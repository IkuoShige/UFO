"""Cross-check the deploy-side runtime against UFO's real training code.

The deploy node rebuilds the observation from scratch in numpy. This suite pins
that reimplementation to the training env's own classes and config, so a change
on either side fails loudly instead of silently degrading the policy.

Run:
    uv run python -m pytest tests/test_k1_ufo_deploy_runtime.py -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_PKG = REPO_ROOT.parent / "booster_k1_locomotion" / "booster_k1_locomotion"
SPEC_JSON = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2" / "export_onnx" / "deploy_spec.json"
OBS_CFG = REPO_ROOT / "humanoidverse" / "config" / "obs" / "bfm_zero_obs.yaml"

pytestmark = pytest.mark.skipif(not DEPLOY_PKG.is_dir(), reason="booster_k1_locomotion checkout not present")

if DEPLOY_PKG.is_dir():
    sys.path.insert(0, str(DEPLOY_PKG))

NUM_DOF = 22


@pytest.fixture(scope="module")
def runtime():
    import ufo_policy_runtime

    return ufo_policy_runtime


@pytest.fixture(scope="module")
def constants():
    import k1_ufo_constants

    return k1_ufo_constants


# --------------------------------------------------------------------------- #
# observation layout vs the training env
# --------------------------------------------------------------------------- #
def test_history_actor_matches_training_history_handler(runtime, constants):
    """Reproduce ``_raw_actor_obs``'s history block with UFO's own HistoryHandler.

    Guards the two things most likely to be wrong in a reimplementation:
    the per-key concatenation order (sorted, not config order) and the
    within-key step order (newest first).
    See humanoidverse/agents/envs/humanoidverse_mjlab.py:938-946.
    """
    from humanoidverse.envs.env_utils.history_handler import HistoryHandler

    cfg = OmegaConf.load(OBS_CFG)
    history_config = cfg.obs.obs_auxiliary
    obs_dims = {"base_ang_vel": 3, "projected_gravity": 3, "dof_pos": NUM_DOF, "dof_vel": NUM_DOF, "actions": NUM_DOF}
    handler = HistoryHandler(1, history_config, obs_dims, "cpu")

    builder = runtime.UfoObsBuilder()
    rng = np.random.default_rng(5)
    default = np.asarray(constants.DEFAULT_ANGLES, dtype=np.float32)

    for _ in range(8):
        dof_pos = rng.standard_normal(NUM_DOF).astype(np.float32)
        dof_vel = rng.standard_normal(NUM_DOF).astype(np.float32)
        ang_vel = rng.standard_normal(3).astype(np.float32)
        gravity = rng.standard_normal(3).astype(np.float32)
        action = rng.standard_normal(NUM_DOF).astype(np.float32)

        # Same scaled quantities the env stores (obs_scales from bfm_zero_obs.yaml).
        scaled = {
            "dof_pos": dof_pos - default,
            "dof_vel": dof_vel,
            "projected_gravity": gravity,
            "base_ang_vel": ang_vel * 0.25,
            "actions": action,
        }

        # --- reference: the env's own code path ---
        hist_cfg = history_config["history_actor"]
        tensors = []
        for key in sorted(hist_cfg.keys()):
            t = handler.query(key)[:, : hist_cfg[key]]
            tensors.append(t.reshape(t.shape[0], -1))
        expected_history = torch.cat(tensors, dim=1).numpy()[0]
        for key in hist_cfg.keys():
            handler.add(key, torch.from_numpy(scaled[key]).unsqueeze(0))

        # --- deploy runtime ---
        obs = builder.build(dof_pos, dof_vel, ang_vel, gravity, action)

        assert obs.shape == (360,)
        np.testing.assert_allclose(obs[72:], expected_history, atol=1e-6)
        # state = dof_pos | dof_vel | projected_gravity | base_ang_vel
        np.testing.assert_allclose(obs[0:22], scaled["dof_pos"], atol=1e-6)
        np.testing.assert_allclose(obs[22:44], scaled["dof_vel"], atol=1e-6)
        np.testing.assert_allclose(obs[44:47], scaled["projected_gravity"], atol=1e-6)
        np.testing.assert_allclose(obs[47:50], scaled["base_ang_vel"], atol=1e-6)
        np.testing.assert_allclose(obs[50:72], scaled["actions"], atol=1e-6)


def test_history_is_zero_at_reset(runtime):
    builder = runtime.UfoObsBuilder()
    obs = builder.build(np.zeros(NUM_DOF), np.zeros(NUM_DOF), np.zeros(3), np.zeros(3), np.zeros(NUM_DOF))
    assert np.all(obs[72:] == 0.0)
    builder.build(np.ones(NUM_DOF), np.ones(NUM_DOF), np.ones(3), np.ones(3), np.ones(NUM_DOF))
    assert np.any(builder.build(np.zeros(NUM_DOF), np.zeros(NUM_DOF), np.zeros(3), np.zeros(3), np.zeros(NUM_DOF))[72:])
    builder.reset()
    obs = builder.build(np.zeros(NUM_DOF), np.zeros(NUM_DOF), np.zeros(3), np.zeros(3), np.zeros(NUM_DOF))
    assert np.all(obs[72:] == 0.0)


def test_dof_pos_is_relative_to_default_pose(runtime, constants):
    """The single most common sim2sim bug: absolute vs offset joint positions."""
    builder = runtime.UfoObsBuilder()
    default = np.asarray(constants.DEFAULT_ANGLES, dtype=np.float32)
    obs = builder.build(default, np.zeros(NUM_DOF), np.zeros(3), np.zeros(3), np.zeros(NUM_DOF))
    np.testing.assert_allclose(obs[0:22], np.zeros(NUM_DOF), atol=1e-6)


def test_projected_gravity_matches_torch_util(runtime):
    """Mirror of humanoidverse/utils/torch_utils.py quat_rotate_inverse(w_last=True)."""
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)  # wxyz
        q_xyzw = torch.tensor([[q[1], q[2], q[3], q[0]]], dtype=torch.float32)
        expected = quat_rotate_inverse(q_xyzw, torch.tensor([[0.0, 0.0, -1.0]]), True).numpy()[0]
        np.testing.assert_allclose(runtime.projected_gravity_from_quat(q), expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# action -> PD target
# --------------------------------------------------------------------------- #
def test_action_pipeline_matches_env(runtime, constants):
    """``_normalized_action`` + JointPositionAction (scale, use_default_offset)."""
    conv = runtime.UfoActionConverter()
    rng = np.random.default_rng(2)
    a_net = np.tanh(rng.standard_normal(NUM_DOF)).astype(np.float32)

    expected_clipped = np.clip(a_net * 5.0, -5.0, 5.0)
    np.testing.assert_allclose(conv.clip_action(a_net), expected_clipped, atol=1e-6)

    expected_q = expected_clipped * np.asarray(constants.PER_JOINT_ACTION_SCALE) + np.asarray(constants.DEFAULT_ANGLES)
    np.testing.assert_allclose(conv.joint_targets(expected_clipped), expected_q, atol=1e-5)

    # Saturated action must clip, not wrap.
    np.testing.assert_allclose(conv.clip_action(np.full(NUM_DOF, 3.0, dtype=np.float32)), np.full(NUM_DOF, 5.0), atol=1e-6)
    # Zero action must command exactly the default pose.
    np.testing.assert_allclose(conv.joint_targets(np.zeros(NUM_DOF)), np.asarray(constants.DEFAULT_ANGLES), atol=1e-6)


# --------------------------------------------------------------------------- #
# z bank / blending
# --------------------------------------------------------------------------- #
def test_z_stays_on_manifold_through_blend(runtime, tmp_path):
    """Every z the actor sees must satisfy ||z|| == sqrt(z_dim) (archi.norm_z)."""
    rng = np.random.default_rng(4)
    z_dim = 256
    skills = {"a": rng.standard_normal(z_dim).astype(np.float32), "b": rng.standard_normal(z_dim).astype(np.float32)}
    seq = rng.standard_normal((7, z_dim)).astype(np.float32)
    path = runtime.ZBank.save(tmp_path / "bank.npz", skills, {"s": seq})

    bank = runtime.ZBank.load(path)
    assert set(bank.skills) == {"a", "b"} and set(bank.sequences) == {"s"}

    ctl = runtime.ZController(bank, initial="a", blend_steps=10)
    np.testing.assert_allclose(np.linalg.norm(ctl.step()), math.sqrt(z_dim), atol=1e-3)

    ctl.select("b")
    seen = [ctl.step() for _ in range(15)]
    for z in seen:
        np.testing.assert_allclose(np.linalg.norm(z), math.sqrt(z_dim), atol=1e-3)
    # Blend actually moves and lands exactly on the target.
    assert not np.allclose(seen[0], seen[-1])
    np.testing.assert_allclose(seen[-1], bank.skills["b"], atol=1e-5)

    # A mean of latents is off-manifold until re-projected.
    blended = 0.5 * (bank.skills["a"] + bank.skills["b"])
    assert abs(np.linalg.norm(blended) - math.sqrt(z_dim)) > 1e-3
    np.testing.assert_allclose(np.linalg.norm(runtime.project_z(blended)), math.sqrt(z_dim), atol=1e-3)


def test_z_sequence_mode_advances_and_loops(runtime, tmp_path):
    rng = np.random.default_rng(9)
    seq = rng.standard_normal((4, 256)).astype(np.float32)
    path = runtime.ZBank.save(tmp_path / "bank.npz", {"a": rng.standard_normal(256).astype(np.float32)}, {"s": seq})
    bank = runtime.ZBank.load(path)

    ctl = runtime.ZController(bank, initial="seq:s", blend_steps=0)
    got = [ctl.step().copy() for _ in range(6)]
    for i, z in enumerate(got):
        np.testing.assert_allclose(z, bank.sequences["s"][i % 4], atol=1e-5)

    ctl.select("seq:s", immediate=True, loop=False)
    tail = [ctl.step().copy() for _ in range(10)]
    np.testing.assert_allclose(tail[-1], bank.sequences["s"][-1], atol=1e-5)


def test_unknown_skill_raises(runtime, tmp_path):
    rng = np.random.default_rng(6)
    path = runtime.ZBank.save(tmp_path / "bank.npz", {"a": rng.standard_normal(256).astype(np.float32)})
    ctl = runtime.ZController(runtime.ZBank.load(path), initial="a", blend_steps=0)
    with pytest.raises(KeyError):
        ctl.select("nope")
    with pytest.raises(KeyError):
        ctl.select("seq:nope")


# --------------------------------------------------------------------------- #
# constants provenance
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SPEC_JSON.exists(), reason="deploy_spec.json not exported")
def test_constants_match_exported_spec(constants):
    """k1_ufo_constants.py is generated from deploy_spec.json; catch drift."""
    spec = json.loads(SPEC_JSON.read_text())
    assert list(constants.JOINT_NAMES) == spec["dof_names"]
    assert constants.NUM_DOF == spec["num_dof"]
    np.testing.assert_allclose(constants.DEFAULT_ANGLES, spec["default_joint_angles"], atol=1e-9)
    np.testing.assert_allclose(constants.JOINT_KP, spec["kp"], atol=1e-9)
    np.testing.assert_allclose(constants.JOINT_KD, spec["kd"], atol=1e-9)
    np.testing.assert_allclose(constants.PER_JOINT_ACTION_SCALE, spec["per_joint_action_scale"], atol=1e-12)
    np.testing.assert_allclose(constants.TORQUE_LIMITS, spec["effort_limit"], atol=1e-9)
    assert constants.ACTOR_OBS_DIM == spec["actor_obs_dim"]
    assert constants.RL_RATE == spec["control_fps"]
    assert list(constants.HISTORY_KEY_ORDER) == spec["history_actor"]["key_order"]


def test_ufo_joint_order_matches_locomotion_constants(constants):
    """Safety-critical: the SDK motor index order must be identical for both policies."""
    import k1_constants as loco

    assert list(constants.JOINT_NAMES) == list(loco.JOINT_NAMES)
    # ...but nothing else may be shared.
    assert list(constants.JOINT_KP) != list(loco.JOINT_KP)
    assert list(constants.DEFAULT_ANGLES) != list(loco.DEFAULT_ANGLES)


@pytest.mark.skipif(not SPEC_JSON.exists(), reason="deploy_spec.json not exported")
def test_joint_order_matches_mujoco_model(constants):
    """Mechanical check of the joint-order chain, not a name-by-eye comparison.

    UFO control_joints order == MJCF hinge-joint order == MJCF actuator order,
    and the hinge joints occupy qpos[7:29] / qvel[6:28] contiguously in that same
    order -- which is what mujoco_sim_node.py indexes (``qpos[7 + i]``) and what
    the booster SDK's serial motor index assumes.
    """
    import mujoco

    spec = json.loads(SPEC_JSON.read_text())
    model = mujoco.MjModel.from_xml_path(spec["xml_path"])
    names = list(constants.JOINT_NAMES)

    hinge = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert hinge == names
    assert actuators == names

    qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    dof_adr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    assert qpos_adr == list(range(7, 7 + constants.NUM_DOF))
    assert dof_adr == list(range(6, 6 + constants.NUM_DOF))
