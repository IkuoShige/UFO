"""Numerical equivalence between the exported K1 UFO ONNX policy and PyTorch.

Guards the deploy contract:
  actor_obs = concat([state(50), last_action(22), history_actor(288), z(256)]) -> 616
  action    = FBcprAuxModel.act(obs_dict, z, mean=True)   # tanh mean, in [-1, 1]

Run:
    uv run python -m pytest tests/test_k1_ufo_onnx_equivalence.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2"
CHECKPOINT_DIR = RUN_DIR / "checkpoint"
EXPORT_DIR = RUN_DIR / "export_onnx"
POLICY_ONNX = EXPORT_DIR / "FBcprAuxModel.onnx"
BACKWARD_ONNX = EXPORT_DIR / "backward_encoder.onnx"
SPEC_JSON = EXPORT_DIR / "deploy_spec.json"

ATOL = 1e-4
RTOL = 1e-4

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR.exists() and POLICY_ONNX.exists() and SPEC_JSON.exists()),
    reason="K1 checkpoint / ONNX export not present (run tools/export_k1_ufo_policy_onnx.py)",
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_JSON.read_text())


@pytest.fixture(scope="module")
def model():
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

    m = load_model_from_checkpoint_dir(str(CHECKPOINT_DIR), device="cpu")
    m.eval()
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def ort_policy():
    import onnxruntime as ort

    return ort.InferenceSession(str(POLICY_ONNX), providers=["CPUExecutionProvider"])


def _project_z(z: np.ndarray) -> np.ndarray:
    return math.sqrt(z.shape[-1]) * z / np.linalg.norm(z, axis=-1, keepdims=True)


def _random_obs(batch: int, spec: dict, seed: int = 0):
    rng = np.random.default_rng(seed)
    dims = spec["actor_input_dims"]
    obs = {key: rng.standard_normal((batch, dims[key])).astype(np.float32) for key in spec["actor_input_keys"]}
    z = _project_z(rng.standard_normal((batch, spec["z_dim"])).astype(np.float32)).astype(np.float32)
    return obs, z


def test_export_signature_matches_spec(spec, ort_policy):
    inputs = ort_policy.get_inputs()
    assert [i.name for i in inputs] == ["actor_obs"]
    assert inputs[0].shape[-1] == spec["actor_obs_dim"] == 616
    outputs = ort_policy.get_outputs()
    assert [o.name for o in outputs] == ["action"]
    assert outputs[0].shape[-1] == spec["output_action_dim"] == 22
    # Key order is load-bearing for the deploy obs builder.
    assert spec["actor_input_keys"] == ["state", "last_action", "history_actor"]
    assert spec["actor_input_dims"] == {"state": 50, "last_action": 22, "history_actor": 288}


def test_policy_onnx_matches_torch(spec, model, ort_policy):
    batch = 100
    obs, z = _random_obs(batch, spec, seed=7)

    actor_obs = np.concatenate([obs[k] for k in spec["actor_input_keys"]] + [z], axis=-1).astype(np.float32)
    assert actor_obs.shape == (batch, spec["actor_obs_dim"])

    ort_action = ort_policy.run(["action"], {"actor_obs": actor_obs})[0]

    with torch.no_grad():
        torch_action = (
            model.act({k: torch.from_numpy(v) for k, v in obs.items()}, torch.from_numpy(z), mean=True)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    max_abs = float(np.max(np.abs(torch_action - ort_action)))
    assert max_abs < ATOL, f"policy ONNX vs torch max_abs={max_abs:.3e}"
    assert np.all(np.abs(ort_action) <= 1.0 + 1e-5), "actor output must be a tanh mean in [-1, 1]"


def test_policy_onnx_batch_one_stream(spec, model, ort_policy):
    """Deploy runs batch=1; make sure per-sample inference matches the batched path."""
    obs, z = _random_obs(8, spec, seed=11)
    for i in range(8):
        actor_obs = np.concatenate([obs[k][i : i + 1] for k in spec["actor_input_keys"]] + [z[i : i + 1]], axis=-1)
        ort_action = ort_policy.run(["action"], {"actor_obs": actor_obs.astype(np.float32)})[0]
        with torch.no_grad():
            torch_action = (
                model.act(
                    {k: torch.from_numpy(v[i : i + 1]) for k, v in obs.items()},
                    torch.from_numpy(z[i : i + 1]),
                    mean=True,
                )
                .cpu()
                .numpy()
            )
        assert np.max(np.abs(torch_action - ort_action)) < ATOL


@pytest.mark.skipif(not BACKWARD_ONNX.exists(), reason="backward encoder ONNX not exported")
def test_backward_encoder_onnx_matches_torch(model):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(BACKWARD_ONNX), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    # last_action is pruned by torch.onnx.export: FBcprAux's backward map filters
    # to key=["state", "privileged_state"] only.
    assert names == ["state", "privileged_state"]

    rng = np.random.default_rng(3)
    batch = 32
    feed = {
        "state": rng.standard_normal((batch, 50)).astype(np.float32),
        "privileged_state": rng.standard_normal((batch, 343)).astype(np.float32),
    }
    ort_z = sess.run(["z"], feed)[0]

    with torch.no_grad():
        obs = {k: torch.from_numpy(v) for k, v in feed.items()}
        obs["last_action"] = torch.zeros(batch, 22)
        torch_z = model.project_z(model.backward_map(obs)).cpu().numpy().astype(np.float32)

    max_abs = float(np.max(np.abs(torch_z - ort_z)))
    assert max_abs < ATOL, f"backward encoder ONNX vs torch max_abs={max_abs:.3e}"
    # z must live on the sphere of radius sqrt(z_dim).
    norms = np.linalg.norm(ort_z, axis=-1)
    assert np.allclose(norms, math.sqrt(ort_z.shape[-1]), atol=1e-3), norms[:5]


def test_deploy_spec_control_constants(spec):
    """Values the deploy node must mirror; drift here is a silent sim2real bug."""
    assert spec["control_fps"] == 50.0 and spec["physics_fps"] == 200.0 and spec["control_decimation"] == 4
    assert spec["normalize_action_to"] == 5.0 and spec["action_clip_value"] == 5.0
    assert spec["action_scale"] == 0.5 and spec["action_rescale"] is True
    assert spec["obs_scales"]["base_ang_vel"] == 0.25
    assert spec["history_actor"]["newest_first"] is True
    assert spec["history_actor"]["key_order"] == ["actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"]
    # per-joint action scale = action_scale * effort_limit / kp
    for i in range(spec["num_dof"]):
        expected = spec["action_scale"] * spec["effort_limit"][i] / spec["kp"][i]
        assert abs(spec["per_joint_action_scale"][i] - expected) < 1e-9
