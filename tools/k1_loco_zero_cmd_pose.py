#!/usr/bin/env python3
"""Measure the steady-state leg pose a K1 IsaacLab locomotion policy holds at ``cmd_vel = 0``.

This is the *hand-off pose*: the posture the walk policy settles into just before it
starts walking, and therefore the posture a get-up policy must end in.

Why a plain-MuJoCo rollout: the trained checkpoints are not available (``logs/`` is
gitignored and absent), only the exported ONNX policies in
``booster_k1_locomotion/assets``. Those are self-contained -- the rsl_rl
``EmpiricalNormalization`` is baked in as ``Sub(mean) -> Div(std)`` at the graph input
(verified on all three files), so raw observations go straight in.

Observation contract (49 dims, single step, no history). Established three ways --
the env cfg, ``history_layout.POLICY_TERM_SPECS``, and the deployed ROS2 node -- and
cross-checked against the baked normalizer statistics:

    [ 0: 3]  base_ang_vel        body-frame gyro, rad/s, unscaled
    [ 3: 6]  projected_gravity   R^T @ (0,0,-1), unit vector
    [ 6: 9]  velocity_commands   (vx, vy, wz)            <- held at 0 here
    [ 9:21]  joint_pos_rel       q - q_default, 12 legs, JOINT_NAMES_K1 order
    [21:33]  joint_vel_rel       qd, 12 legs (default joint vel is 0)
    [33:45]  actions             previous *raw* policy output (pre-scale), 12
    [45:49]  gait_phase          [sin L, cos L, sin R, cos R] -- EXACTLY ZERO when
                                 ||cmd[:3]|| < 0.05 (mdp/observations.py:49-52)

  IsaacLab-K1-Locomotion (commit 852ca36):
    velocity_env_cfg.py:34-42     JOINT_NAMES_K1 (12 legs, order above)
    velocity_env_cfg.py:131       JointPositionActionCfg(scale=0.5, use_default_offset=True)
    velocity_env_cfg.py:339-342   decimation=4, sim.dt=0.005  -> 50 Hz policy / 200 Hz physics
    rough_env_cfg.py:145-166      init_state.joint_pos (the "config default pose")
    rough_env_cfg.py:107-118      delayed_pd_leg: kp=160, kd=4.0, effort/velocity limits, delay 2-7
    rough_env_cfg.py:170-181      feet: kp=50, kd=2.5
    rough_env_cfg.py:200-218      K1PolicyCfg -- the term order above
    mdp/observations.py:25-54     phase_obs -> zeros below cmd_threshold
  Deploy side (booster_k1_locomotion, identical on main and origin/feat/dual_walk):
    rl_policy_isaaclab_node.py:230-266   the same 49-dim builder, gait_phase zeroed at rest
    k1_constants_isaaclab.py:23-49       DEFAULT_ANGLES / kp / kd / ACTION_SCALE=0.5

Robot model. Training used ``assets_soccer/.../K1/K1_locomotion.urdf``, which is
byte-identical to ``K1_22dof.urdf`` except that the ten head/arm joints are ``fixed``
(all at rpy 0 0 0, i.e. **welded at joint angle zero**) and IsaacLab spawns it with
``merge_fixed_joints=True``. So the training robot is a 12-DoF legs-only articulation
with the arms rigid at zero -- there are no "unmatched joints" for IsaacLab to
default. ``--upper-body weld`` (the default) reproduces that by deleting those joints
from the MuJoCo spec. ``--upper-body pd-deploy`` instead PD-holds them at the deploy
``DEFAULT_ANGLES`` (shoulder roll -+1.374), which is what the real robot does; the two
differ, and comparing them measures how much that mismatch moves the leg equilibrium.

Example::

    uv run python tools/k1_loco_zero_cmd_pose.py                      # primary policy
    uv run python tools/k1_loco_zero_cmd_pose.py --all                # all three, table
    uv run python tools/k1_loco_zero_cmd_pose.py --gains deploy       # sensitivity
    uv run python tools/k1_loco_zero_cmd_pose.py --command 0.6 0 0     # harness check: does it walk?
    uv run python tools/k1_loco_zero_cmd_pose.py --init-jitter 0.03 --seed 4   # fixed-point uniqueness
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import onnxruntime as ort

# --------------------------------------------------------------------------- #
# constants transcribed from the two authoritative sources (see module docstring)
# --------------------------------------------------------------------------- #

ASSET_XML = Path("/home/shige/ufo_ws/booster_assets/robots/K1/K1_22dof.xml")
ONNX_DIR = Path("/home/shige/ufo_ws/booster_k1_locomotion/assets")

# The current deployed walk policy (default in every newer launch file). The other
# two are same-lineage 49->12 exports kept for comparison.
PRIMARY_ONNX = "policy_180843_19999.onnx"
ALL_ONNX = (PRIMARY_ONNX, "policy_isaaclab_walk.onnx", "policy_205319_19999.onnx")

# velocity_env_cfg.py:34-42 -- obs/action order, preserve_order=True.
LEG_JOINTS = (
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
UPPER_JOINTS = (
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
)

# rough_env_cfg.py:147-164 -- THE config default pose, and the PD offset the action
# is added to. One vector, used for both ``joint_pos_rel`` and ``q_des``.
DEFAULT_LEG_POSE = np.array(
    [-0.26, 0.0, 0.0, 0.52, -0.26, 0.0,
     -0.26, 0.0, 0.0, 0.52, -0.26, 0.0], dtype=np.float64
)

# k1_constants_isaaclab.py:23-29 -- deploy-side upper-body hold pose (a deploy choice;
# training welds these at zero).
DEPLOY_UPPER_POSE = np.array([0.0, 0.0, 0.3, -1.374, 0.0, -1.2, 0.3, 1.374, 0.0, 1.2])

# rough_env_cfg.py:109-115 / :171-178 -- explicit (DelayedPD) actuators. Per-joint,
# in LEG_JOINTS order.
ISAAC_KP = np.array([160.0] * 4 + [50.0, 50.0] + [160.0] * 4 + [50.0, 50.0])
ISAAC_KD = np.array([4.0] * 4 + [2.5, 2.5] + [4.0] * 4 + [2.5, 2.5])
ISAAC_EFFORT = np.array([68.0, 76.0, 38.3, 112.0, 38.3, 38.3] * 2)
ISAAC_ARMATURE = np.array([0.0478125, 0.0339552, 0.0282528, 0.095625, 0.0282528, 0.0282528] * 2)

# k1_constants_isaaclab.py:32-45 -- what the real robot actually runs.
DEPLOY_KP = np.array([200.0] * 4 + [50.0, 50.0] + [200.0] * 4 + [50.0, 50.0])
DEPLOY_KD = np.array([5.0] * 4 + [1.5, 1.5] + [5.0] * 4 + [1.5, 1.5])
# resource/k1_isaaclab_gains.yaml differs from the .py on kd -- the yaml is what the
# C++ node loads at runtime.
YAML_KD = np.array([3.5] * 4 + [2.5, 2.5] + [3.5] * 4 + [2.5, 2.5])

# K1_22dof.xml:197-208 -- the MJCF/URDF torque limits, much tighter than the
# IsaacLab actuator cfg. Selectable to check whether the standing torques the policy
# asks for are within the modelled hardware limit.
MJCF_EFFORT = np.array([30.0, 35.0, 20.0, 40.0, 20.0, 20.0] * 2)

ACTION_SCALE = 0.5           # velocity_env_cfg.py:131
COMMAND_THRESHOLD = 0.05     # rough_env_cfg.py:59 -- gait_phase is zero below this
OBS_DIM = 49
N_LEG = 12

# mdp/events.py:84-107 compute_cmd_phase_freq. The *current* cfg raised low_freq to
# 1.8 Hz (rough_env_cfg.py:49, dated 2026-08-02) but explicitly notes the deployed
# policies were trained at 1.5, which is also what src/k1_constants_isaaclab.hpp
# (origin/feat/dual_walk) uses. Only needed for the nonzero-command *validation*
# rollout -- at zero command the phase is identically zero.
PHASE_LOW_SPEED, PHASE_HIGH_SPEED = 1.0, 1.8
PHASE_LOW_FREQ, PHASE_HIGH_FREQ = 1.5, 2.0


def cmd_phase_freq(lin_speed: float) -> float:
    """``mdp/events.py:compute_cmd_phase_freq`` -- gait Hz from ``||cmd_xy||``."""
    slope = (PHASE_HIGH_FREQ - PHASE_LOW_FREQ) / (PHASE_HIGH_SPEED - PHASE_LOW_SPEED)
    return PHASE_LOW_FREQ + max(0.0, lin_speed - PHASE_LOW_SPEED) * slope

GAIN_SETS = {
    "isaac": (ISAAC_KP, ISAAC_KD),   # rough_env_cfg actuator cfg == what training used
    "deploy": (DEPLOY_KP, DEPLOY_KD),  # k1_constants_isaaclab.py
    "deploy-yaml": (DEPLOY_KP, YAML_KD),  # resource/k1_isaaclab_gains.yaml
}


# --------------------------------------------------------------------------- #
# math helpers (mirrors booster_k1_locomotion/ufo_policy_runtime.py:41-60)
# --------------------------------------------------------------------------- #

def quat_rotate_inverse_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` (world) into the body frame of ``quat_wxyz`` (w-first)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    q_w, q_vec = q[0], q[1:]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * float(np.dot(q_vec, v)) * 2.0
    return a - b + c


def projected_gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """``mdp.projected_gravity``: gravity unit vector in the trunk frame."""
    return quat_rotate_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0]))


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #

@dataclass
class SimHandles:
    model: mujoco.MjModel
    data: mujoco.MjData
    leg_qpos_adr: np.ndarray
    leg_dof_adr: np.ndarray
    leg_act_ids: np.ndarray
    upper_qpos_adr: np.ndarray
    upper_dof_adr: np.ndarray
    upper_act_ids: np.ndarray
    root_qpos_adr: int
    root_dof_adr: int
    foot_geoms: list[int] = field(default_factory=list)
    foot_body_ids: list[int] = field(default_factory=list)


def build_model(
    xml_path: Path,
    *,
    upper_body: str,
    effort_limit: np.ndarray,
    physics_dt: float,
) -> SimHandles:
    """Compile a MuJoCo model that matches the IsaacLab training articulation.

    Steps, in this order (deleting a joint whose motor still exists fails to compile):
      1. delete all 22 stock ``<motor>`` actuators
      2. ``weld``: delete the 10 head/arm joints, reproducing K1_locomotion.urdf's
         fixed joints + ``merge_fixed_joints=True``
      3. add gear-1 torque motors for the joints we drive, with forcerange set from
         the IsaacLab actuator ``effort_limit``
      4. override leg joint ``armature`` to the IsaacLab actuator cfg values (the MJCF
         already agrees except on the ankles: 0.0565 vs 0.0282528)
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))

    for actuator in list(spec.actuators):
        spec.delete(actuator)

    drive_upper = upper_body != "weld"
    if not drive_upper:
        for joint in list(spec.joints):
            if joint.name in UPPER_JOINTS:
                spec.delete(joint)

    driven = list(LEG_JOINTS) + (list(UPPER_JOINTS) if drive_upper else [])
    # Upper-body forcerange from K1_22dof.xml:187-196 (head 6, arms 14).
    upper_effort = {n: (6.0 if n.startswith(("AAHead", "Head")) else 14.0) for n in UPPER_JOINTS}
    for name in driven:
        limit = (
            float(effort_limit[LEG_JOINTS.index(name)])
            if name in LEG_JOINTS
            else upper_effort[name]
        )
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gear[0] = 1.0
        act.forcelimited = True
        act.forcerange[:] = np.array([-limit, limit])
        act.ctrllimited = True
        act.ctrlrange[:] = np.array([-limit, limit])

    for joint in spec.joints:
        if joint.name in LEG_JOINTS:
            joint.armature = float(ISAAC_ARMATURE[LEG_JOINTS.index(joint.name)])

    model = spec.compile()
    model.opt.timestep = physics_dt
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    data = mujoco.MjData(model)

    def resolve(names):
        qadr, dadr, aids = [], [], []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"joint/actuator '{name}' missing from compiled model")
            qadr.append(model.jnt_qposadr[jid])
            dadr.append(model.jnt_dofadr[jid])
            aids.append(aid)
        return (np.asarray(qadr, np.int64), np.asarray(dadr, np.int64), np.asarray(aids, np.int64))

    leg_q, leg_d, leg_a = resolve(LEG_JOINTS)
    upper_q, upper_d, upper_a = resolve(UPPER_JOINTS) if drive_upper else (
        np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    )

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    if root_jid < 0 or model.jnt_type[root_jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("expected a free root joint 'world_joint'")

    # The invisible box under each foot_link is the actual contact geom.
    foot_geoms, foot_bodies = [], []
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_link")
        foot_bodies.append(bid)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                foot_geoms.append(gid)
    if len(foot_geoms) != 2:
        raise RuntimeError(f"expected 2 foot contact boxes, found {len(foot_geoms)}")

    return SimHandles(
        model=model, data=data,
        leg_qpos_adr=leg_q, leg_dof_adr=leg_d, leg_act_ids=leg_a,
        upper_qpos_adr=upper_q, upper_dof_adr=upper_d, upper_act_ids=upper_a,
        root_qpos_adr=int(model.jnt_qposadr[root_jid]),
        root_dof_adr=int(model.jnt_dofadr[root_jid]),
        foot_geoms=foot_geoms, foot_body_ids=foot_bodies,
    )


def assert_body_frame_ang_vel(sim: SimHandles) -> None:
    """MuJoCo free-joint ``qvel[3:6]`` must equal ``root_ang_vel_b`` (verified, not assumed).

    ``mdp.base_ang_vel`` is ``root_ang_vel_b`` -- the world angular velocity rotated
    into the base frame. If these disagreed the obs would carry a world-frame gyro.
    """
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    bid = int(sim.model.jnt_bodyid[jid])
    probe = mujoco.MjData(sim.model)
    rng = np.random.default_rng(0)
    quat = rng.standard_normal(4)
    probe.qpos[sim.root_qpos_adr + 3: sim.root_qpos_adr + 7] = quat / np.linalg.norm(quat)
    probe.qvel[sim.root_dof_adr: sim.root_dof_adr + 6] = rng.standard_normal(6)
    mujoco.mj_forward(sim.model, probe)
    world6 = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, probe, mujoco.mjtObj.mjOBJ_BODY, bid, world6, 0)
    quat_wxyz = probe.qpos[sim.root_qpos_adr + 3: sim.root_qpos_adr + 7]
    rotated = quat_rotate_inverse_wxyz(quat_wxyz, world6[0:3])
    local = probe.qvel[sim.root_dof_adr + 3: sim.root_dof_adr + 6]
    if not np.allclose(rotated, local, atol=1e-5, rtol=1e-4):
        raise AssertionError(f"ang_vel frame mismatch: rotated_world={rotated} qvel_local={local}")


def _lowest_point(model, data, gid: int) -> float:
    """Lowest world-z of a box geom (exact for a box; the foot contacts are boxes)."""
    pos = data.geom_xpos[gid]
    rot = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    return float(pos[2] - np.abs(rot[2, :]) @ size)


def place_standing(sim: SimHandles, upper_pose: np.ndarray, *,
                   init_pose: np.ndarray | None = None, clearance: float = 2e-3) -> float:
    """Put the robot upright at the config default pose with the feet just above ground.

    IsaacLab spawns at ``init_state.pos=(0,0,0.6)`` and lets the robot drop onto the
    terrain; here we place the feet on the ground instead so the measurement starts
    from a clean static state rather than an impact transient.
    """
    mujoco.mj_resetData(sim.model, sim.data)
    q = sim.data.qpos
    q[sim.root_qpos_adr: sim.root_qpos_adr + 3] = (0.0, 0.0, 1.0)
    q[sim.root_qpos_adr + 3: sim.root_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    q[sim.leg_qpos_adr] = DEFAULT_LEG_POSE if init_pose is None else init_pose
    if sim.upper_qpos_adr.size:
        q[sim.upper_qpos_adr] = upper_pose
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)

    lowest = min(_lowest_point(sim.model, sim.data, gid) for gid in sim.foot_geoms)
    q[sim.root_qpos_adr + 2] = 1.0 - lowest + clearance
    mujoco.mj_forward(sim.model, sim.data)
    if sim.data.ncon != 0:
        raise RuntimeError(f"expected no contacts at spawn, got ncon={sim.data.ncon}")
    return float(q[sim.root_qpos_adr + 2])


# --------------------------------------------------------------------------- #
# actuation
# --------------------------------------------------------------------------- #

class IdealPd:
    """IsaacLab ``IdealPDActuator``: ``tau = kp*(q_des - q) + kd*(0 - qd)``, clamped.

    ``DelayedPDActuator`` is exactly this with the *position target* pushed through a
    per-env delay buffer of 2..7 physics steps (rough_env_cfg.py:116-117); the delay
    only shapes the transient, so ``delay_steps`` defaults to 0 and is swept to
    confirm it induces no limit cycle. Unlike UFO's mjlab actuators there is no
    velocity-derated torque curve -- ``DelayedPDActuatorCfg`` clamps to a flat
    ``effort_limit``.
    """

    def __init__(self, kp, kd, effort_limit, *, delay_steps: int = 0) -> None:
        self.kp = np.asarray(kp, np.float64)
        self.kd = np.asarray(kd, np.float64)
        self.effort_limit = np.asarray(effort_limit, np.float64)
        self.delay_steps = int(delay_steps)
        self._buf: list[np.ndarray] = []
        self.peak_torque = np.zeros_like(self.kp)

    def reset(self, q_des: np.ndarray) -> None:
        self._buf = [q_des.copy() for _ in range(self.delay_steps + 1)]
        self.peak_torque[:] = 0.0

    def torque(self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if self.delay_steps > 0:
            self._buf.append(q_des.copy())
            q_des = self._buf.pop(0)
        raw = self.kp * (q_des - q) - self.kd * qd
        tau = np.clip(raw, -self.effort_limit, self.effort_limit)
        self.peak_torque = np.maximum(self.peak_torque, np.abs(raw))
        return tau


class LocoPolicy:
    """The 49->12 ONNX walk policy, driven exactly as the deploy node drives it."""

    def __init__(self, onnx_path: Path) -> None:
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if inp.name != "obs" or list(inp.shape) != [1, OBS_DIM]:
            raise RuntimeError(f"unexpected ONNX input {inp.name}{inp.shape}; want obs[1,{OBS_DIM}]")
        out = self.session.get_outputs()[0]
        if out.name != "actions" or list(out.shape) != [1, N_LEG]:
            raise RuntimeError(f"unexpected ONNX output {out.name}{out.shape}; want actions[1,{N_LEG}]")
        self._input = inp.name
        self.prev_action = np.zeros(N_LEG)

    def reset(self) -> None:
        self.prev_action[:] = 0.0

    def build_obs(self, q_leg, qd_leg, base_ang_vel, gravity, command, phase_rad=0.0) -> np.ndarray:
        cmd = np.asarray(command, np.float64)
        # mdp/observations.py:41-52 -- sin/cos of the left phase and of left+pi, then
        # forced to *identically zero* below the threshold. That zeroing is the whole
        # reason a zero-command steady state is well defined.
        if np.linalg.norm(cmd[:3]) < COMMAND_THRESHOLD:
            gait_phase = np.zeros(4)
        else:
            gait_phase = np.array([
                math.sin(phase_rad), math.cos(phase_rad),
                math.sin(phase_rad + math.pi), math.cos(phase_rad + math.pi),
            ])
        obs = np.concatenate([
            base_ang_vel,                  # 3
            gravity,                       # 3
            cmd,                           # 3
            q_leg - DEFAULT_LEG_POSE,      # 12  joint_pos_rel
            qd_leg,                        # 12  joint_vel_rel (default vel = 0)
            self.prev_action,              # 12  last_action, raw/pre-scale
            gait_phase,                    # 4
        ])
        assert obs.shape == (OBS_DIM,), obs.shape
        return obs.astype(np.float32)

    def step(self, *args) -> np.ndarray:
        obs = self.build_obs(*args)
        action = self.session.run(None, {self._input: obs.reshape(1, -1)})[0].ravel()
        self.prev_action = action.astype(np.float64)
        # velocity_env_cfg.py:131 -- use_default_offset=True, scale=0.5.
        return DEFAULT_LEG_POSE + ACTION_SCALE * self.prev_action


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #

def rollout(
    onnx: str = PRIMARY_ONNX,
    *,
    upper_body: Literal["weld", "pd-deploy", "pd-zero"] = "weld",
    gains: Literal["isaac", "deploy", "deploy-yaml"] = "isaac",
    effort: Literal["isaac", "mjcf"] = "isaac",
    physics_dt: float = 0.005,
    decimation: int = 4,
    pd_rate_hz: float = 200.0,
    delay_steps: int = 0,
    seconds: float = 15.0,
    settle_seconds: float = 0.0,
    window_seconds: float = 2.0,
    spawn_z: float | None = None,
    command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_jitter: float = 0.0,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    """Hold ``cmd_vel = 0`` and report the leg pose the policy settles into.

    ``command`` is a harness-validation escape hatch: give it a nonzero velocity and
    the gait phase accumulator runs, so the same code path can be checked against
    "does it actually walk at the commanded speed". ``init_jitter`` perturbs the
    initial leg pose to test whether the zero-command fixed point is unique.
    """
    kp, kd = GAIN_SETS[gains]
    effort_limit = ISAAC_EFFORT if effort == "isaac" else MJCF_EFFORT
    upper_pose = {
        "weld": np.zeros(0),
        "pd-deploy": DEPLOY_UPPER_POSE,
        "pd-zero": np.zeros(len(UPPER_JOINTS)),
    }[upper_body]

    sim = build_model(ASSET_XML, upper_body=upper_body, effort_limit=effort_limit,
                      physics_dt=physics_dt)
    assert_body_frame_ang_vel(sim)

    leg_pd = IdealPd(kp, kd, effort_limit, delay_steps=delay_steps)
    n_upper = sim.upper_act_ids.size
    upper_pd = IdealPd(
        np.array([6.0, 6.0] + [30.0] * 8)[:n_upper],   # k1_constants_isaaclab.py:32-38
        np.array([1.0] * 10)[:n_upper],
        np.array([6.0, 6.0] + [14.0] * 8)[:n_upper],
    ) if n_upper else None

    init_pose = DEFAULT_LEG_POSE.copy()
    if init_jitter > 0.0:
        rng = np.random.default_rng(seed)
        init_pose += rng.uniform(-init_jitter, init_jitter, N_LEG)
    spawn_height = place_standing(sim, upper_pose, init_pose=init_pose)
    if spawn_z is not None:
        sim.data.qpos[sim.root_qpos_adr + 2] = spawn_z
        mujoco.mj_forward(sim.model, sim.data)
        spawn_height = spawn_z

    control_dt = decimation * physics_dt
    pd_hold = max(1, int(round(1.0 / (pd_rate_hz * physics_dt))))
    leg_pd.reset(init_pose)
    if upper_pd is not None:
        upper_pd.reset(upper_pose)

    def physics(q_des_leg: np.ndarray) -> None:
        tau_leg = None
        for k in range(decimation):
            if k % pd_hold == 0:
                tau_leg = leg_pd.torque(q_des_leg, sim.data.qpos[sim.leg_qpos_adr],
                                        sim.data.qvel[sim.leg_dof_adr])
                if upper_pd is not None:
                    sim.data.ctrl[sim.upper_act_ids] = upper_pd.torque(
                        upper_pose, sim.data.qpos[sim.upper_qpos_adr],
                        sim.data.qvel[sim.upper_dof_adr])
            sim.data.ctrl[sim.leg_act_ids] = tau_leg
            mujoco.mj_step(sim.model, sim.data)

    # NOTE: settle_seconds defaults to 0 -- the policy is engaged immediately from the
    # static, upright default pose. A pre-engage PD-only hold is *not* neutral: the
    # config default pose is open-loop unstable (finite joint stiffness lets gravity
    # sag the hips, the CoM crosses the toe at ~0.95 s and it topples forward), so a
    # long settle hands the policy an already-falling robot. Measured dependence:
    # settle 0-0.5 s changes the converged pose by <0.023 rad, 1.0 s by 0.10 rad, and
    # >=1.5 s the robot is already down and the policy cannot recover.
    for _ in range(int(round(settle_seconds / control_dt))):
        physics(init_pose)
    settle_pose = sim.data.qpos[sim.leg_qpos_adr].copy()
    settle_height = float(sim.data.qpos[sim.root_qpos_adr + 2])

    policy = LocoPolicy(ONNX_DIR / onnx)
    policy.reset()
    leg_pd.reset(init_pose)
    cmd = np.asarray(command, np.float64)
    phase_rad, walking = 0.0, bool(np.linalg.norm(cmd[:3]) >= COMMAND_THRESHOLD)

    n_steps = int(round(seconds / control_dt))
    log = {k: [] for k in ("q", "qd", "action", "height", "grav_z", "xy", "contacts", "tau",
                                  "lin_vel_b", "yaw_rate")}
    diverged = False
    for _ in range(n_steps):
        q_leg = sim.data.qpos[sim.leg_qpos_adr].copy()
        qd_leg = sim.data.qvel[sim.leg_dof_adr].copy()
        quat = sim.data.qpos[sim.root_qpos_adr + 3: sim.root_qpos_adr + 7].copy()
        base_ang_vel = sim.data.qvel[sim.root_dof_adr + 3: sim.root_dof_adr + 6].copy()
        gravity = projected_gravity_from_quat(quat)

        if walking:
            # mdp/events.py:139 -- phase advances once per env step, before obs.
            phase_rad = (phase_rad + 2.0 * math.pi
                         * cmd_phase_freq(float(np.linalg.norm(cmd[:2])))
                         * control_dt) % (2.0 * math.pi)
        q_des = policy.step(q_leg, qd_leg, base_ang_vel, gravity, cmd, phase_rad)
        physics(q_des)

        log["q"].append(sim.data.qpos[sim.leg_qpos_adr].copy())
        log["qd"].append(sim.data.qvel[sim.leg_dof_adr].copy())
        log["action"].append(policy.prev_action.copy())
        log["height"].append(float(sim.data.qpos[sim.root_qpos_adr + 2]))
        log["grav_z"].append(float(projected_gravity_from_quat(
            sim.data.qpos[sim.root_qpos_adr + 3: sim.root_qpos_adr + 7])[2]))
        log["xy"].append(sim.data.qpos[sim.root_qpos_adr: sim.root_qpos_adr + 2].copy())
        log["contacts"].append([_foot_in_contact(sim, b) for b in sim.foot_body_ids])
        log["tau"].append(sim.data.ctrl[sim.leg_act_ids].copy())
        log["lin_vel_b"].append(quat_rotate_inverse_wxyz(
            sim.data.qpos[sim.root_qpos_adr + 3: sim.root_qpos_adr + 7],
            sim.data.qvel[sim.root_dof_adr: sim.root_dof_adr + 3]).copy())
        log["yaw_rate"].append(float(sim.data.qvel[sim.root_dof_adr + 5]))
        if not np.all(np.isfinite(sim.data.qpos)):
            diverged = True
            break

    return _summarise(
        sim, log, policy, leg_pd,
        onnx=onnx, upper_body=upper_body, gains=gains, effort=effort,
        physics_dt=physics_dt, decimation=decimation, pd_rate_hz=pd_rate_hz,
        delay_steps=delay_steps, control_dt=control_dt, window_seconds=window_seconds,
        spawn_height=spawn_height, settle_height=settle_height, settle_pose=settle_pose,
        command=tuple(float(c) for c in cmd), init_jitter=init_jitter, seed=seed,
        diverged=diverged, quiet=quiet,
    )


def _foot_in_contact(sim: SimHandles, body_id: int) -> bool:
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        for gid in (c.geom1, c.geom2):
            if sim.model.geom_bodyid[gid] == body_id:
                return True
    return False


def _summarise(sim, log, policy, leg_pd, *, window_seconds, control_dt, **meta) -> dict:
    q = np.asarray(log["q"])
    qd = np.asarray(log["qd"])
    height = np.asarray(log["height"])
    grav_z = np.asarray(log["grav_z"])
    xy = np.asarray(log["xy"])
    contacts = np.asarray(log["contacts"])
    tau = np.asarray(log["tau"])
    lin_vel_b = np.asarray(log["lin_vel_b"])
    yaw_rate = np.asarray(log["yaw_rate"])

    n_win = max(1, int(round(window_seconds / control_dt)))
    w = slice(-n_win, None)
    q_w, qd_w = q[w], qd[w]

    pose = q_w.mean(axis=0)
    pose_std = q_w.std(axis=0)
    pose_ptp = q_w.max(axis=0) - q_w.min(axis=0)
    delta = pose - DEFAULT_LEG_POSE

    fell = bool(grav_z[-1] > -0.7 or height[-1] < 0.35)
    # Both feet must stay down for the whole window, otherwise it is taking steps.
    airborne = int((~contacts[w]).sum())
    stepping = bool(airborne > 0 or pose_ptp.max() > 0.05)
    xy_speed = float(np.linalg.norm(xy[-1] - xy[w][0]) / (n_win * control_dt))
    drifting = bool(xy_speed > 0.05)
    status = "fell" if (fell or meta["diverged"]) else (
        "stepping" if stepping else ("drifting" if drifting else "settled"))

    result = dict(
        **{k: v for k, v in meta.items() if k not in ("quiet",)},
        window_seconds=window_seconds, control_dt=control_dt,
        steps=int(q.shape[0]),
        status=status,
        joint_names=list(LEG_JOINTS),
        default_pose=DEFAULT_LEG_POSE.tolist(),
        converged_pose=pose.tolist(),
        pose_std=pose_std.tolist(),
        pose_peak_to_peak=pose_ptp.tolist(),
        delta_vs_default=delta.tolist(),
        delta_max_abs=float(np.abs(delta).max()),
        delta_max_abs_joint=LEG_JOINTS[int(np.argmax(np.abs(delta)))],
        delta_rms=float(np.sqrt((delta ** 2).mean())),
        mean_action=policy.prev_action.tolist(),
        vel_rms=float(np.sqrt((qd_w ** 2).mean())),
        vel_max_abs=float(np.abs(qd_w).max()),
        base_height_mean=float(height[w].mean()),
        base_height_std=float(height[w].std()),
        gravity_z_mean=float(grav_z[w].mean()),
        xy_speed=xy_speed,
        xy_total_travel=float(np.linalg.norm(xy[-1] - xy[0])),
        airborne_foot_samples=airborne,
        settle_pose_delta_max=float(np.abs(meta["settle_pose"] - DEFAULT_LEG_POSE).max()),
        peak_torque=leg_pd.peak_torque.tolist(),
        peak_torque_over_mjcf=float((leg_pd.peak_torque / MJCF_EFFORT).max()),
        torque_saturated=bool(np.any(np.abs(tau[w]) >= 0.999 * leg_pd.effort_limit)),
        # nonzero-command harness validation: does it track the command it was given?
        tracked_lin_vel_b=lin_vel_b[w].mean(axis=0).tolist(),
        tracked_yaw_rate=float(yaw_rate[w].mean()),
    )
    result["settle_pose"] = meta["settle_pose"].tolist()

    if not meta["quiet"]:
        _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print("=" * 92)
    print(f"{r['onnx']}   upper={r['upper_body']} gains={r['gains']} effort={r['effort']} "
          f"delay={r['delay_steps']} dt={r['physics_dt']} pd={r['pd_rate_hz']}Hz")
    print(f"  status = {r['status'].upper()}   "
          f"|qd|rms={r['vel_rms']:.4f} rad/s  base_h={r['base_height_mean']:.4f}"
          f"+-{r['base_height_std']:.4f} m  -gz={-r['gravity_z_mean']:.4f}  "
          f"xy_speed={r['xy_speed']:.4f} m/s  airborne={r['airborne_foot_samples']}")
    if any(abs(c) > 0.0 for c in r["command"]):
        v, wz = r["tracked_lin_vel_b"], r["tracked_yaw_rate"]
        print(f"  command = {tuple(round(c, 3) for c in r['command'])}  ->  tracked "
              f"lin_vel_b=({v[0]:+.3f},{v[1]:+.3f}) yaw_rate={wz:+.3f}")
    print(f"  {'joint':22s} {'default':>9s} {'measured':>9s} {'delta':>9s} "
          f"{'std':>8s} {'p2p':>8s} {'|tau|max':>9s}")
    for i, name in enumerate(r["joint_names"]):
        print(f"  {name:22s} {r['default_pose'][i]:+9.4f} {r['converged_pose'][i]:+9.4f} "
              f"{r['delta_vs_default'][i]:+9.4f} {r['pose_std'][i]:8.4f} "
              f"{r['pose_peak_to_peak'][i]:8.4f} {r['peak_torque'][i]:9.2f}")
    print(f"  delta: max|.| = {r['delta_max_abs']:.4f} rad ({r['delta_max_abs_joint']}), "
          f"RMS = {r['delta_rms']:.4f} rad   "
          f"[{math.degrees(r['delta_max_abs']):.2f} deg / {math.degrees(r['delta_rms']):.2f} deg]")
    print(f"  peak torque / MJCF limit = {r['peak_torque_over_mjcf']:.2f}x   "
          f"saturated in window = {r['torque_saturated']}")


def main(
    onnx: str = PRIMARY_ONNX,
    all: bool = False,
    upper_body: Literal["weld", "pd-deploy", "pd-zero"] = "weld",
    gains: Literal["isaac", "deploy", "deploy-yaml"] = "isaac",
    effort: Literal["isaac", "mjcf"] = "isaac",
    physics_dt: float = 0.005,
    decimation: int = 4,
    pd_rate_hz: float = 200.0,
    delay_steps: int = 0,
    seconds: float = 15.0,
    settle_seconds: float = 0.0,
    window_seconds: float = 2.0,
    spawn_z: float | None = None,
    command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_jitter: float = 0.0,
    seed: int = 0,
    out: Path | None = None,
) -> None:
    """Measure the zero-command steady-state leg pose of a K1 IsaacLab walk policy."""
    targets = list(ALL_ONNX) if all else [onnx]
    results = [
        rollout(t, upper_body=upper_body, gains=gains, effort=effort, physics_dt=physics_dt,
                decimation=decimation, pd_rate_hz=pd_rate_hz, delay_steps=delay_steps,
                seconds=seconds, settle_seconds=settle_seconds, window_seconds=window_seconds,
                spawn_z=spawn_z, command=command, init_jitter=init_jitter, seed=seed)
        for t in targets
    ]
    if len(results) > 1:
        print("=" * 92)
        print(f"{'policy':30s} {'status':10s} {'max|delta|':>11s} {'RMS delta':>10s} "
              f"{'base_h':>8s} {'|qd|rms':>8s}")
        for r in results:
            print(f"{r['onnx']:30s} {r['status']:10s} {r['delta_max_abs']:11.4f} "
                  f"{r['delta_rms']:10.4f} {r['base_height_mean']:8.4f} {r['vel_rms']:8.4f}")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
