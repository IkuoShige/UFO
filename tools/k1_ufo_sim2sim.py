#!/usr/bin/env python3
"""Standalone MuJoCo sim2sim check for the K1 UFO get-up policy. No ROS2 needed.

Drops the robot into a fallen pose, drives it with the exported ONNX meta-policy
plus a latent ``z``, and reports whether it stands up. Writes an mp4.

The MuJoCo model is assembled to mirror the *training* scene exactly:

  * mjlab attaches the robot MJCF (actuators stripped) into a parent spec that
    already holds a ``terrain`` plane, so the compiled model ends up with TWO
    coincident ground planes -- ``robot/ground`` from K1_22dof.xml and mjlab's
    ``terrain``. Reproduced here (see ``--single-ground`` to A/B it).
  * torque-mode ``<motor>`` actuators + a Python PD loop at the physics rate,
    matching mjlab's DcMotorActuator (velocity-derated torque-speed curve).
  * ``MujocoCfg`` solver/integrator settings, which override the MJCF ``<option>``
    block (MjSpec.attach does not propagate child options).

Crucially, the policy is driven through ``booster_k1_locomotion``'s
``ufo_policy_runtime`` -- the same module the ROS2 node uses -- so this script
validates the actual deploy code path, not a re-implementation of it.

Example::

    uv run python tools/k1_ufo_sim2sim.py --z-name getup_opt --video /tmp/getup.mp4
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Must be set before importing mujoco. Headless boxes have no DISPLAY, so default
# to EGL on GPU 0; override by exporting MUJOCO_GL yourself.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_PKG = REPO_ROOT.parent / "booster_k1_locomotion" / "booster_k1_locomotion"
DEFAULT_EXPORT = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2" / "export_onnx"

if not DEPLOY_PKG.is_dir():
    raise SystemExit(f"deploy package not found at {DEPLOY_PKG}; this script drives the real deploy runtime")
sys.path.insert(0, str(DEPLOY_PKG))
from ufo_policy_runtime import (  # noqa: E402
    UfoPolicy,
    ZBank,
    ZController,
    projected_gravity_from_quat,
    quat_rotate_inverse_wxyz,
)

# mjlab SimulationCfg / MujocoCfg defaults as used by make_mjlab_ufo_env_cfg
# (humanoidverse/agents/envs/humanoidverse_mjlab.py:566-585, mjlab/sim/sim.py:86).
MJLAB_OPT = dict(
    integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    solver=mujoco.mjtSolver.mjSOL_NEWTON,
    cone=mujoco.mjtCone.mjCONE_PYRAMIDAL,
    jacobian=mujoco.mjtJacobian.mjJAC_AUTO,
    impratio=1.0,
    iterations=100,
    tolerance=1e-8,
    ls_iterations=50,
    ls_tolerance=0.01,
    ccd_iterations=50,
    gravity=(0.0, 0.0, -9.81),
)


@dataclass
class SimHandles:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_adr: np.ndarray   # per UFO joint, qpos address
    dof_adr: np.ndarray    # per UFO joint, qvel address
    act_ids: np.ndarray    # per UFO joint, actuator id
    root_qpos_adr: int
    root_dof_adr: int


def build_training_equivalent_model(xml_path: Path, spec: dict, *, single_ground: bool = False) -> SimHandles:
    """Compile the same MjModel mjlab compiled during training."""
    child = mujoco.MjSpec.from_file(str(xml_path))
    for actuator in list(child.actuators):
        child.delete(actuator)
    if single_ground:
        for geom in list(child.worldbody.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                child.delete(geom)

    dof_names = spec["dof_names"]
    # mjlab's IdealPdActuator.edit_spec -> create_motor_actuator (utils/spec.py:212):
    # gear 1 torque motors, ctrl/force range = +-effort_limit, joint armature and
    # frictionloss overridden from the training actuator table.
    for i, name in enumerate(dof_names):
        effort = float(spec["effort_limit"][i])
        act = child.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gear[0] = 1.0
        act.forcelimited = True
        act.forcerange[:] = np.array([-effort, effort])
        act.ctrllimited = True
        act.ctrlrange[:] = np.array([-effort, effort])

    parent = mujoco.MjSpec()
    terrain = parent.worldbody.add_body(name="terrain")
    terrain.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.01])
    parent.attach(child, prefix="robot/", frame=parent.worldbody.add_frame())
    model = parent.compile()

    model.opt.integrator = MJLAB_OPT["integrator"]
    model.opt.solver = MJLAB_OPT["solver"]
    model.opt.cone = MJLAB_OPT["cone"]
    model.opt.jacobian = MJLAB_OPT["jacobian"]
    model.opt.timestep = 1.0 / float(spec["physics_fps"])
    model.opt.impratio = MJLAB_OPT["impratio"]
    model.opt.iterations = MJLAB_OPT["iterations"]
    model.opt.tolerance = MJLAB_OPT["tolerance"]
    model.opt.ls_iterations = MJLAB_OPT["ls_iterations"]
    model.opt.ls_tolerance = MJLAB_OPT["ls_tolerance"]
    model.opt.ccd_iterations = MJLAB_OPT["ccd_iterations"]
    model.opt.gravity[:] = MJLAB_OPT["gravity"]

    data = mujoco.MjData(model)
    qpos_adr, dof_adr, act_ids = [], [], []
    for name in dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{name}")
        if jid < 0 or aid < 0:
            raise RuntimeError(f"joint/actuator robot/{name} not found in compiled model")
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
        act_ids.append(aid)

    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/world_joint")
    if root_jid < 0 or model.jnt_type[root_jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("expected a free root joint robot/world_joint")

    return SimHandles(
        model=model,
        data=data,
        qpos_adr=np.asarray(qpos_adr, dtype=np.int64),
        dof_adr=np.asarray(dof_adr, dtype=np.int64),
        act_ids=np.asarray(act_ids, dtype=np.int64),
        root_qpos_adr=int(model.jnt_qposadr[root_jid]),
        root_dof_adr=int(model.jnt_dofadr[root_jid]),
    )


class DcMotorPd:
    """mjlab DcMotorActuator: PD torque, then a velocity-derated torque clamp.

    ``mjlab/actuator/pd_actuator.py:compute`` then ``dc_actuator.py:_clip_effort``.
    Runs once per *physics* step; the position target is held across decimation.
    """

    def __init__(self, spec: dict) -> None:
        self.kp = np.asarray(spec["kp"], dtype=np.float64)
        self.kd = np.asarray(spec["kd"], dtype=np.float64)
        self.effort_limit = np.asarray(spec["effort_limit"], dtype=np.float64)
        self.saturation = self.effort_limit.copy()  # saturation_effort == effort_limit
        self.velocity_limit = np.asarray(spec["velocity_limit"], dtype=np.float64)
        self.vel_at_effort_lim = self.velocity_limit * (1.0 + self.effort_limit / self.saturation)

    def torque(self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        effort = self.kp * (q_des - q) + self.kd * (0.0 - qd)
        v = np.clip(qd, -self.vel_at_effort_lim, self.vel_at_effort_lim)
        top = self.saturation * (1.0 - v / self.velocity_limit)
        bottom = self.saturation * (-1.0 - v / self.velocity_limit)
        return np.clip(effort, np.maximum(bottom, -self.effort_limit), np.minimum(top, self.effort_limit))


def _assert_body_frame_ang_vel(sim: SimHandles) -> None:
    """MuJoCo free-joint ``qvel[3:6]`` is angular velocity in the BODY frame.

    The training env computes ``base_ang_vel = quat_rotate_inverse(base_quat,
    root_ang_vel_world)`` (humanoidverse_mjlab.py:850) where ``root_ang_vel_world``
    comes from ``cvel`` in the world frame. The two must agree, otherwise the
    deploy node would feed a world-frame gyro. Verified, not assumed.
    """
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "robot/world_joint")
    bid = int(sim.model.jnt_bodyid[jid])
    # Run on a throwaway state so the rollout is not perturbed, and so cvel is
    # freshly computed (after mj_step, cvel lags qvel by one integration).
    probe = mujoco.MjData(sim.model)
    rng = np.random.default_rng(0)
    quat = rng.standard_normal(4)
    probe.qpos[sim.root_qpos_adr + 3 : sim.root_qpos_adr + 7] = quat / np.linalg.norm(quat)
    probe.qvel[sim.root_dof_adr : sim.root_dof_adr + 6] = rng.standard_normal(6)
    mujoco.mj_forward(sim.model, probe)

    world6 = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, probe, mujoco.mjtObj.mjOBJ_BODY, bid, world6, 0)
    quat_wxyz = probe.qpos[sim.root_qpos_adr + 3 : sim.root_qpos_adr + 7]
    # mj_objectVelocity returns [angular, linear]; rotate the angular part into body frame.
    rotated = quat_rotate_inverse_wxyz(quat_wxyz, world6[0:3])
    local = probe.qvel[sim.root_dof_adr + 3 : sim.root_dof_adr + 6]
    if not np.allclose(rotated, local, atol=1e-5, rtol=1e-4):
        raise AssertionError(f"base_ang_vel frame mismatch: rotated_world={rotated}, qvel_local={local}")


def set_fallen_pose(sim: SimHandles, spec: dict, *, face_down: bool, height: float = 0.5) -> None:
    """Mirror the training ``lie_down_init`` branch (humanoidverse_mjlab.py:1078).

    Training rotates the motion's root by +-pi/2 about world X and sets z=0.5.
    Here we start from the default joint pose instead of a motion frame.
    """
    mujoco.mj_resetData(sim.model, sim.data)
    q = sim.data.qpos
    q[sim.root_qpos_adr : sim.root_qpos_adr + 3] = [0.0, 0.0, height]
    half = (np.pi / 2.0) * (1.0 if face_down else -1.0) / 2.0
    q[sim.root_qpos_adr + 3 : sim.root_qpos_adr + 7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    q[sim.qpos_adr] = np.asarray(spec["default_joint_angles"], dtype=np.float64)
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def run(
    z_name: str = "getup_opt",
    export_dir: Path = DEFAULT_EXPORT,
    z_bank: Path | None = None,
    video: Path | None = None,
    trace: Path | None = None,
    seconds: float = 6.0,
    settle_seconds: float = 1.0,
    face_down: bool = False,
    single_ground: bool = False,
    blend_steps: int = 25,
    fps: int = 50,
    width: int = 640,
    height_px: int = 480,
) -> dict:
    export_dir = Path(export_dir).expanduser().resolve()
    spec = json.loads((export_dir / "deploy_spec.json").read_text())
    bank_path = Path(z_bank).expanduser() if z_bank else export_dir / "z_bank.npz"

    sim = build_training_equivalent_model(Path(spec["xml_path"]), spec, single_ground=single_ground)
    pd = DcMotorPd(spec)
    decimation = int(spec["control_decimation"])
    control_dt = decimation * sim.model.opt.timestep

    bank = ZBank.load(bank_path)
    if "@" in z_name:
        # ``seq:<name>@<idx>`` -- pin a single frame of a latent sequence as a static
        # z. Handy for hunting a z that both gets up AND holds the stand.
        seq_name, _, idx = z_name.partition("@")
        controller = ZController(bank, blend_steps=blend_steps)
        controller.set_z(bank.sequences[seq_name.removeprefix("seq:")][int(idx)], immediate=True, name=z_name)
    else:
        controller = ZController(bank, initial=z_name, blend_steps=blend_steps)
    policy = UfoPolicy(export_dir / spec["policy_onnx"], controller)

    planes = [
        mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(sim.model.ngeom)
        if sim.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    print(f"[INFO] ground planes in scene: {planes}")
    print(f"[INFO] timestep={sim.model.opt.timestep} decimation={decimation} control_dt={control_dt}")
    print(f"[INFO] z='{z_name}' from {bank_path.name}; available {sorted(bank.names())}")

    default_pose = np.asarray(spec["default_joint_angles"], dtype=np.float64)
    set_fallen_pose(sim, spec, face_down=face_down)

    renderer = None
    frames: list[np.ndarray] = []
    if video is not None:
        renderer = mujoco.Renderer(sim.model, height=height_px, width=width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth, cam.lookat[:] = 3.0, -15.0, 135.0, [0.0, 0.0, 0.4]

    def render():
        if renderer is None:
            return
        cam.lookat[:2] = sim.data.qpos[sim.root_qpos_adr : sim.root_qpos_adr + 2]
        renderer.update_scene(sim.data, cam)
        frames.append(renderer.render())

    def physics(q_des: np.ndarray) -> None:
        for _ in range(decimation):
            q = sim.data.qpos[sim.qpos_adr]
            qd = sim.data.qvel[sim.dof_adr]
            sim.data.ctrl[sim.act_ids] = pd.torque(q_des, q, qd)
            mujoco.mj_step(sim.model, sim.data)

    # Settle: hold the default pose (== zero action) so the robot comes to rest
    # on the ground before the policy is engaged.
    for _ in range(int(settle_seconds / control_dt)):
        physics(default_pose)
        render()
    settle_height = float(sim.data.qpos[sim.root_qpos_adr + 2])

    _assert_body_frame_ang_vel(sim)

    policy.reset()
    heights, uprightness, z_norms = [], [], []
    dof_pos_log = []
    n_steps = int(seconds / control_dt)
    for _ in range(n_steps):
        q = sim.data.qpos[sim.qpos_adr].copy()
        qd = sim.data.qvel[sim.dof_adr].copy()
        quat_wxyz = sim.data.qpos[sim.root_qpos_adr + 3 : sim.root_qpos_adr + 7].copy()
        base_ang_vel = sim.data.qvel[sim.root_dof_adr + 3 : sim.root_dof_adr + 6].copy()
        gravity = projected_gravity_from_quat(quat_wxyz)

        q_des = policy.step(q, qd, base_ang_vel, gravity)
        physics(q_des.astype(np.float64))
        render()

        heights.append(float(sim.data.qpos[sim.root_qpos_adr + 2]))
        uprightness.append(-float(gravity[2]))  # 1.0 = trunk upright, 0 = horizontal
        z_norms.append(float(np.linalg.norm(policy.last_z)))
        dof_pos_log.append(sim.data.qpos[sim.qpos_adr].copy())
        if not np.all(np.isfinite(sim.data.qpos)):
            print("[WARN] simulation diverged")
            break

    heights_a = np.asarray(heights)
    upright_a = np.asarray(uprightness)
    target_h = float(spec["init_root_pos"][2])
    # "Standing" = trunk near-vertical AND root near the nominal standing height.
    standing = (upright_a > 0.9) & (heights_a > 0.8 * target_h)
    first = int(np.argmax(standing)) if standing.any() else -1
    result = {
        "z_name": z_name,
        "steps": len(heights),
        "settle_root_height": settle_height,
        "target_root_height": target_h,
        "final_root_height": float(heights_a[-1]) if len(heights_a) else float("nan"),
        "max_root_height": float(heights_a.max()) if len(heights_a) else float("nan"),
        "final_uprightness": float(upright_a[-1]) if len(upright_a) else float("nan"),
        "max_uprightness": float(upright_a.max()) if len(upright_a) else float("nan"),
        "z_norm": z_norms[-1] if z_norms else float("nan"),
        # Reached a standing pose at any point in the rollout.
        "stood_up_any": bool(standing.any()),
        "time_to_stand_s": float(first * control_dt) if first >= 0 else None,
        "standing_fraction": float(standing.mean()) if len(standing) else 0.0,
        # Still standing at the end of the rollout.
        "stood_up": bool(standing[-1]) if len(standing) else False,
    }
    if trace is not None:
        trace = Path(trace).expanduser()
        trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(trace),
            root_height=heights_a,
            uprightness=upright_a,
            standing=standing,
            dof_pos=np.asarray(dof_pos_log),
            dof_names=np.asarray(spec["dof_names"]),
        )
        print(f"[INFO] wrote trace {trace}")

    if renderer is not None and frames:
        import mediapy as media

        try:  # no system ffmpeg on headless boxes; imageio-ffmpeg ships a static one
            import imageio_ffmpeg

            media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pass
        video = Path(video).expanduser()
        video.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(video), frames, fps=fps)
        print(f"[INFO] wrote {video} ({len(frames)} frames)")

    print("[RESULT] " + json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import tyro

    tyro.cli(run)
