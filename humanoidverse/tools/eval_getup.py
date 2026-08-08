"""Skill evaluation harness for UFO Behavior Foundation Models (FB-CPR), with a get-up task.

The harness answers one question: *given a trained BFM actor, which latent ``z``
reproduces a skill closed-loop, without a reference motion?*

Layout (the generic/skill seam matters -- walk/run/kick reuse the generic half):

  generic
    ``build_eval_env``      MJLab env construction, shared with the inference entrypoints.
    ``ZProvider``           constant z / time-indexed z sequence replay / two-phase schedule.
    ``rollout``             seeded reset -> settle window -> policy rollout, records raw traces.
    ``Trajectory``          raw per-step traces (root height, uprightness, contacts, torques, qpos...).
    ``LatencyBuffer``       harness-side observation/action latency (MJLab has no ctrl-delay term).
    ``aggregate``/``write_*``  CSV + JSON reporting, mp4 rendering of chosen episodes.

  skill-specific (get-up)
    ``FallenPoseBank``      in-distribution (``lie_down_init`` replica) and OOD fallen poses.
    ``StandCriterion``      explicit standing definition (height + uprightness + both feet down).
    ``score_getup``         per-episode success / time-to-stand / stability / effort metrics.

Another skill supplies its own pose bank + criterion + scorer and reuses everything else.

Example
-------
    uv run python -m humanoidverse.tools.eval_getup \
        --model-folder runs/ufo_fb_k1_5090_v2 \
        --candidates runs/getup_eval/candidates.json \
        --bucket indist --num-envs 64 --disable-dr True --disable-obs-noise True \
        --out-dir runs/getup_eval/indist_nominal
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import torch

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer,
    add_bool_arg,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    resolve_inference_robot_config,
)
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_from_angle_axis, quat_mul

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_CONFIG = "configs/robots/k1_22dof.yaml"
DEFAULT_DATA_PATH = "cache/motion_data/k1_lafan1/k1_lafan1_full_ufo.pkl"

# ``lie_down_init`` in humanoidverse/agents/envs/humanoidverse_mjlab.py:1084-1097.
LIE_DOWN_ROOT_HEIGHT = 0.5

# Leg pose the existing IsaacLab walk policy settles into at zero velocity command
# (measured by WS-D, robot fully settled). The get-up skill has to hand off to that policy, so
# how close a candidate's terminal stance is to this pose is a deployment-relevant tie-breaker.
# NB this is *not* the config's declared default (knee 0.52) -- the walk policy holds a more
# crouched stance. Left/right signs follow the sign convention of the default pose in
# configs/robots/k1_22dof.yaml (left roll/yaw positive, left ankle-roll negative).
WALK_HANDOFF_TARGET: dict[str, float] = {
    "Left_Hip_Pitch": -0.452, "Right_Hip_Pitch": -0.452,
    "Left_Hip_Roll": 0.044, "Right_Hip_Roll": -0.044,
    "Left_Hip_Yaw": 0.108, "Right_Hip_Yaw": -0.108,
    "Left_Knee_Pitch": 0.792, "Right_Knee_Pitch": 0.792,
    "Left_Ankle_Pitch": -0.313, "Right_Ankle_Pitch": -0.313,
    "Left_Ankle_Roll": -0.011, "Right_Ankle_Roll": 0.011,
}


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


@dataclass
class StandCriterion:
    """Explicit definition of "standing" for the K1.

    Calibrated against the retargeted LAFAN1 K1 dataset: the Trunk (root) height is
    ~0.535 m median while walking (1st percentile 0.461), ~0.31 m median in the
    ``ground2`` sitting/kneeling clip, and <0.17 m when fallen.  ``root_height=0.45``
    therefore separates standing from kneeling/sitting with margin on both sides.
    The threshold is a judgement call -- it is reported in the JSON summary so
    results can be re-read against a different definition.
    """

    root_height: float = 0.45
    upright: float = 0.90  # -projected_gravity_z == cos(tilt); 0.90 -> tilt <= 25.8 deg
    foot_force: float = 5.0  # N, per foot (K1 weighs 19.67 kg -> ~193 N)
    confirm_s: float = 0.5  # must hold continuously this long to count as "stood up"
    hold_s: float = 2.0  # trailing window that must stay standing for success
    hold_frac: float = 0.95  # fraction of the trailing window that must satisfy the criterion


@dataclass
class RolloutConfig:
    settle_steps: int = 50  # 1.0 s of physics before the policy takes over
    episode_steps: int = 500  # 10.0 s of policy control
    action_latency_max: int = 0  # per-env action delay drawn from {0..max}
    obs_latency_max: int = 0  # per-env observation delay drawn from {0..max}
    record_qpos: bool = True


@dataclass
class PoseBankConfig:
    bucket: str = "indist"
    motion_pool: list[int] | None = None  # in-dist: motion ids to draw reference frames from
    ood_tilts: tuple[str, ...] = ("supine", "prone", "side_left", "side_right")
    ood_yaws: int = 8  # yaw values swept uniformly over [-pi, pi)
    ood_drop_height: float = 0.30
    ood_joint_noise: float = 0.0  # rad, uniform; >0 randomizes the joint pose
    # bucket="motion_frame": reset to an exact motion-library frame, no lie-down transform.
    # Used as the aligned-replay control (the closed-loop analogue of the tracking eval).
    motion_frame_id: int = 0
    motion_frame_time_s: float = 0.0
    motion_frame_jitter_s: float = 0.0


# --------------------------------------------------------------------------------------
# generic: environment
# --------------------------------------------------------------------------------------


def build_eval_env(
    *,
    model_folder: Path,
    data_path: Path,
    robot_config: Path,
    device: str,
    num_envs: int,
    disable_dr: bool,
    disable_obs_noise: bool,
    seed: int,
    hydra_overrides: Sequence[str] | None = None,
):
    """Build the MJLab rollout env exactly like the inference entrypoints do.

    ``max_episode_length_s`` is set far beyond any episode we run so the only
    termination term (``time_out``) never fires; ``rollout`` asserts this.
    """
    env_cfg, use_root_height_obs = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        headless=True,
        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=1.0e5,
    )
    update: dict[str, Any] = {"seed": int(seed)}
    if hydra_overrides:
        # Lets a caller ablate individual DR terms, e.g. domain_rand.push_robots=False.
        update["hydra_overrides"] = list(env_cfg.hydra_overrides) + list(hydra_overrides)
    env_cfg = env_cfg.model_copy(update=update)  # the pydantic config is frozen
    wrapped_env, _ = env_cfg.build(num_envs=num_envs)
    core = wrapped_env._env
    core._motion_lib.load_all_motions()
    core.is_evaluating = True
    return wrapped_env, core, env_cfg, use_root_height_obs


# --------------------------------------------------------------------------------------
# generic: z providers
# --------------------------------------------------------------------------------------


class ZProvider:
    """Maps a policy step index to a (num_envs, z_dim) latent."""

    name: str = "z"
    mode: str = "constant"

    def z_at(self, step: int, num_envs: int) -> torch.Tensor:
        raise NotImplementedError

    def representative_z(self) -> torch.Tensor:
        """Single 256-d vector that best summarizes this provider (for the z-bank)."""
        return self.z_at(0, 1)[0]

    def bank_payload(self, dt: float) -> dict[str, Any]:
        """Raw tensors sufficient to *replay this provider* without the model or motion lib.

        The z-bank is consumed by the deploy node, so a spec dict is not enough -- rebuilding
        a motion-derived z would need the checkpoint and the motion library at deploy time.
        """
        return {"z": self.representative_z().detach().cpu().float()}


class ConstantZ(ZProvider):
    mode = "constant"

    def __init__(self, z: torch.Tensor, name: str):
        self._z = z.reshape(1, -1)
        self.name = name

    def z_at(self, step: int, num_envs: int) -> torch.Tensor:
        return self._z.expand(num_envs, -1)


class SequenceZ(ZProvider):
    """Open-loop, time-indexed replay of a z sequence. Holds the last value past the end."""

    mode = "sequence"

    def __init__(self, z_seq: torch.Tensor, name: str, *, loop: bool = False):
        self._seq = z_seq.reshape(z_seq.shape[0], -1)
        self.name = name
        self.loop = bool(loop)

    def z_at(self, step: int, num_envs: int) -> torch.Tensor:
        n = self._seq.shape[0]
        idx = step % n if self.loop else min(step, n - 1)
        return self._seq[idx : idx + 1].expand(num_envs, -1)

    def representative_z(self) -> torch.Tensor:
        return self._seq[0]

    def bank_payload(self, dt: float) -> dict[str, Any]:
        seq = self._seq.detach().cpu().float()
        return {"z": seq[0], "z_sequence": seq, "loop": self.loop, "dt": dt}


class TwoPhaseZ(ZProvider):
    """``phase_a`` for ``switch_step`` policy steps, then ``phase_b``."""

    mode = "two_phase"

    def __init__(self, phase_a: ZProvider, phase_b: ZProvider, switch_step: int, name: str):
        self.a = phase_a
        self.b = phase_b
        self.switch_step = int(switch_step)
        self.name = name

    def z_at(self, step: int, num_envs: int) -> torch.Tensor:
        if step < self.switch_step:
            return self.a.z_at(step, num_envs)
        return self.b.z_at(step - self.switch_step, num_envs)

    def representative_z(self) -> torch.Tensor:
        return self.a.representative_z()

    def bank_payload(self, dt: float) -> dict[str, Any]:
        return {
            "z": self.a.representative_z().detach().cpu().float(),
            "phase_a": self.a.bank_payload(dt),
            "phase_b": self.b.bank_payload(dt),
            "switch_step": self.switch_step,
            "switch_s": self.switch_step * dt,
            "dt": dt,
        }


# --------------------------------------------------------------------------------------
# generic: latency
# --------------------------------------------------------------------------------------


class LatencyBuffer:
    """Per-env delay line. MJLab drops ``randomize_ctrl_delay``, so latency lives here."""

    def __init__(self, delays: torch.Tensor):
        self.delays = delays.long()
        self.max_delay = int(self.delays.max().item()) if self.delays.numel() else 0
        self._history: list[torch.Tensor] = []

    def push_pop(self, value: torch.Tensor) -> torch.Tensor:
        if self.max_delay == 0:
            return value
        self._history.append(value)
        if len(self._history) > self.max_delay + 1:
            self._history.pop(0)
        # history[-1] is the newest; delay d selects history[-1 - d] (clamped while warming up).
        stacked = torch.stack(self._history, dim=0)  # (H, N, D)
        h = stacked.shape[0]
        pick = torch.clamp(h - 1 - self.delays, min=0)
        idx = pick.view(1, -1, *([1] * (value.dim() - 1))).expand(1, *value.shape)
        return torch.gather(stacked, 0, idx).squeeze(0)

    def reset(self) -> None:
        self._history = []


class DictLatencyBuffer:
    def __init__(self, delays: torch.Tensor):
        self.delays = delays
        self._buffers: dict[str, LatencyBuffer] = {}

    def push_pop(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in obs.items():
            if key not in self._buffers:
                self._buffers[key] = LatencyBuffer(self.delays)
            out[key] = self._buffers[key].push_pop(value) if torch.is_floating_point(value) else value
        return out

    def reset(self) -> None:
        for buffer in self._buffers.values():
            buffer.reset()


# --------------------------------------------------------------------------------------
# skill-specific (get-up): fallen pose bank
# --------------------------------------------------------------------------------------


def _tilt_quat(kind: str, device: str) -> torch.Tensor:
    """World-frame tilt applied to an upright robot. Body x = chest normal, z = head."""
    ang, axis = {
        # chest normal -> +z (face up); head along -x before yaw
        "supine": (-math.pi / 2, [0.0, 1.0, 0.0]),
        # chest normal -> -z (face down)
        "prone": (math.pi / 2, [0.0, 1.0, 0.0]),
        # body up -> -y / +y
        "side_right": (math.pi / 2, [1.0, 0.0, 0.0]),
        "side_left": (-math.pi / 2, [1.0, 0.0, 0.0]),
        "upright": (0.0, [0.0, 0.0, 1.0]),
    }[kind]
    angle = torch.tensor([ang], device=device, dtype=torch.float32)
    return quat_from_angle_axis(angle, torch.tensor([axis], device=device, dtype=torch.float32), w_last=True)


class FallenPoseBank:
    """Fallen initial states, in two clearly separated buckets.

    ``indist`` replicates ``lie_down_init`` exactly: take a motion-library reference
    frame (including its velocities and xy offset), force root z to 0.5, and premultiply
    the root rotation by a +-90 deg rotation about the *world* x axis.  Training draws
    the sign once per reset batch; the harness draws it per env, which leaves the
    per-episode marginal identical and only decorrelates episodes within a batch.
    Training also adds no initial noise (``noise_to_initial_level=0``).

    ``ood`` builds a fallen pose from scratch: an upright default (or joint-randomized)
    pose tilted supine / prone / side-lying and then swept over root yaw.  The yaw sweep
    is the meaningful OOD axis: for an upright source frame, ``lie_down_init`` always
    leaves the head pointing along world +-y, so head direction -- not chest
    orientation -- is what training never varied.
    """

    def __init__(self, core, cfg: PoseBankConfig):
        self.core = core
        self.cfg = cfg
        self.device = core.device
        num_motions = int(core._motion_lib._num_unique_motions)
        pool = cfg.motion_pool if cfg.motion_pool else list(range(num_motions))
        bad = [m for m in pool if m < 0 or m >= num_motions]
        if bad:
            raise ValueError(f"motion_pool entries out of range [0,{num_motions}): {bad}")
        self.pool = np.asarray(pool, dtype=np.int64)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        if self.cfg.bucket == "indist":
            return self._sample_indist(n, rng)
        if self.cfg.bucket == "ood":
            return self._sample_ood(n, rng)
        if self.cfg.bucket == "motion_frame":
            return self._sample_motion_frame(n, rng)
        raise ValueError(f"Unknown bucket: {self.cfg.bucket}")

    def _sample_motion_frame(self, n: int, rng: np.random.Generator):
        """Exact reference frame, untransformed -- the aligned control for z-sequence replay."""
        lib = self.core._motion_lib
        cfg = self.cfg
        motion_ids = torch.full((n,), int(cfg.motion_frame_id), device=self.device, dtype=torch.long)
        jitter = rng.uniform(-cfg.motion_frame_jitter_s, cfg.motion_frame_jitter_s, size=n) if cfg.motion_frame_jitter_s > 0 else np.zeros(n)
        times = torch.as_tensor(
            np.clip(cfg.motion_frame_time_s + jitter, 0.0, float(lib._motion_lengths[cfg.motion_frame_id])),
            device=self.device,
            dtype=torch.float32,
        )
        res = lib.get_motion_state(motion_ids, times, offset=self.core.env_origins[:n])
        root_states = torch.cat([res["root_pos"], res["root_rot"], res["root_vel"], res["root_ang_vel"]], dim=-1)
        dof_states = torch.stack([res["dof_pos"], res["dof_vel"]], dim=-1)
        meta = [
            {"bucket": "motion_frame", "motion_id": int(cfg.motion_frame_id), "motion_time": float(times[i])}
            for i in range(n)
        ]
        return {"root_states": root_states, "dof_states": dof_states}, meta

    def _sample_indist(self, n: int, rng: np.random.Generator):
        lib = self.core._motion_lib
        motion_ids = torch.as_tensor(rng.choice(self.pool, size=n), device=self.device, dtype=torch.long)
        lengths = lib.get_motion_length(motion_ids)
        phases = torch.as_tensor(rng.random(n), device=self.device, dtype=torch.float32)
        times = phases * lengths
        res = lib.get_motion_state(motion_ids, times, offset=self.core.env_origins[:n])

        root_pos = res["root_pos"].clone()
        root_rot = res["root_rot"].clone()
        root_pos[:, 2] = LIE_DOWN_ROOT_HEIGHT
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        angles = torch.as_tensor(signs * (-math.pi / 2), device=self.device, dtype=torch.float32)
        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, 3)
        rot_quat = quat_from_angle_axis(angles, axis, w_last=True)
        root_rot = quat_mul(rot_quat, root_rot, w_last=True)

        root_states = torch.cat([root_pos, root_rot, res["root_vel"], res["root_ang_vel"]], dim=-1)
        dof_states = torch.stack([res["dof_pos"], res["dof_vel"]], dim=-1)
        meta = [
            {"bucket": "indist", "motion_id": int(motion_ids[i]), "motion_time": float(times[i]), "roll_sign": float(signs[i])}
            for i in range(n)
        ]
        return {"root_states": root_states, "dof_states": dof_states}, meta

    def _sample_ood(self, n: int, rng: np.random.Generator):
        cfg = self.cfg
        tilts = list(cfg.ood_tilts)
        yaw_grid = np.linspace(-math.pi, math.pi, cfg.ood_yaws, endpoint=False)
        # Deterministic sweep so every (tilt, yaw) cell is covered evenly.
        cells = [(t, float(y)) for t in tilts for y in yaw_grid]
        order = rng.permutation(len(cells))
        chosen = [cells[order[i % len(cells)]] for i in range(n)]

        quats = []
        for tilt, yaw in chosen:
            q_yaw = quat_from_angle_axis(
                torch.tensor([yaw], device=self.device, dtype=torch.float32),
                torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32),
                w_last=True,
            )
            quats.append(quat_mul(q_yaw, _tilt_quat(tilt, self.device), w_last=True))
        root_rot = torch.cat(quats, dim=0)

        root_pos = self.core.env_origins[:n].clone()
        root_pos[:, 2] = cfg.ood_drop_height
        zeros = torch.zeros((n, 3), device=self.device, dtype=torch.float32)
        root_states = torch.cat([root_pos, root_rot, zeros, zeros], dim=-1)

        dof_pos = self.core.default_dof_pos[:n].clone()
        if cfg.ood_joint_noise > 0.0:
            noise = torch.as_tensor(
                rng.uniform(-cfg.ood_joint_noise, cfg.ood_joint_noise, size=dof_pos.shape),
                device=self.device,
                dtype=torch.float32,
            )
            lower = self.core.hard_dof_pos_limits[:, 0].unsqueeze(0)
            upper = self.core.hard_dof_pos_limits[:, 1].unsqueeze(0)
            dof_pos = torch.clamp(dof_pos + noise, lower, upper)
        dof_states = torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1)
        meta = [{"bucket": "ood", "tilt": chosen[i][0], "yaw": chosen[i][1]} for i in range(n)]
        return {"root_states": root_states, "dof_states": dof_states}, meta


# --------------------------------------------------------------------------------------
# generic: rollout
# --------------------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Raw per-step traces, (T, N, ...) numpy. Task-agnostic."""

    root_height: np.ndarray
    upright: np.ndarray
    foot_force: np.ndarray
    undesired_contact: np.ndarray
    torque: np.ndarray
    dof_vel: np.ndarray
    action: np.ndarray
    qpos: np.ndarray | None
    dt: float
    settle_root_speed: np.ndarray
    settle_ang_speed: np.ndarray
    meta: list[dict[str, Any]]
    env_origins: np.ndarray


@torch.no_grad()
def rollout(
    wrapped_env,
    core,
    model,
    *,
    z_provider: ZProvider,
    pose_bank: FallenPoseBank,
    cfg: RolloutConfig,
    rng: np.random.Generator,
    device: str,
) -> Trajectory:
    num_envs = core.num_envs
    target_states, meta = pose_bank.sample(num_envs, rng)
    observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)

    action_delays = torch.as_tensor(
        rng.integers(0, cfg.action_latency_max + 1, size=num_envs), device=device, dtype=torch.long
    )
    obs_delays = torch.as_tensor(rng.integers(0, cfg.obs_latency_max + 1, size=num_envs), device=device, dtype=torch.long)
    action_buffer = LatencyBuffer(action_delays)
    obs_buffer = DictLatencyBuffer(obs_delays)

    zero_action = torch.zeros((num_envs, core.num_dof), device=device, dtype=torch.float32)
    for _ in range(cfg.settle_steps):
        observation, _r, terminated, truncated, _info = wrapped_env.step(zero_action, to_numpy=False)
        _assert_no_reset(terminated, truncated)

    settle_root_speed = torch.norm(core.robot_root_states[:, 7:10], dim=-1).cpu().numpy()
    settle_ang_speed = torch.norm(core.robot_root_states[:, 10:13], dim=-1).cpu().numpy()

    feet = core.feet_indices
    penalised = core.penalised_contact_indices
    traces: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("root_height", "upright", "foot_force", "undesired_contact", "torque", "dof_vel", "action", "qpos")
    }
    reset_flags: list[torch.Tensor] = []

    # Traces stay on the GPU and are transferred once at the end: a per-step .cpu() on each
    # of ~10 tensors forces ~10 device syncs per step and dominated the rollout cost.
    for step in range(cfg.episode_steps):
        policy_obs = obs_buffer.push_pop(observation)
        z = z_provider.z_at(step, num_envs)
        action = model.act(policy_obs, z, mean=True)
        applied = action_buffer.push_pop(action)
        observation, _r, terminated, truncated, _info = wrapped_env.step(applied, to_numpy=False)
        reset_flags.append(terminated.bool() | truncated.bool())

        traces["root_height"].append(core.robot_root_states[:, 2].clone())
        traces["upright"].append(-core.projected_gravity[:, 2].clone())
        traces["foot_force"].append(torch.norm(core.contact_forces[:, feet, :], dim=-1))
        if penalised.numel() > 0:
            # Matches the training penalty: any |component| > 1 N on a penalized body.
            undesired = torch.any(torch.abs(core.contact_forces[:, penalised, :]) > 1.0, dim=2).any(dim=1)
        else:
            undesired = torch.zeros(num_envs, dtype=torch.bool, device=device)
        traces["undesired_contact"].append(undesired)
        traces["torque"].append(core.torques.clone())
        traces["dof_vel"].append(core.dof_vel.clone())
        traces["action"].append(core.actions.clone())
        if cfg.record_qpos:
            qpos, _qvel = wrapped_env._get_qpos_qvel(to_numpy=False)
            traces["qpos"].append(qpos.clone())

    if bool(torch.stack(reset_flags).any()):
        _raise_reset_error()
    stacked = {
        k: (torch.stack(v, dim=0).detach().cpu().numpy() if v else None) for k, v in traces.items()
    }
    # Root height is absolute; env origins are all at z=0 for a plane terrain, but subtract
    # anyway so the metric is terrain-relative if that ever changes.
    origin_z = core.env_origins[:, 2].cpu().numpy()
    stacked["root_height"] = stacked["root_height"] - origin_z[None, :]
    return Trajectory(
        root_height=stacked["root_height"],
        upright=stacked["upright"],
        foot_force=stacked["foot_force"],
        undesired_contact=stacked["undesired_contact"],
        torque=stacked["torque"],
        dof_vel=stacked["dof_vel"],
        action=stacked["action"],
        qpos=stacked["qpos"],
        dt=float(core.dt),
        settle_root_speed=settle_root_speed,
        settle_ang_speed=settle_ang_speed,
        meta=meta,
        env_origins=core.env_origins.cpu().numpy(),
    )


def _raise_reset_error() -> None:
    raise RuntimeError(
        "Env reported terminated/truncated mid-episode. The harness relies on "
        "max_episode_length_s being large and all termination terms disabled; a reset "
        "would silently resample the motion library and corrupt the episode."
    )


def _assert_no_reset(terminated, truncated) -> None:
    if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
        _raise_reset_error()


# --------------------------------------------------------------------------------------
# skill-specific (get-up): scoring
# --------------------------------------------------------------------------------------


def _first_sustained_true(mask: np.ndarray, window: int) -> int:
    """First index t such that mask[t:t+window] is all True, else -1. mask is (T,)."""
    if window <= 0:
        window = 1
    T = mask.shape[0]
    if T < window:
        return -1
    csum = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    counts = csum[window:] - csum[:-window]
    hits = np.nonzero(counts == window)[0]
    return int(hits[0]) if hits.size else -1


def score_getup(traj: Trajectory, crit: StandCriterion, effort_limits: np.ndarray) -> list[dict[str, Any]]:
    T, N = traj.root_height.shape
    dt = traj.dt
    confirm = max(1, int(round(crit.confirm_s / dt)))
    hold = max(1, int(round(crit.hold_s / dt)))

    both_feet = (traj.foot_force > crit.foot_force).all(axis=2)  # (T, N)
    upright_ok = (traj.root_height >= crit.root_height) & (traj.upright >= crit.upright)
    stand_ok = upright_ok & both_feet

    rows: list[dict[str, Any]] = []
    for i in range(N):
        ok = stand_ok[:, i]
        first = _first_sustained_true(ok, confirm)
        reached = first >= 0
        # Time-to-stand is measured from policy takeover (post-settle) to the first
        # instant of the sustained standing window.
        tts = float(first * dt) if reached else float("nan")

        tail = ok[max(0, T - hold) :]
        held = float(tail.mean())
        success = bool(reached and held >= crit.hold_frac)

        # Relaxed variant: torso up and upright, ignoring momentary double-support loss.
        # Under perturbation (DR pushes every 1-3 s) a robot that stays standing still takes
        # stabilising steps, which breaks the strict both-feet term without it falling. Both
        # numbers are reported so "did not stay perfectly planted" is not read as "fell over".
        up = upright_ok[:, i]
        first_up = _first_sustained_true(up, confirm)
        rose = first_up >= 0
        tail_up_ok = up[max(0, T - hold) :]
        held_upright = float(tail_up_ok.mean())
        success_upright = bool(rose and held_upright >= crit.hold_frac)

        tail_h = traj.root_height[max(0, T - hold) :, i]
        tail_up = np.clip(traj.upright[max(0, T - hold) :, i], -1.0, 1.0)
        # NB: `refell` is loss of the *strict* stance (incl. double support) at the final step --
        # under repeated pushes that is usually a stabilising step, not a fall. `fell_back` is the
        # honest "ended up down" flag: torso below the height/tilt thresholds at the end.
        refell = bool(reached and not ok[-1])
        fell_back = bool(rose and not up[-1])

        torque = np.abs(traj.torque[:, i, :])
        peak_torque = float(torque.max())
        sat = float((torque >= 0.98 * effort_limits[None, :]).mean())
        peak_ratio = float((torque / effort_limits[None, :]).max())
        power = float(np.abs(traj.torque[:, i, :] * traj.dof_vel[:, i, :]).sum(axis=1).mean())

        post = slice(first, T) if reached else slice(0, 0)
        rows.append(
            {
                "success": success,
                "success_upright": success_upright,
                "rose": bool(rose),
                "time_to_rise_s": float(first_up * dt) if rose else float("nan"),
                "held_upright_frac": held_upright,
                # Per-criterion occupancy makes every run self-diagnosing: a crouch-stall
                # (height fails), a sensor problem (feet fail) and a tilt problem look different.
                "frac_height_ok": float((traj.root_height[:, i] >= crit.root_height).mean()),
                "frac_upright_ok": float((traj.upright[:, i] >= crit.upright).mean()),
                "frac_feet_ok": float(both_feet[:, i].mean()),
                "frac_stand_ok": float(ok.mean()),
                "mean_foot_force": float(traj.foot_force[:, i, :].mean()),
                "reached_stand": bool(reached),
                "time_to_stand_s": tts,
                "hold_frac": held,
                "refell": refell,
                "fell_back": fell_back,
                "final_root_height": float(traj.root_height[-1, i]),
                "max_root_height": float(traj.root_height[:, i].max()),
                "final_upright": float(traj.upright[-1, i]),
                "hold_root_height_std": float(tail_h.std()),
                "hold_max_tilt_deg": float(np.degrees(np.arccos(np.clip(tail_up.min(), -1.0, 1.0)))),
                "peak_torque_nm": peak_torque,
                "peak_torque_ratio": peak_ratio,
                "torque_saturation_frac": sat,
                "mean_abs_power_w": power,
                "undesired_contact_frac": float(traj.undesired_contact[:, i].mean()),
                "undesired_contact_after_stand_frac": (
                    float(traj.undesired_contact[post, i].mean()) if reached else float("nan")
                ),
                "settle_root_speed": float(traj.settle_root_speed[i]),
                "settle_ang_speed": float(traj.settle_ang_speed[i]),
            }
        )
    return rows


def add_handoff_metrics(
    traj: Trajectory,
    rows: list[dict[str, Any]],
    dof_names: Sequence[str],
    *,
    hold_s: float,
    target: dict[str, float] = WALK_HANDOFF_TARGET,
) -> None:
    """Terminal leg pose (mean over the hold window) and its distance to the walk hand-off pose.

    Post-processing over qpos that the rollout already recorded -- no extra simulation.
    Only meaningful for episodes that actually ended standing, so it is NaN otherwise.
    """
    if traj.qpos is None:
        return
    hold = max(1, int(round(hold_s / traj.dt)))
    dof = traj.qpos[-hold:, :, 7:]  # (hold, N, ndof), control-joint order
    idx = [dof_names.index(j) for j in target if j in dof_names]
    names = [j for j in target if j in dof_names]
    if not idx:
        return
    tgt = np.asarray([target[j] for j in names], dtype=np.float64)
    mean_pose = dof[:, :, idx].mean(axis=0)  # (N, n_leg)
    for i, row in enumerate(rows):
        if not row.get("reached_stand"):
            row["handoff_max_abs_delta"] = float("nan")
            row["handoff_rms_delta"] = float("nan")
            row["handoff_knee_delta"] = float("nan")
            continue
        delta = mean_pose[i] - tgt
        row["handoff_max_abs_delta"] = float(np.abs(delta).max())
        row["handoff_rms_delta"] = float(np.sqrt((delta ** 2).mean()))
        knee = [k for k, j in enumerate(names) if "Knee" in j]
        row["handoff_knee_delta"] = float(np.mean(delta[knee])) if knee else float("nan")
        for k, j in enumerate(names):
            row[f"pose_{j}"] = float(mean_pose[i][k])


def getup_score(rows: Sequence[dict[str, Any]], episode_s: float) -> float:
    """Documented, arbitrary composite. Sort by ``success_rate`` first; this is the tiebreak.

    100 * success_rate + 10 * speed_bonus + 5 * stability_bonus, range [0, 115].
    """
    if not rows:
        return 0.0
    success = np.array([r["success"] for r in rows], dtype=float)
    rate = float(success.mean())
    ok = [r for r in rows if r["success"]]
    if ok:
        speed = float(np.mean([max(0.0, 1.0 - r["time_to_stand_s"] / 5.0) for r in ok]))
        stab = float(np.mean([max(0.0, 1.0 - r["hold_max_tilt_deg"] / 45.0) for r in ok]))
    else:
        speed = stab = 0.0
    return 100.0 * rate + 10.0 * speed + 5.0 * stab


# --------------------------------------------------------------------------------------
# generic: self-collision analysis (offline, from recorded qpos)
# --------------------------------------------------------------------------------------


def reference_qpos(core, motion_id: int) -> np.ndarray:
    """(T, 7+ndof) MuJoCo-order qpos of a reference motion, for comparison against rollouts."""
    lib = core._motion_lib
    dt = float(core.dt)
    n = int(math.ceil(float(lib._motion_lengths[motion_id]) / dt))
    times = torch.arange(n, device=core.device, dtype=torch.float32) * dt
    ids = torch.full((n,), int(motion_id), device=core.device, dtype=torch.long)
    res = lib.get_motion_state(ids, times)
    root_pos = res["root_pos"].cpu().numpy()
    quat_xyzw = res["root_rot"].cpu().numpy()
    quat_wxyz = np.roll(quat_xyzw, 1, axis=-1)
    return np.concatenate([root_pos, quat_wxyz, res["dof_pos"].cpu().numpy()], axis=-1)


class SelfCollisionChecker:
    """Replays recorded qpos through a CPU MuJoCo model and counts robot-robot contacts.

    The MJLab contact sensor reduces to a per-body net force, which cannot separate
    ground contact from self contact, so this runs offline on the recorded snapshots.
    It sees the policy-rate frames only (not physics substeps), so it under-counts
    brief self contacts.
    """

    def __init__(self, xml_path: Path, expected_nq: int):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if int(self.model.nq) != int(expected_nq):
            raise ValueError(f"Self-collision model nq={self.model.nq}, expected {expected_nq}")
        self.data = mujoco.MjData(self.model)
        self.world_geoms = {
            g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) == 0
        }
        self._body_name = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or f"geom{g}"
            for g in range(self.model.ngeom)
        ]
        self.pair_counts: dict[tuple[str, str], int] = {}

    def analyze(self, qpos_traj: np.ndarray, stride: int = 2) -> dict[str, np.ndarray]:
        """qpos_traj is (T, N, nq).

        Returns per-env ``self_collision_frac`` (fraction of sampled frames with a
        robot-robot contact) and ``ground_penetration_m`` (deepest robot-into-floor
        interpenetration, i.e. ``-min(contact.dist)`` over ground contacts).
        """
        T, N, _ = qpos_traj.shape
        collisions = np.zeros(N, dtype=float)
        penetration = np.zeros(N, dtype=float)
        n_frames = 0
        for t in range(0, T, max(1, stride)):
            n_frames += 1
            for i in range(N):
                self.data.qpos[:] = qpos_traj[t, i].astype(np.float64)
                self.data.qvel[:] = 0.0
                self.mujoco.mj_kinematics(self.model, self.data)
                self.mujoco.mj_collision(self.model, self.data)
                hit = False
                for c in range(int(self.data.ncon)):
                    g1 = int(self.data.contact.geom1[c])
                    g2 = int(self.data.contact.geom2[c])
                    is_world = (g1 in self.world_geoms) or (g2 in self.world_geoms)
                    if is_world:
                        penetration[i] = max(penetration[i], -float(self.data.contact.dist[c]))
                    else:
                        pair = tuple(sorted((self._body_name[g1], self._body_name[g2])))
                        self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
                        if not hit:
                            collisions[i] += 1.0
                            hit = True
        return {
            "self_collision_frac": collisions / max(1, n_frames),
            "ground_penetration_m": penetration,
        }


# --------------------------------------------------------------------------------------
# z sources
# --------------------------------------------------------------------------------------


class MotionZSource:
    """Per-frame z from the *current* model's backward map, plus get-up segment detection.

    Computing z here (rather than reading a stale ``zs_*.pkl``) keeps z consistent with the
    checkpoint under evaluation -- ``backward_map`` is weight-dependent, so a pkl produced
    at an earlier checkpoint encodes a different latent space.
    """

    def __init__(self, core, model, *, use_root_height_obs: bool, device: str):
        self.core = core
        self.model = model
        self.use_root_height_obs = use_root_height_obs
        self.device = device
        self._z_cache: dict[int, torch.Tensor] = {}
        self._h_cache: dict[int, np.ndarray] = {}

    @torch.no_grad()
    def z_sequence(self, motion_id: int) -> torch.Tensor:
        """(T, z_dim). ``z[k]`` corresponds to motion time ``(k+1) * dt`` (see _tracking_z)."""
        if motion_id not in self._z_cache:
            backward_obs, _obs_dict = get_backward_observation(
                self.core, motion_id, use_root_height_obs=self.use_root_height_obs
            )
            sliced = {k: (v[1:].to(self.device) if hasattr(v, "to") else v) for k, v in backward_obs.items()}
            z = self.model.backward_map(sliced)
            self._z_cache[motion_id] = self.model.project_z(z).float()
        return self._z_cache[motion_id]

    def _reference_traces(self, motion_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(root_height, root_speed_xy) sampled at env dt, index-aligned with ``z_sequence``."""
        if motion_id not in self._h_cache:
            lib = self.core._motion_lib
            dt = float(self.core.dt)
            n = int(math.ceil(float(lib._motion_lengths[motion_id]) / dt))
            times = torch.arange(n, device=self.device, dtype=torch.float32) * dt
            ids = torch.full((n,), int(motion_id), device=self.device, dtype=torch.long)
            res = lib.get_motion_state(ids, times)
            height = res["root_pos"][:, 2].cpu().numpy()[1:]
            speed = torch.norm(res["root_vel"][:, :2], dim=-1).cpu().numpy()[1:]
            self._h_cache[motion_id] = (height, speed)
        return self._h_cache[motion_id]

    def root_height(self, motion_id: int) -> np.ndarray:
        return self._reference_traces(motion_id)[0]

    def root_speed(self, motion_id: int) -> np.ndarray:
        return self._reference_traces(motion_id)[1]

    def getup_frames(self, motion_ids: Sequence[int], **kw) -> torch.Tensor:
        """All z frames belonging to any detected get-up segment, pooled across motions."""
        chunks = []
        for mid in motion_ids:
            seq = self.z_sequence(int(mid))
            for lo, hi in self.getup_segments(int(mid), **kw):
                chunks.append(seq[lo:hi])
        if not chunks:
            raise ValueError(f"No get-up segments found in motions {list(motion_ids)}")
        return torch.cat(chunks, dim=0)

    def standing_frames(
        self, motion_ids: Sequence[int], *, min_height: float = 0.50, max_speed: float = 0.15
    ) -> torch.Tensor:
        """z frames where the reference is upright and nearly stationary -- a "stand still" latent."""
        chunks = []
        for mid in motion_ids:
            seq = self.z_sequence(int(mid))
            h, v = self._reference_traces(int(mid))
            n = min(seq.shape[0], h.shape[0])
            mask = (h[:n] > min_height) & (v[:n] < max_speed)
            if mask.any():
                chunks.append(seq[:n][torch.as_tensor(mask, device=seq.device)])
        if not chunks:
            raise ValueError(f"No standing frames found in motions {list(motion_ids)}")
        return torch.cat(chunks, dim=0)

    def getup_segments(
        self, motion_id: int, *, low: float = 0.25, high: float = 0.45, min_s: float = 0.3, max_s: float = 8.0
    ) -> list[tuple[int, int]]:
        """Index ranges (into ``z_sequence``) where the reference root height rises from
        fallen to standing. Returns every such rise.

        ``end`` is the first up-crossing of ``high``; ``start`` is the *lowest* frame in the
        ``max_s`` window before it, i.e. the bottom of the fall -- not merely the last frame
        under ``low``, which would land mid-rise and clip off the hardest part of the motion.
        """
        h = self.root_height(motion_id)
        dt = float(self.core.dt)
        max_back = max(1, int(round(max_s / dt)))
        segments: list[tuple[int, int]] = []
        armed = False
        for t in range(h.shape[0]):
            if h[t] < low:
                armed = True
            elif h[t] > high and armed:
                armed = False
                window_lo = max(0, t - max_back)
                start = window_lo + int(np.argmin(h[window_lo:t])) if t > window_lo else t
                dur = (t - start) * dt
                if h[start] < low and min_s <= dur <= max_s:
                    segments.append((start, t))
        return segments


def load_z_from_file(path: Path, *, key: str | None, index: int | None, device: str) -> torch.Tensor:
    """Accepts .pt/.pth (torch), .npy/.npz (numpy) or .pkl (joblib). Returns (z_dim,) or (T, z_dim)."""
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
    elif suffix == ".npy":
        obj = np.load(path)
    elif suffix == ".npz":
        obj = dict(np.load(path))
    else:
        obj = joblib.load(path)

    if isinstance(obj, dict):
        if key is None:
            raise ValueError(f"{path} holds a dict with keys {sorted(map(str, obj))}; pass 'key'")
        obj = obj[key]
        if isinstance(obj, dict) and "z" in obj:  # z-bank entry
            obj = obj["z"]
    if isinstance(obj, (list, tuple)):
        obj = obj[0] if index is None else obj[index]
    z = torch.as_tensor(np.asarray(obj), dtype=torch.float32, device=device)
    if z.dim() == 2 and index is not None:
        z = z[index]
    return z


def build_z_provider(spec: dict[str, Any], *, model, motion_src: MotionZSource, device: str) -> ZProvider:
    """Materialize one candidate spec into a ZProvider. See ``default_candidates`` for shapes."""
    name = spec.get("name") or spec.get("type", "z")
    ctype = spec["type"]

    if ctype == "two_phase":
        a = build_z_provider({**spec["phase_a"], "name": name + ":a"}, model=model, motion_src=motion_src, device=device)
        b = build_z_provider({**spec["phase_b"], "name": name + ":b"}, model=model, motion_src=motion_src, device=device)
        dt = float(motion_src.core.dt)
        return TwoPhaseZ(a, b, int(round(float(spec["switch_s"]) / dt)), name)

    source = spec.get("source", {})
    kind = source.get("kind", "file")

    if kind in ("motion_segment_mean", "motion_segment", "motion_full"):
        motion_id = int(source["motion_id"])
        seq = motion_src.z_sequence(motion_id)
        if kind == "motion_full":
            lo, hi = 0, seq.shape[0]
        else:
            segments = motion_src.getup_segments(
                motion_id,
                low=float(source.get("low", 0.25)),
                high=float(source.get("high", 0.45)),
            )
            if not segments:
                raise ValueError(f"No get-up segment found in motion {motion_id}")
            seg_idx = int(source.get("segment", 0))
            lo, hi = segments[seg_idx % len(segments)]
            pad = int(round(float(source.get("pad_end_s", 0.0)) / float(motion_src.core.dt)))
            hi = min(seq.shape[0], hi + pad)
        window = seq[lo:hi]
        if kind == "motion_segment_mean":
            return ConstantZ(model.project_z(window.mean(dim=0, keepdim=True))[0], name)
        return SequenceZ(window, name)

    if kind in ("motion_getup_mean", "motion_standing_mean"):
        ids = source.get("motion_ids") or [int(source["motion_id"])]
        if kind == "motion_getup_mean":
            frames = motion_src.getup_frames(ids)
        else:
            frames = motion_src.standing_frames(
                ids, min_height=float(source.get("min_height", 0.50)), max_speed=float(source.get("max_speed", 0.15))
            )
        return ConstantZ(model.project_z(frames.mean(dim=0, keepdim=True))[0], name)

    if kind == "file":
        z = load_z_from_file(Path(source["path"]), key=source.get("key"), index=source.get("index"), device=device)
        if z.dim() == 1:
            return ConstantZ(model.project_z(z.unsqueeze(0))[0], name)
        if spec["type"] == "sequence":
            return SequenceZ(model.project_z(z), name)
        return ConstantZ(model.project_z(z.mean(dim=0, keepdim=True))[0], name)

    if kind == "random":
        gen = torch.Generator(device="cpu").manual_seed(int(source.get("seed", 0)))
        raw = torch.randn((1, int(model.cfg.archi.z_dim)), generator=gen).to(device)
        return ConstantZ(model.project_z(raw)[0], name)

    if kind == "basis":  # structured baseline: +-sqrt(d) on one axis
        z = torch.zeros((1, int(model.cfg.archi.z_dim)), device=device)
        z[0, int(source["axis"]) % z.shape[1]] = float(source.get("sign", 1.0))
        return ConstantZ(model.project_z(z)[0], name)

    if kind == "vector":
        z = torch.as_tensor(np.asarray(source["values"]), dtype=torch.float32, device=device)
        return ConstantZ(model.project_z(z.reshape(1, -1))[0], name)

    raise ValueError(f"Unknown z source kind: {kind}")


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


_SUMMARY_FIELDS = (
    "time_to_stand_s",
    "hold_frac",
    "hold_root_height_std",
    "hold_max_tilt_deg",
    "peak_torque_nm",
    "peak_torque_ratio",
    "torque_saturation_frac",
    "mean_abs_power_w",
    "undesired_contact_frac",
    "self_collision_frac",
    "ground_penetration_m",
    "final_root_height",
    "max_root_height",
    "held_upright_frac",
    "time_to_rise_s",
    "handoff_max_abs_delta",
    "handoff_rms_delta",
    "handoff_knee_delta",
)


def aggregate(rows: Sequence[dict[str, Any]], episode_s: float) -> dict[str, Any]:
    out: dict[str, Any] = {"n_episodes": len(rows)}
    out["success_rate"] = float(np.mean([r["success"] for r in rows])) if rows else 0.0
    out["reached_stand_rate"] = float(np.mean([r["reached_stand"] for r in rows])) if rows else 0.0
    out["refell_rate"] = float(np.mean([r["refell"] for r in rows])) if rows else 0.0
    out["fell_back_rate"] = (
        float(np.mean([r["fell_back"] for r in rows])) if rows and "fell_back" in rows[0] else float("nan")
    )
    for key, flag in (("success_upright_rate", "success_upright"), ("rose_rate", "rose")):
        out[key] = float(np.mean([r[flag] for r in rows])) if rows and flag in rows[0] else float("nan")
    ok = [r for r in rows if r["success"]]
    for f in _SUMMARY_FIELDS:
        vals = [r[f] for r in (ok if f in ("time_to_stand_s", "hold_max_tilt_deg", "hold_root_height_std") else rows) if f in r]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        out[f"{f}_mean"] = float(np.mean(vals)) if vals else float("nan")
        if f in ("time_to_stand_s", "peak_torque_ratio"):
            out[f"{f}_p90"] = float(np.percentile(vals, 90)) if vals else float("nan")
    out["score"] = getup_score(rows, episode_s)
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_episode_csv(
    traj: Trajectory,
    env_index: int,
    metrics: dict[str, Any],
    out_dir: Path,
    *,
    control_joint_names: Sequence[str],
    stem: str,
) -> Path:
    """Write one successful episode as a RobotState CSV matching ``configs/data/k1_lafan1.yaml``.

    Emitted at the native policy rate (1/dt = 50 Hz), *not* resampled to LAFAN1's 30 fps:
    these are the exact simulated states, so resampling would only add interpolation error.
    A dataset manifest consuming these clips must therefore declare ``fps: 50``.

    Recorded qpos is ``[root_pos(3), root_quat_wxyz(4), dof(22)]`` in control-joint order;
    the CSV wants quaternion xyzw, and root xy relative to the clip's own origin, so the
    env-grid offset is removed.
    """
    import csv as _csv

    qpos = traj.qpos[:, env_index, :].astype(np.float64)
    origin = traj.env_origins[env_index]
    root_pos = qpos[:, 0:3] - origin[None, :]
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    root_quat_xyzw = np.stack([x, y, z, w], axis=-1)
    dof = qpos[:, 7:]
    if dof.shape[1] != len(control_joint_names):
        raise ValueError(f"Expected {len(control_joint_names)} dofs in qpos, got {dof.shape[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    header = (
        ["root_pos_x", "root_pos_y", "root_pos_z", "root_rot_x", "root_rot_y", "root_rot_z", "root_rot_w"]
        + [f"dof_pos_{i}" for i in range(dof.shape[1])]
    )
    with csv_path.open("w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        for t in range(qpos.shape[0]):
            writer.writerow([f"{v:.6f}" for v in (*root_pos[t], *root_quat_xyzw[t], *dof[t])])

    sidecar = {
        "fps": round(1.0 / traj.dt, 6),
        "n_frames": int(qpos.shape[0]),
        "duration_s": float(qpos.shape[0] * traj.dt),
        "control_joint_names": list(control_joint_names),
        "root_quat_order": "xyzw",
        "coordinate_system": "z_up",
        "dof_unit": "rad",
        "source": "humanoidverse.tools.eval_getup policy rollout (physically executed)",
        "metrics": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2, default=str) + "\n")
    return csv_path


def render_episode(qpos_traj: np.ndarray, env_index: int, robot_xml: Path, out_path: Path, *, fps: int, render_size: int) -> None:
    import mediapy as media

    renderer = MujocoQposRenderer(robot_xml, render_size=render_size, expected_qpos_size=qpos_traj.shape[-1])
    try:
        frames = [renderer.render_qpos(qpos_traj[t, env_index]) for t in range(qpos_traj.shape[0])]
        media.write_video(str(out_path), frames, fps=fps)
    finally:
        renderer.close()


# --------------------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------------------


def default_candidates() -> list[dict[str, Any]]:
    """Cost-ordered candidate set: existing tracking latents first, baselines last."""
    cands: list[dict[str, Any]] = []
    for motion_id, label in ((16, "fallAndGetUp2_s2"), (17, "fallAndGetUp2_s3"), (55, "pushAndFall1_s4"), (27, "ground2_s2")):
        cands.append(
            {
                "name": f"m{motion_id}_{label}_segmean",
                "type": "constant",
                "source": {"kind": "motion_segment_mean", "motion_id": motion_id, "segment": 0},
            }
        )
        cands.append(
            {
                "name": f"m{motion_id}_{label}_replay",
                "type": "sequence",
                "source": {"kind": "motion_segment", "motion_id": motion_id, "segment": 0},
            }
        )
    for seed in range(4):
        cands.append({"name": f"random_{seed}", "type": "constant", "source": {"kind": "random", "seed": seed}})
    return cands


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Get-up (skill) evaluation harness for UFO BFM policies.")
    p.add_argument("--model-folder", type=Path, default=None)
    p.add_argument("--robot-config", type=Path, default=None)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--data-manifest", type=Path, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--episodes", type=int, default=None, help="Total episodes per candidate (default: --num-envs).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tag", default=None, help="Label for this condition; defaults to <bucket>_<nominal|dr>.")

    p.add_argument("--candidates", type=Path, default=None, help="JSON list of candidate specs.")
    p.add_argument("--emit-default-candidates", type=Path, default=None, help="Write the default candidate JSON and exit.")

    p.add_argument("--bucket", choices=("indist", "ood", "motion_frame"), default="indist")
    p.add_argument("--aligned-motion-id", type=int, default=16, help="motion_frame bucket: reference motion.")
    p.add_argument("--aligned-segment", type=int, default=0, help="motion_frame bucket: reset at this get-up segment's start.")
    p.add_argument("--aligned-jitter-s", type=float, default=0.0)
    p.add_argument("--motion-pool", type=int, nargs="*", default=None, help="in-dist reference motions (default: all).")
    p.add_argument("--ood-tilts", nargs="*", default=["supine", "prone", "side_left", "side_right"])
    p.add_argument("--ood-yaws", type=int, default=8)
    p.add_argument("--ood-drop-height", type=float, default=0.30)
    p.add_argument("--ood-joint-noise", type=float, default=0.0)

    add_bool_arg(p, "--disable-dr", True, "Disable domain randomization (friction/mass/com/push).")
    add_bool_arg(p, "--disable-obs-noise", True, "Disable observation noise.")
    p.add_argument("--hydra-override", nargs="*", default=None,
                   help="Extra hydra overrides, e.g. domain_rand.push_robots=False, to ablate single DR terms.")
    p.add_argument("--action-latency-max", type=int, default=0)
    p.add_argument("--obs-latency-max", type=int, default=0)

    p.add_argument("--settle-steps", type=int, default=50)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--stand-height", type=float, default=0.45)
    p.add_argument("--stand-upright", type=float, default=0.90)
    p.add_argument("--stand-foot-force", type=float, default=5.0)
    p.add_argument("--stand-confirm-s", type=float, default=0.5)
    p.add_argument("--stand-hold-s", type=float, default=2.0)

    add_bool_arg(p, "--self-collision", True, "Run the offline self-collision pass on recorded qpos.")
    p.add_argument("--self-collision-stride", type=int, default=5)
    p.add_argument("--render-best-worst", type=int, default=0, help="Render N best and N worst episodes per candidate.")
    p.add_argument("--render-size", type=int, default=480)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--export-successful-motions", type=Path, default=None,
                   help="Write each successful episode as a RobotState CSV (+JSON sidecar) into this dir.")
    add_bool_arg(p, "--reference-penetration", False, "Report ground penetration of the reference motions.")
    p.add_argument("--reference-penetration-motions", type=int, nargs="*", default=None)
    p.add_argument("--z-bank-out", type=Path, default=None, help="Also write the winning z to this z-bank .pt file.")
    p.add_argument("--z-bank-skill", default="getup")
    args = p.parse_args()
    # Only --emit-default-candidates can run without a model/output; everything else needs both.
    if args.emit_default_candidates is None:
        missing = [n for n, v in (("--model-folder", args.model_folder), ("--out-dir", args.out_dir)) if v is None]
        if missing:
            p.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def _resolve_data_path(args: argparse.Namespace) -> tuple[Path, Path]:
    manifest_robot_config = None
    data_path = args.data_path
    if args.data_manifest is not None:
        if data_path is not None:
            raise SystemExit("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            raise SystemExit("--dataset is required with --data-manifest")
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
        data_path = Path(prepare_manifest_dataset_path(args.data_manifest, args.dataset, split="inference"))
    if data_path is None:
        data_path = PROJECT_ROOT / DEFAULT_DATA_PATH
    robot_config = resolve_inference_robot_config(args.robot_config or (PROJECT_ROOT / DEFAULT_ROBOT_CONFIG), manifest_robot_config)
    return Path(data_path).expanduser().resolve(), robot_config


def main() -> None:
    args = parse_args()
    if args.emit_default_candidates is not None:
        args.emit_default_candidates.parent.mkdir(parents=True, exist_ok=True)
        args.emit_default_candidates.write_text(json.dumps(default_candidates(), indent=2) + "\n")
        print(f"[INFO] Wrote default candidates: {args.emit_default_candidates}")
        return

    data_path, robot_config = _resolve_data_path(args)
    model_folder = args.model_folder.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.bucket}_{'nominal' if args.disable_dr else 'dr'}"

    robot_training = load_robot_training_spec(robot_config)
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    control_joints = list(robot_training.robot.control_joint_names)
    qpos_sorted = sorted(control_joints, key=lambda j: robot_training.robot.joint_qpos_addr[j])
    if qpos_sorted != control_joints:
        raise RuntimeError(
            "Recorded qpos is in control-joint order but rendering/self-collision assume MuJoCo "
            f"qpos-address order; they differ for {robot_training.robot.name}. Reorder before use."
        )

    device = args.device
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=checkpoint_load_device(device))
    model.to(device)
    model.eval()

    wrapped_env, core, env_cfg, use_root_height_obs = build_eval_env(
        model_folder=model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,
        num_envs=args.num_envs,
        disable_dr=bool(args.disable_dr),
        disable_obs_noise=bool(args.disable_obs_noise),
        seed=args.seed,
        hydra_overrides=args.hydra_override,
    )

    print(f"[INFO] tag={tag} bucket={args.bucket} num_envs={core.num_envs} dt={core.dt}")
    print(f"[INFO] disable_dr={args.disable_dr} disable_obs_noise={args.disable_obs_noise} "
          f"action_latency_max={args.action_latency_max} obs_latency_max={args.obs_latency_max}")
    print(f"[INFO] motions={core._motion_lib._num_unique_motions} data={data_path}")

    try:
        motion_src = MotionZSource(core, model, use_root_height_obs=use_root_height_obs, device=device)
        specs = json.loads(args.candidates.read_text()) if args.candidates else default_candidates()

        pose_cfg = PoseBankConfig(
            bucket=args.bucket,
            motion_pool=args.motion_pool,
            ood_tilts=tuple(args.ood_tilts),
            ood_yaws=args.ood_yaws,
            ood_drop_height=args.ood_drop_height,
            ood_joint_noise=args.ood_joint_noise,
            motion_frame_id=args.aligned_motion_id,
            motion_frame_jitter_s=args.aligned_jitter_s,
        )
        if args.bucket == "motion_frame":
            segs = motion_src.getup_segments(args.aligned_motion_id)
            if not segs:
                raise SystemExit(f"No get-up segments in motion {args.aligned_motion_id}")
            lo, _hi = segs[args.aligned_segment % len(segs)]
            # z index k corresponds to motion time (k+1)*dt (see MotionZSource.z_sequence).
            pose_cfg.motion_frame_time_s = float((lo + 1) * core.dt)
            print(f"[INFO] aligned control: motion {args.aligned_motion_id} seg {args.aligned_segment} "
                  f"-> reset at t={pose_cfg.motion_frame_time_s:.2f}s")
        bank = FallenPoseBank(core, pose_cfg)
        roll_cfg = RolloutConfig(
            settle_steps=args.settle_steps,
            episode_steps=args.episode_steps,
            action_latency_max=args.action_latency_max,
            obs_latency_max=args.obs_latency_max,
            record_qpos=bool(args.self_collision) or args.render_best_worst > 0 or args.export_successful_motions is not None,
        )
        crit = StandCriterion(
            root_height=args.stand_height,
            upright=args.stand_upright,
            foot_force=args.stand_foot_force,
            confirm_s=args.stand_confirm_s,
            hold_s=args.stand_hold_s,
        )
        effort_limits = core.torque_limits.cpu().numpy()
        episodes = args.episodes or core.num_envs
        n_batches = int(math.ceil(episodes / core.num_envs))
        episode_s = roll_cfg.episode_steps * float(core.dt)

        checker = SelfCollisionChecker(robot_xml, expected_nq=7 + core.num_dof) if args.self_collision else None
        n_exported = [0]
        self_collision_pairs: dict[str, list[dict[str, Any]]] = {}
        if args.reference_penetration and checker is not None:
            # Baseline for the exported-clip quality claim: how far the *retargeted LAFAN1*
            # reference frames push through the floor, measured the same way.
            for mid in (args.reference_penetration_motions or [16, 17, 55, 27]):
                ref_qpos = reference_qpos(core, int(mid))
                stats = checker.analyze(ref_qpos[:, None, :], stride=max(1, args.self_collision_stride))
                print(f"[REFPEN] motion {mid}: max ground penetration = "
                      f"{float(stats['ground_penetration_m'][0]) * 100:.2f} cm")
        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        z_store: dict[str, dict[str, Any]] = {}

        for spec in specs:
            name = spec.get("name", spec.get("type", "z"))
            try:
                provider = build_z_provider(spec, model=model, motion_src=motion_src, device=device)
            except Exception as exc:  # a bad candidate must not kill the sweep
                print(f"[WARN] Skipping candidate {name}: {exc}")
                continue

            rows: list[dict[str, Any]] = []
            for b in range(n_batches):
                # One generator per (candidate, batch) seeded from the run seed and the batch
                # index only -- every candidate therefore sees the *same* pose bank.
                rng = np.random.default_rng([args.seed, b])
                traj = rollout(
                    wrapped_env, core, model,
                    z_provider=provider, pose_bank=bank, cfg=roll_cfg, rng=rng, device=device,
                )
                batch_rows = score_getup(traj, crit, effort_limits)
                add_handoff_metrics(traj, batch_rows, core.dof_names, hold_s=crit.hold_s)
                if checker is not None and traj.qpos is not None:
                    checker.pair_counts = {}
                    contact_stats = checker.analyze(traj.qpos, stride=args.self_collision_stride)
                    if checker.pair_counts:
                        top_pairs = sorted(checker.pair_counts.items(), key=lambda kv: -kv[1])[:4]
                        self_collision_pairs[name] = [
                            {"pair": list(p), "count": int(k)} for p, k in top_pairs
                        ]
                    for i, r in enumerate(batch_rows):
                        for key, values in contact_stats.items():
                            r[key] = float(values[i])
                for i, r in enumerate(batch_rows):
                    r.update({"candidate": name, "z_mode": provider.mode, "tag": tag, "batch": b, "env": i})
                    r.update(traj.meta[i])
                rows.extend(batch_rows)

                if args.export_successful_motions is not None and traj.qpos is not None:
                    export_dir = args.export_successful_motions.expanduser().resolve()
                    for i, r in enumerate(batch_rows):
                        if not r["success"]:
                            continue
                        stem = f"getup_{tag}_{name}_b{b}_e{i}_score{int(round(100 * r['hold_frac']))}"
                        export_episode_csv(
                            traj, i, r, export_dir, control_joint_names=control_joints, stem=stem
                        )
                        n_exported[0] += 1

                if args.render_best_worst > 0 and b == 0 and traj.qpos is not None:
                    order = sorted(range(len(batch_rows)), key=lambda i: (not batch_rows[i]["success"], batch_rows[i]["time_to_stand_s"] if batch_rows[i]["success"] else 1e9))
                    vids = out_dir / "videos"
                    vids.mkdir(exist_ok=True)
                    for label, picks in (("best", order[: args.render_best_worst]), ("worst", order[-args.render_best_worst :])):
                        for i in picks:
                            render_episode(traj.qpos, i, robot_xml, vids / f"{tag}_{name}_{label}_env{i}.mp4",
                                           fps=args.fps, render_size=args.render_size)

            summary = aggregate(rows, episode_s)
            summary.update({"candidate": name, "z_mode": provider.mode, "tag": tag, "spec": spec})
            summaries.append(summary)
            all_rows.extend(rows)
            z_store[name] = {**provider.bank_payload(float(core.dt)), "mode": provider.mode, "spec": spec}
            print(f"[RESULT] {tag:22s} {name:34s} success={summary['success_rate']:.3f} "
                  f"reached={summary['reached_stand_rate']:.3f} tts={summary['time_to_stand_s_mean']:.2f}s "
                  f"tilt={summary['hold_max_tilt_deg_mean']:.1f}deg score={summary['score']:.1f}", flush=True)

        write_csv(out_dir / f"episodes_{tag}.csv", all_rows)
        summaries.sort(key=lambda s: (-s["success_rate"], -s["score"]))
        payload = {
            "tag": tag,
            "model_folder": str(model_folder),
            "data_path": str(data_path),
            "robot_config": str(robot_config),
            "bucket": args.bucket,
            "num_envs": core.num_envs,
            "episodes_per_candidate": episodes,
            "seed": args.seed,
            "dt": float(core.dt),
            "episode_s": episode_s,
            "settle_s": roll_cfg.settle_steps * float(core.dt),
            "disable_dr": bool(args.disable_dr),
            "disable_obs_noise": bool(args.disable_obs_noise),
            "action_latency_max": args.action_latency_max,
            "obs_latency_max": args.obs_latency_max,
            "stand_criterion": crit.__dict__,
            "pose_bank": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pose_cfg.__dict__.items()},
            "hydra_override": list(args.hydra_override or []),
            "self_collision_pairs": self_collision_pairs,
            "results": summaries,
        }
        (out_dir / f"summary_{tag}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        torch.save(z_store, out_dir / f"z_candidates_{tag}.pt")
        print(f"[INFO] Wrote {out_dir / f'episodes_{tag}.csv'} and {out_dir / f'summary_{tag}.json'}")
        if args.export_successful_motions is not None:
            print(f"[INFO] Exported {n_exported[0]} successful episodes as RobotState CSV "
                  f"({1.0 / float(core.dt):.0f} fps) -> {args.export_successful_motions}")

        if args.z_bank_out is not None and summaries:
            best = summaries[0]
            entry = z_store[best["candidate"]]
            bank_path = args.z_bank_out.expanduser().resolve()
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            existing = torch.load(bank_path, map_location="cpu", weights_only=False) if bank_path.exists() else {}
            existing[args.z_bank_skill] = {
                **{k: v for k, v in entry.items() if k != "spec"},
                "source": {"candidate": best["candidate"], "spec": best["spec"], "tag": tag},
                "score": {k: v for k, v in best.items() if k != "spec"},
            }
            torch.save(existing, bank_path)
            print(f"[INFO] Wrote z-bank entry '{args.z_bank_skill}' -> {bank_path}")
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    main()
