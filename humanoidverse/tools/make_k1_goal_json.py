"""Generate a K1 22-DoF goal JSON for humanoidverse.goal_inference.

`goal_inference.py` hard-requires a robot-specific goal JSON for any robot
that is not the 29-DoF G1 (see `_find_goal_json` in that module). This tool
builds that file for K1 by mining the K1 LAFAN1 motion cache
(`cache/motion_data/k1_lafan1/k1_lafan1_full_ufo.pkl`, produced by the manifest
pipeline from `configs/data/k1_lafan1.yaml`) for frames that are *statistically*
upright/standing, rather than hand-typing joint angles.

Schema (matches `humanoidverse/data/robots/g1/goal_frames_lafan29dof.json`):
a JSON list of `{"motion_id": int, "motion_name": str, "frames": [int, ...]}`
entries, plus extra diagnostic fields ignored by `goal_inference.py`'s
validator (`_validate_goal_entry_dims` only inspects keys in `_GOAL_DOF_KEYS`
/ `_GOAL_QPOS_KEYS`, both absent here since `goal_inference.py` only ever
reads `motion_id` / `motion_name` / `frames` off each entry -- the DOF/qpos
comes from the motion library itself via `motion_id`, not from the JSON).

`motion_id` must match the index MotionLibRobot assigns to each motion when
it loads the cache pkl. With `im_eval=False, min_length=-1` (the default used
by `env._motion_lib.load_data`), the motion lib preserves dict insertion
order, so `motion_id == list(pkl.keys()).index(motion_name)`. This script
loads the exact same pkl (via `prepare_manifest_dataset_path`, `split=
"inference"`, the same call `goal_inference.py` makes for `--data-manifest`)
so the indices line up.

IMPORTANT -- `frames` are control-step indices, not raw mocap frame numbers.
`run_goal_inference` builds its per-motion observation buffer (`gobs`) via
`get_backward_observation`, which resamples the motion at `env.dt` (the
MJLab control step, 0.02s = 50 Hz for the shared simulator config in
`humanoidverse/config/simulator/mujoco.yaml`: sim.fps=200, control_decimation
=4), not at the motion's native capture fps (30 Hz for this LAFAN1 dataset).
`goal["frames"]` values index directly into that `gobs` array
(`value[frame_idx]` in `run_goal_inference`). This script analyzes upright
frames in raw 30 fps motion-frame units (matching the source CSV/pkl), then
converts each chosen frame to a control-step index via
`round(raw_frame / motion_fps / env_dt)` before writing it out. The original
30 fps frame number is preserved per-entry as `diagnostics.source_frame_30fps`
for traceability.

Method
------
For every frame of every motion:
  - torso tilt = angle between the retargeted root quaternion's local +z axis
    and world +z (0 deg == perfectly upright).
  - height deviation = |root height - config `init_state.pos.z`|.
  - leg deviation = L2 norm, over the 12 leg DOFs only, between that frame's
    dof_pos and the config's `default_joint_angles` (itself the median
    upright-frame pose of this same dataset -- used here purely as a
    cross-check reference, not copied into the output).
  - whole-body joint speed = L2 norm of the central-difference dof_pos
    velocity, over all 22 DOFs (catches frames where the legs are static but
    the arms are still swinging mid-motion).

A frame is "quiet-standing" if tilt < 5 deg, height deviation < 0.03 m, and
whole-body speed < 0.5 rad/s. Runs of >=15 consecutive quiet-standing frames
(0.5 s at 30 fps) are kept (drops single-frame coincidences); each run
contributes its center frame as a candidate, tagged with its diagnostics.

Two goal groups are emitted:
  - "standing": the 5 candidates with the lowest leg deviation, one per
    distinct motion (diversity across walk/run/obstacle/etc. motions).
  - "settled_two_feet_stance": the lowest-leg-deviation candidate whose quiet
    run is preceded, earlier in the same fallAndGetUp*/pushAndFall* motion,
    by a frame with height < 0.35 m (i.e. an actual post-recovery standing
    moment, not a pre-fall idle stance) -- directly relevant to a get-up
    policy's target state.

The very first entry in the output list (lowest leg deviation overall) is
the canonical "standing" goal frame used downstream.

Usage
-----
    uv run python -m humanoidverse.tools.make_k1_goal_json

Cross-check output against `load_and_validate_goal_json` (num_dof=22) is
printed at the end; the tool exits non-zero if validation fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R

from humanoidverse.utils.motion_data import prepare_manifest_dataset_path
from humanoidverse.utils.robot_spec import load_robot_training_spec

DEFAULT_ROBOT_CONFIG = "configs/robots/k1_22dof.yaml"
DEFAULT_DATA_MANIFEST = "configs/data/k1_lafan1.yaml"
DEFAULT_DATASET = "k1_lafan1"
DEFAULT_OUTPUT = "humanoidverse/data/robots/k1/goal_frames_k1_22dof.json"

# Leg DOFs come after 10 arm/head DOFs in configs/robots/k1_22dof.yaml's
# control_joints.names (Left_Hip_Pitch .. Right_Ankle_Roll).
LEG_DOF_SLICE = slice(10, 22)

FALL_MOTION_KEYWORDS = ("fallandgetup", "pushandfall")


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, m in enumerate(mask.tolist()):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def analyze_motions(
    motion_data: dict[str, Any],
    *,
    default_vec: np.ndarray,
    default_z: float,
    tilt_deg_max: float,
    height_tol: float,
    speed_max: float,
    min_run_frames: int,
    sim_dt: float,
) -> list[dict[str, Any]]:
    keys = list(motion_data.keys())
    candidates: list[dict[str, Any]] = []
    for motion_id, name in enumerate(keys):
        entry = motion_data[name]
        dof = np.asarray(entry["dof_pos"], dtype=np.float64)
        fps = float(entry["fps"])
        height = np.asarray(entry["root_trans_offset"], dtype=np.float64)[:, 2]
        quat_xyzw = np.asarray(entry["root_quat"], dtype=np.float64)

        up = R.from_quat(quat_xyzw).apply([0.0, 0.0, 1.0])
        tilt_deg = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))
        height_dev = np.abs(height - default_z)
        leg_dev = np.linalg.norm(dof[:, LEG_DOF_SLICE] - default_vec[None, LEG_DOF_SLICE], axis=1)
        full_dev = np.linalg.norm(dof - default_vec[None, :], axis=1)

        dof_vel = np.zeros_like(dof)
        dof_vel[1:-1] = (dof[2:] - dof[:-2]) * (fps / 2.0)
        full_speed = np.linalg.norm(dof_vel, axis=1)

        fallen_mask = height < 0.35
        quiet_mask = (tilt_deg < tilt_deg_max) & (height_dev < height_tol) & (full_speed < speed_max)

        for run_start, run_end in find_runs(quiet_mask):
            if (run_end - run_start) < min_run_frames:
                continue
            frame = (run_start + run_end) // 2
            sim_frame = int(round((frame / fps) / sim_dt))
            candidates.append(
                {
                    "motion_id": motion_id,
                    "motion_name": name,
                    "frame": int(frame),
                    "sim_frame": sim_frame,
                    "run_start": int(run_start),
                    "run_end": int(run_end),
                    "run_len": int(run_end - run_start),
                    "leg_dev": float(leg_dev[frame]),
                    "full_dev": float(full_dev[frame]),
                    "height_m": float(height[frame]),
                    "tilt_deg": float(tilt_deg[frame]),
                    "speed_rad_s": float(full_speed[frame]),
                    "post_recovery": bool(fallen_mask[:run_start].any()) if run_start > 0 else False,
                }
            )
    return candidates


def _diag(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "leg_dev_rad": round(c["leg_dev"], 4),
        "full_dev_rad": round(c["full_dev"], 4),
        "height_m": round(c["height_m"], 4),
        "tilt_deg": round(c["tilt_deg"], 3),
        "speed_rad_s": round(c["speed_rad_s"], 4),
        "quiet_run_frames_30fps": [c["run_start"], c["run_end"]],
        "source_frame_30fps": c["frame"],
    }


def build_goal_entries(
    candidates: list[dict[str, Any]],
    *,
    num_standing: int,
) -> list[dict[str, Any]]:
    # "standing": best (lowest leg deviation) candidate per distinct motion, top N.
    best_per_motion: dict[str, dict[str, Any]] = {}
    for c in candidates:
        name = c["motion_name"]
        if name not in best_per_motion or c["leg_dev"] < best_per_motion[name]["leg_dev"]:
            best_per_motion[name] = c
    standing = sorted(best_per_motion.values(), key=lambda c: c["leg_dev"])[:num_standing]

    # "settled_two_feet_stance": best post-recovery candidate within fall/getup motions.
    fall_candidates = [
        c
        for c in candidates
        if c["post_recovery"] and any(kw in c["motion_name"].lower() for kw in FALL_MOTION_KEYWORDS)
    ]
    fall_candidates.sort(key=lambda c: c["leg_dev"])

    entries: list[dict[str, Any]] = []
    for c in standing:
        entries.append(
            {
                "motion_id": c["motion_id"],
                "motion_name": c["motion_name"],
                "frames": [c["sim_frame"]],
                "goal_type": "standing",
                "diagnostics": _diag(c),
            }
        )
    if fall_candidates:
        c = fall_candidates[0]
        entries.append(
            {
                "motion_id": c["motion_id"],
                "motion_name": c["motion_name"],
                "frames": [c["sim_frame"]],
                "goal_type": "settled_two_feet_stance",
                "diagnostics": _diag(c),
            }
        )
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--data-manifest", type=Path, default=Path(DEFAULT_DATA_MANIFEST))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--tilt-deg-max", type=float, default=5.0)
    parser.add_argument("--height-tol", type=float, default=0.03)
    parser.add_argument("--speed-max", type=float, default=0.5, help="Max whole-body dof_pos speed (rad/s) for a frame to count as 'quiet'.")
    parser.add_argument("--min-run-frames", type=int, default=15, help="Minimum consecutive quiet frames (at the motion's native fps) to keep a run.")
    parser.add_argument("--num-standing", type=int, default=5)
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=0.02,
        help=(
            "MJLab control step (seconds) that goal_inference.py's get_backward_observation() "
            "resamples motions at (sim.fps=200 / control_decimation=4 = 50 Hz in "
            "humanoidverse/config/simulator/mujoco.yaml). Chosen 30fps frames are converted to "
            "this unit before being written to the output JSON."
        ),
    )
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_spec = load_robot_training_spec(args.robot_config)
    control_joint_names = list(training_spec.robot.control_joint_names)
    num_dof = len(control_joint_names)
    default_vec = np.array([training_spec.default_joint_angles[n] for n in control_joint_names], dtype=np.float64)
    default_z = float(training_spec.init_state["pos"][2])

    dataset_path = Path(
        prepare_manifest_dataset_path(
            args.data_manifest,
            args.dataset,
            split="inference",
            rebuild_cache=bool(args.rebuild_motion_cache),
        )
    )
    print(f"[INFO] Motion cache: {dataset_path}")
    motion_data = joblib.load(dataset_path)
    print(f"[INFO] Loaded {len(motion_data)} motions; num_dof={num_dof}")

    first_motion = next(iter(motion_data.values()))
    if list(first_motion["joint_names"]) != control_joint_names:
        raise ValueError(
            "Motion cache joint order does not match robot config control_joints.names; "
            "goal frame dof_pos slices below would silently misalign."
        )

    candidates = analyze_motions(
        motion_data,
        default_vec=default_vec,
        default_z=default_z,
        tilt_deg_max=args.tilt_deg_max,
        height_tol=args.height_tol,
        speed_max=args.speed_max,
        min_run_frames=args.min_run_frames,
        sim_dt=args.sim_dt,
    )
    print(f"[INFO] Found {len(candidates)} quiet-standing runs (>= {args.min_run_frames} frames) across the dataset")

    entries = build_goal_entries(candidates, num_standing=args.num_standing)
    if not entries:
        raise RuntimeError("No goal candidates found; loosen --tilt-deg-max / --height-tol / --speed-max.")

    print(f"[INFO] Goal entries (sim_dt={args.sim_dt}s; first entry is the canonical 'standing' goal):")
    for e in entries:
        diag = e["diagnostics"]
        print(
            f"  [{e['goal_type']:>24s}] motion_id={e['motion_id']:2d} {e['motion_name']:26s} "
            f"sim_frame={e['frames'][0]:5d} (30fps_frame={diag['source_frame_30fps']:5d}) diag={diag}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=2)
    print(f"[INFO] Wrote {args.output} ({len(entries)} entries)")

    # Cross-check with the actual validator goal_inference.py uses.
    from humanoidverse.goal_inference import load_and_validate_goal_json

    validated = load_and_validate_goal_json(args.output, num_dof=num_dof)
    print(f"[INFO] load_and_validate_goal_json OK: {len(validated)} entries, num_dof={num_dof}")


if __name__ == "__main__":
    main()
