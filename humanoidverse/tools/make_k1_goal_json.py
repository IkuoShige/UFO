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

### Arm/hand clearance filter

A WS-A evaluation (`docs/k1_getup_eval.md` Sec. 4.2) found that the previous
version of this generator's top candidates all rest the hands against the
hips (`left_hand_link` <-> `Left_Hip_Yaw` and the mirror), causing continuous
self-collision (~93% of rollout frames) once turned into goal latents and
replayed closed-loop -- a hardware wear/force-estimation problem, even though
those latents were otherwise the most DR-robust candidates found. This is a
geometric property of the candidate pose, not something the tilt/height/speed
filters above can see, so each candidate run's center frame is additionally
checked for **arm clearance**: forward-kinematics the K1 MuJoCo model
(`../booster_assets/robots/K1/K1_22dof.xml`) to that frame's root pose + dof
pose, then compute the exact signed distance (`mujoco.mj_geomDistance`, not a
contact/margin test) between every collision geom on `left_hand_link` /
`right_hand_link` and every collision geom on `Trunk`, `Left_Hip_Roll`,
`Left_Hip_Yaw`, `Right_Hip_Roll`, `Right_Hip_Yaw` -- exactly the body set
implicated in the disqualifying report. The minimum over all those geom pairs
is the candidate's `clearance_m`; candidates below `--min-clearance` (default
0.08 m) are dropped before goal-entry selection.

The threshold is a label for an empirical gap, not a tuned value: across the
92 quiet-standing runs found in the K1 LAFAN1 set, clearance is sharply
bimodal -- one cluster at -0.02 to +0.07 m (hands pressed against the hip,
sometimes penetrating) and another at +0.18 to +0.20 m (hands hanging clear
at the sides), with a zero-candidate gap from 0.068 to 0.179 m between them.
Any threshold in that gap selects the identical 69/92 surviving runs; 0.08 m
was picked as a round number inside it, and the tool prints the survival
count at several thresholds every run so this is auditable rather than
asserted. Surviving candidates trade a somewhat worse `leg_dev` for arms held
away from the body -- `full_dev` (which includes the 10 arm/head DOFs) rises
accordingly; this is the filter doing its job, not a regression against the
leg-focused diagnostics above.

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
import mujoco
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

# Bodies implicated in the self-collision report (docs/k1_getup_eval.md Sec. 4.2):
# hand-link geoms resting against hip/torso geoms.
HAND_BODIES = ("left_hand_link", "right_hand_link")
HIP_TORSO_BODIES = ("Trunk", "Left_Hip_Roll", "Left_Hip_Yaw", "Right_Hip_Roll", "Right_Hip_Yaw")
DEFAULT_MIN_CLEARANCE = 0.08  # meters; see module docstring for the empirical-gap justification.
CLEARANCE_REPORT_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15)


def _collision_geom_ids(model: "mujoco.MjModel", body_name: str) -> list[int]:
    """Geoms on ``body_name`` that actually participate in MuJoCo collision detection

    (``contype != 0 or conaffinity != 0``) -- excludes the contype=0/conaffinity=0 visual
    meshes the K1 XML layers over most collision primitives.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in {model}")
    return [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) == body_id and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)
    ]


def compute_arm_clearance(
    motion_data: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    xml_path: Path,
) -> None:
    """Adds ``clearance_m`` / ``clearance_pair`` to each candidate dict, in place.

    For each candidate's center frame: forward-kinematics the K1 MuJoCo model to that
    frame's root pose + dof_pos (mirrors ``reference_qpos`` in
    ``humanoidverse/tools/eval_getup.py``: ``qpos = [root_pos, roll(root_quat_xyzw, 1),
    dof_pos]``, control-joint order == MuJoCo qpos-address order for K1), then take the
    minimum ``mujoco.mj_geomDistance`` (exact signed distance, not a margin/contact
    test) over every (hand-link geom, hip/torso geom) pair.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    hand_geoms = [g for b in HAND_BODIES for g in _collision_geom_ids(model, b)]
    hip_torso_geoms = [g for b in HIP_TORSO_BODIES for g in _collision_geom_ids(model, b)]

    def body_of(geom_id: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        return str(name) if name else f"body{model.geom_bodyid[geom_id]}"

    qpos = np.zeros(model.nq, dtype=np.float64)
    for c in candidates:
        entry = motion_data[c["motion_name"]]
        frame = c["frame"]
        root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float64)[frame]
        quat_xyzw = np.asarray(entry["root_quat"], dtype=np.float64)[frame]
        dof = np.asarray(entry["dof_pos"], dtype=np.float64)[frame]
        qpos[0:3] = root_pos
        qpos[3:7] = np.roll(quat_xyzw, 1)  # xyzw -> wxyz
        qpos[7:] = dof
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        best_dist = float("inf")
        best_pair: tuple[str, str] | None = None
        for g1 in hand_geoms:
            for g2 in hip_torso_geoms:
                dist = mujoco.mj_geomDistance(model, data, g1, g2, 1.0, None)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (body_of(g1), body_of(g2))
        c["clearance_m"] = float(best_dist)
        c["clearance_pair"] = best_pair


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
        "clearance_m": round(c["clearance_m"], 4),
        "clearance_pair": list(c["clearance_pair"]) if c["clearance_pair"] else None,
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
        "--min-clearance",
        type=float,
        default=DEFAULT_MIN_CLEARANCE,
        help=(
            "Minimum MuJoCo mj_geomDistance (meters) between hand-link and hip/torso collision "
            "geoms at a candidate's center frame; candidates below this are dropped. See module "
            "docstring for the empirical-gap justification of the default."
        ),
    )
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

    xml_path = Path(training_spec.robot.xml_path)
    compute_arm_clearance(motion_data, candidates, xml_path=xml_path)
    clearances = np.array([c["clearance_m"] for c in candidates])
    print(f"[INFO] Arm/hand-vs-hip/torso clearance ({xml_path}): "
          f"min={clearances.min():.4f} median={np.median(clearances):.4f} max={clearances.max():.4f}")
    print("[INFO] Survival by clearance threshold (audit table):")
    for t in sorted(set(CLEARANCE_REPORT_THRESHOLDS) | {args.min_clearance}):
        n = int((clearances >= t).sum())
        marker = "  <-- selected" if t == args.min_clearance else ""
        print(f"    >= {t:.3f} m: {n:3d}/{len(candidates)} runs survive{marker}")

    candidates = [c for c in candidates if c["clearance_m"] >= args.min_clearance]
    if not candidates:
        raise RuntimeError(f"No candidates survive --min-clearance={args.min_clearance}; lower it.")
    print(f"[INFO] {len(candidates)} candidates survive --min-clearance={args.min_clearance} m "
          f"(dropped {len(clearances) - len(candidates)} for hand/hip contact risk)")

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
