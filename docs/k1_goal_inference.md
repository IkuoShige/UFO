# K1 Goal Inference

`humanoidverse/goal_inference.py` needs a robot-specific goal JSON for any
robot that isn't the 29-DoF G1 (`_find_goal_json` hard-rejects other DoF
counts unless `--goal-json` is passed explicitly — see
[`docs/TRAIN_INFERENCE.md`](TRAIN_INFERENCE.md#goal-inference) for the
general CLI). This doc covers the K1-specific goal JSON, how it was
generated, a pipeline bug it exposed (and the fix), and how to reproduce a
"standing" goal latent `z` for K1.

## Files

- `humanoidverse/tools/make_k1_goal_json.py` — generator (see its module
  docstring for the full method). Run with:
  ```bash
  uv run python -m humanoidverse.tools.make_k1_goal_json
  ```
- `humanoidverse/data/robots/k1/goal_frames_k1_22dof.json` — generated output,
  6 entries, validated against `load_and_validate_goal_json(num_dof=22)`.

## How the goal frames were derived

For every frame of every motion in
`cache/motion_data/k1_lafan1/k1_lafan1_full_ufo.pkl` (77 motions, built by the
manifest pipeline from `configs/data/k1_lafan1.yaml`), the generator computes:

- **torso tilt**: angle between the retargeted root quaternion's local +z
  axis and world +z.
- **height deviation**: `|root height - configs/robots/k1_22dof.yaml
  training.init_state.pos.z (0.53)|`.
- **leg deviation**: L2 norm, over the 12 leg DOFs only, between that frame's
  `dof_pos` and the config's `default_joint_angles` (itself the median
  upright-frame pose of this same dataset — used here only as a cross-check
  reference, not copied into the output).
- **whole-body joint speed**: L2 norm of the central-difference `dof_pos`
  velocity over all 22 DOFs, to reject frames where the legs are momentarily
  still but the arms are mid-swing.

A frame counts as "quiet-standing" if tilt < 5°, height deviation < 0.03 m,
and whole-body speed < 0.5 rad/s. Runs of ≥15 consecutive quiet-standing
frames (0.5 s at the dataset's native 30 fps) are kept; each run contributes
its center frame as a candidate, tagged with these diagnostics.

Two goal groups are emitted:

- **`standing`** (5 entries): the lowest-leg-deviation candidate from each of
  the 5 distinct motions with the lowest leg deviation overall — diversity
  across walk/run/obstacle/etc. motions rather than one clip.
- **`settled_two_feet_stance`** (1 entry): the lowest-leg-deviation candidate
  whose quiet run is preceded, earlier in the same
  `fallAndGetUp*`/`pushAndFall*` motion, by a frame with height < 0.35 m —
  i.e. an actual post-recovery standing moment (not a pre-fall idle stance),
  directly relevant to a get-up policy's target state.

The **first entry in the JSON is the canonical "standing" goal**
(`obstacles4_subject2`, lowest leg deviation of all candidates: leg_dev=0.42
rad, tilt=1.9°, height=0.540 m, at `diagnostics.source_frame_30fps=4188`).
Note: at this frame the head is yawed ~0.62 rad (the robot is looking around
mid-obstacle-course), which is baked into this goal's pose — if that matters
downstream, `sprint1_subject2` (2nd entry) is a similar-quality alternative
with a more neutral head pose.

### Units: `frames` are control-step indices, not raw mocap frame numbers

This was the trickiest part. `run_goal_inference` builds its per-motion
observation buffer via `get_backward_observation`, which resamples the
motion at `env.dt` — the MJLab **control step**, `0.02 s` (50 Hz: `sim.fps
=200`, `control_decimation=4` in `humanoidverse/config/simulator/
mujoco.yaml`) — not at the motion's native capture fps (30 Hz for this
LAFAN1 dataset). `goal["frames"]` values index directly into that
`env.dt`-sampled buffer.

The generator analyzes upright frames in raw 30 fps units (matching the
source CSV/pkl), then converts the chosen frame to a control-step index via
`round(raw_frame / motion_fps / env_dt)` before writing it to the JSON. The
original 30 fps frame number is kept per-entry as
`diagnostics.source_frame_30fps` for traceability. Verified exactly: for the
canonical entry, `gobs_dict["dof_pos"][6980]` (from `get_backward_observation`)
and the raw pkl's `dof_pos[4188]` for `obstacles4_subject2` are bit-identical.

If you regenerate against a different simulator config (different
`control_decimation`/`sim.fps`), pass `--sim-dt` accordingly.

### `motion_id` ↔ pkl order

`motion_id` in each entry is the index `MotionLibRobot` assigns when it loads
the cache pkl. With the default `im_eval=False, min_length=-1`, the motion
lib preserves dict insertion order, so `motion_id == list(pkl.keys())
.index(motion_name)`. This is only valid for the current 77-motion cache; if
`k1_lafan1_full_ufo.pkl` is ever rebuilt from different/reordered source
CSVs, re-run the generator (it re-derives `motion_id` from the same
`prepare_manifest_dataset_path` resolver `goal_inference.py` uses, so it's
one command, not a manual fix).

## Bug found and fixed: global vs. local `motion_id` in `run_goal_inference`

Running the tool against the K1 goal JSON (which references 6 different
motions, i.e. `motion_id != 0` for 5 of 6 entries) surfaced a pre-existing,
**robot-agnostic** indexing bug in `run_goal_inference`, not something
specific to K1:

- `env.set_is_evaluating(motion_id)` calls
  `MotionLibRobot.load_motions_for_evaluation(start_idx=motion_id * num_envs)`.
  Since `goal_inference.py` always builds the env with `num_envs=1`, this
  loads **exactly the requested motion into a length-1 local buffer, at local
  index 0** (see `load_motions_for_evaluation` / `load_motions` in
  `humanoidverse/utils/motion_lib/motion_lib_base.py`).
- The very next line, `get_backward_observation(env, motion_id, ...)`,
  indexes that length-1 buffer with the **original global** `motion_id` —
  correct only when `motion_id == 0`. For any other value it raises
  `IndexError: index <motion_id> is out of bounds for dimension 0 with size 1`.

This is not new in this change and would equally break G1's own reference
`goal_frames_lafan29dof.json` (which has entries with `motion_id` 1–7) if run
through `goal_inference.py` as-is — it just hadn't been exercised with a
goal JSON spanning more than one motion before. `humanoidverse/tools/
eval_goal_joint_mae.py:118` has the identical pattern and the identical
latent bug; it was left as-is here (out of scope for this change) but should
get the same fix if/when it's exercised with multi-motion goal JSONs.

**Fix applied** (`humanoidverse/goal_inference.py`, `run_goal_inference`):
load every motion up front, exactly like `tracking_inference.py` already
does —

```python
wrapped_env, _ = env_cfg.build(num_envs=1)
env = wrapped_env._env
env._motion_lib.load_all_motions()
```

With all motions loaded, every subsequent `env.set_is_evaluating(motion_id)`
call becomes a cheap no-op reload (`MotionLibRobot.load_motions_for_evaluation`
early-returns once `all_motions_loaded` is `True`) plus a `reset_all()`, and
`get_backward_observation(env, motion_id, ...)` correctly indexes the full,
77-motion-wide buffers. This does not touch goal-JSON validation or the
non-G1 robot guard. Loading all motions took ~15s locally for the 77-motion
K1 LAFAN1 set (single-process; matches `tracking_inference.py`'s existing
behavior on the same dataset).

## Running goal inference for K1

```bash
nvidia-smi   # check the shared GPU before running
MUJOCO_GL=egl uv run python -m humanoidverse.goal_inference \
  --model-folder runs/ufo_fb_k1_5090_v2 \
  --robot-config configs/robots/k1_22dof.yaml \
  --data-manifest configs/data/k1_lafan1.yaml --dataset k1_lafan1 \
  --goal-json humanoidverse/data/robots/k1/goal_frames_k1_22dof.json \
  --disable-dr --disable-obs-noise --headless
```

This writes `runs/ufo_fb_k1_5090_v2/goal_inference/goal_reaching.pkl`, a dict
of 6 entries keyed `f"{motion_name}_{frame_idx}"` (`frame_idx` in
control-step units, e.g. `obstacles4_subject2_6980`), each a
`(1, 256)` float32 numpy array (`z_dim=256` for this checkpoint).

### Extracting the canonical standing goal `z`

WS-A (get-up evaluation) consumes a plain `(256,)` float32 tensor at
`runs/getup_eval/z_goal_standing.pt`:

```python
import joblib, torch

z_dict = joblib.load("runs/ufo_fb_k1_5090_v2/goal_inference/goal_reaching.pkl")
z = torch.as_tensor(z_dict["obstacles4_subject2_6980"], dtype=torch.float32).squeeze(0)
assert z.shape == (256,)
torch.save(z, "runs/getup_eval/z_goal_standing.pt")
```

This was run locally on the shared RTX 3090 (inference only); the saved
tensor has `norm() == 16.0`.
