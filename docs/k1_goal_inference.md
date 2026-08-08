# K1 Goal Inference

`humanoidverse/goal_inference.py` needs a robot-specific goal JSON for any
robot that isn't the 29-DoF G1 (`_find_goal_json` hard-rejects other DoF
counts unless `--goal-json` is passed explicitly — see
[`docs/TRAIN_INFERENCE.md`](TRAIN_INFERENCE.md#goal-inference) for the
general CLI). This doc covers the K1-specific goal JSON, how it was
generated, a pipeline bug it exposed (and the fix), an arm-clearance
self-collision fix applied after that, and how to reproduce a "standing"
goal latent `z` for K1.

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
(`obstacles4_subject2`, lowest leg deviation of all *clearance-surviving*
candidates: leg_dev=0.486 rad, tilt=3.5°, height=0.548 m, at
`diagnostics.source_frame_30fps=81`). See "Arm/hand clearance filter" below
for why this is a different frame of the same motion than an earlier version
of this doc reported (`source_frame_30fps=4188`, leg_dev=0.42) — that frame
was disqualified for self-collision.

## Arm/hand clearance filter (self-collision fix)

A WS-A closed-loop evaluation (`docs/k1_getup_eval.md` §4.2) found that the
first version of this generator's top candidates all had the hands resting
against the hips — the arm/head DOFs weren't filtered at all, only the 12 leg
DOFs (`leg_dev`) and a whole-body speed gate, so a frame could be "quiet
standing" by that definition while the arms sat in an out-of-distribution,
self-intersecting pose. Once turned into goal latents and replayed
closed-loop, all three of the evaluated goal-derived latents self-collided on
**~93% of rollout frames** (`left_hand_link` ↔ `Left_Hip_Yaw` and the mirror
pair, dominant in every case) against 1.5–2.4% for the deployed/reward-based
candidates — a hardware wear/force-estimation problem, even though these
latents were otherwise the *most* domain-randomization-robust candidates
found (DR success 33–58% vs. the deployed champion's 27%).

This is a geometric property of the candidate pose that the tilt/height/speed
filters cannot see, so `make_k1_goal_json.py` now additionally
forward-kinematics the K1 MuJoCo model
(`../booster_assets/robots/K1/K1_22dof.xml`) to each candidate run's center
frame (root pose + full 22-DOF `dof_pos`) and computes the exact signed
distance (`mujoco.mj_geomDistance` — an analytic query, not a contact/margin
test) between every collision geom on `left_hand_link` / `right_hand_link`
and every collision geom on `Trunk`, `Left_Hip_Roll`, `Left_Hip_Yaw`,
`Right_Hip_Roll`, `Right_Hip_Yaw` — the exact body set implicated in the
report above. The minimum over all those geom pairs is the candidate's
`clearance_m` (now also written into each entry's `diagnostics`); candidates
below `--min-clearance` (default **0.08 m**) are dropped before goal-entry
selection.

**Threshold justification.** Across the 92 quiet-standing runs found in the
K1 LAFAN1 set, clearance is sharply bimodal: one cluster at **-0.02 to
+0.07 m** (hands pressed against the hip, some frames already penetrating —
this is the disqualified family) and another at **+0.18 to +0.20 m** (hands
hanging clear at the sides), with a **zero-candidate gap from 0.068 m to
0.179 m** between them. Any threshold in that gap selects the identical
surviving set, so 0.08 m is not a tuned value — it's a round number placed
inside an empirical gap. The generator prints the survival count at several
thresholds every run, e.g. for the current 77-motion LAFAN1 cache:

| threshold (m) | 0.00 | 0.01 | 0.02 | 0.05 | **0.08** | 0.10 | 0.15 |
|---|---|---|---|---|---|---|---|
| runs surviving / 92 | 84 | 78 | 76 | 72 | **69** | 69 | 69 |

At the selected threshold, 69/92 runs (45 distinct motions) survive; the
5 "standing" entries and the 1 "settled_two_feet_stance" entry are then
picked from that filtered set exactly as before (lowest `leg_dev`, one per
motion / best post-recovery candidate). 3 of the 5 "standing" candidates from
the old (unfiltered) generation already happened to have clear arms and are
unchanged; `obstacles4_subject2` and `sprint1_subject2`'s old center frames
were both hand-on-hip and got replaced by different frames of the same
(`obstacles4_subject2`, `aiming2_subject2`) or a different (`aiming2_subject2`
for the old `sprint1_subject2` slot) motion. The old `settled_two_feet_stance`
candidate (`fallAndGetUp3_subject1`) had no clearance-surviving frame at all
in its motion and was replaced by `fallAndGetUp1_subject4`.

Surviving candidates trade a somewhat worse `leg_dev` for arms held away from
the body: `full_dev` (which includes the 10 arm/head DOFs, previously
unfiltered) rises from ~1.0–1.4 rad to ~2.3–3.1 rad for the "standing" group.
This is the filter doing its job — the leg pose (what the walk-handoff
analysis in `docs/k1_getup_eval.md` §5 scores) is barely affected — not a
regression.

### Before / after: closed-loop self-collision

Re-ran `humanoidverse/tools/eval_getup.py` on the new latents with the exact
condition that produced the disqualifying numbers above
(`bucket=indist`, `num_envs=64`, `seed=0`, `disable_dr=disable_obs_noise=True`,
`settle_steps=50`, `episode_steps=500` — i.e. `runs/getup_eval/
summary_indist_nominal.json`'s config):

| candidate | self-collision frac (before) | self-collision frac (after) | success |
|---|---|---|---|
| `goal_standing_canonical` (`obstacles4_subject2`) | 0.929 | **0.011** | 1.00 |
| `goal_fallAndGetUp3_stand` → `goal_settled_clearance` (`fallAndGetUp1_subject4`) | 0.926 | **0.012** | 1.00 |
| `standing_pooled` (deployed champion, reference) | 0.015 | — (unchanged) | 1.00 |

Both new latents land at or below the deployed champion's self-collision
rate, and success (get up + hold a stationary two-foot stance) stays at
100%. The residual ~1% is transient contact during the settle/get-up
transient (the same pattern `standing_pooled` shows at 1.5%), not sustained
hand-on-hip contact — the dominant pairs (`Left_Hip_Yaw`/`Trunk`/`Left_Hip_Roll`
↔ `left_hand_link`, 15–42 frame-hits each out of ~7000 sampled frames) are an
order of magnitude rarer than the 3300–5300 hits/candidate in the disqualified
version.

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
`diagnostics.source_frame_30fps` for traceability. Verified exactly (against
the pre-clearance-fix canonical entry, `obstacles4_subject2` frame 4188/6980
— the conversion logic is unchanged by the clearance fix, only which frame
gets chosen): `gobs_dict["dof_pos"][6980]` (from `get_backward_observation`)
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
control-step units — after the clearance fix these are
`obstacles4_subject2_135`, `jumps1_subject2_48`, `fallAndGetUp2_subject2_80`,
`dance1_subject2_177`, `aiming2_subject2_45`, `fallAndGetUp1_subject4_8325`),
each a `(1, 256)` float32 numpy array (`z_dim=256` for this checkpoint).
Re-running this command **overwrites** `goal_reaching.pkl` with whatever the
current `goal_frames_k1_22dof.json` says — the old (pre-clearance-fix) keys
(e.g. `obstacles4_subject2_6980`, `sprint1_subject2_5347`,
`fallAndGetUp3_subject1_1963`) no longer exist in it once regenerated.

### Extracting the goal latents

```python
import joblib, torch

z_dict = joblib.load("runs/ufo_fb_k1_5090_v2/goal_inference/goal_reaching.pkl")
out = {}
for k, v in z_dict.items():
    z = torch.as_tensor(v, dtype=torch.float32).squeeze(0)
    assert z.shape == (256,)
    out[k] = z
torch.save(out, "runs/getup_eval/z_goal_clearance.pt")
```

Run locally on the shared RTX 3090 (inference only); every tensor has
`norm() == 16.0` (`project_z` with `norm_z=true`). This dict-of-6 file is
consumed downstream (`kind: "file"`, `key: "<name>"` in an
`eval_getup.py`/`opt_getup_z.py` candidate spec) — the canonical "standing"
entry is `obstacles4_subject2_135`, the post-recovery entry is
`fallAndGetUp1_subject4_8325`.

Historical note: `runs/getup_eval/z_goal_standing.pt` (produced by an earlier
version of this doc, key `obstacles4_subject2_6980`) encodes the
**disqualified**, pre-clearance-fix canonical latent (93% self-collision, see
above) and was left on disk as-is, not regenerated in place — use
`z_goal_clearance.pt` going forward.
