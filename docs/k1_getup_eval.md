# K1 get-up evaluation: which BFM latent `z` gets the robot back on its feet?

Workstream A. Checkpoint under test: `runs/ufo_fb_k1_5090_v2/checkpoint` (FB-CPR / `FBcprAuxModel`,
`z_dim=256`, 192M env steps, K1 22-dof). Harness: `humanoidverse/tools/eval_getup.py`.
Raw results: `runs/getup_eval/` (`episodes_<tag>.csv`, `summary_<tag>.json`, `z_candidates_<tag>.pt`).

## Headline

**The BFM already knows how to get up. The latent that unlocks it is a *standing* latent, not a
get-up latent — and every constant `z` derived from a get-up motion fails completely.**

From a fallen pose drawn from the training `lie_down_init` distribution, a single constant
`z` (`standing_pooled`, saved in `runs/getup_eval/z_bank.pt`) stands the K1 up in **0.87 s** with
**100% success over 128 episodes** (≥97.7% at 95% confidence, rule of three), holds 2.1° max tilt,
saturates a joint torque limit on 0.1% of joint-steps and self-collides on 1.5% of frames. The same
`z` scores **100% at 0.87 s on the out-of-distribution bucket** (supine/prone/side-lying swept over
root yaw) — the OOD sweep is not harder for it at all.

Meanwhile *every* latent built from a get-up motion segment — mean over one segment, mean over all
segments, pooled across four fall/get-up clips — scores **0/64**. Not "gets up slowly": never rises
at all, peaking at 0.15–0.33 m against a 0.53 m standing pose. §2 explains why, and it is the most
transferable result here.

Under the full training domain randomization plus observation noise and action latency, **100% of
episodes still get up and 0/128 ever fall back down** — final root height never drops below 0.526 m.
What degrades is only the strict "both feet planted continuously" term, and the ablation in §4.1
isolates the cause: the DR *pushes*. With physics DR (friction, link mass, COM) but no pushes, the
strict criterion is back to **100%**.

**Recommendation: ship a constant `z`; no distillation, no schedule, no fine-tuning for this skill.**
See §6.

## 1. The harness

`humanoidverse/tools/eval_getup.py` is a standalone rollout harness built on the same
`load_mjlab_env_cfg` path as `tracking_inference.py` / `reward_inference.py`, so the robot config,
actuator model, observation layout and DR switches are identical to training and to the other
inference entrypoints. It is deliberately split into a generic half (rollout loop, z modes, DR,
metrics aggregation, CSV/JSON/mp4 reporting) and a skill-specific half (fallen-pose bank, standing
criterion, get-up scoring), because walk / run / kick are meant to come through the same pipeline.

### 1.1 Why the episodes never terminate early

The MJLab env registers exactly one termination term, `time_out`
(`humanoidverse_mjlab.py:make_mjlab_ufo_env_cfg`), and `_compose_humanoidverse_config` asserts every
HumanoidVerse termination (low height, bad gravity, contact, motion-far, ...) is disabled. The
harness sets `max_episode_length_s = 1e5`, so nothing can reset mid-episode — a fallen robot is
allowed to stay fallen and be scored for it. The rollout still checks the terminated/truncated flags
every step and raises if any fires, because a silent reset would resample the motion library and
corrupt the episode.

### 1.2 Fallen-pose bank

Initial states are written directly through `reset_idx(..., target_states=...)`, which bypasses
motion sampling entirely and writes root state (xyzw) plus joint pos/vel.

**`indist`** replicates training's `lie_down_init` (`humanoidverse_mjlab.py:1084-1097`) exactly: take a
motion-library reference frame (with `offset=env_origins`, keeping the frame's own velocities), force
`root_pos[2] = 0.5`, and premultiply the root rotation by a ±90° rotation about the **world** x axis.
Training adds no initial noise here (`noise_to_initial_level: 0`), so neither does the harness. Two
deliberate deviations, both documented in code: the roll sign is drawn per-env rather than once per
reset batch (identical per-episode marginal, decorrelates episodes within a batch), and reference
frames are drawn from the full 77-clip library rather than the 10 s training clips.

**`ood`** builds a fallen pose from scratch: an upright default pose (optionally joint-randomized),
tilted supine / prone / side-lying, then swept over root yaw:
`R = Rz(yaw) · R_tilt`, dropped from 0.30 m with zero velocity.

The yaw sweep is the axis that matters. Working through the algebra of `lie_down_init` on an
upright source frame, `Rx(∓90°)·Rz(heading)` maps the body z axis (head direction) to world ±y
*for every heading* — the heading only rotates the chest normal. So training's lie-down bank already
spans supine-through-prone chest orientations, but always with the head along world ±y. **Head
direction, not chest orientation, is what training never varied**, and that is what the OOD bucket
sweeps. The OOD side-lying tilts therefore partially overlap the training distribution; supine/prone
at off-axis yaw do not.

**`motion_frame`** is the aligned control: reset onto an exact reference frame with no lie-down
transform, used to give z-sequence replay its best case (see §4).

**Settle window.** After the reset, physics runs for 1.0 s with zero action (PD holds the default
pose) before the policy takes over and before any timer starts. Measured root speed at takeover is
0.016 m/s / 0.13 rad/s, i.e. fully settled. This *is* a deviation from training, where the policy
takes over mid-drop from 0.5 m; `--settle-steps 0` reproduces the training condition.

### 1.3 Standing criterion (explicit, and arbitrary in its thresholds)

An episode counts as standing at step *t* when all three hold:

| quantity | threshold | rationale |
|---|---|---|
| Trunk (root) height | ≥ 0.45 m | K1 default standing pose is 0.53 m. In the retargeted LAFAN1 K1 data the root is 0.535 m median while walking (1st pct 0.461), 0.31 m median in the `ground2` sitting clip, < 0.17 m when fallen. 0.45 separates standing from kneeling with margin both ways. |
| uprightness `-projected_gravity_z` | ≥ 0.90 | tilt ≤ 25.8° |
| both feet contact force | > 5 N each | K1 weighs 19.67 kg (≈193 N), so ≈96 N/foot when standing on two feet |

- **reached_stand**: the three hold *continuously* for 0.5 s. `time_to_stand` is the start of that window, measured from policy takeover.
- **success**: reached_stand **and** the criterion holds for ≥95% of the final 2.0 s of the episode.

So success means "**got up and then held a stationary two-foot stance**". This matters for reading
the table: a candidate that stands up and then walks away scores 0. The per-criterion occupancy
columns (`frac_height_ok`, `frac_upright_ok`, `frac_feet_ok`) are in every CSV row precisely so this
is visible rather than hidden — they are what showed that the random-z rows below fail on the feet
term while satisfying height and uprightness.

The thresholds are a judgement call. They are written into every `summary_*.json`, so results can be
re-read against a different definition without re-running.

### 1.4 Other metrics

Per episode: time-to-stand; post-stand stability (root-height std and max tilt over the hold window,
plus a `refell` flag); peak joint torque, peak torque / effort-limit ratio and saturation fraction
(vs the per-joint `effort_limit` in `configs/robots/k1_22dof.yaml`); mean |τ·ω|; undesired-contact
fraction (Trunk/hip contacts, using the same `|component| > 1 N` test as the training penalty);
self-collision fraction; and max ground penetration.

Self-collision and ground penetration are computed **offline**, by replaying the recorded qpos
through a CPU MuJoCo model and inspecting contact pairs — the MJLab contact sensor reduces to a
per-body net force and cannot separate self contact from floor contact. K1 has no contact
exclusions (`nexclude=0`, all 20 collision geoms `contype=conaffinity=1`), so self-collision is
physically possible. Caveat: this sees policy-rate frames only, so brief sub-step contacts are
under-counted.

### 1.5 z input modes

- **constant** — one z for the whole episode (closed-loop: the policy reacts, the command does not).
- **sequence** — open-loop, time-indexed replay of a per-frame z sequence, holding the last value past the end.
- **two_phase** — provider A for T seconds, then provider B.

z sources: `motion_getup_mean` / `motion_segment_mean` / `motion_segment` (from the model's own
backward map), `motion_standing_mean`, `file` (.pt/.npy/.npz/.pkl, with an optional dict key — this
is how the goal-inference and reward-inference latents drop in), `random`, `basis`, `vector`.

**z is recomputed from the checkpoint under test, not read from the existing `zs_*.pkl`.**
`backward_map` is weight-dependent, so a pkl written at an earlier checkpoint encodes a different
latent space. Verified both directions: the harness's on-the-fly z for motion 16 matches
`tracking_inference/zs_16.pkl` (final checkpoint) to **cosine 1.000000** per frame across all 8194
frames, while the 157M-checkpoint `zs_55.pkl` differs from the final checkpoint's by mean cosine
**0.975** — i.e. the 157M pkls for motions 17/27/65 are stale and were not used.

### 1.6 Get-up segment detection

For candidates derived from get-up motions, the harness locates get-up segments in the reference
itself: the end of a segment is the first up-crossing of root height 0.45 m after having been below
0.25 m, and the start is the *lowest* frame in the preceding 8 s window (not merely the last frame
under 0.25 m, which lands mid-rise and clips off the hardest part). Sanity checks: 12+ segments in
each `fallAndGetUp2` clip spanning h≈0.02–0.09 → 0.45 with durations 1.1–7 s (typically 2–4 s), 3 in
`ground2`, **0 in `walk1`** — the negative control, since walking never dips below 0.25 m.

### 1.7 Determinism and conditions

One `numpy.random.default_rng([seed, batch])` per (candidate, batch) drives motion choice, sample
time, roll sign, OOD tilt/yaw/joint noise and latency draws, so **every candidate sees exactly the
same pose bank**. Reported conditions:

| column | meaning |
|---|---|
| `*_nominal` | `--disable-dr True --disable-obs-noise True`: no DR, no observation noise, no latency |
| `*_dr` | training DR **+** harness latency **+** observation noise |

The DR condition is exactly the training randomization: robot-geom friction 0.5–1.25, link mass
scale 0.95–1.05, torso COM ±2 cm, and velocity pushes of ±0.5 m/s xy / ±0.5 rad/s at random
intervals of 1–3 s; plus observation noise (ang-vel 0.2, projected-gravity 0.05, dof-pos 0.01); plus
harness-side action latency 0–2 steps and observation latency 0–1 step drawn per env. Latency is
implemented in the harness because MJLab's config path silently drops `randomize_ctrl_delay`. Two
honest caveats: MJLab's friction DR randomizes the **robot's** geoms, not the terrain plane (MuJoCo
combines the two, so the effect on the contact pair is similar but not identical to "ground
friction"); and mass/COM/friction are *startup* events, so all candidates within a condition see
identical per-env physics — good for comparability, but it means the DR column samples 64 physics
draws, not 64 independent ones per candidate.

## 2. Why get-up-derived latents fail (the central mechanism)

The most useful result is a negative one. **Every constant `z` derived from a get-up motion
segment fails to stand — 0/64 on every variant tried** (mean over one segment, mean over all
segments of a clip, pooled across `fallAndGetUp2_s2/s3`, `pushAndFall1_s4`, `ground2_s2`). They do
not merely fail to *hold* a stance; they never rise: max root height 0.15–0.33 m against a 0.53 m
standing pose, with `frac_height_ok = 0.00` for the whole 10 s episode.

Rendered, the behaviour is unmistakable: the robot props itself onto hands and knees and stays
there. Root height goes 0.06 → 0.09 → 0.18 → 0.19 and settles; uprightness plateaus at 0.75 (41°).

The explanation is the semantics of the FB latent. `z` conditions the actor toward a target
**state occupancy**, not a trajectory to execute. The average state over a get-up segment is
dominated by the low-to-the-ground part of the motion — a get-up starts fallen and only briefly
ends upright — so the mean latent over that segment encodes "be near the floor, propped up", and
the policy faithfully delivers exactly that. Averaging over the segment also throws away most of
the signal: within a single get-up segment the per-frame `z` values have mean cosine 0.5–0.7 to
their own mean (min as low as 0.05), so a single mean is a poor summary of a fast, non-stationary
motion.

**The latent that gets the robot up is a *standing* latent.** You do not command "get up"; you
command "stand still", and the policy solves getting there.

### 2.1 Full sweep — 27 candidates x 64 episodes per cell

Success rate (strict: got up **and** held a stationary two-foot stance) / mean time-to-stand.

| candidate | z mode | indist nominal | indist dr | ood nominal | ood dr |
|---|---|---|---|---|---|
| `standing_pooled` | constant | 100% / 0.87s | 27% / 1.33s | 100% / 0.87s | 31% / 1.28s |
| `goal_sprint1_stand` | constant | 100% / 0.88s | 33% / 1.34s | 100% / 0.88s | 30% / 1.49s |
| `goal_standing_canonical` | constant | 100% / 0.93s | 39% / 1.47s | 100% / 0.87s | 45% / 1.54s |
| `goal_fallAndGetUp3_stand` | constant | 100% / 0.95s | 58% / 1.41s | 100% / 0.87s | 59% / 1.42s |
| `reward_move_ego_0_0` | constant | 100% / 1.04s | 38% / 1.72s | 100% / 0.96s | 44% / 1.20s |
| `tp_replay_then_stand_3s` | two_phase | 100% / 3.17s | 42% / 3.45s | 100% / 3.18s | 33% / 3.56s |
| `tp_pooled_then_stand_3s` | two_phase | 100% / 3.83s | 31% / 4.33s | 100% / 3.85s | 38% / 4.15s |
| `tp_replay_then_goalstand_4s` | two_phase | 100% / 4.06s | 41% / 4.21s | 100% / 4.06s | 39% / 4.33s |
| `m17_seg1_replay` | sequence | 100% / 6.08s | 52% / 6.56s | 100% / 6.11s | 52% / 6.51s |
| `m16_seg0_replay` | sequence | 100% / 7.97s | 25% / 8.08s | 100% / 7.98s | 17% / 8.10s |
| `m16_seg3_replay` | sequence | 92% / 4.39s | 17% / 4.83s | 88% / 4.38s | 27% / 4.64s |
| `getup_pooled_all` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m16_pooled_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m17_pooled_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m55_pooled_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m27_pooled_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m16_seg0_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m16_seg3_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m16_seg7_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m17_seg1_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m55_seg0_mean` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m16_seg7_replay` | sequence | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `m55_seg0_replay` | sequence | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `random_0` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `random_1` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `random_2` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |
| `random_3` | constant | 0% / —s | 0% / —s | 0% / —s | 0% / —s |

### 2.2 Quality detail, in-distribution / nominal

| candidate | success | time-to-stand | max tilt in hold | torque sat. | self-collision | ground penetration | peak torque |
|---|---|---|---|---|---|---|---|
| `standing_pooled` | 100% | 0.87 s | 2.1° | 0.001 | 0.015 | 0.41 cm | 35.6 Nm |
| `goal_sprint1_stand` | 100% | 0.88 s | 4.0° | 0.094 | 0.926 | 0.53 cm | 33.0 Nm |
| `goal_standing_canonical` | 100% | 0.93 s | 3.6° | 0.112 | 0.929 | 0.48 cm | 33.9 Nm |
| `goal_fallAndGetUp3_stand` | 100% | 0.95 s | 5.1° | 0.119 | 0.926 | 0.45 cm | 33.6 Nm |
| `reward_move_ego_0_0` | 100% | 1.04 s | 2.2° | 0.001 | 0.024 | 0.37 cm | 36.8 Nm |
| `tp_replay_then_stand_3s` | 100% | 3.17 s | 2.4° | 0.004 | 0.114 | 0.51 cm | 34.9 Nm |
| `tp_pooled_then_stand_3s` | 100% | 3.83 s | 2.1° | 0.005 | 0.055 | 0.69 cm | 37.2 Nm |
| `tp_replay_then_goalstand_4s` | 100% | 4.06 s | 3.6° | 0.079 | 0.797 | 0.48 cm | 35.7 Nm |
| `m17_seg1_replay` | 100% | 6.08 s | 2.4° | 0.054 | 0.577 | 0.59 cm | 33.5 Nm |
| `m16_seg0_replay` | 100% | 7.97 s | 6.0° | 0.026 | 0.416 | 0.49 cm | 37.5 Nm |
| `m16_seg3_replay` | 92% | 4.39 s | 4.4° | 0.036 | 0.795 | 0.52 cm | 35.3 Nm |
| `getup_pooled_all` | 0% | — s | —° | 0.012 | 0.151 | 0.78 cm | 33.2 Nm |
| `m16_pooled_mean` | 0% | — s | —° | 0.010 | 0.047 | 0.49 cm | 29.2 Nm |
| `m17_pooled_mean` | 0% | — s | —° | 0.001 | 0.034 | 0.69 cm | 28.5 Nm |
| `m55_pooled_mean` | 0% | — s | —° | 0.002 | 0.049 | 0.36 cm | 30.4 Nm |
| `m27_pooled_mean` | 0% | — s | —° | 0.003 | 0.435 | 0.52 cm | 31.9 Nm |
| `m16_seg0_mean` | 0% | — s | —° | 0.003 | 0.033 | 0.46 cm | 25.7 Nm |
| `m16_seg3_mean` | 0% | — s | —° | 0.002 | 0.024 | 0.45 cm | 29.4 Nm |
| `m16_seg7_mean` | 0% | — s | —° | 0.004 | 0.894 | 0.43 cm | 32.0 Nm |
| `m17_seg1_mean` | 0% | — s | —° | 0.037 | 0.253 | 0.43 cm | 27.1 Nm |
| `m55_seg0_mean` | 0% | — s | —° | 0.001 | 0.062 | 0.39 cm | 28.3 Nm |
| `m16_seg7_replay` | 0% | — s | —° | 0.006 | 0.149 | 0.52 cm | 30.8 Nm |
| `m55_seg0_replay` | 0% | — s | —° | 0.012 | 0.213 | 0.68 cm | 32.7 Nm |
| `random_0` | 0% | — s | —° | 0.002 | 0.068 | 0.34 cm | 28.3 Nm |
| `random_1` | 0% | — s | —° | 0.002 | 0.170 | 0.45 cm | 34.9 Nm |
| `random_2` | 0% | — s | —° | 0.006 | 0.027 | 0.36 cm | 36.2 Nm |
| `random_3` | 0% | — s | —° | 0.005 | 0.039 | 0.32 cm | 32.9 Nm |

### 2.3 Why each candidate failed (per-criterion occupancy, in-distribution / nominal)

Fraction of the 10 s episode each individual criterion held, plus the peak root height reached.

| candidate | height ok | upright ok | both feet ok | max root height | reading |
|---|---|---|---|---|---|
| `standing_pooled` | 0.93 | 0.95 | 0.92 | 0.530 m | stands and stays planted |
| `goal_sprint1_stand` | 0.93 | 0.94 | 0.92 | 0.527 m | stands and stays planted |
| `goal_standing_canonical` | 0.92 | 0.94 | 0.92 | 0.524 m | stands and stays planted |
| `goal_fallAndGetUp3_stand` | 0.92 | 0.94 | 0.92 | 0.518 m | stands and stays planted |
| `reward_move_ego_0_0` | 0.91 | 0.93 | 0.90 | 0.531 m | stands and stays planted |
| `tp_replay_then_stand_3s` | 0.69 | 0.69 | 0.75 | 0.529 m | rises late / intermittently |
| `tp_pooled_then_stand_3s` | 0.63 | 0.65 | 0.63 | 0.530 m | rises late / intermittently |
| `tp_replay_then_goalstand_4s` | 0.72 | 0.70 | 0.72 | 0.526 m | rises late / intermittently |
| `m17_seg1_replay` | 0.46 | 0.50 | 0.51 | 0.524 m | rises late / intermittently |
| `m16_seg0_replay` | 0.36 | 0.32 | 0.47 | 0.532 m | rises late / intermittently |
| `m16_seg3_replay` | 0.72 | 0.70 | 0.70 | 0.531 m | rises late / intermittently |
| `getup_pooled_all` | 0.00 | 0.00 | 0.01 | 0.257 m | **never rises** — stays on the ground |
| `m16_pooled_mean` | 0.00 | 0.00 | 0.80 | 0.216 m | **never rises** — stays on the ground |
| `m17_pooled_mean` | 0.00 | 0.00 | 0.00 | 0.225 m | **never rises** — stays on the ground |
| `m55_pooled_mean` | 0.00 | 0.00 | 0.00 | 0.247 m | **never rises** — stays on the ground |
| `m27_pooled_mean` | 0.00 | 0.00 | 0.45 | 0.271 m | **never rises** — stays on the ground |
| `m16_seg0_mean` | 0.00 | 0.00 | 0.80 | 0.152 m | **never rises** — stays on the ground |
| `m16_seg3_mean` | 0.00 | 0.00 | 0.01 | 0.239 m | **never rises** — stays on the ground |
| `m16_seg7_mean` | 0.00 | 0.00 | 0.86 | 0.330 m | **never rises** — stays on the ground |
| `m17_seg1_mean` | 0.00 | 0.00 | 0.31 | 0.177 m | **never rises** — stays on the ground |
| `m55_seg0_mean` | 0.00 | 0.00 | 0.23 | 0.210 m | **never rises** — stays on the ground |
| `m16_seg7_replay` | 0.84 | 0.81 | 0.22 | 0.534 m | rises and stays up, but keeps stepping |
| `m55_seg0_replay` | 0.73 | 0.70 | 0.18 | 0.530 m | rises and stays up, but keeps stepping |
| `random_0` | 0.70 | 0.76 | 0.22 | 0.470 m | rises and stays up, but keeps stepping |
| `random_1` | 0.88 | 0.89 | 0.06 | 0.543 m | rises and stays up, but keeps stepping |
| `random_2` | 0.88 | 0.92 | 0.02 | 0.504 m | rises and stays up, but keeps stepping |
| `random_3` | 0.85 | 0.88 | 0.11 | 0.526 m | rises and stays up, but keeps stepping |

## 3. Constant z vs z-sequence replay

Time-indexed replay of a per-frame `z` sequence does work, but it is dominated on every axis.

Two things had to be right for replay to work at all:

1. **The replay window must be padded past the end of the get-up.** An unpadded segment replay ends
   exactly when the reference crosses 0.45 m and then holds that final mid-rise latent forever; the
   robot collapses back into a tangle (self-collision in 80% of frames, 0/16 success). Adding 2 s of
   padding so the schedule runs on into the reference's standing phase turns the same candidate into
   92% success. Every replay candidate reported here is padded.
2. **Which segment you pick matters a lot** — `m16_seg0`/`m17_seg1` replays reach 100%, while
   `m16_seg7` and `m55_seg0` replays reach 0%. There is no way to know which is which without
   running the harness.

Even at its best, replay is **4–9× slower** than a constant standing `z` (6.1–8.0 s vs 0.87 s to
stand) and much dirtier (self-collision in 42–80% of frames vs 1.5%). It is open-loop in time: the
schedule advances whether or not the robot is where the reference was, so any slip desynchronises
command from state, and the robot spends the rest of the schedule being told to do something that
no longer matches its situation. A constant `z` cannot desynchronise, because there is nothing to
synchronise — the closed loop runs entirely through the policy's own observations.

The two-phase schedules (get-up or replay `z`, then a standing `z`) also all reach 100% in-distribution,
but at 3.2–4.1 s: phase A only wastes time that the standing latent would have spent already standing.
**A schedule is not merely unnecessary here, it is strictly worse than its own phase B used alone.**

### 3.1 Aligned-replay control (replay's best case)

The episode is reset onto the *exact* reference frame at the start of `m16` get-up
segment 3, so a replay of that same segment's z starts perfectly in sync -- the
closed-loop analogue of the tracking eval. `aligned_settle0` additionally removes the
settle window, matching `tracking_inference`'s protocol (reset to frame, act immediately);
the `aligned_m16seg3` variant kept the harness's 1 s settle, which drives the joints to the
default pose *before* the schedule starts and so is already desynced at t=0.

Only `m16_seg3_replay` is genuinely aligned here; the other rows are controls run from the
same initial state (a different segment's schedule, and the constant standing latents).

**aligned_settle0 (no settle -- true aligned case)**

| candidate | success | time-to-stand | self-collision |
|---|---|---|---|
| `m16_seg3_replay` *(aligned)* | 91% | 4.40 s | 0.749 |
| `standing_pooled` | 100% | 0.51 s | 0.000 |
| `reward_move_ego_0_0` | 100% | 0.58 s | 0.000 |

**aligned_m16seg3 (1 s settle -- desynced at t=0)**

| candidate | success | time-to-stand | self-collision |
|---|---|---|---|
| `m16_seg3_replay` *(aligned)* | 94% | 4.38 s | 0.770 |
| `m16_seg0_replay` *(wrong-schedule control)* | 100% | 7.98 s | 0.383 |
| `standing_pooled` | 100% | 1.00 s | 0.015 |
| `reward_move_ego_0_0` | 100% | 1.17 s | 0.020 |

## 4. Robustness: what the DR column actually says


**confirm_indist_nominal** (n=128 per candidate)

| candidate | rose (got up) | strict stand-still | upright-hold | re-fell | time-to-rise |
|---|---|---|---|---|---|
| `standing_pooled` | 100% | 100% | 100% | 0% | 0.74 s |
| `reward_move_ego_0_0` | 100% | 100% | 100% | 0% | 0.87 s |
| `goal_fallAndGetUp3_stand` | 100% | 100% | 100% | 0% | 0.81 s |
| `goal_standing_canonical` | 100% | 100% | 100% | 0% | 0.80 s |
| `goal_sprint1_stand` | 100% | 100% | 100% | 0% | 0.75 s |
| `m16_seg0_replay` | 100% | 100% | 100% | 0% | 6.79 s |
| `tp_replay_then_stand_3s` | 100% | 100% | 100% | 0% | 3.13 s |
| `random_0` | 80% | 0% | 80% | 0% | 1.30 s |
| `m16_pooled_mean` | 0% | 0% | 0% | 0% | — s |

**confirm_indist_dr** (n=128 per candidate)

| candidate | rose (got up) | strict stand-still | upright-hold | re-fell | time-to-rise |
|---|---|---|---|---|---|
| `standing_pooled` | 100% | 31% | 100% | 12% | 0.92 s |
| `reward_move_ego_0_0` | 100% | 48% | 100% | 9% | 1.13 s |
| `goal_fallAndGetUp3_stand` | 100% | 64% | 100% | 5% | 1.08 s |
| `goal_standing_canonical` | 100% | 48% | 100% | 8% | 1.00 s |
| `goal_sprint1_stand` | 100% | 35% | 100% | 9% | 0.95 s |
| `m16_seg0_replay` | 99% | 19% | 98% | 9% | 6.79 s |
| `tp_replay_then_stand_3s` | 100% | 42% | 100% | 8% | 3.28 s |
| `random_0` | 85% | 0% | 80% | 0% | 2.16 s |
| `m16_pooled_mean` | 0% | 0% | 0% | 0% | — s |

**confirm_ood_dr** (n=128 per candidate)

| candidate | rose (got up) | strict stand-still | upright-hold | re-fell | time-to-rise |
|---|---|---|---|---|---|
| `standing_pooled` | 100% | 32% | 100% | 15% | 0.84 s |
| `reward_move_ego_0_0` | 100% | 45% | 100% | 5% | 1.01 s |
| `goal_fallAndGetUp3_stand` | 100% | 68% | 100% | 2% | 0.90 s |
| `goal_standing_canonical` | 100% | 51% | 100% | 3% | 0.92 s |
| `goal_sprint1_stand` | 100% | 34% | 100% | 12% | 0.94 s |
| `m16_seg0_replay` | 99% | 24% | 98% | 7% | 6.79 s |
| `tp_replay_then_stand_3s` | 100% | 36% | 100% | 10% | 3.30 s |
| `random_0` | 81% | 0% | 79% | 0% | 1.64 s |
| `m16_pooled_mean` | 0% | 0% | 0% | 0% | — s |

Read the DR columns carefully. Strict stand-still success falls to 31–64%, but that is **not** the
robot failing to get up, and **not** the robot falling over:

- `rose` (torso up and upright, sustained 0.5 s) is **100%** under every condition tested.
- `upright-hold` (upright for ≥95% of the final 2 s, ignoring foot contact) is **100%**.
- Checking the raw traces directly: **0 of 128 episodes per candidate ended below the height or
  tilt threshold**; the minimum final root height under full DR is 0.526 m against a 0.530 m
  standing pose, and minimum final uprightness is 0.978 (tilt < 12°).

The `re-fell` column is therefore misleadingly named: it flags episodes that were not in double
support *at the final instant*, which under a push every 1–3 s is a robot mid-stabilising-step, not
a robot on the floor.

### 4.1 DR ablation — which term costs the stand-still criterion

Same three candidates, in-distribution, 128 episodes each, each DR component isolated:

| condition | what is on | `standing_pooled` strict | `reward_move_ego_0_0` | `goal_fallAndGetUp3_stand` |
|---|---|---|---|---|
| `abl_dr_nopush` | friction + link mass + COM only | **100%** | **100%** | **100%** |
| `abl_noise_latency` | obs noise + action/obs latency only | 77% | 81% | 99% |
| `abl_dr_only` | physics DR **+ pushes** | 57% | 61% | 86% |
| `confirm_indist_dr` | everything together | 31% | 48% | 64% |

**Physics randomization costs nothing** — friction 0.5–1.25, link mass ±5% and COM ±2 cm leave the
skill at 100%. The cost comes from the repeated ±0.5 m/s pushes, and secondarily from observation
noise and latency (which training never included: MJLab's config path drops
`randomize_ctrl_delay`). `rose` and `upright-hold` are 100% in every row of this table.


## 5. Terminal stance vs the walk-policy hand-off pose

Mean leg pose over the final 2 s of standing episodes, against the pose the existing
IsaacLab walk policy settles into at zero velocity command (WS-D measurement).
Left-leg joints shown; right leg is the mirror. `—` = candidate never stood.

| candidate | hip pitch | hip roll | hip yaw | knee | ankle pitch | ankle roll | knee Δ | max abs Δ | RMS Δ |
|---|---|---|---|---|---|---|---|---|---|
| **walk-policy target** | -0.452 | +0.044 | +0.108 | +0.792 | -0.313 | -0.011 | — | — | — |
| `goal_standing_canonical` | -0.258 | -0.005 | +0.129 | +0.457 | -0.255 | +0.009 | -0.375 | 0.417 | 0.227 |
| `m16_seg0_replay` | -0.254 | +0.172 | +0.121 | +0.379 | -0.197 | -0.145 | -0.455 | 0.498 | 0.233 |
| `goal_sprint1_stand` | -0.196 | +0.073 | +0.427 | +0.383 | -0.228 | -0.028 | -0.425 | 0.443 | 0.235 |
| `tp_replay_then_stand_3s` | -0.158 | +0.165 | +0.163 | +0.312 | -0.168 | -0.125 | -0.472 | 0.480 | 0.242 |
| `standing_pooled` | -0.175 | +0.156 | +0.141 | +0.309 | -0.148 | -0.119 | -0.477 | 0.483 | 0.242 |
| `reward_move_ego_0_0` | -0.238 | +0.130 | +0.197 | +0.192 | +0.029 | -0.085 | -0.536 | 0.600 | 0.270 |
| `goal_fallAndGetUp3_stand` | -0.040 | +0.230 | +0.113 | +0.215 | -0.087 | -0.224 | -0.551 | 0.577 | 0.348 |

No candidate lands close to the walk-policy stance: every one holds a knee angle of 0.19–0.46 rad
against the 0.79 rad target, i.e. a **0.38–0.55 rad (22–32°) gap**, and all stand *taller* than the
walk policy does. The best of them (`goal_standing_canonical`, RMS 0.227 rad) is meaningfully closer
than the interim `getup` z WS-B measured (knee ≈ 0.15, Δ ≈ 0.64), but it is disqualified on
self-collision (§6). Among the deployable candidates `standing_pooled` (RMS 0.242) beats
`reward_move_ego_0_0` (RMS 0.270), so the hand-off criterion and the primary criteria agree here —
but neither removes the gap. **Closing it is a hand-off problem, not a z-selection problem**: no
latent in this sweep produces the walk policy's crouch, so expect to need either a blend window or a
walk policy tolerant of a taller initial stance.

## 6. Recommendation

**A single constant `z` is sufficient for get-up. No policy specialization is needed for this
skill.** The trained BFM already contains a get-up controller; the only thing that was missing was
knowing which 256-float vector selects it, and it is not the one you would guess.

Concretely:

- **Ship `getup` as a constant latent.** Saved in `runs/getup_eval/z_bank.pt` under the `getup`
  key, with runners-up and the full selection trace (`runs/getup_eval/z_selection.json`).
- **Do not ship a z schedule.** Two-phase and replay schedules were measured, not assumed, and they
  are *worse on every axis that matters*: slower to stand, dirtier (more self-collision), and they
  add an open-loop timing dependency that a constant latent simply does not have. The two-phase
  candidates are literally their own phase B with a delay in front.
- **Do not distill to a fixed-z specialized MLP.** Beyond discarding the shared-latent property that
  makes one actor serve every skill, the measurements give it no motivation: there is no performance
  gap for a specialist to close in-distribution.

### If more robustness is wanted later

The one number that is not ~100% is the *strict stand-still* rate under continuous domain
randomization. §4 shows why: the robot gets up and stays up (re-fall rates are low, torso height
and tilt are unchanged from nominal); what degrades is double-support occupancy, because a push
every 1–3 s makes it take a stabilizing step. If a future requirement is "stands perfectly still
while being shoved", the preferred order is:

1. **Nothing** — first check the requirement is real. A robot that steps to absorb a shove and then
   re-settles is behaving correctly; the strict criterion penalizes it anyway.
2. **Actor fine-tuning that keeps the `z` interface**, regularized toward the BFM (e.g. KL / behavior
   constraint to the frozen actor), so the shared latent space and every other skill survive.

Fine-tuning is the right lever precisely because the deficiency is in the actor's low-level
disturbance rejection, not in the latent: no choice of `z` fixes it, and a schedule cannot either.

### Caveats

- **Simulation only.** MuJoCo/MJLab, no hardware. Sim-to-real for get-up involves large contact
  forces and near-limit torques (§2.2 reports peak torque and effort-limit saturation for this
  reason), and those are exactly the regime where sim contact models are least trustworthy.
- **The standing criterion is a judgement call.** Thresholds are in §1.3 and in every summary JSON.
  The headline conclusion (standing latents work, get-up latents do not) is not
  threshold-sensitive — the failing candidates peak at 0.15–0.33 m against a 0.53 m standing pose —
  but the exact success percentages under DR are.
- **The DR column is the *training* randomization**, not a harder distribution, plus harness-side
  latency that training never saw. It is a regression guard, not a sim-to-real claim.
- **OOD coverage is limited** to tilt x yaw sweeps from a low drop with a neutral joint pose. It does
  not cover terrain, obstacles, or entanglement.

## 7. Reproducing this

```bash
# comparison sweep, one condition (repeat with --bucket ood and the DR flags for the other columns)
uv run python -m humanoidverse.tools.eval_getup \
  --model-folder runs/ufo_fb_k1_5090_v2 --candidates runs/getup_eval/candidates.json \
  --num-envs 64 --settle-steps 50 --episode-steps 500 --self-collision True \
  --bucket indist --disable-dr True --disable-obs-noise True \
  --out-dir runs/getup_eval --tag indist_nominal

# DR condition: training DR + observation noise + harness latency
#   ... --disable-dr False --disable-obs-noise False --action-latency-max 2 --obs-latency-max 1

# ablate a single DR term
#   ... --hydra-override domain_rand.push_robots=False

# export successful episodes as RobotState CSVs, with the reference-motion penetration baseline
uv run python -m humanoidverse.tools.eval_getup ... \
  --export-successful-motions runs/getup_eval/motions_getup --reference-penetration True

# a starting candidate file
uv run python -m humanoidverse.tools.eval_getup --emit-default-candidates my_candidates.json
```

`--num-envs 64` uses ~5.7 GB of VRAM and runs a 27-candidate condition in ~13 min on an RTX 3090.

### Artifacts

| path | contents |
|---|---|
| `runs/getup_eval/z_bank.pt` | skill library. `getup` -> `{z (256,) float32, mode, source, score, runners_up, decision_rule}`. Sequence and two-phase entries additionally carry `z_sequence` / `phase_a` / `phase_b` / `switch_s`, so a consumer can run them without the model or the motion library. |
| `runs/getup_eval/z_selection.json` | every candidate with the numbers the decision rule used, including the disqualification flag |
| `runs/getup_eval/episodes_<tag>.csv` | one row per episode, all metrics |
| `runs/getup_eval/summary_<tag>.json` | per-candidate aggregates + the full run configuration (criterion thresholds, pose bank, DR flags, self-collision body pairs) |
| `runs/getup_eval/z_candidates_<tag>.pt` | every candidate's materialized latent for that run |
| `runs/getup_eval/motions_getup/` | 64 successful get-up episodes as RobotState CSV + JSON sidecars |

### Self-generated motion data

`--export-successful-motions` writes each successful episode in the `robot_state_csv` format of
`configs/data/k1_lafan1.yaml`: `root_pos_{x,y,z}`, `root_rot_{x,y,z,w}` (xyzw), `dof_pos_0..21` in
`control_joints.names` order, plus a JSON sidecar carrying fps, frame count and the episode's full
metrics. Verified to load through `humanoidverse.utils.motion_data.adapters.load_robot_state_csv`
into a valid UFO motion dict, so it can feed the motion library / expert slicer directly — a
manifest entry only needs `fps: 50`.

**Emitted at 50 fps, the native policy rate, not resampled to LAFAN1's 30.** These are the exact
simulated states; resampling would only add interpolation error.

**These clips are dynamically consistent by construction, and measurably cleaner than the
retargeted prior.** Deepest interpenetration with the floor plane, measured identically on both
(max over all collision geoms over the whole clip):

| source | max ground penetration |
|---|---|
| retargeted LAFAN1 `fallAndGetUp2_subject2` (motion 16) | 24.84 cm |
| retargeted LAFAN1 `fallAndGetUp2_subject3` (motion 17) | 17.42 cm |
| retargeted LAFAN1 `pushAndFall1_subject4` (motion 55) | 28.33 cm |
| retargeted LAFAN1 `ground2_subject2` (motion 27) | 20.30 cm |
| **policy rollouts exported here** | **0.04 – 0.25 cm** |

That is a ~100x reduction. Note this is a broader statistic than the 4–11 cm *foot* penetration
recorded in `docs/vastai_k1.md` — it is the deepest contact of any collision geom, so the two
numbers are not directly comparable, but both are measuring the same underlying problem: the
retargeted reference is not physically realizable, and the policy's own trajectories are.

One caveat for downstream use: each exported clip is ~1 s of get-up followed by ~9 s of standing, so
roughly 90% static frames. Trim with the sidecar's `time_to_stand_s` before adding them to a
behaviour prior, or they will bias it toward standing still.
