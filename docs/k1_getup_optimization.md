# Optimizing the K1 get-up latent directly in z-space

Workstream E. Checkpoint under test: `runs/ufo_fb_k1_5090_v2/checkpoint` (FB-CPR / `FBcprAuxModel`,
`z_dim=256`, `norm_z=true`, K1 22-dof). Tool: `humanoidverse/tools/opt_getup_z.py`, which reuses
`humanoidverse/tools/eval_getup.py` (WS-A) for every simulation-side component. Raw artifacts:
`runs/getup_opt/`.

WS-A picked `standing_pooled` out of 27 hand-derived candidates; it saturates nominal success
(100% in-distribution and OOD). This workstream asks whether searching the latent sphere directly
beats it on the axes that are *not* saturated.

## 1. What is being optimized (and what is not)

Nominal success is not optimized — it is a **gate**, not an objective. The composite is

```
J = 100 * nominal_strict_success            # gate: must still get up and hold a stance
  + 100 * stress_DR_upright_stance          # robustness, on the fixed metric (see §2)
  +  25 * max(0, 1 - time_to_stand / 1.5s)  # speed
  +  50 * max(0, 1 - handoff_RMS / 0.35rad) # terminal-pose distance to the walk hand-off pose
  - 200 * max(0, self_collision - 0.05)     # cleanliness, with a hard DQ above 0.50
```

The weights are arbitrary. They were fixed **before** any search rollout was run — they are the
defaults of `ObjectiveWeights`, committed before the first search launched — and they were never
changed afterwards. One thing *was* changed mid-flight and should be read as such: the robustness
term's functional form. A first search was launched with the binary `success_upright` rate, and its
first two iterations showed the term steering the search on pure sampling noise (at iteration 1 the
top sample had a *worse* hand-off distance than the champion, 0.255 vs 0.241 rad, yet scored
+14.5, entirely from a 12-episode binary rate). That run was discarded and the search was restarted
using `held_upright` — the same criterion *before* it is thresholded at 0.95, i.e. the fraction of
the trailing window held upright — which carries far less variance in the same direction.
**Selection and every reported number still use the thresholded `success_upright`;** only the
in-loop search signal is the continuous form. Per-axis numbers are all reported separately, so the
ranking can be recomputed under a different weighting from `runs/getup_opt/final_results.json`
without re-simulating.

`handoff_RMS` is the RMS distance of the mean leg pose over the final 2 s of an episode to the pose
the IsaacLab walk policy settles into at zero velocity command (`docs/k1_loco_handoff_pose.md`):
`Hip_Pitch -0.452, Hip_Roll ±0.044, Hip_Yaw ±0.108, Knee_Pitch +0.792, Ankle_Pitch -0.313,
Ankle_Roll ±0.011`. `standing_pooled` sits at 0.242 rad with a knee 0.48 rad short of the target.

## 2. The DR metric: what was wrong, what was changed, and the ablation that justifies it

**WS-A's number.** `standing_pooled` scored 27–32% "success" under domain randomization. That
number comes from a criterion that requires *both feet loaded continuously* through the trailing
2 s window. Under the training DR — a ±0.5 m/s / ±0.5 rad/s push every 1–3 s — the robot takes a
protective step, loses double support for a few hundred milliseconds, and is scored as a failure
while standing perfectly upright. WS-A itself documented that 0 of 128 DR episodes ended down and
that the minimum final root height was 0.526 m against a 0.530 m standing pose.

**The change.** DR success is redefined as *rose to an upright torso and held it*: root height
≥ 0.45 m and uprightness ≥ 0.90 sustained for 0.5 s, then held for ≥ 95% of the final 2 s, with
**no foot-contact term at all**. Protective stepping is therefore free. This is exactly the
`success_upright` column that `eval_getup.py` already computes, so both the old (`success`) and the
new (`success_upright`) numbers are reported side by side in every table below and in every JSON.
No new criterion was invented, and nothing had to be re-simulated to audit the change.

Optimizing the old metric would have been actively harmful: the latent that maximizes
"never leaves double support while being shoved" is the latent that refuses to step, which is worse
on hardware than one that steps and re-settles.

**The consequence, measured.** The fixed metric is *saturated* at the training DR level. Probe,
9 candidates × 14 episodes, in-distribution, training pushes + observation noise + action/observation
latency (`runs/getup_opt/probe.json`, tier `dr_search`):

| candidate | old strict success | **new upright-stance success** |
|---|---|---|
| `standing_pooled` | 0.500 | **1.000** |
| `reward_move_ego_0_0` | 0.500 | **1.000** |
| `goal_standing_canonical` | 0.429 | **1.000** |
| `goal_sprint1_stand` | 0.571 | **1.000** |
| `goal_fallAndGetUp3_stand` | 0.643 | **1.000** |
| `handoff_pool_100` | 0.357 | **1.000** |
| `handoff_pool_500` | 0.500 | **1.000** |
| `handoff_pool_2000` | 0.429 | **1.000** |
| `standing_mean_all` | 0.429 | **1.000** |

Every candidate is at 1.000. This reproduces WS-A's own `upright-hold` column (100% everywhere) and
it means **there is no robustness headroom left to optimize at the training DR level** — the entire
27% figure was the mis-specified criterion, not a robustness deficit.

**So the search needs a harder tier to have any robustness signal at all.** The push magnitude is
tripled (`domain_rand.max_push_vel_xy=1.5`, `max_push_ang_vel=1.5`, i.e. ±1.5 m/s / ±1.5 rad/s every
1–3 s, plus observation noise and latency). That tier does discriminate (same probe, tier
`stress_search`):

| candidate | old strict success | **new upright-stance success** |
|---|---|---|
| `goal_fallAndGetUp3_stand` | 0.000 | **1.000** |
| `standing_pooled` | 0.071 | **0.857** |
| `reward_move_ego_0_0` | 0.071 | **0.857** |
| `goal_sprint1_stand` | 0.286 | **0.786** |
| `handoff_pool_100` | 0.071 | **0.786** |
| `handoff_pool_500` | 0.071 | **0.786** |
| `standing_mean_all` | 0.000 | **0.786** |
| `goal_standing_canonical` | 0.143 | **0.714** |
| `handoff_pool_2000` | 0.143 | **0.643** |

This stress tier is **out of the training distribution by construction** and is used only as a
*search signal* and as a reported column. It is not a sim-to-real claim, and the deployment gate is
the training-DR tier, where everything is at 1.000.

## 3. Method

### 3.1 Paired population rollouts

A whole population is evaluated in **one** batched rollout. `TiledPoseBank` samples `n_cond`
fallen initial states and tiles them across `n_pop` env blocks — each copy re-offset onto its own
env origin, because mjlab lays all envs out in one shared world on a 5 m grid — and `PopulationZ`
gives block `g` candidate `g`'s latent. Candidates therefore differ *only* in their latent: same
initial poses, same latency draws (`rollout(..., latency_tile=n_cond)`, a small optional extension
to `eval_getup.rollout`), same wall-clock physics. Rankings use common random numbers and every
comparison against the champion is paired.

Two residual asymmetries are handled explicitly: the *startup* physics draws (friction, link mass,
torso COM, default-pose offset) are fixed per env for the life of the process, so with one candidate
per env block they would bias each candidate by a constant — they are therefore **disabled in the
search tier** (`domain_rand.randomize_friction=False`, `...randomize_link_mass=False`,
`...randomize_base_com=False`, `...randomize_default_dof_pos=False`), which costs nothing because
WS-A's ablation measured physics-only DR at 100% strict success. And the candidate→block assignment
is rotated between batches. The full-fidelity final evaluation puts startup physics DR back on.

### 3.2 Search space

Full 256-D search is hopeless at this rollout budget, so the search runs in the span of

* the six strong seed latents (`standing_pooled`, `reward_move_ego_0_0`, the three `goal_*_stand`
  latents, `standing_mean_all`),
* three **crouch** latents pooled from the dataset (§3.3),
* the top 8 principal directions of the backward encoder's output over all upright, slow reference
  frames (253,609 frames, covariance accumulated in one pass over the 77-clip library).

**Provenance caveat on two seeds.** `goal_sprint1_stand` and `goal_fallAndGetUp3_stand` were read
from `runs/ufo_fb_k1_5090_v2/goal_inference/goal_reaching.pkl` at 23:55 on 2026-08-08. WS-F
regenerated that file about an hour later with a *different key set* — both of those keys are gone —
so those two basis vectors, and the probe rows that used them, are valid measurements but are no
longer reproducible from the current pkl. They are cached verbatim in
`runs/getup_opt/z_sources.pt`, which is what the search and the final evaluation actually read.
`opt_getup_z.py` now skips missing pkl keys with a warning instead of aborting. WS-F's
arm-clearance-filtered replacements (`runs/getup_eval/z_goal_clearance.pt`, already
`project_z`-normalized) landed after the basis was built, so they are *not* in the search subspace;
two of them are instead carried through the final evaluation as full candidates (§4).

SVD of that set gives an orthonormal basis of rank **k = 17**; every seed reconstructs through it to
cosine 1.000000. A search point is `z = project_z(B x)` with `x` a unit vector in R^17, so only the
direction matters and `‖z‖ = √256` holds by construction. Because `B` is orthonormal, a great-circle
interpolation in `x` is exactly a great-circle interpolation on the z-sphere.

Random z is not a useful starting point: WS-A measured `random_0..3` at 0/64 success each, so that
claim is taken from their table rather than re-spent as rollouts.

### 3.3 A crouch latent, pooled from the frames that already hold the hand-off pose

Whether the hand-off gap is closable in z-space at all is answered directly rather than left to
CEM. One pass over the motion library measures, for every frame, the RMS distance of its leg pose to
the walk hand-off target, keeping frames that are upright (root > 0.42 m) and slow (< 0.30 m/s):

| statistic over the 253,609 upright reference frames | leg-pose RMS to the hand-off target |
|---|---|
| minimum | **0.044 rad** |
| 1st percentile | 0.119 rad |
| 5th percentile | 0.157 rad |
| 25th percentile | 0.218 rad |
| median | 0.272 rad |

The pose the walk policy holds is therefore *well inside* the retargeted LAFAN1 distribution — the
closest frames sit at 0.044 rad, versus `standing_pooled`'s realized 0.242 rad. Pooling the
backward-encoder z over the 100 / 500 / 2000 closest frames gives `handoff_pool_100/500/2000`
(mean source-frame RMS 0.069 / 0.083 / 0.100 rad).

### 3.4 Warm start, then CEM

`standing_pooled` and `handoff_pool_500` are **1.32 rad apart** on the z-sphere — far outside any
sane CEM step — so iteration −1 sweeps the great circles connecting the champion (and
`standing_mean_all`) to the crouch latent at `t = 0.2 … 1.0`, which both maps the trade-off along the
arc and hands CEM a starting point. CEM then runs from the best arc point: population 12
(always including the champion as an in-iteration control and the current mean), elite 4,
per-coordinate std from the elites with smoothing 0.6 and a floor of 0.02, initial angular spread
0.35 rad, 20 iterations.

Because the champion is re-rolled inside every iteration under that iteration's own conditions,
every candidate carries a `ΔJ` against a *contemporaneous* control, which makes scores comparable
across iterations even though the pose bank is deliberately re-drawn each iteration (fresh
`numpy` generator per iteration; the bucket alternates in-distribution / OOD) to stop the search
from memorizing one bank.

Search fidelity: 12 candidates × 12 conditions = 144 envs, 6 s episodes (1 s settle + 300 policy
steps), one nominal rollout and one stress-DR rollout per iteration. Finalists are re-evaluated at
full fidelity: 10 s episodes, 63 episodes per candidate per condition.

### 3.5 Hold-out

The search only ever draws fallen poses from **even-numbered motions** and from an 8-fold OOD yaw
grid at a 0.30 m drop with an exact default joint pose. The held-out conditions are

* `indist_holdout` — the complementary half of the motion library (odd-numbered clips),
* `ood_holdout` — a 12-fold yaw grid, a 0.35 m drop, and ±0.15 rad uniform joint noise on the
  initial pose,

both with an unseen RNG seed. `standing_pooled` is run through every held-out condition in the same
batched rollouts as the finalists, so hold-out difficulty cancels in the comparison.

## 4. Search trace

### 4.1 Iteration −1: the great circle from the champion to the crouch latent

Twelve points on the arcs `standing_pooled → handoff_pool_500` and `standing_mean_all →
handoff_pool_500` (in-distribution bank, 12 episodes each, 6 s episodes). `t` is the fraction of the
1.32 rad arc travelled. `upright` is the stress-DR upright-stance rate.

| point | hand-off RMS | knee Δ | self-collision | time-to-stand | nominal success | stress upright |
|---|---|---|---|---|---|---|
| `standing_pooled` (t=0) | 0.242 | −0.476 | 0.025 | 0.83 s | 1.00 | 0.83 |
| champion arc, t = 0.20 | 0.199 | −0.414 | **0.022** | 0.84 s | 1.00 | 0.67 |
| champion arc, t = 0.35 | 0.172 | −0.373 | **0.025** | 0.86 s | 1.00 | 0.67 |
| champion arc, t = 0.50 | 0.150 | −0.334 | 0.806 | 0.83 s | 1.00 | 0.75 |
| champion arc, t = 0.65 | 0.139 | −0.313 | 0.842 | 0.84 s | 1.00 | 0.58 |
| champion arc, t = 0.80 | 0.110 | −0.251 | 0.892 | 0.81 s | 1.00 | 0.58 |
| champion arc, t = 1.00 (`handoff_pool_500`) | **0.070** | **−0.138** | 0.894 | 0.78 s | 1.00 | 0.50 |
| `standing_mean_all` arc, t = 0.35 | 0.154 | −0.333 | 0.217 | 0.96 s | 1.00 | 0.67 |
| `standing_mean_all` arc, t = 0.50 | 0.128 | −0.286 | 0.858 | 0.92 s | 1.00 | 0.67 |
| `standing_mean_all` arc, t = 0.65 | 0.108 | −0.247 | 0.881 | 0.88 s | 1.00 | 0.67 |
| `handoff_pool_100` | 0.067 | −0.133 | 0.881 | 0.80 s | 1.00 | 0.58 |
| `handoff_pool_2000` | 0.080 | −0.169 | 0.906 | 0.83 s | 1.00 | 0.58 |

This single sweep contains the workstream's central result. **The hand-off gap is closable in
z-space** — `handoff_pool_500` lands at 0.070 rad RMS with the knee 0.14 rad (not 0.48 rad) short of
the walk policy's crouch, and it still stands up in 0.80 s with 100% nominal success. But **it is
not free**: past `t ≈ 0.35` the self-collision fraction jumps from 0.02 to 0.79–0.90, and the
offending pairs are exactly the ones WS-A found on the goal latents —
`Right_Hip_Roll ↔ right_hand_link` (4988 frame-hits), `Left_Hip_Yaw ↔ left_hand_link` (4614),
`Left_Hip_Roll ↔ left_hand_link` (3730), `Right_Hip_Yaw ↔ right_hand_link` (3094). The deeper the
commanded crouch, the more the arms end up resting on the hips. The transition is a cliff, not a
ramp: on the champion arc self-collision is 0.022–0.025 up to t = 0.35 and 0.806 by t = 0.50.
(The tool originally truncated the arc endpoint names when labelling these points, which made the
two arcs indistinguishable; the rows above were recovered by matching each recorded coordinate
vector back to its exact slerp point, and the labelling is fixed.)

The useful region is therefore the *near* half of the arc, and that is where CEM was started.

### 4.2 CEM

Twelve iterations (warm start + 11), population 12, 12 episodes per candidate per rollout, one
nominal and one stress-DR rollout per iteration. The per-coordinate std hit its floor at iteration 4
and `J_best − J_champion` plateaued, so the run was stopped at iteration 10 of a configured 20.
`standing_pooled` is re-rolled inside every iteration, so each row's `ΔJ` is against a
contemporaneous control on that iteration's own pose bank.

| it | bucket | ΔJ (best − champion) | best: hand-off RMS | knee Δ | self-coll. | tts | stress held-upright | champion: hand-off RMS | champion held-upright |
|---|---|---|---|---|---|---|---|---|---|
| −1 | indist | +8.7 | 0.172 | −0.373 | 0.025 | 0.86 s | 0.897 | 0.242 | 0.905 |
| 0 | ood | +17.0 | 0.170 | −0.371 | 0.014 | 0.95 s | 0.938 | 0.242 | 0.869 |
| 1 | indist | +10.8 | 0.176 | −0.375 | 0.031 | 0.85 s | 0.948 | 0.241 | 0.933 |
| 2 | ood | +17.3 | 0.170 | −0.355 | 0.011 | 0.86 s | 0.995 | 0.240 | 0.929 |
| 3 | indist | +14.0 | 0.163 | −0.342 | 0.017 | 0.83 s | 0.958 | 0.240 | 0.928 |
| 4 | ood | +22.6 | 0.158 | −0.329 | 0.022 | 0.90 s | 0.989 | 0.240 | 0.877 |
| 5 | indist | +14.7 | **0.152** | **−0.314** | 0.014 | 0.80 s | 0.989 | 0.241 | 0.975 |
| 6 | ood | +14.5 | 0.168 | −0.355 | 0.011 | 0.89 s | 0.955 | 0.242 | 0.919 |
| 7 | indist | +24.7 | 0.162 | −0.335 | 0.014 | 0.85 s | 0.997 | 0.240 | 0.867 |
| 8 | ood | +18.0 | 0.156 | −0.330 | 0.019 | 0.89 s | 0.970 | 0.240 | 0.912 |
| 9 | indist | +8.5 | 0.174 | −0.360 | 0.022 | 0.89 s | 0.973 | 0.240 | 0.987 |
| 10 | ood | +17.8 | 0.168 | −0.352 | 0.011 | 0.85 s | 1.000 | 0.239 | 0.928 |

The champion's own hand-off RMS is reproduced as 0.239–0.242 rad in twelve independent
12-episode measurements across two different pose banks, which is a useful check that the metric
itself is stable at this sample size and that the ~0.07 rad improvement is not measurement drift.

CEM converges to hand-off RMS ≈ 0.15–0.17 rad at champion-level self-collision (0.011–0.031),
champion-level or better speed, and stress-DR upright-hold at least as good as the champion's. It
does **not** reach `handoff_pool_500`'s 0.070 rad, because the self-collision penalty (and the hard
disqualification above 0.50) blocks the far half of the arc — which is the honest answer to
"how much of the gap is closable *without* paying for it".

## 5. Final evaluation

Eight finalists, full fidelity: 10 s episodes (1 s settle + 500 policy steps), **48 episodes per
candidate per condition** (16 initial conditions × 3 batches, 128 envs), every candidate rolled out
from the *same* initial conditions in the same batch as `standing_pooled`. Startup physics
randomization is back on for the DR tiers. Twelve conditions = 3 DR tiers × 4 initial-condition
banks, two of which the search never saw.

Finalists: the champion; the converged CEM mean; the three best distinct CEM samples; the pure
crouch latent `handoff_pool_500`; and two of WS-F's arm-clearance-filtered goal latents, carried as
full candidates rather than seeds (they landed after the search basis was built).

### 5.1 Nominal (no DR, no observation noise, no latency)

| condition | candidate | success | time-to-stand | hand-off RMS | knee Δ | self-collision | min final root height |
|---|---|---|---|---|---|---|---|
| in-dist (search) | `standing_pooled` | 1.00 | 0.86 s | 0.242 | −0.477 | 0.012 | 0.528 m |
| | `cem_s7_2` | 1.00 | 0.86 s | **0.155** | **−0.327** | 0.010 | 0.522 m |
| | `cem_mean` | 1.00 | 0.85 s | 0.165 | −0.342 | 0.010 | 0.524 m |
| | `cem_s7_1` | 1.00 | 0.84 s | 0.164 | −0.339 | 0.009 | 0.523 m |
| | `cem_s4_5` | 1.00 | 0.87 s | 0.161 | −0.334 | 0.009 | 0.523 m |
| | `handoff_pool_500` | 1.00 | 0.84 s | **0.071** | **−0.147** | **0.932** | 0.512 m |
| | `wsf_obstacles4_subject2_135` | 1.00 | 0.99 s | 0.312 | −0.565 | 0.009 | 0.528 m |
| | `wsf_fallAndGetUp1_subject4_8325` | 1.00 | 0.99 s | 0.362 | −0.641 | 0.010 | 0.530 m |
| OOD (search) | `standing_pooled` | 1.00 | 0.86 s | 0.241 | −0.476 | 0.004 | 0.529 m |
| | `cem_s7_2` | 1.00 | 0.85 s | **0.154** | −0.326 | 0.010 | 0.523 m |
| | `cem_mean` | 1.00 | 0.83 s | 0.164 | −0.343 | 0.011 | 0.524 m |
| | `handoff_pool_500` | 1.00 | 0.81 s | 0.068 | −0.139 | 0.927 | 0.512 m |
| **in-dist (HELD OUT)** | `standing_pooled` | 1.00 | 0.89 s | 0.243 | −0.477 | 0.014 | — |
| | `cem_s7_2` | 1.00 | 0.90 s | **0.154** | −0.325 | 0.012 | — |
| | `cem_mean` | 1.00 | 0.88 s | 0.164 | −0.341 | 0.014 | — |
| | `handoff_pool_500` | 1.00 | 0.87 s | 0.071 | −0.145 | 0.931 | — |
| | `wsf_obstacles4_subject2_135` | 1.00 | 1.04 s | 0.309 | −0.562 | 0.011 | — |
| **OOD (HELD OUT)** | `standing_pooled` | 1.00 | 0.86 s | 0.242 | — | 0.006 | 0.528 m |
| | `cem_s7_2` | 1.00 | 0.86 s | **0.154** | — | 0.008 | 0.522 m |
| | `cem_mean` | 1.00 | 0.85 s | 0.165 | — | 0.009 | 0.524 m |
| | `handoff_pool_500` | 1.00 | 0.83 s | 0.069 | — | 0.931 | 0.512 m |

**Nominal success is 1.00 for every finalist in every bank, held out or not** — the gate is never
the binding constraint, which is why it was not the objective.

**The held-out numbers are indistinguishable from the search numbers.** `cem_s7_2` measures
0.155 / 0.154 / 0.154 / 0.154 rad across the four banks and `standing_pooled` 0.242 / 0.241 / 0.243 /
0.242. There is no hold-out penalty at all on this axis, and the reason is structural rather than
lucky: the hand-off distance is a property of the *stance the latent commands*, and the latent does
not depend on where the robot started. The full table is in `runs/getup_opt/final_results.json`.

The two WS-F latents are confirmed clean (self-collision 0.005–0.012 against the 0.926–0.929 their
pre-filter counterparts scored in WS-A, an unambiguous fix) but they stand *taller* than the
champion, not more crouched: hand-off RMS 0.309–0.366 rad, worse than `standing_pooled`'s 0.242,
and they are ~0.15 s slower to stand. On the hand-off and speed axes they are not contenders; their
case rests on DR robustness (§5.2).
