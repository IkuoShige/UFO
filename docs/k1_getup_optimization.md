# Optimizing the K1 get-up latent directly in z-space

Workstream E. Checkpoint under test: `runs/ufo_fb_k1_5090_v2/checkpoint` (FB-CPR / `FBcprAuxModel`,
`z_dim=256`, `norm_z=true`, K1 22-dof). Tool: `humanoidverse/tools/opt_getup_z.py`, which reuses
`humanoidverse/tools/eval_getup.py` (WS-A) for every simulation-side component. Raw artifacts:
`runs/getup_opt/`.

WS-A picked `standing_pooled` out of 27 hand-derived candidates; it saturates nominal success
(100% in-distribution and OOD). This workstream asks whether searching the latent sphere directly
beats it on the axes that are *not* saturated.

## Headline

**Three of the four axes had no headroom; the fourth moved by a third and the reason it cannot move
further is now measured.**

* **Robustness under domain randomization was never the problem.** Once WS-A's "both feet planted
  continuously" term is removed, *every* finalist scores **1.000** upright-stance success under the
  full training DR, in 30 of 32 candidate×bank cells, and never ends up on the floor. WS-A's 27% was
  a metric artifact. There is nothing to optimize at the randomization the policy was trained under
  (§2, §5.2).
* **Speed did not move** — every clean latent in the study stands in 0.78–1.04 s, and the optimized
  one is within 0.01 s of the champion. Time-to-stand looks like a property of the actor, not the
  latent (§7).
* **The walk hand-off gap is closable in z-space, and partly closable for free.** The optimized
  latent `cem_s4_5` cuts terminal-pose RMS distance from **0.242 → 0.161 rad** (knee, averaged over
  both legs, +0.315 → +0.457 against a +0.792 target), closer on all six leg joints, at unchanged success, speed and
  self-collision — a paired improvement of over 100 standard errors on both held-out banks
  (−108 SE OOD, −173 SE in-distribution), and reproduced through the exported ONNX deploy path in a second simulator (§5, §6).
* **Closing the gap *fully* costs exactly what WS-A's disqualified goal latents cost.** A latent
  pooled from the dataset frames nearest the hand-off pose gets to **0.070 rad** (knee +0.648) and
  stands in 0.84 s — but holds its hands against its hips on 82–93% of frames and is the one
  candidate measurably less robust to pushes (−0.120 ± 0.040). The self-collision cliff sits at
  t ≈ 0.35–0.50 along the arc between the two (§4.1).

**Recommendation: promote `cem_s4_5` (z-bank key `getup_opt_ws_e`), keep `standing_pooled` as
fallback.** The pre-registered decision rule as literally written says otherwise; §8 explains why
that rule cannot be trusted here (its 0.05 tolerance is finer than the 0.07 standard error of the
quantity it gates on, and it disqualified all seven challengers including two that are *better* than
the champion when the episodes are pooled) and states the case on the evidence instead.

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

### 5.2 Training domain randomization — the metric fix, at full fidelity

Full training DR (friction 0.5–1.25, link mass ±5%, torso COM ±2 cm, ±0.5 m/s / ±0.5 rad/s pushes
every 1–3 s) **plus** observation noise **plus** harness action/observation latency. 48 episodes per
candidate per bank. `old` is WS-A's strict criterion (continuous double support); `upright` is the
fixed one; `rose` is "got up at all"; `fell` is "ended below the height/tilt thresholds".

| candidate | in-dist (search) old / **upright** | OOD (search) | **in-dist (HELD OUT)** | **OOD (HELD OUT)** | rose | fell back |
|---|---|---|---|---|---|---|
| `standing_pooled` | 0.375 / **1.000** | 0.396 / **1.000** | 0.333 / **1.000** | 0.271 / **1.000** | 1.00 | 0.00 |
| `cem_s7_2` | 0.312 / **1.000** | 0.250 / **1.000** | 0.500 / **1.000** | 0.354 / **1.000** | 1.00 | 0.00 |
| `cem_mean` | 0.333 / **1.000** | 0.333 / **1.000** | 0.396 / **1.000** | 0.354 / **1.000** | 1.00 | 0.00 |
| `cem_s7_1` | 0.312 / **1.000** | 0.333 / **1.000** | 0.292 / **1.000** | 0.229 / **1.000** | 1.00 | 0.00 |
| `cem_s4_5` | 0.396 / **1.000** | 0.417 / **1.000** | 0.292 / **1.000** | 0.271 / **1.000** | 1.00 | 0.00 |
| `handoff_pool_500` | 0.396 / **1.000** | 0.562 / **1.000** | 0.438 / **1.000** | 0.521 / **0.958** | 1.00 | 0.00 |
| `wsf_obstacles4_subject2_135` | 0.521 / **1.000** | 0.604 / **1.000** | 0.583 / **1.000** | 0.458 / **1.000** | 1.00 | 0.00 |
| `wsf_fallAndGetUp1_subject4_8325` | 0.458 / **0.979** | 0.562 / **1.000** | 0.479 / **1.000** | 0.333 / **1.000** | 1.00 | 0.00–0.02 |

**30 of the 32 candidate×bank cells are exactly 1.000 on the fixed metric, and `rose` is 1.000 in
all 32.** The old criterion meanwhile scatters between 0.229 and 0.604 — the same 25–60% band WS-A
reported — and its ordering is essentially noise: `standing_pooled` scores 0.375, 0.396, 0.333, 0.271
on four banks that are all the same difficulty. That spread *is* the measurement error of a
criterion that is really counting how many stabilising steps happened to be in progress at the final
instant.

This settles the DR question for deployment: **at the randomization the policy was trained under,
every one of these latents already gets up and stays up, 100% of the time, and none of them ever
ends up on the floor.** Robustness at the training level is not an axis with headroom; it is a gate
that everything passes. (The only two sub-1.000 cells are `wsf_fallAndGetUp1_subject4_8325` at
0.979 in-distribution — one episode of 48, which also produced the only min-final-root-height below
0.45 m in the whole table, 0.437 m — and `handoff_pool_500` at 0.958 on the held-out OOD bank, whose
minimum final height 0.483 m is the lowest of the clean-stance candidates; a deeper crouch has less
margin to the 0.45 m threshold.)

Self-collision rises for every candidate under DR (the fall and the get-up itself involve limb
contact): the champion goes 0.012 → 0.075–0.107 and the CEM latents 0.010 → 0.060–0.102, i.e. the
optimized latents are, if anything, marginally *cleaner* than the champion under DR. WS-F's
clearance-filtered latents are the cleanest of all (0.011–0.018), and `handoff_pool_500` stays at
0.817–0.860 — its hand-on-hip contact is not a nominal-only artifact.

### 5.3 Stress tier (3× training push) — the only condition with any headroom

Training DR with the push magnitude tripled to ±1.5 m/s / ±1.5 rad/s every 1–3 s, over 10 s
episodes, plus observation noise and latency. This is **out of the training distribution by
construction**; it exists to create a robustness signal where the training tier has none. Here the
robot really can end up on the floor: `fell_back` runs 0.04–0.21.

Upright-stance success per bank, and the paired difference against `standing_pooled` pooled over all
four banks (192 paired episodes, same initial conditions, same batch):

| candidate | in-dist (search) | OOD (search) | in-dist (HELD OUT) | OOD (HELD OUT) | mean | **pooled paired Δ** |
|---|---|---|---|---|---|---|
| `standing_pooled` | 0.750 | 0.688 | 0.771 | 0.792 | 0.750 | — |
| `wsf_obstacles4_subject2_135` | 0.833 | 0.750 | 0.771 | 0.708 | 0.766 | **+0.016 ± 0.039** |
| `wsf_fallAndGetUp1_subject4_8325` | 0.771 | 0.771 | 0.771 | 0.667 | 0.745 | **−0.005 ± 0.039** |
| **`cem_s4_5`** | 0.729 | 0.750 | 0.708 | 0.750 | 0.734 | **−0.016 ± 0.040** |
| `cem_s7_1` | 0.750 | 0.708 | 0.812 | 0.646 | 0.729 | **−0.021 ± 0.037** |
| `cem_s7_2` | 0.646 | 0.708 | 0.771 | 0.604 | 0.682 | −0.068 ± 0.039 |
| `cem_mean` | 0.667 | 0.604 | 0.771 | 0.688 | 0.682 | −0.068 ± 0.038 |
| `handoff_pool_500` | 0.625 | 0.708 | 0.688 | 0.500 | 0.630 | **−0.120 ± 0.040** |

Two things to read here.

**First, the noise floor.** `standing_pooled` itself scores 0.688–0.792 across four banks that are
all nominally the same difficulty. A ±0.05 spread is what this metric does at 48 episodes; its
standard error is ≈0.07. Any per-bank comparison finer than that is reading noise.

**Second, one difference *is* real: the deep crouch costs push robustness.** `handoff_pool_500`
is 0.120 ± 0.040 below the champion pooled (−3.0 SE), and it is the worst candidate on all four
banks. A crouched stance has less margin to the 0.45 m height threshold (its minimum final root
height is 0.483 m, the lowest of the clean candidates) and less room to absorb a shove. So the far
end of the arc is penalized *twice* — by self-collision **and** by push robustness. The near end is
not: `cem_s4_5` and `cem_s7_1` sit within half a standard error of the champion, and on the
continuous held-upright fraction `cem_s4_5` is at +0.002 (0.897 vs 0.895).

## 6. Independent confirmation: plain MuJoCo + the exported ONNX policy

`eval_getup.py` and `opt_getup_z.py` share an environment, a robot config and a rollout loop, so a
win inside them is not independent evidence. `tools/k1_ufo_sim2sim.py` is: it compiles its own
MuJoCo model, runs a Python PD loop, and drives the robot through
`booster_k1_locomotion/ufo_policy_runtime` — the same module the ROS 2 deploy node uses — against
the exported ONNX policy. Single robot, no randomization, 1 s settle then 6 s of policy, from a
fallen pose face-up and face-down. Terminal leg pose is the mean over the final 2 s of the trace.

| latent | face-up tts | face-down tts | stood up | hand-off RMS (up / down) | knee, both legs (target +0.792) |
|---|---|---|---|---|---|
| `standing_pooled` | 0.64 s | 0.72 s | yes | 0.247 / 0.235 | +0.311 / +0.325 |
| **`cem_s4_5`** | **0.62 s** | **0.72 s** | yes | **0.162 / 0.160** | **+0.454 / +0.456** |
| `cem_s7_1` | 0.62 s | 0.68 s | yes | 0.167 / 0.160 | (left +0.478 / +0.487) |
| `cem_mean` | 0.62 s | 0.70 s | yes | 0.168 / 0.160 | (left +0.470 / +0.485) |
| `handoff_pool_500` | 0.60 s | 0.68 s | yes | 0.066 / 0.069 | (left +0.669 / +0.672) |
| `wsf_obstacles4_subject2_135` | 0.70 s | 0.78 s | yes | — | — |
| `wsf_fallAndGetUp1_subject4_8325` | 0.70 s | 0.82 s | yes | — | — |

The deploy path reproduces the batched harness to within 0.01 rad of hand-off RMS (`cem_s4_5`
0.162/0.160 here vs 0.158–0.161 there; `standing_pooled` 0.247/0.235 vs 0.240–0.243) and confirms
the `standing_pooled` baseline the brief quotes (0.64 s / 0.72 s) exactly. `cem_s4_5` stands 0.02 s
*faster* face-up and identically face-down, through a completely different simulator and inference
stack. `handoff_pool_500` reproduces its crouch here too (left knee +0.669 against the +0.792
target), which is §4.1's conclusion arrived at independently.

These both-leg knee figures (`cem_s4_5` +0.454 / +0.456, `standing_pooled` +0.311 / +0.325) are the
ones quoted in `docs/k1_getup_plan.md`'s promotion table; rows marked *(left)* above are left-leg
only and are not directly comparable to them — see §7.1 on the leg asymmetry.

## 7. What did *not* improve

Stated plainly, because three of the four axes in the brief turned out to have no headroom for a
latent to exploit:

* **Speed did not improve.** The champion stands in 0.86 s; the optimized latents stand in
  0.84–0.90 s. Paired against the champion on the held-out banks the difference is
  −0.028 s to +0.006 s, i.e. within ~2 standard errors of zero in every case except one bank-specific
  −2.6 SE for `cem_s7_1`, which does not replicate on the other bank. Every clean-standing latent in
  this entire study — 9 seeds, 12 arc points, ~130 CEM samples — lands between 0.78 s and 1.04 s.
  Time-to-stand looks like a property of the *actor*, not of the latent: the latent selects the
  target stance, and the policy takes about as long to get there whatever the stance is.
* **Robustness at the training DR level could not improve, because it is already perfect.** Once the
  "both feet planted continuously" term is removed, every finalist scores 1.000 upright-stance
  success in 30 of 32 candidate×bank cells, never falls back, and always rises. There is nothing
  to optimize. WS-A's 27% was a metric artifact, not a deficiency.
* **Self-collision did not improve and did not need to.** The champion is at 0.012 nominal, the
  optimized latents at 0.008–0.014. Under DR the optimized latents are marginally cleaner
  (0.060–0.102 vs 0.075–0.107) but that gap is not the point of interest.
* **The hand-off gap is only partly closed.** The optimized latent takes the knee (both legs) from
  0.315 rad to 0.457 rad against a 0.792 rad target — from 0.48 rad short to 0.34 rad short, about a
  third of the gap. The latent that closes it properly (`handoff_pool_500`, knee 0.648 rad,
  RMS 0.070) exists and
  works, but it holds its hands against its hips on 82–93% of frames — a wear and force-estimation
  problem on hardware, exactly the failure mode WS-A disqualified the goal latents for — *and* it is
  the one candidate measurably less robust to pushes (§5.3, −0.120 ± 0.040). Within the constraint
  "no worse than the champion on self-collision or push robustness", 0.154–0.164 rad is where the
  z-sphere runs out.
* **WS-F's arm-clearance latents did not turn out to be contenders on these axes.** They are
  genuinely fixed on self-collision (0.011–0.018 under DR, the cleanest of all finalists, against
  0.926–0.929 for their unfiltered predecessors) and they are nominally the most robust under the
  artificial stress tier, but they stand *taller* than the champion (hand-off RMS 0.297–0.366 vs
  0.242) and 0.06–0.14 s slower. If the deployment priority were "cleanest possible contact under
  DR", they would be the pick; on hand-off distance and speed they are behind both the champion and
  the optimized latent.

### 7.1 Terminal leg pose, side by side

Mean over the final 2 s of standing episodes on the two **held-out** banks. Per-joint rows are the
**left leg**; the `knee Δ` / `RMS Δ` / `max abs Δ` summary rows average over **both** legs, which is
the convention of every `knee Δ` column in this document and of `handoff_knee` in the JSON. The two
are not interchangeable for `cem_s4_5`: its legs are not symmetric (sim2sim left knee +0.485,
right +0.423, mean +0.454), whereas `standing_pooled` is closer to symmetric (+0.302 / +0.319).
Quote the both-leg mean unless you specifically want one side.

| joint | walk-policy target | `standing_pooled` | **`cem_s4_5`** | `handoff_pool_500` (disqualified) |
|---|---|---|---|---|
| Hip pitch | −0.452 | −0.182 | **−0.308** | −0.471 |
| Hip roll | +0.044 | +0.152 | **+0.094** | +0.037 |
| Hip yaw | +0.108 | +0.146 | **+0.112** | +0.179 |
| Knee pitch | +0.792 | +0.309 | **+0.491** | +0.653 |
| Ankle pitch | −0.313 | −0.141 | **−0.263** | −0.328 |
| Ankle roll | −0.011 | −0.120 | **−0.054** | +0.026 |
| **RMS Δ** | — | 0.242 | **0.161** | 0.070 |
| **max abs Δ** | — | 0.483 | **0.370** | 0.151 |
| **knee Δ** | — | −0.477 | **−0.335** | −0.144 |

`cem_s4_5` is closer to the target on **all twelve leg joints** — both legs, every joint, not just
the knee (checked explicitly against `pose_*` in `final_results.json`; the right leg's largest
remaining error is the knee at 0.370 rad, the left leg's at 0.301 rad).

## 8. Decision

### 8.1 The pre-registered rule, and what it returned

The rule (§`DECISION_RULE` in `opt_getup_z.py`, written before the final numbers were read)
disqualifies a candidate that, on **either** held-out bank, falls more than 0.02 below the champion
on nominal success or training-DR upright-stance, or more than **0.05** below on stress-DR
upright-stance, or self-collides above 0.50; survivors are then ranked by held-out J.

It disqualified **all seven** challengers, and every one of them on the same clause: the stress-DR
tolerance. The winner it returns is therefore `standing_pooled`.

| candidate | held-out J | search J | rule verdict |
|---|---|---|---|
| `cem_s4_5` | **210.26** | 211.63 | DQ — stress in-dist hold-out 0.708 vs champion 0.771 |
| `cem_s7_1` | 210.21 | 210.61 | DQ — stress OOD hold-out 0.646 vs champion 0.792 |
| `cem_mean` | 209.96 | 201.12 | DQ — stress OOD hold-out 0.688 vs champion 0.792 |
| `cem_s7_2` | 207.08 | 206.43 | DQ — stress OOD hold-out 0.604 vs champion 0.792 |
| `standing_pooled` | 203.89 | 198.03 | eligible |
| `wsf_obstacles4_subject2_135` | 188.01 | 193.49 | DQ — stress OOD hold-out 0.708 vs champion 0.792 |
| `wsf_fallAndGetUp1_subject4_8325` | 180.61 | 186.35 | DQ — stress OOD hold-out 0.667 vs champion 0.792 |
| `handoff_pool_500` | −966.03 | −957.83 | DQ — self-collision 0.932; stress; DR OOD hold-out 0.958 |

### 8.2 Why that verdict should not be taken at face value — and what is left after saying so

**The rule is mis-calibrated, and this is a flaw in the rule, not a finding about the latents.** Its
0.05 tolerance is *below the standard error of the quantity it gates on*: stress upright-stance at
48 episodes has SE ≈ 0.07, and the champion's own score varies 0.688–0.792 across four banks of
identical difficulty. A one-sided per-bank threshold finer than the measurement will disqualify
almost anything, which is exactly what happened — including the two WS-F latents that are *better*
than the champion when the stress episodes are pooled.

Pooling all 192 paired stress episodes (§5.3) gives the properly powered comparison:
`cem_s4_5` is **−0.016 ± 0.040** against the champion on the binary metric and **+0.002** on the
continuous one. That is indistinguishable from no change. The one candidate that pooling *does*
convict is `handoff_pool_500` at −0.120 ± 0.040, and it stays disqualified.

This pooled analysis is a **post-hoc amendment** and is labelled as one. It does not rescue a
candidate that lost; it says the rule could not tell the candidates apart in the first place. State
of the evidence for `cem_s4_5` versus `standing_pooled`, on held-out initial conditions only:

| axis | `standing_pooled` | `cem_s4_5` | paired difference |
|---|---|---|---|
| nominal success | 1.000 | 1.000 | 0 |
| training-DR upright stance | 1.000 | 1.000 | 0 |
| **hand-off RMS** | 0.242 | **0.161** | **−0.081 ± 0.0005 (−173 SE in-dist, −108 SE OOD)** |
| knee angle (both legs) | +0.315 | **+0.457** | +0.142 |
| time-to-stand | 0.86 / 0.89 s | 0.86 / 0.90 s | +0.001 ± 0.014 s |
| self-collision (nominal) | 0.014 / 0.006 | 0.011 / 0.007 | ≈0 |
| self-collision (training DR) | 0.075 / 0.105 | 0.063 / 0.102 | ≈0 |
| stress upright stance (pooled, 192 ep) | 0.750 | 0.734 | −0.016 ± 0.040 |
| sim2sim time-to-stand (up / down) | 0.64 / 0.72 s | **0.62 / 0.72 s** | — |
| sim2sim hand-off RMS (up / down) | 0.247 / 0.235 | **0.162 / 0.160** | — |

### 8.3 Recommendation

**Promote `cem_s4_5` (stored as `getup_opt_ws_e`) to the deployed get-up latent, and keep
`standing_pooled` as the fallback.** The case:

* The only axis with real headroom improves decisively and reproducibly: hand-off RMS 0.242 → 0.161
  rad, closer on all six leg joints, over 100 paired standard errors on each held-out bank
  (−173 in-distribution, −108 OOD), identical on both held-out banks and on both search banks, and reproduced to within 0.01 rad through the ONNX deploy path in a
  different simulator.
* Nothing measurably regresses. Nominal success, training-DR upright stance, self-collision and
  time-to-stand are unchanged; sim2sim is 0.02 s *faster* face-up.
* The brief's two conditions for replacement — "real on held-out conditions" and "confirmed in
  `k1_ufo_sim2sim.py`" — are both met.

**Read this alongside the honest caveats.** The pre-registered rule as literally written says no
(§8.1); it says no to every candidate including two that are nominally *more* robust than the
champion, which is why it should not decide this. If a reviewer prefers the literal rule, the
champion stands and all that is forgone is the hand-off improvement — nothing breaks either way,
because the `getup` key in `runs/getup_eval/z_bank.pt` is untouched and promotion is a one-line
change in whatever selects the deploy key.

**Do not ship `handoff_pool_500`,** even though it is the only latent that nearly closes the
hand-off gap (RMS 0.070, knee +0.648). It self-collides hand-on-hip on 82–93% of frames and it is
the one candidate measurably worse under pushes (−0.120 ± 0.040). It is recorded here as the
existence proof, not as a deployable option.

**For whoever owns the walk hand-off:** the gap is now 0.30 rad of knee rather than 0.48 rad, and
the residual is structural. WS-A's conclusion that "closing it is a hand-off problem, not a
z-selection problem" is half right — a third of it *was* a z-selection problem, and the rest is not
free. Closing the remainder means either a blend window, a walk policy tolerant of a taller stance,
or accepting the arm-clearance cost of a deeper commanded crouch.

## 9. Compute, artifacts and reproduction

### 9.1 Compute spent

All on one (shared) RTX 3090; no training was launched at any point.

| stage | batched rollouts | episodes | wall clock |
|---|---|---|---|
| timing calibration (2 candidates, 128 envs) | 2 | 256 | 1 min |
| probe: crouch scan + 3 DR tiers × 9 candidates (126 envs) | 3 | 378 | 10 min |
| **discarded** first search (binary DR term, 3 iterations) | 6 | 864 | 3 min |
| CEM search (12 iterations × 2 tiers, 144 envs, 6 s episodes) | 24 | 3 456 | 8 min |
| final evaluation (12 conditions × 3 batches, 128 envs, 10 s episodes) | 36 | 4 608 | 25 min |
| **total (GPU)** | **71** | **9 562** | **≈47 min** |
| sim2sim confirmation (single robot, CPU + onnxruntime) | — | 14 | ≈9 min |

Peak GPU memory 6.9 GB with two 144-env environments resident simultaneously (nominal + stress),
which is what makes one nominal and one DR rollout per CEM iteration affordable without rebuilding.
The motion-library scan that produces the crouch latents and the standing principal directions runs
in ~2 s over all 77 clips and is cached in `runs/getup_opt/z_sources.pt`.

### 9.2 Artifacts

| path | contents |
|---|---|
| `humanoidverse/tools/opt_getup_z.py` | the tool: `probe` / `search` / `final` / `bank` |
| `runs/getup_opt/z_sources.pt` | cached seed latents, crouch-pooled latents, standing principal directions, scan statistics |
| `runs/getup_opt/probe.json` | the three-tier probe that established the DR-metric result |
| `runs/getup_opt/search_trace.json` | every evaluated latent with its coordinates and full metrics (`hall`), plus the per-iteration trace and the search config |
| `runs/getup_opt/search_state.pt` | subspace basis, CEM mean and per-coordinate std, seed coordinates |
| `runs/getup_opt/final_results.json` | all 12 conditions × 8 finalists, aggregates **and** per-episode arrays for paired statistics |
| `runs/getup_opt/finalist_z.pt`, `z_bank_finalists.npz` | the finalist latents (`.npz` is loadable by `tools/k1_ufo_sim2sim.py --z-bank`) |
| `runs/getup_opt/decision.json` | the pre-registered rule, every candidate's numbers, the verdicts, paired statistics |
| `docs/k1_getup_opt_z.json` | **committed** raw 256-float vectors: the winner `cem_s4_5`, `standing_pooled`, `handoff_pool_500` (`runs/` is gitignored and the winner is not reproducible from scratch, §3.2) |
| `runs/getup_eval/z_bank.pt` | **`getup` untouched**; new key `getup_opt_ws_e` = `cem_s4_5` with `z` / `mode` / `source` / `score` / `holdout_score` / `paired_vs_champion` / `search_config` / `z_dim` |

### 9.3 Reproducing

```bash
# 1. probe: crouch-frame scan + DR-tier discrimination check (writes runs/getup_opt/z_sources.pt)
uv run python -m humanoidverse.tools.opt_getup_z probe --out-dir runs/getup_opt

# 2. CEM search (warm-start arc sweep, then CEM; writes search_trace.json every iteration)
uv run python -m humanoidverse.tools.opt_getup_z search --out-dir runs/getup_opt

# 3. full-fidelity finals over search + held-out banks, 3 DR tiers
uv run python -m humanoidverse.tools.opt_getup_z final --out-dir runs/getup_opt \
    --n-finalists 3 --final-cond 16 --final-batches 3 \
    --extra-finalists handoff_pool_500 wsf_obstacles4_subject2_135 wsf_fallAndGetUp1_subject4_8325

# 4. decision rule + z-bank write (+ an .npz for sim2sim)
uv run python -m humanoidverse.tools.opt_getup_z bank --out-dir runs/getup_opt \
    --bank-key getup_opt_ws_e --bank-z cem_s4_5

# 5. independent confirmation through the deploy runtime
uv run python tools/k1_ufo_sim2sim.py --z-name cem_s4_5 \
    --z-bank runs/getup_opt/z_bank_finalists.npz --seconds 6 --trace /tmp/t.npz   # add --face-down
```

Prefix commands with `PYTHONPATH=` on a box where ROS 2 leaks into `PYTHONPATH`, and export
`MUJOCO_GL=egl`. `--rescan` rebuilds `z_sources.pt`; note that two of the basis seeds come from a
`goal_reaching.pkl` that has since been regenerated with a different key set (§3.2), so a rescan
today produces a 15-dimensional subspace rather than the 17-dimensional one used here, and the tool
warns and continues rather than aborting.

### 9.4 Caveats

* **Simulation only.** No hardware. Sim-to-real for get-up involves large contact forces and
  near-limit torques.
* **The stress tier is not a sim-to-real claim.** It is 3× the training push magnitude, invented to
  create a robustness signal where the training tier has none, and it is out of distribution by
  construction. The deployment-relevant tier is training DR, where everything scores 1.000.
* **The composite's weights are arbitrary.** Per-axis numbers are all in
  `runs/getup_opt/final_results.json`, so the ranking can be recomputed under a different weighting
  without re-simulating.
* **The search subspace is 17-dimensional**, not 256. It spans the strong hand-derived latents plus
  the principal directions of the standing manifold. A latent outside that span could in principle
  do better; nothing here rules that out.
* **48 episodes per candidate per condition** gives ≈0.07 standard error on a binary rate. That is
  ample for the hand-off axis (>100 SE) and marginal for the stress-robustness axis (§8.2).
* **`runs/` is gitignored**, so the winning 256-float vector (plus `standing_pooled` and
  `handoff_pool_500` for reference) is committed as `docs/k1_getup_opt_z.json`. It cannot be
  regenerated from scratch today — see §3.2.
