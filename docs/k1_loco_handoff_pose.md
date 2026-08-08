# K1 walk-policy hand-off pose — measured

**Question.** A get-up policy must end in the posture the walk policy expects. The
walk policy's declared default pose is `{Hip_Pitch −0.26, Hip_Roll 0, Hip_Yaw 0,
Knee_Pitch 0.52, Ankle_Pitch −0.26, Ankle_Roll 0}` per leg. Is that the pose the
policy actually *holds* at zero velocity command, or does it converge somewhere else?

**Answer.** Somewhere else, by a lot. The deployed walk policy converges to a
noticeably deeper squat: **max |Δ| = 0.284 rad (16.3°) at the knee, RMS Δ = 0.147 rad
(8.4°)** across the 12 legs. The declared default pose is **not** a usable hand-off
target on its own. Details in §4, recommendation in §6.

Measured with `tools/k1_loco_zero_cmd_pose.py`. Every number below is reproducible
from that script; every claim about a convention is traced to a file:line.

---

## 1. What was measured, and why it could be measured at all

The trained checkpoints are not available — `IsaacLab-K1-Locomotion/logs/` is
gitignored and absent. What exists is the exported ONNX in
`booster_k1_locomotion/assets/`. Three of those are the 49→12 walk lineage:

| file | role |
| --- | --- |
| `policy_180843_19999.onnx` | **the currently deployed walk policy** — the default `model_path` in every newer launch file (`launch/k1_isaaclab_deploy.launch.py`, `k1_isaaclab_sim.launch.py`, `k1_sim_webots.launch.py`, the soccer launches). Primary target here. |
| `policy_isaaclab_walk.onnx` | same lineage, secondary |
| `policy_205319_19999.onnx` | same lineage, secondary |
| `model_24999.onnx` | `actor_obs[1,79] → action[1,22]`, a different older 22-DoF policy. Not in scope. |

The kill-switch for this whole exercise would have been a missing observation
normalizer. It is **not** missing: each of the three graphs begins with

```
Sub(obs, normalizer._mean) -> Div(_, normalizer._std) -> Gemm ... Elu ... Gemm
```

i.e. rsl_rl's `EmpiricalNormalization` is baked into the graph
(`agents/history_policy_exporter.py:25-27` and `scripts/rsl_rl/play_onnx.py:165`
both state this contract). Raw observations go straight in. Layer sizes are
49→256→128→128→12 with ELU.

## 2. Observation contract (49 dims, single step, no history)

Established three independent ways — the env cfg, `history_layout.POLICY_TERM_SPECS`,
and the deployed ROS2 node — and all three agree:

| slice | term | units / convention |
| --- | --- | --- |
| `[0:3]` | `base_ang_vel` | body-frame gyro, rad/s, **unscaled** |
| `[3:6]` | `projected_gravity` | `R^T @ (0,0,−1)`, unit vector |
| `[6:9]` | `velocity_commands` | `(vx, vy, ωz)` — held at `(0,0,0)` |
| `[9:21]` | `joint_pos_rel` | `q − q_default`, 12 legs, `JOINT_NAMES_K1` order |
| `[21:33]` | `joint_vel_rel` | `qd` (default joint vel is 0) |
| `[33:45]` | `actions` | previous **raw** policy output, pre-scale |
| `[45:49]` | `gait_phase` | `[sin φ_L, cos φ_L, sin φ_R, cos φ_R]`, **identically zero at rest** |

Sources:

- `.../locomotion/velocity_env_cfg.py:34-42` — `JOINT_NAMES_K1`: 12 leg joints only,
  `Left_{Hip_Pitch,Hip_Roll,Hip_Yaw,Knee_Pitch,Ankle_Pitch,Ankle_Roll}` then the same
  for `Right_`. Head and arms are commented out; the policy does not actuate them.
- `velocity_env_cfg.py:131` — `JointPositionActionCfg(joint_names=JOINT_NAMES_K1,
  preserve_order=True, scale=0.5, use_default_offset=True)`, so
  **`q_target = q_default + 0.5 · action`**, no action clipping.
- `velocity_env_cfg.py:339-342` — `decimation = 4`, `sim.dt = 0.005` → **50 Hz policy,
  200 Hz physics**, matching `k1_constants_isaaclab.py:52-54`.
- `rough_env_cfg.py:200-218` — `K1PolicyCfg`, term order exactly as tabulated;
  `joint_pos`/`joint_vel` use `mdp.joint_pos_rel`/`joint_vel_rel` with
  `preserve_order=True`.
- `history_layout.py:40-48` — `POLICY_TERM_SPECS` documents the identical order.
- `booster_k1_locomotion/booster_k1_locomotion/rl_policy_isaaclab_node.py:230-266` —
  the deployed Python builder, identical layout, `POLICY_OBS_DIM = 49`.

**The single most important detail — `gait_phase` at zero command.**
`mdp/observations.py:41-52`: the phase is computed, then

```python
cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
phase = torch.where(cmd_speed < cmd_threshold, torch.zeros_like(phase), phase)
```

with `cmd_threshold = _COMMAND_THRESHOLD = 0.05` (`rough_env_cfg.py:59`, wired in at
`rough_env_cfg.py:214`). So at zero command the four phase channels are **exactly
zero**, not frozen at some arbitrary phase — which is precisely why a zero-command
steady state is well defined. Both deploy implementations reproduce this: the Python
node zeros immediately (`rl_policy_isaaclab_node.py:243-245`), the C++ node
(`origin/feat/dual_walk:src/rl_policy_isaaclab_node.cpp:1093-1142`) winds the gait
down over one more cycle and then freezes at phase 0 — different transient, same
steady state. Gait *frequency* is therefore irrelevant to this measurement.

**Independent confirmation from the baked normalizer.** The `normalizer._mean` /
`_std` vectors are the empirical observation statistics over training, and they match
the contract term by term: `projected_gravity` mean `(−0.016, −0.001, −0.997)` (a unit
gravity vector, so index 3:6 is right and unscaled); `base_ang_vel` std 0.48–0.64
rad/s and `joint_vel` std 1.6–3.2 rad/s (raw SI, no legged-gym-style scaling);
`gait_phase` std 0.710 ≈ `1/√2` (sin/cos of a near-uniform phase, unscaled). And the
`actions` mean is almost exactly twice the `joint_pos` mean (e.g. left knee `+0.713`
vs `+0.381`), independently confirming `scale = 0.5` with `use_default_offset`.

**No observation noise** was applied (the cfg's `Unoise` terms are training-time
domain randomisation; `K1RoughEnvCfg_PLAY` disables them at
`rough_env_cfg.py:499`, `K1FlatEnvCfg_PLAY` at `flat_env_cfg.py:631`). The measurement is deterministic.

## 3. Rollout harness

Plain MuJoCo, structured after `tools/k1_ufo_sim2sim.py`, but with this policy's
own (completely different) observation layout.

**Robot model — the arms question is settled by the URDF, not by IsaacLab defaults.**
Training loads `assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf`
(`rough_env_cfg.py:73-76`) with `merge_fixed_joints=True`
(`rough_env_cfg.py:121-124`). That URDF is byte-identical to `K1_22dof.urdf` except
that the ten head/arm joints are `type="fixed"` — all with `rpy="0 0 0"` origins, i.e.
**welded at joint angle zero** — and both files carry the same 19.666 kg over the same
23 links. So the training articulation is 12-DoF, legs only, with the upper body a
rigid extension of the trunk at zero angles. There are no "unmatched joints" for
IsaacLab to zero-initialise; the question does not arise. The harness reproduces this
by deleting those ten joints from the MuJoCo spec (`--upper-body weld`, the default);
`--upper-body pd-deploy` instead PD-holds them at the deploy `DEFAULT_ANGLES` for
comparison (§5).

**Actuators.** IsaacLab `DelayedPDActuator` is an ideal PD with the position target
pushed through a 2–7 physics-step delay buffer:
`τ = clip(kp·(q_des − q) − kd·qd, ±effort_limit)`. Values from
`rough_env_cfg.py:107-118` (`legs`: kp 160, kd 4.0, effort 68/76/38.3/112) and
`rough_env_cfg.py:170-181` (`feet`: kp 50, kd 2.5, effort 38.3). Leg `armature` is
overridden onto the MJCF to the actuator-cfg values; the MJCF already agrees except
on the ankles (0.0565 vs 0.0282528). PD runs every physics step at 200 Hz, the policy
at 50 Hz.

**Initial condition — and a real finding.** The robot is placed upright at the config
default pose with the feet 2 mm above the plane (computed from the foot collision-box
geometry), and the policy is engaged immediately. This matters, because:

> **The config default pose is open-loop unstable.** Held by joint PD alone at the
> declared default target, the robot topples *forward* in about 2 s — at any of the
> three gain sets. The CoM starts 9.1 cm inside the front of the support polygon
> (CoM x = +0.0098 m, support x ∈ [−0.0795, +0.1005]), but finite joint stiffness lets
> gravity sag the hips (+0.021 rad) and knees (−0.016 rad), which walks the CoM
> forward; it crosses the toe at ≈0.95 s and the fall is then committed
> (`−g_z`: 1.000 → 0.968 at 1.0 s → 0.823 at 1.5 s → 0 at 2.0 s).
>
> This is not a quirk of the model — it is what the gains predict. The trunk over the
> ankles is an inverted pendulum with destabilising stiffness `m·g·h ≈ 19.67 × 9.81 ×
> 0.5 ≈ 96.5 N·m/rad`, against a total ankle-pitch stiffness of `2 × kp_ankle`. The
> walk gains give 2 × 50 = 100 N·m/rad — marginal even before the hip and knee
> compliance is counted, hence the slow topple. **The get-up deploy gains are worse:**
> `k1_isaaclab_getup_gains.yaml` (on `origin/feat/dual_walk`) is the walk gains × 0.6,
> so 2 × 30 = 60 N·m/rad, comfortably below 96.5. Measured time to `−g_z < 0.7`:
> 1.62 s at walk gains (160/50), 1.50 s at deploy gains (200/50), **1.19 s at the
> get-up gains (96/30)**.

Consequently the pre-engage settle is *not* neutral, and `settle_seconds` defaults
to 0. Measured dependence for the primary policy: settle 0–0.5 s moves the converged
pose by <0.023 rad; 1.0 s moves it 0.104 rad and doubles the left/right asymmetry;
at ≥1.5 s the robot is already down and the policy cannot recover. All headline
numbers use settle 0.

**Validation — the harness actually walks.** With the same code path and a nonzero
command (which turns on the phase accumulator, `mdp/events.py:84-143`), the policies
track the command they are given:

| commanded | `policy_180843` | `policy_isaaclab_walk` | `policy_205319` |
| --- | --- | --- | --- |
| vx 0.3 | 0.310 | 0.243 | 0.238 |
| vx 0.6 | 0.634 | 0.509 | 0.552 |
| vx 1.0 | 1.042 | 0.830 | 0.998 |
| ωz 0.8 | 0.718 | 0.892 | 0.890 |

A wrong observation layout, joint order, action scale, or gain set would not produce
accurate velocity tracking at three speeds across three independently trained
policies. This is the strongest available evidence that the contract in §2 is right.

## 4. Result — the converged zero-command pose

All three policies **stay standing**: they settle to a static pose, not a step-in-place
limit cycle and not a drift. Over the final 2 s: leg velocity RMS 0.0002–0.0023 rad/s,
per-joint peak-to-peak ≤0.002 rad, both feet in continuous contact, trunk upright
(`−g_z` ≥ 0.9996), base xy speed ≤0.0002 m/s. No torque saturation (peak 0.55–0.66× of
even the tighter MJCF limits).

**Primary policy, `policy_180843_19999.onnx`** — the deployed one:

| joint | config default | measured | Δ | Δ (deg) |
| --- | --- | --- | --- | --- |
| Left_Hip_Pitch | −0.2600 | −0.4567 | **−0.1967** | −11.27 |
| Left_Hip_Roll | 0.0000 | +0.0750 | +0.0750 | +4.30 |
| Left_Hip_Yaw | 0.0000 | +0.1426 | +0.1426 | +8.17 |
| Left_Knee_Pitch | +0.5200 | +0.8039 | **+0.2839** | +16.27 |
| Left_Ankle_Pitch | −0.2600 | −0.3183 | −0.0583 | −3.34 |
| Left_Ankle_Roll | 0.0000 | +0.0096 | +0.0096 | +0.55 |
| Right_Hip_Pitch | −0.2600 | −0.4473 | **−0.1873** | −10.73 |
| Right_Hip_Roll | 0.0000 | −0.0131 | −0.0131 | −0.75 |
| Right_Hip_Yaw | 0.0000 | −0.0724 | −0.0724 | −4.15 |
| Right_Knee_Pitch | +0.5200 | +0.7802 | **+0.2602** | +14.91 |
| Right_Ankle_Pitch | −0.2600 | −0.3074 | −0.0474 | −2.72 |
| Right_Ankle_Roll | 0.0000 | −0.0119 | −0.0119 | −0.68 |

**Aggregate: max |Δ| = 0.2839 rad (16.26°), RMS Δ = 0.1470 rad (8.42°).**
Trunk height 0.5042 m, vs 0.5246 m for the exact default pose (2.0 cm lower).

**All three policies:**

| policy | status | max \|Δ\| | RMS Δ | knee Δ (L/R) | base height |
| --- | --- | --- | --- | --- | --- |
| `policy_180843_19999` (deployed) | settled | 0.2839 | 0.1470 | +0.284 / +0.260 | 0.5042 m |
| `policy_isaaclab_walk` | settled | 0.4353 | 0.2209 | +0.435 / +0.425 | 0.4885 m |
| `policy_205319_19999` | settled | 0.1566 | 0.0907 | +0.150 / +0.157 | 0.5131 m |

The three disagree with each other by more than a factor of two in magnitude, but
they agree unanimously on the **shape** of the disagreement with the config: more knee
flexion, more hip-pitch flexion, slightly more ankle-pitch flexion — a deeper squat,
1.2–3.6 cm lower at the trunk. The direction and scale also match the baked
normalizer's training-average `joint_pos_rel` (left hip pitch −0.166, knee +0.381 for
the primary policy), which is an entirely independent source.

**Fixed-point uniqueness.** Repeating with the initial leg pose jittered by
±0.01/0.02/0.03/0.05 rad, 10 seeds each: **41/41 runs settle for every policy**, and
the converged pose is reproducible to a per-joint sd ≤ 0.019 rad. Per-run RMS Δ for
the primary policy spans only 0.1453–0.1571 rad. The zero-command fixed point is
unique and well conditioned.

## 5. Sensitivity — what does *not* change the answer

Primary policy, one variable at a time; "vs baseline" is the largest per-joint change:

| variant | status | max \|Δ\| | RMS Δ | vs baseline |
| --- | --- | --- | --- | --- |
| baseline (kp 160/50, weld, delay 0, dt 0.005) | settled | 0.2839 | 0.1470 | — |
| actuator delay 2 / 4 / 7 physics steps | settled | 0.283 | 0.147 | ≤0.0033 |
| gains = deploy `k1_constants_isaaclab.py` (200/5) | settled | 0.2608 | 0.1378 | 0.0230 |
| gains = deploy `k1_isaaclab_gains.yaml` (200/3.5) | settled | 0.2611 | 0.1384 | 0.0228 |
| arms PD-held at deploy `DEFAULT_ANGLES` | settled | 0.2864 | 0.1483 | 0.0049 |
| arms PD-held at zero (not welded) | settled | 0.3178 | 0.1761 | 0.1227 |
| physics dt 0.001, PD 200 Hz / 1000 Hz | settled | 0.281 | 0.145 | ≤0.0129 |
| spawn at IsaacLab's `z = 0.6` and drop | settled | 0.2889 | 0.1477 | 0.0338 |
| effort limits = MJCF (30/35/20/40/20) | settled | 0.2839 | 0.1470 | 0.0000 |
| 40 s instead of 15 s | settled | 0.2883 | 0.1500 | 0.0174 |

Every variant settles, and RMS Δ stays in 0.138–0.176 rad. Across all 13 variants the
per-joint spread is ≤0.06 rad except `Ankle_Pitch` (0.149 rad) — the sagittal chain
trades hip-pitch against ankle-pitch for a given squat depth, so those two are the
least individually reproducible. The knee, which *is* the squat depth, is stable to
0.06 rad. **The conclusion — a deeper squat, RMS ≈0.15 rad from the config default —
survives every variant tested.** The deploy gains give a slightly *smaller* Δ
(0.138 vs 0.147 RMS), so this is not an artifact of using the training gains.

Two incidental notes. The actuator `effort_limit` in the IsaacLab cfg (68/76/38.3/112
/38.3) is 2–3× the MJCF/URDF `forcerange` (30/35/20/40/20); it makes no difference at
zero command because the standing torques are only ~0.6× of even the tighter limits.
And the `arms PD-held at zero` row moves the answer more than `pd-deploy` does — an
artifact of that variant, whose PD-held arms keep oscillating (leg velocity RMS 0.119
rad/s vs 0.0002–0.0006 in every other variant), not a statement about arm placement.

## 6. Recommendation

**The config default pose is not a usable hand-off target on its own.** The gap is
0.147 rad RMS / 0.284 rad peak for the deployed policy, which is far too large to
wave off: 0.284 rad of knee error fed into `q_target = q_default + 0.5·action` is
0.57 of action range, and the pose the get-up policy would hand over is one that —
as §3 shows — topples forward in 2 s if it is not immediately closed-loop stabilised.

Use the measured pose. For the deployed policy `policy_180843_19999.onnx`, the
mirror-symmetrised converged pose (the raw left/right asymmetry is only 0.035 rad, so
symmetrising costs almost nothing and buys a cleaner target):

| joint | config default | **recommended hand-off** | Δ |
| --- | --- | --- | --- |
| `Hip_Pitch` | −0.26 | **−0.452** | −0.192 |
| `Hip_Roll` | 0.00 | **±0.044** (L +, R −) | ±0.044 |
| `Hip_Yaw` | 0.00 | **±0.108** (L +, R −) | ±0.108 |
| `Knee_Pitch` | +0.52 | **+0.792** | +0.272 |
| `Ankle_Pitch` | −0.26 | **−0.313** | −0.053 |
| `Ankle_Roll` | 0.00 | **±0.011** (L +, R −) | ±0.011 |

Trunk height at this pose: 0.509 m (vs 0.525 m at the config default).
Symmetrised aggregate: max |Δ| 0.272 rad, RMS 0.146 rad.

Practical guidance, in priority order:

1. **Target the knee and hip pitch.** `Knee_Pitch +0.79` and `Hip_Pitch −0.45` carry
   almost all of the delta and are the most reproducible components. Hitting those
   two within ~0.05 rad matters more than any other joint.
2. **`Ankle_Pitch` and `Hip_Yaw` are soft.** Ankle pitch is the least reproducible
   joint under model perturbation (0.149 rad spread) and the three policies disagree
   on the sign of the hip-yaw split. Do not over-fit either; anywhere in
   `Ankle_Pitch ∈ [−0.40, −0.26]`, `|Hip_Yaw| ≤ 0.15` is inside the observed range.
3. **The get-up side currently ends at the wrong pose by construction.** The get-up
   deploy node's own PD offset, `GETUP_DEFAULT_ANGLES`
   (`origin/feat/dual_walk:src/k1_constants_isaaclab_getup.hpp:48-55`), carries exactly
   the config default legs `−0.26, 0, 0, 0.52, −0.26, 0`, and it too uses
   `use_default_offset=True` with `action_scale = 0.5`. So a get-up policy that
   finishes with near-zero action lands precisely on the pose this document measures
   as 0.147 rad RMS away from where the walk policy wants to be — while holding it at
   0.6× gains, which topples in 1.19 s. Either the get-up policy must be *trained* to
   end at the §6 pose, or the hand-off must include an explicit interpolation onto it.
4. **Do not need a perfect match — but do need a prompt hand-off.** The walk policy
   pulled itself from the config default to its own equilibrium in every one of the
   41 jittered multistarts, so it has a usable basin of attraction; ±0.05 rad of
   initial error is harmless. What is *not* harmless is dwelling at the hand-off pose
   under open-loop PD — engage the walk policy within a few hundred ms.
5. **Obtaining the latest trained policy would still improve this**, but is not
   required to act. The repo ships no newer walk policy (see §7), and the three
   policies here bracket the answer: RMS Δ 0.09–0.22 rad, unanimous in direction.
   A measurement on a newer policy would refine the target inside that band; it would
   not change the conclusion that the config default is the wrong target.

## 7. Limitations

**(a) The exported policies are the current ones, but the IsaacLab cfg has moved on.**
`policy_180843_19999.onnx` *is* the deployed walk policy — it is the default in every
newer launch file, and its ONNX blob is byte-identical between `origin/main` and
`origin/feat/dual_walk` (2026-08-07, the newest walking branch), so the repo ships no
newer walk policy. However, `IsaacLab-K1-Locomotion` at `852ca36` has since moved to a
100-step history buffer plus a 4-step MLP history (`history_layout.py:30-33`,
`flat_env_cfg.py:99-197`), whose exporter takes `(command, obs_history)` rather than a
single `obs` (`agents/history_policy_exporter.py:13-27`). No policy from that newer cfg
is exported anywhere locally. So the measurement describes the policy that is actually
deployed today, using an observation contract that the current training cfg no longer
uses. Related: the cfg's `_PHASE_FREQ_LOW` was raised 1.5 → 1.8 Hz on 2026-08-02
(`rough_env_cfg.py:45-49`) with a note that the real robot is still at 1.5 — irrelevant
to this measurement, since the phase is zero at rest, but it confirms the exported
policies predate the current cfg in at least that respect. Training-time PD gains may
likewise have differed from the current 160/4.0; §5 shows the answer is insensitive to
that (deploy gains 200/5 give RMS 0.138 vs 0.147).

**(b) IsaacLab-USD vs MuJoCo-MJCF sim gap.** Training runs on a USD converted from
`K1_locomotion.urdf` inside PhysX; this measurement runs the MJCF `K1_22dof.xml` in
MuJoCo. Kinematics, masses (19.666 kg), joint limits and leg armature were checked to
match, and the ten upper-body joints are welded to reproduce `merge_fixed_joints`. Not
matched: contact solver and material (MuJoCo resolves the foot/ground pair at
`condim = 3` with element-wise-max friction ≈1.0, against PhysX static/dynamic friction
randomised to 0.3–1.0 with multiply combine, `flat_env_cfg.py:311-316`); the MJCF
ground is a flat plane whereas training used `NOISY_FLAT_TERRAIN_CFG`
(`flat_env_cfg.py:36-56`: 70% of tiles carry 0.01–0.04 m uniform height noise,
30% flat); and none of training's link-mass/inertia/gain
randomisation is applied. §5 bounds the resulting uncertainty at ≤0.06 rad per joint
(≤0.15 rad for ankle pitch) — an order of magnitude smaller than the 0.28 rad effect
being reported.

**(c) One more caveat on the harness itself.** `booster_k1_locomotion`'s
`k1_constants_isaaclab.py:59` declares `OBS_DIM = 79`, which does **not** describe
these policies. It is a stale constant for the older 22-DoF `model_24999.onnx`; the
walk node ignores it and uses its own module-level `POLICY_OBS_DIM = 49`
(`rl_policy_isaaclab_node.py:64`, and `src/k1_constants_isaaclab.hpp:26`). Resolved
from the node code, as it must be — but anyone reading only the constants file would
build a 79-wide observation and get a silently garbage pose.

## 8. Context — which "K1 standing pose" is authoritative for what

Four distinct default poses exist across these repos. They are not interchangeable.

| source | legs (HipP, HipR, HipY, Knee, AnkP, AnkR) | arms | authoritative for |
| --- | --- | --- | --- |
| `rough_env_cfg.py:147-164` (walk cfg) | −0.26, 0, 0, 0.52, −0.26, 0 | none — welded at 0 | **the walk policy's PD offset**; this is the pose the 0.5·action is added to, and the one this document measures against |
| `k1_constants_isaaclab.py:23-29` (deploy) | identical to the above | ShP 0.3, ShR ∓1.374, ElP 0, ElY ∓1.2 | what the real robot commands; legs match the cfg exactly, arms are a **deploy-side choice with no training counterpart** |
| `getup_env_cfg.py:162-180` | identical to the above | ShP 0, ShR ∓1.374, ElP 0, ElY 0 | the IsaacLab get-up task's full-body init; same legs, arms differ from deploy on shoulder pitch and elbow yaw |
| `k1_constants.py:20-26` (holosoma 22-DoF) | −0.2, 0, 0, 0.4, −0.25, 0 | ShP 0.2, ShR ∓1.35, ElP 0, ElY ∓0.5 | the old `model_24999.onnx` policy only |
| `k1_constants_standup.py:36-42` (IsaacGymRL) | −0.2, 0, 0, 0.4, −0.25, 0 | all 0 | the separate stand-up policy only |
| `k1_constants_isaaclab_getup.hpp:48-55` (get-up deploy, `origin/feat/dual_walk`) | identical to row 1 | ShP 0, ShR ∓1.3744468, ElP 0, ElY 0 | the get-up deploy node's PD offset — the deploy mirror of row 3, not a fifth pose. `GETUP_OBS_DIM = 75`, 22 DoF, `action_scale 0.5`, gains = walk × 0.6 |

For the walk hand-off, the first row is the reference the policy is *defined* against
(it is the PD offset, so it cannot be changed without invalidating the checkpoint) —
and the third column of §6 is where the policy actually *goes* from there. The arms
are not measured here and do not need to be: the policy does not actuate them, so the
hand-off requirement on the upper body is simply "whatever the deploy node PD-holds",
i.e. row 2. Note that row 2's arm pose has no training counterpart at all (training
welds the arms at zero), which is a real, if apparently benign, morphology mismatch
between training and deployment — §5 shows it moves the leg equilibrium by only
0.005 rad.
