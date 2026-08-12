# K1 UFO no-finetune turf latent selection

## Decision

Rejected on real hardware. The existing `standing_pooled` latent from the frozen
`runs/ufo_fb_k1_5090_v2` actor was tested under the deployment alias `getup_turf`,
but it performed worse than `getup_opt` on artificial turf on 2026-08-12. Use
`getup_opt`; retain this document and the alias only as a reproducible negative
result.

This changes only the 256-dimensional runtime latent `z`. It does not train,
finetune, or modify the actor, backward encoder, optimizer, or checkpoint.

## Why a different existing latent helps

The deployed `getup_opt` (`cem_s4_5`) was selected mainly for the walking-policy
handoff pose. Its historical search intentionally disabled startup physics,
including friction randomization, to make candidate ranking deterministic. It was
therefore never optimized for low-friction contact.

`standing_pooled` was the earlier frozen-policy get-up champion. It leaves a less
crouched final pose and gives up some handoff-pose accuracy, but the corrected
single-ground simulation shows that it recovers upright more often when contact
friction is low. This is a behavior already present in the learned 5090_v2 model,
not new knowledge added after training.

## Search and selection

The pilot evaluated the deployed latent, existing get-up/standing/handoff latents,
and spherical interpolations from `getup_opt` toward those endpoints. Spherical
interpolation preserves the model's required `||z|| = sqrt(256) = 16` constraint.
The sweep used five arc fractions (`0.2, 0.4, 0.6, 0.8, 1.0`) and did not update any
model state.

The endpoint `standing_pooled` beat the intermediate points often enough that a
more expensive CEM search was not justified. It is also simpler to reproduce and
already has nominal get-up and self-collision evidence in
`docs/k1_getup_optimization.md`.

The final comparison tiled both candidates over the same initial-condition bank and
rotated their environment blocks between batches. Each fixed-friction cell used 64
OOD episodes per candidate, then the whole comparison was repeated with a separate
seed. Each batch sampled 16 conditions from four tilt types and a 12-value yaw
grid, with a 0.35 m drop, joint-pose noise, observation noise, action latency of
0--2 control ticks, and observation latency of 0--1 tick. Pushes, mass, COM, and
default-pose randomization were disabled to isolate friction plus deployment-side
sensing/control variation.

Primary metric: rose upright for at least 0.5 s and satisfied the height/tilt
criterion for at least 95% of the final 2 s. The historical strict metric also
requires double-foot contact force and undercounts a robot that is upright but still
taking a recovery step.

| fixed friction | seed 97: `getup_opt` / turf | seed 701: `getup_opt` / turf | pooled delta (128 each) |
| ---: | ---: | ---: | ---: |
| 0.05 | 29.7% / **46.9%** | 32.8% / **45.3%** | **+14.8 points** |
| 0.10 | 54.7% / **68.8%** | 51.6% / **62.5%** | **+12.5 points** |
| 0.20 | 67.2% / **85.9%** | 67.2% / **78.1%** | **+14.8 points** |
| 0.50 | 96.9% / **100%** | 95.3% / **100%** | **+3.9 points** |
| 1.00 | 100% / 100% | 100% / 100% | 0 points |

The paired discordant counts also favor turf/current by 22/3, 20/4, and 23/4 at
friction 0.05, 0.10, and 0.20 respectively. This is a consistent low-friction
effect, not a single aggregate driven by one seed.

Plain-MuJoCo sim2sim was also run with one ground plane at friction 0.05 and 0.20.
`standing_pooled` stood up and remained upright from both face-up and face-down
starts in all four probes. Time to stand was 0.72--2.34 s; the slow case was the
face-down start at friction 0.05.

## Reproduce

Build the arc sweep and run the paired evaluator:

```bash
uv run python tools/build_k1_turf_z_sweep.py

uv run python -m humanoidverse.tools.eval_getup_z_turf \
  --candidates runs/getup_turf/z_arc_sweep_candidates.json \
  --names getup_opt opt_to_standing_t100 \
  --output runs/getup_turf/paired_confirm.json \
  --frictions 0.05 0.1 0.2 0.5 1.0 \
  --conditions 16 --batches 4 --seed 701
```

Build a small deployment bank from the committed latent record. This works even if
the gitignored search cache is absent:

```bash
uv run python tools/build_k1_turf_z_bank.py
```

The output is
`runs/ufo_fb_k1_5090_v2/export_onnx/z_bank_turf.npz` and contains:

- `z/getup_turf`: selected `standing_pooled` vector;
- `z/getup_opt`: current deployed vector for rollback;
- `z/standing_pooled`: explicit source-name alias.

For historical reproduction in `booster_k1_locomotion`, place that file in the
get-up bundle as `z_bank.npz` and launch simulation with
`getup_z:=getup_turf`. Do not select it on hardware; use
`getup_z:=getup_opt`. The ONNX actor and deploy spec do not change.

## Limit and hardware result

This was a meaningful simulator experiment, but not an effective real-turf
mitigation. The simulator models scalar Coulomb friction; turf pile compliance, toe
catching, local height changes, and directional drag are absent. It also did not
reach 100% under the extreme 0.05--0.20 OOD stress cells.

The hardware A/B gate rejected `getup_turf`: it was worse than the previous
`getup_opt` deployment on real artificial turf. This demonstrates that the scalar
Coulomb-friction simulation result did not transfer; it must not be used as evidence
that the alias is safer or more robust on K1. The separate friction finetune remains
the active experiment because it can change the policy itself, but it still requires
the same hardware gate after checkpoint and latent regeneration.
