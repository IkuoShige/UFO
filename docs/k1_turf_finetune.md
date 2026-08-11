# K1 UFO turf-friction adaptation

Purpose: adapt `runs/ufo_fb_k1_5090_v2` to the slippery artificial-turf surface
observed during real K1 deployment through `booster_k1_locomotion`, while preserving
the reference UFO/FB-CPR training recipe and its general behavior repertoire.

## Root cause and fix

The pre-fix K1 MJLab scene contained two coincident planes:

- `robot/ground`, embedded in `K1_22dof.xml`, friction 0.4;
- MJLab `terrain`, friction 1.0.

MuJoCo uses the elementwise maximum for equal-priority geom friction. The old DR
changed only robot geom friction, so the untouched terrain coefficient 1.0 prevented
effective ground-contact friction from dropping below 1.0. The duplicate contacts
also made the ground artificially stiff.

The corrected path:

1. strips world-level plane geoms from a robot MJCF before MJLab attaches it;
2. keeps exactly one scene-owned `terrain` plane;
3. samples one tangential-friction value per environment;
4. writes that same value to all robot geoms and the terrain geom, so hands, knees,
   feet, and ground share one material draw and the requested range is the effective
   contact range.

This path is used for both fresh training and checkpoint resume. The default range
remains the UFO reference setting `[0.5, 1.25]`. `--friction-range LOW HIGH` is an
explicit override for turf adaptation.

## Verification completed locally

- Compiled K1 scene: one plane, named `terrain`.
- Eight-world GPU probe with friction 0.2–0.6: observed coefficients 0.258–0.508;
  robot/terrain difference within every world was exactly 0.
- Fresh-training smoke: 16 environments, 2,048 steps, friction 0.2–0.6, exit 0.
- Checkpoint-resume/update smoke: copied a source run at 2,048 steps into a new
  run, loaded its agent and replay buffer, then continued only to 12,288 steps.
  The resumed FB agent completed 32 optimizer updates and checkpointed with exit
  0; the source train status remained at 2,048 steps.
- Standalone deploy-runtime sim2sim now uses one ground by default and accepts
  `--ground-friction` for fixed-coefficient probes.

No full training or `5090_v2` checkpoint mutation was performed locally.

### Pre-finetune low-friction baseline

`5090_v2` + its deployed `getup_opt` was evaluated with the corrected one-ground
scene, identical 16 OOD poses/seed, observation noise, action latency 0–2 ticks,
and observation latency 0–1 tick. Pushes, mass, COM, and default-pose DR were off so
the comparison isolates friction plus deploy-side sensing/control variation.

| fixed friction | strict success | upright success | rose at least once |
| ---: | ---: | ---: | ---: |
| 0.05 | 3/16 (18.8%) | 3/16 (18.8%) | 11/16 (68.8%) |
| 0.20 | 7/16 (43.8%) | 10/16 (62.5%) | 13/16 (81.2%) |
| 0.50 | 8/16 (50.0%) | 16/16 (100%) | 16/16 (100%) |

The historical strict criterion also requires final double-foot force and is known
to undercount a stable robot that is still taking a recovery step. `upright success`
is therefore the primary turf gate; the strict number remains recorded for continuity.
These are small-sample smoke baselines, not final confidence estimates.

## Fresh reference training

No friction flag is needed. The reference range `[0.5, 1.25]` is now applied to the
effective contact instead of being masked by the second plane.

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/k1_22dof.yaml \
  --data-manifest configs/data/k1_lafan1.yaml \
  --gpu-ids single \
  --num-envs 512 \
  --num-env-steps 192000000 \
  --update-z-every-step 100 \
  --buffer-size 2000000 \
  --work-dir runs/ufo_fb_k1_5090_friction_fixed
```

Apart from the physics bug fix, this keeps the existing UFO FB recipe and 5090-proven
buffer size.

## Safe `5090_v2` turf finetune

Do not resume inside the source directory. `--resume-from` copies the full checkpoint,
optimizer, replay buffer, and train status into a new work directory before loading.
It refuses to overwrite an existing target checkpoint.

`--num-env-steps` is the final total global step target, not the number of extra steps.
For a source at 192M, 200M means an 8M-step first phase.

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/k1_22dof.yaml \
  --data-manifest configs/data/k1_lafan1.yaml \
  --gpu-ids single \
  --num-envs 512 \
  --num-env-steps 200000000 \
  --update-z-every-step 100 \
  --buffer-size 2000000 \
  --friction-range 0.05 1.25 \
  --resume-from runs/ufo_fb_k1_5090_v2 \
  --work-dir runs/ufo_fb_k1_5090_v2_turf_ft
```

Resume restores `5090_v2`'s saved agent configuration and optimizer state, including
the reference learning rates. Do not add `--lr-scale` to this command: the checkpoint
values, not a newly built agent preset, govern a resumed agent.

If interrupted after staging, rerun the same command **without** `--resume-from`.
The existing target checkpoint then resumes normally. Keep the original
`runs/ufo_fb_k1_5090_v2` unchanged as the rollback baseline.

The 0.05 lower bound is a stress setting, not a measured turf coefficient. Before a
longer continuation, compare fixed-friction evaluation at 0.05, 0.1, 0.2, 0.5, and
1.0. Replace the stress lower bound if a measured turf coefficient becomes available.

## Evaluation gate

Evaluate old and finetuned checkpoints with the same initial-pose seed and:

- four OOD orientations;
- fixed friction grid: 0.05, 0.1, 0.2, 0.5, 1.0;
- observation noise;
- action latency 0–2 ticks and observation latency 0–1 tick;
- nominal and reference `[0.5, 1.25]` regression conditions.

The pre-finetune candidate specification is
`docs/k1_getup_turf_candidates.json`. After finetuning, do not reuse its old `z`:
export the new actor/backward encoder, regenerate the get-up latent for the new
checkpoint, then run the gate. A checkpoint update and latent update are one atomic
deployment unit.

## Modeling limit

This fix and override cover Coulomb tangential friction. Real turf may also add pile
compliance, local height variation, toe catching, or direction-dependent resistance.
If corrected low-friction simulation does not reproduce the remaining real failure,
extend the ground model only after collecting video/telemetry evidence; do not hide a
contact-model mismatch by blindly widening friction further.
