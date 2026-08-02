# Training Booster K1 on vast.ai

This documents the Booster K1 bring-up on top of UFO's [robot-config training
path](robot_config_training.md), and how to run it on a rented GPU box on
[vast.ai](https://vast.ai) via the repo's `Dockerfile`.

## What's already set up in this repo

- `configs/robots/k1_22dof.yaml` — curated robot config for the Booster K1
  (22 DoF), generated from `booster_assets/robots/K1/K1_22dof.xml` (+URDF)
  with `humanoidverse.tools.robot_inspect`, then manually reviewed (feet,
  hands, key_bodies, contact bodies matched against the XML body names;
  `init_state.pos.z` corrected from the XML-derived draft value to `0.55`,
  matching the actual root height observed in the K1 motion data — the XML's
  own `Trunk` body placement of `z=1.0` is just a visualization default, not
  a standing height).
- `humanoidverse/config/robot/k1/k1_22dof_auto.yaml` — matching Hydra robot
  config draft, referenced by `configs/robots/k1_22dof.yaml`'s
  `training.hydra_robot: k1/k1_22dof_auto`.
- `configs/data/k1_lafan1.yaml` — data manifest for the retargeted LAFAN1 K1
  motion set from
  [`wu0712/retargeted_lafan1_for_booster_k1`](https://huggingface.co/datasets/wu0712/retargeted_lafan1_for_booster_k1)
  (77 motion clips, headered `robot_state_csv`, `root_pos_*`/`root_rot_*`/
  `dof_pos_N` columns — column names differ from UFO's CSV defaults, hence
  the explicit `columns:` block in the manifest).
- `scripts/download_k1_lafan1_data.sh` — downloads and stages that dataset
  under `humanoidverse/data/k1_lafan1/`.
- `Dockerfile` — CUDA 12.8 image with `uv sync`, `booster_assets` cloned as a
  sibling directory, the K1 LAFAN1 CSVs staged, and the motion cache
  pre-built. `pyproject.toml` pins `torch`/`torchvision` to PyTorch's cu128
  index (see the comment above `[tool.uv.sources]`) instead of the PyPI
  default cu126 build, because cu126 wheels have no Blackwell (sm_100/sm_120)
  kernels — on an RTX 50-series GPU (e.g. RTX 5090) that fails at env
  creation with `CUDA error: no kernel image is available for execution on
  the device`. cu128 wheels remain backward-compatible with older
  Ampere/Ada GPUs (verified on a local RTX 3090).

This was smoke-tested locally (`--smoke`, single GPU) end to end: robot XML
load -> env/observation/reward manager construction -> motion library load
(1692 training clips, 496671 frames) -> one training step. It has **not**
been through a full training run, and the PD gains / default pose in
`configs/robots/k1_22dof.yaml` are still XML-derived draft values — treat
this as a working starting point, not a tuned config. See
`metadata.warnings` in that file for exactly what was auto-generated vs.
reviewed.

LAFAN1 itself is licensed CC BY-NC-ND 4.0 (non-commercial, no derivatives) —
keep that in mind for anything trained on this data.

## 1. Build and push the image

From the UFO repo root (needs a GPU for the smoke-tested build steps below
to make sense, though the `uv sync` / data-fetch build steps themselves are
CPU-only):

```bash
docker build -t <dockerhub-user>/ufo-k1:latest .
docker push <dockerhub-user>/ufo-k1:latest
```

The image bakes in the full uv-managed Python env, `booster_assets`, the K1
LAFAN1 motion CSVs, and a pre-built motion cache — it's ready to train as
soon as it starts, no extra setup step on the rented box. Expect several GB
(CUDA + torch + mujoco + ~650MB of K1 motion data/cache).

If you don't want to publish images, vast.ai also supports pulling from a
private registry (GHCR, private Docker Hub repo) — configure registry
credentials in the vast.ai console when creating the instance.

## 2. Launch on vast.ai

- Search for an offer with the GPU count/VRAM you want (an RTX 3090/4090 is
  enough for a smoke test; scale up GPU count for a full FB/TeCH run — see
  `README.md`'s G1 quick start for the multi-GPU command shape).
- Create the instance with:
  - **Image**: `<dockerhub-user>/ufo-k1:latest`
  - **Disk space**: at least 30-40 GB (image + checkpoints + W&B logs).
  - **On-start command / entrypoint**: the image's `ENTRYPOINT` is `bash`,
    so either open an interactive shell (`vastai ssh <instance-id>` or the
    web terminal) and run commands manually, or override the on-start
    command with a full `run_train.sh` invocation (see below) if you want
    training to start automatically.
- If you want W&B logging, set `WANDB_API_KEY` as an environment variable on
  the instance (vast.ai lets you set env vars per-instance) and add
  `--use-wandb --wandb-run-name ...` to the training command.

## 3. Run training

Smoke test first, exactly like the local run that validated this config:

```bash
cd /workspace/UFO
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/k1_22dof.yaml \
  --data-manifest configs/data/k1_lafan1.yaml \
  --gpu-ids single \
  --smoke \
  --buffer-size 20000 \
  --work-dir /tmp/ufo_smoke_k1
```

`--buffer-size 20000` matters for a smoke test — the full training default
(`5120000` per GPU) tries to allocate a replay buffer sized for real
training and can OOM on a single low-VRAM GPU before any actual training
step runs.

Then a full run, scaling `--gpu-ids` / `--num-envs` / `--buffer-size` to
whatever the rented instance has (drop `--buffer-size` to use the
default once you have enough VRAM/GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/k1_22dof.yaml \
  --data-manifest configs/data/k1_lafan1.yaml \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_k1 \
  --update-z-every-step 100 \
  --buffer-size 5120000
```

Swap `--agent fb` for `--agent tech` for TeCH training, same as the G1 path
in the main README.

## 4. Before treating results as meaningful

Per `docs/robot_config_training.md`, this is still an experimental path for
new robots. Before trusting a K1 policy:

- Watch reward curves for the first hour of training — PD gains/action
  scale are XML-derived drafts and may need retuning if the robot never
  stabilizes standing.
- Non-G1 reward inference currently only covers root/locomotion tasks; goal
  inference needs a K1-specific goal JSON if you go beyond that.
- The `deploy` branch / ONNX export path is G1-oriented; K1 real-robot
  deployment is out of scope for what's set up here.

## 5. Getting checkpoints and logs back off the box

`--work-dir runs/ufo_fb_k1` writes checkpoints/config/logs under
`/workspace/UFO/runs/ufo_fb_k1` inside the container. Either `scp`/`rsync`
that directory off before terminating the instance, or point `--work-dir` at
a vast.ai-mounted persistent volume if you're using one.
