# UFO training image for Booster K1, targeting GPU rental hosts (e.g. vast.ai).
#
# Build (from the UFO repo root):
#   docker build -t <you>/ufo-k1:latest .
#
# The image bakes in:
#   - system deps + uv-managed Python 3.10 env (uv sync, from pyproject.toml/uv.lock)
#   - booster_assets (K1 MJCF/URDF) cloned as a sibling of UFO, as required by
#     configs/robots/k1_22dof.yaml's relative xml_path
#   - the retargeted LAFAN1 -> K1 motion CSVs (scripts/download_k1_lafan1_data.sh)
#
# See docs/vastai_k1.md for how to push this image and launch it on vast.ai.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    unzip \
    build-essential \
    libgl1 \
    libegl1 \
    libosmesa6-dev \
    libglfw3 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# booster_assets holds the K1 MJCF/URDF; UFO's robot configs expect it as a
# sibling directory (../booster_assets relative to the UFO repo root).
ARG BOOSTER_ASSETS_REPO=https://github.com/IkuoShige/booster_assets.git
RUN git clone --depth 1 "${BOOSTER_ASSETS_REPO}" /workspace/booster_assets

COPY . /workspace/UFO
WORKDIR /workspace/UFO

RUN uv sync --frozen

RUN bash scripts/download_k1_lafan1_data.sh

# Warm the motion-data cache (root_pos/quat/dof_pos -> ufo_pkl) so the first
# training run doesn't pay the CSV-parsing cost.
RUN uv run python -c "\
from humanoidverse.utils.motion_data.manifest import prepare_motion_manifest; \
prepare_motion_manifest('configs/data/k1_lafan1.yaml', rebuild_cache=True)"

ENTRYPOINT ["/bin/bash"]
