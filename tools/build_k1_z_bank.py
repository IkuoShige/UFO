#!/usr/bin/env python3
"""Build a z-bank (.npz) of BFM skill latents for the K1 UFO policy.

The BFM actor is skill-agnostic: one network, the skill is selected purely by the
256-dim latent ``z``. A z-bank is a named collection of such latents so the deploy
node can switch skills at runtime (get-up -> walk -> kick ...).

Layout (consumed by ``booster_k1_locomotion/ufo_policy_runtime.py::ZBank``):
    ``z/<name>``   (256,)   a single skill latent
    ``seq/<name>`` (T, 256) a time-indexed latent sequence for open-loop replay

All entries are re-projected onto the ``||z|| = sqrt(256)`` sphere on write and
again on read (FBcprAuxModel.project_z, archi.norm_z=True). A plain mean of
latents is off-manifold until re-projected -- never skip this.

Example::

    uv run python tools/build_k1_z_bank.py \
        --source runs/ufo_fb_k1_5090_v2/tracking_inference_157M/zs_17.pkl \
        --output runs/ufo_fb_k1_5090_v2/export_onnx/z_bank.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent / "booster_k1_locomotion" / "booster_k1_locomotion"))

DEFAULT_SOURCE = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2" / "tracking_inference_157M" / "zs_17.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2" / "export_onnx" / "z_bank.npz"


def load_zs(path: Path) -> np.ndarray:
    zs = joblib.load(str(path))
    arr = np.asarray(zs.detach().cpu().numpy() if hasattr(zs, "detach") else zs, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected (T, z_dim) latents in {path}, got {arr.shape}")
    return arr


def main(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    prefix: str = "getup",
    seq_stride: int = 1,
    getup_lo: int = 100,
    getup_hi: int = 140,
) -> None:
    from ufo_policy_runtime import ZBank, project_z  # noqa: E402

    source, output = Path(source).expanduser(), Path(output).expanduser()
    zs = load_zs(source)
    total = zs.shape[0]
    print(f"[INFO] {source.name}: {zs.shape} latents, norms {np.linalg.norm(zs, axis=-1)[:3]}")

    lo, hi = int(total * 0.45), int(total * 0.55)
    skills = {
        # Interim get-up latent, validated in tools/k1_ufo_sim2sim.py: from a
        # fallen pose (face up or face down) the robot stands within ~0.8 s and
        # holds the stand at the nominal 0.53 m root height. Taken from the
        # early "getting up" window of the fallAndGetUp tracking rollout.
        # WS-A is searching for a better one; swap it in here.
        prefix: project_z(zs[getup_lo:getup_hi].mean(axis=0)),
        # Mean of a mid-sequence window, re-projected.
        f"{prefix}_mid": project_z(zs[lo:hi].mean(axis=0)),
        # Whole-clip mean: the "average behaviour" of this tracking rollout.
        f"{prefix}_mean": project_z(zs.mean(axis=0)),
        # A single mid-sequence frame, no averaging.
        f"{prefix}_frame": project_z(zs[total // 2]),
    }
    sequences = {f"{prefix}_track": zs[::max(int(seq_stride), 1)]}

    path = ZBank.save(output, skills, sequences)
    bank = ZBank.load(path)
    print(f"[INFO] wrote {path}")
    print(f"[INFO] skills={sorted(bank.skills)} sequences={{{', '.join(f'{k}:{v.shape}' for k, v in bank.sequences.items())}}}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
