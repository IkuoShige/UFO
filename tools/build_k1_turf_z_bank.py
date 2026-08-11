#!/usr/bin/env python3
"""Build a minimal, no-finetune K1 turf get-up z-bank.

The vectors are read from the committed 5090_v2 latent record, so this does not
depend on gitignored ``runs/`` search artifacts.  ``getup_opt`` remains available
for rollback; ``getup_turf`` and ``standing_pooled`` are aliases for the selected
low-friction latent.  The actor checkpoint is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "docs" / "k1_getup_opt_z.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "ufo_fb_k1_5090_v2" / "export_onnx" / "z_bank_turf.npz"


def load_latent(source: Path, name: str) -> np.ndarray:
    payload = json.loads(source.read_text())
    try:
        z = np.asarray(payload["latents"][name]["z"], dtype=np.float32)
        z_dim = int(payload["z_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid latent record {source}: {exc}") from exc
    if z.shape != (z_dim,):
        raise ValueError(f"Expected latent {name!r} shape {(z_dim,)}, got {z.shape}")
    target_norm = math.sqrt(z_dim)
    norm = float(np.linalg.norm(z))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Latent {name!r} has invalid norm {norm}")
    return (z * (target_norm / norm)).astype(np.float32)


def build_bank(source: Path, output: Path) -> Path:
    current = load_latent(source, "cem_s4_5")
    turf = load_latent(source, "standing_pooled")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **{
            "z/getup_opt": current,
            "z/getup_turf": turf,
            "z/standing_pooled": turf,
        },
    )
    print(f"[INFO] wrote {output}")
    print("[INFO] skills=['getup_opt', 'getup_turf', 'standing_pooled']; norms=16")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_bank(args.source.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
