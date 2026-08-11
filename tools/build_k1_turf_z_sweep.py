#!/usr/bin/env python3
"""Build an interpretable low-friction get-up latent sweep for K1.

The frozen FB policy constrains constant latents to the radius-sqrt(z_dim)
sphere.  This tool follows great-circle arcs from the deployed ``getup_opt``
latent toward three existing behaviors that expose different low-friction
trade-offs, then writes an NPZ z-bank and an ``eval_getup`` candidate file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation preserving the source latent radius."""

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    radius = float(np.linalg.norm(a))
    if radius <= 0.0 or not np.isclose(np.linalg.norm(b), radius, rtol=1e-4, atol=1e-5):
        raise ValueError("Latent endpoints must have the same non-zero norm")
    au = a / radius
    bu = b / radius
    dot = float(np.clip(np.dot(au, bu), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-7:
        out = (1.0 - t) * au + t * bu
    else:
        out = (math.sin((1.0 - t) * omega) * au + math.sin(t * omega) * bu) / math.sin(omega)
    out *= radius / np.linalg.norm(out)
    return out.astype(np.float32)


def build_sweep(
    *,
    deploy_bank: Path,
    finalists: Path,
    output_bank: Path,
    output_candidates: Path,
    fractions: tuple[float, ...],
) -> None:
    with np.load(deploy_bank) as bank:
        getup_opt = np.asarray(bank["z/getup_opt"], dtype=np.float32)
        endpoints = {
            "standing": np.asarray(bank["z/standing_pooled"], dtype=np.float32),
            "getup_original": np.asarray(bank["z/getup"], dtype=np.float32),
        }
    finalist_bank = torch.load(finalists, map_location="cpu", weights_only=False)
    endpoints["handoff"] = torch.as_tensor(finalist_bank["handoff_pool_500"]).numpy().astype(np.float32)

    vectors: dict[str, np.ndarray] = {"getup_opt": getup_opt}
    for endpoint_name, endpoint in endpoints.items():
        for fraction in fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"Sweep fractions must satisfy 0 < t <= 1, got {fraction}")
            name = f"opt_to_{endpoint_name}_t{int(round(100 * fraction)):03d}"
            vectors[name] = slerp(getup_opt, endpoint, fraction)

    output_bank.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_bank, **{f"z/{name}": value for name, value in vectors.items()})
    candidates = [
        {
            "name": name,
            "type": "constant",
            "source": {"kind": "file", "path": str(output_bank), "key": f"z/{name}"},
        }
        for name in vectors
    ]
    output_candidates.write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"[INFO] wrote {len(vectors)} latents to {output_bank}")
    print(f"[INFO] wrote eval candidates to {output_candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deploy-bank",
        type=Path,
        default=Path("runs/ufo_fb_k1_5090_v2/export_onnx/z_bank.npz"),
    )
    parser.add_argument("--finalists", type=Path, default=Path("runs/getup_opt/finalist_z.pt"))
    parser.add_argument("--output-bank", type=Path, default=Path("runs/getup_turf/z_arc_sweep.npz"))
    parser.add_argument(
        "--output-candidates",
        type=Path,
        default=Path("runs/getup_turf/z_arc_sweep_candidates.json"),
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_sweep(
        deploy_bank=args.deploy_bank,
        finalists=args.finalists,
        output_bank=args.output_bank,
        output_candidates=args.output_candidates,
        fractions=tuple(args.fractions),
    )


if __name__ == "__main__":
    main()
