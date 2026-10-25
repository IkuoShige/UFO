#!/usr/bin/env python3
"""Audit a UFO checkpoint for NaN/Inf without constructing the agent or simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import safetensors
import torch

DEFAULT_CHUNK_ELEMENTS = 8_000_000


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "train_status.json").is_file():
        return path
    checkpoint = path / "checkpoint"
    if (checkpoint / "train_status.json").is_file():
        return checkpoint
    raise FileNotFoundError(f"No checkpoint/train_status.json found under {path}")


def _torch_tensor_issue(tensor: torch.Tensor, *, chunk_elements: int) -> str | None:
    if tensor.numel() == 0 or not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
        return None
    flat = tensor.detach().reshape(-1)
    bad_count = 0
    first_bad: list[int] = []
    for start in range(0, flat.numel(), chunk_elements):
        finite = torch.isfinite(flat[start : start + chunk_elements])
        if bool(finite.all().item()):
            continue
        local = (~finite).nonzero(as_tuple=False).reshape(-1)
        bad_count += int(local.numel())
        if len(first_bad) < 5:
            first_bad.extend((local[: 5 - len(first_bad)] + start).cpu().tolist())
    if bad_count == 0:
        return None
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} bad_count={bad_count} flat_indices={first_bad}"


def _iter_torch_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "<root>", value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_torch_tensors(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_torch_tensors(item, child)


def audit_model(checkpoint: Path, *, chunk_elements: int) -> tuple[int, list[str]]:
    model_path = checkpoint / "model" / "model.safetensors"
    if not model_path.is_file():
        return 0, [f"missing model file: {model_path}"]
    checked = 0
    issues: list[str] = []
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            checked += tensor.numel()
            issue = _torch_tensor_issue(tensor, chunk_elements=chunk_elements)
            if issue is not None:
                issues.append(f"model.{name}: {issue}")
    return checked, issues


def audit_optimizers(checkpoint: Path, *, chunk_elements: int) -> tuple[int, list[str]]:
    optimizer_path = checkpoint / "optimizers.pth"
    if not optimizer_path.is_file():
        return 0, [f"missing optimizer file: {optimizer_path}"]
    try:
        state = torch.load(optimizer_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # PyTorch versions before mmap support.
        state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    checked = 0
    issues: list[str] = []
    for name, tensor in _iter_torch_tensors(state, "optimizers"):
        checked += tensor.numel()
        issue = _torch_tensor_issue(tensor, chunk_elements=chunk_elements)
        if issue is not None:
            issues.append(f"{name}: {issue}")
    return checked, issues


def _numpy_dtype_is_float(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)


def _hdf5_dataset_issue(dataset: h5py.Dataset, *, chunk_elements: int) -> tuple[int, str | None]:
    if dataset.size == 0 or not _numpy_dtype_is_float(dataset.dtype):
        return int(dataset.size), None
    row_elements = int(np.prod(dataset.shape[1:], dtype=np.int64)) if dataset.ndim > 1 else 1
    rows_per_chunk = max(1, chunk_elements // max(row_elements, 1))
    bad_count = 0
    first_bad: list[list[int]] = []
    if dataset.ndim == 0:
        chunk_bounds = [(0, 0)]
    else:
        chunk_bounds = [(start, min(start + rows_per_chunk, dataset.shape[0])) for start in range(0, dataset.shape[0], rows_per_chunk)]
    for start, stop in chunk_bounds:
        values = np.asarray(dataset[()] if dataset.ndim == 0 else dataset[start:stop])
        finite = np.isfinite(values)
        if bool(finite.all()):
            continue
        local_bad = np.argwhere(~finite)
        bad_count += int(local_bad.shape[0])
        if len(first_bad) < 5:
            for index in local_bad[: 5 - len(first_bad)]:
                global_index = index.tolist()
                if dataset.ndim > 0 and global_index:
                    global_index[0] += start
                first_bad.append(global_index)
    if bad_count == 0:
        return int(dataset.size), None
    return int(dataset.size), (f"shape={dataset.shape} dtype={dataset.dtype} bad_count={bad_count} indices={first_bad}")


def audit_buffers(checkpoint: Path, *, chunk_elements: int) -> tuple[int, list[str]]:
    buffers_dir = checkpoint / "buffers"
    buffer_paths = sorted(buffers_dir.glob("**/buffer.hdf5"))
    if not buffer_paths:
        return 0, [f"missing replay buffer under: {buffers_dir}"]
    checked = 0
    issues: list[str] = []
    for buffer_path in buffer_paths:
        with h5py.File(buffer_path, "r") as handle:
            datasets: list[tuple[str, h5py.Dataset]] = []

            def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
                if isinstance(item, h5py.Dataset):
                    datasets.append((name, item))

            handle.visititems(collect)
            for name, dataset in datasets:
                count, issue = _hdf5_dataset_issue(dataset, chunk_elements=chunk_elements)
                checked += count
                if issue is not None:
                    relative = buffer_path.relative_to(checkpoint)
                    issues.append(f"{relative}:{name}: {issue}")
    return checked, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Run directory or checkpoint directory")
    parser.add_argument("--skip-optimizers", action="store_true")
    parser.add_argument("--skip-buffer", action="store_true")
    parser.add_argument("--chunk-elements", type=int, default=DEFAULT_CHUNK_ELEMENTS)
    args = parser.parse_args()
    if args.chunk_elements <= 0:
        parser.error("--chunk-elements must be positive")
    return args


def main() -> int:
    args = parse_args()
    checkpoint = _resolve_checkpoint(args.checkpoint)
    status = json.loads((checkpoint / "train_status.json").read_text())
    print(
        "Auditing "
        f"{checkpoint} (global_time={status.get('global_time', status.get('time'))}, "
        f"optimizer_steps={status.get('optimizer_steps')})"
    )

    sections = [("model", audit_model)]
    if not args.skip_optimizers:
        sections.append(("optimizers", audit_optimizers))
    if not args.skip_buffer:
        sections.append(("buffer", audit_buffers))

    all_issues: list[str] = []
    for label, audit in sections:
        checked, issues = audit(checkpoint, chunk_elements=args.chunk_elements)
        all_issues.extend(issues)
        print(f"{label}: checked {checked:,} values, issues={len(issues)}")

    if all_issues:
        print("FAILED: checkpoint contains non-finite or missing state")
        for issue in all_issues[:50]:
            print(f"  - {issue}")
        if len(all_issues) > 50:
            print(f"  ... {len(all_issues) - 50} more issues")
        return 1
    print("PASS: all audited floating-point checkpoint state is finite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
