from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import safetensors.torch
import torch

from humanoidverse.agents.fb.agent import FBAgent

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_checkpoint_finite.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location("audit_checkpoint_finite", _AUDIT_PATH)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
audit_buffers = _AUDIT_MODULE.audit_buffers
audit_model = _AUDIT_MODULE.audit_model
audit_optimizers = _AUDIT_MODULE.audit_optimizers


class _SequenceBuffer:
    def __init__(self, values):
        self.values = iter(values)

    def sample(self, _batch_size):
        return next(self.values)


class FBNonfiniteGuardTest(unittest.TestCase):
    def test_bad_replay_sample_is_retried_before_use(self) -> None:
        agent = object.__new__(FBAgent)
        agent.cfg = SimpleNamespace(train=SimpleNamespace(batch_size=1, nonfinite_batch_retries=2))
        buffer = _SequenceBuffer(
            [
                {"observation": torch.tensor([[math.nan]])},
                {"observation": torch.tensor([[1.0]])},
            ]
        )

        batch, retries = agent._sample_finite_batch(buffer, label="train_batch")

        self.assertEqual(retries, 1)
        self.assertEqual(batch["observation"].item(), 1.0)

    def test_bad_gradients_are_rejected_before_optimizer_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter.grad = torch.tensor([math.nan])

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            FBAgent._check_or_clip_gradients([parameter], None)

        self.assertEqual(parameter.item(), 1.0)
        optimizer.zero_grad(set_to_none=True)

    def test_checkpoint_auditor_locates_nonfinite_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir)
            (checkpoint / "model").mkdir()
            (checkpoint / "buffers" / "train").mkdir(parents=True)
            safetensors.torch.save_file(
                {"finite": torch.ones(2), "bad": torch.tensor([0.0, math.inf])},
                checkpoint / "model" / "model.safetensors",
            )
            torch.save(
                {"actor_optimizer": {"state": {0: {"exp_avg": torch.tensor([math.nan])}}}},
                checkpoint / "optimizers.pth",
            )
            with h5py.File(checkpoint / "buffers" / "train" / "buffer.hdf5", "w") as handle:
                handle.create_dataset("observation", data=np.array([[1.0], [np.nan]], dtype=np.float32))

            _, model_issues = audit_model(checkpoint, chunk_elements=1)
            _, optimizer_issues = audit_optimizers(checkpoint, chunk_elements=1)
            _, buffer_issues = audit_buffers(checkpoint, chunk_elements=1)

        self.assertEqual(len(model_issues), 1)
        self.assertIn("model.bad", model_issues[0])
        self.assertEqual(len(optimizer_issues), 1)
        self.assertIn("exp_avg", optimizer_issues[0])
        self.assertEqual(len(buffer_issues), 1)
        self.assertIn("observation", buffer_issues[0])


if __name__ == "__main__":
    unittest.main()
