from __future__ import annotations

import unittest

from humanoidverse.training.workspace import _should_save_final_checkpoint


class WorkspaceCheckpointingTest(unittest.TestCase):
    def test_final_checkpoint_is_due_after_progress_past_periodic_checkpoint(self) -> None:
        self.assertTrue(
            _should_save_final_checkpoint(
                global_time=200_000_000,
                last_checkpoint_global_time=198_400_000,
            )
        )

    def test_periodic_checkpoint_at_final_step_is_not_duplicated(self) -> None:
        self.assertFalse(
            _should_save_final_checkpoint(
                global_time=200_000_000,
                last_checkpoint_global_time=200_000_000,
            )
        )

    def test_partial_rollout_budget_still_saves_latest_completed_step(self) -> None:
        self.assertTrue(
            _should_save_final_checkpoint(
                global_time=199_999_744,
                last_checkpoint_global_time=198_400_000,
            )
        )

    def test_resume_without_progress_does_not_rewrite_checkpoint(self) -> None:
        self.assertFalse(
            _should_save_final_checkpoint(
                global_time=198_400_000,
                last_checkpoint_global_time=198_400_000,
            )
        )


if __name__ == "__main__":
    unittest.main()
