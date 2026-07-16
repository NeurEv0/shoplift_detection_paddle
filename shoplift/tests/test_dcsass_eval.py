from __future__ import annotations

import unittest

from shoplift.eval.dcsass_eval import _metrics, _select_samples


class DcsassEvalTest(unittest.TestCase):
    def test_select_samples_can_balance_by_label(self) -> None:
        samples = [
            {"clip_id": "n0", "label": 0},
            {"clip_id": "n1", "label": 0},
            {"clip_id": "n2", "label": 0},
            {"clip_id": "p0", "label": 1},
            {"clip_id": "p1", "label": 1},
        ]

        selected = _select_samples(
            samples,
            clip_ids=None,
            max_clips=None,
            max_clips_per_label=2,
        )

        self.assertEqual([sample["clip_id"] for sample in selected], ["n0", "n1", "p0", "p1"])

    def test_select_samples_preserves_requested_clip_order(self) -> None:
        samples = [
            {"clip_id": "a", "label": 0},
            {"clip_id": "b", "label": 1},
        ]

        selected = _select_samples(
            samples,
            clip_ids=["b", "a"],
            max_clips=None,
            max_clips_per_label=None,
        )

        self.assertEqual([sample["clip_id"] for sample in selected], ["b", "a"])

    def test_metrics_counts_visualizations(self) -> None:
        metrics = _metrics(
            [
                {"label": 1, "pred_label": 0, "visualization_path": "outputs/a.mp4"},
                {"label": 0, "pred_label": 0, "visualization_path": ""},
            ]
        )

        self.assertEqual(metrics["visualization_count"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)


if __name__ == "__main__":
    unittest.main()
