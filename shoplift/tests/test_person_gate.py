from __future__ import annotations

import unittest

from shoplift.core.types import DetectionBox, FrameMeta, Tracklet
from shoplift.vision import PersonGate, evaluate_person_gate


class PersonGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameMeta(
            frame_id=3,
            timestamp_ms=100,
            camera_id="camera-gate",
            width=1280,
            height=720,
        )

    def test_empty_frame_skips_heavy_modules(self) -> None:
        result = evaluate_person_gate([], frame=self.frame)

        self.assertFalse(result.has_person)
        self.assertTrue(result.skipped_heavy_modules)
        self.assertFalse(result.should_run_heavy_modules)
        self.assertEqual(result.reason, "no_person_skip_heavy_modules")

    def test_person_detection_triggers_heavy_modules(self) -> None:
        person_box = DetectionBox(
            box_id="det-1",
            frame_id=self.frame.frame_id,
            timestamp_ms=self.frame.timestamp_ms,
            category="person",
            bbox=(10, 20, 100, 220),
            score=0.8,
        )
        item_box = DetectionBox(
            box_id="det-2",
            frame_id=self.frame.frame_id,
            timestamp_ms=self.frame.timestamp_ms,
            category="item",
            bbox=(200, 220, 240, 280),
            score=0.9,
        )

        result = evaluate_person_gate([person_box, item_box], frame=self.frame)

        self.assertTrue(result.has_person)
        self.assertFalse(result.skipped_heavy_modules)
        self.assertEqual(result.person_boxes, (person_box,))

    def test_person_tracklet_outputs_track_id(self) -> None:
        tracked_box = DetectionBox(
            box_id="mot-1",
            frame_id=self.frame.frame_id,
            timestamp_ms=self.frame.timestamp_ms,
            category="person",
            bbox=(30, 40, 120, 260),
            score=0.82,
            track_id="person-9",
        )
        tracklet = Tracklet(track_id="person-9", category="person", boxes=(tracked_box,))

        result = evaluate_person_gate(tracklets=[tracklet], frame=self.frame)

        self.assertEqual(result.person_track_ids, ("person-9",))
        self.assertEqual(result.person_boxes, (tracked_box,))
        self.assertEqual(result.to_dict()["person_track_ids"], ["person-9"])

    def test_gate_metrics_accumulate_skip_and_trigger_rates(self) -> None:
        gate = PersonGate(min_score=0.5)
        person_box = DetectionBox(
            box_id="det-1",
            frame_id=self.frame.frame_id,
            timestamp_ms=self.frame.timestamp_ms,
            category="person",
            bbox=(10, 20, 100, 220),
            score=0.8,
        )

        gate.evaluate([], frame=self.frame)
        gate.evaluate([person_box], frame=self.frame)

        self.assertEqual(gate.metrics.total_frames, 2)
        self.assertEqual(gate.metrics.skipped_frames, 1)
        self.assertEqual(gate.metrics.triggered_frames, 1)
        self.assertEqual(gate.metrics.skip_rate, 0.5)
        self.assertEqual(gate.metrics.trigger_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
