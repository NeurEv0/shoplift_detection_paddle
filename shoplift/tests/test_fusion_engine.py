"""Unit tests for the PP-TSM open × pipeline risk fusion engine."""

from __future__ import annotations

import unittest

from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.fusion.fusion_engine import (
    OPEN_PACKAGING_TAG,
    PACKAGE_OPENING_EVENT_TYPE,
    FusionConfig,
    FusionEngine,
    ProductFrameEvidence,
    ProductGate,
)
from shoplift.pptsm_open.types import OpenPackagingEvent, OpenWindow


def _win(start: int, end: int, prob: float, frame_id: int | None = None) -> OpenWindow:
    return OpenWindow(start_timestamp_ms=start, end_timestamp_ms=end,
                      open_prob=prob, frame_id=frame_id)


def _open_event(
    camera: str = "cam",
    person: str = "p1",
    starts: tuple[int, ...] = (1000, 1100, 1200),
    probs: tuple[float, ...] = (0.7, 0.8, 0.9),
) -> OpenPackagingEvent:
    windows = tuple(
        _win(start, start + 100, prob, index)
        for index, (start, prob) in enumerate(zip(starts, probs))
    )
    return OpenPackagingEvent(camera, person, windows)


def _evidence(frames: list[tuple[int, int, bool]]) -> list[ProductFrameEvidence]:
    return [ProductFrameEvidence(frame_id=frame_id, timestamp_ms=ts, is_holding_product=flag)
            for frame_id, ts, flag in frames]


def _confirmed_evidence(
    person: str = "p1",
    n: int = 14,
    start_ts: int = 1000,
    step_ms: int = 10,
) -> dict[str, list[ProductFrameEvidence]]:
    return {person: _evidence([(i, start_ts + i * step_ms, True) for i in range(n)])}


def _pipeline_event(
    person: str = "p1",
    camera: str = "cam",
    start_ms: int = 1000,
    end_ms: int = 1300,
    level: str = "medium",
    score: float = 0.5,
    event_type: str = "bag_concealment",
) -> RiskEvent:
    relation = RelationEvidence(
        relation_type="item_enter_container",
        frame_id=0,
        timestamp_ms=start_ms,
        score=0.6,
        reason_tags=("entered_private_container",),
        person_track_id=person,
    )
    return RiskEvent(
        event_id=f"evt-{camera}-{person}-{event_type}-{start_ms}",
        camera_id=camera,
        timestamp_ms=end_ms,
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
        person_track_id=person,
        event_type=event_type,
        risk_score=score,
        risk_level=level,  # type: ignore[arg-type]
        reason_tags=("entered_private_container",),
        evidence=(relation,),
    )


class ProductGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductGate(FusionConfig(min_product_confirm_frames=3))

    def test_default_min_product_confirm_frames_is_14(self) -> None:
        self.assertEqual(FusionConfig().min_product_confirm_frames, 14)

    def test_confirms_consecutive_three(self) -> None:
        evidence = _evidence([(0, 1000, True), (1, 1040, True), (2, 1080, True)])
        self.assertTrue(self.gate.confirms(_open_event(), evidence))

    def test_rejects_short_run(self) -> None:
        evidence = _evidence([(0, 1000, True), (1, 1040, True), (2, 1080, False)])
        self.assertFalse(self.gate.confirms(_open_event(), evidence))

    def test_rejects_discontinuous(self) -> None:
        evidence = _evidence([(0, 1000, True), (1, 1040, False), (2, 1080, True)])
        self.assertFalse(self.gate.confirms(_open_event(), evidence))

    def test_ignores_out_of_window(self) -> None:
        evidence = _evidence([(0, 500, True), (1, 540, True), (2, 580, True)])
        self.assertFalse(self.gate.confirms(_open_event(), evidence))

    def test_gap_in_frame_ids_breaks_run(self) -> None:
        # frames 0,1 are True but frame 2 is missing (occlusion): not "consecutive 3"
        evidence = _evidence([(0, 1000, True), (1, 1040, True), (3, 1120, True)])
        self.assertFalse(self.gate.confirms(_open_event(), evidence))


class FusionEngineTest(unittest.TestCase):
    def test_no_escalation_pipeline_unchanged(self) -> None:
        engine = FusionEngine()
        pipeline_event = _pipeline_event(level="medium", score=0.5)
        open_event = _open_event()
        result = engine.fuse([pipeline_event], [open_event], _confirmed_evidence())

        pipe_out = [event for event in result.events if event.event_type == "bag_concealment"]
        self.assertEqual(len(pipe_out), 1)
        self.assertEqual(pipe_out[0].risk_level, "medium")
        self.assertNotIn(OPEN_PACKAGING_TAG, pipe_out[0].reason_tags)
        self.assertEqual(result.escalated_event_ids, ())
        self.assertEqual(len(result.package_opening_events), 1)

    def test_standalone_package_opening(self) -> None:
        engine = FusionEngine()
        result = engine.fuse([], [_open_event()], _confirmed_evidence())

        self.assertEqual(len(result.package_opening_events), 1)
        out = result.package_opening_events[0]
        self.assertEqual(out.event_type, PACKAGE_OPENING_EVENT_TYPE)
        self.assertEqual(out.risk_level, "high")
        self.assertEqual(out.person_track_id, "p1")
        self.assertIn(OPEN_PACKAGING_TAG, out.reason_tags)
        self.assertIn("pptsm_open", out.metadata)
        self.assertEqual(len(result.events), 1)

    def test_product_gate_discards_without_evidence(self) -> None:
        engine = FusionEngine()
        result = engine.fuse([], [_open_event()], {})
        self.assertEqual(result.package_opening_events, ())
        self.assertEqual(len(result.discarded_open_events), 1)

    def test_product_gate_rejects_short_run(self) -> None:
        engine = FusionEngine()
        evidence = {"p1": _evidence([(i, 1000 + i * 10, True) for i in range(13)])}
        result = engine.fuse([], [_open_event()], evidence)
        self.assertEqual(result.package_opening_events, ())
        self.assertEqual(len(result.discarded_open_events), 1)

    def test_merge_same_track_adjacent(self) -> None:
        engine = FusionEngine()
        e1 = _open_event(starts=(1000, 1100, 1200))          # 1000-1300
        e2 = _open_event(starts=(1300, 1400, 1500))          # 1300-1600
        evidence = {
            "p1": _evidence(
                [(i, 1000 + i * 10, True) for i in range(14)]
                + [(100 + i, 1300 + i * 10, True) for i in range(14)]
            )
        }
        result = engine.fuse([], [e1, e2], evidence)

        self.assertEqual(len(result.package_opening_events), 1)
        out = result.package_opening_events[0]
        self.assertEqual(out.start_timestamp_ms, 1000)
        self.assertEqual(out.end_timestamp_ms, 1600)

    def test_merge_different_track_spatial_overlap(self) -> None:
        engine = FusionEngine()
        e1 = _open_event(person="p1", starts=(1000, 1100, 1200))
        e2 = _open_event(person="p2", starts=(1300, 1400, 1500))
        evidence = {
            "p1": _evidence([(i, 1000 + i * 10, True) for i in range(14)]),
            "p2": _evidence([(100 + i, 1300 + i * 10, True) for i in range(14)]),
        }
        boxes = {
            "p1": [(0, 1300, [0.0, 0.0, 100.0, 200.0])],
            "p2": [(0, 1300, [0.0, 0.0, 100.0, 200.0])],
        }
        result = engine.fuse([], [e1, e2], evidence, boxes)

        self.assertEqual(len(result.package_opening_events), 1)
        out = result.package_opening_events[0]
        self.assertEqual(out.metadata["merged_source_track_ids"], ["p1", "p2"])

    def test_no_merge_different_track_no_spatial_overlap(self) -> None:
        engine = FusionEngine()
        e1 = _open_event(person="p1", starts=(1000, 1100, 1200))
        e2 = _open_event(person="p2", starts=(1300, 1400, 1500))
        evidence = {
            "p1": _evidence([(i, 1000 + i * 10, True) for i in range(14)]),
            "p2": _evidence([(100 + i, 1300 + i * 10, True) for i in range(14)]),
        }
        boxes = {
            "p1": [(0, 1300, [0.0, 0.0, 100.0, 200.0])],
            "p2": [(0, 1300, [1000.0, 0.0, 1100.0, 200.0])],
        }
        result = engine.fuse([], [e1, e2], evidence, boxes)

        self.assertEqual(len(result.package_opening_events), 2)

    def test_no_merge_time_gap_beyond_threshold(self) -> None:
        engine = FusionEngine(FusionConfig(merge_temporal_gap_ms=100))
        e1 = _open_event(starts=(1000, 1100, 1200))          # ends 1300
        e2 = _open_event(starts=(2000, 2100, 2200))          # starts 2000, gap 700
        evidence = {
            "p1": _evidence(
                [(i, 1000 + i * 10, True) for i in range(14)]
                + [(100 + i, 2000 + i * 10, True) for i in range(14)]
            )
        }
        result = engine.fuse([], [e1, e2], evidence)

        self.assertEqual(len(result.package_opening_events), 2)


class SchemaCompatibilityTest(unittest.TestCase):
    """Fusion outputs must round-trip through the P0 RiskEvent contract."""

    def _payload(self, event: RiskEvent) -> dict:
        from shoplift.events.event_schema import risk_event_to_payload

        return risk_event_to_payload(event)

    def _assert_valid(self, event: RiskEvent) -> None:
        from shoplift.events.event_schema import validate_risk_event_payload

        errors = validate_risk_event_payload(self._payload(event))
        self.assertEqual(errors, [], msg=f"schema errors: {errors}")

    def test_standalone_package_opening_payload_validates(self) -> None:
        engine = FusionEngine()
        result = engine.fuse([], [_open_event()], _confirmed_evidence())
        self.assertEqual(len(result.package_opening_events), 1)
        self._assert_valid(result.package_opening_events[0])

    def test_unchanged_pipeline_payload_validates(self) -> None:
        engine = FusionEngine()
        pipeline_event = _pipeline_event(person="p2", level="medium")
        result = engine.fuse([pipeline_event], [_open_event(person="p1")],
                             _confirmed_evidence("p1"))
        unchanged = [event for event in result.events if event.person_track_id == "p2"]
        self.assertEqual(len(unchanged), 1)
        self._assert_valid(unchanged[0])

    def test_merged_package_opening_payload_validates(self) -> None:
        engine = FusionEngine()
        e1 = _open_event(starts=(1000, 1100, 1200))
        e2 = _open_event(starts=(1300, 1400, 1500))
        evidence = {
            "p1": _evidence(
                [(i, 1000 + i * 10, True) for i in range(14)]
                + [(100 + i, 1300 + i * 10, True) for i in range(14)]
            )
        }
        result = engine.fuse([], [e1, e2], evidence)
        self.assertEqual(len(result.package_opening_events), 1)
        self._assert_valid(result.package_opening_events[0])


if __name__ == "__main__":
    unittest.main()
