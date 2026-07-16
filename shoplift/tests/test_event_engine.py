from __future__ import annotations

import unittest

from shoplift.core.types import RelationEvidence
from shoplift.events.event_engine import ShopliftingEventEngine
from shoplift.events.event_schema import validate_risk_event_payload
from shoplift.events.state_machine import CONFIRMED_RISK_EVENT


def _evidence(
    relation_type: str,
    frame_id: int,
    tags: tuple[str, ...],
    *,
    score: float = 0.9,
    container_track_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationEvidence:
    return RelationEvidence(
        relation_type=relation_type,
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        score=score,
        reason_tags=tags,
        person_track_id="person-1",
        hand_track_id="hand-1-right" if relation_type == "hand_item_contact" else None,
        item_track_id="item-1",
        container_track_id=container_track_id,
        metadata=metadata or {},
    )


class EventEngineTest(unittest.TestCase):
    def test_engine_emits_risk_event_for_confirmed_concealment(self) -> None:
        engine = ShopliftingEventEngine()

        engine.process_relations(
            relations=(
                _evidence(
                    "hand_item_contact",
                    1,
                    ("hand_item_overlap", "temporal_consistent", "motion_aligned"),
                ),
            ),
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
        )
        engine.process_relations(
            relations=(
                _evidence(
                    "item_follow_person",
                    2,
                    ("item_owned_by_person", "follow_motion_aligned"),
                ),
            ),
            frame_id=2,
            timestamp_ms=66,
            camera_id="camera-1",
        )
        engine.process_relations(
            relations=(
                _evidence(
                    "item_enter_container",
                    3,
                    ("entered_private_container", "entry_temporal_consistent"),
                    container_track_id="bag-1",
                    metadata={"is_normal_container": False, "container_kind": "private"},
                ),
            ),
            frame_id=3,
            timestamp_ms=99,
            camera_id="camera-1",
        )
        result = engine.process_relations(
            relations=(
                _evidence(
                    "item_disappeared_after_entry",
                    4,
                    ("item_disappeared", "after_container_entry", "after_private_container_entry"),
                    score=0.95,
                    container_track_id="bag-1",
                ),
            ),
            frame_id=4,
            timestamp_ms=132,
            camera_id="camera-1",
        )

        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event.event_type, "bag_concealment")
        self.assertEqual(event.risk_level, "high")
        self.assertIn("item_disappeared", event.reason_tags)
        self.assertEqual(result.states[-1].state, CONFIRMED_RISK_EVENT)
        self.assertEqual(validate_risk_event_payload(event.to_dict() | {"schema_version": "shoplift.risk_event.v1"}), [])

    def test_engine_does_not_emit_for_normal_container(self) -> None:
        engine = ShopliftingEventEngine()

        result = engine.process_relations(
            relations=(
                _evidence(
                    "item_enter_container",
                    1,
                    ("entered_normal_container", "entry_temporal_consistent"),
                    score=0.4,
                    container_track_id="basket-1",
                    metadata={"is_normal_container": True, "container_kind": "normal"},
                ),
                _evidence(
                    "item_disappeared_after_entry",
                    2,
                    ("item_disappeared", "after_container_entry", "normal_container_exempted"),
                    score=0.25,
                    container_track_id="basket-1",
                    metadata={"is_normal_container": True, "container_kind": "normal"},
                ),
            ),
            frame_id=2,
            timestamp_ms=66,
            camera_id="camera-1",
        )

        self.assertEqual(result.events, ())

    def test_engine_does_not_emit_for_contact_only(self) -> None:
        engine = ShopliftingEventEngine()

        result = engine.process_relations(
            relations=(
                _evidence(
                    "hand_item_contact",
                    1,
                    ("hand_item_overlap", "temporal_consistent"),
                ),
            ),
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
        )

        self.assertEqual(result.events, ())


if __name__ == "__main__":
    unittest.main()
