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
    item_track_id: str = "item-1",
    score: float = 0.9,
    container_track_id: str | None = None,
    hand_track_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationEvidence:
    return RelationEvidence(
        relation_type=relation_type,
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        score=score,
        reason_tags=tags,
        person_track_id="person-1",
        hand_track_id=hand_track_id or ("hand-1-right" if relation_type == "hand_item_contact" else None),
        item_track_id=item_track_id,
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

    def test_engine_emits_clothing_concealment(self) -> None:
        engine = ShopliftingEventEngine()

        engine.process_relations(
            relations=(
                _evidence("hand_item_contact", 1, ("hand_item_overlap", "temporal_consistent"), item_track_id="item-2"),
                _evidence("item_follow_person", 1, ("item_owned_by_person",), item_track_id="item-2"),
            ),
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
        )
        engine.process_relations(
            relations=(
                _evidence(
                    "item_enter_container",
                    2,
                    ("entered_clothing_region", "entry_temporal_consistent"),
                    item_track_id="item-2",
                    metadata={"container_kind": "clothing"},
                ),
            ),
            frame_id=2,
            timestamp_ms=66,
            camera_id="camera-1",
        )
        result = engine.process_relations(
            relations=(
                _evidence(
                    "item_disappeared_after_entry",
                    3,
                    ("item_disappeared", "after_clothing_region_entry"),
                    item_track_id="item-2",
                ),
            ),
            frame_id=3,
            timestamp_ms=99,
            camera_id="camera-1",
        )

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_type, "clothing_concealment")
        self.assertEqual(result.events[0].risk_level, "high")

    def test_engine_emits_special_container_concealment(self) -> None:
        engine = ShopliftingEventEngine()

        engine.process_relations(
            relations=(
                _evidence("hand_item_contact", 1, ("hand_item_overlap", "temporal_consistent"), item_track_id="item-3"),
                _evidence("item_follow_person", 1, ("item_owned_by_person",), item_track_id="item-3"),
            ),
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
        )
        engine.process_relations(
            relations=(
                _evidence(
                    "item_enter_container",
                    2,
                    ("entered_special_container", "entry_temporal_consistent"),
                    item_track_id="item-3",
                    container_track_id="stroller-1",
                    metadata={"container_kind": "special"},
                ),
            ),
            frame_id=2,
            timestamp_ms=66,
            camera_id="camera-1",
        )
        result = engine.process_relations(
            relations=(
                _evidence(
                    "item_disappeared_after_entry",
                    3,
                    ("item_disappeared", "after_special_container_entry"),
                    item_track_id="item-3",
                    container_track_id="stroller-1",
                ),
            ),
            frame_id=3,
            timestamp_ms=99,
            camera_id="camera-1",
        )

        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].event_type, "special_container_concealment")
        self.assertEqual(result.events[0].risk_level, "high")

    def test_engine_emits_near_body_suspicious_for_contact_plus_follow(self) -> None:
        engine = ShopliftingEventEngine()

        first = engine.process_relations(
            relations=(
                _evidence(
                    "hand_item_contact",
                    1,
                    ("hand_item_overlap", "temporal_consistent"),
                    item_track_id="item-4",
                ),
            ),
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
        )
        second = engine.process_relations(
            relations=(
                _evidence(
                    "item_follow_person",
                    2,
                    ("item_owned_by_person", "follow_motion_aligned"),
                    item_track_id="item-4",
                ),
            ),
            frame_id=2,
            timestamp_ms=66,
            camera_id="camera-1",
        )

        self.assertEqual(first.events, ())
        self.assertEqual(len(second.events), 1)
        self.assertEqual(second.events[0].event_type, "near_body_suspicious")
        self.assertIn(second.events[0].risk_level, {"low", "medium"})

    def test_engine_emits_bulk_pickup_to_bag_for_multiple_items(self) -> None:
        engine = ShopliftingEventEngine()

        for frame_id in (1, 2, 3, 4):
            relations = []
            for item_index in (1, 2, 3):
                item_track_id = f"item-{item_index}"
                if frame_id == 1:
                    relations.append(
                        _evidence(
                            "hand_item_contact",
                            frame_id,
                            ("hand_item_overlap", "temporal_consistent"),
                            item_track_id=item_track_id,
                        )
                    )
                elif frame_id == 2:
                    relations.append(
                        _evidence(
                            "item_follow_person",
                            frame_id,
                            ("item_owned_by_person", "follow_motion_aligned"),
                            item_track_id=item_track_id,
                        )
                    )
                elif frame_id == 3:
                    relations.append(
                        _evidence(
                            "item_enter_container",
                            frame_id,
                            ("entered_private_container", "entry_temporal_consistent"),
                            item_track_id=item_track_id,
                            container_track_id="bag-bulk-1",
                            metadata={"is_normal_container": False, "container_kind": "private"},
                        )
                    )
                else:
                    relations.append(
                        _evidence(
                            "item_disappeared_after_entry",
                            frame_id,
                            ("item_disappeared", "after_container_entry", "after_private_container_entry"),
                            item_track_id=item_track_id,
                            container_track_id="bag-bulk-1",
                            score=0.95,
                        )
                    )

            result = engine.process_relations(
                relations=tuple(relations),
                frame_id=frame_id,
                timestamp_ms=frame_id * 33,
                camera_id="camera-1",
            )

        self.assertTrue(any(event.event_type == "bulk_pickup_to_bag" for event in result.events))

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
