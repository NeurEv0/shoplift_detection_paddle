from __future__ import annotations

import unittest

from shoplift.core.types import RelationEvidence
from shoplift.events.state_machine import (
    CONFIRMED_RISK_EVENT,
    ITEM_PICKED,
    RESOLVED_OR_DOWNGRADED,
    SUSPECTED_CONCEALMENT,
    SuspiciousActionStateMachine,
)


def _evidence(
    relation_type: str,
    frame_id: int,
    tags: tuple[str, ...],
    *,
    score: float = 0.8,
    item_track_id: str = "item-1",
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
        item_track_id=item_track_id,
        container_track_id=container_track_id,
        metadata=metadata or {},
    )


class StateMachineTest(unittest.TestCase):
    def test_concealment_path_reaches_confirmed_risk_event(self) -> None:
        machine = SuspiciousActionStateMachine()

        picked = machine.update(
            (
                _evidence(
                    "hand_item_contact",
                    1,
                    ("hand_item_overlap", "temporal_consistent"),
                ),
            ),
            frame_id=1,
            timestamp_ms=33,
        )
        suspected = machine.update(
            (
                _evidence(
                    "item_enter_container",
                    2,
                    ("entered_private_container", "entry_temporal_consistent"),
                    container_track_id="bag-1",
                    metadata={"is_normal_container": False},
                ),
            ),
            frame_id=2,
            timestamp_ms=66,
        )
        confirmed = machine.update(
            (
                _evidence(
                    "item_disappeared_after_entry",
                    3,
                    ("item_disappeared", "after_container_entry", "after_private_container_entry"),
                    score=0.95,
                    container_track_id="bag-1",
                ),
            ),
            frame_id=3,
            timestamp_ms=99,
        )

        self.assertEqual(picked[-1].state, ITEM_PICKED)
        self.assertEqual(suspected[-1].state, SUSPECTED_CONCEALMENT)
        self.assertEqual(confirmed[-1].state, CONFIRMED_RISK_EVENT)
        self.assertEqual(confirmed[-1].suggested_event_type, "bag_concealment")
        self.assertIn("item_disappeared", confirmed[-1].reason_tags)

    def test_normal_container_entry_resolves_or_downgrades(self) -> None:
        machine = SuspiciousActionStateMachine()

        snapshots = machine.update(
            (
                _evidence(
                    "item_enter_container",
                    1,
                    ("entered_normal_container", "entry_temporal_consistent"),
                    container_track_id="basket-1",
                    metadata={"is_normal_container": True},
                ),
            ),
            frame_id=1,
            timestamp_ms=33,
        )

        self.assertEqual(snapshots[-1].state, RESOLVED_OR_DOWNGRADED)
        self.assertEqual(snapshots[-1].suggested_event_type, "normal_container_placement")

    def test_bulk_pickup_tags_suggest_bulk_event(self) -> None:
        machine = SuspiciousActionStateMachine()

        snapshots = machine.update(
            (
                _evidence(
                    "item_enter_container",
                    1,
                    ("entered_private_container", "bulk_pickup"),
                    container_track_id="bag-1",
                    metadata={"is_normal_container": False, "container_kind": "private"},
                ),
            ),
            frame_id=1,
            timestamp_ms=33,
        )

        self.assertEqual(snapshots[-1].suggested_event_type, "bulk_pickup_to_bag")


if __name__ == "__main__":
    unittest.main()
