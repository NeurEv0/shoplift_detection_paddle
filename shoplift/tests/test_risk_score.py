from __future__ import annotations

import unittest

from shoplift.core.types import RelationEvidence
from shoplift.events.state_machine import ActionStateSnapshot
from shoplift.rules.risk_score import RiskScorer


def _evidence(
    relation_type: str,
    frame_id: int,
    tags: tuple[str, ...],
    *,
    item_track_id: str = "item-1",
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
        item_track_id=item_track_id,
        container_track_id=container_track_id,
        metadata=metadata or {},
    )


def _snapshot(
    *,
    event_type: str,
    reason_tags: tuple[str, ...],
    evidence: tuple[RelationEvidence, ...],
    metadata: dict[str, object] | None = None,
) -> ActionStateSnapshot:
    return ActionStateSnapshot(
        state="confirmed_risk_event",
        person_track_id="person-1",
        item_track_id="item-1",
        frame_id=max(item.frame_id for item in evidence),
        timestamp_ms=max(item.timestamp_ms for item in evidence),
        reason_tags=reason_tags,
        evidence=evidence,
        suggested_event_type=event_type,
        metadata=metadata or {},
    )


class RiskScoreTest(unittest.TestCase):
    def test_bag_concealment_scores_high(self) -> None:
        scorer = RiskScorer()
        snapshot = _snapshot(
            event_type="bag_concealment",
            reason_tags=(
                "hand_item_contact",
                "item_owned_by_person",
                "entered_private_container",
                "after_private_container_entry",
                "item_disappeared",
            ),
            evidence=(
                _evidence("hand_item_contact", 1, ("hand_item_overlap", "temporal_consistent")),
                _evidence("item_follow_person", 2, ("item_owned_by_person", "follow_motion_aligned")),
                _evidence(
                    "item_enter_container",
                    3,
                    ("entered_private_container", "entry_temporal_consistent"),
                    container_track_id="bag-1",
                    metadata={"container_kind": "private"},
                ),
                _evidence(
                    "item_disappeared_after_entry",
                    4,
                    ("item_disappeared", "after_container_entry", "after_private_container_entry"),
                    container_track_id="bag-1",
                ),
            ),
        )

        result = scorer.score_snapshot(snapshot, event_type="bag_concealment")

        self.assertEqual(result.risk_level, "high")
        self.assertGreaterEqual(result.risk_score, 0.75)
        self.assertIn("entered_private_container", result.reason_tags)

    def test_normal_container_is_downgraded(self) -> None:
        scorer = RiskScorer()
        snapshot = _snapshot(
            event_type="normal_container_placement",
            reason_tags=("entered_normal_container", "entry_temporal_consistent"),
            evidence=(
                _evidence(
                    "item_enter_container",
                    1,
                    ("entered_normal_container", "entry_temporal_consistent"),
                    container_track_id="basket-1",
                    metadata={"is_normal_container": True, "container_kind": "normal"},
                ),
            ),
            metadata={"is_normal_container": True, "container_kind": "normal"},
        )

        result = scorer.score_snapshot(snapshot, event_type="normal_container_placement")

        self.assertEqual(result.risk_level, "low")
        self.assertLessEqual(result.risk_score, 0.44)

    def test_low_visibility_is_capped(self) -> None:
        scorer = RiskScorer()
        snapshot = _snapshot(
            event_type="bag_concealment",
            reason_tags=("hand_item_contact", "entered_private_container", "item_disappeared"),
            evidence=(
                _evidence("hand_item_contact", 1, ("hand_item_overlap",), metadata={"occlusion_ratio": 0.8}),
                _evidence(
                    "item_disappeared_after_entry",
                    2,
                    ("item_disappeared", "after_private_container_entry"),
                    container_track_id="bag-1",
                    metadata={"visibility": "severe_occlusion"},
                ),
            ),
        )

        result = scorer.score_snapshot(snapshot, event_type="bag_concealment")

        self.assertIn("low_visibility", result.reason_tags)
        self.assertLessEqual(result.risk_score, 0.55)
        self.assertIn(result.risk_level, {"low", "medium"})

    def test_bulk_pickup_to_bag_reaches_high(self) -> None:
        scorer = RiskScorer()
        snapshot = _snapshot(
            event_type="bulk_pickup_to_bag",
            reason_tags=(
                "hand_item_contact",
                "item_owned_by_person",
                "entered_private_container",
                "item_disappeared",
                "bulk_pickup",
            ),
            evidence=(
                _evidence("hand_item_contact", 1, ("hand_item_overlap", "temporal_consistent")),
                _evidence("item_follow_person", 2, ("item_owned_by_person", "follow_motion_aligned")),
                _evidence(
                    "item_enter_container",
                    3,
                    ("entered_private_container", "entry_temporal_consistent"),
                    container_track_id="bag-1",
                    metadata={"container_kind": "private"},
                ),
            ),
            metadata={"bulk_item_count": 3, "bulk_pickup": True, "container_track_id": "bag-1"},
        )

        result = scorer.score_snapshot(snapshot, event_type="bulk_pickup_to_bag")

        self.assertEqual(result.risk_level, "high")
        self.assertGreaterEqual(result.risk_score, 0.75)
        self.assertIn("bulk_pickup", result.reason_tags)


if __name__ == "__main__":
    unittest.main()
