from __future__ import annotations

import unittest

from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.rules.validators import RiskRuleValidator


def _evidence(
    relation_type: str,
    frame_id: int,
    tags: tuple[str, ...],
    *,
    item_track_id: str = "item-1",
    container_track_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RelationEvidence:
    return RelationEvidence(
        relation_type=relation_type,
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        score=0.9,
        reason_tags=tags,
        person_track_id="person-1",
        hand_track_id="hand-1-right" if relation_type == "hand_item_contact" else None,
        item_track_id=item_track_id,
        container_track_id=container_track_id,
        metadata=metadata or {},
    )


def _event(
    *,
    event_type: str,
    risk_score: float,
    risk_level: str,
    reason_tags: tuple[str, ...],
    evidence: tuple[RelationEvidence, ...],
    metadata: dict[str, object] | None = None,
) -> RiskEvent:
    return RiskEvent(
        event_id="evt-1",
        camera_id="camera-1",
        timestamp_ms=99,
        person_track_id="person-1",
        event_type=event_type,
        risk_score=risk_score,
        risk_level=risk_level,
        reason_tags=reason_tags,
        evidence=evidence,
        metadata=metadata or {},
    )


class RiskRuleValidatorTest(unittest.TestCase):
    def test_high_risk_requires_two_reason_tags(self) -> None:
        validator = RiskRuleValidator()
        event = _event(
            event_type="bag_concealment",
            risk_score=0.82,
            risk_level="high",
            reason_tags=("item_disappeared",),
            evidence=(
                _evidence(
                    "item_disappeared_after_entry",
                    1,
                    ("item_disappeared", "after_private_container_entry"),
                    container_track_id="bag-1",
                ),
            ),
        )

        result = validator.apply(event)

        self.assertFalse(result.is_valid)
        self.assertIn("high_risk_requires_multiple_reason_tags", {violation.code for violation in result.violations})
        self.assertIn(result.event.risk_level, {"low", "medium"})
        self.assertLess(result.event.risk_score, 0.75)

    def test_single_frame_contact_is_downgraded(self) -> None:
        validator = RiskRuleValidator()
        event = _event(
            event_type="near_body_suspicious",
            risk_score=0.8,
            risk_level="high",
            reason_tags=("hand_item_contact", "item_owned_by_person"),
            evidence=(
                _evidence(
                    "hand_item_contact",
                    1,
                    ("hand_item_overlap",),
                    metadata={"contact_frames": 1},
                ),
            ),
        )

        result = validator.apply(event)

        self.assertIn("single_frame_contact_must_not_be_high", {violation.code for violation in result.violations})
        self.assertEqual(result.event.risk_level, "low")
        self.assertLessEqual(result.event.risk_score, 0.44)

    def test_low_visibility_adds_tag_and_caps_to_medium(self) -> None:
        validator = RiskRuleValidator()
        event = _event(
            event_type="bag_concealment",
            risk_score=0.86,
            risk_level="high",
            reason_tags=("hand_item_contact", "entered_private_container", "item_disappeared"),
            evidence=(
                _evidence(
                    "item_enter_container",
                    1,
                    ("entered_private_container", "entry_temporal_consistent"),
                    container_track_id="bag-1",
                    metadata={"visibility": "severe_occlusion"},
                ),
                _evidence(
                    "item_disappeared_after_entry",
                    2,
                    ("item_disappeared", "after_private_container_entry"),
                    container_track_id="bag-1",
                    metadata={"occlusion_ratio": 0.85},
                ),
            ),
        )

        result = validator.apply(event)

        self.assertIn("low_visibility_must_be_downgraded", {violation.code for violation in result.violations})
        self.assertIn("low_visibility", result.event.reason_tags)
        self.assertIn(result.event.risk_level, {"low", "medium"})
        self.assertLessEqual(result.event.risk_score, 0.55)

    def test_normal_container_is_not_high(self) -> None:
        validator = RiskRuleValidator()
        event = _event(
            event_type="bag_concealment",
            risk_score=0.9,
            risk_level="high",
            reason_tags=("entered_normal_container", "item_disappeared"),
            evidence=(
                _evidence(
                    "item_enter_container",
                    1,
                    ("entered_normal_container", "entry_temporal_consistent"),
                    container_track_id="basket-1",
                    metadata={"is_normal_container": True, "container_kind": "normal"},
                ),
            ),
        )

        result = validator.apply(event)

        self.assertIn("normal_container_must_not_be_high", {violation.code for violation in result.violations})
        self.assertIn(result.event.risk_level, {"low", "medium"})
        self.assertLess(result.event.risk_score, 0.75)


if __name__ == "__main__":
    unittest.main()
