"""Shoplifting event engine for P1 relation and state-machine outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.events.state_machine import (
    ActionStateSnapshot,
    CONFIRMED_RISK_EVENT,
    SuspiciousActionStateMachine,
)
from shoplift.tracking.association import (
    AssociationConfig,
    AssociationFrame,
    AssociationResult,
    ShopliftingRelationAssociator,
)


@dataclass(frozen=True)
class EventEngineResult:
    """Structured output produced by one event-engine update."""

    frame_id: int
    timestamp_ms: int
    camera_id: str
    relations: tuple[RelationEvidence, ...]
    states: tuple[ActionStateSnapshot, ...]
    events: tuple[RiskEvent, ...]
    metadata: dict[str, object] = field(default_factory=dict)


class ShopliftingEventEngine:
    """Convert frame-level visual structures into reviewable risk events."""

    def __init__(
        self,
        *,
        association_config: AssociationConfig | None = None,
        relation_associator: ShopliftingRelationAssociator | None = None,
        state_machine: SuspiciousActionStateMachine | None = None,
    ) -> None:
        self.association_config = association_config or AssociationConfig()
        self.relation_associator = relation_associator or ShopliftingRelationAssociator(self.association_config)
        self.state_machine = state_machine or SuspiciousActionStateMachine()
        self._emitted_keys: set[tuple[str, str, str]] = set()

    def process_frame(self, frame: AssociationFrame) -> EventEngineResult:
        association_result = self.relation_associator.update(frame)
        return self.process_relations(
            relations=association_result.relations,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            camera_id=frame.camera_id,
            association_result=association_result,
        )

    def process_relations(
        self,
        *,
        relations: Sequence[RelationEvidence],
        frame_id: int,
        timestamp_ms: int,
        camera_id: str,
        association_result: AssociationResult | None = None,
    ) -> EventEngineResult:
        relation_tuple = tuple(relations)
        states = self.state_machine.update(
            relation_tuple,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )
        events: list[RiskEvent] = []
        for snapshot in states:
            if snapshot.state != CONFIRMED_RISK_EVENT:
                continue
            key = (snapshot.person_track_id, snapshot.item_track_id, snapshot.suggested_event_type or "")
            if key in self._emitted_keys:
                continue
            event = self._risk_event_from_snapshot(snapshot, camera_id)
            if event is None:
                continue
            events.append(event)
            self._emitted_keys.add(key)
            self.state_machine.mark_event_emitted(snapshot)

        relation_counts = _count_relations(relation_tuple)
        metadata = {
            "relation_count": len(relation_tuple),
            "state_count": len(states),
            "event_count": len(events),
            "relation_counts": relation_counts,
        }
        if association_result is not None:
            metadata["association"] = association_result.metadata

        return EventEngineResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            camera_id=camera_id,
            relations=relation_tuple,
            states=states,
            events=tuple(events),
            metadata=metadata,
        )

    def _risk_event_from_snapshot(
        self,
        snapshot: ActionStateSnapshot,
        camera_id: str,
    ) -> RiskEvent | None:
        if not snapshot.evidence:
            return None
        reason_tags = tuple(sorted(set(snapshot.reason_tags)))
        if "normal_container_exempted" in reason_tags or "entered_normal_container" in reason_tags:
            return None

        risk_score = _score_snapshot(snapshot)
        if "low_visibility" in reason_tags or "possible_occlusion" in reason_tags:
            risk_score = min(risk_score, 0.55)
        risk_level = _risk_level(risk_score)

        event_type = snapshot.suggested_event_type or "near_body_suspicious"
        event_id = (
            f"evt-{camera_id}-{snapshot.person_track_id}-"
            f"{snapshot.item_track_id}-{event_type}-{snapshot.frame_id}"
        )
        return RiskEvent(
            event_id=event_id,
            camera_id=camera_id,
            timestamp_ms=snapshot.timestamp_ms,
            start_timestamp_ms=_first_timestamp(snapshot.evidence),
            end_timestamp_ms=snapshot.timestamp_ms,
            person_track_id=snapshot.person_track_id,
            event_type=event_type,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=_mean_score(snapshot.evidence),
            reason_tags=reason_tags,
            evidence=tuple(snapshot.evidence),
            metadata={
                "state": snapshot.state,
                "item_track_id": snapshot.item_track_id,
                "source": "shoplifting_event_engine_p1",
            },
        )


def _score_snapshot(snapshot: ActionStateSnapshot) -> float:
    relation_scores = {evidence.relation_type: evidence.score for evidence in snapshot.evidence}
    base = 0.15
    if "hand_item_contact" in relation_scores:
        base += 0.18 * relation_scores["hand_item_contact"]
    if "item_follow_person" in relation_scores:
        base += 0.16 * relation_scores["item_follow_person"]
    if "item_enter_container" in relation_scores:
        base += 0.26 * relation_scores["item_enter_container"]
    if "item_disappeared_after_entry" in relation_scores:
        base += 0.32 * relation_scores["item_disappeared_after_entry"]

    reason_tags = set(snapshot.reason_tags)
    if "entered_special_container" in reason_tags or "entered_clothing_region" in reason_tags:
        base += 0.06
    if "after_private_container_entry" in reason_tags or "after_special_container_entry" in reason_tags:
        base += 0.05
    if len(reason_tags) >= 4:
        base += 0.05
    return max(0.0, min(1.0, base))


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _mean_score(evidences: Sequence[RelationEvidence]) -> float:
    if not evidences:
        return 0.0
    return sum(evidence.score for evidence in evidences) / len(evidences)


def _first_timestamp(evidences: Sequence[RelationEvidence]) -> int | None:
    if not evidences:
        return None
    return min(evidence.timestamp_ms for evidence in evidences)


def _count_relations(relations: Sequence[RelationEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relation in relations:
        counts[relation.relation_type] = counts.get(relation.relation_type, 0) + 1
    return counts


__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "EventEngineResult",
    "RelationEvidence",
    "RiskEvent",
    "ShopliftingEventEngine",
]
