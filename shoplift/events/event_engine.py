"""Shoplifting event engine for P1 relation and state-machine outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.events.state_machine import (
    ActionStateSnapshot,
    CONFIRMED_RISK_EVENT,
    NEAR_BODY_OR_CONTAINER,
    ITEM_PICKED,
    SUSPECTED_CONCEALMENT,
    SuspiciousActionStateMachine,
)
from shoplift.rules.risk_score import RiskScorer
from shoplift.rules.validators import RiskRuleValidator
from shoplift.tracking.association import (
    AssociationConfig,
    AssociationFrame,
    AssociationResult,
    ShopliftingRelationAssociator,
)


CONCEALMENT_EVENT_TYPES = {
    "bag_concealment",
    "clothing_concealment",
    "special_container_concealment",
}


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


@dataclass
class _BulkWindowEntry:
    snapshot: ActionStateSnapshot
    timestamp_ms: int


class ShopliftingEventEngine:
    """Convert frame-level visual structures into reviewable risk events."""

    def __init__(
        self,
        *,
        association_config: AssociationConfig | None = None,
        relation_associator: ShopliftingRelationAssociator | None = None,
        state_machine: SuspiciousActionStateMachine | None = None,
        risk_scorer: RiskScorer | None = None,
        rule_validator: RiskRuleValidator | None = None,
    ) -> None:
        self.association_config = association_config or AssociationConfig()
        self.relation_associator = relation_associator or ShopliftingRelationAssociator(self.association_config)
        self.state_machine = state_machine or SuspiciousActionStateMachine()
        self.risk_scorer = risk_scorer or RiskScorer()
        self.rule_validator = rule_validator or RiskRuleValidator()
        self._emitted_keys: set[tuple[str, str, str]] = set()
        self._bulk_windows: dict[tuple[str, str], dict[str, _BulkWindowEntry]] = {}

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
            event_type = self._candidate_event_type(snapshot)
            if event_type is None:
                continue

            bulk_event = self._maybe_emit_bulk_event(snapshot, camera_id, event_type)
            if bulk_event is not None:
                events.append(bulk_event)

            event = self._build_event(snapshot, camera_id, event_type)
            if event is None:
                continue

            key = (event.person_track_id, self._item_track_id(snapshot), event.event_type)
            if key in self._emitted_keys:
                continue
            self._emitted_keys.add(key)
            events.append(event)
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

    def _candidate_event_type(self, snapshot: ActionStateSnapshot) -> str | None:
        suggested = snapshot.suggested_event_type or "near_body_suspicious"
        if snapshot.state == CONFIRMED_RISK_EVENT:
            if suggested in CONCEALMENT_EVENT_TYPES | {"bulk_pickup_to_bag"}:
                return suggested
            if suggested == "near_body_suspicious" and self._supports_near_body(snapshot):
                return suggested
            return None

        if snapshot.state in {ITEM_PICKED, NEAR_BODY_OR_CONTAINER, SUSPECTED_CONCEALMENT}:
            if self._supports_near_body(snapshot):
                return "near_body_suspicious"
        return None

    def _supports_near_body(self, snapshot: ActionStateSnapshot) -> bool:
        relation_types = {evidence.relation_type for evidence in snapshot.evidence}
        if len(relation_types) < 2:
            return False
        if "hand_item_contact" not in relation_types:
            return False
        if not (
            "item_follow_person" in relation_types
            or "item_enter_container" in relation_types
            or "item_disappeared_after_entry" in relation_types
        ):
            return False
        return True

    def _maybe_emit_bulk_event(
        self,
        snapshot: ActionStateSnapshot,
        camera_id: str,
        event_type: str,
    ) -> RiskEvent | None:
        if event_type not in CONCEALMENT_EVENT_TYPES:
            return None
        if snapshot.state != CONFIRMED_RISK_EVENT:
            return None

        container_track_id = self._snapshot_container_track_id(snapshot)
        if not container_track_id:
            return None

        key = (snapshot.person_track_id, container_track_id)
        window = self._bulk_windows.setdefault(key, {})
        window[self._item_track_id(snapshot)] = _BulkWindowEntry(
            snapshot=snapshot,
            timestamp_ms=snapshot.timestamp_ms,
        )
        self._prune_bulk_window(key, snapshot.timestamp_ms)

        if len(window) < self.risk_scorer.config.bulk_item_count_threshold:
            return None

        bulk_key = (snapshot.person_track_id, container_track_id, "bulk_pickup_to_bag")
        if bulk_key in self._emitted_keys:
            return None

        combined_snapshot = self._combine_bulk_snapshot(snapshot, window.values(), container_track_id)
        event = self._build_event(combined_snapshot, camera_id, "bulk_pickup_to_bag")
        if event is None:
            return None

        self._emitted_keys.add(bulk_key)
        return event

    def _combine_bulk_snapshot(
        self,
        snapshot: ActionStateSnapshot,
        entries: Sequence[_BulkWindowEntry],
        container_track_id: str,
    ) -> ActionStateSnapshot:
        evidence = _dedupe_evidence(
            evidence
            for entry in entries
            for evidence in entry.snapshot.evidence
        )
        reason_tags = _unique_tags(
            snapshot.reason_tags,
            ("bulk_pickup_to_bag", "bulk_pickup"),
            *(entry.snapshot.reason_tags for entry in entries),
        )
        metadata = dict(snapshot.metadata)
        metadata.update(
            {
                "bulk_item_count": len(entries),
                "bulk_item_track_ids": tuple(sorted(self._item_track_id(entry.snapshot) for entry in entries)),
                "container_track_id": container_track_id,
                "bulk_pickup": True,
            }
        )
        return replace(
            snapshot,
            evidence=evidence,
            reason_tags=reason_tags,
            metadata=metadata,
        )

    def _build_event(
        self,
        snapshot: ActionStateSnapshot,
        camera_id: str,
        event_type: str,
    ) -> RiskEvent | None:
        if not snapshot.evidence:
            return None

        score = self.risk_scorer.score_snapshot(snapshot, event_type=event_type)
        event = RiskEvent(
            event_id=self._build_event_id(camera_id, snapshot, event_type),
            camera_id=camera_id,
            timestamp_ms=snapshot.timestamp_ms,
            start_timestamp_ms=_first_timestamp(snapshot.evidence),
            end_timestamp_ms=snapshot.timestamp_ms,
            person_track_id=snapshot.person_track_id,
            event_type=event_type,
            risk_score=score.risk_score,
            risk_level=score.risk_level,
            confidence=_mean_score(snapshot.evidence),
            reason_tags=score.reason_tags,
            evidence=tuple(snapshot.evidence),
            metadata={
                "state": snapshot.state,
                "item_track_id": snapshot.item_track_id,
                "event_type": event_type,
                "source": "shoplifting_event_engine_p1",
                "score_components": score.components,
                **snapshot.metadata,
            },
        )
        validation = self.rule_validator.apply(event)
        if validation.event is None:
            return None
        normalized_event = validation.event
        if validation.violations:
            metadata = dict(normalized_event.metadata)
            metadata["rule_violations"] = [violation.code for violation in validation.violations]
            metadata["rule_violation_messages"] = [violation.message for violation in validation.violations]
            metadata["rule_changed"] = validation.changed
            normalized_event = replace(normalized_event, metadata=metadata)
        return normalized_event

    def _prune_bulk_window(self, key: tuple[str, str], timestamp_ms: int) -> None:
        window = self._bulk_windows.get(key)
        if not window:
            return
        threshold = max(4000, self.risk_scorer.config.high_risk_min_evidence_frames * 400)
        for item_track_id, entry in list(window.items()):
            if timestamp_ms - entry.timestamp_ms > threshold:
                del window[item_track_id]
        if not window:
            del self._bulk_windows[key]

    def _build_event_id(self, camera_id: str, snapshot: ActionStateSnapshot, event_type: str) -> str:
        return (
            f"evt-{camera_id}-{snapshot.person_track_id}-"
            f"{self._item_track_id(snapshot)}-{event_type}-{snapshot.frame_id}"
        )

    @staticmethod
    def _snapshot_container_track_id(snapshot: ActionStateSnapshot) -> str | None:
        for evidence in reversed(snapshot.evidence):
            if evidence.container_track_id:
                return evidence.container_track_id
        value = snapshot.metadata.get("container_track_id")
        if isinstance(value, str) and value:
            return value
        return None

    @staticmethod
    def _item_track_id(snapshot: ActionStateSnapshot) -> str:
        return snapshot.item_track_id


def _dedupe_evidence(evidences: Sequence[RelationEvidence] | Sequence[Sequence[RelationEvidence]]) -> tuple[RelationEvidence, ...]:
    flattened: list[RelationEvidence] = []
    for item in evidences:
        if isinstance(item, RelationEvidence):
            flattened.append(item)
            continue
        flattened.extend(item)

    seen: set[tuple[object, ...]] = set()
    deduped: list[RelationEvidence] = []
    for evidence in sorted(flattened, key=lambda item: (item.timestamp_ms, item.frame_id, item.relation_type, item.item_track_id or "")):
        key = (
            evidence.relation_type,
            evidence.frame_id,
            evidence.timestamp_ms,
            evidence.person_track_id,
            evidence.hand_track_id,
            evidence.item_track_id,
            evidence.container_track_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(evidence)
    return tuple(deduped)


def _unique_tags(*groups: Sequence[str]) -> tuple[str, ...]:
    tags: list[str] = []
    for group in groups:
        for tag in group:
            if tag and tag not in tags:
                tags.append(tag)
    return tuple(tags)


def _count_relations(relations: Sequence[RelationEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relation in relations:
        counts[relation.relation_type] = counts.get(relation.relation_type, 0) + 1
    return counts


def _mean_score(evidences: Sequence[RelationEvidence]) -> float:
    if not evidences:
        return 0.0
    return sum(evidence.score for evidence in evidences) / len(evidences)


def _first_timestamp(evidences: Sequence[RelationEvidence]) -> int | None:
    if not evidences:
        return None
    return min(evidence.timestamp_ms for evidence in evidences)


__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "EventEngineResult",
    "RelationEvidence",
    "RiskEvent",
    "ShopliftingEventEngine",
]
