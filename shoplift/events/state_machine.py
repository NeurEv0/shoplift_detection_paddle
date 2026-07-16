"""Suspicious action state machine for P1 relation evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from shoplift.core.types import RelationEvidence


EVENT_STATES = (
    "observing",
    "item_picked",
    "near_body_or_container",
    "suspected_concealment",
    "confirmed_risk_event",
    "resolved_or_downgraded",
)

OBSERVING = "observing"
ITEM_PICKED = "item_picked"
NEAR_BODY_OR_CONTAINER = "near_body_or_container"
SUSPECTED_CONCEALMENT = "suspected_concealment"
CONFIRMED_RISK_EVENT = "confirmed_risk_event"
RESOLVED_OR_DOWNGRADED = "resolved_or_downgraded"


@dataclass(frozen=True)
class ActionStateSnapshot:
    """Reviewable state-machine output for one person-item context."""

    state: str
    person_track_id: str
    item_track_id: str
    frame_id: int
    timestamp_ms: int
    reason_tags: tuple[str, ...]
    evidence: tuple[RelationEvidence, ...]
    previous_state: str | None = None
    suggested_event_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED_RISK_EVENT


@dataclass
class _ActionContext:
    person_track_id: str
    item_track_id: str
    state: str = OBSERVING
    started_frame_id: int | None = None
    started_timestamp_ms: int | None = None
    last_frame_id: int | None = None
    last_timestamp_ms: int | None = None
    reason_tags: set[str] = field(default_factory=set)
    evidence: list[RelationEvidence] = field(default_factory=list)
    suggested_event_type: str | None = None
    event_emitted: bool = False


class SuspiciousActionStateMachine:
    """Advance suspicious-action states from relation evidence."""

    def __init__(self, *, max_evidence_items: int = 12) -> None:
        if max_evidence_items <= 0:
            raise ValueError("max_evidence_items must be positive")
        self.max_evidence_items = max_evidence_items
        self._contexts: dict[tuple[str, str], _ActionContext] = {}

    def update(
        self,
        evidences: Sequence[RelationEvidence],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> tuple[ActionStateSnapshot, ...]:
        snapshots: list[ActionStateSnapshot] = []
        for evidence in evidences:
            if not evidence.person_track_id or not evidence.item_track_id:
                continue
            key = (evidence.person_track_id, evidence.item_track_id)
            context = self._contexts.setdefault(
                key,
                _ActionContext(
                    person_track_id=evidence.person_track_id,
                    item_track_id=evidence.item_track_id,
                ),
            )
            previous_state = context.state
            self._append_evidence(context, evidence)
            context.started_frame_id = context.started_frame_id if context.started_frame_id is not None else evidence.frame_id
            context.started_timestamp_ms = (
                context.started_timestamp_ms
                if context.started_timestamp_ms is not None
                else evidence.timestamp_ms
            )
            context.last_frame_id = frame_id
            context.last_timestamp_ms = timestamp_ms
            context.suggested_event_type = _suggested_event_type(context)
            context.state = self._next_state(context, evidence)

            snapshots.append(
                ActionStateSnapshot(
                    state=context.state,
                    previous_state=previous_state,
                    person_track_id=context.person_track_id,
                    item_track_id=context.item_track_id,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    reason_tags=tuple(sorted(context.reason_tags)),
                    evidence=tuple(context.evidence),
                    suggested_event_type=context.suggested_event_type,
                    metadata={
                        "started_frame_id": context.started_frame_id,
                        "started_timestamp_ms": context.started_timestamp_ms,
                        "last_relation_type": evidence.relation_type,
                        "event_emitted": context.event_emitted,
                    },
                )
            )
        return tuple(snapshots)

    def mark_event_emitted(self, snapshot: ActionStateSnapshot) -> None:
        key = (snapshot.person_track_id, snapshot.item_track_id)
        if key in self._contexts:
            self._contexts[key].event_emitted = True

    def context_state(self, person_track_id: str, item_track_id: str) -> str | None:
        context = self._contexts.get((person_track_id, item_track_id))
        return context.state if context else None

    def _append_evidence(self, context: _ActionContext, evidence: RelationEvidence) -> None:
        context.evidence.append(evidence)
        if len(context.evidence) > self.max_evidence_items:
            context.evidence = context.evidence[-self.max_evidence_items :]
        context.reason_tags.add(evidence.relation_type)
        context.reason_tags.update(evidence.reason_tags)

    def _next_state(self, context: _ActionContext, evidence: RelationEvidence) -> str:
        tags = set(evidence.reason_tags)
        all_tags = set(context.reason_tags)

        if evidence.relation_type == "hand_item_contact":
            if context.state in {OBSERVING, RESOLVED_OR_DOWNGRADED}:
                return ITEM_PICKED
            return context.state

        if evidence.relation_type == "item_follow_person":
            if context.state == OBSERVING:
                return ITEM_PICKED
            if "gap_filled" in tags and context.state == CONFIRMED_RISK_EVENT:
                return context.state
            return context.state if context.state != RESOLVED_OR_DOWNGRADED else ITEM_PICKED

        if evidence.relation_type == "item_enter_container":
            if "entered_normal_container" in tags or evidence.metadata.get("is_normal_container"):
                return RESOLVED_OR_DOWNGRADED
            if "entered_private_container" in tags or "entered_special_container" in tags or "entered_clothing_region" in tags:
                return SUSPECTED_CONCEALMENT
            return NEAR_BODY_OR_CONTAINER

        if evidence.relation_type == "item_disappeared_after_entry":
            if "normal_container_exempted" in tags:
                return RESOLVED_OR_DOWNGRADED
            if "low_visibility" in tags or "possible_occlusion" in tags:
                return RESOLVED_OR_DOWNGRADED
            has_entry_disappearance = "after_container_entry" in all_tags or bool(
                {
                    "after_private_container_entry",
                    "after_special_container_entry",
                    "after_clothing_region_entry",
                }
                & all_tags
            )
            if "item_disappeared" in all_tags and has_entry_disappearance:
                return CONFIRMED_RISK_EVENT
            return SUSPECTED_CONCEALMENT

        return context.state


def _suggested_event_type(context: _ActionContext) -> str:
    tags = context.reason_tags
    if "bulk_pickup_to_bag" in tags or "bulk_pickup" in tags or "bulk_item_count" in tags:
        return "bulk_pickup_to_bag"
    if "entered_clothing_region" in tags or "after_clothing_region_entry" in tags:
        return "clothing_concealment"
    if "entered_special_container" in tags or "after_special_container_entry" in tags:
        return "special_container_concealment"
    if "entered_private_container" in tags or "after_private_container_entry" in tags:
        return "bag_concealment"
    if "entered_normal_container" in tags or "normal_container_exempted" in tags:
        return "normal_container_placement"
    if "hand_item_contact" in tags or "item_follow_person" in tags:
        return "near_body_suspicious"
    return "near_body_suspicious"


__all__ = [
    "ActionStateSnapshot",
    "CONFIRMED_RISK_EVENT",
    "EVENT_STATES",
    "ITEM_PICKED",
    "NEAR_BODY_OR_CONTAINER",
    "OBSERVING",
    "RESOLVED_OR_DOWNGRADED",
    "SUSPECTED_CONCEALMENT",
    "SuspiciousActionStateMachine",
]
