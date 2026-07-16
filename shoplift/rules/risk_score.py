"""Risk scoring for shoplifting risk events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shoplift.core.types import RelationEvidence, RiskLevel
from shoplift.events.state_machine import ActionStateSnapshot


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _risk_level(score: float, *, medium_threshold: float, high_threshold: float) -> RiskLevel:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _unique_tags(*groups: Sequence[str]) -> tuple[str, ...]:
    tags: list[str] = []
    for group in groups:
        for tag in group:
            if tag and tag not in tags:
                tags.append(tag)
    return tuple(tags)


def _evidence_frame_span(evidence: Sequence[RelationEvidence]) -> int:
    if not evidence:
        return 1
    frame_ids = [item.frame_id for item in evidence]
    return max(frame_ids) - min(frame_ids) + 1


def _relation_counts(evidence: Sequence[RelationEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.relation_type] = counts.get(item.relation_type, 0) + 1
    return counts


def _metadata_value(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _container_kind_from_snapshot(snapshot: ActionStateSnapshot) -> str:
    tags = set(snapshot.reason_tags)
    metadata_sources: list[Mapping[str, Any]] = [snapshot.metadata]
    metadata_sources.extend(evidence.metadata for evidence in snapshot.evidence)

    for metadata in metadata_sources:
        kind = str(_metadata_value(metadata, "container_kind", "container_type") or "").strip().lower()
        if kind in {"private", "bag", "special", "clothing", "normal"}:
            if kind == "bag":
                return "private"
            return kind
        if metadata.get("is_normal_container") is True:
            return "normal"

    if "entered_normal_container" in tags or "normal_container_exempted" in tags:
        return "normal"
    if "entered_clothing_region" in tags or "after_clothing_region_entry" in tags:
        return "clothing"
    if "entered_special_container" in tags or "after_special_container_entry" in tags:
        return "special"
    if "entered_private_container" in tags or "after_private_container_entry" in tags:
        return "private"
    return "unknown"


@dataclass(frozen=True)
class RiskScoreBreakdown:
    """Breakdown of a single risk score calculation."""

    risk_score: float
    risk_level: RiskLevel
    reason_tags: tuple[str, ...]
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskScoringConfig:
    """Weights and caps used by the P1 risk scorer."""

    action_type_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "clothing_concealment": 0.34,
            "bag_concealment": 0.32,
            "special_container_concealment": 0.34,
            "bulk_pickup_to_bag": 0.38,
            "near_body_suspicious": 0.18,
            "normal_container_placement": 0.05,
        }
    )
    container_type_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "private": 0.16,
            "clothing": 0.18,
            "special": 0.18,
            "normal": -0.20,
            "unknown": 0.04,
        }
    )
    continuous_evidence_weight: float = 0.16
    relation_diversity_weight: float = 0.08
    model_confidence_weight: float = 0.20
    area_risk_weight: float = 0.12
    normal_shopping_downgrade: float = 0.35
    low_visibility_score_cap: float = 0.55
    low_confidence_score_cap: float = 0.55
    low_confidence_threshold: float = 0.35
    single_frame_contact_score_cap: float = 0.44
    medium_threshold: float = 0.45
    high_threshold: float = 0.75
    bulk_item_count_threshold: int = 3
    high_risk_min_evidence_frames: int = 8

    def __post_init__(self) -> None:
        if self.bulk_item_count_threshold <= 1:
            raise ValueError("bulk_item_count_threshold must be greater than 1")
        if self.high_risk_min_evidence_frames <= 0:
            raise ValueError("high_risk_min_evidence_frames must be positive")


class RiskScorer:
    """Convert state-machine snapshots into calibrated risk scores."""

    def __init__(self, config: RiskScoringConfig | None = None) -> None:
        self.config = config or RiskScoringConfig()

    def score_snapshot(
        self,
        snapshot: ActionStateSnapshot,
        *,
        event_type: str | None = None,
    ) -> RiskScoreBreakdown:
        event_type = event_type or snapshot.suggested_event_type or "near_body_suspicious"
        reason_tags = set(snapshot.reason_tags)
        evidence = tuple(snapshot.evidence)
        components: dict[str, float] = {}

        action_weight = float(self.config.action_type_weights.get(event_type, 0.12))
        container_kind = _container_kind_from_snapshot(snapshot)
        container_weight = float(self.config.container_type_weights.get(container_kind, 0.0))

        frame_span = _evidence_frame_span(evidence)
        evidence_frames = max(frame_span, len(evidence))
        continuous_ratio = min(1.0, evidence_frames / max(1, self.config.high_risk_min_evidence_frames))
        relation_diversity = min(1.0, len(_relation_counts(evidence)) / 3.0)

        continuous_component = self.config.continuous_evidence_weight * continuous_ratio
        diversity_component = self.config.relation_diversity_weight * relation_diversity
        confidence_component = self.config.model_confidence_weight * _mean([item.score for item in evidence])

        area_risk_component = self.config.area_risk_weight * self._area_risk_factor(snapshot)
        normal_downgrade = self.config.normal_shopping_downgrade if container_kind == "normal" else 0.0

        score = (
            action_weight
            + container_weight
            + continuous_component
            + diversity_component
            + confidence_component
            + area_risk_component
            - normal_downgrade
        )

        tags = set(reason_tags)
        if container_kind == "normal":
            tags.update({"normal_container_exempted", "normal_shopping"})
        if self._is_low_visibility(snapshot):
            tags.add("low_visibility")
            score = min(score, self.config.low_visibility_score_cap)
        if self._is_low_confidence(snapshot):
            tags.add("low_confidence")
            score = min(score, self.config.low_confidence_score_cap)
        if self._is_single_frame_contact(snapshot, event_type):
            score = min(score, self.config.single_frame_contact_score_cap)
        if event_type == "bulk_pickup_to_bag":
            item_count = self._item_count(snapshot)
            if item_count >= self.config.bulk_item_count_threshold:
                score = max(score, 0.78)
                tags.add("bulk_pickup")
            else:
                score = min(score, 0.55)

        score = _clamp(score)
        risk_level = _risk_level(score, medium_threshold=self.config.medium_threshold, high_threshold=self.config.high_threshold)
        if risk_level == "high" and len(tags) < 2:
            score = min(score, self.config.high_threshold - 0.01)
            risk_level = _risk_level(score, medium_threshold=self.config.medium_threshold, high_threshold=self.config.high_threshold)

        return RiskScoreBreakdown(
            risk_score=score,
            risk_level=risk_level,
            reason_tags=tuple(sorted(tags)),
            components={
                "action_weight": action_weight,
                "container_weight": container_weight,
                "continuous_component": continuous_component,
                "diversity_component": diversity_component,
                "confidence_component": confidence_component,
                "area_risk_component": area_risk_component,
                "normal_downgrade": normal_downgrade,
            },
            metadata={
                "event_type": event_type,
                "container_kind": container_kind,
                "evidence_frames": evidence_frames,
                "frame_span": frame_span,
            },
        )

    def _area_risk_factor(self, snapshot: ActionStateSnapshot) -> float:
        sources: list[Mapping[str, Any]] = [snapshot.metadata]
        sources.extend(evidence.metadata for evidence in snapshot.evidence)

        best = 0.0
        for source in sources:
            raw = _metadata_value(source, "area_risk_weight", "zone_risk_weight")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                best = max(best, max(0.0, min(1.0, float(raw))))
                continue
            label = str(_metadata_value(source, "area_risk", "zone_risk", "risk_zone") or "").strip().lower()
            if label in {"high", "high_risk", "danger", "red"}:
                best = max(best, 1.0)
            elif label in {"medium", "warn", "yellow"}:
                best = max(best, 0.5)
            elif label in {"low", "green"}:
                best = max(best, 0.0)
        if any(bool(source.get("high_risk_zone")) for source in sources if isinstance(source, Mapping)):
            best = max(best, 1.0)
        if any(bool(source.get("blind_spot")) for source in sources if isinstance(source, Mapping)):
            best = max(best, 0.5)
        return best

    def _is_low_visibility(self, snapshot: ActionStateSnapshot) -> bool:
        sources: list[Mapping[str, Any]] = [snapshot.metadata]
        sources.extend(evidence.metadata for evidence in snapshot.evidence)
        tags = set(snapshot.reason_tags)
        for source in sources:
            if source.get("low_visibility") is True or source.get("low_confidence") is True:
                return True
            visibility = str(_metadata_value(source, "visibility", "visibility_level") or "").strip().lower()
            if visibility in {"low", "poor", "occluded", "severe_occlusion"}:
                return True
            occlusion = _metadata_value(source, "occlusion_ratio", "visibility_ratio")
            if isinstance(occlusion, (int, float)) and not isinstance(occlusion, bool) and float(occlusion) >= 0.6:
                return True
        return bool({"low_visibility", "possible_occlusion", "severe_occlusion"} & tags)

    def _is_low_confidence(self, snapshot: ActionStateSnapshot) -> bool:
        if snapshot.evidence and _mean([item.score for item in snapshot.evidence]) < self.config.low_confidence_threshold:
            return True
        for source in [snapshot.metadata, *[evidence.metadata for evidence in snapshot.evidence]]:
            confidence = _metadata_value(source, "confidence", "score")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                if float(confidence) < self.config.low_confidence_threshold:
                    return True
        return False

    def _is_single_frame_contact(self, snapshot: ActionStateSnapshot, event_type: str) -> bool:
        if event_type not in {"near_body_suspicious", "bag_concealment", "clothing_concealment", "special_container_concealment", "bulk_pickup_to_bag"}:
            return False
        if not snapshot.evidence:
            return False
        relation_types = {item.relation_type for item in snapshot.evidence}
        if relation_types != {"hand_item_contact"} and "item_enter_container" not in relation_types:
            return False
        contact_frames = [int(_metadata_value(item.metadata, "contact_frames", "continuous_frames") or 1) for item in snapshot.evidence]
        return max(contact_frames, default=1) <= 1 and len(snapshot.evidence) == 1

    def _item_count(self, snapshot: ActionStateSnapshot) -> int:
        candidates = [snapshot.metadata]
        candidates.extend(evidence.metadata for evidence in snapshot.evidence)
        for source in candidates:
            value = _metadata_value(source, "item_count", "picked_item_count", "bulk_item_count")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return len({item.item_track_id for item in snapshot.evidence if item.item_track_id})


def score_snapshot(snapshot: ActionStateSnapshot, *, event_type: str | None = None) -> RiskScoreBreakdown:
    return RiskScorer().score_snapshot(snapshot, event_type=event_type)


def risk_level(score: float, *, medium_threshold: float = 0.45, high_threshold: float = 0.75) -> RiskLevel:
    return _risk_level(score, medium_threshold=medium_threshold, high_threshold=high_threshold)


__all__ = [
    "RiskScoreBreakdown",
    "RiskScorer",
    "RiskScoringConfig",
    "risk_level",
    "score_snapshot",
]
