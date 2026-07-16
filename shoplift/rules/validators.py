"""Rule validation and normalization for risk events."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from shoplift.core.types import RiskEvent
from shoplift.rules.risk_score import risk_level


ValidationSeverity = Literal["info", "warning", "error"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _container_kind(event: RiskEvent) -> str:
    tags = set(event.reason_tags)
    metadata_sources = [event.metadata, *[item.metadata for item in event.evidence]]
    for source in metadata_sources:
        kind = source.get("container_kind")
        if isinstance(kind, str):
            normalized = kind.strip().lower()
            if normalized == "bag":
                return "private"
            if normalized in {"private", "clothing", "special", "normal"}:
                return normalized
        if source.get("is_normal_container") is True:
            return "normal"
    if "normal_container_exempted" in tags or "entered_normal_container" in tags:
        return "normal"
    if "entered_clothing_region" in tags or "after_clothing_region_entry" in tags:
        return "clothing"
    if "entered_special_container" in tags or "after_special_container_entry" in tags:
        return "special"
    if "entered_private_container" in tags or "after_private_container_entry" in tags:
        return "private"
    return "unknown"


def _low_visibility(event: RiskEvent) -> bool:
    tags = set(event.reason_tags)
    if {"low_visibility", "possible_occlusion", "severe_occlusion"} & tags:
        return True
    for source in [event.metadata, *[item.metadata for item in event.evidence]]:
        if source.get("low_visibility") is True or source.get("low_confidence") is True:
            return True
        visibility = source.get("visibility") or source.get("visibility_level")
        if isinstance(visibility, str) and visibility.strip().lower() in {"low", "poor", "occluded", "severe_occlusion"}:
            return True
        occlusion = source.get("occlusion_ratio") or source.get("visibility_ratio")
        if _is_number(occlusion) and float(occlusion) >= 0.6:
            return True
    return False


def _single_frame_contact(event: RiskEvent) -> bool:
    if len(event.evidence) != 1:
        return False
    evidence = event.evidence[0]
    if evidence.relation_type != "hand_item_contact":
        return False
    contact_frames = evidence.metadata.get("contact_frames") or evidence.metadata.get("continuous_frames")
    if _is_number(contact_frames):
        return int(contact_frames) <= 1
    return True


def _high_risk_reason_count(event: RiskEvent) -> int:
    return len(set(event.reason_tags))


def _level_cap(level: str, medium_threshold: float, high_threshold: float) -> float:
    if level == "low":
        return max(0.0, medium_threshold - 0.01)
    if level == "medium":
        return max(0.0, high_threshold - 0.01)
    return 1.0


@dataclass(frozen=True)
class RiskRuleConfig:
    """Thresholds used by the P1 rule validator."""

    medium_threshold: float = 0.45
    high_threshold: float = 0.75
    high_risk_min_reason_tags: int = 2
    low_visibility_risk_cap: str = "medium"
    normal_container_risk_cap: str = "medium"
    single_frame_contact_risk_cap: str = "low"
    low_visibility_score_cap: float = 0.55
    low_confidence_score_cap: float = 0.55
    single_frame_contact_score_cap: float = 0.44
    normal_container_score_cap: float = 0.44

    def __post_init__(self) -> None:
        if self.high_risk_min_reason_tags <= 0:
            raise ValueError("high_risk_min_reason_tags must be positive")


@dataclass(frozen=True)
class RuleViolation:
    """One validation problem or normalization adjustment."""

    code: str
    message: str
    severity: ValidationSeverity = "warning"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleValidationResult:
    """Validation result plus the normalized event payload."""

    event: RiskEvent | None
    violations: tuple[RuleViolation, ...]
    changed: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.violations


class RiskRuleValidator:
    """Validate and normalize shoplifting risk events."""

    def __init__(self, config: RiskRuleConfig | None = None) -> None:
        self.config = config or RiskRuleConfig()

    def validate(self, event: RiskEvent) -> tuple[RuleViolation, ...]:
        violations: list[RuleViolation] = []
        if event.risk_level == "high" and _high_risk_reason_count(event) < self.config.high_risk_min_reason_tags:
            violations.append(
                RuleViolation(
                    code="high_risk_requires_multiple_reason_tags",
                    message="high risk events must have at least two reason tags",
                )
            )
        if event.risk_level == "high" and _single_frame_contact(event):
            violations.append(
                RuleViolation(
                    code="single_frame_contact_must_not_be_high",
                    message="single-frame contact evidence cannot remain high risk",
                )
            )
        if event.risk_level == "high" and _container_kind(event) == "normal":
            violations.append(
                RuleViolation(
                    code="normal_container_must_not_be_high",
                    message="normal basket/cart placement must not become a high-risk event",
                )
            )
        if event.risk_level == "high" and _low_visibility(event):
            violations.append(
                RuleViolation(
                    code="low_visibility_must_be_downgraded",
                    message="severe occlusion or low visibility must be downgraded",
                )
            )
        return tuple(violations)

    def apply(self, event: RiskEvent) -> RuleValidationResult:
        violations = list(self.validate(event))
        if not violations:
            return RuleValidationResult(event=event, violations=(), changed=False)

        reason_tags = set(event.reason_tags)
        metadata = dict(event.metadata)
        risk_score = event.risk_score
        changed = False

        container_kind = _container_kind(event)
        if container_kind == "normal":
            reason_tags.update({"normal_container_exempted", "normal_shopping"})
            risk_score = min(risk_score, _level_cap(self.config.normal_container_risk_cap, self.config.medium_threshold, self.config.high_threshold))
            risk_score = min(risk_score, self.config.normal_container_score_cap)
            changed = True

        if _low_visibility(event):
            reason_tags.add("low_visibility")
            risk_score = min(risk_score, _level_cap(self.config.low_visibility_risk_cap, self.config.medium_threshold, self.config.high_threshold))
            risk_score = min(risk_score, self.config.low_visibility_score_cap)
            metadata.setdefault("visibility", "low")
            changed = True

        if _single_frame_contact(event):
            reason_tags.add("single_frame_contact")
            risk_score = min(risk_score, _level_cap(self.config.single_frame_contact_risk_cap, self.config.medium_threshold, self.config.high_threshold))
            risk_score = min(risk_score, self.config.single_frame_contact_score_cap)
            changed = True

        if event.risk_level == "high" and _high_risk_reason_count(event) < self.config.high_risk_min_reason_tags:
            risk_score = min(risk_score, self.config.high_threshold - 0.01)
            changed = True

        if any(v.code == "low_visibility_must_be_downgraded" for v in violations):
            reason_tags.add("low_visibility")
            metadata.setdefault("low_visibility", True)
            changed = True

        risk_score = max(0.0, min(1.0, risk_score))
        adjusted_level = risk_level(
            risk_score,
            medium_threshold=self.config.medium_threshold,
            high_threshold=self.config.high_threshold,
        )
        if adjusted_level == "high" and len(reason_tags) < self.config.high_risk_min_reason_tags:
            risk_score = min(risk_score, self.config.high_threshold - 0.01)
            adjusted_level = risk_level(
                risk_score,
                medium_threshold=self.config.medium_threshold,
                high_threshold=self.config.high_threshold,
            )
            changed = True

        adjusted = replace(
            event,
            risk_score=risk_score,
            risk_level=adjusted_level,
            reason_tags=tuple(sorted(reason_tags)),
            metadata=metadata,
        )
        return RuleValidationResult(event=adjusted, violations=tuple(violations), changed=changed)


__all__ = [
    "RiskRuleConfig",
    "RiskRuleValidator",
    "RuleValidationResult",
    "RuleViolation",
]
