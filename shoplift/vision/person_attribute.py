"""Person attribute postprocessing and proxy-item generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from shoplift.core.types import (
    AttributePrediction,
    BBox,
    DetectionBox,
    FrameMeta,
    HandRegion,
    PersonAttribute,
    ProxyItemRegion,
    Tracklet,
)


HAND_STATE_LABELS = ("empty", "holding_object", "holding_product", "uncertain")
HAND_VISIBILITY_LABELS = ("clear", "partial_occluded", "not_judgable")
BODY_ORIENTATION_LABELS = ("front", "side", "back", "unknown")
OCCLUSION_LEVEL_LABELS = ("none", "light", "heavy")


@dataclass(frozen=True)
class PersonAttributeConfig:
    min_holding_product_score: float = 0.5
    not_judgable_score_threshold: float = 0.7
    heavy_occlusion_score_threshold: float = 0.7
    low_confidence_multiplier: float = 0.6
    proxy_bbox_padding_ratio: float = 0.0


@dataclass(frozen=True)
class PersonAttributeFrameResult:
    person_attributes: tuple[PersonAttribute, ...] = field(default_factory=tuple)
    proxy_item_regions: tuple[ProxyItemRegion, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def proxy_item_boxes(self) -> tuple[DetectionBox, ...]:
        return tuple(region.to_detection_box() for region in self.proxy_item_regions)

    def to_dict(self) -> dict[str, object]:
        return {
            "person_attributes": [attribute.to_dict() for attribute in self.person_attributes],
            "proxy_item_regions": [region.to_dict() for region in self.proxy_item_regions],
            "metadata": self.metadata,
        }


class PersonAttributePostProcessor:
    """Normalize raw multi-head model predictions into person attributes."""

    def __init__(self, config: PersonAttributeConfig | None = None) -> None:
        self.config = config or PersonAttributeConfig()

    def build_attribute(
        self,
        *,
        frame: FrameMeta,
        person_track: Tracklet,
        raw: Mapping[str, object],
    ) -> PersonAttribute:
        person_box = _latest_track_box(person_track)
        if person_box is None:
            raise ValueError("person_track must contain at least one box")

        left_state = _prediction(raw.get("left_hand_state"), HAND_STATE_LABELS, "uncertain")
        left_visibility = _prediction(raw.get("left_hand_visibility"), HAND_VISIBILITY_LABELS, "not_judgable")
        right_state = _prediction(raw.get("right_hand_state"), HAND_STATE_LABELS, "uncertain")
        right_visibility = _prediction(raw.get("right_hand_visibility"), HAND_VISIBILITY_LABELS, "not_judgable")
        orientation = _prediction(raw.get("body_orientation"), BODY_ORIENTATION_LABELS, "unknown")
        occlusion = _prediction(raw.get("occlusion_level"), OCCLUSION_LEVEL_LABELS, "heavy")

        left_state = self._enforce_visibility(left_state, left_visibility)
        right_state = self._enforce_visibility(right_state, right_visibility)

        return PersonAttribute(
            attribute_id=f"attr-{frame.frame_id}-{person_track.track_id}",
            person_track_id=person_track.track_id,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            bbox=person_box.bbox,
            left_hand_state=left_state,
            left_hand_visibility=left_visibility,
            right_hand_state=right_state,
            right_hand_visibility=right_visibility,
            body_orientation=orientation,
            occlusion_level=occlusion,
            metadata={
                "source": str(raw.get("source", "person_attribute_model")),
                "raw": {
                    key: value
                    for key, value in raw.items()
                    if key
                    not in {
                        "left_hand_state",
                        "left_hand_visibility",
                        "right_hand_state",
                        "right_hand_visibility",
                        "body_orientation",
                        "occlusion_level",
                    }
                },
            },
        )

    def _enforce_visibility(
        self,
        state: AttributePrediction,
        visibility: AttributePrediction,
    ) -> AttributePrediction:
        if (
            visibility.label == "not_judgable"
            and visibility.score >= self.config.not_judgable_score_threshold
            and state.label != "uncertain"
        ):
            return AttributePrediction("uncertain", max(state.score, visibility.score))
        return state


class ProxyItemRegionBuilder:
    """Create proxy item regions from held-product attributes and hand ROIs."""

    def __init__(self, config: PersonAttributeConfig | None = None) -> None:
        self.config = config or PersonAttributeConfig()

    def build(
        self,
        *,
        frame: FrameMeta,
        person_attributes: Sequence[PersonAttribute],
        hand_regions: Sequence[HandRegion],
    ) -> tuple[ProxyItemRegion, ...]:
        hands_by_key = {
            (hand.person_track_id, hand.side): hand
            for hand in hand_regions
            if hand.side in {"left", "right"}
        }
        regions: list[ProxyItemRegion] = []
        for attribute in person_attributes:
            for side in ("left", "right"):
                hand = hands_by_key.get((attribute.person_track_id, side))
                if hand is None:
                    continue
                state = getattr(attribute, f"{side}_hand_state")
                visibility = getattr(attribute, f"{side}_hand_visibility")
                region = self._build_one(frame, attribute, hand, state, visibility)
                if region is not None:
                    regions.append(region)
        return tuple(regions)

    def _build_one(
        self,
        frame: FrameMeta,
        attribute: PersonAttribute,
        hand: HandRegion,
        state: AttributePrediction,
        visibility: AttributePrediction,
    ) -> ProxyItemRegion | None:
        if state.label != "holding_product":
            return None
        if state.score < self.config.min_holding_product_score:
            return None
        if (
            visibility.label == "not_judgable"
            and visibility.score >= self.config.not_judgable_score_threshold
        ):
            return None

        confidence = min(1.0, state.score * 0.75 + hand.score * 0.25)
        reason_tags = ["holding_product", "hand_roi_proxy"]
        if visibility.label == "partial_occluded":
            confidence *= self.config.low_confidence_multiplier
            reason_tags.append("partial_occluded")
        if (
            attribute.occlusion_level.label == "heavy"
            and attribute.occlusion_level.score >= self.config.heavy_occlusion_score_threshold
        ):
            confidence *= self.config.low_confidence_multiplier
            reason_tags.append("heavy_occlusion")

        proxy_bbox = _clip_bbox(
            _expand_bbox(hand.bbox, self.config.proxy_bbox_padding_ratio),
            frame,
        )
        return ProxyItemRegion(
            proxy_item_id=f"proxy-item-{frame.frame_id}-{attribute.person_track_id}-{hand.side}",
            proxy_item_track_id=f"proxy-item-{attribute.person_track_id}-{hand.side}",
            person_track_id=attribute.person_track_id,
            hand_track_id=hand.hand_track_id,
            hand_side=hand.side,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            proxy_bbox=proxy_bbox,
            source_hand_roi=hand.bbox,
            confidence=max(0.0, min(1.0, confidence)),
            state_label=state.label,
            state_score=state.score,
            visibility_label=visibility.label,
            visibility_score=visibility.score,
            is_precise_item_bbox=False,
            metadata={
                "attribute_id": attribute.attribute_id,
                "reason_tags": tuple(reason_tags),
                "body_orientation": attribute.body_orientation.label,
                "occlusion_level": attribute.occlusion_level.label,
            },
        )


class RuleBasedPersonAttributeEstimator:
    """Deterministic early-integration estimator used before trained weights exist.

    The estimator is intentionally conservative: by default it marks visible
    hands as empty and unavailable hands as uncertain. Tests or fixture backends
    can pass explicit raw predictions to exercise held-product behavior.
    """

    def __init__(self, config: PersonAttributeConfig | None = None) -> None:
        self.postprocessor = PersonAttributePostProcessor(config)

    def estimate(
        self,
        *,
        frame: FrameMeta,
        person_tracks: Sequence[Tracklet],
        hand_regions: Sequence[HandRegion],
    ) -> tuple[PersonAttribute, ...]:
        hands_by_person: dict[str, set[str]] = {}
        for hand in hand_regions:
            hands_by_person.setdefault(hand.person_track_id, set()).add(hand.side)

        attributes: list[PersonAttribute] = []
        for track in person_tracks:
            visible_sides = hands_by_person.get(track.track_id, set())
            raw = {
                "source": "rule_based_person_attribute_estimator",
                "left_hand_state": ("empty", 0.55) if "left" in visible_sides else ("uncertain", 0.6),
                "left_hand_visibility": ("clear", 0.55) if "left" in visible_sides else ("not_judgable", 0.6),
                "right_hand_state": ("empty", 0.55) if "right" in visible_sides else ("uncertain", 0.6),
                "right_hand_visibility": ("clear", 0.55) if "right" in visible_sides else ("not_judgable", 0.6),
                "body_orientation": ("unknown", 0.5),
                "occlusion_level": ("light", 0.5),
            }
            attributes.append(
                self.postprocessor.build_attribute(
                    frame=frame,
                    person_track=track,
                    raw=raw,
                )
            )
        return tuple(attributes)


def build_proxy_item_regions(
    *,
    frame: FrameMeta,
    person_attributes: Sequence[PersonAttribute],
    hand_regions: Sequence[HandRegion],
    config: PersonAttributeConfig | None = None,
) -> tuple[ProxyItemRegion, ...]:
    return ProxyItemRegionBuilder(config).build(
        frame=frame,
        person_attributes=person_attributes,
        hand_regions=hand_regions,
    )


def _prediction(value: object, labels: Sequence[str], default_label: str) -> AttributePrediction:
    if isinstance(value, AttributePrediction):
        return value
    if isinstance(value, Mapping):
        label = str(value.get("label", default_label))
        score = float(value.get("score", 1.0 if label == default_label else 0.0))
    elif (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], str)
    ):
        label = str(value[0])
        score = float(value[1])
    elif isinstance(value, str):
        label = value
        score = 1.0
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        scores = [float(item) for item in value]
        if not scores:
            label = default_label
            score = 1.0
        else:
            index = max(range(len(scores)), key=scores.__getitem__)
            label = labels[index] if index < len(labels) else default_label
            score = scores[index]
    else:
        label = default_label
        score = 1.0
    if label not in labels:
        label = default_label
        score = 0.0
    return AttributePrediction(label=label, score=max(0.0, min(1.0, score)))


def _latest_track_box(tracklet: Tracklet) -> DetectionBox | None:
    if not tracklet.boxes:
        return None
    return max(tracklet.boxes, key=lambda box: box.frame_id)


def _expand_bbox(bbox: BBox, ratio: float) -> BBox:
    if ratio <= 0.0:
        return bbox
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


def _clip_bbox(bbox: BBox, frame: FrameMeta) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(frame.width), x1)),
        max(0.0, min(float(frame.height), y1)),
        max(0.0, min(float(frame.width), x2)),
        max(0.0, min(float(frame.height), y2)),
    )


__all__ = [
    "BODY_ORIENTATION_LABELS",
    "HAND_STATE_LABELS",
    "HAND_VISIBILITY_LABELS",
    "OCCLUSION_LEVEL_LABELS",
    "PersonAttributeConfig",
    "PersonAttributeFrameResult",
    "PersonAttributePostProcessor",
    "ProxyItemRegionBuilder",
    "RuleBasedPersonAttributeEstimator",
    "build_proxy_item_regions",
]
