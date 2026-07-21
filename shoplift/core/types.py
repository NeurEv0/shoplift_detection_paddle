"""Typed data structures passed between shoplift pipeline modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
RiskLevel = Literal["low", "medium", "high"]
HandSide = Literal["left", "right", "unknown"]
HandState = Literal["empty", "holding_object", "holding_product", "uncertain"]
HandVisibility = Literal["clear", "partial_occluded", "not_judgable"]
BodyOrientation = Literal["front", "side", "back", "unknown"]
OcclusionLevel = Literal["none", "light", "heavy"]

_RISK_LEVELS = {"low", "medium", "high"}
_HAND_SIDES = {"left", "right", "unknown"}
_HAND_STATES = {"empty", "holding_object", "holding_product", "uncertain"}
_HAND_VISIBILITIES = {"clear", "partial_occluded", "not_judgable"}
_BODY_ORIENTATIONS = {"front", "side", "back", "unknown"}
_OCCLUSION_LEVELS = {"none", "light", "heavy"}


def _validate_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_score(value: float, field_name: str = "score") -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


def _validate_enum(value: str, allowed: set[str], field_name: str) -> None:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_text}")


def _normalize_bbox(bbox: BBox) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must be ordered as [x1, y1, x2, y2]")
    return (x1, y1, x2, y2)


def _normalize_points(points: tuple[Point, ...]) -> tuple[Point, ...]:
    normalized = []
    for point in points:
        if len(point) != 2:
            raise ValueError("points must contain exactly two coordinates")
        normalized.append((float(point[0]), float(point[1])))
    return tuple(normalized)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class FrameMeta:
    """Frame identity and timing metadata shared by all module outputs."""

    frame_id: int
    timestamp_ms: int
    camera_id: str
    width: int
    height: int
    source_uri: str | None = None

    def __post_init__(self) -> None:
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        _validate_non_empty(self.camera_id, "camera_id")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class DetectionBox:
    """Single object detection or tracked bbox in pixel coordinates."""

    box_id: str
    frame_id: int
    category: str
    bbox: BBox
    score: float
    track_id: str | None = None
    timestamp_ms: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.box_id, "box_id")
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_empty(self.category, "category")
        _validate_score(float(self.score))
        if self.timestamp_ms is not None:
            _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        object.__setattr__(self, "score", float(self.score))

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class Tracklet:
    """Time-ordered detections belonging to a single tracked entity."""

    track_id: str
    category: str
    boxes: tuple[DetectionBox, ...] = field(default_factory=tuple)
    start_frame_id: int | None = None
    end_frame_id: int | None = None
    timestamps_ms: tuple[int, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.track_id, "track_id")
        _validate_non_empty(self.category, "category")
        boxes = tuple(self.boxes)
        timestamps = tuple(int(value) for value in self.timestamps_ms)
        start = self.start_frame_id
        end = self.end_frame_id
        if boxes:
            frame_ids = [box.frame_id for box in boxes]
            start = min(frame_ids) if start is None else start
            end = max(frame_ids) if end is None else end
        if start is not None:
            _validate_non_negative(start, "start_frame_id")
        if end is not None:
            _validate_non_negative(end, "end_frame_id")
        if start is not None and end is not None and end < start:
            raise ValueError("end_frame_id must be greater than or equal to start_frame_id")
        for timestamp_ms in timestamps:
            _validate_non_negative(timestamp_ms, "timestamps_ms")
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "start_frame_id", start)
        object.__setattr__(self, "end_frame_id", end)
        object.__setattr__(self, "timestamps_ms", timestamps)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class HandRegion:
    """Hand ROI derived from keypoints, hand detector, or fused tracking."""

    hand_track_id: str
    person_track_id: str
    frame_id: int
    timestamp_ms: int
    side: HandSide
    bbox: BBox
    score: float
    source_keypoints: tuple[Point, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.hand_track_id, "hand_track_id")
        _validate_non_empty(self.person_track_id, "person_track_id")
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        if self.side not in _HAND_SIDES:
            raise ValueError("side must be one of: left, right, unknown")
        _validate_score(float(self.score))
        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "source_keypoints", _normalize_points(tuple(self.source_keypoints)))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class BodyPose:
    """Full-body keypoints and skeleton evidence for one tracked person."""

    pose_id: str
    person_track_id: str
    frame_id: int
    timestamp_ms: int
    keypoints: tuple[Point, ...]
    scores: tuple[float, ...]
    score: float
    keypoint_names: tuple[str, ...] = field(default_factory=tuple)
    skeleton_edges: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.pose_id, "pose_id")
        _validate_non_empty(self.person_track_id, "person_track_id")
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        _validate_score(float(self.score))
        scores = tuple(float(score) for score in self.scores)
        for score in scores:
            _validate_score(score, "scores")
        edges = tuple((int(start), int(end)) for start, end in self.skeleton_edges)
        for start, end in edges:
            _validate_non_negative(start, "skeleton_edges")
            _validate_non_negative(end, "skeleton_edges")
        object.__setattr__(self, "keypoints", _normalize_points(tuple(self.keypoints)))
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "keypoint_names", tuple(str(name) for name in self.keypoint_names))
        object.__setattr__(self, "skeleton_edges", edges)
        if self.bbox is not None:
            object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class AttributePrediction:
    """Single categorical person-attribute prediction."""

    label: str
    score: float

    def __post_init__(self) -> None:
        _validate_non_empty(self.label, "label")
        _validate_score(float(self.score))
        object.__setattr__(self, "score", float(self.score))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class PersonAttribute:
    """Shoplift-specific person attributes for one tracked person and frame."""

    attribute_id: str
    person_track_id: str
    frame_id: int
    timestamp_ms: int
    bbox: BBox
    left_hand_state: AttributePrediction
    left_hand_visibility: AttributePrediction
    right_hand_state: AttributePrediction
    right_hand_visibility: AttributePrediction
    body_orientation: AttributePrediction
    occlusion_level: AttributePrediction
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.attribute_id, "attribute_id")
        _validate_non_empty(self.person_track_id, "person_track_id")
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        object.__setattr__(self, "bbox", _normalize_bbox(self.bbox))
        _validate_enum(self.left_hand_state.label, _HAND_STATES, "left_hand_state")
        _validate_enum(self.right_hand_state.label, _HAND_STATES, "right_hand_state")
        _validate_enum(self.left_hand_visibility.label, _HAND_VISIBILITIES, "left_hand_visibility")
        _validate_enum(self.right_hand_visibility.label, _HAND_VISIBILITIES, "right_hand_visibility")
        _validate_enum(self.body_orientation.label, _BODY_ORIENTATIONS, "body_orientation")
        _validate_enum(self.occlusion_level.label, _OCCLUSION_LEVELS, "occlusion_level")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ProxyItemRegion:
    """Item-position proxy derived from a held-product hand ROI."""

    proxy_item_id: str
    proxy_item_track_id: str
    person_track_id: str
    hand_track_id: str
    hand_side: HandSide
    frame_id: int
    timestamp_ms: int
    proxy_bbox: BBox
    source_hand_roi: BBox
    confidence: float
    state_label: str
    state_score: float
    visibility_label: str
    visibility_score: float
    is_precise_item_bbox: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.proxy_item_id, "proxy_item_id")
        _validate_non_empty(self.proxy_item_track_id, "proxy_item_track_id")
        _validate_non_empty(self.person_track_id, "person_track_id")
        _validate_non_empty(self.hand_track_id, "hand_track_id")
        if self.hand_side not in {"left", "right"}:
            raise ValueError("hand_side must be one of: left, right")
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        _validate_score(float(self.confidence), "confidence")
        _validate_score(float(self.state_score), "state_score")
        _validate_score(float(self.visibility_score), "visibility_score")
        _validate_enum(self.state_label, _HAND_STATES, "state_label")
        _validate_enum(self.visibility_label, _HAND_VISIBILITIES, "visibility_label")
        object.__setattr__(self, "proxy_bbox", _normalize_bbox(self.proxy_bbox))
        object.__setattr__(self, "source_hand_roi", _normalize_bbox(self.source_hand_roi))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "state_score", float(self.state_score))
        object.__setattr__(self, "visibility_score", float(self.visibility_score))

    def to_detection_box(self) -> DetectionBox:
        """Expose the proxy as an item box so the existing associator can consume it."""

        return DetectionBox(
            box_id=self.proxy_item_id,
            frame_id=self.frame_id,
            timestamp_ms=self.timestamp_ms,
            category="item",
            bbox=self.proxy_bbox,
            score=self.confidence,
            track_id=self.proxy_item_track_id,
            attributes={
                **self.metadata,
                "source": "person_attribute_hand_roi",
                "is_proxy_item_region": True,
                "is_precise_item_bbox": self.is_precise_item_bbox,
                "person_track_id": self.person_track_id,
                "hand_track_id": self.hand_track_id,
                "hand_side": self.hand_side,
                "source_hand_roi": self.source_hand_roi,
                "state_label": self.state_label,
                "state_score": self.state_score,
                "visibility_label": self.visibility_label,
                "visibility_score": self.visibility_score,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class RelationEvidence:
    """Explainable evidence for a hand-item-container temporal relation."""

    relation_type: str
    frame_id: int
    timestamp_ms: int
    score: float
    reason_tags: tuple[str, ...]
    person_track_id: str | None = None
    hand_track_id: str | None = None
    item_track_id: str | None = None
    container_track_id: str | None = None
    evidence_boxes: dict[str, BBox] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.relation_type, "relation_type")
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        _validate_score(float(self.score))
        tags = tuple(self.reason_tags)
        if not tags:
            raise ValueError("reason_tags must contain at least one tag")
        for tag in tags:
            _validate_non_empty(tag, "reason_tags")
        normalized_boxes = {
            name: _normalize_bbox(box) for name, box in self.evidence_boxes.items()
        }
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "reason_tags", tags)
        object.__setattr__(self, "evidence_boxes", normalized_boxes)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class RiskEvent:
    """Structured, reviewable risk event emitted by the event engine."""

    event_id: str
    camera_id: str
    timestamp_ms: int
    person_track_id: str
    event_type: str
    risk_score: float
    risk_level: RiskLevel
    reason_tags: tuple[str, ...]
    evidence: tuple[RelationEvidence, ...]
    start_timestamp_ms: int | None = None
    end_timestamp_ms: int | None = None
    confidence: float | None = None
    clip_uri: str | None = None
    debug_visualization_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.event_id, "event_id")
        _validate_non_empty(self.camera_id, "camera_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")
        _validate_non_empty(self.person_track_id, "person_track_id")
        _validate_non_empty(self.event_type, "event_type")
        _validate_score(float(self.risk_score), "risk_score")
        if self.risk_level not in _RISK_LEVELS:
            raise ValueError("risk_level must be one of: low, medium, high")
        if self.start_timestamp_ms is not None:
            _validate_non_negative(self.start_timestamp_ms, "start_timestamp_ms")
        if self.end_timestamp_ms is not None:
            _validate_non_negative(self.end_timestamp_ms, "end_timestamp_ms")
        if (
            self.start_timestamp_ms is not None
            and self.end_timestamp_ms is not None
            and self.end_timestamp_ms < self.start_timestamp_ms
        ):
            raise ValueError("end_timestamp_ms must be greater than or equal to start_timestamp_ms")
        if self.confidence is not None:
            _validate_score(float(self.confidence), "confidence")
        tags = tuple(self.reason_tags)
        if not tags:
            raise ValueError("reason_tags must contain at least one tag")
        evidence = tuple(self.evidence)
        if not evidence:
            raise ValueError("evidence must contain at least one item")
        object.__setattr__(self, "risk_score", float(self.risk_score))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "reason_tags", tags)
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)
