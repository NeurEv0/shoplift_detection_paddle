"""Temporal hand-item-container relation association for P1."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable, Sequence

from shoplift.core.types import BBox, BodyPose, DetectionBox, HandRegion, RelationEvidence, Tracklet


PRIVATE_CONTAINER_CATEGORIES = frozenset({"bag"})
NORMAL_CONTAINER_CATEGORIES = frozenset({"basket", "cart", "checkout_bag"})
SPECIAL_CONTAINER_CATEGORIES = frozenset({"stroller", "helmet"})
EXTENSION_REGION_CATEGORIES = frozenset({"clothing_region", "pocket_region"})


@dataclass(frozen=True)
class AssociationConfig:
    """Thresholds for relation association.

    Defaults mirror the P1 example config closely enough for unit tests and
    offline prototypes while keeping the module independent from YAML parsing.
    """

    min_contact_frames: int = 3
    min_entry_frames: int = 3
    disappeared_after_entry_frames: int = 10
    max_contact_distance_px: float = 32.0
    contact_distance_ratio: float = 0.75
    min_contact_iou: float = 0.01
    min_motion_cosine: float = 0.35
    max_follow_distance_px: float = 160.0
    follow_score_threshold: float = 0.45
    ownership_conflict_margin: float = 0.12
    max_missing_frames: int = 5
    max_frame_gap: int = 1
    min_entry_overlap_ratio: float = 0.25
    low_confidence_score: float = 0.35
    normal_container_categories: tuple[str, ...] = ("basket", "cart", "checkout_bag")

    def __post_init__(self) -> None:
        if self.min_contact_frames <= 0:
            raise ValueError("min_contact_frames must be positive")
        if self.min_entry_frames <= 0:
            raise ValueError("min_entry_frames must be positive")
        if self.disappeared_after_entry_frames <= 0:
            raise ValueError("disappeared_after_entry_frames must be positive")
        if self.max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")


@dataclass(frozen=True)
class TrackedDetection:
    """Detection with a stable short-term track id."""

    detection: DetectionBox
    track_id: str


@dataclass(frozen=True)
class AssociationFrame:
    """One frame of structured visual evidence for relation association."""

    frame_id: int
    timestamp_ms: int
    camera_id: str
    person_tracks: tuple[Tracklet, ...] = field(default_factory=tuple)
    body_poses: tuple[BodyPose, ...] = field(default_factory=tuple)
    hand_regions: tuple[HandRegion, ...] = field(default_factory=tuple)
    items: tuple[DetectionBox, ...] = field(default_factory=tuple)
    containers: tuple[DetectionBox, ...] = field(default_factory=tuple)
    extension_regions: tuple[DetectionBox, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AssociationResult:
    """Relations inferred for one frame."""

    frame_id: int
    timestamp_ms: int
    camera_id: str
    relations: tuple[RelationEvidence, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def by_type(self, relation_type: str) -> tuple[RelationEvidence, ...]:
        return tuple(evidence for evidence in self.relations if evidence.relation_type == relation_type)


@dataclass
class _ItemTrackState:
    track_id: str
    last_box: DetectionBox
    last_frame_id: int
    missing_frames: int = 0


@dataclass
class _ContactState:
    consecutive_frames: int = 0
    last_frame_id: int | None = None
    last_hand_center: tuple[float, float] | None = None
    last_item_center: tuple[float, float] | None = None


@dataclass
class _OwnershipState:
    person_track_id: str
    item_track_id: str
    last_item_box: DetectionBox
    last_person_box: DetectionBox | None
    last_frame_id: int
    score: float
    missing_frames: int = 0
    evidence_frames: int = 1


@dataclass
class _EntryState:
    consecutive_frames: int = 0
    last_frame_id: int | None = None


@dataclass
class _DisappearanceState:
    item_track_id: str
    container_track_id: str
    person_track_id: str | None
    entry_evidence: RelationEvidence
    last_visible_frame_id: int
    missing_frames: int = 0
    reported: bool = False


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def bbox_intersection_area(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def bbox_iou(first: BBox, second: BBox) -> float:
    intersection = bbox_intersection_area(first, second)
    union = bbox_area(first) + bbox_area(second) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def bbox_distance(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return sqrt(dx * dx + dy * dy)


def point_bbox_distance(point: tuple[float, float], bbox: BBox) -> float:
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, x - x2, 0.0)
    dy = max(y1 - y, y - y2, 0.0)
    return sqrt(dx * dx + dy * dy)


def expand_bbox(bbox: BBox, padding: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 - padding, y1 - padding, x2 + padding, y2 + padding)


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def cosine_similarity(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    min_magnitude: float = 1.0,
) -> float | None:
    ax, ay = first
    bx, by = second
    first_mag = sqrt(ax * ax + ay * ay)
    second_mag = sqrt(bx * bx + by * by)
    if first_mag < min_magnitude or second_mag < min_magnitude:
        return None
    return (ax * bx + ay * by) / (first_mag * second_mag)


def latest_track_box(tracklet: Tracklet) -> DetectionBox | None:
    if not tracklet.boxes:
        return None
    return max(tracklet.boxes, key=lambda box: box.frame_id)


def detection_track_id(detection: DetectionBox, prefix: str = "det") -> str:
    if detection.track_id:
        return detection.track_id
    metadata_id = detection.attributes.get("track_id") or detection.attributes.get(f"{prefix}_track_id")
    if metadata_id:
        return str(metadata_id)
    return detection.box_id


def container_kind(detection: DetectionBox, config: AssociationConfig | None = None) -> str:
    category = str(detection.category).strip().lower()
    source_category = str(detection.attributes.get("source_category", "")).strip().lower()
    normal_categories = set((config or AssociationConfig()).normal_container_categories)
    if detection.attributes.get("is_normal_container") is True:
        return "normal"
    if category in normal_categories or source_category in normal_categories:
        return "normal"
    if category in PRIVATE_CONTAINER_CATEGORIES:
        return "private"
    if category in SPECIAL_CONTAINER_CATEGORIES:
        return "special"
    if category in EXTENSION_REGION_CATEGORIES:
        return "clothing"
    return "unknown"


def _relation_key(evidence: RelationEvidence) -> tuple[str | None, str | None]:
    return evidence.person_track_id, evidence.item_track_id


class ItemTrackStitcher:
    """Assign short-term item ids when the detector does not provide them."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._states: dict[str, _ItemTrackState] = {}
        self._next_id = 1

    def update(self, items: Sequence[DetectionBox], frame_id: int) -> tuple[TrackedDetection, ...]:
        assigned: list[TrackedDetection] = []
        used_track_ids: set[str] = set()

        for item in items:
            explicit_id = item.track_id or item.attributes.get("item_track_id")
            if explicit_id:
                track_id = str(explicit_id)
            else:
                track_id = self._match_existing_track(item, frame_id, used_track_ids)
                if track_id is None:
                    track_id = f"item-{self._next_id}"
                    self._next_id += 1

            used_track_ids.add(track_id)
            self._states[track_id] = _ItemTrackState(
                track_id=track_id,
                last_box=item,
                last_frame_id=frame_id,
                missing_frames=0,
            )
            assigned.append(TrackedDetection(detection=item, track_id=track_id))

        for track_id in list(self._states):
            if track_id in used_track_ids:
                continue
            state = self._states[track_id]
            state.missing_frames += 1
            if state.missing_frames > self.config.max_missing_frames:
                del self._states[track_id]

        return tuple(assigned)

    def _match_existing_track(
        self,
        item: DetectionBox,
        frame_id: int,
        used_track_ids: set[str],
    ) -> str | None:
        best_track_id: str | None = None
        best_score = 0.0
        item_center = item.center
        for track_id, state in self._states.items():
            if track_id in used_track_ids:
                continue
            if frame_id - state.last_frame_id > self.config.max_missing_frames + 1:
                continue
            distance = _point_distance(item_center, state.last_box.center)
            iou = bbox_iou(item.bbox, state.last_box.bbox)
            if distance > self.config.max_follow_distance_px and iou <= 0.0:
                continue
            score = iou + max(0.0, 1.0 - distance / self.config.max_follow_distance_px)
            if score > best_score:
                best_score = score
                best_track_id = track_id
        return best_track_id


class HandItemContactAssociator:
    """Detect stable hand-item contact relations."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._states: dict[tuple[str, str, str], _ContactState] = {}

    def update(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        hands: Sequence[HandRegion],
        items: Sequence[TrackedDetection],
    ) -> tuple[RelationEvidence, ...]:
        evidences: list[RelationEvidence] = []
        active_keys: set[tuple[str, str, str]] = set()

        for hand in hands:
            for tracked_item in items:
                item = tracked_item.detection
                item_track_id = tracked_item.track_id
                key = (hand.person_track_id, hand.hand_track_id, item_track_id)
                active_keys.add(key)
                state = self._states.setdefault(key, _ContactState())
                metrics = self._contact_metrics(hand, item, state)

                if metrics["is_contact"]:
                    if state.last_frame_id is not None and frame_id - state.last_frame_id <= self.config.max_frame_gap:
                        state.consecutive_frames += 1
                    else:
                        state.consecutive_frames = 1
                    state.last_frame_id = frame_id
                    state.last_hand_center = bbox_center(hand.bbox)
                    state.last_item_center = item.center
                else:
                    state.consecutive_frames = 0
                    state.last_frame_id = frame_id
                    state.last_hand_center = bbox_center(hand.bbox)
                    state.last_item_center = item.center
                    continue

                if state.consecutive_frames < self.config.min_contact_frames:
                    continue

                tags = list(metrics["reason_tags"])
                tags.append("temporal_consistent")
                motion_similarity = metrics["motion_similarity"]
                if motion_similarity is not None and motion_similarity >= self.config.min_motion_cosine:
                    tags.append("motion_aligned")
                if hand.score < self.config.low_confidence_score or item.score < self.config.low_confidence_score:
                    tags.append("low_confidence")

                score = min(
                    1.0,
                    float(metrics["base_score"])
                    + 0.2 * min(1.0, state.consecutive_frames / max(1, self.config.min_contact_frames))
                    + (0.1 if "motion_aligned" in tags else 0.0),
                )
                if "low_confidence" in tags:
                    score = min(score, 0.55)

                evidences.append(
                    RelationEvidence(
                        relation_type="hand_item_contact",
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        score=score,
                        reason_tags=tuple(dict.fromkeys(tags)),
                        person_track_id=hand.person_track_id,
                        hand_track_id=hand.hand_track_id,
                        item_track_id=item_track_id,
                        evidence_boxes={"hand": hand.bbox, "item": item.bbox},
                        metadata={
                            "contact_frames": state.consecutive_frames,
                            "distance_px": metrics["distance_px"],
                            "iou": metrics["iou"],
                            "motion_similarity": motion_similarity,
                            "item_box_id": item.box_id,
                        },
                    )
                )

        self._drop_inactive(active_keys)
        return tuple(evidences)

    def _contact_metrics(
        self,
        hand: HandRegion,
        item: DetectionBox,
        state: _ContactState,
    ) -> dict[str, object]:
        distance = bbox_distance(hand.bbox, item.bbox)
        hand_w, hand_h = bbox_size(hand.bbox)
        item_w, item_h = bbox_size(item.bbox)
        size_threshold = max(hand_w, hand_h, item_w, item_h) * self.config.contact_distance_ratio
        threshold = max(self.config.max_contact_distance_px, size_threshold)
        iou = bbox_iou(hand.bbox, item.bbox)
        hand_center_in_item = point_in_bbox(bbox_center(hand.bbox), expand_bbox(item.bbox, 4.0))
        item_center_in_hand = point_in_bbox(item.center, expand_bbox(hand.bbox, 4.0))

        is_close = distance <= threshold
        is_overlap = iou >= self.config.min_contact_iou or hand_center_in_item or item_center_in_hand
        is_contact = is_close or is_overlap

        tags: list[str] = []
        if is_close:
            tags.append("hand_item_distance_close")
        if is_overlap:
            tags.append("hand_item_overlap")

        motion_similarity: float | None = None
        if state.last_hand_center is not None and state.last_item_center is not None:
            current_hand_center = bbox_center(hand.bbox)
            current_item_center = item.center
            hand_vector = (
                current_hand_center[0] - state.last_hand_center[0],
                current_hand_center[1] - state.last_hand_center[1],
            )
            item_vector = (
                current_item_center[0] - state.last_item_center[0],
                current_item_center[1] - state.last_item_center[1],
            )
            motion_similarity = cosine_similarity(hand_vector, item_vector)

        distance_score = max(0.0, 1.0 - distance / max(1.0, threshold))
        overlap_score = min(1.0, iou * 4.0 + (0.25 if hand_center_in_item or item_center_in_hand else 0.0))
        base_score = 0.55 * distance_score + 0.35 * overlap_score
        if motion_similarity is not None:
            base_score += 0.1 * max(0.0, motion_similarity)

        return {
            "is_contact": is_contact,
            "reason_tags": tags or ["hand_item_near"],
            "distance_px": distance,
            "iou": iou,
            "motion_similarity": motion_similarity,
            "base_score": min(1.0, base_score),
        }

    def _drop_inactive(self, active_keys: set[tuple[str, str, str]]) -> None:
        for key in list(self._states):
            state = self._states[key]
            if key not in active_keys and state.consecutive_frames == 0:
                del self._states[key]


class ItemFollowPersonAssociator:
    """Associate item tracks with the most likely person track."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._states: dict[str, _OwnershipState] = {}

    def update(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        person_tracks: Sequence[Tracklet],
        items: Sequence[TrackedDetection],
        contact_evidence: Sequence[RelationEvidence] = (),
    ) -> tuple[RelationEvidence, ...]:
        person_boxes = {
            person.track_id: box
            for person in person_tracks
            if (box := latest_track_box(person)) is not None
        }
        contact_by_item = {
            evidence.item_track_id: evidence
            for evidence in contact_evidence
            if evidence.item_track_id and evidence.person_track_id
        }
        evidences: list[RelationEvidence] = []
        seen_item_ids: set[str] = set()

        for tracked in items:
            item = tracked.detection
            item_track_id = tracked.track_id
            seen_item_ids.add(item_track_id)
            best = self._best_person_for_item(
                item=item,
                item_track_id=item_track_id,
                person_boxes=person_boxes,
                contact_evidence=contact_by_item.get(item_track_id),
            )
            if best is None:
                continue
            person_track_id, person_box, score, tags, ambiguous = best
            if ambiguous or score < self.config.follow_score_threshold:
                continue

            previous = self._states.get(item_track_id)
            evidence_frames = (previous.evidence_frames + 1) if previous and previous.person_track_id == person_track_id else 1
            self._states[item_track_id] = _OwnershipState(
                person_track_id=person_track_id,
                item_track_id=item_track_id,
                last_item_box=item,
                last_person_box=person_box,
                last_frame_id=frame_id,
                score=score,
                missing_frames=0,
                evidence_frames=evidence_frames,
            )

            evidences.append(
                RelationEvidence(
                    relation_type="item_follow_person",
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    score=score,
                    reason_tags=tuple(dict.fromkeys(tags)),
                    person_track_id=person_track_id,
                    item_track_id=item_track_id,
                    evidence_boxes={"item": item.bbox, "person": person_box.bbox},
                    metadata={
                        "follow_frames": evidence_frames,
                        "item_box_id": item.box_id,
                        "assignment_ambiguous": False,
                    },
                )
            )

        for item_track_id, state in list(self._states.items()):
            if item_track_id in seen_item_ids:
                continue
            state.missing_frames += 1
            if state.missing_frames > self.config.max_missing_frames:
                del self._states[item_track_id]
                continue
            if frame_id - state.last_frame_id > self.config.max_missing_frames + 1:
                del self._states[item_track_id]
                continue
            boxes = {"item": state.last_item_box.bbox}
            if state.last_person_box is not None:
                boxes["person"] = state.last_person_box.bbox
            evidences.append(
                RelationEvidence(
                    relation_type="item_follow_person",
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    score=max(0.1, state.score * 0.7),
                    reason_tags=("item_owned_by_person", "gap_filled"),
                    person_track_id=state.person_track_id,
                    item_track_id=item_track_id,
                    evidence_boxes=boxes,
                    metadata={
                        "follow_frames": state.evidence_frames,
                        "missing_frames": state.missing_frames,
                        "gap_filled": True,
                    },
                )
            )

        return tuple(evidences)

    def person_for_items(self) -> dict[str, str]:
        return {item_id: state.person_track_id for item_id, state in self._states.items()}

    def _best_person_for_item(
        self,
        *,
        item: DetectionBox,
        item_track_id: str,
        person_boxes: dict[str, DetectionBox],
        contact_evidence: RelationEvidence | None,
    ) -> tuple[str, DetectionBox, float, list[str], bool] | None:
        candidates: list[tuple[str, DetectionBox, float, list[str]]] = []
        previous = self._states.get(item_track_id)

        for person_track_id, person_box in person_boxes.items():
            score = 0.0
            tags: list[str] = []
            distance = _point_distance(item.center, person_box.center)
            distance_score = max(0.0, 1.0 - distance / max(1.0, self.config.max_follow_distance_px))
            expanded_person = expand_bbox(person_box.bbox, max(16.0, max(bbox_size(item.bbox))))

            if contact_evidence and contact_evidence.person_track_id == person_track_id:
                score += 0.45 * contact_evidence.score
                tags.append("hand_item_contact_owner")
            if previous and previous.person_track_id == person_track_id:
                score += 0.25
                tags.append("previous_item_owner")
            if point_in_bbox(item.center, expanded_person):
                score += 0.2
                tags.append("item_near_person")
            if distance_score > 0.0:
                score += 0.2 * distance_score

            motion_similarity = None
            if previous and previous.person_track_id == person_track_id and previous.last_person_box:
                item_vector = (
                    item.center[0] - previous.last_item_box.center[0],
                    item.center[1] - previous.last_item_box.center[1],
                )
                person_vector = (
                    person_box.center[0] - previous.last_person_box.center[0],
                    person_box.center[1] - previous.last_person_box.center[1],
                )
                motion_similarity = cosine_similarity(item_vector, person_vector)
                if motion_similarity is not None and motion_similarity >= self.config.min_motion_cosine:
                    score += 0.1 * motion_similarity
                    tags.append("follow_motion_aligned")

            if tags:
                tags.insert(0, "item_owned_by_person")
                candidates.append((person_track_id, person_box, min(1.0, score), tags))

        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate[2], reverse=True)
        best = candidates[0]
        second_score = candidates[1][2] if len(candidates) > 1 else -1.0
        ambiguous = second_score >= 0.0 and best[2] - second_score < self.config.ownership_conflict_margin
        return best[0], best[1], best[2], best[3], ambiguous


class ContainerEntryDetector:
    """Detect item entry into containers or extension regions."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._states: dict[tuple[str, str], _EntryState] = {}

    def update(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        items: Sequence[TrackedDetection],
        containers: Sequence[DetectionBox],
        person_by_item: dict[str, str] | None = None,
    ) -> tuple[RelationEvidence, ...]:
        person_by_item = person_by_item or {}
        evidences: list[RelationEvidence] = []
        active_keys: set[tuple[str, str]] = set()

        for tracked_item in items:
            item = tracked_item.detection
            item_track_id = tracked_item.track_id
            for container in containers:
                container_track_id = detection_track_id(container, "container")
                key = (item_track_id, container_track_id)
                metrics = self._entry_metrics(item, container)
                if not metrics["is_entry"]:
                    if key in self._states:
                        self._states[key].consecutive_frames = 0
                    continue

                active_keys.add(key)
                state = self._states.setdefault(key, _EntryState())
                if state.last_frame_id is not None and frame_id - state.last_frame_id <= self.config.max_frame_gap:
                    state.consecutive_frames += 1
                else:
                    state.consecutive_frames = 1
                state.last_frame_id = frame_id
                if state.consecutive_frames < self.config.min_entry_frames:
                    continue

                kind = container_kind(container, self.config)
                tags = self._entry_tags(kind)
                if state.consecutive_frames >= self.config.min_entry_frames:
                    tags.append("entry_temporal_consistent")
                if item.score < self.config.low_confidence_score or container.score < self.config.low_confidence_score:
                    tags.append("low_confidence")

                score = min(
                    1.0,
                    float(metrics["base_score"])
                    + 0.2 * min(1.0, state.consecutive_frames / max(1, self.config.min_entry_frames)),
                )
                if kind == "normal":
                    score = min(score, 0.45)
                if "low_confidence" in tags:
                    score = min(score, 0.55)

                evidences.append(
                    RelationEvidence(
                        relation_type="item_enter_container",
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        score=score,
                        reason_tags=tuple(dict.fromkeys(tags)),
                        person_track_id=person_by_item.get(item_track_id),
                        item_track_id=item_track_id,
                        container_track_id=container_track_id,
                        evidence_boxes={"item": item.bbox, "container": container.bbox},
                        metadata={
                            "entry_frames": state.consecutive_frames,
                            "container_category": container.category,
                            "container_kind": kind,
                            "is_normal_container": kind == "normal",
                            "item_box_id": item.box_id,
                            "container_box_id": container.box_id,
                            "item_center_inside_container": metrics["center_inside"],
                            "item_overlap_ratio": metrics["overlap_ratio"],
                        },
                    )
                )

        for key in list(self._states):
            if key not in active_keys and self._states[key].consecutive_frames == 0:
                del self._states[key]
        return tuple(evidences)

    def _entry_metrics(self, item: DetectionBox, container: DetectionBox) -> dict[str, object]:
        intersection = bbox_intersection_area(item.bbox, container.bbox)
        item_area = max(1.0, bbox_area(item.bbox))
        overlap_ratio = intersection / item_area
        center_inside = point_in_bbox(item.center, container.bbox)
        is_entry = center_inside or overlap_ratio >= self.config.min_entry_overlap_ratio
        base_score = 0.45 * (1.0 if center_inside else 0.0) + 0.35 * min(1.0, overlap_ratio)
        return {
            "is_entry": is_entry,
            "center_inside": center_inside,
            "overlap_ratio": overlap_ratio,
            "base_score": base_score,
        }

    @staticmethod
    def _entry_tags(kind: str) -> list[str]:
        if kind == "normal":
            return ["entered_normal_container", "normal_container"]
        if kind == "special":
            return ["entered_special_container"]
        if kind == "clothing":
            return ["entered_clothing_region"]
        if kind == "private":
            return ["entered_private_container"]
        return ["entered_container"]


class DisappearanceAfterEntryDetector:
    """Detect item disappearance after a container-entry relation."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._states: dict[tuple[str, str], _DisappearanceState] = {}

    def update(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        visible_item_ids: Iterable[str],
        entry_evidence: Sequence[RelationEvidence] = (),
    ) -> tuple[RelationEvidence, ...]:
        visible = set(visible_item_ids)
        for evidence in entry_evidence:
            if not evidence.item_track_id or not evidence.container_track_id:
                continue
            key = (evidence.item_track_id, evidence.container_track_id)
            state = self._states.get(key)
            if state is None:
                self._states[key] = _DisappearanceState(
                    item_track_id=evidence.item_track_id,
                    container_track_id=evidence.container_track_id,
                    person_track_id=evidence.person_track_id,
                    entry_evidence=evidence,
                    last_visible_frame_id=frame_id,
                )
            else:
                state.entry_evidence = evidence
                state.person_track_id = evidence.person_track_id or state.person_track_id
                state.last_visible_frame_id = frame_id
                state.missing_frames = 0
                state.reported = False

        evidences: list[RelationEvidence] = []
        for key, state in list(self._states.items()):
            item_track_id, container_track_id = key
            if item_track_id in visible:
                state.last_visible_frame_id = frame_id
                state.missing_frames = 0
                state.reported = False
                continue

            if frame_id <= state.last_visible_frame_id:
                continue
            state.missing_frames += 1
            if state.missing_frames < self.config.disappeared_after_entry_frames:
                continue
            if state.reported:
                continue

            tags = ["item_disappeared", "after_container_entry"]
            entry_tags = set(state.entry_evidence.reason_tags)
            if "entered_normal_container" in entry_tags or state.entry_evidence.metadata.get("is_normal_container"):
                tags.append("normal_container_exempted")
                score = 0.25
            else:
                if "entered_private_container" in entry_tags:
                    tags.append("after_private_container_entry")
                if "entered_special_container" in entry_tags:
                    tags.append("after_special_container_entry")
                if "entered_clothing_region" in entry_tags:
                    tags.append("after_clothing_region_entry")
                if "low_confidence" in entry_tags:
                    tags.append("low_visibility")
                score = min(1.0, state.entry_evidence.score + 0.15)
                if "low_visibility" in tags:
                    score = min(score, 0.55)

            boxes = dict(state.entry_evidence.evidence_boxes)
            evidences.append(
                RelationEvidence(
                    relation_type="item_disappeared_after_entry",
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    score=score,
                    reason_tags=tuple(dict.fromkeys(tags)),
                    person_track_id=state.person_track_id,
                    item_track_id=item_track_id,
                    container_track_id=container_track_id,
                    evidence_boxes=boxes,
                    metadata={
                        "missing_frames": state.missing_frames,
                        "entry_frame_id": state.entry_evidence.frame_id,
                        "entry_timestamp_ms": state.entry_evidence.timestamp_ms,
                        "container_kind": state.entry_evidence.metadata.get("container_kind"),
                        "is_normal_container": state.entry_evidence.metadata.get("is_normal_container", False),
                    },
                )
            )
            state.reported = True

        return tuple(evidences)


class ShopliftingRelationAssociator:
    """Run P1 relation detectors over frame-level structured results."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self.item_stitcher = ItemTrackStitcher(self.config)
        self.hand_item_contact = HandItemContactAssociator(self.config)
        self.item_follow_person = ItemFollowPersonAssociator(self.config)
        self.container_entry = ContainerEntryDetector(self.config)
        self.disappearance_after_entry = DisappearanceAfterEntryDetector(self.config)

    def update(self, frame: AssociationFrame) -> AssociationResult:
        tracked_items = self.item_stitcher.update(frame.items, frame.frame_id)
        relation_groups: list[tuple[RelationEvidence, ...]] = []

        hand_contact = self.hand_item_contact.update(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            hands=frame.hand_regions,
            items=tracked_items,
        )
        contact = _dedupe_contact_evidence(hand_contact)
        relation_groups.append(contact)

        follow = self.item_follow_person.update(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            person_tracks=frame.person_tracks,
            items=tracked_items,
            contact_evidence=contact,
        )
        relation_groups.append(follow)

        person_by_item = self.item_follow_person.person_for_items()
        all_containers = tuple(frame.containers) + tuple(frame.extension_regions)
        entry = self.container_entry.update(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            items=tracked_items,
            containers=all_containers,
            person_by_item=person_by_item,
        )
        relation_groups.append(entry)

        disappeared = self.disappearance_after_entry.update(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            visible_item_ids=(tracked.track_id for tracked in tracked_items),
            entry_evidence=entry,
        )
        relation_groups.append(disappeared)

        relations = tuple(evidence for group in relation_groups for evidence in group)
        return AssociationResult(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            camera_id=frame.camera_id,
            relations=relations,
            metadata={
                "tracked_item_count": len(tracked_items),
                "relation_count": len(relations),
                "relation_counts": _count_by_relation_type(relations),
            },
        )


def _count_by_relation_type(relations: Sequence[RelationEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relation in relations:
        counts[relation.relation_type] = counts.get(relation.relation_type, 0) + 1
    return counts


def _dedupe_contact_evidence(evidences: Sequence[RelationEvidence]) -> tuple[RelationEvidence, ...]:
    best_by_key: dict[tuple[str | None, str | None, int], RelationEvidence] = {}
    for evidence in evidences:
        key = (evidence.person_track_id, evidence.item_track_id, evidence.frame_id)
        previous = best_by_key.get(key)
        if previous is None or evidence.score > previous.score:
            best_by_key[key] = evidence
    return tuple(best_by_key.values())


def _point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "AssociationResult",
    "ContainerEntryDetector",
    "DetectionBox",
    "DisappearanceAfterEntryDetector",
    "HandItemContactAssociator",
    "HandRegion",
    "ItemFollowPersonAssociator",
    "ItemTrackStitcher",
    "RelationEvidence",
    "ShopliftingRelationAssociator",
    "TrackedDetection",
    "Tracklet",
    "bbox_area",
    "bbox_center",
    "bbox_distance",
    "bbox_intersection_area",
    "bbox_iou",
    "container_kind",
    "detection_track_id",
    "point_bbox_distance",
]
