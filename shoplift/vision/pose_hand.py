"""Hand ROI extraction from body pose keypoints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shoplift.core.types import BBox, FrameMeta, HandRegion, Point, Tracklet


COCO_KEYPOINT_INDICES: Mapping[str, int] = {
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
}


@dataclass(frozen=True)
class PersonPose:
    """Normalized body keypoints for one tracked person."""

    person_track_id: str
    keypoints: tuple[Point, ...]
    scores: tuple[float, ...] = field(default_factory=tuple)
    person_bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.person_track_id:
            raise ValueError("person_track_id must be a non-empty string")
        object.__setattr__(self, "keypoints", tuple(_normalize_point(point) for point in self.keypoints))
        object.__setattr__(self, "scores", tuple(_safe_score(score) for score in self.scores))
        if self.person_bbox is not None:
            object.__setattr__(self, "person_bbox", _normalize_bbox(self.person_bbox))


@dataclass(frozen=True)
class HandRegionExtractor:
    """Derive left/right hand regions from wrist and arm keypoints."""

    min_keypoint_score: float = 0.2
    hand_min_size_px: float = 18.0
    forearm_extension_ratio: float = 0.4
    hand_side_from_forearm: float = 1.1
    hand_scale_from_forearm: float | None = None
    hand_scale_from_person: float = 0.08
    min_forearm_length_px: float = 8.0
    keypoint_indices: Mapping[str, int] = field(default_factory=lambda: dict(COCO_KEYPOINT_INDICES))
    source: str = "pose_hand"

    def extract(
        self,
        frame: FrameMeta,
        person_poses: Sequence[PersonPose],
    ) -> tuple[HandRegion, ...]:
        hand_regions: list[HandRegion] = []
        for person_index, pose in enumerate(person_poses):
            hand_regions.extend(self.extract_for_person(frame, pose, person_index=person_index))
        return tuple(hand_regions)

    def extract_for_person(
        self,
        frame: FrameMeta,
        person_pose: PersonPose,
        *,
        person_index: int = 0,
    ) -> tuple[HandRegion, ...]:
        regions: list[HandRegion] = []
        for side in ("left", "right"):
            region = self._extract_side(frame, person_pose, side=side, person_index=person_index)
            if region is not None:
                regions.append(region)
        return tuple(regions)

    def _extract_side(
        self,
        frame: FrameMeta,
        person_pose: PersonPose,
        *,
        side: str,
        person_index: int,
    ) -> HandRegion | None:
        wrist_name = f"{side}_wrist"
        elbow_name = f"{side}_elbow"
        wrist_index = self.keypoint_indices[wrist_name]
        elbow_index = self.keypoint_indices[elbow_name]
        wrist = _point_at(person_pose.keypoints, wrist_index)
        if wrist is None:
            return None

        wrist_score = _score_at(person_pose.scores, wrist_index)
        if wrist_score < self.min_keypoint_score:
            return None

        elbow = _point_at(person_pose.keypoints, elbow_index)
        elbow_score = _score_at(person_pose.scores, elbow_index, default=0.0)
        crop = self._hand_crop(wrist, elbow, elbow_score, person_pose.person_bbox)
        bbox = _clip_bbox(
            (
                crop.center[0] - crop.side_length / 2.0,
                crop.center[1] - crop.side_length / 2.0,
                crop.center[0] + crop.side_length / 2.0,
                crop.center[1] + crop.side_length / 2.0,
            ),
            frame,
        )
        source_points = (wrist, elbow) if elbow is not None and elbow_score >= self.min_keypoint_score else (wrist,)
        score = min(1.0, (wrist_score + max(elbow_score, wrist_score)) / 2.0)
        return HandRegion(
            hand_track_id=f"hand-{person_pose.person_track_id}-{side}",
            person_track_id=person_pose.person_track_id,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            side=side,
            bbox=bbox,
            score=score,
            source_keypoints=source_points,
            metadata={
                **person_pose.metadata,
                "source": self.source,
                "person_index": person_index,
                "wrist_index": wrist_index,
                "wrist_score": wrist_score,
                "elbow_index": elbow_index,
                "elbow_score": elbow_score,
                "crop_strategy": crop.strategy,
                "crop_center": crop.center,
                "crop_side_length": crop.side_length,
                "forearm_vector": crop.forearm_vector,
                "forearm_length": crop.forearm_length,
                "forearm_angle_deg": crop.angle_deg,
                "axis_aligned": True,
                "forearm_extension_ratio": self.forearm_extension_ratio,
                "hand_side_from_forearm": self._effective_hand_side_from_forearm(),
            },
        )

    def _hand_crop(
        self,
        wrist: Point,
        elbow: Point | None,
        elbow_score: float,
        person_bbox: BBox | None,
    ) -> "_HandCrop":
        if elbow is not None and elbow_score >= self.min_keypoint_score:
            vector = (wrist[0] - elbow[0], wrist[1] - elbow[1])
            forearm_length = math.hypot(vector[0], vector[1])
            if forearm_length >= self.min_forearm_length_px:
                center = (
                    wrist[0] + vector[0] * self.forearm_extension_ratio,
                    wrist[1] + vector[1] * self.forearm_extension_ratio,
                )
                return _HandCrop(
                    center=center,
                    side_length=max(self.hand_min_size_px, forearm_length * self._effective_hand_side_from_forearm()),
                    strategy="forearm_guided_axis_aligned",
                    forearm_vector=vector,
                    forearm_length=forearm_length,
                    angle_deg=math.degrees(math.atan2(vector[1], vector[0])),
                )

        side_length = self.hand_min_size_px
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            side_length = max(side_length, min(x2 - x1, y2 - y1) * self.hand_scale_from_person * 2.0)
        return _HandCrop(
            center=wrist,
            side_length=side_length,
            strategy="wrist_fallback_axis_aligned",
            forearm_vector=None,
            forearm_length=None,
            angle_deg=None,
        )

    def _effective_hand_side_from_forearm(self) -> float:
        if self.hand_scale_from_forearm is not None:
            return self.hand_scale_from_forearm * 2.0
        return self.hand_side_from_forearm


@dataclass(frozen=True)
class _HandCrop:
    center: Point
    side_length: float
    strategy: str
    forearm_vector: Point | None
    forearm_length: float | None
    angle_deg: float | None


def build_person_poses(
    keypoints: Sequence[Sequence[Point]],
    scores: Sequence[Sequence[float]] | None = None,
    person_tracks: Sequence[Tracklet] | None = None,
) -> tuple[PersonPose, ...]:
    """Bind normalized per-person keypoints to person track ids by index."""

    person_tracks = tuple(person_tracks or ())
    score_rows = tuple(scores or ())
    poses: list[PersonPose] = []
    for person_index, person_keypoints in enumerate(keypoints):
        person_track = person_tracks[person_index] if person_index < len(person_tracks) else None
        person_track_id = person_track.track_id if person_track is not None else f"person-{person_index}"
        person_bbox = person_track.boxes[-1].bbox if person_track and person_track.boxes else None
        person_scores = score_rows[person_index] if person_index < len(score_rows) else ()
        poses.append(
            PersonPose(
                person_track_id=person_track_id,
                keypoints=tuple(person_keypoints),
                scores=tuple(person_scores),
                person_bbox=person_bbox,
                metadata={"person_index": person_index},
            )
        )
    return tuple(poses)


def extract_hand_regions(
    frame: FrameMeta,
    keypoints: Sequence[Sequence[Point]],
    scores: Sequence[Sequence[float]] | None = None,
    person_tracks: Sequence[Tracklet] | None = None,
    *,
    min_keypoint_score: float = 0.2,
) -> tuple[HandRegion, ...]:
    extractor = HandRegionExtractor(min_keypoint_score=min_keypoint_score)
    return extractor.extract(frame, build_person_poses(keypoints, scores, person_tracks))


def _point_at(points: Sequence[Point], index: int) -> Point | None:
    if index >= len(points):
        return None
    point = points[index]
    if point[0] <= 0.0 and point[1] <= 0.0:
        return None
    return point


def _score_at(scores: Sequence[float], index: int, default: float = 1.0) -> float:
    if len(scores) == 1:
        return scores[0]
    if index >= len(scores):
        return default
    return scores[index]


def _normalize_point(point: Point) -> Point:
    if len(point) < 2:
        raise ValueError("keypoints must contain at least two coordinates")
    return (float(point[0]), float(point[1]))


def _normalize_bbox(bbox: BBox) -> BBox:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must be ordered as [x1, y1, x2, y2]")
    return (x1, y1, x2, y2)


def _clip_bbox(bbox: BBox, frame: FrameMeta) -> BBox:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(frame.width), x1)),
        max(0.0, min(float(frame.height), y1)),
        max(0.0, min(float(frame.width), x2)),
        max(0.0, min(float(frame.height), y2)),
    )


def _safe_score(value: float) -> float:
    score = float(value)
    return max(0.0, min(1.0, score))


__all__ = [
    "COCO_KEYPOINT_INDICES",
    "HandRegion",
    "HandRegionExtractor",
    "PersonPose",
    "build_person_poses",
    "extract_hand_regions",
]
