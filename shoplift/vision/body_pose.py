"""Full-body pose evidence built from COCO-style keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from shoplift.core.types import BodyPose, FrameMeta, Point, Tracklet


COCO_PERSON_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_PERSON_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass(frozen=True)
class BodyPoseBuilder:
    """Bind keypoints to person tracks and expose pose-level confidence."""

    min_keypoint_score: float = 0.2
    keypoint_names: tuple[str, ...] = field(default_factory=lambda: COCO_PERSON_KEYPOINT_NAMES)
    skeleton_edges: tuple[tuple[int, int], ...] = field(default_factory=lambda: COCO_PERSON_SKELETON)
    source: str = "body_pose"

    def build(
        self,
        frame: FrameMeta,
        keypoints: Sequence[Sequence[Point]],
        scores: Sequence[Sequence[float]] | None = None,
        person_tracks: Sequence[Tracklet] | None = None,
    ) -> tuple[BodyPose, ...]:
        person_tracks = tuple(person_tracks or ())
        score_rows = tuple(scores or ())
        poses: list[BodyPose] = []
        for person_index, person_keypoints in enumerate(keypoints):
            person_track = person_tracks[person_index] if person_index < len(person_tracks) else None
            person_track_id = person_track.track_id if person_track is not None else f"person-{person_index}"
            person_bbox = person_track.boxes[-1].bbox if person_track and person_track.boxes else None
            person_scores = tuple(
                _safe_score(score)
                for score in (score_rows[person_index] if person_index < len(score_rows) else ())
            )
            pose_score, visible_count = self._pose_score(person_scores)
            poses.append(
                BodyPose(
                    pose_id=f"pose-{person_track_id}",
                    person_track_id=person_track_id,
                    frame_id=frame.frame_id,
                    timestamp_ms=frame.timestamp_ms,
                    keypoints=tuple(tuple(point) for point in person_keypoints),
                    scores=person_scores,
                    score=pose_score,
                    keypoint_names=self.keypoint_names,
                    skeleton_edges=self.skeleton_edges,
                    bbox=person_bbox,
                    metadata={
                        "source": self.source,
                        "person_index": person_index,
                        "visible_keypoint_count": visible_count,
                        "total_keypoint_count": len(person_keypoints),
                        "min_keypoint_score": self.min_keypoint_score,
                    },
                )
            )
        return tuple(poses)

    def _pose_score(self, scores: Sequence[float]) -> tuple[float, int]:
        visible_scores = [score for score in scores if score >= self.min_keypoint_score]
        if not visible_scores:
            return 0.0, 0
        return min(1.0, sum(visible_scores) / len(visible_scores)), len(visible_scores)


def _safe_score(value: float) -> float:
    score = float(value)
    return max(0.0, min(1.0, score))


__all__ = [
    "BodyPoseBuilder",
    "COCO_PERSON_KEYPOINT_NAMES",
    "COCO_PERSON_SKELETON",
]
