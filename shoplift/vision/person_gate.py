"""Person detection and tracking gate.

The gate is the first cheap decision point in the offline prototype: if no
person-level evidence is present, downstream heavy modules such as keypoint,
item/container detection, and relation analysis can be skipped for that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from shoplift.core.types import DetectionBox, FrameMeta, Tracklet


@dataclass(frozen=True)
class PersonGateResult:
    frame: FrameMeta | None
    has_person: bool
    should_run_heavy_modules: bool
    skipped_heavy_modules: bool
    person_boxes: tuple[DetectionBox, ...]
    person_tracklets: tuple[Tracklet, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def person_track_ids(self) -> tuple[str, ...]:
        return tuple(tracklet.track_id for tracklet in self.person_tracklets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict() if self.frame is not None else None,
            "has_person": self.has_person,
            "should_run_heavy_modules": self.should_run_heavy_modules,
            "skipped_heavy_modules": self.skipped_heavy_modules,
            "person_boxes": [box.to_dict() for box in self.person_boxes],
            "person_tracklets": [tracklet.to_dict() for tracklet in self.person_tracklets],
            "person_track_ids": list(self.person_track_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PersonGateMetrics:
    total_frames: int = 0
    triggered_frames: int = 0
    skipped_frames: int = 0
    person_box_count: int = 0
    person_track_count: int = 0

    @property
    def trigger_rate(self) -> float:
        return self.triggered_frames / self.total_frames if self.total_frames else 0.0

    @property
    def skip_rate(self) -> float:
        return self.skipped_frames / self.total_frames if self.total_frames else 0.0

    def update(self, result: PersonGateResult) -> "PersonGateMetrics":
        return PersonGateMetrics(
            total_frames=self.total_frames + 1,
            triggered_frames=self.triggered_frames + int(result.has_person),
            skipped_frames=self.skipped_frames + int(result.skipped_heavy_modules),
            person_box_count=self.person_box_count + len(result.person_boxes),
            person_track_count=self.person_track_count + len(result.person_tracklets),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "triggered_frames": self.triggered_frames,
            "skipped_frames": self.skipped_frames,
            "person_box_count": self.person_box_count,
            "person_track_count": self.person_track_count,
            "trigger_rate": self.trigger_rate,
            "skip_rate": self.skip_rate,
        }


@dataclass
class PersonGate:
    min_score: float = 0.45
    person_category: str = "person"
    skip_when_empty: bool = True
    metrics: PersonGateMetrics = field(default_factory=PersonGateMetrics)

    def evaluate(
        self,
        detections: Sequence[DetectionBox] | None = None,
        tracklets: Sequence[Tracklet] | None = None,
        frame: FrameMeta | None = None,
    ) -> PersonGateResult:
        person_tracklets = self._person_tracklets(tracklets or ())
        person_boxes = self._person_boxes(detections or (), person_tracklets)
        has_person = bool(person_boxes or person_tracklets)
        skipped = self.skip_when_empty and not has_person
        if has_person:
            reason = "person_present"
        elif skipped:
            reason = "no_person_skip_heavy_modules"
        else:
            reason = "no_person_continue"

        result = PersonGateResult(
            frame=frame,
            has_person=has_person,
            should_run_heavy_modules=not skipped,
            skipped_heavy_modules=skipped,
            person_boxes=person_boxes,
            person_tracklets=person_tracklets,
            reason=reason,
        )
        self.metrics = self.metrics.update(result)
        return result

    def evaluate_frame_result(self, frame_result: Any) -> PersonGateResult:
        frame = getattr(frame_result, "frame", None)
        detections = getattr(frame_result, "detections", ())
        person_tracks = getattr(frame_result, "person_tracks", ())
        return self.evaluate(detections=detections, tracklets=person_tracks, frame=frame)

    def reset_metrics(self) -> None:
        self.metrics = PersonGateMetrics()

    def _person_boxes(
        self,
        detections: Sequence[DetectionBox],
        person_tracklets: Sequence[Tracklet],
    ) -> tuple[DetectionBox, ...]:
        boxes = [
            detection
            for detection in detections
            if detection.category == self.person_category and detection.score >= self.min_score
        ]
        tracked_boxes = [
            box
            for tracklet in person_tracklets
            for box in tracklet.boxes[-1:]
            if box.score >= self.min_score
        ]

        by_key: dict[tuple[int, tuple[float, float, float, float], str | None], DetectionBox] = {}
        for box in [*boxes, *tracked_boxes]:
            by_key[(box.frame_id, box.bbox, box.track_id)] = box
        return tuple(by_key.values())

    def _person_tracklets(self, tracklets: Sequence[Tracklet]) -> tuple[Tracklet, ...]:
        return tuple(
            tracklet
            for tracklet in tracklets
            if tracklet.category == self.person_category
            and (
                not tracklet.boxes
                or any(box.score >= self.min_score for box in tracklet.boxes[-1:])
            )
        )


def evaluate_person_gate(
    detections: Sequence[DetectionBox] | None = None,
    min_score: float = 0.45,
    tracklets: Sequence[Tracklet] | None = None,
    frame: FrameMeta | None = None,
) -> PersonGateResult:
    return PersonGate(min_score=min_score).evaluate(
        detections=detections,
        tracklets=tracklets,
        frame=frame,
    )
