"""Person gate primitives for deciding whether heavy modules should run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from shoplift.core.types import DetectionBox


@dataclass(frozen=True)
class PersonGateResult:
    has_person: bool
    skipped_heavy_modules: bool
    person_boxes: tuple[DetectionBox, ...]


def evaluate_person_gate(
    detections: Sequence[DetectionBox],
    min_score: float = 0.45,
) -> PersonGateResult:
    person_boxes = tuple(
        detection
        for detection in detections
        if detection.category == "person" and detection.score >= min_score
    )
    has_person = bool(person_boxes)
    return PersonGateResult(
        has_person=has_person,
        skipped_heavy_modules=not has_person,
        person_boxes=person_boxes,
    )
