"""Typed results for the PP-TSM open-binary (拆包装) inference.

These structures are produced by the sliding-window inference and eventization,
then consumed by the fusion engine. They are pure data: no Paddle / GPU / weights
dependency, so the fusion layer is unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _validate_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_prob(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


@dataclass(frozen=True)
class OpenWindow:
    """One PP-TSM open classification window on a single person timeline.

    ``open_prob`` is the softmax probability of class ``1 = open(拆包装)``.
    ``frame_id`` is the window's anchor frame on the resampled person timeline;
    it is optional and only used to enrich downstream evidence.
    """

    start_timestamp_ms: int
    end_timestamp_ms: int
    open_prob: float
    frame_id: int | None = None

    def __post_init__(self) -> None:
        _validate_non_negative(self.start_timestamp_ms, "start_timestamp_ms")
        _validate_non_negative(self.end_timestamp_ms, "end_timestamp_ms")
        if self.end_timestamp_ms < self.start_timestamp_ms:
            raise ValueError("end_timestamp_ms must be >= start_timestamp_ms")
        _validate_prob(float(self.open_prob), "open_prob")
        if self.frame_id is not None:
            _validate_non_negative(self.frame_id, "frame_id")
        object.__setattr__(self, "open_prob", float(self.open_prob))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenPackagingEvent:
    """One eventized 拆包装 episode.

    Represents ``>= min_consecutive_windows`` consecutive open windows whose
    probability is ``>= theta`` on one person timeline (v2 ep26 event rule).
    """

    camera_id: str
    person_track_id: str
    windows: tuple[OpenWindow, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.camera_id, "camera_id")
        _validate_non_empty(self.person_track_id, "person_track_id")
        windows = tuple(self.windows)
        if not windows:
            raise ValueError("windows must be non-empty")
        object.__setattr__(self, "windows", windows)

    @property
    def start_timestamp_ms(self) -> int:
        return self.windows[0].start_timestamp_ms

    @property
    def end_timestamp_ms(self) -> int:
        return self.windows[-1].end_timestamp_ms

    @property
    def peak_open_prob(self) -> float:
        return max(window.open_prob for window in self.windows)

    @property
    def mean_open_prob(self) -> float:
        return sum(window.open_prob for window in self.windows) / len(self.windows)

    @property
    def consecutive_windows(self) -> int:
        return len(self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "person_track_id": self.person_track_id,
            "start_timestamp_ms": self.start_timestamp_ms,
            "end_timestamp_ms": self.end_timestamp_ms,
            "peak_open_prob": self.peak_open_prob,
            "mean_open_prob": self.mean_open_prob,
            "consecutive_windows": self.consecutive_windows,
            "windows": [window.to_dict() for window in self.windows],
            "metadata": dict(self.metadata),
        }


__all__ = ["OpenPackagingEvent", "OpenWindow"]
