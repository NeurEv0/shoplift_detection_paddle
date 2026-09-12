"""Eventize a per-window open-probability timeline into 拆包装 events.

Rule (v2 ep26 定版, see docs/pptsm_open_binary_report.md §9):
  沿人像时间线滑窗(win=16, stride=2) → 单窗 open 概率;
  连续 >= min_consecutive_windows 窗概率 >= theta 记为一次"拆包事件".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from shoplift.pptsm_open.types import OpenPackagingEvent, OpenWindow


@dataclass(frozen=True)
class OpenEventizeConfig:
    """Tunable parameters for open-window eventization."""

    theta: float = 0.6
    min_consecutive_windows: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be between 0.0 and 1.0")
        if self.min_consecutive_windows <= 0:
            raise ValueError("min_consecutive_windows must be positive")


def eventize_open_windows(
    windows: Iterable[OpenWindow],
    *,
    camera_id: str,
    person_track_id: str,
    theta: float = 0.6,
    min_consecutive_windows: int = 3,
) -> list[OpenPackagingEvent]:
    """Group a person timeline's open windows into 拆包装 events.

    ``windows`` must belong to a single ``(camera_id, person_track_id)``. Windows
    are sorted by ``start_timestamp_ms``; a run of ``>= min_consecutive_windows``
    consecutive windows each with ``open_prob >= theta`` becomes one event.
    """
    if not camera_id:
        raise ValueError("camera_id must be a non-empty string")
    if not person_track_id:
        raise ValueError("person_track_id must be a non-empty string")

    ordered = sorted(windows, key=lambda window: window.start_timestamp_ms)
    events: list[OpenPackagingEvent] = []
    run: list[OpenWindow] = []

    for window in ordered:
        if window.open_prob >= theta:
            run.append(window)
            continue
        if len(run) >= min_consecutive_windows:
            events.append(OpenPackagingEvent(camera_id, person_track_id, tuple(run)))
        run = []

    if len(run) >= min_consecutive_windows:
        events.append(OpenPackagingEvent(camera_id, person_track_id, tuple(run)))

    return events


__all__ = ["OpenEventizeConfig", "eventize_open_windows"]
