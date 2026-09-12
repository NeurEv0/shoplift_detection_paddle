"""PP-TSM open × pipeline risk fusion engine.

Fuses PP-TSM 拆包装 events (product-gated, then merged) with the PaddleDetection
pipeline's ``RiskEvent`` output and produces the final reportable events:

  1. 商品判定门 (ProductGate): 拆包事件窗口内该 person 是否出现「连续
     >= min_product_confirm_frames 帧 holding_product / proxy_item_region」证据
     —— 拆的是不是商品。稀疏确认:不要求全程可见,窗口内出现过即可;
     全程看不到(无证据)则保守丢弃(拆自己包/非商品不算)。
  2. 同一 person(或相邻 track)时间重叠/相邻的 open 段合并成一个事件
     (轨迹碎片、事件碎片都会并回来)。
  3. 每个合并后的拆包装段 → 独立新增 ``package_opening`` 事件(high),
     不抬升、不改管线事件(只要有拆包装就新建事件,不管管线状态)。

Pure Python: no Paddle / GPU / weights dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.pptsm_open.types import OpenPackagingEvent

PACKAGE_OPENING_EVENT_TYPE = "package_opening"
OPEN_PACKAGING_TAG = "open_packaging"
PPTSM_OPEN_RELATION = "pptsm_open"
PPTSM_OPEN_EVIDENCE_TAG = "pptsm_open_evidence"
FUSION_SOURCE = "pptsm_open_fusion"


def _validate_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two ``[x0, y0, x1, y1]`` boxes."""
    ax0, ay0, ax1, ay1 = (float(v) for v in box_a)
    bx0, by0, bx1, by1 = (float(v) for v in box_b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class ProductFrameEvidence:
    """One processed frame of product-holding evidence for one person.

    ``is_holding_product`` is True when the person's hand is holding a store
    product (``holding_product``) or a proxy item region exists for that hand.
    """

    frame_id: int
    timestamp_ms: int
    is_holding_product: bool

    def __post_init__(self) -> None:
        _validate_non_negative(self.frame_id, "frame_id")
        _validate_non_negative(self.timestamp_ms, "timestamp_ms")


@dataclass(frozen=True)
class FusionConfig:
    """Tunable parameters for open-eventization, product gating and fusion."""

    # 事件化(与 v2 ep26 定版一致)
    theta: float = 0.6
    min_consecutive_windows: int = 3

    # 商品判定门:拆包窗口内「连续 >= N 帧」holding_product 才算"拆的是商品"
    min_product_confirm_frames: int = 14
    # 构建 holding_product 证据时,手状态分数下限
    min_holding_product_score: float = 0.5

    # 拆包装段合并:时间重叠或间隔 <= 该值(ms)视为同一次拆包
    merge_temporal_gap_ms: int = 5000
    # 不同 track 合并时需 bbox IoU >= 该值(确认是同一人因轨迹碎片换了 id)
    merge_iou_threshold: float = 0.3

    # 独立拆包事件(每个合并后的拆包段都产出)的配置
    standalone_event_type: str = PACKAGE_OPENING_EVENT_TYPE
    standalone_risk_score: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be between 0.0 and 1.0")
        if self.min_consecutive_windows <= 0:
            raise ValueError("min_consecutive_windows must be positive")
        if self.min_product_confirm_frames <= 0:
            raise ValueError("min_product_confirm_frames must be positive")
        if not 0.0 <= self.min_holding_product_score <= 1.0:
            raise ValueError("min_holding_product_score must be between 0.0 and 1.0")
        if self.merge_temporal_gap_ms < 0:
            raise ValueError("merge_temporal_gap_ms must be non-negative")
        if not 0.0 <= self.merge_iou_threshold <= 1.0:
            raise ValueError("merge_iou_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.standalone_risk_score <= 1.0:
            raise ValueError("standalone_risk_score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class FusionResult:
    """Structured output of one fusion pass."""

    events: tuple[RiskEvent, ...] = ()
    escalated_event_ids: tuple[str, ...] = ()
    package_opening_events: tuple[RiskEvent, ...] = ()
    discarded_open_events: tuple[OpenPackagingEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ProductGate:
    """Decide whether a 拆包装 episode is really opening a store product."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def confirms(
        self,
        open_event: OpenPackagingEvent,
        evidence: Sequence[ProductFrameEvidence],
    ) -> bool:
        """Return True if the open event's window contains a consecutive run of
        ``>= min_product_confirm_frames`` frames marked as holding a product."""
        in_window = [
            item
            for item in evidence
            if open_event.start_timestamp_ms <= item.timestamp_ms <= open_event.end_timestamp_ms
        ]
        return self._max_true_run(in_window) >= self.config.min_product_confirm_frames

    @staticmethod
    def _max_true_run(evidence: Sequence[ProductFrameEvidence]) -> int:
        """Longest run of *consecutive source frames* (frame_id adjacency) that
        are holding_product. A gap (missing frame, e.g. occlusion) or a False
        frame breaks the run — matching the "连续 >= N 帧" requirement."""
        best = 0
        run = 0
        prev_frame_id: int | None = None
        ordered = sorted(evidence, key=lambda entry: (entry.frame_id, entry.timestamp_ms))
        for item in ordered:
            if not item.is_holding_product:
                run = 0
                prev_frame_id = None
                continue
            if prev_frame_id is not None and item.frame_id == prev_frame_id + 1:
                run += 1
            else:
                run = 1
            best = max(best, run)
            prev_frame_id = item.frame_id
        return best


class FusionEngine:
    """Fuse pipeline risk events with product-gated, merged 拆包装 events."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self.product_gate = ProductGate(self.config)

    def fuse(
        self,
        pipeline_events: Sequence[RiskEvent],
        open_events: Sequence[OpenPackagingEvent],
        product_evidence: Mapping[str, Sequence[ProductFrameEvidence]] | None = None,
        person_boxes: Mapping[str, Sequence[tuple[int, int, list[float]]]] | None = None,
    ) -> FusionResult:
        """Run the fusion and return the final reportable events.

        ``product_evidence`` maps ``person_track_id`` -> per-frame
        ``ProductFrameEvidence`` (used by the product gate). ``person_boxes``
        maps ``person_track_id`` -> ``(frame_id, timestamp_ms, bbox)`` and is
        used to confirm that two *different* tracks are the same person before
        merging (track fragmentation).
        """
        evidence = {
            person_id: tuple(entries)
            for person_id, entries in (product_evidence or {}).items()
        }
        pipeline = list(pipeline_events)
        discarded: list[OpenPackagingEvent] = []
        gated: list[OpenPackagingEvent] = []

        for open_event in open_events:
            person_evidence = evidence.get(open_event.person_track_id, ())
            if not self.product_gate.confirms(open_event, person_evidence):
                discarded.append(open_event)
                continue
            gated.append(open_event)

        merged = self._merge_open_events(gated, person_boxes)
        package_opening = [self._build_package_opening(event) for event in merged]

        return FusionResult(
            events=tuple(pipeline) + tuple(package_opening),
            escalated_event_ids=(),
            package_opening_events=tuple(package_opening),
            discarded_open_events=tuple(discarded),
            metadata={
                "pipeline_event_count": len(pipeline),
                "open_event_count": len(open_events),
                "gated_open_event_count": len(gated),
                "merged_open_event_count": len(merged),
                "escalated_count": 0,
                "package_opening_count": len(package_opening),
                "discarded_open_count": len(discarded),
            },
        )

    # -- merge ---------------------------------------------------------------

    def _merge_open_events(
        self,
        events: Sequence[OpenPackagingEvent],
        person_boxes: Mapping[str, Sequence[tuple[int, int, list[float]]]] | None,
    ) -> list[OpenPackagingEvent]:
        """Greedily coalesce temporally-overlapping/adjacent open events.

        Two events merge when they are (a) time-overlapping or within
        ``merge_temporal_gap_ms`` of each other, AND (b) from the same track, or
        from different tracks whose person boxes spatially overlap (same person
        whose track id changed).
        """
        if not events:
            return []
        ordered = sorted(
            events, key=lambda event: (event.start_timestamp_ms, event.end_timestamp_ms)
        )
        merged: list[OpenPackagingEvent] = []
        current = ordered[0]
        for nxt in ordered[1:]:
            if self._should_merge(current, nxt, person_boxes):
                current = self._combine(current, nxt, person_boxes)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged

    def _should_merge(
        self,
        a: OpenPackagingEvent,
        b: OpenPackagingEvent,
        person_boxes: Mapping[str, Sequence[tuple[int, int, list[float]]]] | None,
    ) -> bool:
        if a.camera_id != b.camera_id:
            return False
        gap = self.config.merge_temporal_gap_ms
        if a.end_timestamp_ms + gap < b.start_timestamp_ms:
            return False
        if b.end_timestamp_ms + gap < a.start_timestamp_ms:
            return False
        if a.person_track_id == b.person_track_id:
            return True
        return self._tracks_overlap(a, b, person_boxes)

    def _tracks_overlap(
        self,
        a: OpenPackagingEvent,
        b: OpenPackagingEvent,
        person_boxes: Mapping[str, Sequence[tuple[int, int, list[float]]]] | None,
    ) -> bool:
        if not person_boxes:
            return False
        box_a = a.metadata.get("_end_bbox")
        if box_a is None:
            box_a = self._bbox_near(a.person_track_id, a.end_timestamp_ms, person_boxes)
        box_b = self._bbox_near(b.person_track_id, b.start_timestamp_ms, person_boxes)
        if box_a is None or box_b is None:
            return False
        return _iou(box_a, box_b) >= self.config.merge_iou_threshold

    @staticmethod
    def _bbox_near(
        track_id: str,
        timestamp_ms: int,
        person_boxes: Mapping[str, Sequence[tuple[int, int, list[float]]]],
    ) -> list[float] | None:
        track = person_boxes.get(track_id)
        if not track:
            return None
        best: list[float] | None = None
        best_delta: int | None = None
        for _frame_id, ts, bbox in track:
            delta = abs(ts - timestamp_ms)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = bbox
        return best

    def _combine(
        self,
        a: OpenPackagingEvent,
        b: OpenPackagingEvent,
        person_boxes: Mapping[str, Sequence[tuple[int, int, list[float]]]] | None,
    ) -> OpenPackagingEvent:
        windows = sorted(
            a.windows + b.windows, key=lambda window: window.start_timestamp_ms
        )
        source_tracks = list(a.metadata.get("merged_source_track_ids", [a.person_track_id]))
        if b.person_track_id not in source_tracks:
            source_tracks.append(b.person_track_id)
        metadata = dict(a.metadata)
        metadata["merged_source_track_ids"] = source_tracks
        if person_boxes:
            end_bbox = self._bbox_near(b.person_track_id, b.end_timestamp_ms, person_boxes)
            if end_bbox is not None:
                metadata["_end_bbox"] = [float(value) for value in end_bbox]
        return OpenPackagingEvent(
            camera_id=a.camera_id,
            person_track_id=a.person_track_id,
            windows=tuple(windows),
            metadata=metadata,
        )

    # -- output --------------------------------------------------------------

    def _build_package_opening(self, open_event: OpenPackagingEvent) -> RiskEvent:
        peak = max(open_event.windows, key=lambda window: window.open_prob)
        evidence = RelationEvidence(
            relation_type=PPTSM_OPEN_RELATION,
            frame_id=peak.frame_id if peak.frame_id is not None else 0,
            timestamp_ms=peak.start_timestamp_ms,
            score=peak.open_prob,
            reason_tags=(OPEN_PACKAGING_TAG,),
            person_track_id=open_event.person_track_id,
            metadata=self._open_event_evidence(open_event),
        )
        metadata: dict[str, Any] = {
            "source": FUSION_SOURCE,
            "pptsm_open": self._open_event_evidence(open_event),
        }
        source_tracks = open_event.metadata.get("merged_source_track_ids")
        if source_tracks:
            metadata["merged_source_track_ids"] = list(source_tracks)
        return RiskEvent(
            event_id=self._build_event_id(open_event),
            camera_id=open_event.camera_id,
            timestamp_ms=open_event.end_timestamp_ms,
            start_timestamp_ms=open_event.start_timestamp_ms,
            end_timestamp_ms=open_event.end_timestamp_ms,
            person_track_id=open_event.person_track_id,
            event_type=self.config.standalone_event_type,
            risk_score=self.config.standalone_risk_score,
            risk_level="high",
            confidence=open_event.peak_open_prob,
            reason_tags=(OPEN_PACKAGING_TAG, PPTSM_OPEN_EVIDENCE_TAG),
            evidence=(evidence,),
            metadata=metadata,
        )

    def _build_event_id(self, open_event: OpenPackagingEvent) -> str:
        return (
            f"evt-{open_event.camera_id}-{open_event.person_track_id}-"
            f"{self.config.standalone_event_type}-{open_event.start_timestamp_ms}"
        )

    @staticmethod
    def _open_event_evidence(open_event: OpenPackagingEvent) -> dict[str, Any]:
        return {
            "start_timestamp_ms": open_event.start_timestamp_ms,
            "end_timestamp_ms": open_event.end_timestamp_ms,
            "peak_open_prob": open_event.peak_open_prob,
            "mean_open_prob": open_event.mean_open_prob,
            "consecutive_windows": open_event.consecutive_windows,
            "window_timestamps_ms": [window.start_timestamp_ms for window in open_event.windows],
            "window_probs": [window.open_prob for window in open_event.windows],
        }


__all__ = [
    "FUSION_SOURCE",
    "OPEN_PACKAGING_TAG",
    "PACKAGE_OPENING_EVENT_TYPE",
    "PPTSM_OPEN_EVIDENCE_TAG",
    "PPTSM_OPEN_RELATION",
    "FusionConfig",
    "FusionEngine",
    "FusionResult",
    "ProductFrameEvidence",
    "ProductGate",
]
