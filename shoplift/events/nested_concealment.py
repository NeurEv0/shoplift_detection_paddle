"""Two-stage "private container hidden inside a public store basket" detector.

语义(2026-09-10 Step0 定稿 v1.3, 用户确认方向 A: 主人绑定+延续见下注):
  场景: 偷盗者把自己的私有容器(bag/backpack/handbag/suitcase, canonical "bag")
  放入店内公有篮(仅 source=basket; 购物车 cart / 顾客自有袋 plastic_bag 均排除)
  后, 包被篮壁遮挡(检出消失) → 藏匿窗口武装; 之后**该包的主人**(放入前手近袋者)
  累计伸手入篮 >= owner_dwell_frames 帧 → 触发 private_container_concealment。

关键机制(数据锚点, 见 .tmp/ 侦查与 docs/stage_b §8.6):
  - mouth margin: 包底边须越过篮口线 mouth_margin_px(20) 才算"进入"——防搁篮口误判;
  - bag 短时跟踪(帧间 IoU>=0.3, gap<=3) + owner 绑定: 包可见期间"手中心距包中心
    <80px"的人即主人(cc015 person-71 在 f510-552 携带期绑定, MOT 空洞期不受影响);
  - entry(>=min_entry_frames 帧)后包检出消失 hide_confirm_frames(2) 帧 → armed,
    记录 owner(袋跟踪主人, 兜底回看 arm 前 40 帧手入篮多数者);
  - armed 内只计 owner 的在篮帧累计(严格逐帧), >= owner_dwell_frames(50) 即触发;
    每 (锚, owner) 只发一次; armed 后包再可见 > blip_tolerance_frames(40) 或
    idle_expire_ms(12s) 无 owner 活动 → 解除(累计清零);
  - 每帧 basket/bag 检测先做 NMS(同类 IoU>0.35 合并) → 同一物理篮只建一个锚点。

d03(堂食区, 容器全映射为 category=basket): cart source 保留 'cart'、顾客自有袋
source 'plastic_bag' → 均排除; source='basket' 的容器 d03 仅 2 个检测 → 0 武装 → 0 误报。

cc015 person-71 事件锚点(fx 记录): bag 主人=person-71; 消失 f615 → armed;
owner 在篮帧 615-646(31) + 807-828(22) 累计 53 >= 50 → 触发 f~828 (33.1s)。
MOT 换号(71→172→183→208)后的同人活动 v1 不接续(见 docs 待办), 避免双事件。

注: v1.2 曾用 K=3×L=15"多次短 dip"判定; 实测 person-71 是 18/21/22 帧三段中停留
且 arm 会砍首段 → K 语义不稳。按用户指示改为"主人累计长停留"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from shoplift.core.types import BBox, DetectionBox, HandRegion, RelationEvidence, RiskEvent
from shoplift.rules.risk_score import risk_level
from shoplift.tracking.association import AssociationFrame

EVENT_TYPE = "private_container_concealment"

# 事件 reason tags(避开 normal/低可见度 关键字, 防 validator 把事件降档)
TAG_ENTRY = "private_container_entered_public_container"
TAG_HIDDEN = "private_container_hidden"
TAG_DWELL = "owner_hand_long_dwell_in_public_container"
TAG_CONSISTENT = "entry_temporal_consistent"

_FRAMES_PER_SECOND = 25


@dataclass(frozen=True)
class NestedConcealmentConfig:
    """Tunable thresholds for the nested private-into-public concealment rule.

    YAML section ``nested_concealment`` mirrors these fields 1:1 (rules_loader).
    """

    mouth_margin_px: float = 20.0
    min_entry_frames: int = 3
    hide_confirm_frames: int = 2
    blip_tolerance_frames: int = 40
    owner_dwell_frames: int = 115  # 主人累计在篮停留 >= 4.6s(115帧=25fps) → 触发(用户定稿, 不加同篮接续)
    idle_expire_ms: int = 12_000
    allowed_public_sources: tuple[str, ...] = ("basket",)
    owner_near_distance_px: float = 80.0
    event_type: str = EVENT_TYPE
    base_score: float = 0.38
    entry_confirm_score: float = 0.15
    hide_confirm_score: float = 0.15
    per_episode_score: float = 0.10
    medium_threshold: float = 0.45
    high_threshold: float = 0.75

    def __post_init__(self) -> None:
        for name in ("min_entry_frames", "hide_confirm_frames", "blip_tolerance_frames",
                     "owner_dwell_frames"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.idle_expire_ms <= 0:
            raise ValueError("idle_expire_ms must be positive")


def _bbox_center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _iou(first: BBox, second: BBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = width * height
    if inter_area <= 0:
        return 0.0
    union_area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter_area
    return inter_area / max(1.0, union_area)


def _source_category(detection: DetectionBox) -> str:
    return str(detection.attributes.get("source_category") or detection.category).strip().lower()


def _nms(detections: Sequence[DetectionBox]) -> list[DetectionBox]:
    """同类重复检测合并(IoU>0.35 取高分), 保框最大 score."""
    kept: list[DetectionBox] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        dup = False
        for existing in kept:
            if _iou(existing.bbox, det.bbox) > 0.35:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


class _BasketAnchor:
    """One store-basket window state machine (idle -> hiding -> armed)."""

    __slots__ = ("anchor_id", "box", "last_frame_id", "phase", "entry_consec",
                 "hide_count", "vis_run", "arm_frame_id", "last_activity_frame_id",
                 "owner", "owner_cum", "owner_cur_run", "fired")

    def __init__(self, anchor_id: str, box: BBox, frame_id: int) -> None:
        self.anchor_id = anchor_id
        self.box = box
        self.last_frame_id = frame_id
        self.phase = "idle"
        self.entry_consec = 0
        self.hide_count = 0
        self.vis_run = 0
        self.arm_frame_id: int | None = None
        self.last_activity_frame_id = frame_id
        self.owner: str | None = None
        self.owner_cum = 0
        self.owner_cur_run = 0
        self.fired = False

    def reset(self) -> None:
        self.phase = "idle"
        self.entry_consec = 0
        self.hide_count = 0
        self.vis_run = 0
        self.arm_frame_id = None
        self.owner = None
        self.owner_cum = 0
        self.owner_cur_run = 0


class _BagTrack:
    __slots__ = ("track_id", "last_box", "last_frame_id", "start_frame_id",
                 "owner", "last_owner_frame_id")

    def __init__(self, track_id: str, box: BBox, frame_id: int) -> None:
        self.track_id = track_id
        self.last_box = box
        self.last_frame_id = frame_id
        self.start_frame_id = frame_id
        self.owner: str | None = None
        self.last_owner_frame_id: int | None = None


class NestedConcealmentDetector:
    """Frame-level windowed detector; call :meth:`update` once per frame."""

    def __init__(self, config: NestedConcealmentConfig | None = None,
                 debug_sink: list[dict[str, Any]] | None = None) -> None:
        self.config = config or NestedConcealmentConfig()
        self._anchors: list[_BasketAnchor] = []
        self._next_anchor_id = 1
        self._bag_tracks: list[_BagTrack] = []
        self._next_bag_track_id = 1
        self._recent_hands: list[tuple[int, list[tuple[str, BBox]]]] = []
        self._debug = debug_sink

    # ------------------------------------------------------------------ API
    def update(self, frame: AssociationFrame) -> tuple[RiskEvent, ...]:
        baskets = self._nms_public_baskets(frame.containers)
        bag_dets = _nms([det for det in frame.containers
                         if det.category.strip().lower() == "bag"])
        hands = frame.hand_regions
        events: list[RiskEvent] = []

        self._update_bag_tracks(bag_dets, hands, frame)
        self._recent_hands.append((frame.frame_id, [
            (hand.person_track_id, hand.bbox) for hand in hands
        ]))
        if len(self._recent_hands) > 60:
            self._recent_hands.pop(0)

        self._match_anchors(baskets, frame.frame_id)
        for anchor in self._anchors:
            anchor.last_frame_id = frame.frame_id
            is_rel = self._bag_released_into(bag_dets, anchor.box)
            is_vis = self._bag_visible_at(bag_dets, anchor.box)
            nb, nbg = len(baskets), len(bag_dets)

            if anchor.phase == "idle":
                if is_rel:
                    anchor.entry_consec += 1
                    if anchor.entry_consec >= self.config.min_entry_frames:
                        anchor.phase = "hiding"
                        anchor.entry_consec = 0
                        self._trace(frame, anchor, "hiding", is_rel, is_vis, nb, nbg)
                else:
                    anchor.entry_consec = 0
            elif anchor.phase == "hiding":
                if is_rel:
                    pass
                elif is_vis:
                    anchor.phase = "idle"
                    anchor.hide_count = 0
                    self._trace(frame, anchor, "idle", is_rel, is_vis, nb, nbg,
                                "bag-visible-again")
                else:
                    anchor.hide_count += 1
                    if anchor.hide_count >= self.config.hide_confirm_frames:
                        anchor.phase = "armed"
                        anchor.arm_frame_id = frame.frame_id
                        anchor.last_activity_frame_id = frame.frame_id
                        anchor.hide_count = 0
                        anchor.vis_run = 0
                        anchor.owner = self._resolve_owner(frame, anchor)
                        anchor.owner_cum = 0
                        anchor.owner_cur_run = 0
                        self._trace(frame, anchor, "armed", is_rel, is_vis, nb, nbg,
                                    f"owner={anchor.owner}")
            else:  # armed
                if is_vis:
                    anchor.vis_run += 1
                    if anchor.vis_run > self.config.blip_tolerance_frames:
                        anchor.reset()
                        self._trace(frame, anchor, "idle", is_rel, is_vis, nb, nbg,
                                    "blip-disarm")
                        continue
                else:
                    anchor.vis_run = 0
                self._advance_owner_dwell(anchor, frame, events)
                idle_frames = frame.frame_id - anchor.last_activity_frame_id
                if idle_frames * (1000 / _FRAMES_PER_SECOND) > self.config.idle_expire_ms:
                    self._trace(frame, anchor, "idle", is_rel, is_vis, nb, nbg,
                                "idle-expire")
                    anchor.reset()

        self._anchors = [a for a in self._anchors
                         if frame.frame_id - a.last_frame_id <= 10 * _FRAMES_PER_SECOND]
        self._bag_tracks = [t for t in self._bag_tracks
                            if frame.frame_id - t.last_frame_id <= 15 * _FRAMES_PER_SECOND]
        return tuple(events)

    # ------------------------------------------------------------ internals
    def _nms_public_baskets(self, containers: Sequence[DetectionBox]) -> list[DetectionBox]:
        return _nms([
            det for det in containers
            if det.category.strip().lower() == "basket"
            and _source_category(det) in self.config.allowed_public_sources
        ])

    def _match_anchors(self, baskets: list[DetectionBox], frame_id: int) -> None:
        for basket in baskets:
            best = None
            best_iou = 0.0
            for anchor in self._anchors:
                score = _iou(anchor.box, basket.bbox)
                if score > best_iou:
                    best_iou = score
                    best = anchor
            if best is not None and best_iou >= 0.2:
                best.box = basket.bbox
            else:
                # 新建前再查一次: 与既有锚 IoU>0.35(近似同物理篮)则并进去, 避免双锚
                merged = False
                for anchor in self._anchors:
                    if _iou(anchor.box, basket.bbox) > 0.35:
                        anchor.box = basket.bbox
                        merged = True
                        break
                if not merged:
                    anchor = _BasketAnchor(f"nested-pub-{self._next_anchor_id}",
                                           basket.bbox, frame_id)
                    self._next_anchor_id += 1
                    self._anchors.append(anchor)
        self._merge_overlapping_anchors()

    def _merge_overlapping_anchors(self) -> None:
        """重叠锚(IoU>0.35)合并: 保留更"深"的一个, 状态取并集(避免双事件)."""
        merged = True
        while merged:
            merged = False
            for i in range(len(self._anchors)):
                for j in range(i + 1, len(self._anchors)):
                    a, b = self._anchors[i], self._anchors[j]
                    if _iou(a.box, b.box) <= 0.35:
                        continue
                    depth = {"idle": 0, "hiding": 1, "armed": 2}
                    if depth[a.phase] < depth[b.phase]:
                        a, b = b, a
                    if b.phase == "armed" and a.phase != "armed":
                        a.phase = "armed"
                        a.arm_frame_id = b.arm_frame_id
                        a.owner = b.owner
                        a.owner_cum = b.owner_cum
                        a.owner_cur_run = b.owner_cur_run
                    a.fired = a.fired or b.fired
                    a.owner_cum = max(a.owner_cum, b.owner_cum)
                    if a.owner is None:
                        a.owner = b.owner
                    if a.arm_frame_id is None or (b.arm_frame_id is not None
                                                  and b.arm_frame_id < a.arm_frame_id):
                        a.arm_frame_id = b.arm_frame_id
                    a.entry_consec = max(a.entry_consec, b.entry_consec)
                    self._anchors.remove(b)
                    merged = True
                    break
                if merged:
                    break

    def _update_bag_tracks(self, bag_dets: list[DetectionBox], hands: Sequence[HandRegion],
                           frame: AssociationFrame) -> None:
        used: set[str] = set()
        for det in bag_dets:
            best = None
            best_iou = 0.0
            for track in self._bag_tracks:
                if track.track_id in used:
                    continue
                score = _iou(track.last_box, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best = track
            if best is not None and best_iou >= 0.3 and frame.frame_id - best.last_frame_id <= 3:
                best.last_box = det.bbox
                best.last_frame_id = frame.frame_id
                used.add(best.track_id)
                for hand in hands:
                    if self._hand_near_bag(hand, det.bbox):
                        best.owner = hand.person_track_id
                        best.last_owner_frame_id = frame.frame_id
                        break
            else:
                track = _BagTrack(f"nested-bag-{self._next_bag_track_id}",
                                  det.bbox, frame.frame_id)
                self._next_bag_track_id += 1
                self._bag_tracks.append(track)
                for hand in hands:
                    if self._hand_near_bag(hand, det.bbox):
                        track.owner = hand.person_track_id
                        track.last_owner_frame_id = frame.frame_id
                        break

    def _hand_near_bag(self, hand: HandRegion, bag_box: BBox) -> bool:
        hx, hy = _bbox_center(hand.bbox)
        bx1, by1, bx2, by2 = bag_box
        c = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
        return abs(hx - c[0]) < self.config.owner_near_distance_px and abs(hy - c[1]) < self.config.owner_near_distance_px

    def _bag_released_into(self, bags: list[DetectionBox], basket_box: BBox) -> bool:
        x1, y1, x2, y2 = basket_box
        for bag in bags:
            bx1, by1, bx2, by2 = bag.bbox
            cx = (bx1 + bx2) / 2.0
            if by2 >= y1 + self.config.mouth_margin_px and x1 - 20 <= cx <= x2 + 20:
                return True
        return False

    def _bag_visible_at(self, bags: list[DetectionBox], basket_box: BBox) -> bool:
        x1, y1, x2, y2 = basket_box
        for bag in bags:
            bx1, by1, bx2, by2 = bag.bbox
            cx = (bx1 + bx2) / 2.0
            if by2 >= y1 - 10 and x1 - 40 <= cx <= x2 + 40:
                return True
        return False

    def _resolve_owner(self, frame: AssociationFrame, anchor: _BasketAnchor) -> str | None:
        # 1) 袋跟踪: 找离锚最近且刚消失的 bag track 的主人
        candidates: list[tuple[float, _BagTrack]] = []
        x1, y1, x2, y2 = anchor.box
        anchor_c = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        for track in self._bag_tracks:
            if track.owner is None:
                continue
            if frame.frame_id - (track.last_owner_frame_id or 0) > 120:
                continue
            bx1, by1, bx2, by2 = track.last_box
            tc = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            dist = abs(tc[0] - anchor_c[0]) + abs(tc[1] - anchor_c[1])
            candidates.append((dist, track))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1].owner
        # 2) 兜底: arm 前 40 帧手入篮多数者
        votes: dict[str, int] = {}
        for (fr_id, fr_hands) in self._recent_hands:
            if frame.frame_id - fr_id > 40:
                continue
            for (pid, hbox) in fr_hands:
                hc = _bbox_center(hbox)
                if x1 - 5 <= hc[0] <= x2 + 5 and y1 <= hc[1] <= y2:
                    votes[pid] = votes.get(pid, 0) + 1
        if votes:
            return max(votes, key=votes.get)
        return None

    def _advance_owner_dwell(self, anchor: _BasketAnchor, frame: AssociationFrame,
                             events: list[RiskEvent]) -> None:
        if anchor.owner is None or anchor.fired:
            return
        x1, y1, x2, y2 = anchor.box
        inside = False
        for hand in frame.hand_regions:
            if hand.person_track_id != anchor.owner:
                continue
            hx, hy = _bbox_center(hand.bbox)
            if x1 - 5 <= hx <= x2 + 5 and y1 <= hy <= y2:
                inside = True
                break
        if inside:
            anchor.owner_cum += 1
            anchor.owner_cur_run += 1
            anchor.last_activity_frame_id = frame.frame_id
            if anchor.owner_cum >= self.config.owner_dwell_frames:
                anchor.fired = True
                event = self._build_event(anchor, frame)
                if event is not None:
                    events.append(event)
                    self._trace(frame, anchor, "fired", False, False, 0, 0,
                                f"owner={anchor.owner} cum={anchor.owner_cum}")

    def _build_event(self, anchor: _BasketAnchor,
                     frame: AssociationFrame) -> RiskEvent | None:
        score = min(
            1.0,
            self.config.base_score
            + self.config.entry_confirm_score
            + self.config.hide_confirm_score
            + self.config.per_episode_score,
        )
        level = risk_level(score, medium_threshold=self.config.medium_threshold,
                           high_threshold=self.config.high_threshold)
        start_ms = (anchor.arm_frame_id or frame.frame_id) * (1000 // _FRAMES_PER_SECOND)
        end_ms = frame.timestamp_ms
        evidence = (
            RelationEvidence(
                relation_type=self.config.event_type,
                frame_id=anchor.arm_frame_id or frame.frame_id,
                timestamp_ms=start_ms,
                score=1.0,
                reason_tags=(TAG_ENTRY, TAG_HIDDEN),
                person_track_id=anchor.owner or "",
                evidence_boxes={"container": anchor.box},
                metadata={
                    "public_container_kind": "store_basket",
                    "nested_phase": "armed",
                    "owner": anchor.owner,
                },
            ),
            RelationEvidence(
                relation_type=self.config.event_type,
                frame_id=frame.frame_id,
                timestamp_ms=end_ms,
                score=1.0,
                reason_tags=(TAG_DWELL,),
                person_track_id=anchor.owner or "",
                evidence_boxes={"container": anchor.box},
                metadata={
                    "public_container_kind": "store_basket",
                    "nested_phase": "dwell",
                    "owner_dwell_frames": anchor.owner_cum,
                },
            ),
        )
        return RiskEvent(
            event_id=f"evt-{frame.camera_id}-{anchor.owner}-{self.config.event_type}-{frame.frame_id}",
            camera_id=frame.camera_id,
            timestamp_ms=end_ms,
            person_track_id=anchor.owner or "unknown",
            event_type=self.config.event_type,
            risk_score=score,
            risk_level=level,
            reason_tags=(TAG_ENTRY, TAG_HIDDEN, TAG_DWELL, TAG_CONSISTENT),
            evidence=evidence,
            start_timestamp_ms=start_ms,
            end_timestamp_ms=end_ms,
            confidence=1.0,
            metadata={
                "state": "confirmed_risk_event",
                "source": "shoplifting_event_engine_p1",
                "event_type": self.config.event_type,
                "nested": {
                    "anchor_id": anchor.anchor_id,
                    "arm_frame_id": anchor.arm_frame_id,
                    "owner": anchor.owner,
                    "owner_dwell_frames": anchor.owner_cum,
                    "public_container_kind": "store_basket",
                },
            },
        )

    def _trace(self, frame: AssociationFrame, anchor: _BasketAnchor, phase_to: str,
               is_rel: bool, is_vis: bool, n_baskets: int, n_bags: int,
               note: str = "") -> None:
        if self._debug is None:
            return
        self._debug.append({
            "frame": frame.frame_id,
            "anchor": anchor.anchor_id,
            "phase_to": phase_to,
            "note": note,
        })


__all__ = [
    "EVENT_TYPE",
    "NestedConcealmentConfig",
    "NestedConcealmentDetector",
    "TAG_CONSISTENT",
    "TAG_DWELL",
    "TAG_ENTRY",
    "TAG_HIDDEN",
]
