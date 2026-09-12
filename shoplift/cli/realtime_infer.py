"""Real-time stream inference (USB camera / RTSP) with GPU/CPU resource profiling.

把离线两段式链路(管线 → PP-TSM open 融合)合并成一条流式单 pass,并在推理全程
持续采样 GPU 显存 / CPU 内存 / 时延 / FPS,用来从「单流占用」外推「10 流超市部署」
的硬件配置。

整体流程(单流,流式):
  stream frame
    ├─ PP-Human MOT        (行人检测 + 跟踪)
    ├─ keypoint            (姿态 + 手部 ROI)
    ├─ person attribute    (手持状态 holding_product) + proxy item region
    ├─ item container det  (容器/包裹检测)
    ├─ 管线风险事件引擎     (RiskEvent)
    ├─ 增量 PP-TSM open    (人像 16 帧滑窗 @3.125fps → open 概率 → 事件化)
    └─ 商品门 + 融合        (package_opening 事件)

监控指标(每帧):
  - paddle gpu allocated / reserved / max_allocated (MB)  —— 显存「波动」与「稳态峰值」
  - 进程级 GPU 显存(pynvml, 或 nvidia-smi 兜底, MB)
  - 进程 RSS (MB)
  - 各阶段时延 + FPS

产物(output-dir 下):
  - metrics.csv     每帧资源/时延样本
  - summary.json    模型加载显存拆解 + 稳态/峰值/波动 + 10 流外推
  - events.jsonl    流式产出的融合事件(逐条追加)

用法(服务器 conda 环境, 项目根目录):
  python -m shoplift.cli.realtime_infer \
      --pipeline-config shoplift/configs/pipeline.cc015_v7b_classw102_test.yml \
      --source rtsp://user:pass@10.0.0.5:554/stream \
      --open-config shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml \
      --open-weights outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams \
      --paddlevideo-root third_party/PaddleVideo \
      --fusion-config shoplift/configs/fusion.example.yml \
      --output-dir outputs/realtime/single_stream \
      --duration 300

--source 也支持: 数字(USB 摄像头索引, 如 0)、本地视频路径(测试回放)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from shoplift.cli.offline_analyze import (
    FrameMeta,
    FramePacket,
    OfflineConfig,
    analyze_packet_components,
    create_vision_backend,
    load_offline_config,
)
from shoplift.configs.rules_loader import load_rules_config
from shoplift.events.event_engine import ShopliftingEventEngine
from shoplift.events.event_schema import risk_event_to_payload
from shoplift.fusion.fusion_engine import (
    FusionConfig,
    FusionEngine,
    ProductFrameEvidence,
)
from shoplift.pptsm_open.eventize import eventize_open_windows
from shoplift.pptsm_open.infer import OpenModel, TimelineFrame
from shoplift.pptsm_open.types import OpenPackagingEvent, OpenWindow
from shoplift.rules.risk_score import RiskScorer
from shoplift.rules.validators import RiskRuleValidator
from shoplift.tracking.association import AssociationFrame
from shoplift.vision import (
    ItemContainerDetectionAdapter,
    PersonGate,
    RuleBasedPersonAttributeEstimator,
)

DEFAULT_TIMELINE_FPS = 3.125
STOP_SENTINEL = object()


def _quiet(fn):
    """抑制模型加载/前向时的 stdout/stderr 刷屏(如 Paddle 的 'Test iter 0')。
    若 fn 抛异常,把捕获的 stderr 原样写回真实 stderr 再抛出,保证报错可见。"""
    import contextlib
    import io
    import sys

    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return fn()
    except BaseException:
        if err.getvalue():
            sys.stderr.write(err.getvalue())
        raise


# ---------------------------------------------------------------------------
# 资源监控
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """采样 paddle 显存 + 进程 GPU 显存 + 进程 RSS。

    paddle 层(allocated/reserved)是每帧主指标,能捕捉「显存波动」;
    pynvml / nvidia-smi 给出进程级真实占用(含 CUDA context 开销),用于核对与汇报。
    """

    def __init__(self, gpu_pid: int | None = None) -> None:
        self._nvml_ok = False
        self._nvml = None
        self._handle = None
        self._pid = gpu_pid or os.getpid()
        self._last_process_mem_ts = 0.0
        self._process_mem_cache: dict[str, float] = {}
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
            self._nvml_ok = True
        except Exception:
            self._nvml_ok = False

    # -- paddle 显存 --------------------------------------------------------
    @staticmethod
    def paddle_mem() -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            import paddle

            out["gpu_allocated_mb"] = paddle.device.cuda.memory_allocated() / 1e6
            out["gpu_reserved_mb"] = paddle.device.cuda.memory_reserved() / 1e6
            out["gpu_max_allocated_mb"] = paddle.device.cuda.max_memory_allocated() / 1e6
            out["gpu_max_reserved_mb"] = paddle.device.cuda.max_memory_reserved() / 1e6
        except Exception:
            pass
        return out

    # -- 进程级 GPU 显存(pynvml 优先, nvidia-smi 回退) ----------------------
    def process_gpu_mem(self, refresh_interval: float = 2.0) -> dict[str, float]:
        """进程级 GPU 显存。优先 pynvml(快);不可用时回退 nvidia-smi --query-compute-apps。

        nvidia-smi 是子进程调用(慢),为避免拖慢每帧采样,结果按 refresh_interval 秒缓存。
        """
        now = time.monotonic()
        if self._process_mem_cache and now - self._last_process_mem_ts < refresh_interval:
            return self._process_mem_cache
        self._last_process_mem_ts = now

        out: dict[str, float] = {}
        if self._nvml_ok:
            try:
                for proc in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                    if proc.pid == self._pid:
                        out["gpu_process_used_mb"] = proc.usedGpuMemory / 1e6
                        break
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                out["gpu_total_mb"] = mem.total / 1e6
                out["gpu_used_mb"] = mem.used / 1e6
            except Exception:
                out = {}

        if "gpu_process_used_mb" not in out:
            # 回退:nvidia-smi 进程级(含 CUDA context 的真实占用)
            try:
                raw = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                for line in raw.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[0] == str(self._pid):
                        out["gpu_process_used_mb"] = float(parts[1])
                        break
                if "gpu_total_mb" not in out:
                    out.update(self.nvidia_smi())
            except Exception:
                pass

        self._process_mem_cache = out
        return out

    # -- 进程 RSS ----------------------------------------------------------
    @staticmethod
    def rss_mb() -> float:
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:
            return -1.0

    def sample(self) -> dict[str, float]:
        s: dict[str, float] = {"cpu_rss_mb": self.rss_mb()}
        s.update(self.paddle_mem())
        s.update(self.process_gpu_mem())
        return s

    @staticmethod
    def nvidia_smi() -> dict[str, float]:
        """authoritative snapshot via nvidia-smi(较慢,周期性调用即可)。"""
        try:
            raw = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            used, total = (float(v) for v in raw.split(",")[:2])
            return {"smi_gpu_used_mb": used, "smi_gpu_total_mb": total}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# 流输入(后台读线程 + 有界队列,解耦 RTSP 阻塞读与推理)
# ---------------------------------------------------------------------------

class StreamSource:
    """从 USB 摄像头 / RTSP / 本地视频读取帧,后台线程缓冲。

    ``source``: 纯数字 → USB 索引;rtsp/http 开头 → 网络流;否则 → 本地文件。
    """

    def __init__(
        self,
        source: str,
        *,
        camera_id: str,
        frame_stride: int = 1,
        queue_size: int = 8,
        source_fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.frame_stride = max(1, frame_stride)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._opened = False
        self._fps: float = source_fps or 0.0
        self._width = width
        self._height = height
        self._reader_error: Exception | None = None

        if source.lstrip("-").isdigit():
            self._open = int(source)
        elif source.lower().startswith(("rtsp://", "http://", "rtmp://")):
            self._open = source
        else:
            self._open = source

        self._thread = threading.Thread(target=self._read_loop, name="stream-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        cap = cv2.VideoCapture(self._open)
        if not cap.isOpened():
            self._reader_error = OSError(f"cannot open stream source: {self._open}")
            self._queue.put(STOP_SENTINEL)
            return
        self._opened = True
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if self._fps <= 0 and fps and fps > 0:
                self._fps = float(fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._size = (width, height)
        except Exception:
            self._size = (0, 0)

        frame_id = 0
        try:
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                if frame_id % self.frame_stride != 0:
                    frame_id += 1
                    continue
                try:
                    self._queue.put((frame_id, image), timeout=1.0)
                except queue.Full:
                    # 推理跟不上时丢帧(保持实时,不无限堆积)
                    pass
                frame_id += 1
        finally:
            cap.release()
            self._queue.put(STOP_SENTINEL)

    @property
    def fps(self) -> float:
        return self._fps if self._fps and self._fps > 0 else 25.0

    @property
    def size(self) -> tuple[int, int]:
        return getattr(self, "_size", (0, 0))

    def next(self) -> FramePacket | None:
        if self._reader_error is not None:
            raise self._reader_error
        while True:
            try:
                item = self._queue.get(timeout=0.5)
                break
            except queue.Empty:
                if self._stop.is_set():
                    return None
        if item is STOP_SENTINEL:
            return None
        frame_id, image = item
        height, width = image.shape[:2]
        fps = self.fps
        timestamp_ms = int(round(frame_id * 1000.0 / fps))
        meta = FrameMeta(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            camera_id=self.camera_id,
            width=width,
            height=height,
            source_uri=str(self._open),
        )
        return FramePacket(
            frame=meta,
            image=image,
            source_frame_id=frame_id,
            source_uri=str(self._open),
            input_type="stream",
            fps=fps,
        )

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# 增量 PP-TSM open
# ---------------------------------------------------------------------------

@dataclass
class _TrackState:
    start_frame: int
    timeline: list[TimelineFrame] = field(default_factory=list)
    windows: list[OpenWindow] = field(default_factory=list)
    finalized: int = 0  # 已 emit 的「已完成」事件数


def _crop_padded_bgr(image: Any, bbox: list[float], padding: float) -> Any | None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx0 = max(0, int(x0 - w * padding))
    cy0 = max(0, int(y0 - h * padding))
    cx1 = min(width - 1, int(x1 + w * padding))
    cy1 = min(height - 1, int(y1 + h * padding))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return np.ascontiguousarray(image[cy0:cy1, cx0:cx1])


class RealtimeOpenEngine:
    """在流式人像时间线上增量滑窗 + 事件化 PP-TSM open。

    时间线按 timeline_fps(默认 3.125fps,与训练 master cadence 一致)重采样;
    每凑满 16 帧(且按 stride 步进)跑一次 open 模型;事件化后把「已完成」的
    拆包装事件逐条产出(尾部的进行中 run 在 track 结束时 finalize)。
    """

    def __init__(
        self,
        model: OpenModel,
        *,
        timeline_fps: float,
        source_fps: float,
        win: int = 16,
        stride: int = 2,
        crop_padding: float = 0.15,
        theta: float = 0.6,
        min_consecutive_windows: int = 3,
    ) -> None:
        self.model = model
        self.timeline_fps = timeline_fps
        self.step = max(1, round(source_fps / timeline_fps))
        self.win = win
        self.stride = stride
        self.crop_padding = crop_padding
        self.theta = theta
        self.min_consecutive = min_consecutive_windows
        self._tracks: dict[str, _TrackState] = {}
        self.window_inferences = 0  # 统计 open 前向次数

    def on_frame(
        self,
        camera_id: str,
        image: Any,
        tracks: list[tuple[str, list[float], int, int]],
    ) -> list[OpenPackagingEvent]:
        """``tracks``: [(track_id, bbox, frame_id, timestamp_ms)]。

        返回本帧新 finalize 的拆包装事件。
        """
        from PIL import Image

        new_events: list[OpenPackagingEvent] = []
        for track_id, bbox, frame_id, timestamp_ms in tracks:
            st = self._tracks.get(track_id)
            if st is None:
                st = _TrackState(start_frame=frame_id)
                self._tracks[track_id] = st

            # 时间线重采样:对齐 track 起点,每 step 帧采一次
            if (frame_id - st.start_frame) % self.step != 0:
                continue
            crop = _crop_padded_bgr(image, bbox, self.crop_padding)
            if crop is None:
                continue
            rgb = np.ascontiguousarray(crop[..., ::-1])  # BGR -> RGB
            st.timeline.append(
                TimelineFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    image=Image.fromarray(rgb),
                )
            )

            n = len(st.timeline)
            if n < self.win or (n - self.win) % self.stride != 0:
                continue
            chunk = st.timeline[n - self.win : n]
            open_prob = self.model.open_prob([f.image for f in chunk])
            self.window_inferences += 1
            anchor = chunk[self.win // 2]
            st.windows.append(
                OpenWindow(
                    start_timestamp_ms=chunk[0].timestamp_ms,
                    end_timestamp_ms=chunk[-1].timestamp_ms,
                    open_prob=open_prob,
                    frame_id=anchor.frame_id,
                )
            )
            events = eventize_open_windows(
                st.windows,
                camera_id=camera_id,
                person_track_id=track_id,
                theta=self.theta,
                min_consecutive_windows=self.min_consecutive,
            )
            if len(events) > st.finalized:
                new_events.extend(events[st.finalized :])
                st.finalized = len(events)

        return new_events

    def finalize_track(self, camera_id: str, track_id: str) -> list[OpenPackagingEvent]:
        """track 结束(离开画面)时,把尾部进行中的 run 强制收尾。"""
        st = self._tracks.get(track_id)
        if st is None:
            return []
        # 追加一个假 break 窗口,让 eventize 把尾部 run finalize
        if st.windows:
            last = st.windows[-1]
            fake = OpenWindow(
                start_timestamp_ms=last.end_timestamp_ms + 1,
                end_timestamp_ms=last.end_timestamp_ms + 1,
                open_prob=0.0,
            )
            st.windows.append(fake)
        events = eventize_open_windows(
            st.windows,
            camera_id=camera_id,
            person_track_id=track_id,
            theta=self.theta,
            min_consecutive_windows=self.min_consecutive,
        )
        new_events = events[st.finalized :]
        st.finalized = len(events)
        return new_events

    def track_count(self) -> int:
        return len(self._tracks)


# ---------------------------------------------------------------------------
# 融合辅助(从单帧结果构建 product evidence / person boxes)
# ---------------------------------------------------------------------------

def _attr_holding(attr: Any, min_score: float) -> bool:
    for side in ("left_hand_state", "right_hand_state"):
        pred = getattr(attr, side, None)
        if pred is not None and pred.label == "holding_product" and pred.score >= min_score:
            return True
    return False


def build_product_evidence(
    frame_id: int,
    timestamp_ms: int,
    person_tracks: tuple,
    person_attributes: tuple,
    proxy_item_regions: tuple,
    min_holding_product_score: float,
) -> dict[str, ProductFrameEvidence]:
    holding: set[str] = set()
    for attr in person_attributes:
        if attr.person_track_id and _attr_holding(attr, min_holding_product_score):
            holding.add(attr.person_track_id)
    for proxy in proxy_item_regions:
        if proxy.person_track_id:
            holding.add(proxy.person_track_id)

    out: dict[str, ProductFrameEvidence] = {}
    for track in person_tracks:
        out[track.track_id] = ProductFrameEvidence(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            is_holding_product=track.track_id in holding,
        )
    return out


def _draw_bbox(frame, bbox, color, label, thickness=2) -> None:
    if bbox is None or len(bbox) != 4:
        return
    import cv2

    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            frame,
            str(label),
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _visible_keypoint(keypoints, scores, index, min_score):
    if index >= len(keypoints):
        return None
    score = float(scores[index]) if index < len(scores) else 1.0
    if score < min_score:
        return None
    point = keypoints[index]
    if not isinstance(point, (tuple, list)) or len(point) < 2:
        return None
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if x <= 0 and y <= 0:
        return None
    return (x, y)


def _draw_body_pose(frame, body_pose, min_score=0.2) -> None:
    import cv2

    keypoints = getattr(body_pose, "keypoints", ()) or ()
    scores = getattr(body_pose, "scores", ()) or ()
    edges = getattr(body_pose, "skeleton_edges", ()) or ()
    line_color = (60, 255, 120)
    point_color = (0, 190, 255)
    for edge in edges:
        start, end = int(edge[0]), int(edge[1])
        s = _visible_keypoint(keypoints, scores, start, min_score)
        e = _visible_keypoint(keypoints, scores, end, min_score)
        if s is None or e is None:
            continue
        cv2.line(frame, s, e, line_color, 2, cv2.LINE_AA)
    for index in range(len(keypoints)):
        point = _visible_keypoint(keypoints, scores, index, min_score)
        if point is not None:
            cv2.circle(frame, point, 3, point_color, -1, cv2.LINE_AA)


def draw_frame(image, analysis, highlight_ids=None):
    """复刻离线管线的 debug 叠加:人框+track_id、姿态骨架/关键点、手部框、代理商品区、商品/容器框。"""

    frame = image.copy()
    highlight = set(highlight_ids or ())
    person_tracks = getattr(analysis, "person_tracks", ()) or ()

    # 1. 人框 + track_id(报警人红框高亮)
    for track in person_tracks:
        if not track.boxes:
            continue
        if track.track_id in highlight:
            _draw_bbox(frame, track.boxes[-1].bbox, (0, 0, 255), f"ALERT {track.track_id}", 3)
        else:
            _draw_bbox(frame, track.boxes[-1].bbox, (80, 180, 255), track.track_id)

    # 2. 姿态骨架 + 关键点
    for pose in getattr(analysis, "body_poses", ()) or ():
        min_score = float((getattr(pose, "metadata", None) or {}).get("min_keypoint_score", 0.2))
        _draw_body_pose(frame, pose, min_score)

    # 3. 手部框
    for hand in getattr(analysis, "hand_regions", ()) or ():
        _draw_bbox(frame, hand.bbox, (255, 120, 0), f"{hand.side}_hand")

    # 4. 代理商品区(手持商品的手部 ROI 代理框)
    for proxy in getattr(analysis, "proxy_item_regions", ()) or ():
        _draw_bbox(frame, proxy.proxy_bbox, (0, 255, 255), f"proxy_{proxy.hand_side}")

    # 5. 商品 + 容器框(带类别文字)
    item_container = getattr(analysis, "item_container", None)
    if item_container is not None:
        for it in getattr(item_container, "items", ()) or ():
            _draw_bbox(frame, it.bbox, (0, 220, 0), getattr(it, "category", "item"))
        for c in getattr(item_container, "containers", ()) or ():
            _draw_bbox(frame, c.bbox, (255, 0, 180), getattr(c, "category", "container"))

    return frame


def save_alert_shot(packet, analysis, person_id, event_type, timestamp_ms, out_dir) -> None:
    """报警瞬间截图:高亮该 person 的标注帧存成 jpg。"""
    try:
        import cv2

        frame = draw_frame(packet.image, analysis, highlight_ids={person_id})
        safe_id = str(person_id).replace("/", "_").replace("\\", "_")
        shot = out_dir / f"alert_{event_type}_{safe_id}_{timestamp_ms}.jpg"
        cv2.imwrite(str(shot), frame)
        print(f"[shot] {shot.name}", flush=True)
    except Exception as exc:
        print(f"[shot] failed: {exc}", flush=True)


def _resolve_output_dir(base: Path, auto_suffix: bool) -> Path:
    """每次启动新建一个带时间戳的输出目录,避免覆盖上一次结果。"""
    if not auto_suffix:
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parent = base.parent
    name = base.name
    candidate = parent / f"{name}_{stamp}"
    i = 1
    while candidate.exists():
        candidate = parent / f"{name}_{stamp}_{i}"
        i += 1
    return candidate


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> OfflineConfig:
    ns = argparse.Namespace(
        config=args.pipeline_config,
        input=None,
        output=None,
        dry_run=False,
        frame_stride=args.frame_stride,
        max_frames=None,
        no_debug=True,
        debug_frames=False,
        backend=args.backend,
    )
    config = load_offline_config(ns)
    if args.camera_id:
        config = replace(config, camera_id=args.camera_id)
    return config


# ---------------------------------------------------------------------------
# 模型加载 + 显存拆解
# ---------------------------------------------------------------------------

def load_backend_with_breakdown(
    backend: Any,
    monitor: ResourceMonitor,
    breakdown: dict[str, float],
    per_model: bool,
) -> None:
    """加载管线各模型,采样显存拆解。"""
    backend._prepare_import_paths()
    cfg = backend.config

    if not per_model:
        # 一次性触发惰性加载
        backend._ensure_initialized()
        breakdown["pipeline_all"] = monitor.sample()
        return

    if cfg.mot.enabled:
        backend._mot_predictor = backend._create_mot_predictor()
        breakdown["pipeline_mot"] = monitor.sample()
    if cfg.keypoint.enabled:
        backend._keypoint_predictor = backend._create_keypoint_predictor()
        breakdown["pipeline_keypoint"] = monitor.sample()
    if cfg.person_attribute.enabled:
        backend._create_attribute_predictor()
        breakdown["pipeline_attribute"] = monitor.sample()
    if cfg.item_container.enabled:
        backend._item_detector = backend._create_item_detector()
        breakdown["pipeline_item_container"] = monitor.sample()
    # 补齐 keypoint 依赖的 crop/translate 工具(惰性)
    if cfg.keypoint.enabled:
        backend._ensure_initialized()


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    # 抑制 Paddle/PaddleVideo INFO 日志(如 "Test iter 0"),须在首次 import paddle 前设置
    os.environ.setdefault("GLOG_minloglevel", "2")
    import logging

    for _name in ("paddlevideo", "ppdet", "PaddleClas", "paddle"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    config = build_config(args)
    monitor = ResourceMonitor()
    source = StreamSource(
        args.source,
        camera_id=config.camera_id,
        frame_stride=args.frame_stride,
        source_fps=args.source_fps,
        width=args.width,
        height=args.height,
    )
    breakdown: dict[str, float] = {}

    # ---- 1. 加载管线模型 ----
    print("[load] baseline", flush=True)
    breakdown["baseline"] = monitor.sample()
    backend = create_vision_backend(config.backend)
    _quiet(lambda: load_backend_with_breakdown(backend, monitor, breakdown, args.per_model))
    print(f"[load] pipeline loaded: {monitor.sample().get('gpu_reserved_mb', 0):.0f} MB reserved", flush=True)

    # ---- 2. 加载 PP-TSM open(动态图) ----
    open_model: Optional[OpenModel] = None
    open_engine: Optional[RealtimeOpenEngine] = None
    fusion_engine: Optional[FusionEngine] = None
    if args.open_config and args.open_weights:
        import paddle

        # 后端加载时内部调用了 paddle.enable_static();open 模型是 dygraph,需切回动态图
        paddle.disable_static()
        open_model = _quiet(
            lambda: OpenModel.load(
                args.open_config,
                args.open_weights,
                device=args.device,
                paddlevideo_root=args.paddlevideo_root,
            )
        )
        breakdown["open_model"] = monitor.sample()
        print(f"[load] open model loaded: {monitor.sample().get('gpu_reserved_mb', 0):.0f} MB reserved", flush=True)

        fusion_config, inference = _load_fusion_config(args.fusion_config)
        fusion_engine = FusionEngine(fusion_config)
        open_engine = RealtimeOpenEngine(
            open_model,
            timeline_fps=float(args.timeline_fps or inference.get("timeline_fps", DEFAULT_TIMELINE_FPS)),
            source_fps=source.fps,
            win=int(inference.get("win", 16)),
            stride=int(inference.get("stride", 2)),
            crop_padding=float(
                args.crop_padding
                if args.crop_padding is not None
                else inference.get("crop_padding", 0.15)
            ),
            theta=fusion_config.theta,
            min_consecutive_windows=fusion_config.min_consecutive_windows,
        )
    else:
        print("[load] open model disabled (--open-config/--open-weights not both set)", flush=True)

    # 进程级真实显存(静态权重 + CUDA context),用于 10 流外推
    process_mem_after_load = monitor.process_gpu_mem().get("gpu_process_used_mb", 0.0)
    peak_process_mem = process_mem_after_load
    print(f"[load] process gpu mem after load: {process_mem_after_load:.1f} MB", flush=True)

    # ---- 3. 事件引擎 + 融合状态 ----
    person_gate = PersonGate(
        min_score=config.modules.person_gate_min_score,
        skip_when_empty=config.modules.person_gate_skip_when_empty,
    )
    rules = load_rules_config(config.rules_config)
    event_engine = ShopliftingEventEngine(
        association_config=rules.association,
        risk_scorer=RiskScorer(rules.risk_scoring),
        rule_validator=RiskRuleValidator(rules.rules),
        nested_concealment_config=rules.nested_concealment,
    )
    item_container_adapter = ItemContainerDetectionAdapter(
        min_score=config.modules.item_container_min_score,
        allowed_categories=frozenset(config.modules.item_container_classes)
        if config.modules.item_container_classes is not None
        else None,
    )
    attribute_estimator = RuleBasedPersonAttributeEstimator()

    pipeline_events: list = []
    open_events: list[OpenPackagingEvent] = []
    product_evidence: dict[str, list[ProductFrameEvidence]] = {}
    person_boxes: dict[str, list[tuple[int, int, list[float]]]] = {}
    emitted_event_ids: set[str] = set()
    active_tracks_last_seen: dict[str, int] = {}

    # ---- 4. 输出(每次启动自动新建带时间戳的目录,避免覆盖上次结果) ----
    out_dir = _resolve_output_dir(Path(args.output_dir), not args.no_run_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output-dir] {out_dir}", flush=True)
    metrics_path = out_dir / "metrics.csv"
    events_path = out_dir / "events.jsonl"
    summary_path = out_dir / "summary.json"

    fieldnames = [
        "frame_id", "timestamp_ms", "wall_s",
        "pipeline_ms", "event_ms", "open_ms", "fusion_ms", "total_ms", "fps",
        "gpu_allocated_mb", "gpu_reserved_mb", "gpu_max_allocated_mb", "gpu_max_reserved_mb",
        "gpu_process_used_mb", "gpu_total_mb", "gpu_used_mb", "cpu_rss_mb",
        "n_tracks", "n_open_windows", "n_pipeline_events", "n_package_opening",
    ]
    csv_file = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    events_file = events_path.open("w", encoding="utf-8")
    debug_video_path = out_dir / "debug.mp4"
    video_writer = None
    video_fps = getattr(source, "fps", 0) or 15.0

    samples: list[dict[str, float]] = []
    latencies: list[float] = []
    n_frames = 0
    n_open_windows_total = 0
    n_package_opening_total = 0
    start_wall = time.perf_counter()
    last_summary = start_wall
    stop_flag = threading.Event()

    def _handle_sig(signum, frame):
        stop_flag.set()
        source.close()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print(f"[run] streaming from {args.source} (fps={source.fps:.2f}, stride={args.frame_stride})", flush=True)

    try:
        while not stop_flag.is_set():
            t0 = time.perf_counter()
            packet = source.next()
            if packet is None:
                break

            # 管线推理(静音 PaddleDetection deploy 预测器的 'Test iter 0' 刷屏)
            t1 = time.perf_counter()
            analysis = _quiet(
                lambda: analyze_packet_components(
                    packet,
                    backend=backend,
                    person_gate=person_gate,
                    item_container_adapter=item_container_adapter,
                    attribute_estimator=attribute_estimator,
                    modules=config.modules,
                )
            )
            t2 = time.perf_counter()

            # 管线事件引擎
            event_result = event_engine.process_frame(
                AssociationFrame(
                    frame_id=packet.frame.frame_id,
                    timestamp_ms=packet.frame.timestamp_ms,
                    camera_id=packet.frame.camera_id,
                    person_tracks=analysis.person_tracks,
                    body_poses=analysis.body_poses,
                    hand_regions=analysis.hand_regions,
                    items=analysis.item_container.items
                    + tuple(r.to_detection_box() for r in analysis.proxy_item_regions),
                    containers=analysis.item_container.containers,
                    extension_regions=analysis.item_container.extension_regions,
                    metadata={"source_uri": packet.source_uri, "input_type": packet.input_type},
                )
            )
            for evt in event_result.events:
                events_file.write(json.dumps(risk_event_to_payload(evt), ensure_ascii=False) + "\n")
                if evt.risk_level == "high":
                    print(
                        f"[ALERT] {evt.event_type} person={evt.person_track_id} "
                        f"score={evt.risk_score:.2f} t={evt.timestamp_ms}ms",
                        flush=True,
                    )
                    save_alert_shot(
                        packet, analysis, evt.person_track_id,
                        evt.event_type, evt.timestamp_ms, out_dir,
                    )
                elif evt.risk_level == "medium":
                    print(
                        f"[EVENT] {evt.event_type} person={evt.person_track_id} "
                        f"level={evt.risk_level} t={evt.timestamp_ms}ms",
                        flush=True,
                    )
            events_file.flush()
            pipeline_events.extend(event_result.events)
            t3 = time.perf_counter()

            # 增量 PP-TSM open + 融合
            new_open_events: list[OpenPackagingEvent] = []
            track_rows: list[tuple[str, list[float], int, int]] = []
            for track in analysis.person_tracks:
                if not track.boxes:
                    continue
                box = track.boxes[-1]
                track_rows.append(
                    (track.track_id, list(box.bbox), packet.frame.frame_id, packet.frame.timestamp_ms)
                )
                person_boxes.setdefault(track.track_id, []).append(
                    (packet.frame.frame_id, packet.frame.timestamp_ms, list(box.bbox))
                )
                active_tracks_last_seen[track.track_id] = packet.frame.frame_id

            if open_engine is not None:
                new_open_events = open_engine.on_frame(
                    packet.frame.camera_id, packet.image, track_rows
                )
                n_open_windows_total = open_engine.window_inferences
            t4 = time.perf_counter()

            # 商品证据(holding_product)累积
            evidence = build_product_evidence(
                packet.frame.frame_id,
                packet.frame.timestamp_ms,
                analysis.person_tracks,
                analysis.person_attributes,
                analysis.proxy_item_regions,
                config.modules.person_attribute_min_holding_product_score,
            )
            for tid, ev in evidence.items():
                product_evidence.setdefault(tid, []).append(ev)
            open_events.extend(new_open_events)

            # 周期性跑一次融合(纯 Python,开销小)
            fusion_ms = 0.0
            if fusion_engine is not None and open_events and (n_frames % args.fusion_every == 0):
                t4b = time.perf_counter()
                result = fusion_engine.fuse(pipeline_events, open_events, product_evidence, person_boxes)
                for evt in result.package_opening_events:
                    if evt.event_id not in emitted_event_ids:
                        emitted_event_ids.add(evt.event_id)
                        events_file.write(
                            json.dumps(risk_event_to_payload(evt), ensure_ascii=False) + "\n"
                        )
                        events_file.flush()
                        n_package_opening_total += 1
                        print(
                            f"[ALERT] {evt.event_type} person={evt.person_track_id} "
                            f"prob={evt.confidence:.3f} "
                            f"t={evt.start_timestamp_ms}-{evt.end_timestamp_ms}ms",
                            flush=True,
                        )
                        save_alert_shot(
                            packet, analysis, evt.person_track_id,
                            evt.event_type, evt.start_timestamp_ms, out_dir,
                        )
                fusion_ms = (time.perf_counter() - t4b) * 1000.0
            t5 = time.perf_counter()

            # 资源采样 + 记录
            sample = monitor.sample()
            if sample.get("gpu_process_used_mb"):
                peak_process_mem = max(peak_process_mem, sample["gpu_process_used_mb"])
            sample["frame_id"] = packet.frame.frame_id
            sample["timestamp_ms"] = packet.frame.timestamp_ms
            sample["wall_s"] = round(time.perf_counter() - start_wall, 3)
            sample["pipeline_ms"] = (t2 - t1) * 1000.0
            sample["event_ms"] = (t3 - t2) * 1000.0
            sample["open_ms"] = (t4 - t3) * 1000.0
            sample["fusion_ms"] = fusion_ms
            sample["total_ms"] = (t5 - t0) * 1000.0
            sample["fps"] = 1.0 / max(1e-6, t5 - t0)
            sample["n_tracks"] = len(track_rows)
            sample["n_open_windows"] = n_open_windows_total
            sample["n_pipeline_events"] = len(pipeline_events)
            sample["n_package_opening"] = n_package_opening_total
            writer.writerow({k: sample.get(k, "") for k in fieldnames})
            csv_file.flush()
            samples.append(sample)
            latencies.append(sample["total_ms"])
            n_frames += 1

            # 可视化:标注帧写入 debug 视频(纯 CPU,在时延/显存采样之后,不影响测量)
            if packet.image is not None:
                try:
                    import cv2

                    annotated = draw_frame(packet.image, analysis)
                    if video_writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(debug_video_path), fourcc, video_fps, (w, h)
                        )
                    if video_writer.isOpened():
                        video_writer.write(annotated)
                except Exception:
                    pass

            # 周期性汇总(默认每 5s 或 --summary-every 秒)
            if time.perf_counter() - last_summary >= args.summary_every:
                last_summary = time.perf_counter()
                _print_summary(samples, latencies, n_frames, monitor)

            if args.max_frames and n_frames >= args.max_frames:
                break
            if args.duration and (time.perf_counter() - start_wall) >= args.duration:
                break
    finally:
        source.close()
        csv_file.close()
        events_file.close()
        if video_writer is not None:
            video_writer.release()

    # ---- 收尾:finalize 进行中的 open run ----
    if open_engine is not None and fusion_engine is not None:
        for tid in list(open_engine._tracks.keys()):
            tail = open_engine.finalize_track(config.camera_id, tid)
            open_events.extend(tail)
        result = fusion_engine.fuse(pipeline_events, open_events, product_evidence, person_boxes)
        with events_path.open("a", encoding="utf-8") as f:
            for evt in result.package_opening_events:
                if evt.event_id not in emitted_event_ids:
                    emitted_event_ids.add(evt.event_id)
                    f.write(json.dumps(risk_event_to_payload(evt), ensure_ascii=False) + "\n")
                    n_package_opening_total += 1
                    print(
                        f"[ALERT] {evt.event_type} person={evt.person_track_id} "
                        f"prob={evt.confidence:.3f} "
                        f"t={evt.start_timestamp_ms}-{evt.end_timestamp_ms}ms",
                        flush=True,
                    )

    summary = _build_summary(
        samples, latencies, n_frames, breakdown, monitor,
        n_open_windows_total, len(pipeline_events), n_package_opening_total,
        config.camera_id,
        process_mem_after_load=process_mem_after_load,
        peak_process_mem=peak_process_mem,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _load_fusion_config(path: str | Path) -> tuple[FusionConfig, dict[str, Any]]:
    from shoplift.cli.fuse_report import load_fusion_config

    return load_fusion_config(path)


def _print_summary(samples, latencies, n_frames, monitor: ResourceMonitor) -> None:
    if not samples:
        return
    last = samples[-1]

    lat = sorted(latencies)
    p50 = lat[len(lat) // 2]
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    alloc = [s.get("gpu_allocated_mb", 0) for s in samples if "gpu_allocated_mb" in s]
    fluct = (max(alloc) - min(alloc)) if alloc else 0.0
    print(
        f"[live] frames={n_frames} fps={1.0/max(1e-6, sum(lat)/len(lat)):.1f} "
        f"lat p50/p95={p50:.0f}/{p95:.0f}ms "
        f"gpu_reserved={last.get('gpu_reserved_mb', 0):.0f}MB "
        f"gpu_fluct={fluct:.0f}MB cpu_rss={last.get('cpu_rss_mb', 0):.0f}MB",
        flush=True,
    )


def _build_summary(
    samples, latencies, n_frames, breakdown, monitor: ResourceMonitor,
    n_open_windows, n_pipeline_events, n_package_opening, camera_id: str = "",
    process_mem_after_load: float = 0.0, peak_process_mem: float = 0.0,
) -> dict[str, Any]:
    alloc = [s["gpu_allocated_mb"] for s in samples if "gpu_allocated_mb" in s]
    reserved = [s["gpu_reserved_mb"] for s in samples if "gpu_reserved_mb" in s]
    rss = [s["cpu_rss_mb"] for s in samples if s.get("cpu_rss_mb", -1) >= 0]
    lat = sorted(latencies)

    steady_reserved = reserved[-1] if reserved else 0.0
    peak_reserved = max(reserved) if reserved else 0.0
    peak_allocated = max(alloc) if alloc else 0.0
    fluctuation = (max(alloc) - min(alloc)) if alloc else 0.0
    peak_rss = max(rss) if rss else 0.0

    avg_fps = n_frames / max(1e-6, sum(lat) / 1000.0)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0

    smi = monitor.nvidia_smi()

    # 进程级真实显存(含静态管线 + CUDA context),用于 10 流外推
    static_process = process_mem_after_load if process_mem_after_load > 0 else 0.0
    dynamic_process = max(0.0, peak_process_mem - static_process)

    streams = 10
    if peak_process_mem > 0:
        single_full = peak_process_mem
        shared_process = static_process + streams * dynamic_process
        multi_process = peak_process_mem * streams
        estimate_note = (
            "基于进程级真实显存(nvidia-smi --query-compute-apps):"
            "静态(模型权重+CUDA context)单进程只占 1 份,dynamic(每流激活缓冲)按 10 流叠加。"
        )
    else:
        load_samples = [v.get("gpu_reserved_mb", 0.0) for v in breakdown.values()]
        static_mb = max(load_samples) if load_samples else 0.0
        dynamic_mb = max(0.0, peak_reserved - static_mb)
        single_full = peak_reserved
        shared_process = static_mb + streams * dynamic_mb
        multi_process = peak_reserved * streams
        estimate_note = (
            "进程级显存不可用,回退到 paddle reserved(仅统计 PP-TSM open 模型,"
            "严重低估静态管线,建议 pip install nvidia-ml-py)。"
        )

    return {
        "input": {"camera_id": camera_id, "frames": n_frames, "avg_fps": round(avg_fps, 2)},
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1)},
        "gpu_memory_mb": {
            "paddle_dygraph_steady_reserved": round(steady_reserved, 1),
            "paddle_dygraph_peak_reserved": round(peak_reserved, 1),
            "paddle_dygraph_peak_allocated": round(peak_allocated, 1),
            "paddle_dygraph_fluctuation_allocated": round(fluctuation, 1),
            "process_after_load_static_mb": round(static_process, 1),
            "process_peak_mb": round(peak_process_mem, 1),
            "process_dynamic_per_stream_mb": round(dynamic_process, 1),
            "nvidia_smi_whole_gpu": {k: round(v, 1) for k, v in smi.items()},
        },
        "cpu_memory_mb": {"peak_rss": round(peak_rss, 1)},
        "model_load_breakdown_mb": {
            "note": "paddle reserved 视角;静态管线显示 0 属正常(dygraph API 看不到静态图推理显存)",
            **{k: round(v.get("gpu_reserved_mb", 0.0), 1) for k, v in breakdown.items()},
        },
        "counters": {
            "open_window_inferences": n_open_windows,
            "pipeline_events": n_pipeline_events,
            "package_opening_events": n_package_opening,
        },
        "ten_stream_estimate_mb": {
            "single_stream_peak": round(single_full, 1),
            "shared_process_batched": round(shared_process, 1),
            "multi_process_x10": round(multi_process, 1),
            "note": estimate_note,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-config", type=Path, required=True, help="管线 yml(含 backend/modules/rules)")
    p.add_argument("--source", required=True, help="摄像头索引(数字)/ RTSP URL / 本地视频路径")
    p.add_argument("--camera-id", default=None, help="覆盖 yml 的 camera_id")
    p.add_argument("--frame-stride", type=int, default=1, help="每 N 源帧处理 1 帧")
    p.add_argument("--source-fps", type=float, default=None, help="流 fps(自动探测失败时指定,默认 25)")
    p.add_argument("--width", type=int, default=None, help="摄像头采集宽度(可选)")
    p.add_argument("--height", type=int, default=None, help="摄像头采集高度(可选)")
    p.add_argument("--backend", choices=("model_free", "paddledet_pphuman"), default=None)
    p.add_argument("--device", default="gpu", help="open 模型 device(默认 gpu)")

    p.add_argument("--open-config", type=Path, default=None, help="PP-TSM open 训练 yaml")
    p.add_argument("--open-weights", type=Path, default=None, help="PP-TSM open .pdparams 权重")
    p.add_argument("--paddlevideo-root", type=Path, default=Path("third_party/PaddleVideo"))
    p.add_argument("--fusion-config", type=Path, default=Path("shoplift/configs/fusion.example.yml"))
    p.add_argument("--timeline-fps", type=float, default=None)
    p.add_argument("--crop-padding", type=float, default=None)
    p.add_argument("--fusion-every", type=int, default=5, help="每 N 帧跑一次融合")

    p.add_argument("--output-dir", type=Path, default=Path("outputs/realtime/single_stream"))
    p.add_argument("--no-run-suffix", action="store_true",
                   help="不自动加时间戳,直接写 --output-dir(会覆盖上一次结果)")
    p.add_argument("--max-frames", type=int, default=None, help="处理 N 帧后停止")
    p.add_argument("--duration", type=float, default=None, help="运行 N 秒后停止")
    p.add_argument("--summary-every", type=float, default=5.0, help="每 N 秒打印一次实时汇总")
    p.add_argument("--per-model", action="store_true", help="逐个模型加载并采样显存拆解")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
