"""多流并行实时推理(真并行:N worker 线程,每路独立跑完整管线)。

与 multi_stream_bench.py(串行 round-robin)的区别:
  - N 条流各用一个 worker 线程并行跑,模拟真实多摄像头「同时」推理;
  - 模型只加载一次(静态底座共享),但 paddle 静态图 predictor 在多线程下
    可能有线程亲和性 / 锁竞争,这里直接实测真实显存与吞吐;
  - 输出与单流 / 串行多流对齐:metrics.csv / events.jsonl / debug 视频 / summary.json。

用法(同源视频,N 路并行):
  python -m shoplift.cli.multi_stream_parallel \
      --pipeline-config shoplift/configs/pipeline.cc015_v7b_classw102_test.yml \
      --source datasets/test/test_videos/yulong_store/cc015bb3e050a8bc67ebbb2e6fd9951d.mp4 \
      --open-config shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml \
      --open-weights outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams \
      --paddlevideo-root third_party/PaddleVideo \
      --fusion-config shoplift/configs/fusion.example.yml \
      --num-streams 4 --duration 60 --output-dir outputs/realtime/multi_parallel

用法(多路不同源,逗号分隔):
      ... --sources /dev/video0,/dev/video1 ...

注意:单个 /dev/videoX 摄像头只能被打开一次;要测 N 路真并行,请用 N 个不同源
(N 个视频路径或 N 个摄像头),或同一视频路径(每路独立 reader 从头读)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2

from shoplift.cli.offline_analyze import (
    analyze_packet_components,
    create_vision_backend,
)
from shoplift.cli.realtime_infer import (
    DEFAULT_TIMELINE_FPS,
    RealtimeOpenEngine,
    ResourceMonitor,
    StreamSource,
    _load_fusion_config,
    _quiet,
    _resolve_output_dir,
    build_config,
    build_product_evidence,
    draw_frame,
    load_backend_with_breakdown,
    save_alert_shot,
)
from shoplift.configs.rules_loader import load_rules_config
from shoplift.events.event_engine import ShopliftingEventEngine
from shoplift.events.event_schema import risk_event_to_payload
from shoplift.fusion.fusion_engine import FusionEngine, FusionConfig
from shoplift.pptsm_open.infer import OpenModel
from shoplift.pptsm_open.types import OpenPackagingEvent
from shoplift.rules.risk_score import RiskScorer
from shoplift.rules.validators import RiskRuleValidator
from shoplift.tracking.association import AssociationFrame
from shoplift.vision import (
    ItemContainerDetectionAdapter,
    PersonGate,
    RuleBasedPersonAttributeEstimator,
)

FIELD_NAMES = [
    "stream_id", "frame_id", "timestamp_ms", "wall_s",
    "pipeline_ms", "event_ms", "open_ms", "fusion_ms", "total_ms", "fps",
    "gpu_allocated_mb", "gpu_reserved_mb", "gpu_max_allocated_mb", "gpu_max_reserved_mb",
    "gpu_process_used_mb", "gpu_total_mb", "gpu_used_mb", "cpu_rss_mb",
    "n_tracks", "n_open_windows", "n_pipeline_events", "n_package_opening",
]

_STOP = object()


def _is_camera_source(source: str) -> bool:
    """摄像头/索引源(同一设备通常只能开一次):数字索引或 /dev/video*。"""
    s = str(source).strip()
    return s.isdigit() or s.startswith("/dev/video")


class _BroadcastSource:
    """从一个共享队列取帧:单 reader 读一次,把同一帧复制给 N 个并行消费者。"""

    def __init__(self, q: "queue.Queue[Any]", camera_id: str, fps: float, stop_flag: threading.Event) -> None:
        self._q = q
        self.camera_id = camera_id
        self.fps = fps
        self._stop = stop_flag

    def next(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _STOP:
                return None
            return item
        return None


@dataclass
class _StreamRuntime:
    """一条流的全部状态(open/融合/事件引擎各流独立)。"""
    idx: int
    source: Any
    camera_id: str
    open_engine: Optional[RealtimeOpenEngine]
    fusion_engine: Optional[FusionEngine]
    event_engine: ShopliftingEventEngine
    pipeline_events: list = field(default_factory=list)
    open_events: list = field(default_factory=list)
    product_evidence: dict = field(default_factory=dict)
    person_boxes: dict = field(default_factory=dict)
    emitted_event_ids: set = field(default_factory=set)
    active_tracks_last_seen: dict = field(default_factory=dict)
    n_frames: int = 0
    n_open_windows_total: int = 0
    n_package_opening_total: int = 0
    video_writer: Any = None
    debug_video_path: Optional[Path] = None
    video_fps: float = 15.0
    alert_dir: Optional[Path] = None
    diag_window_idx: dict = field(default_factory=dict)


@dataclass
class _Shared:
    backend: Any
    open_model: Optional[OpenModel]
    fusion_config: Optional[FusionConfig]
    inference: dict[str, Any]
    person_gate: PersonGate
    item_container_adapter: ItemContainerDetectionAdapter
    attribute_estimator: RuleBasedPersonAttributeEstimator
    modules: Any
    monitor: ResourceMonitor
    writer: Any
    csv_file: Any
    events_file: Any
    samples: list
    latencies: list
    peak_process_mem: list
    peak_rss: list
    start_wall: float
    args: argparse.Namespace
    stop_flag: threading.Event
    lock: threading.Lock = field(default_factory=threading.Lock)
    diag_hand_file: Any = None
    diag_open_file: Any = None


def _process_frame(rt: _StreamRuntime, packet: Any, sh: _Shared) -> None:
    """处理一条流的一帧:完整管线 + 事件 + open + 融合 + 采样 + debug 视频。

    推理(analyze / open / event / fusion)不持锁、真正并行;
    只有「共享输出 + 监控采样」在锁内串行化。
    """
    t1 = time.perf_counter()
    analysis = analyze_packet_components(
        packet,
        backend=sh.backend,
        person_gate=sh.person_gate,
        item_container_adapter=sh.item_container_adapter,
        attribute_estimator=sh.attribute_estimator,
        modules=sh.modules,
    )
    t2 = time.perf_counter()

    hand_rows: list[dict[str, Any]] = []
    if sh.diag_hand_file is not None:
        for attr in analysis.person_attributes:
            head = attr.metadata.get("head_probs") or {}
            hand_rows.append({
                "stream_id": rt.idx,
                "frame_id": packet.frame.frame_id,
                "timestamp_ms": packet.frame.timestamp_ms,
                "track_id": attr.person_track_id,
                "left_state_probs": head.get("left_hand_state"),
                "right_state_probs": head.get("right_hand_state"),
                "left_visibility_probs": head.get("left_hand_visibility"),
                "right_visibility_probs": head.get("right_hand_visibility"),
                "left_state": attr.left_hand_state.label,
                "left_score": round(float(attr.left_hand_state.score), 6),
                "right_state": attr.right_hand_state.label,
                "right_score": round(float(attr.right_hand_state.score), 6),
                "body_orientation": attr.body_orientation.label,
            })

    event_result = rt.event_engine.process_frame(
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
    rt.pipeline_events.extend(event_result.events)
    t3 = time.perf_counter()

    new_open_events: list[OpenPackagingEvent] = []
    track_rows: list[tuple[str, list[float], int, int]] = []
    for track in analysis.person_tracks:
        if not track.boxes:
            continue
        box = track.boxes[-1]
        track_rows.append((track.track_id, list(box.bbox), packet.frame.frame_id, packet.frame.timestamp_ms))
        rt.person_boxes.setdefault(track.track_id, []).append(
            (packet.frame.frame_id, packet.frame.timestamp_ms, list(box.bbox))
        )
        rt.active_tracks_last_seen[track.track_id] = packet.frame.frame_id

    if rt.open_engine is not None:
        new_open_events = rt.open_engine.on_frame(rt.camera_id, packet.image, track_rows)
        rt.n_open_windows_total = rt.open_engine.window_inferences
    t4 = time.perf_counter()

    open_rows: list[dict[str, Any]] = []
    if sh.diag_open_file is not None and rt.open_engine is not None:
        for tid, st in rt.open_engine._tracks.items():
            dumped = rt.diag_window_idx.get(tid, 0)
            while dumped < len(st.windows):
                w = st.windows[dumped]
                open_rows.append({
                    "stream_id": rt.idx,
                    "track_id": tid,
                    "start_ms": w.start_timestamp_ms,
                    "end_ms": w.end_timestamp_ms,
                    "open_prob": round(float(w.open_prob), 6),
                    "frame_id": w.frame_id,
                })
                dumped += 1
            rt.diag_window_idx[tid] = dumped

    evidence = build_product_evidence(
        packet.frame.frame_id,
        packet.frame.timestamp_ms,
        analysis.person_tracks,
        analysis.person_attributes,
        analysis.proxy_item_regions,
        sh.modules.person_attribute_min_holding_product_score,
    )
    for tid, ev in evidence.items():
        rt.product_evidence.setdefault(tid, []).append(ev)
    rt.open_events.extend(new_open_events)

    fusion_ms = 0.0
    fusion_alerts: list[Any] = []
    if rt.fusion_engine is not None and rt.open_events and (rt.n_frames % sh.args.fusion_every == 0):
        t4b = time.perf_counter()
        result = rt.fusion_engine.fuse(rt.pipeline_events, rt.open_events, rt.product_evidence, rt.person_boxes)
        for evt in result.package_opening_events:
            if evt.event_id not in rt.emitted_event_ids:
                rt.emitted_event_ids.add(evt.event_id)
                rt.n_package_opening_total += 1
                fusion_alerts.append(evt)
        fusion_ms = (time.perf_counter() - t4b) * 1000.0
    t5 = time.perf_counter()

    # ---- 共享资源(锁内):监控采样 + 峰值更新 + csv/events/diag 写 ----
    with sh.lock:
        sample = sh.monitor.sample()
        if sample.get("gpu_process_used_mb"):
            sh.peak_process_mem[0] = max(sh.peak_process_mem[0], sample["gpu_process_used_mb"])
        if sample.get("cpu_rss_mb", -1) >= 0:
            sh.peak_rss[0] = max(sh.peak_rss[0], sample["cpu_rss_mb"])
        sample["stream_id"] = rt.idx
        sample["frame_id"] = packet.frame.frame_id
        sample["timestamp_ms"] = packet.frame.timestamp_ms
        sample["wall_s"] = round(time.perf_counter() - sh.start_wall, 3)
        sample["pipeline_ms"] = (t2 - t1) * 1000.0
        sample["event_ms"] = (t3 - t2) * 1000.0
        sample["open_ms"] = (t4 - t3) * 1000.0
        sample["fusion_ms"] = fusion_ms
        sample["total_ms"] = (t5 - t1) * 1000.0
        sample["fps"] = 1.0 / max(1e-6, t5 - t1)
        sample["n_tracks"] = len(track_rows)
        sample["n_open_windows"] = rt.n_open_windows_total
        sample["n_pipeline_events"] = len(rt.pipeline_events)
        sample["n_package_opening"] = rt.n_package_opening_total
        sh.writer.writerow({k: sample.get(k, "") for k in FIELD_NAMES})
        sh.csv_file.flush()
        sh.samples.append(sample)
        sh.latencies.append(sample["total_ms"])

        for evt in event_result.events:
            sh.events_file.write(json.dumps(risk_event_to_payload(evt), ensure_ascii=False) + "\n")
        for evt in fusion_alerts:
            sh.events_file.write(json.dumps(risk_event_to_payload(evt), ensure_ascii=False) + "\n")
        sh.events_file.flush()

        if sh.diag_hand_file is not None:
            for row in hand_rows:
                sh.diag_hand_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if sh.diag_open_file is not None:
            for row in open_rows:
                sh.diag_open_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    rt.n_frames += 1

    # 告警打印 + 截图(每流独立目录,不锁)
    for evt in event_result.events:
        if evt.risk_level == "high":
            print(
                f"[ALERT s{rt.idx}] {evt.event_type} person={evt.person_track_id} "
                f"score={evt.risk_score:.2f} t={evt.timestamp_ms}ms",
                flush=True,
            )
            save_alert_shot(packet, analysis, evt.person_track_id, evt.event_type, evt.timestamp_ms, rt.alert_dir)
        elif evt.risk_level == "medium":
            print(
                f"[EVENT s{rt.idx}] {evt.event_type} person={evt.person_track_id} "
                f"level={evt.risk_level} t={evt.timestamp_ms}ms",
                flush=True,
            )
    for evt in fusion_alerts:
        print(
            f"[ALERT s{rt.idx}] {evt.event_type} person={evt.person_track_id} "
            f"prob={evt.confidence:.3f} t={evt.start_timestamp_ms}-{evt.end_timestamp_ms}ms",
            flush=True,
        )
        save_alert_shot(
            packet, analysis, evt.person_track_id, evt.event_type, evt.start_timestamp_ms, rt.alert_dir
        )

    # debug 视频(每流独立 writer,不锁)
    if packet.image is not None:
        try:
            annotated = draw_frame(packet.image, analysis)
            if rt.video_writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rt.video_writer = cv2.VideoWriter(str(rt.debug_video_path), fourcc, rt.video_fps, (w, h))
            if rt.video_writer.isOpened():
                rt.video_writer.write(annotated)
        except Exception:
            pass


def _finalize_stream(rt: _StreamRuntime, sh: _Shared) -> None:
    if rt.open_engine is None or rt.fusion_engine is None:
        return
    for tid in list(rt.open_engine._tracks.keys()):
        tail = rt.open_engine.finalize_track(rt.camera_id, tid)
        rt.open_events.extend(tail)
    result = rt.fusion_engine.fuse(rt.pipeline_events, rt.open_events, rt.product_evidence, rt.person_boxes)
    alerts: list[Any] = []
    for evt in result.package_opening_events:
        if evt.event_id not in rt.emitted_event_ids:
            rt.emitted_event_ids.add(evt.event_id)
            rt.n_package_opening_total += 1
            alerts.append(evt)
    with sh.lock:
        for evt in alerts:
            sh.events_file.write(json.dumps(risk_event_to_payload(evt), ensure_ascii=False) + "\n")
        sh.events_file.flush()
    for evt in alerts:
        print(
            f"[ALERT s{rt.idx}] {evt.event_type} person={evt.person_track_id} "
            f"prob={evt.confidence:.3f} t={evt.start_timestamp_ms}-{evt.end_timestamp_ms}ms",
            flush=True,
        )


def _make_open_engine(
    open_model: Optional[OpenModel], fusion_config: Optional[FusionConfig],
    inference: dict[str, Any], source_fps: float, args: argparse.Namespace,
) -> Optional[RealtimeOpenEngine]:
    if open_model is None or fusion_config is None:
        return None
    return RealtimeOpenEngine(
        open_model,
        timeline_fps=float(args.timeline_fps or inference.get("timeline_fps", DEFAULT_TIMELINE_FPS)),
        source_fps=source_fps,
        win=int(inference.get("win", 16)),
        stride=int(inference.get("stride", 2)),
        crop_padding=float(args.crop_padding if args.crop_padding is not None else inference.get("crop_padding", 0.15)),
        theta=fusion_config.theta,
        min_consecutive_windows=fusion_config.min_consecutive_windows,
    )


def _build_summary(
    runtimes: list[_StreamRuntime], shared: _Shared, num_streams: int,
    static_process: float, elapsed: float,
) -> dict[str, Any]:
    peak_process = shared.peak_process_mem[0]
    dynamic_total = max(0.0, peak_process - static_process)
    # 并行:N 路同时推理,峰值动态 = N × 单次推理动态;除以 N 即「每并发推理」的增量
    dynamic_per_stream = dynamic_total / max(1, num_streams)
    implied_single = static_process + dynamic_per_stream

    total_frames = len(shared.samples)
    aggregate_fps = total_frames / max(1e-6, elapsed)

    lat = sorted(shared.latencies)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0

    smi = shared.monitor.nvidia_smi()
    shared_10 = static_process + 10 * dynamic_per_stream

    per_stream = []
    total_open = total_events = total_pkg = 0
    for rt in runtimes:
        per_stream.append({
            "stream": rt.idx,
            "frames": rt.n_frames,
            "open_windows": rt.n_open_windows_total,
            "pipeline_events": len(rt.pipeline_events),
            "package_opening": rt.n_package_opening_total,
        })
        total_open += rt.n_open_windows_total
        total_events += len(rt.pipeline_events)
        total_pkg += rt.n_package_opening_total

    return {
        "mode": "parallel",
        "input": {
            "num_streams": num_streams,
            "total_frames": total_frames,
            "elapsed_s": round(elapsed, 1),
        },
        "gpu_memory_mb": {
            "static_after_load": round(static_process, 1),
            "n_stream_peak": round(peak_process, 1),
            "dynamic_total": round(dynamic_total, 1),
            "dynamic_per_concurrent_stream": round(dynamic_per_stream, 1),
            "implied_single_peak": round(implied_single, 1),
            "nvidia_smi_whole_gpu": {k: round(v, 1) for k, v in smi.items()},
        },
        "throughput": {
            "aggregate_fps": round(aggregate_fps, 2),
            "per_stream_fps": round(aggregate_fps / max(1, num_streams), 2),
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
        },
        "cpu_memory_mb": {"peak_rss": round(shared.peak_rss[0], 1)},
        "counters": {
            "per_stream": per_stream,
            "total_open_windows": total_open,
            "total_pipeline_events": total_events,
            "total_package_opening": total_pkg,
        },
        "ten_stream_estimate_mb": {
            "shared_process_batched": round(shared_10, 1),
            "note": (
                "并行 N 流实测:静态底座只占 1 份,dynamic 按实测每并发流增量 "
                f"({dynamic_per_stream:.1f}MB)×10 叠加(峰值=静态+10×每并发增量)。"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-config", type=Path, required=True)
    p.add_argument("--source", required=True, help="默认同源:摄像头索引/RTSP URL/本地视频路径")
    p.add_argument("--sources", default=None, help="逗号分隔的多路源(覆盖 --source/--num-streams),用于 N 路不同源")
    p.add_argument("--camera-id", default=None)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--source-fps", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--backend", choices=("model_free", "paddledet_pphuman"), default=None)
    p.add_argument("--device", default="gpu")

    p.add_argument("--open-config", type=Path, default=None)
    p.add_argument("--open-weights", type=Path, default=None)
    p.add_argument("--paddlevideo-root", type=Path, default=Path("third_party/PaddleVideo"))
    p.add_argument("--fusion-config", type=Path, default=Path("shoplift/configs/fusion.example.yml"))
    p.add_argument("--timeline-fps", type=float, default=None)
    p.add_argument("--crop-padding", type=float, default=None)
    p.add_argument("--fusion-every", type=int, default=5, help="每 N 帧跑一次融合")

    p.add_argument("--num-streams", type=int, default=2)
    p.add_argument("--broadcast", action="store_true", help="单 reader 读一次源(摄像头),帧复制给 N 个并行 worker")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/realtime/multi_parallel"))
    p.add_argument("--no-run-suffix", action="store_true")
    p.add_argument("--max-frames", type=int, default=None, help="每条流处理 N 帧后停")
    p.add_argument("--duration", type=float, default=None, help="运行 N 秒后停")
    p.add_argument("--per-model", action="store_true")
    p.add_argument("--summary-every", type=float, default=2.0, help="终端 [live] 打印间隔(秒)")
    p.add_argument(
        "--dump-diag",
        action="store_true",
        help="诊断输出:diag_hand.jsonl(每帧每人手部状态完整概率) + diag_open.jsonl(每窗口 open 概率)",
    )
    return p


def _worker_loop(rt: _StreamRuntime, sh: _Shared) -> None:
    while not sh.stop_flag.is_set():
        if sh.args.duration and (time.perf_counter() - sh.start_wall) >= sh.args.duration:
            break
        try:
            packet = rt.source.next()
        except Exception as exc:
            print(f"[error] stream {rt.idx} 源读取失败: {exc!r}", flush=True)
            break
        if packet is None:
            break
        try:
            _process_frame(rt, packet, sh)
        except Exception as exc:
            print(f"[error] stream {rt.idx} frame {packet.frame.frame_id}: {exc!r}", flush=True)
            import traceback

            traceback.print_exc()
            break
        if sh.args.max_frames and rt.n_frames >= sh.args.max_frames:
            break


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("GLOG_minloglevel", "2")
    import logging

    for _name in ("paddlevideo", "ppdet", "PaddleClas", "paddle"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    # 解析多路源
    if args.sources:
        source_list = [s.strip() for s in args.sources.split(",") if s.strip()]
        num_streams = len(source_list)
    else:
        num_streams = max(1, args.num_streams)
        source_list = [args.source] * num_streams

    # Ctrl-C 尽早注册:第一次=优雅停止,第二次=强制退出
    stop_flag = threading.Event()
    _readers_box: list[Any] = []
    _interrupted = [False]

    def _sig(signum, frame):
        if _interrupted[0]:
            print("\n[interrupt] 强制退出", flush=True)
            os._exit(130)
        _interrupted[0] = True
        print("\n[interrupt] 收到 Ctrl-C,停止中(再按一次强制退出)...", flush=True)
        stop_flag.set()
        for src in _readers_box:
            try:
                src.close()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    config = build_config(args)
    monitor = ResourceMonitor()

    out_dir = _resolve_output_dir(Path(args.output_dir), not args.no_run_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output-dir] {out_dir}", flush=True)

    # ---- 1. 加载模型(只一次,全流共享) ----
    print("[load] baseline", flush=True)
    backend = create_vision_backend(config.backend)
    breakdown: dict[str, float] = {}
    _quiet(lambda: load_backend_with_breakdown(backend, monitor, breakdown, args.per_model))

    open_model: Optional[OpenModel] = None
    fusion_config: Optional[FusionConfig] = None
    inference: dict[str, Any] = {}
    if args.open_config and args.open_weights:
        import paddle

        paddle.disable_static()
        open_model = _quiet(
            lambda: OpenModel.load(
                args.open_config, args.open_weights, device=args.device, paddlevideo_root=args.paddlevideo_root
            )
        )
        fusion_config, inference = _load_fusion_config(args.fusion_config)

    static_process = monitor.process_gpu_mem().get("gpu_process_used_mb", 0.0)
    print(f"[load] static process gpu mem: {static_process:.1f} MB", flush=True)
    if stop_flag.is_set():
        print("[interrupt] 加载阶段已收到 Ctrl-C,退出", flush=True)
        return 130

    person_gate = PersonGate(
        min_score=config.modules.person_gate_min_score,
        skip_when_empty=config.modules.person_gate_skip_when_empty,
    )
    item_container_adapter = ItemContainerDetectionAdapter(
        min_score=config.modules.item_container_min_score,
        allowed_categories=frozenset(config.modules.item_container_classes)
        if config.modules.item_container_classes is not None
        else None,
    )
    attribute_estimator = RuleBasedPersonAttributeEstimator()
    rules = load_rules_config(config.rules_config)

    # ---- 2. N 个 reader(独立或广播:单 reader 读一次复制给 N 个并行 worker) ----
    base_cam = config.camera_id
    sources: list[Any] = []
    readers: list[Any] = []
    broadcast_queues: list[Any] = []
    broadcast_reader: Optional[StreamSource] = None
    use_broadcast = args.broadcast or (
        num_streams > 1 and len(set(source_list)) == 1 and _is_camera_source(source_list[0])
    )
    if use_broadcast and not args.broadcast:
        print("[broadcast] 检测到单摄像头源且 num_streams>1,自动开启 --broadcast(读一次复制给 N 个并行 worker)", flush=True)
    if use_broadcast and num_streams > 1:
        broadcast_reader = StreamSource(
            source_list[0], camera_id=base_cam, frame_stride=args.frame_stride,
            source_fps=args.source_fps, width=args.width, height=args.height,
        )
        readers = [broadcast_reader]
        broadcast_queues = [queue.Queue(maxsize=8) for _ in range(num_streams)]
        for i in range(num_streams):
            cam = base_cam if num_streams == 1 else f"{base_cam}_s{i}"
            sources.append(_BroadcastSource(broadcast_queues[i], cam, broadcast_reader.fps, stop_flag))
    else:
        for i in range(num_streams):
            cam = base_cam if num_streams == 1 else f"{base_cam}_s{i}"
            src = StreamSource(
                source_list[i], camera_id=cam, frame_stride=args.frame_stride,
                source_fps=args.source_fps, width=args.width, height=args.height,
            )
            sources.append(src)
            readers.append(src)
    _readers_box[:] = readers

    # ---- 3. 每条流独立 runtime ----
    runtimes: list[_StreamRuntime] = []
    for i, source in enumerate(sources):
        cam = base_cam if num_streams == 1 else f"{base_cam}_s{i}"
        alert_dir = out_dir if num_streams == 1 else out_dir / f"stream_{i}"
        alert_dir.mkdir(parents=True, exist_ok=True)
        runtimes.append(
            _StreamRuntime(
                idx=i,
                source=source,
                camera_id=cam,
                open_engine=_make_open_engine(open_model, fusion_config, inference, source.fps, args),
                fusion_engine=FusionEngine(fusion_config) if fusion_config is not None else None,
                event_engine=ShopliftingEventEngine(
                    association_config=rules.association,
                    risk_scorer=RiskScorer(rules.risk_scoring),
                    rule_validator=RiskRuleValidator(rules.rules),
                    nested_concealment_config=rules.nested_concealment,
                ),
                video_fps=getattr(source, "fps", 0) or 15.0,
                debug_video_path=(out_dir / "debug.mp4") if num_streams == 1 else (out_dir / f"debug_s{i}.mp4"),
                alert_dir=alert_dir,
            )
        )

    # ---- 4. 共享输出 ----
    csv_path = out_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES)
    writer.writeheader()
    events_file = (out_dir / "events.jsonl").open("w", encoding="utf-8")

    diag_hand_file = None
    diag_open_file = None
    if args.dump_diag:
        diag_hand_file = (out_dir / "diag_hand.jsonl").open("w", encoding="utf-8")
        diag_open_file = (out_dir / "diag_open.jsonl").open("w", encoding="utf-8")
        print("[dump-diag] 手部状态 -> diag_hand.jsonl ; open 窗口 -> diag_open.jsonl", flush=True)

    shared = _Shared(
        backend=backend,
        open_model=open_model,
        fusion_config=fusion_config,
        inference=inference,
        person_gate=person_gate,
        item_container_adapter=item_container_adapter,
        attribute_estimator=attribute_estimator,
        modules=config.modules,
        monitor=monitor,
        writer=writer,
        csv_file=csv_file,
        events_file=events_file,
        samples=[],
        latencies=[],
        peak_process_mem=[static_process],
        peak_rss=[0.0],
        start_wall=time.perf_counter(),
        args=args,
        stop_flag=stop_flag,
        diag_hand_file=diag_hand_file,
        diag_open_file=diag_open_file,
    )

    # broadcast fanout:单 reader 读帧,复制到 N 个队列给 N 个并行 worker
    if broadcast_reader is not None:

        def _fanout() -> None:
            try:
                while not stop_flag.is_set():
                    packet = broadcast_reader.next()
                    if packet is None:
                        break
                    for q in broadcast_queues:
                        try:
                            q.put(packet, timeout=1.0)
                        except queue.Full:
                            pass
            except Exception as exc:
                print(f"[broadcast] 读源失败: {exc!r}", flush=True)
            finally:
                for q in broadcast_queues:
                    try:
                        q.put_nowait(_STOP)
                    except queue.Full:
                        pass

        threading.Thread(target=_fanout, name="broadcaster", daemon=True).start()

    print(
        f"[run] {num_streams} parallel streams (fps={sources[0].fps:.2f}, "
        f"stride={args.frame_stride}, summary_every={args.summary_every:.0f}s)",
        flush=True,
    )

    # ---- 5. N 个 worker 线程并行 ----
    workers = [
        threading.Thread(target=_worker_loop, args=(runtimes[i], shared), name=f"worker-{i}", daemon=True)
        for i in range(num_streams)
    ]
    for w in workers:
        w.start()

    # ---- 6. 主线程:[live] 打印 + 等待停止 ----
    last_print = 0.0
    try:
        while True:
            time.sleep(min(args.summary_every, 0.5))
            now = time.perf_counter()
            if now - last_print >= args.summary_every:
                last_print = now
                with shared.lock:
                    n = len(shared.samples)
                    el = max(1e-6, now - shared.start_wall)
                    lat = sorted(shared.latencies)
                    p50 = lat[len(lat) // 2] if lat else 0.0
                    peak = shared.peak_process_mem[0]
                    rss = shared.peak_rss[0]
                print(
                    f"[live] frames={n} fps={n / el:.1f} lat_p50={p50:.0f}ms "
                    f"peak_gpu={peak:.0f}MB rss={rss:.0f}MB",
                    flush=True,
                )
            if stop_flag.is_set():
                break
            if args.duration and (now - shared.start_wall) >= args.duration:
                break
            if all(not w.is_alive() for w in workers):
                break
    finally:
        stop_flag.set()
        for src in readers:
            src.close()
        for q in broadcast_queues:
            try:
                q.put_nowait(_STOP)
            except Exception:
                pass
        for w in workers:
            w.join(timeout=5.0)

        # 1. finalize(每流,写事件需锁)
        for rt in runtimes:
            try:
                _finalize_stream(rt, shared)
            except Exception as exc:
                print(f"[finalize s{rt.idx}] error: {exc!r}", flush=True)

        # 2. 落盘数据文件 + summary(放 release 之前)
        try:
            with shared.lock:
                events_file.flush()
                csv_file.flush()
                if diag_hand_file is not None:
                    diag_hand_file.flush()
                if diag_open_file is not None:
                    diag_open_file.flush()
        except Exception:
            pass
        try:
            elapsed = time.perf_counter() - shared.start_wall
            summary = _build_summary(runtimes, shared, num_streams, static_process, elapsed)
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            print(f"[summary] error: {exc!r}", flush=True)
            import traceback

            traceback.print_exc()

        # 3. 释放视频(放最后)
        for rt in runtimes:
            if rt.video_writer is not None:
                try:
                    rt.video_writer.release()
                except Exception:
                    pass

        # 4. 关文件
        try:
            csv_file.close()
            events_file.close()
            if diag_hand_file is not None:
                diag_hand_file.close()
            if diag_open_file is not None:
                diag_open_file.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
