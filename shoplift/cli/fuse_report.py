"""Offline PP-TSM open × pipeline fusion report CLI.

Run the 4-step fusion pipeline and write the final reportable events:

  1. PP-TSM open 生产推理:复用管线 MOT 人时间线(person crop)→ 16 帧滑窗 open 概率
  2. 事件化:连续 >= min_consecutive_windows 窗 >= theta → 拆包事件
  3. 商品判定门:拆包窗口内连续 >= min_product_confirm_frames 帧 holding_product
  4. 融合:管线同人事件 → 抬到 high + open_packaging;管线无事件 → 独立 package_opening(high)

Usage (from project root, server conda env):
  python -m shoplift.cli.fuse_report \
      --events outputs/<user>/<run>/events.json \
      --frame-results outputs/<user>/<run>/frame_results.jsonl \
      --video datasets/test/test_videos/yulong_store/<video> \
      --config shoplift/configs/paddlevideo/pptsm_open_v2_fight16.yaml \
      --weights outputs/wqh/paddlevideo/pptsm_open_v2_fight16/ppTSM_epoch_00026.pdparams \
      --paddlevideo-root third_party/PaddleVideo \
      --fusion-config shoplift/configs/fusion.example.yml \
      --output outputs/<user>/<run>/fused_events.json

To test the fusion/上报 layer without GPU/weights, pass pre-computed open events:
  python -m shoplift.cli.fuse_report --events ... --frame-results ... \
      --open-events-json outputs/.../open_events.json --fusion-config ... --output ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.events.event_schema import risk_event_to_payload
from shoplift.fusion.fusion_engine import (
    FusionConfig,
    FusionEngine,
    ProductFrameEvidence,
)
from shoplift.pptsm_open.types import OpenPackagingEvent, OpenWindow

DEFAULT_FUSION_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fusion.example.yml"


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------

def load_fusion_config(path: str | Path | None = None) -> tuple[FusionConfig, dict[str, Any]]:
    """Load ``fusion`` and ``inference`` sections from the fusion yml."""
    config_path = Path(path) if path is not None else DEFAULT_FUSION_CONFIG
    if not config_path.exists():
        raise OSError(f"fusion config does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the fusion config") from exc

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"fusion config must be a mapping: {config_path}")

    fusion_section = data.get("fusion", {}) or {}
    field_names = {item.name for item in fields(FusionConfig)}
    unknown = set(fusion_section) - field_names
    if unknown:
        raise ValueError(
            f"unknown fusion config keys: {sorted(unknown)}; "
            f"allowed: {sorted(field_names)}"
        )
    fusion_config = FusionConfig(**fusion_section)
    inference = dict(data.get("inference", {}) or {})
    return fusion_config, inference


# ---------------------------------------------------------------------------
# input parsing
# ---------------------------------------------------------------------------

def _attr_holding_product(attr: Mapping[str, Any], min_score: float) -> bool:
    for side in ("left_hand_state", "right_hand_state"):
        head = attr.get(side) or {}
        if head.get("label") == "holding_product":
            try:
                if float(head.get("score", 0.0)) >= min_score:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def read_frame_results(
    path: str | Path,
    *,
    min_holding_product_score: float,
) -> tuple[dict[str, list[tuple[int, int, list[float]]]], dict[str, list[ProductFrameEvidence]]]:
    """One streaming pass over ``frame_results.jsonl``.

    Returns ``(person_boxes, product_evidence)``:
      - ``person_boxes``: person_track_id -> [(source_frame_id, timestamp_ms, bbox)]
      - ``product_evidence``: person_track_id -> [ProductFrameEvidence]
    """
    person_boxes: dict[str, list[tuple[int, int, list[float]]]] = defaultdict(list)
    product_evidence: dict[str, list[ProductFrameEvidence]] = defaultdict(list)

    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            frame = line.get("frame", {})
            frame_id = int(frame.get("frame_id", 0))
            timestamp_ms = int(frame.get("timestamp_ms", 0))

            attrs = line.get("person_attributes", []) or []
            proxies = line.get("proxy_item_regions", []) or []
            holding: set[str] = set()
            for attr in attrs:
                person_id = attr.get("person_track_id")
                if person_id and _attr_holding_product(attr, min_holding_product_score):
                    holding.add(person_id)
            for proxy in proxies:
                person_id = proxy.get("person_track_id")
                if person_id:
                    holding.add(person_id)

            for track in line.get("person_tracks", []) or []:
                person_id = track.get("track_id")
                if not person_id:
                    continue
                boxes = track.get("boxes", []) or []
                if boxes:
                    box = boxes[0]
                    bbox = box.get("bbox")
                    if bbox:
                        person_boxes[person_id].append(
                            (
                                int(box.get("frame_id", frame_id)),
                                int(box.get("timestamp_ms", timestamp_ms)),
                                [float(value) for value in bbox],
                            )
                        )
                product_evidence[person_id].append(
                    ProductFrameEvidence(frame_id, timestamp_ms, person_id in holding)
                )

    return dict(person_boxes), dict(product_evidence)


def load_pipeline_events(path: str | Path) -> list[RiskEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("events", [])
    return [_risk_event_from_dict(item) for item in data]


def load_open_events(path: str | Path) -> list[OpenPackagingEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("open_events", [])
    return [_open_event_from_dict(item) for item in data]


def _risk_event_from_dict(item: Mapping[str, Any]) -> RiskEvent:
    evidence = tuple(
        RelationEvidence(
            relation_type=entry["relation_type"],
            frame_id=int(entry["frame_id"]),
            timestamp_ms=int(entry["timestamp_ms"]),
            score=float(entry["score"]),
            reason_tags=tuple(entry["reason_tags"]),
            person_track_id=entry.get("person_track_id"),
            hand_track_id=entry.get("hand_track_id"),
            item_track_id=entry.get("item_track_id"),
            container_track_id=entry.get("container_track_id"),
            evidence_boxes=dict(entry.get("evidence_boxes", {}) or {}),
            metadata=dict(entry.get("metadata", {}) or {}),
        )
        for entry in item.get("evidence", [])
    )
    return RiskEvent(
        event_id=item["event_id"],
        camera_id=item["camera_id"],
        timestamp_ms=int(item["timestamp_ms"]),
        person_track_id=item["person_track_id"],
        event_type=item["event_type"],
        risk_score=float(item["risk_score"]),
        risk_level=item["risk_level"],
        reason_tags=tuple(item["reason_tags"]),
        evidence=evidence,
        start_timestamp_ms=item.get("start_timestamp_ms"),
        end_timestamp_ms=item.get("end_timestamp_ms"),
        confidence=item.get("confidence"),
        clip_uri=item.get("clip_uri"),
        debug_visualization_uri=item.get("debug_visualization_uri"),
        metadata=dict(item.get("metadata", {}) or {}),
    )


def _open_event_from_dict(item: Mapping[str, Any]) -> OpenPackagingEvent:
    windows = tuple(
        OpenWindow(
            start_timestamp_ms=int(window["start_timestamp_ms"]),
            end_timestamp_ms=int(window["end_timestamp_ms"]),
            open_prob=float(window["open_prob"]),
            frame_id=window.get("frame_id"),
        )
        for window in item.get("windows", [])
    )
    return OpenPackagingEvent(
        camera_id=item["camera_id"],
        person_track_id=item["person_track_id"],
        windows=windows,
        metadata=dict(item.get("metadata", {}) or {}),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PP-TSM open × pipeline fusion report")
    parser.add_argument("--events", required=True, help="pipeline events.json")
    parser.add_argument("--frame-results", required=True, help="pipeline frame_results.jsonl")
    parser.add_argument("--video", default=None, help="source video (required for inference)")
    parser.add_argument("--config", default=None, help="PP-TSM open training yaml")
    parser.add_argument("--weights", default=None, help="PP-TSM open .pdparams checkpoint")
    parser.add_argument("--paddlevideo-root", default="third_party/PaddleVideo")
    parser.add_argument("--open-events-json", default=None,
                        help="pre-computed open events json (skip inference)")
    parser.add_argument("--fusion-config", default=str(DEFAULT_FUSION_CONFIG))
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--output", required=True, help="fused events json path")
    parser.add_argument("--report", default=None, help="fusion report json path (optional)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fusion_config, inference = load_fusion_config(args.fusion_config)

    pipeline_events = load_pipeline_events(args.events)
    person_boxes, product_evidence = read_frame_results(
        args.frame_results,
        min_holding_product_score=fusion_config.min_holding_product_score,
    )

    camera_id = args.camera_id
    if camera_id is None and pipeline_events:
        camera_id = pipeline_events[0].camera_id
    if camera_id is None:
        camera_id = "unknown"

    if args.open_events_json is not None:
        open_events = load_open_events(args.open_events_json)
    else:
        if not (args.video and args.config and args.weights):
            print("[error] inference requires --video, --config, --weights "
                  "(or pass --open-events-json)", file=sys.stderr)
            return 2
        open_events = _infer_open_events(
            person_boxes,
            camera_id=camera_id,
            video=args.video,
            config=args.config,
            weights=args.weights,
            paddlevideo_root=args.paddlevideo_root,
            fusion_config=fusion_config,
            inference=inference,
        )

    engine = FusionEngine(fusion_config)
    result = engine.fuse(pipeline_events, open_events, product_evidence, person_boxes)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [risk_event_to_payload(event) for event in result.events],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = Path(args.report) if args.report else output_path.with_name("fusion_report.json")
    report_path.write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "escalated_event_ids": list(result.escalated_event_ids),
                "discarded_open_events": [event.to_dict() for event in result.discarded_open_events],
                "open_event_count": len(open_events),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] fused events -> {output_path} ({len(result.events)} events)", flush=True)
    print(f"[done] fusion report -> {report_path}", flush=True)
    print(f"       pipeline={len(pipeline_events)} open={len(open_events)} "
          f"gated={result.metadata.get('gated_open_event_count', 0)} "
          f"merged={result.metadata.get('merged_open_event_count', 0)} "
          f"package_opening={len(result.package_opening_events)} "
          f"discarded_open={len(result.discarded_open_events)}", flush=True)
    return 0


def _infer_open_events(
    person_boxes: Mapping[str, list[tuple[int, int, list[float]]]],
    *,
    camera_id: str,
    video: str,
    config: str,
    weights: str,
    paddlevideo_root: str,
    fusion_config: FusionConfig,
    inference: Mapping[str, Any],
) -> list[OpenPackagingEvent]:
    from shoplift.pptsm_open.eventize import eventize_open_windows
    from shoplift.pptsm_open.infer import OpenModel, build_person_timeline, infer_open_windows

    timeline_fps = float(inference.get("timeline_fps", 3.125))
    device = str(inference.get("device", "gpu"))
    win = int(inference.get("win", 16))
    stride = int(inference.get("stride", 2))
    crop_padding = float(inference.get("crop_padding", 0.15))

    model = OpenModel.load(config, weights, device=device, paddlevideo_root=paddlevideo_root)

    open_events: list[OpenPackagingEvent] = []
    print(f"[open] inferring {len(person_boxes)} person tracks ...", flush=True)
    for person_id, track in sorted(person_boxes.items()):
        anchors = [(frame_id, bbox) for frame_id, _ts, bbox in sorted(track)]
        timeline = build_person_timeline(
            video,
            anchors,
            timeline_fps=timeline_fps,
            padding=crop_padding,
        )
        if not timeline:
            print(f"[open] {person_id}: no timeline, skipped", flush=True)
            continue
        windows = infer_open_windows(
            model,
            timeline,
            camera_id=camera_id,
            person_track_id=person_id,
            win=win,
            stride=stride,
        )
        person_open = eventize_open_windows(
            windows,
            camera_id=camera_id,
            person_track_id=person_id,
            theta=fusion_config.theta,
            min_consecutive_windows=fusion_config.min_consecutive_windows,
        )
        open_events.extend(person_open)
        print(
            f"[open] {person_id}: timeline={len(timeline)} "
            f"windows={len(windows)} events={len(person_open)}",
            flush=True,
        )
    print(f"[open] done, {len(open_events)} open events total", flush=True)
    return open_events


if __name__ == "__main__":
    raise SystemExit(main())
