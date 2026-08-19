"""Run the P0 offline shoplift analysis chain on a video or frame directory."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from shoplift.core.types import (
    BodyPose,
    DetectionBox,
    FrameMeta,
    HandRegion,
    PersonAttribute,
    ProxyItemRegion,
    Tracklet,
)
from shoplift.events.event_engine import ShopliftingEventEngine
from shoplift.events.event_schema import risk_event_to_payload
from shoplift.tracking.association import AssociationFrame
from shoplift.vision import (
    ItemContainerDetectionAdapter,
    ItemContainerResult,
    PersonGate,
    PersonGateResult,
    PersonAttributeConfig,
    RuleBasedPersonAttributeEstimator,
    build_proxy_item_regions,
)


FRAME_RESULT_SCHEMA_VERSION = "shoplift.frame_result.v1"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"})


def default_user_output_root() -> Path:
    """Per-user output root: outputs/<user>/, overridable via $SHOPLIFT_USER."""
    user = os.environ.get("SHOPLIFT_USER") or os.environ.get("USER") or "default"
    return Path("outputs") / user


@dataclass(frozen=True)
class RuntimeOptions:
    frame_stride: int = 1
    max_frames: int | None = None
    save_debug_visualization: bool = True
    save_debug_frames: bool = False


@dataclass(frozen=True)
class ModuleOptions:
    person_gate_enabled: bool = True
    person_gate_min_score: float = 0.45
    person_gate_skip_when_empty: bool = True
    pose_recognition_enabled: bool = True
    pose_recognition_min_keypoint_score: float = 0.2
    pose_hand_enabled: bool = False
    pose_hand_min_keypoint_score: float = 0.2
    person_attribute_enabled: bool = True
    person_attribute_min_holding_product_score: float = 0.5
    proxy_item_enabled: bool = True
    item_container_enabled: bool = True
    item_container_min_score: float = 0.35
    item_container_classes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BackendOptions:
    backend_type: str = "model_free"
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    frame_jsonl: Path
    event_json: Path
    debug_dir: Path
    debug_video: Path


@dataclass(frozen=True)
class OfflineConfig:
    camera_id: str
    input_path: Path | None
    input_type: str | None
    runtime: RuntimeOptions
    modules: ModuleOptions
    backend: BackendOptions
    outputs: OutputPaths


@dataclass(frozen=True)
class FramePacket:
    frame: FrameMeta
    image: Any
    source_frame_id: int
    source_uri: str
    input_type: str
    fps: float | None = None


@dataclass(frozen=True)
class VisionBackendResult:
    detections: tuple[DetectionBox, ...] = ()
    person_tracks: tuple[Tracklet, ...] = ()
    body_poses: tuple[BodyPose, ...] = ()
    hand_regions: tuple[HandRegion, ...] = ()
    person_attributes: tuple[PersonAttribute, ...] = ()
    proxy_item_regions: tuple[ProxyItemRegion, ...] = ()
    metadata: dict[str, Any] | None = None


class VisionBackend(Protocol):
    def analyze(self, packet: FramePacket) -> VisionBackendResult:
        ...


class ModelFreeVisionBackend:
    """A deterministic backend for wiring the offline pipeline before model weights."""

    def analyze(self, packet: FramePacket) -> VisionBackendResult:
        return VisionBackendResult(metadata={"backend": "model_free"})


@dataclass(frozen=True)
class FrameAnalysis:
    payload: dict[str, Any]
    person_tracks: tuple[Tracklet, ...]
    body_poses: tuple[BodyPose, ...]
    hand_regions: tuple[HandRegion, ...]
    person_attributes: tuple[PersonAttribute, ...]
    proxy_item_regions: tuple[ProxyItemRegion, ...]
    item_container: ItemContainerResult


@dataclass(frozen=True)
class OfflineAnalysisSummary:
    input_path: Path
    input_type: str
    processed_frames: int
    frame_jsonl: Path
    event_json: Path
    debug_visualization: Path | None
    person_gate_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": str(self.input_path),
            "input_type": self.input_type,
            "processed_frames": self.processed_frames,
            "frame_jsonl": str(self.frame_jsonl),
            "event_json": str(self.event_json),
            "debug_visualization": str(self.debug_visualization)
            if self.debug_visualization is not None
            else None,
            "person_gate_metrics": self.person_gate_metrics,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("shoplift/configs/pipeline.example.yml"))
    parser.add_argument("--input", type=Path, default=None, help="Video path or frame directory.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without processing frames.")
    parser.add_argument("--frame-stride", type=int, default=None, help="Process one frame every N source frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames.")
    parser.add_argument("--no-debug", action="store_true", help="Do not write debug visualization artifacts.")
    parser.add_argument(
        "--debug-frames",
        action="store_true",
        help="For video input, also save sampled debug frames as JPG images.",
    )
    parser.add_argument(
        "--backend",
        choices=("model_free", "paddledet_pphuman"),
        default=None,
        help="Override the configured vision backend.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_offline_config(args)
        if args.dry_run:
            print(json.dumps(dry_run_payload(config), ensure_ascii=False, indent=2))
            return 0
        summary = run_offline_analysis(config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 2

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def load_offline_config(args: argparse.Namespace) -> OfflineConfig:
    config_data = _load_yaml(args.config)
    input_path = args.input or _path_or_none(_nested_get(config_data, ("input", "path")))
    input_type = None if args.input is not None else _str_or_none(_nested_get(config_data, ("input", "type")))
    output_root = args.output or default_user_output_root()

    runtime_section = _mapping_at(config_data, "runtime")
    runtime = RuntimeOptions(
        frame_stride=_positive_int(
            args.frame_stride
            if args.frame_stride is not None
            else runtime_section.get("frame_stride", 1),
            "frame_stride",
        ),
        max_frames=_optional_positive_int(
            args.max_frames if args.max_frames is not None else runtime_section.get("max_frames"),
            "max_frames",
        ),
        save_debug_visualization=bool(
            runtime_section.get("save_debug_visualization", True)
        )
        and not args.no_debug,
        save_debug_frames=(
            bool(runtime_section.get("save_debug_frames", False))
            or bool(getattr(args, "debug_frames", False))
        )
        and not args.no_debug,
    )

    modules_section = _mapping_at(config_data, "modules")
    person_gate_section = _mapping_at(modules_section, "person_gate")
    pose_recognition_section = _mapping_at(modules_section, "pose_recognition")
    pose_hand_section = _mapping_at(modules_section, "pose_hand")
    person_attribute_section = _mapping_at(modules_section, "person_attribute")
    proxy_item_section = _mapping_at(modules_section, "proxy_item")
    item_container_section = _mapping_at(modules_section, "item_container")
    configured_classes = item_container_section.get("classes")
    modules = ModuleOptions(
        person_gate_enabled=bool(person_gate_section.get("enabled", True)),
        person_gate_min_score=float(person_gate_section.get("min_score", 0.45)),
        person_gate_skip_when_empty=bool(person_gate_section.get("skip_when_empty", True)),
        pose_recognition_enabled=bool(pose_recognition_section.get("enabled", True)),
        pose_recognition_min_keypoint_score=float(pose_recognition_section.get("min_keypoint_score", 0.2)),
        pose_hand_enabled=bool(pose_hand_section.get("enabled", False)),
        pose_hand_min_keypoint_score=float(pose_hand_section.get("min_keypoint_score", 0.2)),
        person_attribute_enabled=bool(person_attribute_section.get("enabled", True)),
        person_attribute_min_holding_product_score=float(
            person_attribute_section.get("min_holding_product_score", 0.5)
        ),
        proxy_item_enabled=bool(proxy_item_section.get("enabled", True)),
        item_container_enabled=bool(item_container_section.get("enabled", True)),
        item_container_min_score=float(item_container_section.get("min_score", 0.35)),
        item_container_classes=tuple(str(item) for item in configured_classes)
        if isinstance(configured_classes, Sequence) and not isinstance(configured_classes, (str, bytes))
        else None,
    )

    backend_section = _mapping_at(config_data, "backend")
    backend_type = args.backend or str(backend_section.get("type", "model_free"))
    backend = BackendOptions(
        backend_type=backend_type,
        options={key: value for key, value in backend_section.items() if key != "type"},
    )

    outputs_section = _mapping_at(config_data, "outputs")
    outputs = output_paths_from_config(outputs_section, output_root, force_output_root=args.output is not None)

    return OfflineConfig(
        camera_id=str(config_data.get("camera_id", "demo-camera-001")),
        input_path=input_path,
        input_type=input_type,
        runtime=runtime,
        modules=modules,
        backend=backend,
        outputs=outputs,
    )


def run_offline_analysis(
    config: OfflineConfig,
    backend: VisionBackend | None = None,
) -> OfflineAnalysisSummary:
    if config.input_path is None:
        raise ValueError("input path is required; pass --input or configure input.path")
    input_path = config.input_path
    if not input_path.exists():
        raise OSError(f"input path does not exist: {input_path}")

    input_type = detect_input_type(input_path, config.input_type)
    backend = backend or create_vision_backend(config.backend)
    person_gate = PersonGate(
        min_score=config.modules.person_gate_min_score,
        skip_when_empty=config.modules.person_gate_skip_when_empty,
    )
    event_engine = ShopliftingEventEngine()
    emitted_events = []
    item_container_adapter = ItemContainerDetectionAdapter(
        min_score=config.modules.item_container_min_score,
        allowed_categories=frozenset(config.modules.item_container_classes)
        if config.modules.item_container_classes is not None
        else None,
    )
    attribute_estimator = RuleBasedPersonAttributeEstimator()

    config.outputs.frame_jsonl.parent.mkdir(parents=True, exist_ok=True)
    config.outputs.event_json.parent.mkdir(parents=True, exist_ok=True)
    if config.runtime.save_debug_visualization:
        config.outputs.debug_dir.mkdir(parents=True, exist_ok=True)

    processed_frames = 0
    debug_writer = DebugVisualizationWriter(
        input_type=input_type,
        output_dir=config.outputs.debug_dir,
        output_video=config.outputs.debug_video,
        enabled=config.runtime.save_debug_visualization,
        frame_stride=config.runtime.frame_stride,
        save_frame_images=config.runtime.save_debug_frames,
    )
    try:
        with config.outputs.frame_jsonl.open("w", encoding="utf-8") as jsonl:
            for packet in iter_frames(
                input_path,
                input_type=input_type,
                camera_id=config.camera_id,
                frame_stride=config.runtime.frame_stride,
                max_frames=config.runtime.max_frames,
            ):
                analysis = analyze_packet_components(
                    packet,
                    backend=backend,
                    person_gate=person_gate,
                    item_container_adapter=item_container_adapter,
                    attribute_estimator=attribute_estimator,
                    modules=config.modules,
                )
                frame_result = analysis.payload
                event_result = event_engine.process_frame(
                    AssociationFrame(
                        frame_id=packet.frame.frame_id,
                        timestamp_ms=packet.frame.timestamp_ms,
                        camera_id=packet.frame.camera_id,
                        person_tracks=analysis.person_tracks,
                        body_poses=analysis.body_poses,
                        hand_regions=analysis.hand_regions,
                        items=analysis.item_container.items
                        + tuple(region.to_detection_box() for region in analysis.proxy_item_regions),
                        containers=analysis.item_container.containers,
                        extension_regions=analysis.item_container.extension_regions,
                        metadata={
                            "source_uri": packet.source_uri,
                            "input_type": packet.input_type,
                        },
                    )
                )
                emitted_events.extend(event_result.events)
                frame_result["metadata"]["event_engine"] = {
                    "relation_count": len(event_result.relations),
                    "state_count": len(event_result.states),
                    "event_count": len(event_result.events),
                    "relation_counts": event_result.metadata.get("relation_counts", {}),
                }
                jsonl.write(json.dumps(frame_result, ensure_ascii=False, separators=(",", ":")))
                jsonl.write("\n")
                debug_writer.write(packet, frame_result)
                processed_frames += 1
    finally:
        debug_writer.close()

    with config.outputs.event_json.open("w", encoding="utf-8") as events_file:
        json.dump([risk_event_to_payload(event) for event in emitted_events], events_file, ensure_ascii=False, indent=2)
        events_file.write("\n")

    return OfflineAnalysisSummary(
        input_path=input_path,
        input_type=input_type,
        processed_frames=processed_frames,
        frame_jsonl=config.outputs.frame_jsonl,
        event_json=config.outputs.event_json,
        debug_visualization=debug_writer.artifact_path,
        person_gate_metrics=person_gate.metrics.to_dict(),
    )


def create_vision_backend(options: BackendOptions) -> VisionBackend:
    backend_type = options.backend_type.strip().lower()
    if backend_type in {"model_free", "none"}:
        return ModelFreeVisionBackend()
    if backend_type in {"paddledet_pphuman", "pphuman", "paddledetection"}:
        from shoplift.backends import PaddleDetPPHumanBackend, PaddleDetPPHumanBackendConfig

        return PaddleDetPPHumanBackend(
            PaddleDetPPHumanBackendConfig.from_mapping(options.options or {})
        )
    raise ValueError(f"unsupported backend.type: {options.backend_type}")


def output_paths_from_config(
    outputs_section: Mapping[str, Any],
    output_root: Path,
    *,
    force_output_root: bool = False,
) -> OutputPaths:
    if force_output_root:
        return OutputPaths(
            root=output_root,
            frame_jsonl=output_root / "frame_results.jsonl",
            event_json=output_root / "events.json",
            debug_dir=output_root / "debug",
            debug_video=output_root / "debug_visualization.mp4",
        )

    def _resolve(key: str, default_name: str) -> Path:
        raw = outputs_section.get(key)
        if not raw:
            return output_root / default_name
        path = Path(str(raw))
        return path if path.is_absolute() else output_root / path

    return OutputPaths(
        root=output_root,
        frame_jsonl=_resolve("frame_jsonl", "frame_results.jsonl"),
        event_json=_resolve("event_json", "events.json"),
        debug_dir=_resolve("debug_visualization_dir", "debug"),
        debug_video=_resolve("debug_visualization_video", "debug_visualization.mp4"),
    )


def analyze_packet(
    packet: FramePacket,
    *,
    backend: VisionBackend,
    person_gate: PersonGate,
    item_container_adapter: ItemContainerDetectionAdapter,
    modules: ModuleOptions,
) -> dict[str, Any]:
    return analyze_packet_components(
        packet,
        backend=backend,
        person_gate=person_gate,
        item_container_adapter=item_container_adapter,
        attribute_estimator=RuleBasedPersonAttributeEstimator(),
        modules=modules,
    ).payload


def analyze_packet_components(
    packet: FramePacket,
    *,
    backend: VisionBackend,
    person_gate: PersonGate,
    item_container_adapter: ItemContainerDetectionAdapter,
    attribute_estimator: RuleBasedPersonAttributeEstimator,
    modules: ModuleOptions,
) -> FrameAnalysis:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    backend_result = backend.analyze(packet)
    timings["vision_backend"] = _elapsed_ms(started)

    started = time.perf_counter()
    if modules.person_gate_enabled:
        gate_result = person_gate.evaluate(
            detections=backend_result.detections,
            tracklets=backend_result.person_tracks,
            frame=packet.frame,
        )
    else:
        gate_result = PersonGateResult(
            frame=packet.frame,
            has_person=True,
            should_run_heavy_modules=True,
            skipped_heavy_modules=False,
            person_boxes=(),
            person_tracklets=backend_result.person_tracks,
            reason="person_gate_disabled",
        )
    timings["person_gate"] = _elapsed_ms(started)

    body_poses = backend_result.body_poses if modules.pose_recognition_enabled else ()
    hand_regions = backend_result.hand_regions if modules.pose_hand_enabled else ()
    person_attributes = backend_result.person_attributes if modules.person_attribute_enabled else ()
    proxy_item_regions = backend_result.proxy_item_regions if modules.proxy_item_enabled else ()
    if gate_result.skipped_heavy_modules:
        body_poses = ()
        hand_regions = ()
        person_attributes = ()
        proxy_item_regions = ()

    if (
        modules.person_attribute_enabled
        and not gate_result.skipped_heavy_modules
        and not person_attributes
    ):
        started = time.perf_counter()
        person_attributes = attribute_estimator.estimate(
            frame=packet.frame,
            person_tracks=backend_result.person_tracks,
            hand_regions=hand_regions,
        )
        timings["person_attribute"] = _elapsed_ms(started)

    if (
        modules.proxy_item_enabled
        and modules.person_attribute_enabled
        and not gate_result.skipped_heavy_modules
        and not proxy_item_regions
    ):
        started = time.perf_counter()
        proxy_item_regions = build_proxy_item_regions(
            frame=packet.frame,
            person_attributes=person_attributes,
            hand_regions=hand_regions,
            config=PersonAttributeConfig(
                min_holding_product_score=modules.person_attribute_min_holding_product_score
            ),
        )
        timings["proxy_item"] = _elapsed_ms(started)

    started = time.perf_counter()
    item_container_result = (
        item_container_adapter.adapt(backend_result.detections)
        if modules.item_container_enabled and not gate_result.skipped_heavy_modules
        else ItemContainerResult(metadata={"skipped_by_person_gate": gate_result.skipped_heavy_modules})
    )
    timings["item_container"] = _elapsed_ms(started)

    payload = {
        "schema_version": FRAME_RESULT_SCHEMA_VERSION,
        "frame": packet.frame.to_dict(),
        "person_gate": gate_result.to_dict(),
        "person_tracks": [tracklet.to_dict() for tracklet in backend_result.person_tracks],
        "body_poses": [body_pose.to_dict() for body_pose in body_poses],
        "hand_regions": [hand_region.to_dict() for hand_region in hand_regions],
        "person_attributes": [attribute.to_dict() for attribute in person_attributes],
        "proxy_item_regions": [region.to_dict() for region in proxy_item_regions],
        "item_container": item_container_result.to_dict(),
        "metadata": {
            "input_type": packet.input_type,
            "source_frame_id": packet.source_frame_id,
            "source_uri": packet.source_uri,
            "backend": (backend_result.metadata or {}).get("backend"),
            "backend_metadata": backend_result.metadata or {},
            "timings_ms": timings,
        },
    }
    return FrameAnalysis(
        payload=payload,
        person_tracks=backend_result.person_tracks,
        body_poses=tuple(body_poses),
        hand_regions=tuple(hand_regions),
        person_attributes=tuple(person_attributes),
        proxy_item_regions=tuple(proxy_item_regions),
        item_container=item_container_result,
    )


def iter_frames(
    input_path: Path,
    *,
    input_type: str,
    camera_id: str,
    frame_stride: int,
    max_frames: int | None,
) -> Iterator[FramePacket]:
    if input_type == "frame_dir":
        yield from iter_frame_dir(
            input_path,
            camera_id=camera_id,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        return
    if input_type == "video":
        yield from iter_video(
            input_path,
            camera_id=camera_id,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        return
    raise ValueError(f"unsupported input type: {input_type}")


def iter_frame_dir(
    frame_dir: Path,
    *,
    camera_id: str,
    frame_stride: int,
    max_frames: int | None,
) -> Iterator[FramePacket]:
    cv2 = _import_cv2()
    frame_paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frame_paths:
        raise ValueError(f"frame directory contains no supported images: {frame_dir}")

    processed = 0
    for source_index, frame_path in enumerate(frame_paths):
        if source_index % frame_stride != 0:
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"failed to read frame image: {frame_path}")
        height, width = image.shape[:2]
        frame = FrameMeta(
            frame_id=source_index,
            timestamp_ms=source_index * 33,
            camera_id=camera_id,
            width=width,
            height=height,
            source_uri=str(frame_path),
        )
        yield FramePacket(
            frame=frame,
            image=image,
            source_frame_id=source_index,
            source_uri=str(frame_path),
            input_type="frame_dir",
        )
        processed += 1
        if max_frames is not None and processed >= max_frames:
            return


def iter_video(
    video_path: Path,
    *,
    camera_id: str,
    frame_stride: int,
    max_frames: int | None,
) -> Iterator[FramePacket]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    processed = 0
    source_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source_index % frame_stride != 0:
                source_index += 1
                continue
            height, width = image.shape[:2]
            timestamp_ms = int(round(source_index * 1000.0 / fps)) if fps > 0 else source_index * 33
            frame = FrameMeta(
                frame_id=source_index,
                timestamp_ms=timestamp_ms,
                camera_id=camera_id,
                width=width,
                height=height,
                source_uri=str(video_path),
            )
            yield FramePacket(
                frame=frame,
                image=image,
                source_frame_id=source_index,
                source_uri=str(video_path),
                input_type="video",
                fps=fps if fps > 0 else None,
            )
            processed += 1
            source_index += 1
            if max_frames is not None and processed >= max_frames:
                return
    finally:
        capture.release()


class DebugVisualizationWriter:
    def __init__(
        self,
        *,
        input_type: str,
        output_dir: Path,
        output_video: Path,
        enabled: bool,
        frame_stride: int = 1,
        save_frame_images: bool = False,
    ) -> None:
        self.input_type = input_type
        self.output_dir = output_dir
        self.output_video = output_video
        self.enabled = enabled
        self.frame_stride = max(1, frame_stride)
        self.save_frame_images = save_frame_images
        self._writer: Any | None = None
        self._artifact_path: Path | None = None
        self._video_failed = False

    @property
    def artifact_path(self) -> Path | None:
        return self._artifact_path

    def write(self, packet: FramePacket, frame_result: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        cv2 = _import_cv2()
        image = packet.image.copy()
        draw_debug_overlay(image, frame_result)
        wrote_frame_image = False
        if self.input_type == "video" and self.save_frame_images:
            self._write_debug_image(cv2, image, packet)
            wrote_frame_image = True
        if self.input_type == "video" and not self._video_failed:
            if self._writer is None:
                self.output_video.parent.mkdir(parents=True, exist_ok=True)
                fps = self._output_video_fps(packet)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self.output_video),
                    fourcc,
                    fps,
                    (packet.frame.width, packet.frame.height),
                )
                if not self._writer.isOpened():
                    self._writer.release()
                    self._writer = None
                    self._video_failed = True
                    self._artifact_path = self.output_dir
                else:
                    self._artifact_path = self.output_video
            if self._writer is not None:
                self._writer.write(image)
                self._artifact_path = self.output_video
                return
        if wrote_frame_image:
            return
        self._write_debug_image(cv2, image, packet)

    def _write_debug_image(self, cv2: Any, image: Any, packet: FramePacket) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        debug_path = self.output_dir / f"frame_{packet.frame.frame_id:06d}.jpg"
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError(f"failed to write debug image: {debug_path}")
        debug_path.write_bytes(encoded.tobytes())
        self._artifact_path = self.output_dir

    def _output_video_fps(self, packet: FramePacket) -> float:
        source_fps = packet.fps or 30.0
        return max(1.0, source_fps / self.frame_stride)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def draw_debug_overlay(image: Any, frame_result: Mapping[str, Any]) -> None:
    for tracklet in frame_result.get("person_tracks", []):
        for box in tracklet.get("boxes", [])[-1:]:
            _draw_bbox(image, box.get("bbox"), (80, 180, 255), tracklet.get("track_id"))
    for box in frame_result.get("person_gate", {}).get("person_boxes", []):
        _draw_bbox(image, box.get("bbox"), (0, 220, 255), box.get("track_id") or box.get("category"))
    for body_pose in frame_result.get("body_poses", []):
        _draw_body_pose(image, body_pose)
    for hand in frame_result.get("hand_regions", []):
        _draw_bbox(image, hand.get("bbox"), (255, 120, 0), f"{hand.get('side')}_hand")
    for proxy in frame_result.get("proxy_item_regions", []):
        _draw_bbox(
            image,
            proxy.get("proxy_bbox"),
            (0, 255, 255),
            f"proxy_{proxy.get('hand_side')}",
        )
    item_container = frame_result.get("item_container", {})
    for item in item_container.get("items", []):
        _draw_bbox(image, item.get("bbox"), (0, 220, 0), item.get("category"))
    for container in item_container.get("containers", []):
        _draw_bbox(image, container.get("bbox"), (255, 0, 180), container.get("category"))
    for region in item_container.get("extension_regions", []):
        _draw_bbox(image, region.get("bbox"), (160, 160, 160), region.get("category"))


def detect_input_type(input_path: Path, configured_type: str | None = None) -> str:
    if configured_type:
        normalized = configured_type.strip().lower()
        if normalized in {"frames", "frame_dir", "image_dir"}:
            return "frame_dir"
        if normalized == "video":
            return "video"
        raise ValueError(f"unsupported configured input.type: {configured_type}")
    if input_path.is_dir():
        return "frame_dir"
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"cannot infer input type for: {input_path}")


def dry_run_payload(config: OfflineConfig) -> dict[str, Any]:
    return {
        "config": {
            "camera_id": config.camera_id,
            "input": str(config.input_path) if config.input_path is not None else None,
            "input_type": config.input_type,
            "frame_stride": config.runtime.frame_stride,
            "max_frames": config.runtime.max_frames,
            "save_debug_visualization": config.runtime.save_debug_visualization,
            "save_debug_frames": config.runtime.save_debug_frames,
            "backend": config.backend.backend_type,
        },
        "outputs": {
            "root": str(config.outputs.root),
            "frame_jsonl": str(config.outputs.frame_jsonl),
            "event_json": str(config.outputs.event_json),
            "debug_dir": str(config.outputs.debug_dir),
            "debug_video": str(config.outputs.debug_video),
        },
        "status": "dry_run_ok",
    }


def _draw_bbox(image: Any, bbox: Sequence[float] | None, color: tuple[int, int, int], label: Any) -> None:
    if bbox is None or len(bbox) != 4:
        return
    cv2 = _import_cv2()
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        text = str(label)
        cv2.putText(
            image,
            text,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_body_pose(image: Any, body_pose: Mapping[str, Any]) -> None:
    cv2 = _import_cv2()
    keypoints = body_pose.get("keypoints") or []
    scores = body_pose.get("scores") or []
    edges = body_pose.get("skeleton_edges") or []
    metadata = body_pose.get("metadata") or {}
    min_score = float(metadata.get("min_keypoint_score", 0.2))
    line_color = (60, 255, 120)
    point_color = (0, 190, 255)
    for edge in edges:
        if not isinstance(edge, Sequence) or len(edge) != 2:
            continue
        start_index, end_index = int(edge[0]), int(edge[1])
        start = _visible_keypoint(keypoints, scores, start_index, min_score)
        end = _visible_keypoint(keypoints, scores, end_index, min_score)
        if start is None or end is None:
            continue
        cv2.line(image, start, end, line_color, 2, cv2.LINE_AA)
    for index in range(len(keypoints)):
        point = _visible_keypoint(keypoints, scores, index, min_score)
        if point is not None:
            cv2.circle(image, point, 3, point_color, -1, cv2.LINE_AA)


def _visible_keypoint(
    keypoints: Sequence[Any],
    scores: Sequence[Any],
    index: int,
    min_score: float,
) -> tuple[int, int] | None:
    if index >= len(keypoints):
        return None
    score = float(scores[index]) if index < len(scores) else 1.0
    if score < min_score:
        return None
    point = keypoints[index]
    if not isinstance(point, Sequence) or len(point) < 2:
        return None
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if x <= 0 and y <= 0:
        return None
    return (x, y)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise OSError(f"config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, content) in enumerate(lines):
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("simple YAML parser expected a list parent")
            parent.append(_parse_simple_yaml_scalar(content[2:].strip()))
            continue

        key, separator, raw_value = content.partition(":")
        if not separator:
            raise ValueError(f"simple YAML parser expected key: value, got: {content}")
        key = key.strip()
        value = raw_value.strip()
        if value == "":
            next_is_list = (
                index + 1 < len(lines)
                and lines[index + 1][0] > indent
                and lines[index + 1][1].startswith("- ")
            )
            child: Any = [] if next_is_list else {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_simple_yaml_scalar(value)
    return root


def _parse_simple_yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for offline analysis") from exc
    return cv2


def _nested_get(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping_at(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _path_or_none(value: Any) -> Path | None:
    return Path(value) if value else None


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


if __name__ == "__main__":
    raise SystemExit(main())
