"""PaddleDetection / PP-Human inference backend.

The backend is intentionally a thin adapter around PaddleDetection deploy
predictors. It does not invent detections when model weights are missing:
enabled modules must point to local exported inference model directories.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from shoplift.adapters.paddledet_adapter import (
    PaddleDetectionAdapter,
    SHOPLIFT_CLASS_ID_TO_CATEGORY,
)
from shoplift.core.types import DetectionBox, Tracklet

if TYPE_CHECKING:
    from shoplift.cli.offline_analyze import FramePacket, VisionBackendResult


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_path(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def _resolve_path(path: Path | None, *, project_root: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else project_root / path


def _mapping_at(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return {}
    data = yaml.safe_load(text) or {}
    return data if isinstance(data, dict) else {}


def _is_url(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _require_local_model_dir(path: Path | str | None, module_name: str) -> str:
    if path is None:
        raise RuntimeError(f"{module_name} model_dir is required when the module is enabled")
    text = str(path)
    if _is_url(text):
        raise RuntimeError(
            f"{module_name} model_dir points to a URL; download/export the model first and configure a local directory: {text}"
        )
    resolved = Path(text)
    if not resolved.exists():
        raise RuntimeError(f"{module_name} model_dir does not exist: {resolved}")
    if not resolved.is_dir():
        raise RuntimeError(f"{module_name} model_dir must be a directory: {resolved}")
    infer_cfg = resolved / "infer_cfg.yml"
    inference_yml = resolved / "inference.yml"
    if not infer_cfg.exists() and not inference_yml.exists():
        raise RuntimeError(
            f"{module_name} model_dir must contain infer_cfg.yml or inference.yml: {resolved}"
        )
    return str(resolved)


@dataclass(frozen=True)
class PaddleDetModuleConfig:
    """One PaddleDetection predictor module configuration."""

    enabled: bool = False
    model_dir: Path | None = None
    batch_size: int = 1
    threshold: float = 0.5


@dataclass(frozen=True)
class PaddleDetMOTConfig(PaddleDetModuleConfig):
    """PP-Human MOT module configuration."""

    enabled: bool = True
    tracker_config: Path | None = None
    skip_frame_num: int = -1


@dataclass(frozen=True)
class PaddleDetPPHumanBackendConfig:
    """Configuration for the PP-Human backend."""

    paddledetection_root: Path = Path("src/PaddleDetection-release-2.9")
    pphuman_config: Path | None = Path("src/PaddleDetection-release-2.9/deploy/pipeline/config/infer_cfg_pphuman.yml")
    device: str = "cpu"
    run_mode: str = "paddle"
    cpu_threads: int = 1
    enable_mkldnn: bool = False
    trt_min_shape: int = 1
    trt_max_shape: int = 1280
    trt_opt_shape: int = 640
    trt_calib_mode: bool = False
    output_dir: Path = Path("outputs/paddledet")
    mot: PaddleDetMOTConfig = field(default_factory=PaddleDetMOTConfig)
    keypoint: PaddleDetModuleConfig = field(default_factory=PaddleDetModuleConfig)
    item_container: PaddleDetModuleConfig = field(default_factory=PaddleDetModuleConfig)
    item_class_id_to_category: Mapping[int, str] = field(default_factory=lambda: dict(SHOPLIFT_CLASS_ID_TO_CATEGORY))

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        project_root: Path | None = None,
    ) -> "PaddleDetPPHumanBackendConfig":
        project_root = project_root or Path.cwd()
        root = _resolve_path(_as_path(mapping.get("paddledetection_root")), project_root=project_root) or project_root / "src/PaddleDetection-release-2.9"
        pphuman_config = _resolve_path(_as_path(mapping.get("pphuman_config")), project_root=project_root)
        if pphuman_config is None:
            pphuman_config = root / "deploy/pipeline/config/infer_cfg_pphuman.yml"
        pphuman_cfg = _load_yaml(pphuman_config)

        mot_mapping = _mapping_at(mapping, "mot")
        kpt_mapping = _mapping_at(mapping, "keypoint")
        item_mapping = _mapping_at(mapping, "item_container")
        pphuman_mot = _mapping_at(pphuman_cfg, "MOT")
        pphuman_kpt = _mapping_at(pphuman_cfg, "KPT")

        tracker_config = _resolve_path(
            _as_path(mot_mapping.get("tracker_config") or pphuman_mot.get("tracker_config")),
            project_root=project_root,
        )
        if tracker_config is None:
            tracker_config = root / "deploy/pipeline/config/tracker_config.yml"

        return cls(
            paddledetection_root=root,
            pphuman_config=pphuman_config,
            device=str(mapping.get("device", "cpu")).lower(),
            run_mode=str(mapping.get("run_mode", "paddle")),
            cpu_threads=int(mapping.get("cpu_threads", 1)),
            enable_mkldnn=_as_bool(mapping.get("enable_mkldnn"), False),
            trt_min_shape=int(mapping.get("trt_min_shape", 1)),
            trt_max_shape=int(mapping.get("trt_max_shape", 1280)),
            trt_opt_shape=int(mapping.get("trt_opt_shape", 640)),
            trt_calib_mode=_as_bool(mapping.get("trt_calib_mode"), False),
            output_dir=_resolve_path(_as_path(mapping.get("output_dir")), project_root=project_root) or project_root / "outputs/paddledet",
            mot=PaddleDetMOTConfig(
                enabled=_as_bool(mot_mapping.get("enabled"), _as_bool(pphuman_mot.get("enable"), True)),
                model_dir=_resolve_path(_as_path(mot_mapping.get("model_dir") or pphuman_mot.get("model_dir")), project_root=project_root),
                tracker_config=tracker_config,
                batch_size=int(mot_mapping.get("batch_size", pphuman_mot.get("batch_size", 1))),
                threshold=float(mot_mapping.get("threshold", pphuman_cfg.get("crop_thresh", 0.5))),
                skip_frame_num=int(mot_mapping.get("skip_frame_num", pphuman_mot.get("skip_frame_num", -1))),
            ),
            keypoint=PaddleDetModuleConfig(
                enabled=_as_bool(kpt_mapping.get("enabled"), False),
                model_dir=_resolve_path(_as_path(kpt_mapping.get("model_dir") or pphuman_kpt.get("model_dir")), project_root=project_root),
                batch_size=int(kpt_mapping.get("batch_size", pphuman_kpt.get("batch_size", 8))),
                threshold=float(kpt_mapping.get("threshold", pphuman_cfg.get("kpt_thresh", 0.2))),
            ),
            item_container=PaddleDetModuleConfig(
                enabled=_as_bool(item_mapping.get("enabled"), False),
                model_dir=_resolve_path(_as_path(item_mapping.get("model_dir")), project_root=project_root),
                batch_size=int(item_mapping.get("batch_size", 1)),
                threshold=float(item_mapping.get("threshold", 0.35)),
            ),
        )


class PaddleDetPPHumanBackend:
    """Run PP-Human MOT/KPT and optional item-container detection per frame."""

    def __init__(self, config: PaddleDetPPHumanBackendConfig | None = None) -> None:
        self.config = config or PaddleDetPPHumanBackendConfig()
        self.person_adapter = PaddleDetectionAdapter(
            min_detection_score=self.config.mot.threshold,
            min_keypoint_score=self.config.keypoint.threshold,
        )
        self.item_adapter = PaddleDetectionAdapter(
            class_id_to_category=self.config.item_class_id_to_category,
            min_detection_score=self.config.item_container.threshold,
        )
        self._mot_predictor: Any | None = None
        self._keypoint_predictor: Any | None = None
        self._item_detector: Any | None = None
        self._crop_image_with_mot: Any | None = None
        self._translate_to_ori_images: Any | None = None

    def analyze(self, packet: "FramePacket") -> "VisionBackendResult":
        from shoplift.cli.offline_analyze import VisionBackendResult

        self._ensure_initialized()

        person_tracks: tuple[Tracklet, ...] = ()
        hand_regions = ()
        detections: tuple[DetectionBox, ...] = ()

        frame_rgb = self._bgr_to_rgb(packet.image)
        metadata: dict[str, Any] = {
            "backend": "paddledet_pphuman",
            "mot_enabled": self.config.mot.enabled,
            "keypoint_enabled": self.config.keypoint.enabled,
            "item_container_enabled": self.config.item_container.enabled,
        }

        if self._mot_predictor is not None:
            mot_result = self._mot_predictor.predict_image(
                [frame_rgb],
                visual=False,
                frame_count=packet.source_frame_id,
            )
            mot_payload = mot_result[0] if mot_result else ([], [], [])
            person_tracks = self.person_adapter.convert_mot_result(
                mot_payload,
                packet.frame,
                source="pphuman_mot",
            )
            metadata["person_track_count"] = len(person_tracks)

        if self._keypoint_predictor is not None and person_tracks:
            keypoint_payload = self._predict_keypoints(frame_rgb, packet, person_tracks)
            hand_regions = self.person_adapter.convert_keypoint_result(
                keypoint_payload,
                packet.frame,
                person_tracks,
            )
            metadata["hand_region_count"] = len(hand_regions)
        elif self.config.keypoint.enabled:
            metadata["hand_region_count"] = 0

        if self._item_detector is not None:
            det_result = self._item_detector.predict_image([frame_rgb], visual=False)
            detections = self.item_adapter.convert_detection_result(
                det_result,
                packet.frame,
                source="item_container_det",
            )
            metadata["detection_count"] = len(detections)
        elif self.config.item_container.enabled:
            metadata["detection_count"] = 0

        return VisionBackendResult(
            detections=detections,
            person_tracks=person_tracks,
            hand_regions=hand_regions,
            metadata=metadata,
        )

    def _ensure_initialized(self) -> None:
        self._prepare_import_paths()
        if self.config.mot.enabled and self._mot_predictor is None:
            self._mot_predictor = self._create_mot_predictor()
        if self.config.keypoint.enabled and self._keypoint_predictor is None:
            self._keypoint_predictor = self._create_keypoint_predictor()
        if self.config.item_container.enabled and self._item_detector is None:
            self._item_detector = self._create_item_detector()
        if self.config.keypoint.enabled and self._crop_image_with_mot is None:
            from pipe_utils import crop_image_with_mot
            from python.keypoint_postprocess import translate_to_ori_images

            self._crop_image_with_mot = crop_image_with_mot
            self._translate_to_ori_images = translate_to_ori_images

    def _prepare_import_paths(self) -> None:
        root = self.config.paddledetection_root.resolve()
        paths = [
            root,
            root / "deploy",
            root / "deploy/pipeline",
            root / "deploy/python",
            root / "deploy/pptracking/python",
        ]
        for path in reversed(paths):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

    def _create_mot_predictor(self) -> Any:
        try:
            import paddle
            from pptracking.python.mot_sde_infer import SDE_Detector
        except ImportError as exc:
            raise RuntimeError(
                f"PaddleDetection PP-Human MOT dependencies are not importable: {exc}"
            ) from exc

        paddle.enable_static()
        model_dir = _require_local_model_dir(self.config.mot.model_dir, "MOT")
        tracker_config = self.config.mot.tracker_config
        if tracker_config is None or not tracker_config.exists():
            raise RuntimeError(f"MOT tracker_config does not exist: {tracker_config}")
        return SDE_Detector(
            model_dir=model_dir,
            tracker_config=str(tracker_config),
            device=self.config.device.upper(),
            run_mode=self.config.run_mode,
            batch_size=self.config.mot.batch_size,
            trt_min_shape=self.config.trt_min_shape,
            trt_max_shape=self.config.trt_max_shape,
            trt_opt_shape=self.config.trt_opt_shape,
            trt_calib_mode=self.config.trt_calib_mode,
            cpu_threads=self.config.cpu_threads,
            enable_mkldnn=self.config.enable_mkldnn,
            output_dir=str(self.config.output_dir),
            threshold=self.config.mot.threshold,
            skip_frame_num=self.config.mot.skip_frame_num,
        )

    def _create_keypoint_predictor(self) -> Any:
        try:
            import paddle
            from python.keypoint_infer import KeyPointDetector
        except ImportError as exc:
            raise RuntimeError(
                f"PaddleDetection keypoint dependencies are not importable: {exc}"
            ) from exc

        paddle.enable_static()
        model_dir = _require_local_model_dir(self.config.keypoint.model_dir, "KeyPoint")
        return KeyPointDetector(
            model_dir=model_dir,
            device=self.config.device.upper(),
            run_mode=self.config.run_mode,
            batch_size=self.config.keypoint.batch_size,
            trt_min_shape=self.config.trt_min_shape,
            trt_max_shape=self.config.trt_max_shape,
            trt_opt_shape=self.config.trt_opt_shape,
            trt_calib_mode=self.config.trt_calib_mode,
            cpu_threads=self.config.cpu_threads,
            enable_mkldnn=self.config.enable_mkldnn,
            output_dir=str(self.config.output_dir),
            threshold=self.config.keypoint.threshold,
            use_dark=False,
        )

    def _create_item_detector(self) -> Any:
        try:
            import paddle
            from python.infer import Detector
        except ImportError as exc:
            raise RuntimeError(
                f"PaddleDetection detector dependencies are not importable: {exc}"
            ) from exc

        paddle.enable_static()
        model_dir = _require_local_model_dir(self.config.item_container.model_dir, "Item/container")
        return Detector(
            model_dir=model_dir,
            device=self.config.device.upper(),
            run_mode=self.config.run_mode,
            batch_size=self.config.item_container.batch_size,
            trt_min_shape=self.config.trt_min_shape,
            trt_max_shape=self.config.trt_max_shape,
            trt_opt_shape=self.config.trt_opt_shape,
            trt_calib_mode=self.config.trt_calib_mode,
            cpu_threads=self.config.cpu_threads,
            enable_mkldnn=self.config.enable_mkldnn,
            output_dir=str(self.config.output_dir),
            threshold=self.config.item_container.threshold,
        )

    def _predict_keypoints(
        self,
        frame_rgb: Any,
        packet: "FramePacket",
        person_tracks: tuple[Tracklet, ...],
    ) -> dict[str, Any]:
        if self._crop_image_with_mot is None or self._translate_to_ori_images is None:
            return {"keypoint": [[], []], "bbox": []}
        mot_rows = self._mot_rows_from_tracks(person_tracks)
        if not mot_rows:
            return {"keypoint": [[], []], "bbox": []}

        import numpy as np

        mot_res = {"boxes": np.array(mot_rows, dtype=np.float32)}
        crop_input, new_bboxes, ori_bboxes = self._crop_image_with_mot(frame_rgb, mot_res)
        if not crop_input:
            return {"keypoint": [[], []], "bbox": []}

        kpt_pred = self._keypoint_predictor.predict_image(crop_input, visual=False)
        keypoint_vector, score_vector = self._translate_to_ori_images(
            kpt_pred,
            np.array(new_bboxes),
        )
        return {
            "keypoint": [
                keypoint_vector.tolist(),
                score_vector.tolist(),
            ]
            if len(keypoint_vector) > 0
            else [[], []],
            "bbox": ori_bboxes,
            "metadata": {"source_frame_id": packet.source_frame_id},
        }

    @staticmethod
    def _mot_rows_from_tracks(person_tracks: tuple[Tracklet, ...]) -> list[list[float]]:
        rows: list[list[float]] = []
        for track in person_tracks:
            if not track.boxes:
                continue
            box = track.boxes[-1]
            raw_track_id = box.attributes.get("raw_track_id", track.track_id.replace("person-", ""))
            try:
                numeric_track_id = float(raw_track_id)
            except (TypeError, ValueError):
                numeric_track_id = float(len(rows) + 1)
            rows.append(
                [
                    numeric_track_id,
                    0.0,
                    box.score,
                    box.bbox[0],
                    box.bbox[1],
                    box.bbox[2],
                    box.bbox[3],
                ]
            )
        return rows

    @staticmethod
    def _bgr_to_rgb(image: Any) -> Any:
        if hasattr(image, "shape") and len(image.shape) >= 3:
            return image[:, :, ::-1]
        return image


__all__ = [
    "PaddleDetModuleConfig",
    "PaddleDetMOTConfig",
    "PaddleDetPPHumanBackend",
    "PaddleDetPPHumanBackendConfig",
]
