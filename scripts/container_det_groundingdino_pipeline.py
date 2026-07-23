"""GroundingDINO-assisted labeling pipeline for container detection.

The pipeline has two normal stages:

1. ``prelabel`` runs GroundingDINO over frame images and writes both raw JSONL
   candidates and Label Studio pre-annotation tasks.
2. ``export-coco`` converts reviewed Label Studio JSON, or trusted JSONL
   candidates, into the COCO layout consumed by the PaddleDetection RT-DETR
   container baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("datasets/container_det")
DEFAULT_LABEL_LIST = DEFAULT_DATASET_ROOT / "label_list.txt"
DEFAULT_WORK_DIR = DEFAULT_DATASET_ROOT / "groundingdino_work"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_STUDIO_IMAGE_PREFIX = "/data/local-files/?d="

DEFAULT_PROMPT_ALIASES: dict[str, tuple[str, ...]] = {
    "bag": (
        "bag",
        "shopping bag",
        "paper bag",
        "tote bag",
        "open bag",
        "soft bag",
    ),
    "backpack": (
        "backpack",
        "school bag",
        "sling backpack",
        "shoulder backpack",
    ),
    "handbag": (
        "handbag",
        "purse",
        "clutch",
        "small tote",
    ),
    "suitcase": (
        "suitcase",
        "luggage",
        "rolling luggage",
        "travel case",
    ),
    "basket": (
        "shopping basket",
        "retail basket",
        "handheld basket",
        "basket",
    ),
    "cart": (
        "shopping cart",
        "shopping trolley",
        "trolley",
        "cart",
    ),
    "plastic_bag": (
        "plastic bag",
        "produce bag",
        "transparent plastic bag",
    ),
    "stroller": (
        "stroller",
        "baby stroller",
        "pram",
    ),
    "helmet": (
        "helmet",
        "motorcycle helmet",
        "bicycle helmet",
        "hard shell helmet",
    ),
}

LABEL_COLORS = (
    "#e11d48",
    "#2563eb",
    "#16a34a",
    "#ca8a04",
    "#7c3aed",
    "#0891b2",
    "#ea580c",
    "#4f46e5",
    "#be123c",
)


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    task_image_path: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    score: float
    phrase: str
    bbox: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    source: str = "groundingdino"

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "score": round(float(self.score), 6),
            "phrase": self.phrase,
            "bbox": [round(value, 3) for value in self.bbox],
            "bbox_xyxy": [round(value, 3) for value in self.bbox_xyxy],
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DetectionCandidate":
        label = str(payload.get("label") or payload.get("category") or "")
        bbox = payload.get("bbox") or payload.get("bbox_xywh")
        if not _valid_box_list(bbox):
            raise ValueError(f"detection is missing bbox: {payload}")
        xywh = tuple(float(value) for value in bbox)  # type: ignore[arg-type]
        xyxy_payload = payload.get("bbox_xyxy")
        if _valid_box_list(xyxy_payload):
            xyxy = tuple(float(value) for value in xyxy_payload)  # type: ignore[arg-type]
        else:
            x, y, width, height = xywh
            xyxy = (x, y, x + width, y + height)
        return cls(
            label=label,
            score=float(payload.get("score", 1.0)),
            phrase=str(payload.get("phrase") or label),
            bbox=xywh,  # type: ignore[arg-type]
            bbox_xyxy=xyxy,  # type: ignore[arg-type]
            source=str(payload.get("source") or "manual"),
        )


@dataclass(frozen=True)
class PredictionRecord:
    image_path: str
    source_image: Path
    width: int
    height: int
    detections: tuple[DetectionCandidate, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "source_image": str(self.source_image),
            "width": self.width,
            "height": self.height,
            "detections": [detection.to_dict() for detection in self.detections],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PrelabelSummary:
    image_count: int
    detection_count: int
    skipped_unknown_phrase_count: int
    skipped_small_box_count: int
    predictions_jsonl: Path
    label_studio_tasks: Path
    label_studio_config: Path
    summary_json: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "image_count": self.image_count,
            "detection_count": self.detection_count,
            "skipped_unknown_phrase_count": self.skipped_unknown_phrase_count,
            "skipped_small_box_count": self.skipped_small_box_count,
            "predictions_jsonl": str(self.predictions_jsonl),
            "label_studio_tasks": str(self.label_studio_tasks),
            "label_studio_config": str(self.label_studio_config),
            "summary_json": str(self.summary_json),
        }


@dataclass(frozen=True)
class CocoExportSummary:
    input_record_count: int
    selected_record_count: int
    skipped_empty_record_count: int
    skipped_missing_image_count: int
    split_image_counts: dict[str, int]
    split_annotation_counts: dict[str, int]
    category_counts: dict[str, int]
    output_dir: Path
    summary_json: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "input_record_count": self.input_record_count,
            "selected_record_count": self.selected_record_count,
            "skipped_empty_record_count": self.skipped_empty_record_count,
            "skipped_missing_image_count": self.skipped_missing_image_count,
            "split_image_counts": self.split_image_counts,
            "split_annotation_counts": self.split_annotation_counts,
            "category_counts": self.category_counts,
            "output_dir": str(self.output_dir),
            "summary_json": str(self.summary_json),
        }


class GroundingDINORunner:
    """Small adapter around the official GroundingDINO inference helpers."""

    def __init__(
        self,
        *,
        config: Path,
        weights: Path,
        device: str,
        box_threshold: float,
        text_threshold: float,
        prompt: str,
    ) -> None:
        try:
            from groundingdino.util.inference import load_image, load_model, predict
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GroundingDINO is required for prelabeling. Install it from the "
                "official repository, then pass --config and --weights."
            ) from exc

        self._load_image = load_image
        self._predict = predict
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompt = prompt
        try:
            self.model = load_model(str(config), str(weights), device=device)
        except TypeError:
            self.model = load_model(str(config), str(weights))
            if hasattr(self.model, "to"):
                self.model.to(device)

    def predict_image(self, image_path: Path) -> tuple[Any, Any, Any]:
        _, image = self._load_image(str(image_path))
        try:
            return self._predict(
                model=self.model,
                image=image,
                caption=self.prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
        except TypeError:
            return self._predict(
                model=self.model,
                image=image,
                caption=self.prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )


def read_label_list(path: Path = DEFAULT_LABEL_LIST) -> tuple[str, ...]:
    path = _resolve_from_root(path)
    if not path.exists():
        raise FileNotFoundError(f"label list does not exist: {path}")
    labels = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    duplicates = [label for label in labels if labels.count(label) > 1]
    if duplicates:
        raise ValueError(f"duplicate labels in {path}: {sorted(set(duplicates))}")
    if not labels:
        raise ValueError(f"label list is empty: {path}")
    return labels


def build_prompt(labels: Sequence[str]) -> str:
    phrases: list[str] = []
    for label in labels:
        aliases = DEFAULT_PROMPT_ALIASES.get(label, ())
        for phrase in (*aliases, label.replace("_", " ")):
            normalized = _normalize_phrase(phrase)
            if normalized and normalized not in phrases:
                phrases.append(normalized)
    return " . ".join(phrases) + " ."


def canonical_label_from_phrase(
    phrase: str,
    labels: Sequence[str],
) -> str | None:
    normalized = _normalize_phrase(phrase)
    if not normalized:
        return None

    alias_entries: list[tuple[str, str]] = []
    for label in labels:
        aliases = DEFAULT_PROMPT_ALIASES.get(label, ())
        for alias in (*aliases, label, label.replace("_", " ")):
            alias_entries.append((_normalize_phrase(alias), label))
    alias_entries = sorted(set(alias_entries), key=lambda item: len(item[0]), reverse=True)

    exact = {alias: label for alias, label in alias_entries}
    if normalized in exact:
        return exact[normalized]

    padded = f" {normalized} "
    for alias, label in alias_entries:
        if f" {alias} " in padded:
            return label
    return None


def prelabel_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    config: Path,
    weights: Path,
    label_list: Path = DEFAULT_LABEL_LIST,
    image_root: Path | None = None,
    label_studio_root: Path = ROOT,
    label_studio_image_prefix: str = LABEL_STUDIO_IMAGE_PREFIX,
    prompt: str | None = None,
    box_threshold: float = 0.28,
    text_threshold: float = 0.22,
    min_area: float = 64.0,
    nms_iou: float = 0.65,
    device: str = "cuda",
    max_images: int | None = None,
    model_version: str = "groundingdino_container_v1",
    save_visualizations: bool = False,
    overwrite: bool = False,
    runner: GroundingDINORunner | None = None,
) -> PrelabelSummary:
    labels = read_label_list(label_list)
    prompt = prompt or build_prompt(labels)
    input_path = _resolve_from_root(input_path)
    output_dir = _resolve_from_root(output_dir)
    config = _resolve_from_root(config)
    weights = _resolve_from_root(weights)
    label_studio_root = _resolve_from_root(label_studio_root)
    samples = load_image_samples(
        input_path,
        image_root=image_root,
        label_studio_root=label_studio_root,
        max_images=max_images,
    )
    if not samples:
        raise FileNotFoundError(f"no images found for prelabeling: {input_path}")

    predictions_path = output_dir / "predictions.jsonl"
    tasks_path = output_dir / "label_studio_tasks.json"
    config_path = output_dir / "label_studio_config.xml"
    summary_path = output_dir / "summary.json"
    _ensure_can_write(
        (predictions_path, tasks_path, config_path, summary_path),
        overwrite=overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = runner or GroundingDINORunner(
        config=config,
        weights=weights,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        prompt=prompt,
    )

    records: list[PredictionRecord] = []
    skipped_unknown = 0
    skipped_small = 0
    for sample in _progress(samples, "GroundingDINO"):
        width, height = read_image_size(sample.image_path)
        boxes, logits, phrases = runner.predict_image(sample.image_path)
        detections, unknown_count, small_count = detections_from_groundingdino_outputs(
            boxes,
            logits,
            phrases,
            width=width,
            height=height,
            labels=labels,
            min_area=min_area,
            nms_iou=nms_iou,
        )
        skipped_unknown += unknown_count
        skipped_small += small_count
        records.append(
            PredictionRecord(
                image_path=sample.task_image_path,
                source_image=sample.image_path,
                width=width,
                height=height,
                detections=tuple(detections),
                metadata=sample.metadata,
            )
        )

    write_predictions_jsonl(predictions_path, records, overwrite=overwrite)
    write_label_studio_tasks(
        tasks_path,
        records,
        label_studio_image_prefix=label_studio_image_prefix,
        model_version=model_version,
        overwrite=overwrite,
    )
    write_label_studio_config(config_path, labels, overwrite=overwrite)
    if save_visualizations:
        visualization_dir = output_dir / "visualizations"
        for record in records:
            save_detection_visualization(
                record.source_image,
                visualization_dir / Path(record.source_image.name),
                record.detections,
                overwrite=overwrite,
            )

    summary = PrelabelSummary(
        image_count=len(records),
        detection_count=sum(len(record.detections) for record in records),
        skipped_unknown_phrase_count=skipped_unknown,
        skipped_small_box_count=skipped_small,
        predictions_jsonl=predictions_path,
        label_studio_tasks=tasks_path,
        label_studio_config=config_path,
        summary_json=summary_path,
    )
    write_json(summary_path, summary.to_dict(), overwrite=overwrite)
    return summary


def detections_from_groundingdino_outputs(
    boxes: Any,
    logits: Any,
    phrases: Any,
    *,
    width: int,
    height: int,
    labels: Sequence[str],
    min_area: float,
    nms_iou: float,
) -> tuple[list[DetectionCandidate], int, int]:
    detections: list[DetectionCandidate] = []
    skipped_unknown = 0
    skipped_small = 0
    for box, score, phrase in zip(_as_sequence(boxes), _as_sequence(logits), _as_sequence(phrases)):
        phrase_text = str(phrase)
        label = canonical_label_from_phrase(phrase_text, labels)
        if label is None:
            skipped_unknown += 1
            continue
        xywh, xyxy = normalized_cxcywh_to_xywh(box, image_width=width, image_height=height)
        if xywh[2] * xywh[3] < min_area:
            skipped_small += 1
            continue
        detections.append(
            DetectionCandidate(
                label=label,
                score=_score_to_float(score),
                phrase=phrase_text,
                bbox=xywh,
                bbox_xyxy=xyxy,
            )
        )
    return nms_candidates(detections, iou_threshold=nms_iou), skipped_unknown, skipped_small


def normalized_cxcywh_to_xywh(
    box: Any,
    *,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    values = _float_list(box)
    if len(values) != 4:
        raise ValueError(f"expected a 4-value box, got: {values}")
    cx, cy, width, height = values
    x1 = (cx - width / 2.0) * image_width
    y1 = (cy - height / 2.0) * image_height
    x2 = (cx + width / 2.0) * image_width
    y2 = (cy + height / 2.0) * image_height
    clipped = clip_xyxy((x1, y1, x2, y2), image_width=image_width, image_height=image_height)
    x1, y1, x2, y2 = clipped
    return (x1, y1, x2 - x1, y2 - y1), clipped


def clip_xyxy(
    box: Sequence[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0.0, min(float(image_width), x1))
    top = max(0.0, min(float(image_height), y1))
    right = max(0.0, min(float(image_width), x2))
    bottom = max(0.0, min(float(image_height), y2))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid box after clipping: {box}")
    return left, top, right, bottom


def nms_candidates(
    detections: Sequence[DetectionCandidate],
    *,
    iou_threshold: float,
) -> list[DetectionCandidate]:
    if iou_threshold >= 1.0:
        return list(detections)
    selected: list[DetectionCandidate] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        duplicate = any(
            candidate.label == existing.label
            and bbox_iou(candidate.bbox_xyxy, existing.bbox_xyxy) >= iou_threshold
            for existing in selected
        )
        if not duplicate:
            selected.append(candidate)
    return selected


def bbox_iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - inter_area
    return inter_area / union if union > 0 else 0.0


def load_image_samples(
    input_path: Path,
    *,
    image_root: Path | None = None,
    label_studio_root: Path = ROOT,
    max_images: int | None = None,
) -> list[ImageSample]:
    input_path = _resolve_from_root(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        samples = load_manifest_samples(
            input_path,
            image_root=image_root,
            label_studio_root=label_studio_root,
        )
    else:
        samples = load_image_file_samples(
            input_path,
            label_studio_root=label_studio_root,
        )
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        samples = samples[:max_images]
    return samples


def load_manifest_samples(
    manifest_path: Path,
    *,
    image_root: Path | None = None,
    label_studio_root: Path = ROOT,
) -> list[ImageSample]:
    manifest_path = _resolve_from_root(manifest_path)
    source_image_root = _resolve_manifest_image_root(manifest_path, image_root)
    samples: list[ImageSample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=2):
            raw_image_path = row.get("source_image") or row.get("image_path")
            if not raw_image_path:
                raise ValueError(f"manifest row {row_number} is missing image_path")
            image_path = _resolve_source_image_path(raw_image_path, source_image_root)
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not image_path.exists():
                raise FileNotFoundError(f"image does not exist for row {row_number}: {image_path}")
            samples.append(
                ImageSample(
                    image_path=image_path,
                    task_image_path=task_relative_path(image_path, label_studio_root),
                    metadata={key: str(value) for key, value in row.items() if _has_value(value)},
                )
            )
    return samples


def load_image_file_samples(
    input_path: Path,
    *,
    label_studio_root: Path = ROOT,
) -> list[ImageSample]:
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {input_path}")
        images = [input_path]
    else:
        images = [
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    return [
        ImageSample(
            image_path=image_path,
            task_image_path=task_relative_path(image_path, label_studio_root),
        )
        for image_path in sorted(images)
    ]


def write_predictions_jsonl(
    path: Path,
    records: Sequence[PredictionRecord],
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write((path,), overwrite=overwrite)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_label_studio_tasks(
    path: Path,
    records: Sequence[PredictionRecord],
    *,
    label_studio_image_prefix: str = LABEL_STUDIO_IMAGE_PREFIX,
    model_version: str = "groundingdino_container_v1",
    overwrite: bool,
) -> None:
    _ensure_can_write((path,), overwrite=overwrite)
    tasks = [
        label_studio_task(
            record,
            label_studio_image_prefix=label_studio_image_prefix,
            model_version=model_version,
        )
        for record in records
    ]
    write_json(path, tasks, overwrite=overwrite)


def label_studio_task(
    record: PredictionRecord,
    *,
    label_studio_image_prefix: str = LABEL_STUDIO_IMAGE_PREFIX,
    model_version: str = "groundingdino_container_v1",
) -> dict[str, object]:
    results = [
        label_studio_rectangle_result(
            record,
            detection,
            index=index,
        )
        for index, detection in enumerate(record.detections)
    ]
    score = max((detection.score for detection in record.detections), default=0.0)
    data = {
        "image": label_studio_image_prefix + quote(record.image_path, safe="/:"),
        "source_image": str(record.source_image),
        "width": record.width,
        "height": record.height,
        **record.metadata,
    }
    return {
        "data": data,
        "predictions": [
            {
                "model_version": model_version,
                "score": round(float(score), 6),
                "result": results,
            }
        ],
    }


def label_studio_rectangle_result(
    record: PredictionRecord,
    detection: DetectionCandidate,
    *,
    index: int,
) -> dict[str, object]:
    x, y, width, height = detection.bbox
    result_id = hashlib.sha1(
        f"{record.image_path}:{index}:{detection.label}:{detection.bbox}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": record.width,
        "original_height": record.height,
        "image_rotation": 0,
        "score": round(float(detection.score), 6),
        "value": {
            "rotation": 0,
            "x": _percent(x, record.width),
            "y": _percent(y, record.height),
            "width": _percent(width, record.width),
            "height": _percent(height, record.height),
            "rectanglelabels": [detection.label],
        },
    }


def write_label_studio_config(
    path: Path,
    labels: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write((path,), overwrite=overwrite)
    label_tags = "\n".join(
        f'    <Label value="{label}" background="{LABEL_COLORS[index % len(LABEL_COLORS)]}"/>'
        for index, label in enumerate(labels)
    )
    body = (
        '<View>\n'
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{label_tags}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def save_detection_visualization(
    input_image: Path,
    output_image: Path,
    detections: Sequence[DetectionCandidate],
    *,
    overwrite: bool,
) -> None:
    if output_image.exists() and not overwrite:
        raise FileExistsError(f"visualization already exists: {output_image}")
    from PIL import Image, ImageDraw

    with Image.open(input_image) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        color = LABEL_COLORS[abs(hash(detection.label)) % len(LABEL_COLORS)]
        x1, y1, x2, y2 = detection.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{detection.label} {detection.score:.2f}"
        text_box = draw.textbbox((x1, y1), text)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), text, fill="white")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image)


def export_coco_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    input_format: str = "auto",
    label_list: Path = DEFAULT_LABEL_LIST,
    image_root: Path | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 2026,
    min_score: float = 0.0,
    include_empty: bool = True,
    use_predictions_when_unreviewed: bool = False,
    overwrite: bool = False,
    missing_image_policy: str = "error",
) -> CocoExportSummary:
    labels = read_label_list(label_list)
    records = load_labeled_records(
        input_path,
        input_format=input_format,
        labels=labels,
        image_root=image_root,
        min_score=min_score,
        use_predictions_when_unreviewed=use_predictions_when_unreviewed,
    )
    return materialize_coco_dataset(
        records,
        output_dir,
        labels=labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        include_empty=include_empty,
        overwrite=overwrite,
        missing_image_policy=missing_image_policy,
    )


def load_labeled_records(
    input_path: Path,
    *,
    input_format: str,
    labels: Sequence[str],
    image_root: Path | None,
    min_score: float,
    use_predictions_when_unreviewed: bool,
) -> list[PredictionRecord]:
    input_path = _resolve_from_root(input_path)
    resolved_format = detect_input_format(input_path, input_format)
    if resolved_format == "predictions-jsonl":
        return load_prediction_records(
            input_path,
            labels=labels,
            image_root=image_root,
            min_score=min_score,
        )
    if resolved_format == "label-studio-json":
        return load_label_studio_records(
            input_path,
            labels=labels,
            image_root=image_root,
            min_score=min_score,
            use_predictions_when_unreviewed=use_predictions_when_unreviewed,
        )
    raise ValueError(f"unsupported input format: {input_format}")


def detect_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if path.suffix.lower() == ".jsonl":
        return "predictions-jsonl"
    if path.suffix.lower() == ".json":
        return "label-studio-json"
    raise ValueError(f"cannot infer input format from suffix: {path}")


def load_prediction_records(
    path: Path,
    *,
    labels: Sequence[str],
    image_root: Path | None,
    min_score: float,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object at line {line_number}")
            source_image = resolve_label_studio_or_local_path(
                str(payload.get("source_image") or payload.get("image_path") or ""),
                image_root=image_root,
            )
            width = int(payload.get("width") or 0)
            height = int(payload.get("height") or 0)
            if (width <= 0 or height <= 0) and source_image.exists():
                width, height = read_image_size(source_image)
            detections = tuple(
                detection
                for detection in (
                    DetectionCandidate.from_mapping(item)
                    for item in payload.get("detections") or []
                    if isinstance(item, Mapping)
                )
                if detection.label in labels and detection.score >= min_score
            )
            records.append(
                PredictionRecord(
                    image_path=str(payload.get("image_path") or task_relative_path(source_image, ROOT)),
                    source_image=source_image,
                    width=width,
                    height=height,
                    detections=detections,
                    metadata=_string_mapping(payload.get("metadata")),
                )
            )
    return records


def load_label_studio_records(
    path: Path,
    *,
    labels: Sequence[str],
    image_root: Path | None,
    min_score: float,
    use_predictions_when_unreviewed: bool,
) -> list[PredictionRecord]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else payload
    if not isinstance(tasks, list):
        raise ValueError(f"Label Studio export must be a list or object with tasks: {path}")

    records: list[PredictionRecord] = []
    for task_index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        data = _string_mapping(task.get("data"))
        source_value = data.get("source_image") or data.get("image") or data.get("image_path")
        source_image = resolve_label_studio_or_local_path(source_value, image_root=image_root)
        result_items, source = _reviewed_label_studio_results(
            task,
            use_predictions_when_unreviewed=use_predictions_when_unreviewed,
        )
        if result_items is None:
            continue
        width, height = _dimensions_from_results(result_items)
        if width <= 0 or height <= 0:
            width = int(float(data.get("width") or 0))
            height = int(float(data.get("height") or 0))
        if (width <= 0 or height <= 0) and source_image.exists():
            width, height = read_image_size(source_image)
        detections = tuple(
            detection
            for detection in (
                detection_from_label_studio_result(
                    item,
                    image_width=width,
                    image_height=height,
                    source=source,
                )
                for item in result_items
                if isinstance(item, Mapping)
            )
            if detection is not None and detection.label in labels and detection.score >= min_score
        )
        records.append(
            PredictionRecord(
                image_path=data.get("image_path") or data.get("image") or f"task_{task_index:06d}",
                source_image=source_image,
                width=width,
                height=height,
                detections=detections,
                metadata=data,
            )
        )
    return records


def _reviewed_label_studio_results(
    task: Mapping[str, Any],
    *,
    use_predictions_when_unreviewed: bool,
) -> tuple[list[Any] | None, str]:
    annotations = [
        annotation
        for annotation in task.get("annotations") or []
        if isinstance(annotation, Mapping) and not annotation.get("was_cancelled")
    ]
    if annotations:
        return list(annotations[-1].get("result") or []), "label_studio_annotation"
    if use_predictions_when_unreviewed:
        predictions = [item for item in task.get("predictions") or [] if isinstance(item, Mapping)]
        if predictions:
            return list(predictions[-1].get("result") or []), "label_studio_prediction"
    return None, "unreviewed"


def detection_from_label_studio_result(
    result: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    source: str,
) -> DetectionCandidate | None:
    if result.get("type") != "rectanglelabels":
        return None
    value = result.get("value")
    if not isinstance(value, Mapping):
        return None
    labels = value.get("rectanglelabels") or value.get("labels")
    if not labels:
        nested = value.get("values")
        if isinstance(nested, Mapping):
            labels = nested.get("rectanglelabels") or nested.get("labels")
    if not isinstance(labels, list) or not labels:
        return None
    label = str(labels[0])
    width = float(value.get("width", 0.0)) * image_width / 100.0
    height = float(value.get("height", 0.0)) * image_height / 100.0
    x = float(value.get("x", 0.0)) * image_width / 100.0
    y = float(value.get("y", 0.0)) * image_height / 100.0
    if width <= 0 or height <= 0:
        return None
    xyxy = clip_xyxy((x, y, x + width, y + height), image_width=image_width, image_height=image_height)
    x1, y1, x2, y2 = xyxy
    return DetectionCandidate(
        label=label,
        score=_label_studio_score(result.get("score")),
        phrase=label,
        bbox=(x1, y1, x2 - x1, y2 - y1),
        bbox_xyxy=xyxy,
        source=source,
    )


def materialize_coco_dataset(
    records: Sequence[PredictionRecord],
    output_dir: Path,
    *,
    labels: Sequence[str],
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 2026,
    include_empty: bool = True,
    overwrite: bool = False,
    missing_image_policy: str = "error",
) -> CocoExportSummary:
    if missing_image_policy not in {"error", "skip"}:
        raise ValueError("missing_image_policy must be 'error' or 'skip'")
    if not 0.0 <= val_ratio <= 1.0 or not 0.0 <= test_ratio <= 1.0:
        raise ValueError("split ratios must be between 0 and 1")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be less than 1")

    output_dir = _resolve_from_root(output_dir)
    selected: list[PredictionRecord] = []
    skipped_empty = 0
    skipped_missing = 0
    for record in records:
        if not include_empty and not record.detections:
            skipped_empty += 1
            continue
        if not record.source_image.exists():
            if missing_image_policy == "skip":
                skipped_missing += 1
                continue
            raise FileNotFoundError(f"source image does not exist: {record.source_image}")
        selected.append(record)
    if not selected:
        raise ValueError("no labeled records are available for COCO export")

    assignments = split_records(selected, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    annotation_dir = output_dir / "annotations"
    image_root = output_dir / "images"
    planned_jsons = tuple(annotation_dir / f"instances_{split}.json" for split in ("train", "val", "test"))
    _ensure_can_write(planned_jsons, overwrite=overwrite)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (image_root / split).mkdir(parents=True, exist_ok=True)

    category_counts = {label: 0 for label in labels}
    split_image_counts: dict[str, int] = {}
    split_annotation_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_records_for_coco = assignments.get(split, [])
        coco = build_coco_for_split(
            split_records_for_coco,
            split=split,
            output_image_root=image_root,
            labels=labels,
            overwrite=overwrite,
        )
        split_image_counts[split] = len(coco["images"])
        split_annotation_counts[split] = len(coco["annotations"])
        for annotation in coco["annotations"]:
            label = labels[int(annotation["category_id"])]
            category_counts[label] += 1
        write_json(annotation_dir / f"instances_{split}.json", coco, overwrite=overwrite)

    summary = CocoExportSummary(
        input_record_count=len(records),
        selected_record_count=len(selected),
        skipped_empty_record_count=skipped_empty,
        skipped_missing_image_count=skipped_missing,
        split_image_counts=split_image_counts,
        split_annotation_counts=split_annotation_counts,
        category_counts={key: value for key, value in category_counts.items() if value > 0},
        output_dir=output_dir,
        summary_json=annotation_dir / "export_summary.json",
    )
    write_json(summary.summary_json, summary.to_dict(), overwrite=True)
    return summary


def split_records(
    records: Sequence[PredictionRecord],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[PredictionRecord]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    test_count, val_count = _split_counts(len(records), val_ratio=val_ratio, test_ratio=test_ratio)
    test_indices = set(indices[:test_count])
    val_indices = set(indices[test_count : test_count + val_count])
    return {
        "train": [record for index, record in enumerate(records) if index not in test_indices | val_indices],
        "val": [record for index, record in enumerate(records) if index in val_indices],
        "test": [record for index, record in enumerate(records) if index in test_indices],
    }


def build_coco_for_split(
    records: Sequence[PredictionRecord],
    *,
    split: str,
    output_image_root: Path,
    labels: Sequence[str],
    overwrite: bool,
) -> dict[str, object]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    used_filenames: set[str] = set()
    annotation_id = 1
    for image_id, record in enumerate(records):
        file_name = unique_output_filename(record.source_image, used_filenames)
        destination = output_image_root / split / file_name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"output image already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.source_image, destination)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(record.width),
                "height": int(record.height),
                "source_image": str(record.source_image),
            }
        )
        for detection in record.detections:
            if detection.label not in label_to_id:
                continue
            x, y, width, height = detection.bbox
            if width <= 0 or height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_id[detection.label],
                    "bbox": [round(x, 3), round(y, 3), round(width, 3), round(height, 3)],
                    "area": round(width * height, 3),
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": label, "supercategory": "container"}
            for index, label in enumerate(labels)
        ],
    }


def unique_output_filename(source_image: Path, used_filenames: set[str]) -> str:
    suffix = source_image.suffix.lower() or ".jpg"
    stem = safe_stem(source_image.stem)
    candidate = f"{stem}{suffix}"
    key = candidate.lower()
    if key not in used_filenames:
        used_filenames.add(key)
        return candidate
    digest = hashlib.sha1(str(source_image).encode("utf-8")).hexdigest()[:8]
    candidate = f"{stem}_{digest}{suffix}"
    key = candidate.lower()
    counter = 1
    while key in used_filenames:
        candidate = f"{stem}_{digest}_{counter:02d}{suffix}"
        key = candidate.lower()
        counter += 1
    used_filenames.add(key)
    return candidate


def resolve_label_studio_or_local_path(
    value: str | None,
    *,
    image_root: Path | None,
) -> Path:
    if not value:
        raise ValueError("image path is empty")
    raw_value = unquote(str(value))
    parsed = urlparse(raw_value)
    if parsed.scheme in {"http", "https"}:
        raise ValueError(f"cannot resolve remote image URL without source_image metadata: {value}")
    if parsed.scheme == "file":
        file_path = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:/", file_path):
            file_path = file_path[1:]
        return Path(file_path)
    if raw_value.startswith("/data/local-files/"):
        query = parse_qs(parsed.query)
        values = query.get("d") or query.get("path")
        if not values:
            raise ValueError(f"Label Studio local-files URL is missing d=: {value}")
        raw_value = unquote(values[0])
    path = Path(raw_value)
    if path.is_absolute():
        return path
    root = _resolve_from_root(image_root) if image_root is not None else ROOT
    _validate_relative_parts(raw_value)
    return root / path


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def write_json(
    path: Path,
    payload: object,
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write((path,), overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_relative_path(path: Path, label_studio_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(label_studio_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "image"


def _resolve_manifest_image_root(manifest_path: Path, image_root: Path | None) -> Path:
    if image_root is not None:
        return _resolve_from_root(image_root)
    candidate = manifest_path.parent / "images"
    return candidate if candidate.exists() else manifest_path.parent


def _resolve_source_image_path(raw_image_path: str, image_root: Path) -> Path:
    path = Path(raw_image_path)
    if path.is_absolute():
        return path
    _validate_relative_parts(raw_image_path)
    return image_root / path


def _resolve_from_root(path: Path | None) -> Path:
    if path is None:
        return ROOT
    return path if path.is_absolute() else ROOT / path


def _ensure_can_write(paths: Iterable[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"output file already exists: {path}; use --overwrite")


def _normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _percent(value: float, total: int) -> float:
    if total <= 0:
        raise ValueError("image dimension must be positive")
    return round(max(0.0, min(100.0, value * 100.0 / total)), 4)


def _as_sequence(value: Any) -> Sequence[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return [value]


def _float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected a list-like value, got: {value}")
    return [float(item) for item in value]


def _score_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return float(max(value))
    return float(value)


def _valid_box_list(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return True


def _validate_relative_parts(raw_path: str) -> None:
    normalized = raw_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"relative image path must not contain '..': {raw_path}")


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if _has_value(item)}


def _has_value(value: object) -> bool:
    return value is not None and value != ""


def _dimensions_from_results(result_items: Sequence[Any]) -> tuple[int, int]:
    for item in result_items:
        if not isinstance(item, Mapping):
            continue
        width = int(item.get("original_width") or 0)
        height = int(item.get("original_height") or 0)
        if width > 0 and height > 0:
            return width, height
    return 0, 0


def _label_studio_score(value: object) -> float:
    if value is None or value == "":
        return 1.0
    return float(value)


def _split_counts(total: int, *, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if total <= 1:
        return 0, 0
    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))
    if test_ratio > 0 and test_count == 0:
        test_count = 1
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    max_holdout = total - 1
    while test_count + val_count > max_holdout:
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break
    return test_count, val_count


def _progress(items: Sequence[ImageSample], description: str) -> Iterable[ImageSample]:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return items
    return tqdm(items, desc=description)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prelabel = subparsers.add_parser("prelabel", help="Run GroundingDINO and create Label Studio tasks.")
    prelabel.add_argument("--input", required=True, type=Path, help="Image directory, image file, or CSV manifest.")
    prelabel.add_argument("--output", type=Path, default=DEFAULT_WORK_DIR, help="Output work directory.")
    prelabel.add_argument("--label-list", type=Path, default=DEFAULT_LABEL_LIST)
    prelabel.add_argument("--image-root", type=Path, default=None, help="Image root for CSV image_path values.")
    prelabel.add_argument("--label-studio-root", type=Path, default=ROOT, help="Root served by Label Studio local files.")
    prelabel.add_argument("--label-studio-image-prefix", default=LABEL_STUDIO_IMAGE_PREFIX)
    prelabel.add_argument("--config", required=True, type=Path, help="GroundingDINO config path.")
    prelabel.add_argument("--weights", required=True, type=Path, help="GroundingDINO checkpoint path.")
    prelabel.add_argument("--device", default="cuda", help="GroundingDINO device, e.g. cuda or cpu.")
    prelabel.add_argument("--box-threshold", type=float, default=0.28)
    prelabel.add_argument("--text-threshold", type=float, default=0.22)
    prelabel.add_argument("--min-area", type=float, default=64.0)
    prelabel.add_argument("--nms-iou", type=float, default=0.65)
    prelabel.add_argument("--prompt", default=None, help="Override the generated GroundingDINO text prompt.")
    prelabel.add_argument("--max-images", type=int, default=None)
    prelabel.add_argument("--model-version", default="groundingdino_container_v1")
    prelabel.add_argument("--save-visualizations", action="store_true")
    prelabel.add_argument("--overwrite", action="store_true")

    export = subparsers.add_parser("export-coco", help="Export reviewed labels to PaddleDetection COCO layout.")
    export.add_argument("--input", required=True, type=Path, help="Label Studio JSON export or predictions.jsonl.")
    export.add_argument("--input-format", default="auto", choices=("auto", "label-studio-json", "predictions-jsonl"))
    export.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT, help="COCO dataset output root.")
    export.add_argument("--label-list", type=Path, default=DEFAULT_LABEL_LIST)
    export.add_argument("--image-root", type=Path, default=None, help="Root for relative local image paths.")
    export.add_argument("--val-ratio", type=float, default=0.2)
    export.add_argument("--test-ratio", type=float, default=0.0)
    export.add_argument("--seed", type=int, default=2026)
    export.add_argument("--min-score", type=float, default=0.0)
    export.add_argument("--drop-empty", action="store_true", help="Drop images with no boxes.")
    export.add_argument("--use-predictions-when-unreviewed", action="store_true")
    export.add_argument("--skip-missing-images", action="store_true")
    export.add_argument("--overwrite", action="store_true")

    config = subparsers.add_parser("write-label-studio-config", help="Write Label Studio XML config.")
    config.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT / "label_studio_config.xml")
    config.add_argument("--label-list", type=Path, default=DEFAULT_LABEL_LIST)
    config.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prelabel":
        summary = prelabel_dataset(
            args.input,
            args.output,
            config=args.config,
            weights=args.weights,
            label_list=args.label_list,
            image_root=args.image_root,
            label_studio_root=args.label_studio_root,
            label_studio_image_prefix=args.label_studio_image_prefix,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_area=args.min_area,
            nms_iou=args.nms_iou,
            device=args.device,
            max_images=args.max_images,
            model_version=args.model_version,
            save_visualizations=args.save_visualizations,
            overwrite=args.overwrite,
        )
    elif args.command == "export-coco":
        summary = export_coco_dataset(
            args.input,
            args.output,
            input_format=args.input_format,
            label_list=args.label_list,
            image_root=args.image_root,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_score=args.min_score,
            include_empty=not args.drop_empty,
            use_predictions_when_unreviewed=args.use_predictions_when_unreviewed,
            overwrite=args.overwrite,
            missing_image_policy="skip" if args.skip_missing_images else "error",
        )
    elif args.command == "write-label-studio-config":
        labels = read_label_list(args.label_list)
        output = _resolve_from_root(args.output)
        write_label_studio_config(output, labels, overwrite=args.overwrite)
        summary = {"label_studio_config": str(output), "label_count": len(labels)}
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(summary.to_dict() if hasattr(summary, "to_dict") else summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
