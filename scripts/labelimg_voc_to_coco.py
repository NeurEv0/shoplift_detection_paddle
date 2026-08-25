"""Convert labelImg Pascal VOC XML annotations to PaddleDetection COCO JSON.

labelImg (PascalVOC format) produces one XML per image under an ``Annotations``
directory. This script parses those XMLs and writes the COCO layout consumed by
the container RT-DETR baseline:

    output/
      annotations/instances_train.json
      annotations/instances_val.json
      annotations/instances_test.json
      annotations/export_summary.json
      images/{train,val,test}/<copied images>

Category ids follow the order of ``label_list.txt`` (id 0 = first line), which
must match the labelImg ``classes.txt`` order. New classes must be appended at
the end of the label list to keep existing category ids stable.

Difficult boxes (labelImg ``difficult``) are filtered by default so low-quality
boxes do not pollute training; use ``--keep-difficult`` to keep them with
``ignore=1``.

Example:

    python scripts/labelimg_voc_to_coco.py ^
      --xml-dir datasets/container_det/manual_work/xml ^
      --image-dir datasets/container_det/manual_work/jpg ^
      --label-list datasets/container_det/label_list.txt ^
      --output datasets/container_det ^
      --val-ratio 0.2 --include-empty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_LIST = ROOT / "datasets" / "container_det" / "label_list.txt"
DEFAULT_OUTPUT = ROOT / "datasets" / "container_det"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class VocObject:
    """One labeled object inside a Pascal VOC annotation."""

    name: str
    bbox: tuple[float, float, float, float]  # xyxy pixel coordinates
    difficult: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class VocAnnotation:
    """Parsed Pascal VOC annotation for one image."""

    xml_path: Path
    image_path: Path
    width: int
    height: int
    objects: tuple[VocObject, ...] = field(default_factory=tuple)
    difficult_seen: int = 0


@dataclass(frozen=True)
class CocoExportSummary:
    """Summary of a single COCO export run."""

    xml_count: int
    selected_record_count: int
    skipped_empty_record_count: int
    skipped_missing_image_count: int
    split_image_counts: dict[str, int]
    split_annotation_counts: dict[str, int]
    category_counts: dict[str, int]
    difficult_box_count: int
    output_dir: Path
    summary_json: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "xml_count": self.xml_count,
            "selected_record_count": self.selected_record_count,
            "skipped_empty_record_count": self.skipped_empty_record_count,
            "skipped_missing_image_count": self.skipped_missing_image_count,
            "split_image_counts": self.split_image_counts,
            "split_annotation_counts": self.split_annotation_counts,
            "category_counts": self.category_counts,
            "difficult_box_count": self.difficult_box_count,
            "output_dir": str(self.output_dir),
            "summary_json": str(self.summary_json),
        }


def read_label_list(path: Path) -> tuple[str, ...]:
    path = _resolve_from_root(path)
    if not path.exists():
        raise FileNotFoundError(f"label list does not exist: {path}")
    # utf-8-sig tolerates the UTF-8 BOM that Windows editors (Notepad/PowerShell)
    # prepend to text files; without it the first label would parse as '\ufeffbag'.
    labels = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    duplicates = [label for label in labels if labels.count(label) > 1]
    if duplicates:
        raise ValueError(f"duplicate labels in {path}: {sorted(set(duplicates))}")
    if not labels:
        raise ValueError(f"label list is empty: {path}")
    return labels


def collect_voc_xmls(xml_dir: Path) -> list[Path]:
    if not xml_dir.exists():
        raise FileNotFoundError(f"xml directory does not exist: {xml_dir}")
    xmls = sorted(
        path
        for path in xml_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".xml"
    )
    if not xmls:
        raise ValueError(f"no Pascal VOC XML files found in: {xml_dir}")
    return xmls


def parse_voc_annotation(
    xml_path: Path,
    *,
    image_dir: Path,
    labels: Sequence[str],
    keep_difficult: bool,
) -> VocAnnotation:
    """Parse one labelImg XML into a VocAnnotation (errors are raised with file context)."""

    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"malformed VOC XML {xml_path}: {exc}") from exc

    filename = _text(root, "filename")
    if not filename:
        raise ValueError(f"{xml_path}: <filename> is missing")

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"{xml_path}: <size> is missing")
    width = _int_text(size_node, "width", xml_path)
    height = _int_text(size_node, "height", xml_path)
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path}: invalid image size {width}x{height}")

    image_path = image_dir / filename
    if not image_path.exists():
        raise FileNotFoundError(
            f"{xml_path}: referenced image not found: {image_path}"
        )

    label_set = set(labels)
    objects: list[VocObject] = []
    difficult_seen = 0
    for index, obj_node in enumerate(root.findall("object")):
        name = _text(obj_node, "name")
        if not name:
            raise ValueError(f"{xml_path}: object[{index}] has no <name>")
        if name not in label_set:
            raise ValueError(
                f"{xml_path}: object name {name!r} is not in label list {labels}"
            )
        bndbox = obj_node.find("bndbox")
        if bndbox is None:
            raise ValueError(f"{xml_path}: object {name!r} has no <bndbox>")
        xmin = _float_text(bndbox, "xmin", xml_path)
        ymin = _float_text(bndbox, "ymin", xml_path)
        xmax = _float_text(bndbox, "xmax", xml_path)
        ymax = _float_text(bndbox, "ymax", xml_path)
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"{xml_path}: object {name!r} has invalid bbox")
        x1, y1, x2, y2 = _clip_bbox((xmin, ymin, xmax, ymax), width, height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"{xml_path}: object {name!r} bbox is fully outside the image"
            )
        difficult = _bool_text(obj_node, "difficult")
        truncated = _bool_text(obj_node, "truncated")
        if difficult:
            difficult_seen += 1
        if difficult and not keep_difficult:
            continue
        objects.append(
            VocObject(
                name=name,
                bbox=(x1, y1, x2, y2),
                difficult=difficult,
                truncated=truncated,
            )
        )

    return VocAnnotation(
        xml_path=xml_path,
        image_path=image_path,
        width=width,
        height=height,
        objects=tuple(objects),
        difficult_seen=difficult_seen,
    )


def load_annotations(
    xml_dir: Path,
    *,
    image_dir: Path,
    labels: Sequence[str],
    keep_difficult: bool,
    missing_image_policy: str,
) -> tuple[list[VocAnnotation], int, int]:
    """Parse all XMLs; return (records, skipped_missing_image_count, difficult_box_count)."""

    if missing_image_policy not in {"error", "skip"}:
        raise ValueError("missing_image_policy must be 'error' or 'skip'")
    records: list[VocAnnotation] = []
    skipped_missing = 0
    difficult_count = 0
    for xml_path in collect_voc_xmls(xml_dir):
        try:
            annotation = parse_voc_annotation(
                xml_path,
                image_dir=image_dir,
                labels=labels,
                keep_difficult=keep_difficult,
            )
        except FileNotFoundError:
            if missing_image_policy == "skip":
                skipped_missing += 1
                continue
            raise
        difficult_count += annotation.difficult_seen
        records.append(annotation)
    if not records:
        raise ValueError("no usable VOC annotations found")
    return records, skipped_missing, difficult_count


def split_annotations(
    records: Sequence[VocAnnotation],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[VocAnnotation]]:
    if not 0.0 <= val_ratio <= 1.0 or not 0.0 <= test_ratio <= 1.0:
        raise ValueError("split ratios must be between 0 and 1")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

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


def build_coco_split(
    records: Sequence[VocAnnotation],
    *,
    split: str,
    output_image_root: Path,
    labels: Sequence[str],
    overwrite: bool,
) -> dict[str, Any]:
    """Copy images and build the COCO dict for one split."""

    label_to_id = {label: index for index, label in enumerate(labels)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    annotation_id = 1

    for image_id, record in enumerate(records):
        file_name = _unique_output_filename(record.image_path, used_filenames)
        destination = output_image_root / split / file_name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"output image already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image_path, destination)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(record.width),
                "height": int(record.height),
                "source_image": str(record.image_path),
                "source_xml": str(record.xml_path),
            }
        )
        for obj in record.objects:
            x1, y1, x2, y2 = obj.bbox
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_id[obj.name],
                    "bbox": [round(x1, 3), round(y1, 3), round(width, 3), round(height, 3)],
                    "area": round(width * height, 3),
                    "iscrowd": 0,
                    "ignore": 1 if obj.difficult else 0,
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


def materialize_coco(
    records: Sequence[VocAnnotation],
    output_dir: Path,
    *,
    labels: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_empty: bool,
    overwrite: bool,
    skipped_missing: int,
    difficult_box_count: int,
) -> CocoExportSummary:
    if not records:
        raise ValueError("no labeled records are available for COCO export")

    selected: list[VocAnnotation] = []
    skipped_empty = 0
    for record in records:
        if not include_empty and not record.objects:
            skipped_empty += 1
            continue
        selected.append(record)
    if not selected:
        raise ValueError("no records with objects remain after filtering")

    assignments = split_annotations(
        selected, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    annotation_dir = output_dir / "annotations"
    image_root = output_dir / "images"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    category_counts = {label: 0 for label in labels}
    split_image_counts: dict[str, int] = {}
    split_annotation_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_records = assignments.get(split, [])
        coco = build_coco_split(
            split_records,
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
        _write_json(
            annotation_dir / f"instances_{split}.json",
            coco,
            overwrite=overwrite,
        )

    summary = CocoExportSummary(
        xml_count=len(records),
        selected_record_count=len(selected),
        skipped_empty_record_count=skipped_empty,
        skipped_missing_image_count=skipped_missing,
        split_image_counts=split_image_counts,
        split_annotation_counts=split_annotation_counts,
        category_counts={key: value for key, value in category_counts.items() if value > 0},
        difficult_box_count=difficult_box_count,
        output_dir=output_dir,
        summary_json=annotation_dir / "export_summary.json",
    )
    _write_json(summary.summary_json, summary.to_dict(), overwrite=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, required=True, help="Directory of labelImg Pascal VOC XML files.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing the annotated images.")
    parser.add_argument("--label-list", type=Path, default=DEFAULT_LABEL_LIST, help="label_list.txt; order defines category ids.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="COCO output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for splitting.")
    parser.add_argument("--include-empty", action="store_true", help="Keep images without boxes as negative samples.")
    parser.add_argument("--keep-difficult", action="store_true", help="Keep difficult boxes with ignore=1 instead of filtering them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images/JSON files.")
    parser.add_argument("--missing-image-policy", choices=("error", "skip"), default="error")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = read_label_list(args.label_list)
    xml_dir = _resolve_from_root(args.xml_dir)
    image_dir = _resolve_from_root(args.image_dir)
    output_dir = _resolve_from_root(args.output)

    records, skipped_missing, difficult_count = load_annotations(
        xml_dir,
        image_dir=image_dir,
        labels=labels,
        keep_difficult=args.keep_difficult,
        missing_image_policy=args.missing_image_policy,
    )
    summary = materialize_coco(
        records,
        output_dir,
        labels=labels,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        include_empty=args.include_empty,
        overwrite=args.overwrite,
        skipped_missing=skipped_missing,
        difficult_box_count=difficult_count,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_from_root(path: Path) -> Path:
    if path is None:
        return ROOT
    return path if path.is_absolute() else ROOT / path


def _text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _int_text(node: ET.Element, tag: str, xml_path: Path) -> int:
    raw = _text(node, tag)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{xml_path}: <{tag}> is not an integer: {raw!r}") from exc


def _float_text(node: ET.Element, tag: str, xml_path: Path) -> float:
    raw = _text(node, tag)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{xml_path}: <{tag}> is not a number: {raw!r}") from exc


def _bool_text(node: ET.Element, tag: str) -> bool:
    raw = _text(node, tag)
    if raw in {"", "0", "false", "False"}:
        return False
    return True


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _split_counts(total: int, *, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))
    if test_count + val_count >= total and total > 0:
        test_count = max(0, test_count - 1) if test_count else 0
        val_count = max(0, val_count - 1) if val_count else 0
    return test_count, val_count


def _unique_output_filename(source_image: Path, used_filenames: set[str]) -> str:
    suffix = source_image.suffix.lower() or ".jpg"
    stem = _safe_stem(source_image.stem)
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


def _safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._") or "image"


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
