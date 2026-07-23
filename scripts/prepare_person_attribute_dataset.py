"""Prepare person-attribute labeling crops from supermarket videos or frames.

The script samples video frames, optionally uses existing offline-analysis
JSONL person track boxes, saves person crops, and writes a CSV annotation
template compatible with ``shoplift.models.person_attribute.dataset``.

It can also consume an existing frame-image directory or a CSV manifest created
by a previous full-frame sampling pass. This is the preferred path after
running the existing person detector on extracted store frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shoplift.models.person_attribute.dataset import REQUIRED_COLUMNS  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_LABELS = {
    "left_hand_state": "uncertain",
    "left_hand_visibility": "not_judgable",
    "right_hand_state": "uncertain",
    "right_hand_visibility": "not_judgable",
    "body_orientation": "unknown",
    "occlusion_level": "heavy",
}
METADATA_COLUMNS = (
    "label_status",
    "source_video",
    "source_image",
    "frame_id",
    "timestamp_ms",
    "person_track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_source",
)


@dataclass(frozen=True)
class CropCandidate:
    frame_id: int
    timestamp_ms: int
    bbox: tuple[float, float, float, float] | None
    person_track_id: str
    crop_source: str


@dataclass(frozen=True)
class PrepareSummary:
    video_count: int
    sampled_frame_count: int
    crop_count: int
    annotation_path: Path
    image_dir: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "video_count": self.video_count,
            "sampled_frame_count": self.sampled_frame_count,
            "crop_count": self.crop_count,
            "annotation_path": str(self.annotation_path),
            "image_dir": str(self.image_dir),
        }


@dataclass(frozen=True)
class CandidateIndex:
    by_source_uri: dict[str, dict[int, list[CropCandidate]]]
    by_frame_id: dict[int, list[CropCandidate]]


@dataclass(frozen=True)
class FrameSample:
    image_path: Path
    source_image: str
    source_video: str
    frame_id: int
    timestamp_ms: int


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required to prepare person-attribute crops. "
            "Install dependencies from requirements.txt first."
        ) from exc
    return cv2


def collect_videos(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported video suffix: {input_path}")
        return [input_path]
    videos = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        raise FileNotFoundError(f"no videos found under: {input_path}")
    return sorted(videos)


def collect_frame_images(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {input_path}")
        return [input_path]
    images = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise FileNotFoundError(f"no images found under: {input_path}")
    return sorted(images)


def load_jsonl_candidates(frame_jsonl: Path) -> CandidateIndex:
    by_source_uri: dict[str, dict[int, list[CropCandidate]]] = {}
    by_frame_id: dict[int, list[CropCandidate]] = {}
    with frame_jsonl.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL row must be an object at line {line_number}: {frame_jsonl}")
            frame = _mapping(payload.get("frame"), "frame")
            frame_id = int(frame.get("frame_id", payload.get("frame_id", 0)))
            timestamp_ms = int(frame.get("timestamp_ms", payload.get("timestamp_ms", 0)))
            tracks = payload.get("person_tracks") or []
            source_uri = _source_uri_from_payload(payload)
            frame_candidates = [_candidate_from_track(track, frame_id, timestamp_ms) for track in tracks]
            frame_candidates = [candidate for candidate in frame_candidates if candidate is not None]
            if frame_candidates:
                if source_uri is None:
                    by_frame_id.setdefault(frame_id, []).extend(frame_candidates)
                else:
                    for source_key in _source_keys(source_uri):
                        by_source_uri.setdefault(source_key, {}).setdefault(frame_id, []).extend(frame_candidates)
    return CandidateIndex(by_source_uri=by_source_uri, by_frame_id=by_frame_id)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    raise ValueError(f"{name} must be an object")


def _source_uri_from_payload(payload: Mapping[str, Any]) -> str | None:
    metadata = _mapping(payload.get("metadata"), "metadata")
    frame = _mapping(payload.get("frame"), "frame")
    source_uri = metadata.get("source_uri") or frame.get("source_uri")
    if source_uri is None:
        return None
    return str(source_uri)


def _source_keys(source_uri: str) -> tuple[str, ...]:
    path = Path(source_uri)
    keys = [source_uri]
    if path.name:
        keys.append(path.name)
    if path.stem:
        keys.append(path.stem)
    try:
        keys.append(str(path.resolve()))
    except OSError:
        pass
    return tuple(dict.fromkeys(keys))


def _candidate_from_track(
    track: object,
    frame_id: int,
    timestamp_ms: int,
) -> CropCandidate | None:
    if not isinstance(track, Mapping):
        return None
    boxes = track.get("boxes") or []
    if not boxes:
        return None
    latest = boxes[-1]
    if not isinstance(latest, Mapping):
        return None
    bbox = latest.get("bbox")
    if not _valid_bbox(bbox):
        return None
    return CropCandidate(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        bbox=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
        person_track_id=str(track.get("track_id") or latest.get("track_id") or "unknown"),
        crop_source="person_track",
    )


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def full_frame_candidate(frame_id: int, timestamp_ms: int) -> CropCandidate:
    return CropCandidate(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        bbox=None,
        person_track_id="unassigned",
        crop_source="full_frame",
    )


def crop_image(
    frame: Any,
    bbox: tuple[float, float, float, float] | None,
    *,
    padding_ratio: float,
) -> tuple[Any, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    if bbox is None:
        return frame, (0, 0, width, height)
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop after clipping: {(left, top, right, bottom)}")
    return frame[top:bottom, left:right], (left, top, right, bottom)


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    split: str = "train",
    frame_stride: int = 30,
    frame_stride_map: Mapping[str, int] | None = None,
    frame_jsonl: Path | None = None,
    padding_ratio: float = 0.08,
    full_frame_without_tracks: bool = True,
    overwrite: bool = False,
) -> PrepareSummary:
    cv2 = _load_cv2()
    videos = collect_videos(input_path)
    candidate_index = (
        load_jsonl_candidates(frame_jsonl)
        if frame_jsonl is not None
        else CandidateIndex(by_source_uri={}, by_frame_id={})
    )
    image_dir = output_dir / "images" / split
    annotation_path = output_dir / f"{split}.csv"
    if annotation_path.exists() and not overwrite:
        raise FileExistsError(f"annotation file already exists: {annotation_path}")
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sampled_frame_count = 0
    for video_path in videos:
        video_stride = _stride_for_video(video_path, frame_stride, frame_stride_map or {})
        source_candidates = _frame_candidates_for_video(candidate_index, video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"failed to open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_id = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_id % video_stride != 0:
                    frame_id += 1
                    continue
                sampled_frame_count += 1
                timestamp_ms = int(round((frame_id / fps) * 1000)) if fps > 0 else 0
                frame_candidates = source_candidates.get(frame_id, [])
                if not frame_candidates and full_frame_without_tracks:
                    frame_candidates = [full_frame_candidate(frame_id, timestamp_ms)]
                for person_index, candidate in enumerate(frame_candidates):
                    crop, clipped_bbox = crop_image(frame, candidate.bbox, padding_ratio=padding_ratio)
                    rel_image_path = _build_relative_image_path(
                        split,
                        video_path,
                        frame_id,
                        candidate.person_track_id,
                        person_index,
                    )
                    abs_image_path = output_dir / "images" / rel_image_path
                    abs_image_path.parent.mkdir(parents=True, exist_ok=True)
                    if abs_image_path.exists() and not overwrite:
                        raise FileExistsError(f"image file already exists: {abs_image_path}")
                    write_image(abs_image_path, crop, cv2=cv2)
                    rows.append(
                        _annotation_row(
                            rel_image_path,
                            source_video=video_path,
                            frame_id=frame_id,
                            timestamp_ms=candidate.timestamp_ms or timestamp_ms,
                            person_track_id=candidate.person_track_id,
                            bbox=clipped_bbox,
                            crop_source=candidate.crop_source,
                        )
                    )
                frame_id += 1
        finally:
            capture.release()

    _write_annotation(annotation_path, rows)
    return PrepareSummary(
        video_count=len(videos),
        sampled_frame_count=sampled_frame_count,
        crop_count=len(rows),
        annotation_path=annotation_path,
        image_dir=image_dir,
    )


def prepare_frame_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    split: str = "full",
    frame_jsonl: Path | None = None,
    source_annotation: Path | None = None,
    padding_ratio: float = 0.08,
    full_frame_without_tracks: bool = False,
    overwrite: bool = False,
) -> PrepareSummary:
    cv2 = _load_cv2()
    candidate_index = (
        load_jsonl_candidates(frame_jsonl)
        if frame_jsonl is not None
        else CandidateIndex(by_source_uri={}, by_frame_id={})
    )
    samples = (
        load_frame_samples_from_annotation(source_annotation)
        if source_annotation is not None
        else load_frame_samples_from_images(input_path)
    )
    image_dir = output_dir / "images" / split
    annotation_path = output_dir / f"{split}.csv"
    if annotation_path.exists() and not overwrite:
        raise FileExistsError(f"annotation file already exists: {annotation_path}")
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source_index, sample in enumerate(samples):
        frame = read_image(sample.image_path, cv2=cv2)
        if frame is None:
            raise ValueError(f"failed to read frame image: {sample.image_path}")
        frame_candidates = _frame_candidates_for_image(candidate_index, sample)
        if not frame_candidates and full_frame_without_tracks:
            frame_candidates = [full_frame_candidate(sample.frame_id, sample.timestamp_ms)]
        for person_index, candidate in enumerate(frame_candidates):
            crop, clipped_bbox = crop_image(frame, candidate.bbox, padding_ratio=padding_ratio)
            rel_image_path = _build_relative_frame_image_path(
                split,
                sample.image_path,
                source_index,
                candidate.frame_id,
                candidate.person_track_id,
                person_index,
            )
            abs_image_path = output_dir / "images" / rel_image_path
            abs_image_path.parent.mkdir(parents=True, exist_ok=True)
            if abs_image_path.exists() and not overwrite:
                raise FileExistsError(f"image file already exists: {abs_image_path}")
            write_image(abs_image_path, crop, cv2=cv2)
            rows.append(
                _annotation_row(
                    rel_image_path,
                    source_video=sample.source_video,
                    source_image=sample.source_image,
                    frame_id=sample.frame_id,
                    timestamp_ms=sample.timestamp_ms,
                    person_track_id=candidate.person_track_id,
                    bbox=clipped_bbox,
                    crop_source=candidate.crop_source,
                )
            )

    _write_annotation(annotation_path, rows)
    return PrepareSummary(
        video_count=0,
        sampled_frame_count=len(samples),
        crop_count=len(rows),
        annotation_path=annotation_path,
        image_dir=image_dir,
    )


def load_frame_samples_from_images(input_path: Path) -> list[FrameSample]:
    samples = []
    for index, path in enumerate(collect_frame_images(input_path)):
        samples.append(
            FrameSample(
                image_path=path,
                source_image=str(path),
                source_video="",
                frame_id=index,
                timestamp_ms=index * 33,
            )
        )
    return samples


def load_frame_samples_from_annotation(annotation_path: Path) -> list[FrameSample]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"source annotation does not exist: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    image_root = annotation_path.parent / "images"
    samples = []
    for index, row in enumerate(rows):
        raw_image_path = row.get("source_image") or row.get("image_path")
        if not raw_image_path:
            raise ValueError(f"source annotation row is missing image_path: {annotation_path}")
        image_path = Path(raw_image_path)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        samples.append(
            FrameSample(
                image_path=image_path,
                source_image=str(image_path),
                source_video=str(row.get("source_video") or ""),
                frame_id=_int_or_default(row.get("frame_id"), index),
                timestamp_ms=_int_or_default(row.get("timestamp_ms"), index * 33),
            )
        )
    return samples


def _frame_candidates_for_image(
    candidate_index: CandidateIndex,
    sample: FrameSample,
) -> list[CropCandidate]:
    source_values = [sample.source_image, str(sample.image_path)]
    for source_value in source_values:
        for source_key in _source_keys(source_value):
            candidates_by_frame = candidate_index.by_source_uri.get(source_key)
            if candidates_by_frame is None:
                continue
            if sample.frame_id in candidates_by_frame:
                return candidates_by_frame[sample.frame_id]
            flattened = [candidate for values in candidates_by_frame.values() for candidate in values]
            if flattened:
                return flattened
    return candidate_index.by_frame_id.get(sample.frame_id, [])


def write_image(path: Path, image: Any, *, cv2: Any) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def read_image(path: Path, *, cv2: Any) -> Any | None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return cv2.imread(str(path))
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _frame_candidates_for_video(
    candidate_index: CandidateIndex,
    video_path: Path,
) -> dict[int, list[CropCandidate]]:
    for source_key in (str(video_path.resolve()), str(video_path), video_path.name, video_path.stem):
        candidates = candidate_index.by_source_uri.get(source_key)
        if candidates is not None:
            return candidates
    return candidate_index.by_frame_id


def _stride_for_video(
    video_path: Path,
    default_stride: int,
    frame_stride_map: Mapping[str, int],
) -> int:
    for source_key in _source_keys(str(video_path)):
        if source_key in frame_stride_map:
            return frame_stride_map[source_key]
    for source_key in _source_keys(video_path.name):
        if source_key in frame_stride_map:
            return frame_stride_map[source_key]
    return default_stride


def _build_relative_image_path(
    split: str,
    video_path: Path,
    frame_id: int,
    person_track_id: str,
    person_index: int,
) -> Path:
    safe_track_id = _safe_stem(person_track_id)
    filename = f"{_safe_stem(video_path.stem)}_f{frame_id:06d}_{safe_track_id}_{person_index:02d}.jpg"
    return Path(split) / filename


def _build_relative_frame_image_path(
    split: str,
    image_path: Path,
    source_index: int,
    frame_id: int,
    person_track_id: str,
    person_index: int,
) -> Path:
    safe_track_id = _safe_stem(person_track_id)
    filename = (
        f"{_safe_stem(image_path.stem)}_s{source_index:06d}_"
        f"f{frame_id:06d}_{safe_track_id}_{person_index:02d}.jpg"
    )
    return Path(split) / filename


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "sample"


def _annotation_row(
    rel_image_path: Path,
    *,
    source_video: Path | str,
    source_image: Path | str = "",
    frame_id: int,
    timestamp_ms: int,
    person_track_id: str,
    bbox: tuple[int, int, int, int],
    crop_source: str,
) -> dict[str, object]:
    row: dict[str, object] = {"image_path": rel_image_path.as_posix(), **DEFAULT_LABELS}
    row.update(
        {
            "label_status": "unreviewed",
            "source_video": str(source_video),
            "source_image": str(source_image),
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "person_track_id": person_track_id,
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "crop_source": crop_source,
        }
    )
    return row


def _write_annotation(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    fieldnames = [*REQUIRED_COLUMNS, *METADATA_COLUMNS]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Video file, video directory, image file, or image directory.")
    parser.add_argument(
        "--input-type",
        default="auto",
        choices=("auto", "video", "image_dir"),
        help="Treat --input as videos or already-extracted frame images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/person_attribute_working"),
        help="Output dataset directory.",
    )
    parser.add_argument("--split", default="full", choices=("full", "train", "val"), help="Dataset split.")
    parser.add_argument("--frame-stride", type=int, default=30, help="Sample one frame every N frames.")
    parser.add_argument(
        "--frame-stride-map",
        default=None,
        help="Optional per-video overrides, e.g. 'a.mp4=20;b.mpg=90'. Keys may be paths, filenames, or stems.",
    )
    parser.add_argument(
        "--frame-jsonl",
        type=Path,
        default=None,
        help="Optional offline_analyze frame_results.jsonl containing person_tracks.",
    )
    parser.add_argument(
        "--source-annotation",
        type=Path,
        default=None,
        help="Optional previous full-frame CSV manifest. Relative image_path values resolve under <csv_dir>/images.",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.08, help="Padding around person bbox.")
    parser.add_argument(
        "--no-full-frame",
        action="store_true",
        help="Do not save full-frame samples when no person track bbox is available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images/CSV.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.padding_ratio < 0:
        raise ValueError("--padding-ratio must be non-negative")
    frame_stride_map = _parse_frame_stride_map(args.frame_stride_map)
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    frame_jsonl = args.frame_jsonl if args.frame_jsonl is None or args.frame_jsonl.is_absolute() else ROOT / args.frame_jsonl
    source_annotation = (
        args.source_annotation
        if args.source_annotation is None or args.source_annotation.is_absolute()
        else ROOT / args.source_annotation
    )
    input_type = _detect_prepare_input_type(input_path, args.input_type, source_annotation)
    if input_type == "image_dir":
        summary = prepare_frame_dataset(
            input_path,
            output_dir,
            split=args.split,
            frame_jsonl=frame_jsonl,
            source_annotation=source_annotation,
            padding_ratio=args.padding_ratio,
            full_frame_without_tracks=not args.no_full_frame,
            overwrite=args.overwrite,
        )
    else:
        summary = prepare_dataset(
            input_path,
            output_dir,
            split=args.split,
            frame_stride=args.frame_stride,
            frame_stride_map=frame_stride_map,
            frame_jsonl=frame_jsonl,
            padding_ratio=args.padding_ratio,
            full_frame_without_tracks=not args.no_full_frame,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _parse_frame_stride_map(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    result: dict[str, int] = {}
    for item in re.split(r"[;,]", value):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --frame-stride-map item: {item}")
        key, raw_stride = item.split("=", 1)
        key = key.strip()
        stride = int(raw_stride.strip())
        if not key:
            raise ValueError("empty key in --frame-stride-map")
        if stride <= 0:
            raise ValueError(f"frame stride must be positive for {key}")
        result[key] = stride
    return result


def _detect_prepare_input_type(
    input_path: Path,
    configured_type: str,
    source_annotation: Path | None,
) -> str:
    if configured_type != "auto":
        return configured_type
    if source_annotation is not None:
        return "image_dir"
    if input_path.is_file():
        return "image_dir" if input_path.suffix.lower() in IMAGE_SUFFIXES else "video"
    images = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if images:
        return "image_dir"
    return "video"


def _int_or_default(value: object, default: int) -> int:
    try:
        if value in {None, ""}:
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
