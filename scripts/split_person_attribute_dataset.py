"""Split reviewed person-attribute annotations into train/val datasets.

The default paths target the current Yulong store crop-labeling workflow:

```
python scripts/split_person_attribute_dataset.py
```

This reads ``datasets/person_attribute/person_attribute_yulong_store_crops/full.csv``,
keeps rows with ``label_status=reviewed``, copies the corresponding crop images,
and writes train/val CSVs under ``datasets/person_attribute/splits``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shoplift.models.person_attribute.dataset import REQUIRED_COLUMNS  # noqa: E402
from shoplift.models.person_attribute.labels import HEAD_SPECS_BY_NAME  # noqa: E402


DEFAULT_SOURCE_DATASET = Path("datasets/person_attribute/person_attribute_yulong_store_crops")
DEFAULT_ANNOTATION = Path("full.csv")
DEFAULT_OUTPUT = Path("datasets/person_attribute/splits")
DEFAULT_STATUS = ("reviewed",)
KNOWN_SOURCE_SPLIT_DIRS = {"full", "train", "val", "test"}


@dataclass(frozen=True)
class SplitEntry:
    row: dict[str, str]
    source_image_path: Path
    original_index: int


@dataclass(frozen=True)
class SplitSummary:
    source_annotation: Path
    output_dir: Path
    train_annotation: Path
    val_annotation: Path
    source_row_count: int
    selected_row_count: int
    skipped_missing_image_count: int
    train_count: int
    val_count: int
    copied_image_count: int
    status_counts: dict[str, int]
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source_annotation": str(self.source_annotation),
            "output_dir": str(self.output_dir),
            "train_annotation": str(self.train_annotation),
            "val_annotation": str(self.val_annotation),
            "source_row_count": self.source_row_count,
            "selected_row_count": self.selected_row_count,
            "skipped_missing_image_count": self.skipped_missing_image_count,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "copied_image_count": self.copied_image_count,
            "status_counts": self.status_counts,
            "dry_run": self.dry_run,
        }


def split_labeled_dataset(
    source_dataset: Path,
    output_dir: Path,
    *,
    annotation: Path = DEFAULT_ANNOTATION,
    image_root: Path | None = None,
    statuses: Sequence[str] = DEFAULT_STATUS,
    val_ratio: float = 0.2,
    val_count: int | None = None,
    seed: int = 2026,
    train_split: str = "train",
    val_split: str = "val",
    overwrite: bool = False,
    dry_run: bool = False,
    missing_image_policy: str = "error",
) -> SplitSummary:
    """Create train/val split CSVs and image folders from completed labels."""

    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if val_count is not None and val_count < 0:
        raise ValueError("--val-count must be non-negative")
    if missing_image_policy not in {"error", "skip"}:
        raise ValueError("missing_image_policy must be 'error' or 'skip'")
    if train_split == val_split:
        raise ValueError("train and val split names must be different")

    annotation_path = _resolve_annotation_path(source_dataset, annotation)
    source_image_root = _resolve_image_root(source_dataset, image_root)
    output_dir = _resolve_from_root(output_dir)
    statuses = tuple(_normalize_statuses(statuses))

    rows, fieldnames, status_counts = _read_annotation_rows(annotation_path)
    selected_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if _row_status_matches(row, fieldnames, statuses)
    ]
    entries, skipped_missing = _build_entries(
        selected_rows,
        source_image_root=source_image_root,
        annotation_path=annotation_path,
        missing_image_policy=missing_image_policy,
    )
    _validate_labels(entries, annotation_path)
    if not entries:
        raise ValueError(
            "no rows are available for splitting; check label_status or --status"
        )

    val_size = _resolve_val_count(
        len(entries),
        val_ratio=val_ratio,
        val_count=val_count,
    )
    val_indices = _choose_val_indices(len(entries), val_size, seed)
    train_entries = [
        entry for index, entry in enumerate(entries) if index not in val_indices
    ]
    val_entries = [
        entry for index, entry in enumerate(entries) if index in val_indices
    ]

    output_fieldnames = _ordered_fieldnames(fieldnames)
    train_rows, train_copied = _materialize_split(
        train_entries,
        split_name=train_split,
        output_dir=output_dir,
        fieldnames=output_fieldnames,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    val_rows, val_copied = _materialize_split(
        val_entries,
        split_name=val_split,
        output_dir=output_dir,
        fieldnames=output_fieldnames,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    train_annotation = output_dir / f"{train_split}.csv"
    val_annotation = output_dir / f"{val_split}.csv"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(train_annotation, train_rows, output_fieldnames, overwrite=overwrite)
        _write_csv(val_annotation, val_rows, output_fieldnames, overwrite=overwrite)
        _write_summary(
            output_dir / "split_summary.json",
            source_annotation=annotation_path,
            output_dir=output_dir,
            train_annotation=train_annotation,
            val_annotation=val_annotation,
            source_row_count=len(rows),
            selected_row_count=len(entries),
            skipped_missing_image_count=skipped_missing,
            train_count=len(train_entries),
            val_count=len(val_entries),
            copied_image_count=train_copied + val_copied,
            status_counts=status_counts,
        )

    return SplitSummary(
        source_annotation=annotation_path,
        output_dir=output_dir,
        train_annotation=train_annotation,
        val_annotation=val_annotation,
        source_row_count=len(rows),
        selected_row_count=len(entries),
        skipped_missing_image_count=skipped_missing,
        train_count=len(train_entries),
        val_count=len(val_entries),
        copied_image_count=train_copied + val_copied,
        status_counts=status_counts,
        dry_run=dry_run,
    )


def _resolve_annotation_path(source_dataset: Path, annotation: Path) -> Path:
    source_dataset = _resolve_from_root(source_dataset)
    annotation_path = annotation if annotation.is_absolute() else source_dataset / annotation
    if not annotation_path.exists():
        raise FileNotFoundError(f"annotation CSV does not exist: {annotation_path}")
    return annotation_path


def _resolve_image_root(source_dataset: Path, image_root: Path | None) -> Path:
    source_dataset = _resolve_from_root(source_dataset)
    if image_root is None:
        image_root = source_dataset / "images"
    elif not image_root.is_absolute():
        image_root = source_dataset / image_root
    if not image_root.exists():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    return image_root


def _resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _normalize_statuses(statuses: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for status in statuses:
        for item in status.split(","):
            item = item.strip()
            if item:
                values.append(item)
    return tuple(dict.fromkeys(values)) or DEFAULT_STATUS


def _read_annotation_rows(
    annotation_path: Path,
) -> tuple[list[dict[str, str]], list[str], dict[str, int]]:
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"annotation is missing required columns: {missing}")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("label_status") or "<missing>"
        status_counts[status] = status_counts.get(status, 0) + 1
    return rows, fieldnames, status_counts


def _row_status_matches(
    row: Mapping[str, str],
    fieldnames: Sequence[str],
    statuses: Sequence[str],
) -> bool:
    if "all" in statuses:
        return True
    if "label_status" not in fieldnames:
        return True
    return row.get("label_status", "") in statuses


def _build_entries(
    indexed_rows: Iterable[tuple[int, dict[str, str]]],
    *,
    source_image_root: Path,
    annotation_path: Path,
    missing_image_policy: str,
) -> tuple[list[SplitEntry], int]:
    entries: list[SplitEntry] = []
    skipped_missing = 0
    for original_index, row in indexed_rows:
        image_path = _source_image_path(row.get("image_path", ""), source_image_root)
        if not image_path.exists():
            if missing_image_policy == "skip":
                skipped_missing += 1
                continue
            raise FileNotFoundError(
                f"image does not exist for row {original_index + 2} in "
                f"{annotation_path}: {image_path}"
            )
        if not image_path.is_file():
            raise ValueError(f"image path is not a file: {image_path}")
        entries.append(
            SplitEntry(
                row=dict(row),
                source_image_path=image_path,
                original_index=original_index,
            )
        )
    return entries, skipped_missing


def _source_image_path(raw_image_path: str, source_image_root: Path) -> Path:
    if not raw_image_path:
        raise ValueError("annotation row is missing image_path")
    image_path = Path(raw_image_path)
    if image_path.is_absolute():
        return image_path
    _validate_relative_parts(raw_image_path)
    return source_image_root / image_path


def _validate_relative_parts(raw_path: str) -> None:
    normalized = raw_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"relative image_path must not contain '..': {raw_path}")


def _validate_labels(entries: Sequence[SplitEntry], annotation_path: Path) -> None:
    for entry in entries:
        for name, spec in HEAD_SPECS_BY_NAME.items():
            value = entry.row.get(name, "")
            if value not in spec.labels:
                raise ValueError(
                    f"unsupported label for {name} at row {entry.original_index + 2} "
                    f"in {annotation_path}: {value}"
                )


def _resolve_val_count(
    sample_count: int,
    *,
    val_ratio: float,
    val_count: int | None,
) -> int:
    if sample_count <= 1:
        return 0
    if val_count is None:
        count = int(round(sample_count * val_ratio))
        if val_ratio > 0:
            count = max(1, count)
    else:
        count = val_count
    return min(max(0, count), sample_count - 1)


def _choose_val_indices(sample_count: int, val_count: int, seed: int) -> set[int]:
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    return set(indices[:val_count])


def _ordered_fieldnames(fieldnames: Sequence[str]) -> list[str]:
    required = [name for name in REQUIRED_COLUMNS if name in fieldnames]
    rest = [name for name in fieldnames if name not in required]
    return [*required, *rest]


def _materialize_split(
    entries: Sequence[SplitEntry],
    *,
    split_name: str,
    output_dir: Path,
    fieldnames: Sequence[str],
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    used_paths: set[str] = set()
    copied = 0
    for entry in entries:
        rel_image_path = _unique_output_image_path(
            entry.row["image_path"],
            split_name=split_name,
            used_paths=used_paths,
        )
        destination = output_dir / "images" / rel_image_path
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"output image already exists: {destination}; use --overwrite to replace"
            )
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.source_image_path, destination)
            copied += 1

        output_row = dict(entry.row)
        output_row["image_path"] = rel_image_path.as_posix()
        rows.append({name: output_row.get(name, "") for name in fieldnames})
    return rows, copied


def _unique_output_image_path(
    raw_image_path: str,
    *,
    split_name: str,
    used_paths: set[str],
) -> Path:
    candidate = _output_image_path(raw_image_path, split_name)
    key = candidate.as_posix().lower()
    if key not in used_paths:
        used_paths.add(key)
        return candidate

    parent = candidate.parent
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        deduped = parent / f"{stem}_dup{counter:02d}{suffix}"
        key = deduped.as_posix().lower()
        if key not in used_paths:
            used_paths.add(key)
            return deduped
        counter += 1


def _output_image_path(raw_image_path: str, split_name: str) -> Path:
    image_path = Path(raw_image_path)
    if image_path.is_absolute():
        tail = Path(image_path.name)
    else:
        normalized = raw_image_path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part not in {"", "."}]
        if parts and parts[0].lower() in KNOWN_SOURCE_SPLIT_DIRS:
            parts = parts[1:]
        if not parts:
            raise ValueError(f"cannot build output image path from: {raw_image_path}")
        if any(part == ".." for part in parts):
            raise ValueError(f"relative image_path must not contain '..': {raw_image_path}")
        tail = Path(*parts)
    return Path(split_name) / tail


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, str]],
    fieldnames: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"annotation file already exists: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_summary(path: Path, **payload: object) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help="Source labeled dataset directory.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
        help="Annotation CSV path or path relative to --source-dataset.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Source image root. Defaults to <source-dataset>/images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output split dataset directory.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Rows to include by label_status. Can be repeated or comma-separated. Use 'all' to include every row.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Exact validation sample count. Overrides --val-ratio.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip selected rows whose image files are missing instead of failing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = split_labeled_dataset(
        args.source_dataset,
        args.output,
        annotation=args.annotation,
        image_root=args.image_root,
        statuses=args.status or DEFAULT_STATUS,
        val_ratio=args.val_ratio,
        val_count=args.val_count,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        missing_image_policy="skip" if args.skip_missing_images else "error",
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
