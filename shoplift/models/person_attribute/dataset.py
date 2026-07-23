"""Dataset utilities for person-attribute model training."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from shoplift.models.person_attribute.backbones import require_paddle
from shoplift.models.person_attribute.labels import HEAD_SPECS_BY_NAME


REQUIRED_COLUMNS = (
    "image_path",
    "left_hand_state",
    "left_hand_visibility",
    "right_hand_state",
    "right_hand_visibility",
    "body_orientation",
    "occlusion_level",
)


@dataclass(frozen=True)
class PersonAttributeSample:
    image_path: Path
    labels: dict[str, int]
    metadata: dict[str, Any]


def read_annotation(path: Path, *, image_root: Path = Path(".")) -> tuple[PersonAttributeSample, ...]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = _read_jsonl(path)
    else:
        raise ValueError(f"unsupported annotation format: {path}")

    samples = []
    for row in rows:
        _validate_row(row, path)
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = image_root / image_path
        labels = {
            name: HEAD_SPECS_BY_NAME[name].label_to_index(str(row[name]))
            for name in HEAD_SPECS_BY_NAME
        }
        metadata = {key: value for key, value in row.items() if key not in REQUIRED_COLUMNS}
        samples.append(PersonAttributeSample(image_path=image_path, labels=labels, metadata=metadata))
    return tuple(samples)


class PersonAttributeDataset:
    def __init__(
        self,
        annotation: Path,
        *,
        image_root: Path = Path("."),
        image_size: tuple[int, int] = (192, 256),
        training: bool = True,
    ) -> None:
        paddle = require_paddle()
        self._dataset_base = paddle.io.Dataset
        self.samples = read_annotation(annotation, image_root=image_root)
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _load_image(sample.image_path, self.image_size, self.training)
        labels = {name: value for name, value in sample.labels.items()}
        return image, labels


def make_dataloader(
    dataset: PersonAttributeDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    paddle = require_paddle()
    return paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        return_list=True,
    )


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data = json.loads(line)
                if not isinstance(data, Mapping):
                    raise ValueError(f"JSONL rows must be mappings: {path}")
                yield dict(data)


def _validate_row(row: Mapping[str, Any], path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in row or row[column] in {None, ""}]
    if missing:
        raise ValueError(f"annotation row in {path} is missing columns: {missing}")


def _load_image(path: Path, image_size: tuple[int, int], training: bool):
    import cv2
    import numpy as np

    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 3)
    image = (image - mean) / std
    return image.transpose(2, 0, 1)

__all__ = [
    "PersonAttributeDataset",
    "PersonAttributeSample",
    "REQUIRED_COLUMNS",
    "make_dataloader",
    "read_annotation",
]
