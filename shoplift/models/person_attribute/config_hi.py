"""Configuration helpers for person-attribute training and export.

Stage B variant (config_hi):在原 config.py 基础上增加逐头类别权重 loss_class_weights 解析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "pphgnetv2"
    arch: str = "S"
    paddledetection_root: Path = Path("src/PaddleDetection-release-2.9")
    pretrained: Path | None = None
    freeze_backbone_epochs: int = 0


@dataclass(frozen=True)
class DataConfig:
    train_annotation: Path
    val_annotation: Path | None = None
    test_annotation: Path | None = None
    image_root: Path = Path(".")
    image_width: int = 192
    image_height: int = 256
    batch_size: int = 32
    num_workers: int = 0


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 30
    warmup_epochs: int = 1


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path = Path("models/shoplift/person_attribute/inference")
    output_format: str = "concat"


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path = Path("outputs/person_attribute")
    device: str = "cpu"
    seed: int = 2026
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(
        default_factory=lambda: DataConfig(train_annotation=Path("datasets/person_attribute/train.csv"))
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    # 逐头类别权重:head 名 -> 按 spec.labels 顺序的每类权重(可选)。
    # 例如提高 holding_product 召回:{"right_hand_state": [1.0, 1.0, 3.0, 0.7], ...}
    loss_class_weights: dict[str, tuple[float, ...]] = field(default_factory=dict)


def load_train_config(path: Path) -> TrainConfig:
    data = _load_yaml(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"config must be a mapping: {path}")
    return TrainConfig(
        output_dir=Path(data.get("output_dir", "outputs/person_attribute")),
        device=str(data.get("device", "cpu")),
        seed=int(data.get("seed", 2026)),
        backbone=_load_backbone(data.get("backbone", {})),
        data=_load_data(data.get("data", {})),
        optimizer=_load_optimizer(data.get("optimizer", {})),
        export=_load_export(data.get("export", {})),
        loss_class_weights=_load_class_weights(data.get("loss", {})),
    )


def _load_backbone(data: Any) -> BackboneConfig:
    mapping = data if isinstance(data, Mapping) else {}
    pretrained = mapping.get("pretrained")
    return BackboneConfig(
        name=str(mapping.get("name", "pphgnetv2")),
        arch=str(mapping.get("arch", "S")),
        paddledetection_root=Path(mapping.get("paddledetection_root", "src/PaddleDetection-release-2.9")),
        pretrained=Path(pretrained) if pretrained else None,
        freeze_backbone_epochs=int(mapping.get("freeze_backbone_epochs", 0)),
    )


def _load_data(data: Any) -> DataConfig:
    mapping = data if isinstance(data, Mapping) else {}
    return DataConfig(
        train_annotation=Path(mapping.get("train_annotation", "datasets/person_attribute/train.csv")),
        val_annotation=Path(mapping["val_annotation"]) if mapping.get("val_annotation") else None,
        test_annotation=Path(mapping["test_annotation"]) if mapping.get("test_annotation") else None,
        image_root=Path(mapping.get("image_root", ".")),
        image_width=int(mapping.get("image_width", 192)),
        image_height=int(mapping.get("image_height", 256)),
        batch_size=int(mapping.get("batch_size", 32)),
        num_workers=int(mapping.get("num_workers", 0)),
    )


def _load_optimizer(data: Any) -> OptimizerConfig:
    mapping = data if isinstance(data, Mapping) else {}
    return OptimizerConfig(
        learning_rate=float(mapping.get("learning_rate", 0.001)),
        weight_decay=float(mapping.get("weight_decay", 0.0001)),
        epochs=int(mapping.get("epochs", 30)),
        warmup_epochs=int(mapping.get("warmup_epochs", 1)),
    )


def _load_export(data: Any) -> ExportConfig:
    mapping = data if isinstance(data, Mapping) else {}
    return ExportConfig(
        output_dir=Path(mapping.get("output_dir", "models/shoplift/person_attribute/inference")),
        output_format=str(mapping.get("output_format", "concat")),
    )


def _load_class_weights(data: Any) -> dict[str, tuple[float, ...]]:
    mapping = data if isinstance(data, Mapping) else {}
    weights = mapping.get("class_weights")
    if weights is None:
        return {}
    if not isinstance(weights, Mapping):
        raise ValueError("loss.class_weights must be a mapping of head name -> per-class weights")
    result: dict[str, tuple[float, ...]] = {}
    for head, values in weights.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"loss.class_weights[{head}] must be a non-empty list of floats")
        result[str(head)] = tuple(float(value) for value in values)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(text)
    data = yaml.safe_load(text) or {}
    return data if isinstance(data, dict) else {}


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for indent, content in lines:
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key, separator, raw_value = content.partition(":")
        if not separator:
            raise ValueError(f"simple YAML parser expected key: value, got: {content}")
        key = key.strip()
        value = raw_value.strip()
        if value == "":
            child: dict[str, Any] = {}
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


__all__ = [
    "BackboneConfig",
    "DataConfig",
    "ExportConfig",
    "OptimizerConfig",
    "TrainConfig",
    "load_train_config",
]
