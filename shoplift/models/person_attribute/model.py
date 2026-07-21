"""Multi-task person-attribute classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shoplift.models.person_attribute.backbones import build_backbone, require_paddle
from shoplift.models.person_attribute.labels import HEAD_SPECS, total_output_dim

try:
    import paddle as _paddle

    _LayerBase = _paddle.nn.Layer
except ImportError:
    _LayerBase = object


class PersonAttributeModel:
    """Backbone + shared projection + six classification heads."""

    def __init__(
        self,
        *,
        backbone_name: str = "pphgnetv2",
        backbone_arch: str = "S",
        paddledetection_root: Path = Path("src/PaddleDetection-release-2.9"),
        dropout: float = 0.2,
        embedding_dim: int = 512,
        output_format: str = "dict",
    ) -> None:
        self.layer = _PersonAttributeLayer(
            backbone_name=backbone_name,
            backbone_arch=backbone_arch,
            paddledetection_root=paddledetection_root,
            dropout=dropout,
            embedding_dim=embedding_dim,
            output_format=output_format,
        )

    def __call__(self, x: Any) -> Any:
        return self.layer(x)

    def state_dict(self) -> dict[str, Any]:
        return self.layer.state_dict()

    def set_state_dict(self, state: dict[str, Any]) -> None:
        self.layer.set_state_dict(state)

    def parameters(self):
        return self.layer.parameters()

    def train(self) -> None:
        self.layer.train()

    def eval(self) -> None:
        self.layer.eval()


class _PersonAttributeLayer(_LayerBase):
    def __init__(
        self,
        *,
        backbone_name: str,
        backbone_arch: str,
        paddledetection_root: Path,
        dropout: float,
        embedding_dim: int,
        output_format: str,
    ) -> None:
        super().__init__()
        paddle = require_paddle()
        nn = paddle.nn
        backbone, out_channels = build_backbone(
            name=backbone_name,
            arch=backbone_arch,
            paddledetection_root=paddledetection_root,
        )
        self.backbone = backbone.backbone if hasattr(backbone, "backbone") else backbone.model
        self.backbone_adapter = backbone
        self.output_format = output_format
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(out_channels, embedding_dim),
            nn.BatchNorm1D(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.LayerDict(
            {
                spec.name: nn.Linear(embedding_dim, spec.num_classes)
                for spec in HEAD_SPECS
            }
        )

    def forward(self, x: Any) -> Any:
        features = self.backbone_adapter(x)
        pooled = self.flatten(self.pool(features))
        embedding = self.proj(pooled)
        outputs = {name: head(embedding) for name, head in self.heads.items()}
        if self.output_format == "concat":
            paddle = require_paddle()
            return paddle.concat([outputs[spec.name] for spec in HEAD_SPECS], axis=1)
        return outputs


def build_person_attribute_layer(
    *,
    backbone_name: str = "pphgnetv2",
    backbone_arch: str = "S",
    paddledetection_root: Path = Path("src/PaddleDetection-release-2.9"),
    output_format: str = "dict",
) -> Any:
    return _PersonAttributeLayer(
        backbone_name=backbone_name,
        backbone_arch=backbone_arch,
        paddledetection_root=paddledetection_root,
        dropout=0.2,
        embedding_dim=512,
        output_format=output_format,
    )


def load_pretrained(layer: Any, path: Path, *, strict: bool = False) -> None:
    paddle = require_paddle()
    state = paddle.load(str(path))
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    current = layer.state_dict()
    filtered = {}
    for key, value in state.items():
        target_key = key
        if target_key not in current and f"backbone.{key}" in current:
            target_key = f"backbone.{key}"
        if target_key in current and list(current[target_key].shape) == list(value.shape):
            filtered[target_key] = value
    if strict and len(filtered) != len(current):
        missing = sorted(set(current) - set(filtered))
        raise ValueError(f"pretrained weights did not cover all model params: {missing[:20]}")
    current.update(filtered)
    layer.set_state_dict(current)


__all__ = [
    "PersonAttributeModel",
    "build_person_attribute_layer",
    "load_pretrained",
    "total_output_dim",
]
