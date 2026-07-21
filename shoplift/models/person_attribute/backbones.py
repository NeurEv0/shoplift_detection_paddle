"""Backbone builders for the person-attribute model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def require_paddle() -> Any:
    try:
        import paddle
    except ImportError as exc:
        raise RuntimeError("paddlepaddle is required for person-attribute model training") from exc
    return paddle


def build_backbone(
    *,
    name: str,
    arch: str = "S",
    paddledetection_root: Path = Path("src/PaddleDetection-release-2.9"),
) -> tuple[Any, int]:
    """Build a backbone and return `(layer, output_channels)`."""

    normalized = name.strip().lower()
    if normalized in {"tiny_cnn", "tiny"}:
        return TinyCNN(), 256
    if normalized in {"pphgnet", "pphgnetv2", "hgnetv2"}:
        return _build_pphgnetv2(arch=arch, paddledetection_root=paddledetection_root)
    if normalized in {"lcnet", "pplcnet", "pp_lcnet"}:
        return _build_lcnet(arch=arch, paddledetection_root=paddledetection_root)
    raise ValueError(f"unsupported backbone: {name}")


def _build_pphgnetv2(*, arch: str, paddledetection_root: Path) -> tuple[Any, int]:
    _ensure_paddledet_path(paddledetection_root)
    from ppdet.modeling.backbones.hgnet_v2 import PPHGNetV2

    model = PPHGNetV2(arch=arch.upper(), return_idx=[3], freeze_at=-1, freeze_norm=False)
    out_channels = int(model._out_channels[3])
    return PaddleDetectionBackboneAdapter(model), out_channels


def _build_lcnet(*, arch: str, paddledetection_root: Path) -> tuple[Any, int]:
    _ensure_paddledet_path(paddledetection_root)
    from ppdet.modeling.backbones.lcnet import LCNet

    scale = float(arch) if str(arch).replace(".", "", 1).isdigit() else 1.0
    model = LCNet(scale=scale, feature_maps=[5])
    out_channels = int(model._out_channels[-1])
    return PaddleDetectionBackboneAdapter(model), out_channels


def _ensure_paddledet_path(root: Path) -> None:
    resolved = root.resolve()
    text = str(resolved)
    if text not in sys.path:
        sys.path.insert(0, text)


class PaddleDetectionBackboneAdapter:
    """Adapt PPDet backbones that expect `{"image": tensor}` inputs."""

    def __init__(self, backbone: Any) -> None:
        self.backbone = backbone

    def __call__(self, x: Any) -> Any:
        outputs = self.backbone({"image": x})
        if isinstance(outputs, (list, tuple)):
            return outputs[-1]
        return outputs

    def parameters(self):
        return self.backbone.parameters()

    def train(self) -> None:
        self.backbone.train()

    def eval(self) -> None:
        self.backbone.eval()


class TinyCNN:
    """Small fallback CNN for smoke tests and CPU-only prototyping."""

    def __init__(self) -> None:
        paddle = require_paddle()
        nn = paddle.nn
        self.model = nn.Sequential(
            nn.Conv2D(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
        )

    def __call__(self, x: Any) -> Any:
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


__all__ = [
    "PaddleDetectionBackboneAdapter",
    "TinyCNN",
    "build_backbone",
    "require_paddle",
]

