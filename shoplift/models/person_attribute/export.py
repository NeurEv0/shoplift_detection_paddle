"""Export a trained person-attribute model for Paddle Inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from shoplift.models.person_attribute.backbones import require_paddle
from shoplift.models.person_attribute.config import load_train_config
from shoplift.models.person_attribute.model import build_person_attribute_layer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("shoplift/configs/person_attribute.example.yml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--format", choices=("concat", "six_heads"), default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    output_dir = args.output_dir or config.export.output_dir
    output_format = args.format or config.export.output_format
    export(config, weights=args.weights, output_dir=output_dir, output_format=output_format)
    return 0


def export(config, *, weights: Path, output_dir: Path, output_format: str) -> None:
    paddle = require_paddle()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_person_attribute_layer(
        backbone_name=config.backbone.name,
        backbone_arch=config.backbone.arch,
        paddledetection_root=config.backbone.paddledetection_root,
        output_format="concat" if output_format == "concat" else "dict",
    )
    model.set_state_dict(paddle.load(str(weights)))
    model.eval()
    static_model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 3, config.data.image_height, config.data.image_width],
                dtype="float32",
                name="image",
            )
        ],
    )
    paddle.jit.save(static_model, str(output_dir / "inference"))
    _write_infer_cfg(output_dir, config, output_format)


def _write_infer_cfg(output_dir: Path, config, output_format: str) -> None:
    text = "\n".join(
        [
            "model_name: shoplift_person_attribute",
            "output_format: " + output_format,
            "Preprocess:",
            "  - type: Resize",
            f"    target_size: [{config.data.image_width}, {config.data.image_height}]",
            "  - type: NormalizeImage",
            "    mean: [0.485, 0.456, 0.406]",
            "    std: [0.229, 0.224, 0.225]",
            "  - type: Permute",
            "label_heads:",
            "  left_hand_state: [empty, holding_object, holding_product, uncertain]",
            "  left_hand_visibility: [clear, partial_occluded, not_judgable]",
            "  right_hand_state: [empty, holding_object, holding_product, uncertain]",
            "  right_hand_visibility: [clear, partial_occluded, not_judgable]",
            "  body_orientation: [front, side, back, unknown]",
            "  occlusion_level: [none, light, heavy]",
            "",
        ]
    )
    (output_dir / "infer_cfg.yml").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

