"""Stage B 评估入口(高分辨率/类别权重实验),不改动 v1 的 evaluate.py。

用法(与 v1 相同,但 --config 用 hi384 配置,模块名带 _hi):
  python -m shoplift.models.person_attribute.evaluate_hi \\
      --config shoplift/configs/person_attribute_hi384.yml \\
      --checkpoint outputs/wqh/person_attribute/hi384/best.pdparams --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shoplift.models.person_attribute.backbones import require_paddle
from shoplift.models.person_attribute.config_hi import TrainConfig, load_train_config
from shoplift.models.person_attribute.dataset import PersonAttributeDataset, make_dataloader
from shoplift.models.person_attribute.model import build_person_attribute_layer
from shoplift.models.person_attribute.train_hi import _run_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("shoplift/configs/person_attribute_hi384.yml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pdparams to evaluate")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the result JSON (default: next to the checkpoint)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    evaluate(config, checkpoint=args.checkpoint, split=args.split, output=args.output)
    return 0


def evaluate(
    config: TrainConfig,
    *,
    checkpoint: Path,
    split: str = "test",
    output: Path | None = None,
) -> dict[str, Any]:
    paddle = require_paddle()
    paddle.set_device(config.device)

    model = build_person_attribute_layer(
        backbone_name=config.backbone.name,
        backbone_arch=config.backbone.arch,
        paddledetection_root=config.backbone.paddledetection_root,
        output_format="dict",
    )
    model.set_state_dict(paddle.load(str(checkpoint)))
    model.eval()

    annotation = {
        "train": config.data.train_annotation,
        "val": config.data.val_annotation,
        "test": config.data.test_annotation,
    }[split]
    if annotation is None:
        raise ValueError(f"no annotation configured for split '{split}'")

    dataset = PersonAttributeDataset(
        annotation,
        image_root=config.data.image_root,
        image_size=(config.data.image_width, config.data.image_height),
        training=False,
    )
    loader = make_dataloader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    metrics = _run_epoch(
        model,
        loader,
        optimizer=None,
        class_weights=config.loss_class_weights,
    )
    result = {"checkpoint": str(checkpoint), "split": split, **metrics}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if output is None:
        output = checkpoint.parent / f"eval_{split}_{checkpoint.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
