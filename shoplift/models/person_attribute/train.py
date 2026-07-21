"""Train the shoplift person-attribute model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shoplift.models.person_attribute.backbones import require_paddle
from shoplift.models.person_attribute.config import TrainConfig, load_train_config
from shoplift.models.person_attribute.dataset import PersonAttributeDataset, make_dataloader
from shoplift.models.person_attribute.labels import HEAD_SPECS
from shoplift.models.person_attribute.model import build_person_attribute_layer, load_pretrained


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("shoplift/configs/person_attribute.example.yml"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config)
    train(config, resume=args.resume)
    return 0


def train(config: TrainConfig, *, resume: Path | None = None) -> None:
    paddle = require_paddle()
    paddle.set_device(config.device)
    paddle.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_person_attribute_layer(
        backbone_name=config.backbone.name,
        backbone_arch=config.backbone.arch,
        paddledetection_root=config.backbone.paddledetection_root,
        output_format="dict",
    )
    if config.backbone.pretrained is not None:
        load_pretrained(model, config.backbone.pretrained, strict=False)
    if resume is not None:
        model.set_state_dict(paddle.load(str(resume)))

    train_dataset = PersonAttributeDataset(
        config.data.train_annotation,
        image_root=config.data.image_root,
        image_size=(config.data.image_width, config.data.image_height),
        training=True,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = None
    if config.data.val_annotation is not None:
        val_dataset = PersonAttributeDataset(
            config.data.val_annotation,
            image_root=config.data.image_root,
            image_size=(config.data.image_width, config.data.image_height),
            training=False,
        )
        val_loader = make_dataloader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

    optimizer = paddle.optimizer.AdamW(
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        parameters=model.parameters(),
    )

    best_score = -1.0
    for epoch in range(1, config.optimizer.epochs + 1):
        _set_backbone_trainable(model, epoch > config.backbone.freeze_backbone_epochs)
        model.train()
        metrics = _run_epoch(model, train_loader, optimizer=optimizer)
        payload = {"epoch": epoch, "train": metrics}
        if val_loader is not None:
            model.eval()
            val_metrics = _run_epoch(model, val_loader, optimizer=None)
            payload["val"] = val_metrics
            score = float(val_metrics["mean_accuracy"])
            if score > best_score:
                best_score = score
                paddle.save(model.state_dict(), str(config.output_dir / "best.pdparams"))
        paddle.save(model.state_dict(), str(config.output_dir / "last.pdparams"))
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _run_epoch(model: Any, loader: Any, *, optimizer: Any | None) -> dict[str, Any]:
    paddle = require_paddle()
    total_loss = 0.0
    total_samples = 0
    correct = {spec.name: 0 for spec in HEAD_SPECS}
    total = {spec.name: 0 for spec in HEAD_SPECS}

    for images, labels in loader:
        logits = model(images)
        losses = []
        batch_size = int(images.shape[0])
        for spec in HEAD_SPECS:
            target = labels[spec.name]
            loss = paddle.nn.functional.cross_entropy(logits[spec.name], target)
            losses.append(loss * spec.loss_weight)
            pred = paddle.argmax(logits[spec.name], axis=1)
            correct[spec.name] += int((pred == target).astype("int64").sum().item())
            total[spec.name] += batch_size
        loss = sum(losses)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    accuracies = {
        name: (correct[name] / total[name] if total[name] else 0.0)
        for name in correct
    }
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": accuracies,
        "mean_accuracy": sum(accuracies.values()) / max(1, len(accuracies)),
    }


def _set_backbone_trainable(model: Any, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for param in backbone.parameters():
        param.stop_gradient = not trainable


if __name__ == "__main__":
    raise SystemExit(main())
