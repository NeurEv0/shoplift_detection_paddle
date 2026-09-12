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
    parser.add_argument("--config", type=Path, default=Path("shoplift/configs/person_attribute.yml"))
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
    test_loader = None
    if config.data.test_annotation is not None:
        test_dataset = PersonAttributeDataset(
            config.data.test_annotation,
            image_root=config.data.image_root,
            image_size=(config.data.image_width, config.data.image_height),
            training=False,
        )
        test_loader = make_dataloader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

    lr_scheduler = _build_lr_scheduler(paddle, config)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        weight_decay=config.optimizer.weight_decay,
        parameters=model.parameters(),
    )

    writer = _make_writer(config.output_dir)

    best_score = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, config.optimizer.epochs + 1):
        _set_backbone_trainable(model, epoch > config.backbone.freeze_backbone_epochs)
        model.train()
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer)
        payload: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "lr": float(lr_scheduler.get_lr()),
        }
        val_metrics = None
        if val_loader is not None:
            model.eval()
            val_metrics = _run_epoch(model, val_loader, optimizer=None)
            payload["val"] = val_metrics
            score = float(val_metrics["mean_accuracy"])
            if score > best_score:
                best_score = score
                paddle.save(model.state_dict(), str(config.output_dir / "best.pdparams"))
            val_loss = float(val_metrics["loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                paddle.save(
                    model.state_dict(),
                    str(config.output_dir / "best_val_loss.pdparams"),
                )
        _log_tensorboard(
            writer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=float(lr_scheduler.get_lr()),
        )
        paddle.save(model.state_dict(), str(config.output_dir / "last.pdparams"))
        lr_scheduler.step()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    if writer is not None:
        writer.close()

    if test_loader is not None:
        model.eval()
        test_metrics = _run_epoch(model, test_loader, optimizer=None)
        (config.output_dir / "test_summary.json").write_text(
            json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"epoch": "test", "test": test_metrics}, ensure_ascii=False, sort_keys=True))


def _build_lr_scheduler(paddle: Any, config: TrainConfig) -> Any:
    base_lr = config.optimizer.learning_rate
    epochs = max(1, config.optimizer.epochs)
    warmup_epochs = config.optimizer.warmup_epochs
    cosine = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr,
        T_max=epochs,
        eta_min=0.0,
    )
    if warmup_epochs > 0:
        return paddle.optimizer.lr.LinearWarmup(
            learning_rate=cosine,
            warmup_steps=int(warmup_epochs),
            start_lr=0.0,
            end_lr=base_lr,
        )
    return cosine


def _make_writer(output_dir: Path) -> Any:
    try:
        from tensorboardX import SummaryWriter

        return SummaryWriter(logdir=str(output_dir / "tensorboard"))
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        print(f"[train] tensorboardX unavailable, skipping tensorboard logging: {exc}")
        return None


def _log_tensorboard(
    writer: Any,
    *,
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any] | None,
    lr: float,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", train_metrics["loss"], epoch)
    writer.add_scalar("train/mean_accuracy", train_metrics["mean_accuracy"], epoch)
    for name in train_metrics["accuracy"]:
        writer.add_scalar(f"train/accuracy/{name}", train_metrics["accuracy"][name], epoch)
    if val_metrics is not None:
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/mean_accuracy", val_metrics["mean_accuracy"], epoch)
        for name in val_metrics["accuracy"]:
            writer.add_scalar(f"val/accuracy/{name}", val_metrics["accuracy"][name], epoch)
    writer.add_scalar("lr", lr, epoch)


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
