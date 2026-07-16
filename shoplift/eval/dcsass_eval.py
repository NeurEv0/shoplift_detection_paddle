"""Clip-level DCSASS Shoplifting evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from shoplift.cli.offline_analyze import (
    OutputPaths,
    load_offline_config,
    run_offline_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/DCSASS_Shoplifting"))
    parser.add_argument("--config", type=Path, default=Path("shoplift/configs/pipeline.example.yml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dcsass_eval"))
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-clips-per-label", type=int, default=None)
    parser.add_argument("--clip-id", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--backend", choices=("model_free", "paddledet_pphuman"), default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_dcsass_eval(args)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


def run_dcsass_eval(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root
    manifest_path = dataset_root / "metadata" / "manifest.csv"
    if not manifest_path.exists():
        raise OSError(f"DCSASS manifest does not exist: {manifest_path}")
    if not 0.0 <= float(args.score_threshold) <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")

    base_config = load_offline_config(
        argparse.Namespace(
            config=args.config,
            input=None,
            output=args.output,
            dry_run=False,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            no_debug=args.no_debug,
            backend=args.backend,
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)
    samples = _select_samples(
        list(_iter_manifest(dataset_root, manifest_path)),
        clip_ids=args.clip_id,
        max_clips=args.max_clips,
        max_clips_per_label=args.max_clips_per_label,
    )

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        clip_output = args.output / "clips" / sample["clip_id"]
        clip_config = replace(
            base_config,
            camera_id=sample["clip_id"],
            input_path=sample["video_path"],
            input_type="video",
            outputs=OutputPaths(
                root=clip_output,
                frame_jsonl=clip_output / "frame_results.jsonl",
                event_json=clip_output / "events.json",
                debug_dir=clip_output / "debug",
                debug_video=clip_output / "debug_visualization.mp4",
            ),
        )
        try:
            summary = run_offline_analysis(clip_config)
            events = json.loads(clip_config.outputs.event_json.read_text(encoding="utf-8"))
            prediction = _prediction_from_events(events, threshold=float(args.score_threshold))
            predictions.append(
                {
                    **sample,
                    "processed_frames": summary.processed_frames,
                    **prediction,
                    "visualization_path": str(summary.debug_visualization or ""),
                    "error": "",
                }
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            predictions.append(
                {
                    **sample,
                    "processed_frames": 0,
                    "pred_label": 0,
                    "max_risk_score": 0.0,
                    "max_risk_level": "none",
                    "event_count": 0,
                    "visualization_path": "",
                    "error": str(exc),
                }
            )

    metrics = _metrics(predictions)
    metrics_path = args.output / "metrics.json"
    predictions_path = args.output / "predictions.csv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_predictions(predictions_path, predictions)
    return {"metrics": metrics, "predictions": predictions}


def _iter_manifest(dataset_root: Path, manifest_path: Path) -> Iterable[dict[str, Any]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative_path = Path(str(row["relative_path"]).replace("\\", "/"))
            yield {
                "clip_id": row["clip_id"],
                "category": row["category"],
                "label": int(row["label"]),
                "original_video": row["original_video"],
                "relative_path": str(relative_path),
                "video_path": dataset_root / relative_path,
            }


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    clip_ids: list[str] | None,
    max_clips: int | None,
    max_clips_per_label: int | None,
) -> list[dict[str, Any]]:
    if max_clips is not None and max_clips <= 0:
        raise ValueError("--max-clips must be positive")
    if max_clips_per_label is not None and max_clips_per_label <= 0:
        raise ValueError("--max-clips-per-label must be positive")

    if clip_ids:
        by_id = {str(sample["clip_id"]): sample for sample in samples}
        missing = [clip_id for clip_id in clip_ids if clip_id not in by_id]
        if missing:
            raise ValueError(f"clip_id not found in manifest: {', '.join(missing)}")
        selected = [by_id[clip_id] for clip_id in clip_ids]
    else:
        selected = samples

    if max_clips_per_label is not None:
        counts: dict[int, int] = {}
        balanced: list[dict[str, Any]] = []
        for sample in selected:
            label = int(sample["label"])
            if counts.get(label, 0) >= max_clips_per_label:
                continue
            balanced.append(sample)
            counts[label] = counts.get(label, 0) + 1
        selected = balanced

    if max_clips is not None:
        selected = selected[:max_clips]
    return selected


def _prediction_from_events(events: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    if not events:
        return {
            "pred_label": 0,
            "max_risk_score": 0.0,
            "max_risk_level": "none",
            "event_count": 0,
        }
    max_event = max(events, key=lambda event: float(event.get("risk_score", 0.0)))
    max_score = float(max_event.get("risk_score", 0.0))
    max_level = str(max_event.get("risk_level", "none"))
    return {
        "pred_label": int(max_score >= threshold),
        "max_risk_score": max_score,
        "max_risk_level": max_level,
        "event_count": len(events),
    }


def _metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in predictions if row["label"] == 1 and row["pred_label"] == 1)
    fp = sum(1 for row in predictions if row["label"] == 0 and row["pred_label"] == 1)
    tn = sum(1 for row in predictions if row["label"] == 0 and row["pred_label"] == 0)
    fn = sum(1 for row in predictions if row["label"] == 1 and row["pred_label"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(predictions)
    errors = sum(1 for row in predictions if row.get("error"))
    visualizations = sum(1 for row in predictions if row.get("visualization_path"))
    return {
        "sample_count": total,
        "positive_count": sum(1 for row in predictions if row["label"] == 1),
        "negative_count": sum(1 for row in predictions if row["label"] == 0),
        "error_count": errors,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
        "visualization_count": visualizations,
    }


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    fieldnames = [
        "clip_id",
        "category",
        "label",
        "pred_label",
        "max_risk_score",
        "max_risk_level",
        "event_count",
        "processed_frames",
        "visualization_path",
        "original_video",
        "relative_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
