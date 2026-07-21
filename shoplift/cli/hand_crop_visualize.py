"""Run forearm-guided hand cropping on an image directory and save evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from shoplift.cli.offline_analyze import (
    BackendOptions,
    OutputPaths,
    RuntimeOptions,
    load_offline_config,
    run_offline_analysis,
)


DEFAULT_CONFIG = Path("shoplift/configs/pipeline.test_videos.yml")
DEFAULT_INPUT = Path("datasets/test_props/origins")
DEFAULT_OUTPUT = Path("outputs/hand_crops/test_props")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Image directory to process.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for JSONL, overlays, and crops.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--backend", choices=("paddledet_pphuman",), default="paddledet_pphuman")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _build_hand_crop_config(args)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "input": str(config.input_path),
                        "output": str(config.outputs.root),
                        "backend": config.backend.backend_type,
                        "pose_hand_enabled": config.modules.pose_hand_enabled,
                        "derive_hand_regions": (config.backend.options or {})
                        .get("keypoint", {})
                        .get("derive_hand_regions"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        summary = run_offline_analysis(config)
        crop_summary = save_hand_crops(config.outputs.frame_jsonl, config.outputs.root / "crops")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 2

    payload = {
        **summary.to_dict(),
        "crop_dir": str(config.outputs.root / "crops"),
        "saved_crop_count": crop_summary["saved_crop_count"],
        "hand_region_count": crop_summary["hand_region_count"],
        "hand_crop_summary": str(config.outputs.root / "hand_crop_summary.json"),
    }
    (config.outputs.root / "hand_crop_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_hand_crop_config(args: argparse.Namespace):
    config = load_offline_config(
        argparse.Namespace(
            config=args.config,
            input=args.input,
            output=args.output,
            dry_run=args.dry_run,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            no_debug=False,
            debug_frames=True,
            backend=args.backend,
        )
    )
    backend_options = dict(config.backend.options or {})
    keypoint_options = dict(backend_options.get("keypoint") or {})
    keypoint_options["enabled"] = True
    keypoint_options["derive_hand_regions"] = True
    backend_options["keypoint"] = keypoint_options
    item_options = dict(backend_options.get("item_container") or {})
    item_options["enabled"] = False
    backend_options["item_container"] = item_options

    return replace(
        config,
        input_path=args.input,
        input_type="frame_dir",
        runtime=RuntimeOptions(
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            save_debug_visualization=True,
            save_debug_frames=True,
        ),
        modules=replace(
            config.modules,
            pose_recognition_enabled=True,
            pose_hand_enabled=True,
            item_container_enabled=False,
        ),
        backend=BackendOptions(backend_type=args.backend, options=backend_options),
        outputs=OutputPaths(
            root=args.output,
            frame_jsonl=args.output / "frame_results.jsonl",
            event_json=args.output / "events.json",
            debug_dir=args.output / "overlays",
            debug_video=args.output / "debug_visualization.mp4",
        ),
    )


def save_hand_crops(frame_jsonl: Path, crop_dir: Path) -> dict[str, int]:
    cv2 = _import_cv2()
    crop_dir.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, Any] = {}
    saved_crop_count = 0
    hand_region_count = 0
    for line in frame_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame_result = json.loads(line)
        source_uri = str(frame_result.get("metadata", {}).get("source_uri") or "")
        if not source_uri:
            continue
        image = source_cache.get(source_uri)
        if image is None:
            image = cv2.imread(source_uri)
            if image is None:
                continue
            source_cache[source_uri] = image
        frame_id = int(frame_result.get("frame", {}).get("frame_id", 0))
        source_stem = Path(source_uri).stem
        for index, hand in enumerate(frame_result.get("hand_regions", [])):
            hand_region_count += 1
            crop = _crop_image(image, hand.get("bbox"))
            if crop is None:
                continue
            side = str(hand.get("side") or "unknown")
            strategy = str((hand.get("metadata") or {}).get("crop_strategy") or "hand_crop")
            crop_path = crop_dir / f"{source_stem}_frame{frame_id:06d}_{side}_{index}_{strategy}.jpg"
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            crop_path.write_bytes(encoded.tobytes())
            saved_crop_count += 1
    return {"saved_crop_count": saved_crop_count, "hand_region_count": hand_region_count}


def _crop_image(image: Any, bbox: Sequence[float] | None) -> Any | None:
    if bbox is None or len(bbox) != 4:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for hand crop visualization") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
