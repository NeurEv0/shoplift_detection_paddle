"""Run video-only inference and persist visualization artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from shoplift.cli.offline_analyze import (
    VIDEO_EXTENSIONS,
    OutputPaths,
    detect_input_type,
    dry_run_payload,
    load_offline_config,
    run_offline_analysis,
)


DEFAULT_CONFIG = Path("shoplift/configs/pipeline.test_videos.yml")
DEFAULT_INPUT = Path("datasets/test_videos")
DEFAULT_OUTPUT_ROOT = Path("outputs/inference_visualization/test_videos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Video path or video directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output root. Defaults to outputs/inference_visualization/test_videos for directories.")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without processing frames.")
    parser.add_argument("--frame-stride", type=int, default=None, help="Process one frame every N source frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue a directory batch if one video fails.")
    parser.add_argument(
        "--backend",
        choices=("model_free", "paddledet_pphuman"),
        default=None,
        help="Override the configured vision backend.",
    )
    parser.add_argument(
        "--no-debug-frames",
        action="store_true",
        help="Only write the visualization video, not sampled JPG frames.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    offline_args = argparse.Namespace(
        config=args.config,
        input=args.input,
        output=args.output,
        dry_run=args.dry_run,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        no_debug=False,
        debug_frames=not args.no_debug_frames,
        backend=args.backend,
    )
    try:
        config = load_offline_config(offline_args)
        if args.output is None:
            config = replace(config, outputs=replace(config.outputs, root=config.outputs.frame_jsonl.parent))
        if config.input_path is None:
            raise ValueError("video input path is required")
        if config.input_path.is_dir():
            summary = run_video_directory(config, args)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        configured_type = (config.input_type or "").strip().lower()
        input_type = detect_input_type(
            config.input_path,
            None if configured_type in {"video_dir", "videos"} else config.input_type,
        )
        if input_type != "video":
            raise ValueError(f"video_infer_visualize only accepts video input, got: {input_type}")
        output_root = args.output
        if output_root is None and configured_type in {"video_dir", "videos"}:
            output_root = DEFAULT_OUTPUT_ROOT / _safe_video_name(config.input_path)
        config = _video_config(config, config.input_path, output_root, save_debug_frames=not args.no_debug_frames)
        if args.dry_run:
            print(json.dumps(dry_run_payload(config), ensure_ascii=False, indent=2))
            return 0
        summary = run_offline_analysis(config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 2

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def run_video_directory(config, args: argparse.Namespace) -> dict:
    videos = discover_video_files(config.input_path)
    if not videos:
        raise ValueError(f"video directory contains no supported videos: {config.input_path}")

    output_root = args.output or DEFAULT_OUTPUT_ROOT
    summaries: list[dict] = []
    failures: list[dict[str, str]] = []
    for video_path in videos:
        video_config = _video_config(
            config,
            video_path,
            output_root / _safe_video_name(video_path),
            save_debug_frames=not args.no_debug_frames,
        )
        if args.dry_run:
            summaries.append(dry_run_payload(video_config))
            continue
        try:
            summaries.append(run_offline_analysis(video_config).to_dict())
        except (OSError, ValueError, RuntimeError) as exc:
            if not args.continue_on_error:
                raise
            failures.append({"input": str(video_path), "error": str(exc)})

    batch_summary = {
        "input": str(config.input_path),
        "input_type": "video_dir",
        "video_count": len(videos),
        "output_root": str(output_root),
        "summaries": summaries,
        "failures": failures,
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "batch_summary.json"
        summary_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        batch_summary["batch_summary"] = str(summary_path)
    return batch_summary


def discover_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _video_config(config, video_path: Path, output_root: Path | None, *, save_debug_frames: bool):
    if output_root is None:
        output_root = config.outputs.frame_jsonl.parent
    return replace(
        config,
        camera_id=video_path.stem,
        input_path=video_path,
        input_type="video",
        runtime=replace(
            config.runtime,
            save_debug_visualization=True,
            save_debug_frames=save_debug_frames,
        ),
        outputs=OutputPaths(
            root=output_root,
            frame_jsonl=output_root / "frame_results.jsonl",
            event_json=output_root / "events.json",
            debug_dir=output_root / "debug_frames",
            debug_video=output_root / "debug_visualization.mp4",
        ),
    )


def _safe_video_name(video_path: Path) -> str:
    raw_name = video_path.stem.replace("/", "_").replace("\\", "_").strip()
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    if not ascii_name:
        return f"video_{digest}"
    if ascii_name != raw_name:
        return f"{ascii_name}_{digest}"
    return ascii_name


if __name__ == "__main__":
    raise SystemExit(main())
