"""Reassemble numbered MP4 segments into a single video file."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATTERN = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required to reassemble video segments. "
            "Install dependencies from requirements.txt first."
        ) from exc
    return cv2


def extract_segment_index(path: Path) -> int:
    match = INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            f"unable to parse numeric segment index from filename: {path.name}"
        )
    return int(match.group(1))


def build_default_output_path(input_dir: Path) -> Path:
    if input_dir.suffix:
        return input_dir.with_name(f"{input_dir.stem}_merged{input_dir.suffix}")
    return input_dir.with_name(f"{input_dir.name}_merged.mp4")


def collect_segments(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    segments = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"]
    if not segments:
        raise FileNotFoundError(f"no mp4 segment files found in: {input_dir}")
    return sorted(segments, key=extract_segment_index)


def validate_contiguous_indices(paths: list[Path]) -> None:
    indices = [extract_segment_index(path) for path in paths]
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(
            "segment indices are not contiguous: "
            f"found {indices}, expected {expected}"
        )


def probe_video(path: Path) -> VideoInfo:
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open video segment: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0:
        raise RuntimeError(f"invalid fps reported for segment: {path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid frame size reported for segment: {path}")

    return VideoInfo(fps=fps, width=width, height=height)


def _same_video_layout(left: VideoInfo, right: VideoInfo) -> bool:
    return (
        abs(left.fps - right.fps) < 0.01
        and left.width == right.width
        and left.height == right.height
    )


def reassemble_segments(
    segment_paths: list[Path],
    output_path: Path,
    *,
    codec: str = "mp4v",
) -> None:
    cv2 = _load_cv2()
    reference = probe_video(segment_paths[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        reference.fps,
        (reference.width, reference.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"failed to open output video for writing: {output_path}")

    try:
        for index, segment_path in enumerate(segment_paths, start=1):
            info = probe_video(segment_path)
            if not _same_video_layout(reference, info):
                raise ValueError(
                    "segment video properties do not match the first segment: "
                    f"{segment_path.name}"
                )

            capture = cv2.VideoCapture(str(segment_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"failed to open video segment: {segment_path}")

            frame_count = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    writer.write(frame)
                    frame_count += 1
            finally:
                capture.release()

            print(
                f"[{index}/{len(segment_paths)}] merged {segment_path.name} "
                f"({frame_count} frames)"
            )
    finally:
        writer.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing numbered MP4 segments such as *_0.mp4, *_1.mp4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path. Defaults to a sibling file named *_merged.mp4.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc codec to use for output writing. Defaults to mp4v.",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Skip contiguous-index validation if segment numbers have gaps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_path = args.output if args.output is not None else build_default_output_path(input_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    segment_paths = collect_segments(input_dir)
    if not args.allow_gaps:
        validate_contiguous_indices(segment_paths)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output file already exists: {output_path}. Use --overwrite to replace it."
        )

    print(f"input directory: {input_dir}")
    print(f"segment count: {len(segment_paths)}")
    print(f"output file: {output_path}")

    reassemble_segments(segment_paths, output_path, codec=args.codec)
    print("video reassembly completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
