from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from shoplift.cli.video_infer_visualize import _safe_video_name, discover_video_files, main


class VideoInferVisualizeTest(unittest.TestCase):
    def test_directory_dry_run_creates_per_video_output_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            input_dir.mkdir()
            (input_dir / "a.mp4").write_bytes(b"")
            (input_dir / "b.mpg").write_bytes(b"")
            (input_dir / "note.txt").write_text("ignore", encoding="utf-8")
            config_path = self.write_config(root / "pipeline.yml", input_dir)
            output_root = root / "outputs"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(input_dir),
                        "--output",
                        str(output_root),
                        "--backend",
                        "model_free",
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["input_type"], "video_dir")
            self.assertEqual(payload["video_count"], 2)
            debug_videos = [
                summary["outputs"]["debug_video"].replace("\\", "/")
                for summary in payload["summaries"]
            ]
            self.assertTrue(any(path.endswith("a/debug_visualization.mp4") for path in debug_videos))
            self.assertTrue(any(path.endswith("b/debug_visualization.mp4") for path in debug_videos))

    def test_single_file_override_ignores_video_dir_config_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "videos"
            input_dir.mkdir()
            video_path = input_dir / "sample.mp4"
            video_path.write_bytes(b"")
            config_path = self.write_config(root / "pipeline.yml", input_dir)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(video_path),
                        "--backend",
                        "model_free",
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["config"]["input_type"], "video")

    def test_discover_video_files_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_bytes(b"")
            (root / "b.mpg").write_bytes(b"")
            (root / "c.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual([path.name for path in discover_video_files(root)], ["a.mp4", "b.mpg"])

    def test_safe_video_name_avoids_non_ascii_output_segments(self) -> None:
        safe_name = _safe_video_name(Path("D03_堂食区_修复版.mpg"))

        self.assertTrue(safe_name.startswith("D03_"))
        self.assertTrue(safe_name.isascii())

    def write_config(self, path: Path, input_path: Path) -> Path:
        path.write_text(
            "\n".join(
                [
                    "camera_id: test-video",
                    "input:",
                    "  type: video_dir",
                    f"  path: '{input_path.as_posix()}'",
                    "runtime:",
                    "  frame_stride: 30",
                    "  max_frames: null",
                    "  save_debug_visualization: true",
                    "  save_debug_frames: true",
                    "backend:",
                    "  type: model_free",
                    "modules:",
                    "  person_gate:",
                    "    enabled: true",
                    "    min_score: 0.45",
                    "    skip_when_empty: true",
                    "  pose_hand:",
                    "    enabled: false",
                    "    min_keypoint_score: 0.2",
                    "  item_container:",
                    "    enabled: false",
                    "outputs:",
                    "  frame_jsonl: outputs/frame_results.jsonl",
                    "  event_json: outputs/events.json",
                    "  debug_visualization_dir: outputs/debug_frames",
                    "  debug_visualization_video: outputs/debug_visualization.mp4",
                ]
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
