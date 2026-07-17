from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from shoplift.cli.offline_analyze import (
    BackendOptions,
    ModuleOptions,
    OfflineConfig,
    OutputPaths,
    RuntimeOptions,
    VisionBackendResult,
    main,
    run_offline_analysis,
)
from shoplift.core.types import BodyPose, DetectionBox, HandRegion, Tracklet


class OfflineAnalyzeTest(unittest.TestCase):
    def test_frame_directory_cli_writes_jsonl_events_and_debug_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "frames"
            output_dir = root / "outputs"
            frame_dir.mkdir()
            self.write_image(frame_dir / "frame_000.jpg")
            self.write_image(frame_dir / "frame_001.jpg")
            self.write_image(frame_dir / "frame_002.jpg")
            config_path = self.write_config(root / "pipeline.yml", frame_dir)

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(frame_dir),
                        "--output",
                        str(output_dir),
                        "--max-frames",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            jsonl_path = output_dir / "frame_results.jsonl"
            lines = self.read_jsonl(jsonl_path)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["schema_version"], "shoplift.frame_result.v1")
            self.assertEqual(lines[0]["metadata"]["input_type"], "frame_dir")
            self.assertTrue(lines[0]["person_gate"]["skipped_heavy_modules"])
            self.assertEqual(json.loads((output_dir / "events.json").read_text(encoding="utf-8")), [])
            self.assertEqual(len(list((output_dir / "debug").glob("*.jpg"))), 2)

    def test_backend_results_are_written_to_pose_and_item_container_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            self.write_image(frame_dir / "frame_000.jpg")
            output_dir = root / "outputs"
            config = OfflineConfig(
                camera_id="camera-test",
                input_path=frame_dir,
                input_type=None,
                runtime=RuntimeOptions(max_frames=1, save_debug_visualization=False),
                modules=ModuleOptions(item_container_classes=("product", "handbag")),
                backend=BackendOptions(),
                outputs=OutputPaths(
                    root=output_dir,
                    frame_jsonl=output_dir / "frame_results.jsonl",
                    event_json=output_dir / "events.json",
                    debug_dir=output_dir / "debug",
                    debug_video=output_dir / "debug.mp4",
                ),
            )

            summary = run_offline_analysis(config, backend=FixtureBackend())

            self.assertEqual(summary.processed_frames, 1)
            payload = self.read_jsonl(output_dir / "frame_results.jsonl")[0]
            self.assertFalse(payload["person_gate"]["skipped_heavy_modules"])
            self.assertEqual(payload["body_poses"][0]["person_track_id"], "person-42")
            self.assertEqual(payload["hand_regions"], [])
            self.assertEqual([box["category"] for box in payload["item_container"]["items"]], ["item"])
            self.assertEqual([box["category"] for box in payload["item_container"]["containers"]], ["bag"])
            self.assertEqual(payload["metadata"]["backend"], "fixture")

    def test_video_input_is_processed_when_video_writer_is_available(self) -> None:
        cv2 = self.import_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is not available")
            for _ in range(3):
                writer.write(self.blank_image(width=64, height=48))
            writer.release()

            output_dir = root / "outputs"
            config = OfflineConfig(
                camera_id="camera-video",
                input_path=video_path,
                input_type=None,
                runtime=RuntimeOptions(max_frames=2, save_debug_visualization=False),
                modules=ModuleOptions(),
                backend=BackendOptions(),
                outputs=OutputPaths(
                    root=output_dir,
                    frame_jsonl=output_dir / "frame_results.jsonl",
                    event_json=output_dir / "events.json",
                    debug_dir=output_dir / "debug",
                    debug_video=output_dir / "debug.mp4",
                ),
            )

            summary = run_offline_analysis(config)

            self.assertEqual(summary.input_type, "video")
            self.assertEqual(summary.processed_frames, 2)
            self.assertEqual(len(self.read_jsonl(output_dir / "frame_results.jsonl")), 2)

    def test_video_debug_can_write_sampled_frame_images(self) -> None:
        cv2 = self.import_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                6.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV video writer is not available")
            for _ in range(4):
                writer.write(self.blank_image(width=64, height=48))
            writer.release()

            output_dir = root / "outputs"
            config = OfflineConfig(
                camera_id="camera-video",
                input_path=video_path,
                input_type=None,
                runtime=RuntimeOptions(
                    frame_stride=2,
                    max_frames=2,
                    save_debug_visualization=True,
                    save_debug_frames=True,
                ),
                modules=ModuleOptions(),
                backend=BackendOptions(),
                outputs=OutputPaths(
                    root=output_dir,
                    frame_jsonl=output_dir / "frame_results.jsonl",
                    event_json=output_dir / "events.json",
                    debug_dir=output_dir / "debug",
                    debug_video=output_dir / "debug.mp4",
                ),
            )

            summary = run_offline_analysis(config)

            self.assertEqual(summary.processed_frames, 2)
            self.assertTrue((output_dir / "debug.mp4").exists())
            self.assertEqual(len(list((output_dir / "debug").glob("*.jpg"))), 2)

    def write_config(self, path: Path, input_path: Path) -> Path:
        path.write_text(
            "\n".join(
                [
                    "camera_id: camera-cli",
                    "input:",
                    "  type: frame_dir",
                    f"  path: '{input_path.as_posix()}'",
                    "runtime:",
                    "  frame_stride: 1",
                    "  max_frames: null",
                    "modules:",
                    "  person_gate:",
                    "    enabled: true",
                    "    min_score: 0.45",
                    "    skip_when_empty: true",
                    "  pose_hand:",
                    "    enabled: true",
                    "    min_keypoint_score: 0.2",
                    "  item_container:",
                    "    enabled: true",
                    "    min_score: 0.35",
                    "    classes: [item, product, bag, handbag]",
                    "outputs:",
                    "  frame_jsonl: unused.jsonl",
                    "  event_json: unused-events.json",
                    "  debug_visualization_dir: unused-debug",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def write_image(self, path: Path) -> None:
        cv2 = self.import_cv2()
        self.assertTrue(cv2.imwrite(str(path), self.blank_image()))

    def blank_image(self, *, width: int = 80, height: int = 60):
        cv2 = self.import_cv2()
        return cv2.UMat(height, width, cv2.CV_8UC3).get()

    def import_cv2(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        return cv2

    def read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FixtureBackend:
    def analyze(self, packet) -> VisionBackendResult:
        person_box = DetectionBox(
            box_id="person-box",
            frame_id=packet.frame.frame_id,
            timestamp_ms=packet.frame.timestamp_ms,
            category="person",
            bbox=(5, 5, 30, 50),
            score=0.9,
            track_id="person-42",
        )
        item_box = DetectionBox(
            box_id="product-box",
            frame_id=packet.frame.frame_id,
            timestamp_ms=packet.frame.timestamp_ms,
            category="product",
            bbox=(35, 20, 45, 35),
            score=0.8,
        )
        bag_box = DetectionBox(
            box_id="handbag-box",
            frame_id=packet.frame.frame_id,
            timestamp_ms=packet.frame.timestamp_ms,
            category="handbag",
            bbox=(45, 10, 70, 55),
            score=0.86,
        )
        tracklet = Tracklet(track_id="person-42", category="person", boxes=(person_box,))
        hand = HandRegion(
            hand_track_id="hand-person-42-right",
            person_track_id="person-42",
            frame_id=packet.frame.frame_id,
            timestamp_ms=packet.frame.timestamp_ms,
            side="right",
            bbox=(28, 22, 40, 36),
            score=0.75,
        )
        body_pose = BodyPose(
            pose_id="pose-person-42",
            person_track_id="person-42",
            frame_id=packet.frame.frame_id,
            timestamp_ms=packet.frame.timestamp_ms,
            keypoints=((10, 10), (20, 20)),
            scores=(0.9, 0.8),
            score=0.85,
            keypoint_names=("nose", "left_eye"),
            skeleton_edges=((0, 1),),
            bbox=person_box.bbox,
        )
        return VisionBackendResult(
            detections=(item_box, bag_box),
            person_tracks=(tracklet,),
            body_poses=(body_pose,),
            hand_regions=(hand,),
            metadata={"backend": "fixture"},
        )


if __name__ == "__main__":
    unittest.main()
