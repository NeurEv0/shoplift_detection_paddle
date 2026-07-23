from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_person_attribute_dataset import (
    collect_videos,
    load_jsonl_candidates,
    prepare_frame_dataset,
    prepare_dataset,
)


class PreparePersonAttributeDatasetTest(unittest.TestCase):
    def test_jsonl_candidates_are_loaded_from_person_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path = root / "frame_results.jsonl"
            self.write_frame_jsonl(jsonl_path, "sample.mp4")

            index = load_jsonl_candidates(jsonl_path)

            self.assertIn("sample.mp4", index.by_source_uri)
            self.assertEqual(index.by_source_uri["sample.mp4"][0][0].person_track_id, "person-7")
            self.assertEqual(index.by_source_uri["sample.mp4"][0][0].bbox, (10.0, 8.0, 34.0, 42.0))

    def test_prepare_dataset_writes_crops_and_annotation_template(self) -> None:
        cv2 = self.import_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            self.write_video(video_path, cv2)
            jsonl_path = root / "frame_results.jsonl"
            self.write_frame_jsonl(jsonl_path, str(video_path))

            output_dir = root / "dataset"
            summary = prepare_dataset(
                video_path,
                output_dir,
                frame_jsonl=jsonl_path,
                frame_stride=1,
                padding_ratio=0.0,
            )

            rows = self.read_csv(output_dir / "train.csv")
            self.assertEqual(summary.video_count, 1)
            self.assertEqual(summary.sampled_frame_count, 2)
            self.assertEqual(summary.crop_count, 2)
            self.assertEqual(rows[0]["image_path"], "train/sample_f000000_person-7_00.jpg")
            self.assertEqual(rows[0]["left_hand_state"], "uncertain")
            self.assertEqual(rows[0]["left_hand_visibility"], "not_judgable")
            self.assertEqual(rows[0]["label_status"], "unreviewed")
            self.assertEqual(rows[0]["crop_source"], "person_track")
            self.assertEqual(rows[1]["crop_source"], "full_frame")

            first_crop = cv2.imread(str(output_dir / "images" / rows[0]["image_path"]))
            second_crop = cv2.imread(str(output_dir / "images" / rows[1]["image_path"]))
            self.assertEqual(first_crop.shape[:2], (34, 24))
            self.assertEqual(second_crop.shape[:2], (48, 64))

    def test_prepare_frame_dataset_uses_source_annotation_and_jsonl_boxes(self) -> None:
        cv2 = self.import_cv2()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dataset = root / "source_dataset"
            source_image_dir = source_dataset / "images" / "full"
            source_image_dir.mkdir(parents=True)
            image_path = source_image_dir / "frame_000.jpg"
            self.write_image(image_path, cv2, width=64, height=48)
            annotation_path = source_dataset / "full.csv"
            self.write_full_frame_annotation(annotation_path)
            jsonl_path = root / "frame_results.jsonl"
            self.write_frame_jsonl(jsonl_path, str(image_path))

            output_dir = root / "cropped_dataset"
            summary = prepare_frame_dataset(
                source_dataset / "images",
                output_dir,
                split="full",
                frame_jsonl=jsonl_path,
                source_annotation=annotation_path,
                padding_ratio=0.0,
            )

            rows = self.read_csv(output_dir / "full.csv")
            self.assertEqual(summary.video_count, 0)
            self.assertEqual(summary.sampled_frame_count, 1)
            self.assertEqual(summary.crop_count, 1)
            self.assertEqual(rows[0]["crop_source"], "person_track")
            self.assertEqual(rows[0]["source_image"], str(image_path))
            self.assertEqual(rows[0]["source_video"], "sample.mp4")
            crop = cv2.imread(str(output_dir / "images" / rows[0]["image_path"]))
            self.assertEqual(crop.shape[:2], (34, 24))

    def test_collect_videos_accepts_single_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"placeholder")

            self.assertEqual(collect_videos(video_path), [video_path])

    def write_frame_jsonl(self, path: Path, source_uri: str) -> None:
        payload = {
            "schema_version": "shoplift.frame_result.v1",
            "frame": {
                "frame_id": 0,
                "timestamp_ms": 0,
                "camera_id": "camera-test",
                "width": 64,
                "height": 48,
                "source_uri": source_uri,
            },
            "person_tracks": [
                {
                    "track_id": "person-7",
                    "category": "person",
                    "boxes": [
                        {
                            "box_id": "person-7-0",
                            "frame_id": 0,
                            "category": "person",
                            "bbox": [10, 8, 34, 42],
                            "score": 0.9,
                            "track_id": "person-7",
                        }
                    ],
                }
            ],
            "metadata": {"source_uri": source_uri},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_video(self, path: Path, cv2) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (64, 48),
        )
        if not writer.isOpened():
            self.skipTest("OpenCV video writer is not available")
        for index in range(2):
            frame = self.blank_image(cv2, width=64, height=48, value=40 + index * 50)
            writer.write(frame)
        writer.release()

    def write_image(self, path: Path, cv2, *, width: int = 64, height: int = 48) -> None:
        image = self.blank_image(cv2, width=width, height=height, value=80)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())

    def write_full_frame_annotation(self, path: Path) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_path",
                    "left_hand_state",
                    "left_hand_visibility",
                    "right_hand_state",
                    "right_hand_visibility",
                    "body_orientation",
                    "occlusion_level",
                    "label_status",
                    "source_video",
                    "frame_id",
                    "timestamp_ms",
                    "person_track_id",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "crop_source",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_path": "full/frame_000.jpg",
                    "left_hand_state": "uncertain",
                    "left_hand_visibility": "not_judgable",
                    "right_hand_state": "uncertain",
                    "right_hand_visibility": "not_judgable",
                    "body_orientation": "unknown",
                    "occlusion_level": "heavy",
                    "label_status": "unreviewed",
                    "source_video": "sample.mp4",
                    "frame_id": "0",
                    "timestamp_ms": "0",
                    "person_track_id": "unassigned",
                    "bbox_x1": "0",
                    "bbox_y1": "0",
                    "bbox_x2": "64",
                    "bbox_y2": "48",
                    "crop_source": "full_frame",
                }
            )

    def blank_image(self, cv2, *, width: int, height: int, value: int):
        import numpy as np

        return np.full((height, width, 3), value, dtype=np.uint8)

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    def import_cv2(self):
        try:
            import cv2
        except ModuleNotFoundError:
            self.skipTest("OpenCV is not installed")
        return cv2


if __name__ == "__main__":
    unittest.main()
