from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.container_det_groundingdino_pipeline import (
    DetectionCandidate,
    PredictionRecord,
    build_prompt,
    canonical_label_from_phrase,
    detections_from_groundingdino_outputs,
    label_studio_task,
    load_label_studio_records,
    materialize_coco_dataset,
)


class ContainerDetGroundingDINOPipelineTest(unittest.TestCase):
    labels = (
        "bag",
        "backpack",
        "handbag",
        "suitcase",
        "basket",
        "cart",
        "plastic_bag",
        "stroller",
        "helmet",
    )

    def test_prompt_and_phrase_mapping_cover_container_aliases(self) -> None:
        prompt = build_prompt(self.labels)

        self.assertIn("shopping cart", prompt)
        self.assertIn("plastic bag", prompt)
        self.assertEqual(canonical_label_from_phrase("a shopping trolley", self.labels), "cart")
        self.assertEqual(canonical_label_from_phrase("transparent plastic bag", self.labels), "plastic_bag")
        self.assertIsNone(canonical_label_from_phrase("person", self.labels))

    def test_groundingdino_outputs_convert_to_pixel_boxes_and_nms(self) -> None:
        detections, skipped_unknown, skipped_small = detections_from_groundingdino_outputs(
            boxes=[
                [0.5, 0.5, 0.5, 0.5],
                [0.51, 0.51, 0.5, 0.5],
                [0.1, 0.1, 0.02, 0.02],
                [0.7, 0.7, 0.2, 0.2],
            ],
            logits=[0.91, 0.62, 0.88, 0.8],
            phrases=["shopping basket", "basket", "helmet", "person"],
            width=200,
            height=100,
            labels=self.labels,
            min_area=20,
            nms_iou=0.5,
        )

        self.assertEqual(skipped_unknown, 1)
        self.assertEqual(skipped_small, 1)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "basket")
        self.assertEqual(detections[0].bbox, (50.0, 25.0, 100.0, 50.0))

    def test_label_studio_task_uses_percent_rectangles(self) -> None:
        record = PredictionRecord(
            image_path="datasets/container_det/images/full/frame.jpg",
            source_image=Path("frame.jpg"),
            width=200,
            height=100,
            detections=(
                DetectionCandidate(
                    label="backpack",
                    score=0.75,
                    phrase="backpack",
                    bbox=(20, 10, 40, 50),
                    bbox_xyxy=(20, 10, 60, 60),
                ),
            ),
        )

        task = label_studio_task(record)
        result = task["predictions"][0]["result"][0]  # type: ignore[index]

        self.assertEqual(task["data"]["image"], "/data/local-files/?d=datasets/container_det/images/full/frame.jpg")  # type: ignore[index]
        self.assertEqual(result["value"]["rectanglelabels"], ["backpack"])  # type: ignore[index]
        self.assertEqual(result["value"]["x"], 10.0)  # type: ignore[index]
        self.assertEqual(result["value"]["height"], 50.0)  # type: ignore[index]

    def test_label_studio_export_materializes_paddledetection_coco(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images" / "frame.jpg"
            self.write_image(image_path)
            export_path = root / "label_studio_export.json"
            export_path.write_text(
                json.dumps(
                    [
                        {
                            "data": {
                                "image": f"/data/local-files/?d={image_path.as_posix()}",
                                "source_image": str(image_path),
                            },
                            "annotations": [
                                {
                                    "result": [
                                        {
                                            "type": "rectanglelabels",
                                            "from_name": "label",
                                            "to_name": "image",
                                            "original_width": 200,
                                            "original_height": 100,
                                            "value": {
                                                "x": 10,
                                                "y": 20,
                                                "width": 30,
                                                "height": 40,
                                                "rectanglelabels": ["bag"],
                                            },
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = load_label_studio_records(
                export_path,
                labels=self.labels,
                image_root=None,
                min_score=0.0,
                use_predictions_when_unreviewed=False,
            )
            summary = materialize_coco_dataset(
                records,
                root / "coco",
                labels=self.labels,
                val_ratio=0.0,
                test_ratio=0.0,
                include_empty=True,
                overwrite=False,
            )

            coco = json.loads((root / "coco" / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.selected_record_count, 1)
            self.assertEqual(summary.split_image_counts["train"], 1)
            self.assertEqual(coco["categories"][0]["name"], "bag")
            self.assertEqual(coco["categories"][0]["id"], 0)
            self.assertEqual(coco["annotations"][0]["category_id"], 0)
            self.assertEqual(coco["annotations"][0]["bbox"], [20.0, 20.0, 60.0, 40.0])
            self.assertTrue((root / "coco" / "images" / "train" / "frame.jpg").exists())

    def write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder image bytes")


if __name__ == "__main__":
    unittest.main()
