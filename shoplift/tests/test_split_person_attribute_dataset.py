from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.split_person_attribute_dataset import split_labeled_dataset


class SplitPersonAttributeDatasetTest(unittest.TestCase):
    def test_split_uses_reviewed_rows_and_copies_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dataset = root / "source"
            self.write_source_dataset(
                source_dataset,
                [
                    self.row("full/person_a.jpg", "reviewed", "front"),
                    self.row("full/person_b.jpg", "reviewed", "side"),
                    self.row("full/person_c.jpg", "unreviewed", "back"),
                ],
            )
            output_dir = root / "splits"

            summary = split_labeled_dataset(
                source_dataset,
                output_dir,
                val_ratio=0.5,
                seed=7,
            )

            train_rows = self.read_csv(output_dir / "train.csv")
            val_rows = self.read_csv(output_dir / "val.csv")
            output_rows = [*train_rows, *val_rows]
            self.assertEqual(summary.source_row_count, 3)
            self.assertEqual(summary.selected_row_count, 2)
            self.assertEqual(summary.train_count, 1)
            self.assertEqual(summary.val_count, 1)
            self.assertEqual(summary.copied_image_count, 2)
            self.assertEqual({row["label_status"] for row in output_rows}, {"reviewed"})
            self.assertEqual(
                {Path(row["image_path"]).parts[0] for row in output_rows},
                {"train", "val"},
            )
            self.assertFalse(any("person_c" in row["image_path"] for row in output_rows))
            for row in output_rows:
                self.assertTrue((output_dir / "images" / row["image_path"]).exists())

    def test_split_rejects_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dataset = root / "source"
            bad_row = self.row("full/person_a.jpg", "reviewed", "front")
            bad_row["left_hand_state"] = "bad_label"
            self.write_source_dataset(source_dataset, [bad_row])

            with self.assertRaisesRegex(ValueError, "unsupported label"):
                split_labeled_dataset(source_dataset, root / "splits")

    def test_split_rejects_missing_images_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dataset = root / "source"
            self.write_source_dataset(
                source_dataset,
                [self.row("full/missing.jpg", "reviewed", "front")],
                write_images=False,
            )

            with self.assertRaisesRegex(FileNotFoundError, "image does not exist"):
                split_labeled_dataset(source_dataset, root / "splits")

    def row(self, image_path: str, status: str, orientation: str) -> dict[str, str]:
        return {
            "image_path": image_path,
            "left_hand_state": "empty",
            "left_hand_visibility": "clear",
            "right_hand_state": "holding_object",
            "right_hand_visibility": "partial_occluded",
            "body_orientation": orientation,
            "occlusion_level": "light",
            "label_status": status,
            "source_video": "sample.mp4",
            "frame_id": "1",
        }

    def write_source_dataset(
        self,
        source_dataset: Path,
        rows: list[dict[str, str]],
        *,
        write_images: bool = True,
    ) -> None:
        annotation_path = source_dataset / "full.csv"
        image_root = source_dataset / "images"
        image_root.mkdir(parents=True)
        fieldnames = list(rows[0])
        with annotation_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if write_images:
            for row in rows:
                image_path = image_root / row["image_path"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"fake image bytes")

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
