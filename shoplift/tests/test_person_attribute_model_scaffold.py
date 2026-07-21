from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shoplift.models.person_attribute.config import load_train_config
from shoplift.models.person_attribute.dataset import read_annotation
from shoplift.models.person_attribute.labels import HEAD_SPECS, total_output_dim


class PersonAttributeModelScaffoldTest(unittest.TestCase):
    def test_label_heads_match_backend_output_contract(self) -> None:
        self.assertEqual([spec.num_classes for spec in HEAD_SPECS], [4, 3, 4, 3, 4, 3])
        self.assertEqual(total_output_dim(), 21)

    def test_read_annotation_maps_labels_to_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "train.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "image_path,left_hand_state,left_hand_visibility,right_hand_state,right_hand_visibility,body_orientation,occlusion_level",
                        "img.jpg,holding_product,clear,empty,partial_occluded,side,light",
                    ]
                ),
                encoding="utf-8",
            )

            samples = read_annotation(csv_path, image_root=root / "images")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].labels["left_hand_state"], 2)
            self.assertEqual(samples[0].labels["right_hand_visibility"], 1)
            self.assertEqual(samples[0].image_path, root / "images" / "img.jpg")

    def test_load_train_config_resolves_model_training_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "config.yml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "output_dir: ./outputs/person_attribute",
                        "device: cpu",
                        "backbone:",
                        "  name: tiny_cnn",
                        "  arch: smoke",
                        "data:",
                        "  image_root: ./images",
                        "  train_annotation: ./train.csv",
                        "optimizer:",
                        "  epochs: 2",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_train_config(cfg_path)

            self.assertEqual(config.backbone.name, "tiny_cnn")
            self.assertEqual(config.data.train_annotation, Path("./train.csv"))
            self.assertEqual(config.optimizer.epochs, 2)


if __name__ == "__main__":
    unittest.main()

