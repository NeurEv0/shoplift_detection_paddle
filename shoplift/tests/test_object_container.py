from __future__ import annotations

import unittest

from shoplift.core.types import DetectionBox
from shoplift.vision.object_container import (
    ItemContainerDetectionAdapter,
    adapt_item_container_detections,
    canonical_item_container_category,
    detection_role,
)


class ObjectContainerTest(unittest.TestCase):
    def box(self, box_id: str, category: str, score: float = 0.8) -> DetectionBox:
        return DetectionBox(
            box_id=box_id,
            frame_id=2,
            timestamp_ms=66,
            category=category,
            bbox=(10, 20, 50, 80),
            score=score,
        )

    def test_aliases_are_normalized_and_grouped(self) -> None:
        result = adapt_item_container_detections(
            [
                self.box("product-1", "product"),
                self.box("backpack-1", "backpack"),
                self.box("cart-1", "cart"),
                self.box("plastic-bag-1", "plastic_bag"),
                self.box("stroller-1", "stroller"),
                self.box("helmet-1", "helmet"),
                self.box("pocket-1", "pocket_region"),
            ],
            min_score=0.35,
        )

        self.assertEqual([box.category for box in result.items], ["item"])
        self.assertEqual([box.category for box in result.containers], ["bag", "basket", "basket", "stroller", "helmet"])
        self.assertEqual([box.category for box in result.extension_regions], ["clothing_region"])
        self.assertTrue(result.containers[1].attributes["is_normal_container"])
        self.assertTrue(result.containers[2].attributes["is_normal_container"])
        self.assertEqual(result.containers[0].attributes["source_category"], "backpack")
        self.assertEqual(result.to_dict()["metadata"]["container_count"], 5)

    def test_low_score_and_unsupported_categories_are_skipped(self) -> None:
        result = adapt_item_container_detections(
            [
                self.box("item-low", "item", 0.2),
                self.box("person-1", "person", 0.95),
                self.box("bag-1", "bag", 0.7),
            ],
            min_score=0.35,
        )

        self.assertEqual([box.box_id for box in result.detections], ["bag-1"])
        self.assertEqual(result.metadata["skipped_low_score"], 1)
        self.assertEqual(result.metadata["skipped_unsupported"], 1)

    def test_allowed_categories_accept_aliases(self) -> None:
        adapter = ItemContainerDetectionAdapter(
            min_score=0.35,
            allowed_categories=frozenset({"product", "handbag"}),
        )

        result = adapter.adapt(
            [
                self.box("product-1", "product"),
                self.box("handbag-1", "handbag"),
                self.box("cart-1", "cart"),
            ]
        )

        self.assertEqual([box.category for box in result.detections], ["item", "bag"])
        self.assertEqual(result.metadata["allowed_categories"], ["bag", "item"])

    def test_extension_regions_can_be_excluded(self) -> None:
        result = adapt_item_container_detections(
            [self.box("pocket-1", "pocket_region")],
            include_extension_regions=False,
        )

        self.assertEqual(result.detections, ())
        self.assertEqual(result.metadata["skipped_unsupported"], 1)

    def test_category_helpers(self) -> None:
        self.assertEqual(canonical_item_container_category("HandBag"), "bag")
        self.assertEqual(canonical_item_container_category("plastic_bag"), "basket")
        self.assertEqual(detection_role("cart"), "container")
        self.assertEqual(detection_role("pocket_region"), "extension_region")


if __name__ == "__main__":
    unittest.main()
