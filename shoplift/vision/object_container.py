"""Item and container detection scaffold."""

from shoplift.core.types import DetectionBox

ITEM_AND_CONTAINER_CLASSES = {
    "item",
    "product",
    "bag",
    "backpack",
    "handbag",
    "basket",
    "cart",
    "stroller",
    "helmet",
    "clothing_region",
    "pocket_region",
}

__all__ = ["DetectionBox", "ITEM_AND_CONTAINER_CLASSES"]
