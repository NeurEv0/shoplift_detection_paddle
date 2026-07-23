"""Item and container detection normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from shoplift.core.types import DetectionBox


DetectionRole = Literal["item", "container", "extension_region"]

ITEM_CATEGORY = "item"
PERSONAL_CONTAINER_CATEGORY = "bag"
NORMAL_CONTAINER_CATEGORY = "basket"
SPECIAL_CONTAINER_CATEGORIES = frozenset({"stroller", "helmet"})
EXTENSION_REGION_CATEGORY = "clothing_region"

ITEM_AND_CONTAINER_ALIASES = {
    "product": ITEM_CATEGORY,
    "backpack": PERSONAL_CONTAINER_CATEGORY,
    "handbag": PERSONAL_CONTAINER_CATEGORY,
    "suitcase": PERSONAL_CONTAINER_CATEGORY,
    "cart": NORMAL_CONTAINER_CATEGORY,
    "plastic_bag": NORMAL_CONTAINER_CATEGORY,
    "pocket_region": EXTENSION_REGION_CATEGORY,
}

ITEM_AND_CONTAINER_CLASSES = frozenset(
    {
        ITEM_CATEGORY,
        "product",
        PERSONAL_CONTAINER_CATEGORY,
        "backpack",
        "handbag",
        "suitcase",
        NORMAL_CONTAINER_CATEGORY,
        "cart",
        "plastic_bag",
        "stroller",
        "helmet",
        EXTENSION_REGION_CATEGORY,
        "pocket_region",
    }
)

CANONICAL_ITEM_CONTAINER_CLASSES = frozenset(
    {
        ITEM_CATEGORY,
        PERSONAL_CONTAINER_CATEGORY,
        NORMAL_CONTAINER_CATEGORY,
        "stroller",
        "helmet",
        EXTENSION_REGION_CATEGORY,
    }
)

ITEM_CATEGORIES = frozenset({ITEM_CATEGORY})
CONTAINER_CATEGORIES = frozenset(
    {
        PERSONAL_CONTAINER_CATEGORY,
        NORMAL_CONTAINER_CATEGORY,
        *SPECIAL_CONTAINER_CATEGORIES,
    }
)
EXTENSION_REGION_CATEGORIES = frozenset({EXTENSION_REGION_CATEGORY})
NORMAL_CONTAINER_CATEGORIES = frozenset({NORMAL_CONTAINER_CATEGORY})


@dataclass(frozen=True)
class ItemContainerResult:
    """Grouped item/container detections for one frame or batch slice."""

    detections: tuple[DetectionBox, ...] = field(default_factory=tuple)
    items: tuple[DetectionBox, ...] = field(default_factory=tuple)
    containers: tuple[DetectionBox, ...] = field(default_factory=tuple)
    extension_regions: tuple[DetectionBox, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [detection.to_dict() for detection in self.detections],
            "items": [detection.to_dict() for detection in self.items],
            "containers": [detection.to_dict() for detection in self.containers],
            "extension_regions": [detection.to_dict() for detection in self.extension_regions],
            "metadata": {
                "detection_count": len(self.detections),
                "item_count": len(self.items),
                "container_count": len(self.containers),
                "extension_region_count": len(self.extension_regions),
                **self.metadata,
            },
        }


@dataclass(frozen=True)
class ItemContainerDetectionAdapter:
    """Filter and normalize coarse product/container detections."""

    min_score: float = 0.35
    allowed_categories: frozenset[str] | None = None
    include_extension_regions: bool = True

    def adapt(self, detections: Sequence[DetectionBox]) -> ItemContainerResult:
        normalized: list[DetectionBox] = []
        items: list[DetectionBox] = []
        containers: list[DetectionBox] = []
        extension_regions: list[DetectionBox] = []
        skipped_low_score = 0
        skipped_unsupported = 0

        for detection in detections:
            normalized_detection = self._normalize_detection(detection)
            if normalized_detection is None:
                if detection.score < self.min_score:
                    skipped_low_score += 1
                else:
                    skipped_unsupported += 1
                continue

            role = normalized_detection.attributes["detection_role"]
            normalized.append(normalized_detection)
            if role == "item":
                items.append(normalized_detection)
            elif role == "container":
                containers.append(normalized_detection)
            elif role == "extension_region":
                extension_regions.append(normalized_detection)

        return ItemContainerResult(
            detections=tuple(normalized),
            items=tuple(items),
            containers=tuple(containers),
            extension_regions=tuple(extension_regions),
            metadata={
                "skipped_low_score": skipped_low_score,
                "skipped_unsupported": skipped_unsupported,
                "allowed_categories": sorted(self._allowed_categories()) if self.allowed_categories else None,
            },
        )

    def _normalize_detection(self, detection: DetectionBox) -> DetectionBox | None:
        if detection.score < self.min_score:
            return None

        source_category = detection.category
        canonical_category = canonical_item_container_category(source_category)
        if canonical_category not in CANONICAL_ITEM_CONTAINER_CLASSES:
            return None
        if canonical_category not in self._allowed_categories():
            return None
        role = detection_role(canonical_category)
        if role == "extension_region" and not self.include_extension_regions:
            return None

        return DetectionBox(
            box_id=detection.box_id,
            frame_id=detection.frame_id,
            category=canonical_category,
            bbox=detection.bbox,
            score=detection.score,
            track_id=detection.track_id,
            timestamp_ms=detection.timestamp_ms,
            attributes={
                **detection.attributes,
                "source_category": detection.attributes.get("source_category", source_category),
                "canonical_category": canonical_category,
                "detection_role": role,
                "is_normal_container": canonical_category in NORMAL_CONTAINER_CATEGORIES,
            },
        )

    def _allowed_categories(self) -> frozenset[str]:
        if self.allowed_categories is None:
            return CANONICAL_ITEM_CONTAINER_CLASSES
        return frozenset(
            canonical_category
            for category in self.allowed_categories
            if (canonical_category := canonical_item_container_category(category))
            in CANONICAL_ITEM_CONTAINER_CLASSES
        )


def canonical_item_container_category(category: str) -> str:
    normalized = str(category).strip().lower()
    return ITEM_AND_CONTAINER_ALIASES.get(normalized, normalized)


def detection_role(category: str) -> DetectionRole:
    canonical_category = canonical_item_container_category(category)
    if canonical_category in ITEM_CATEGORIES:
        return "item"
    if canonical_category in CONTAINER_CATEGORIES:
        return "container"
    if canonical_category in EXTENSION_REGION_CATEGORIES:
        return "extension_region"
    raise ValueError(f"unsupported item/container category: {category}")


def adapt_item_container_detections(
    detections: Sequence[DetectionBox],
    *,
    min_score: float = 0.35,
    allowed_categories: Sequence[str] | None = None,
    include_extension_regions: bool = True,
) -> ItemContainerResult:
    adapter = ItemContainerDetectionAdapter(
        min_score=min_score,
        allowed_categories=frozenset(allowed_categories) if allowed_categories is not None else None,
        include_extension_regions=include_extension_regions,
    )
    return adapter.adapt(detections)


__all__ = [
    "CANONICAL_ITEM_CONTAINER_CLASSES",
    "CONTAINER_CATEGORIES",
    "DetectionBox",
    "DetectionRole",
    "EXTENSION_REGION_CATEGORIES",
    "ITEM_AND_CONTAINER_ALIASES",
    "ITEM_AND_CONTAINER_CLASSES",
    "ITEM_CATEGORIES",
    "ItemContainerDetectionAdapter",
    "ItemContainerResult",
    "NORMAL_CONTAINER_CATEGORIES",
    "adapt_item_container_detections",
    "canonical_item_container_category",
    "detection_role",
]
