"""Vision modules for person, hand, item, and container signals."""

from shoplift.vision.object_container import (
    CANONICAL_ITEM_CONTAINER_CLASSES,
    CONTAINER_CATEGORIES,
    EXTENSION_REGION_CATEGORIES,
    ITEM_AND_CONTAINER_ALIASES,
    ITEM_AND_CONTAINER_CLASSES,
    ITEM_CATEGORIES,
    NORMAL_CONTAINER_CATEGORIES,
    ItemContainerDetectionAdapter,
    ItemContainerResult,
    adapt_item_container_detections,
    canonical_item_container_category,
    detection_role,
)
from shoplift.vision.person_gate import (
    PersonGate,
    PersonGateMetrics,
    PersonGateResult,
    evaluate_person_gate,
)
from shoplift.vision.pose_hand import (
    COCO_KEYPOINT_INDICES,
    HandRegionExtractor,
    PersonPose,
    build_person_poses,
    extract_hand_regions,
)

__all__ = [
    "CANONICAL_ITEM_CONTAINER_CLASSES",
    "COCO_KEYPOINT_INDICES",
    "CONTAINER_CATEGORIES",
    "EXTENSION_REGION_CATEGORIES",
    "HandRegionExtractor",
    "ITEM_AND_CONTAINER_ALIASES",
    "ITEM_AND_CONTAINER_CLASSES",
    "ITEM_CATEGORIES",
    "ItemContainerDetectionAdapter",
    "ItemContainerResult",
    "NORMAL_CONTAINER_CATEGORIES",
    "PersonGate",
    "PersonGateMetrics",
    "PersonGateResult",
    "PersonPose",
    "adapt_item_container_detections",
    "build_person_poses",
    "canonical_item_container_category",
    "detection_role",
    "evaluate_person_gate",
    "extract_hand_regions",
]
