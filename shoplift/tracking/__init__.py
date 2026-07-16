"""Tracking and association primitives."""

from shoplift.tracking.association import (
    AssociationConfig,
    AssociationFrame,
    AssociationResult,
    ContainerEntryDetector,
    DisappearanceAfterEntryDetector,
    HandItemContactAssociator,
    ItemFollowPersonAssociator,
    ShopliftingRelationAssociator,
    TrackedDetection,
)
from shoplift.tracking.track_types import (
    DetectionBox,
    FrameMeta,
    HandRegion,
    RelationEvidence,
    RiskEvent,
    Tracklet,
)

__all__ = [
    "AssociationConfig",
    "AssociationFrame",
    "AssociationResult",
    "ContainerEntryDetector",
    "DetectionBox",
    "DisappearanceAfterEntryDetector",
    "FrameMeta",
    "HandItemContactAssociator",
    "HandRegion",
    "ItemFollowPersonAssociator",
    "RelationEvidence",
    "RiskEvent",
    "ShopliftingRelationAssociator",
    "Tracklet",
    "TrackedDetection",
]
