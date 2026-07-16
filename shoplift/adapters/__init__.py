"""Adapters around upstream vision frameworks."""

from shoplift.adapters.paddledet_adapter import (
    DEFAULT_CATEGORY_ALIASES,
    DEFAULT_CLASS_ID_TO_CATEGORY,
    SHOPLIFT_CLASS_ID_TO_CATEGORY,
    PaddleDetectionAdapter,
    PaddleDetectionEnvironment,
    PaddleDetectionFrameResult,
    ensure_ppdet_path,
)

__all__ = [
    "DEFAULT_CATEGORY_ALIASES",
    "DEFAULT_CLASS_ID_TO_CATEGORY",
    "SHOPLIFT_CLASS_ID_TO_CATEGORY",
    "PaddleDetectionAdapter",
    "PaddleDetectionEnvironment",
    "PaddleDetectionFrameResult",
    "ensure_ppdet_path",
]
