"""Adapters around upstream vision frameworks."""

from shoplift.adapters.paddledet_adapter import PaddleDetectionEnvironment, ensure_ppdet_path

__all__ = ["PaddleDetectionEnvironment", "ensure_ppdet_path"]
