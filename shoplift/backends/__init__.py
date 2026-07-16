"""Vision backends used by the offline shoplift pipeline."""

from shoplift.backends.paddledet_pphuman_backend import (
    PaddleDetPPHumanBackend,
    PaddleDetPPHumanBackendConfig,
)

__all__ = [
    "PaddleDetPPHumanBackend",
    "PaddleDetPPHumanBackendConfig",
]
