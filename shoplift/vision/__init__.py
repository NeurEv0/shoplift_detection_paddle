"""Vision module scaffolds for person, hand, item, and container signals."""

from shoplift.vision.person_gate import (
    PersonGate,
    PersonGateMetrics,
    PersonGateResult,
    evaluate_person_gate,
)

__all__ = [
    "PersonGate",
    "PersonGateMetrics",
    "PersonGateResult",
    "evaluate_person_gate",
]
