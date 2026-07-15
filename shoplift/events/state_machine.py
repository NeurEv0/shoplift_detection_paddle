"""Event state-machine placeholders for P1."""

EVENT_STATES = (
    "observing",
    "item_picked",
    "near_body_or_container",
    "suspected_concealment",
    "confirmed_risk_event",
    "resolved_or_downgraded",
)

__all__ = ["EVENT_STATES"]
