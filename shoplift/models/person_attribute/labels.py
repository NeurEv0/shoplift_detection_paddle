"""Label definitions for the shoplift person-attribute model."""

from __future__ import annotations

from dataclasses import dataclass


HAND_STATE_LABELS = ("empty", "holding_object", "holding_product", "uncertain")
HAND_VISIBILITY_LABELS = ("clear", "partial_occluded", "not_judgable")
BODY_ORIENTATION_LABELS = ("front", "side", "back", "unknown")
OCCLUSION_LEVEL_LABELS = ("none", "light", "heavy")


@dataclass(frozen=True)
class HeadSpec:
    name: str
    labels: tuple[str, ...]
    loss_weight: float = 1.0

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def label_to_index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise ValueError(f"unsupported label for {self.name}: {label}") from exc


HEAD_SPECS = (
    HeadSpec("left_hand_state", HAND_STATE_LABELS, 2.0),
    HeadSpec("left_hand_visibility", HAND_VISIBILITY_LABELS, 1.5),
    HeadSpec("right_hand_state", HAND_STATE_LABELS, 2.0),
    HeadSpec("right_hand_visibility", HAND_VISIBILITY_LABELS, 1.5),
    HeadSpec("body_orientation", BODY_ORIENTATION_LABELS, 1.0),
    HeadSpec("occlusion_level", OCCLUSION_LEVEL_LABELS, 1.0),
)

HEAD_SPECS_BY_NAME = {spec.name: spec for spec in HEAD_SPECS}


def total_output_dim() -> int:
    return sum(spec.num_classes for spec in HEAD_SPECS)


__all__ = [
    "BODY_ORIENTATION_LABELS",
    "HAND_STATE_LABELS",
    "HAND_VISIBILITY_LABELS",
    "OCCLUSION_LEVEL_LABELS",
    "HEAD_SPECS",
    "HEAD_SPECS_BY_NAME",
    "HeadSpec",
    "total_output_dim",
]

