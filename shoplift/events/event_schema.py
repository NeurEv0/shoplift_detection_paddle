"""Risk event JSON schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shoplift.core.types import RiskEvent


SCHEMA_VERSION = "shoplift.risk_event.v1"
RISK_EVENT_SCHEMA_PATH = Path(__file__).with_name("risk_event.schema.json")
RISK_LEVELS = {"low", "medium", "high"}


def load_risk_event_schema(path: Path | None = None) -> dict[str, Any]:
    schema_path = path or RISK_EVENT_SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def risk_event_to_payload(event: RiskEvent) -> dict[str, Any]:
    payload = event.to_dict()
    payload.setdefault("schema_version", SCHEMA_VERSION)
    return payload


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_string(payload: dict[str, Any], field: str, errors: list[str]) -> None:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")


def _check_non_negative_int(payload: dict[str, Any], field: str, errors: list[str]) -> None:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{field} must be a non-negative integer")


def _check_score(payload: dict[str, Any], field: str, errors: list[str]) -> None:
    value = payload.get(field)
    if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
        errors.append(f"{field} must be a number between 0.0 and 1.0")


def _check_tags(payload: dict[str, Any], field: str, errors: list[str]) -> None:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty string array")
        return
    for tag in value:
        if not isinstance(tag, str) or not tag:
            errors.append(f"{field} must only contain non-empty strings")
            return


def _check_bbox(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 4:
        errors.append(f"{field} must be a four-number bbox")
        return
    if not all(_is_number(item) for item in value):
        errors.append(f"{field} must only contain numbers")
        return
    x1, y1, x2, y2 = [float(item) for item in value]
    if x2 < x1 or y2 < y1:
        errors.append(f"{field} must be ordered as [x1, y1, x2, y2]")


def _validate_evidence_item(item: Any, index: int, errors: list[str]) -> None:
    prefix = f"evidence[{index}]"
    if not isinstance(item, dict):
        errors.append(f"{prefix} must be an object")
        return
    _check_string(item, "relation_type", errors)
    _check_non_negative_int(item, "frame_id", errors)
    _check_non_negative_int(item, "timestamp_ms", errors)
    _check_score(item, "score", errors)
    _check_tags(item, "reason_tags", errors)
    evidence_boxes = item.get("evidence_boxes", {})
    if evidence_boxes is not None:
        if not isinstance(evidence_boxes, dict):
            errors.append(f"{prefix}.evidence_boxes must be an object")
        else:
            for name, bbox in evidence_boxes.items():
                _check_bbox(bbox, f"{prefix}.evidence_boxes.{name}", errors)


def validate_risk_event_payload(
    payload: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> list[str]:
    """Validate the P0 RiskEvent contract without requiring jsonschema."""

    del schema  # The file remains the source of truth; this is a small subset.
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be an object"]

    required = (
        "schema_version",
        "event_id",
        "camera_id",
        "timestamp_ms",
        "person_track_id",
        "event_type",
        "risk_score",
        "risk_level",
        "reason_tags",
        "evidence",
    )
    for field in required:
        if field not in payload:
            errors.append(f"{field} is required")

    if errors:
        return errors

    _check_string(payload, "schema_version", errors)
    _check_string(payload, "event_id", errors)
    _check_string(payload, "camera_id", errors)
    _check_non_negative_int(payload, "timestamp_ms", errors)
    _check_string(payload, "person_track_id", errors)
    _check_string(payload, "event_type", errors)
    _check_score(payload, "risk_score", errors)
    if payload.get("risk_level") not in RISK_LEVELS:
        errors.append("risk_level must be one of: low, medium, high")
    _check_tags(payload, "reason_tags", errors)

    for optional_int in ("start_timestamp_ms", "end_timestamp_ms"):
        if optional_int in payload and payload[optional_int] is not None:
            _check_non_negative_int(payload, optional_int, errors)
    if (
        isinstance(payload.get("start_timestamp_ms"), int)
        and isinstance(payload.get("end_timestamp_ms"), int)
        and payload["end_timestamp_ms"] < payload["start_timestamp_ms"]
    ):
        errors.append("end_timestamp_ms must be greater than or equal to start_timestamp_ms")

    if "confidence" in payload and payload["confidence"] is not None:
        _check_score(payload, "confidence", errors)

    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must be a non-empty array")
    else:
        for index, item in enumerate(evidence):
            _validate_evidence_item(item, index, errors)

    return errors


def assert_valid_risk_event_payload(payload: dict[str, Any]) -> None:
    errors = validate_risk_event_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
