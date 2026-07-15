from __future__ import annotations

import json
import unittest

from shoplift.events.event_schema import (
    RISK_EVENT_SCHEMA_PATH,
    assert_valid_risk_event_payload,
    load_risk_event_schema,
    validate_risk_event_payload,
)


EXAMPLE_PATH = RISK_EVENT_SCHEMA_PATH.parent / "examples" / "risk_event.example.json"


class EventSchemaTest(unittest.TestCase):
    def test_schema_and_example_are_loadable(self) -> None:
        schema = load_risk_event_schema()
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["title"], "Shoplift RiskEvent")
        self.assertEqual(validate_risk_event_payload(example), [])

    def test_schema_validator_reports_missing_required_fields(self) -> None:
        errors = validate_risk_event_payload({"event_id": "evt-1"})

        self.assertIn("camera_id is required", errors)
        self.assertIn("evidence is required", errors)

    def test_assert_valid_risk_event_payload_raises_on_invalid_score(self) -> None:
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        example["risk_score"] = 2.0

        with self.assertRaisesRegex(ValueError, "risk_score"):
            assert_valid_risk_event_payload(example)


if __name__ == "__main__":
    unittest.main()
