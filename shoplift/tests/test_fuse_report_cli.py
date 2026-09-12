"""Integration tests for the PP-TSM open × pipeline fusion report CLI.

Exercises the pure path (``--open-events-json``): reads pipeline events +
frame_results, fuses pre-computed open events through the product gate, and
writes fused events — no GPU / PaddleVideo / weights needed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shoplift.cli import fuse_report
from shoplift.core.types import RelationEvidence, RiskEvent
from shoplift.events.event_schema import risk_event_to_payload
from shoplift.pptsm_open.types import OpenPackagingEvent, OpenWindow


def _pipeline_event(
    person: str,
    start_ms: int,
    end_ms: int,
    level: str = "medium",
    score: float = 0.5,
    event_type: str = "bag_concealment",
) -> dict:
    relation = RelationEvidence(
        relation_type="item_enter_container",
        frame_id=0,
        timestamp_ms=start_ms,
        score=0.6,
        reason_tags=("entered_private_container",),
        person_track_id=person,
    )
    event = RiskEvent(
        event_id=f"evt-cam-{person}-{event_type}-{start_ms}",
        camera_id="cam",
        timestamp_ms=end_ms,
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
        person_track_id=person,
        event_type=event_type,
        risk_score=score,
        risk_level=level,  # type: ignore[arg-type]
        reason_tags=("entered_private_container",),
        evidence=(relation,),
    )
    return risk_event_to_payload(event)


def _open_event(person: str) -> dict:
    windows = tuple(
        OpenWindow(start_timestamp_ms=1000 + index * 100,
                   end_timestamp_ms=1100 + index * 100,
                   open_prob=0.7 + index * 0.1,
                   frame_id=index)
        for index in range(3)
    )
    return OpenPackagingEvent("cam", person, windows).to_dict()


def _frame_line(frame_id: int, timestamp_ms: int, holding: list[str]) -> dict:
    """One frame_results line; persons in ``holding`` are holding_product."""
    person_ids = ["person-1", "person-2", "person-3"]
    default_bboxes = {
        "person-1": [10.0, 10.0, 110.0, 210.0],
        "person-2": [400.0, 10.0, 500.0, 210.0],
        "person-3": [800.0, 10.0, 900.0, 210.0],
    }
    person_tracks = []
    attributes = []
    for person in person_ids:
        bbox = default_bboxes[person]
        person_tracks.append({
            "track_id": person,
            "category": "person",
            "boxes": [{
                "box_id": f"{person}-{frame_id}",
                "frame_id": frame_id,
                "category": "person",
                "bbox": bbox,
                "score": 0.9,
                "track_id": person,
                "timestamp_ms": timestamp_ms,
            }],
            "start_frame_id": frame_id,
            "end_frame_id": frame_id,
            "timestamps_ms": [timestamp_ms],
            "metadata": {},
        })
        holding_label = "holding_product" if person in holding else "empty"
        holding_score = 0.9 if person in holding else 0.9
        attributes.append({
            "attribute_id": f"{person}-{frame_id}",
            "person_track_id": person,
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "bbox": bbox,
            "left_hand_state": {"label": holding_label, "score": holding_score},
            "left_hand_visibility": {"label": "clear", "score": 0.9},
            "right_hand_state": {"label": "empty", "score": 0.9},
            "right_hand_visibility": {"label": "clear", "score": 0.9},
            "body_orientation": {"label": "front", "score": 0.9},
            "occlusion_level": {"label": "none", "score": 0.9},
            "metadata": {},
        })
    return {
        "schema_version": "shoplift.frame_result.v1",
        "frame": {"frame_id": frame_id, "timestamp_ms": timestamp_ms,
                  "camera_id": "cam", "width": 1280, "height": 720, "source_uri": None},
        "person_gate": {},
        "person_tracks": person_tracks,
        "body_poses": [],
        "hand_regions": [],
        "person_attributes": attributes,
        "proxy_item_regions": [],
        "item_container": {},
        "metadata": {},
    }


class ReadFrameResultsTest(unittest.TestCase):
    def test_parses_boxes_and_product_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_results.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(_frame_line(0, 0, ["person-1"])) + "\n")
                handle.write(json.dumps(_frame_line(1, 40, ["person-1"])) + "\n")
                handle.write(json.dumps(_frame_line(2, 80, ["person-1"])) + "\n")

            boxes, evidence = fuse_report.read_frame_results(
                path, min_holding_product_score=0.5
            )
            self.assertIn("person-1", boxes)
            self.assertEqual(len(boxes["person-1"]), 3)
            self.assertIn("person-1", evidence)
            # 3 consecutive holding_product frames
            flags = [entry.is_holding_product for entry in evidence["person-1"]]
            self.assertEqual(flags, [True, True, True])
            # person-2/person-3 present but NOT holding
            self.assertEqual([e.is_holding_product for e in evidence["person-2"]],
                             [False, False, False])


class FuseReportCliTest(unittest.TestCase):
    def test_end_to_end_via_open_events_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.json").write_text(
                json.dumps([
                    _pipeline_event("person-1", 1000, 1300),
                    _pipeline_event("person-2", 1000, 1300),
                ]),
                encoding="utf-8",
            )
            with (root / "frame_results.jsonl").open("w", encoding="utf-8") as handle:
                # person-1 & person-3 hold product (3 consecutive frames); person-2 not
                for frame_id, ts in ((0, 1000), (1, 1040), (2, 1080)):
                    handle.write(json.dumps(
                        _frame_line(frame_id, ts, ["person-1", "person-3"])) + "\n")
            (root / "open_events.json").write_text(
                json.dumps([
                    _open_event("person-1"),   # product-confirmed + pipeline event -> escalate
                    _open_event("person-3"),   # product-confirmed + no pipeline -> standalone
                    _open_event("person-4"),   # no product evidence -> discarded
                ]),
                encoding="utf-8",
            )
            (root / "fusion.yml").write_text(
                "fusion:\n  min_product_confirm_frames: 3\n", encoding="utf-8",
            )

            code = fuse_report.main([
                "--events", str(root / "events.json"),
                "--frame-results", str(root / "frame_results.jsonl"),
                "--open-events-json", str(root / "open_events.json"),
                "--fusion-config", str(root / "fusion.yml"),
                "--output", str(root / "fused_events.json"),
                "--report", str(root / "fusion_report.json"),
            ])
            self.assertEqual(code, 0)

            fused = json.loads((root / "fused_events.json").read_text(encoding="utf-8"))
            self.assertEqual(len(fused), 4)  # 2 pipeline (unchanged) + 2 standalone

            by_type: dict[str, list[dict]] = {}
            for event in fused:
                by_type.setdefault(event["event_type"], []).append(event)

            # pipeline events unchanged (no escalation)
            bag = by_type["bag_concealment"]
            self.assertEqual(len(bag), 2)
            for event in bag:
                self.assertEqual(event["risk_level"], "medium")
                self.assertNotIn("open_packaging", event["reason_tags"])

            # standalone package_opening events (person-1 + person-3)
            pkg = by_type["package_opening"]
            self.assertEqual(len(pkg), 2)
            self.assertEqual({event["person_track_id"] for event in pkg},
                             {"person-1", "person-3"})
            for event in pkg:
                self.assertEqual(event["risk_level"], "high")
                self.assertIn("open_packaging", event["reason_tags"])

            report = json.loads((root / "fusion_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["metadata"]["discarded_open_count"], 1)
            self.assertEqual(report["metadata"]["escalated_count"], 0)
            self.assertEqual(len(report["discarded_open_events"]), 1)
            self.assertEqual(report["discarded_open_events"][0]["person_track_id"], "person-4")


if __name__ == "__main__":
    unittest.main()
