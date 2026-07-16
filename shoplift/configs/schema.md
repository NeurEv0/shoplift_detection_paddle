# Shoplift P0 Data Contracts

All coordinates use pixel-space bounding boxes ordered as `[x1, y1, x2, y2]`.
Frame timing uses `frame_id` and `timestamp_ms`.

## Core Structures

The canonical Python data structures are defined in `shoplift/core/types.py` and
re-exported from `shoplift/tracking/track_types.py`.

| Structure | Purpose |
|---|---|
| `FrameMeta` | Frame identity, timestamp, camera id, and image size. |
| `DetectionBox` | Generic detection or tracked object box with category and score. |
| `Tracklet` | Time-ordered detections belonging to one tracked entity. |
| `HandRegion` | Hand ROI tied to a person track and left/right side. |
| `RelationEvidence` | Explainable hand-item-container relation evidence. |
| `RiskEvent` | Reviewable risk event emitted to downstream systems. |

## PaddleDetection Adapter

`shoplift/adapters/paddledet_adapter.py` converts PaddleDetection outputs without
requiring Paddle or NumPy at import time.

| PaddleDetection output | Internal structure |
|---|---|
| Detection row `[class_id, score, x1, y1, x2, y2]` | `DetectionBox` |
| PP-Human/MOT row `[track_id, class_id, score, x1, y1, x2, y2]` | one-frame `Tracklet` |
| SDE tracker `online_tlwhs`, `online_scores`, `online_ids` | one-frame `Tracklet` |
| PP-Human keypoint `[keypoints, scores]` | left/right `HandRegion` when wrist score passes threshold |

## Person Gate

`shoplift/vision/person_gate.py` provides the P0 person gate.

| Structure | Purpose |
|---|---|
| `PersonGateResult` | Per-frame decision: has person, should run heavy modules, skipped heavy modules, person boxes, and person track ids. |
| `PersonGateMetrics` | Accumulates total frames, skipped frames, triggered frames, `skip_rate`, and `trigger_rate`. |
| `PersonGate` | Stateful gate that accepts `DetectionBox`, `Tracklet`, or adapter frame results. |

## Event JSON Schema

- Schema: `shoplift/events/risk_event.schema.json`
- Example: `shoplift/events/examples/risk_event.example.json`
- Helper module: `shoplift/events/event_schema.py`

Required `RiskEvent` fields:

| Field | Description |
|---|---|
| `schema_version` | Contract version, currently `shoplift.risk_event.v1`. |
| `event_id` | Unique event id. |
| `camera_id` | Source camera id. |
| `timestamp_ms` | Event timestamp in milliseconds. |
| `person_track_id` | Associated person track id. |
| `event_type` | Event type such as `bag_concealment`. |
| `risk_score` | Float in `[0, 1]`. |
| `risk_level` | `low`, `medium`, or `high`. |
| `reason_tags` | Explainable reason tags for review and visualization. |
| `evidence` | One or more `RelationEvidence` entries. |

The schema represents suspicious visual evidence only. It must not be treated as
an automatic legal or operational determination of theft.
