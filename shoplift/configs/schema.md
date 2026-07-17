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
| `BodyPose` | Full-body keypoints, scores, skeleton edges, pose score, and bound person track. |
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
| PP-Human keypoint `[keypoints, scores]` | `BodyPose`; optional left/right `HandRegion` only when explicitly enabled |

## Person Gate

`shoplift/vision/person_gate.py` provides the P0 person gate.

| Structure | Purpose |
|---|---|
| `PersonGateResult` | Per-frame decision: has person, should run heavy modules, skipped heavy modules, person boxes, and person track ids. |
| `PersonGateMetrics` | Accumulates total frames, skipped frames, triggered frames, `skip_rate`, and `trigger_rate`. |
| `PersonGate` | Stateful gate that accepts `DetectionBox`, `Tracklet`, or adapter frame results. |

## Body Pose

`shoplift/vision/body_pose.py` exposes PP-Human keypoints as first-class pose
evidence instead of treating them only as a hand ROI intermediate.

| Structure | Purpose |
|---|---|
| `BodyPoseBuilder` | Binds COCO-style keypoints and scores to `person_track_id`. |
| `COCO_PERSON_KEYPOINT_NAMES` | The 17 keypoint names used by the PP-Human pose model. |
| `COCO_PERSON_SKELETON` | Skeleton edge pairs used by debug visualization. |

Each `BodyPose` contains all keypoints, per-keypoint scores, skeleton edges,
pose-level score, optional person bbox, and visibility metadata.

Current default policy is pose-only. `pose_recognition` and `pose_hand` are
treated as OR evidence sources, but `pose_hand` is disabled by default because
manual review marked hand ROI as redundant. Pose wrist keypoints can still feed
contact evidence without emitting `HandRegion` objects.

## Pose Hand

`shoplift/vision/pose_hand.py` derives hand ROIs from body keypoints without
requiring Paddle or NumPy at import time.

| Structure | Purpose |
|---|---|
| `PersonPose` | Normalized COCO-style keypoints, scores, optional person bbox, and bound `person_track_id`. |
| `HandRegionExtractor` | Creates left/right `HandRegion` entries from wrist and elbow keypoints. |
| `extract_hand_regions` | Convenience function for normalized keypoints plus optional `Tracklet` binding. |

The P0 extractor uses COCO indices `left_elbow=7`, `right_elbow=8`,
`left_wrist=9`, and `right_wrist=10`. Wrist keypoints below
`min_keypoint_score` are filtered. When elbow confidence is high, the hand ROI
size is based on forearm length; otherwise it falls back to the person bbox
when available.

## Item And Container Detection

`shoplift/vision/object_container.py` normalizes coarse product/container
detections and groups them for downstream relation analysis.

| Canonical category | Aliases | Role |
|---|---|---|
| `item` | `product` | item |
| `bag` | `backpack`, `handbag` | container |
| `basket` | `cart` | normal container |
| `stroller` | - | container |
| `helmet` | - | container |
| `clothing_region` | `pocket_region` | extension region |

`ItemContainerDetectionAdapter` preserves model metadata while adding
`source_category`, `canonical_category`, `detection_role`, and
`is_normal_container` attributes to each normalized `DetectionBox`.

## Offline Frame Result

`shoplift/cli/offline_analyze.py` writes one JSON object per processed frame to
`frame_results.jsonl`.

Required top-level fields:

| Field | Description |
|---|---|
| `schema_version` | Frame result contract version, currently `shoplift.frame_result.v1`. |
| `frame` | `FrameMeta` for the processed source frame. |
| `person_gate` | `PersonGateResult`, including skip/trigger decision and person track ids. |
| `person_tracks` | Person `Tracklet` entries available for this frame. |
| `body_poses` | Bound `BodyPose` entries available for this frame. |
| `hand_regions` | Bound `HandRegion` entries available for this frame; empty by default in pose-only mode. |
| `item_container` | Grouped item/container/extension-region detections. |
| `metadata` | Input type, source frame id, source URI, backend id, and module timing metrics. |

The CLI supports video files and frame directories. The default model-free
backend emits empty detections while exercising frame IO, person gate decisions,
JSONL output, event-engine wiring, and debug visualization generation.

The optional `paddledet_pphuman` backend wraps PaddleDetection deploy
predictors for PP-Human MOT, optional keypoint-to-body-pose/hand ROI, and optional
item/container detection. It requires local exported model directories and does
not auto-download models during offline analysis.

`events.json` is generated by `ShopliftingEventEngine` from each frame's
`AssociationFrame`. It remains empty when relation evidence is insufficient or
when the configured backend does not provide item/container evidence.

## Event JSON Schema

- Schema: `shoplift/events/risk_event.schema.json`
- Example: `shoplift/events/examples/risk_event.example.json`
- Helper module: `shoplift/events/event_schema.py`
- Event engine: `shoplift/events/event_engine.py`
- Risk scoring: `shoplift/rules/risk_score.py`
- Rule validation: `shoplift/rules/validators.py`

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

## P1 Event Types

`ShopliftingEventEngine` consumes relation evidence and state-machine snapshots
to emit the first batch of reviewable event types:

| Event type | Meaning |
|---|---|
| `bag_concealment` | Item entered a private bag/container and later disappeared. |
| `clothing_concealment` | Item entered a clothing or pocket region and later disappeared. |
| `special_container_concealment` | Item entered a special container such as stroller/helmet and later disappeared. |
| `bulk_pickup_to_bag` | Multiple item tracks entered the same private container in a short window. |
| `near_body_suspicious` | Contact plus person-follow/near-body evidence exists, but concealment is not confirmed. |

## P1 Risk Scoring And Validation

`RiskScorer` combines action type weight, container type weight, temporal
evidence, relation diversity, model confidence, area risk, and normal-shopping
downgrade into `risk_score` and `risk_level`.

`RiskRuleValidator` normalizes unsafe outputs:

| Rule | Behavior |
|---|---|
| High risk needs multiple reason tags | Downgrade high-risk events with insufficient explanation. |
| Normal basket/cart placement | Cap or downgrade normal-container events and add normal-shopping tags. |
| Low visibility or severe occlusion | Add `low_visibility` and cap the risk. |
| Single-frame contact | Cap contact-only events so they cannot become high risk. |
