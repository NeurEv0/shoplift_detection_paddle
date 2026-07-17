from __future__ import annotations

import unittest

from shoplift.core.types import (
    BodyPose,
    DetectionBox,
    FrameMeta,
    HandRegion,
    RelationEvidence,
    RiskEvent,
    Tracklet,
)


class CoreTypesTest(unittest.TestCase):
    def test_core_types_are_json_ready(self) -> None:
        frame = FrameMeta(
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
            width=1920,
            height=1080,
        )
        person_box = DetectionBox(
            box_id="person-box-1",
            frame_id=1,
            timestamp_ms=33,
            category="person",
            bbox=(10, 20, 110, 220),
            score=0.91,
            track_id="person-1",
        )
        tracklet = Tracklet(track_id="person-1", category="person", boxes=(person_box,))
        hand = HandRegion(
            hand_track_id="hand-1-right",
            person_track_id="person-1",
            frame_id=1,
            timestamp_ms=33,
            side="right",
            bbox=(80, 150, 120, 190),
            score=0.78,
            source_keypoints=((100, 170),),
        )
        body_pose = BodyPose(
            pose_id="pose-person-1",
            person_track_id="person-1",
            frame_id=1,
            timestamp_ms=33,
            keypoints=((100, 100), (110, 110)),
            scores=(0.9, 0.8),
            score=0.85,
            keypoint_names=("nose", "left_eye"),
            skeleton_edges=((0, 1),),
            bbox=person_box.bbox,
        )
        evidence = RelationEvidence(
            relation_type="hand_item_contact",
            frame_id=1,
            timestamp_ms=33,
            score=0.73,
            reason_tags=("bbox_overlap",),
            person_track_id="person-1",
            hand_track_id=hand.hand_track_id,
            item_track_id="item-1",
            evidence_boxes={"hand": hand.bbox, "item": (112, 154, 140, 188)},
        )
        event = RiskEvent(
            event_id="evt-1",
            camera_id=frame.camera_id,
            timestamp_ms=33,
            person_track_id=tracklet.track_id,
            event_type="near_body_suspicious",
            risk_score=0.48,
            risk_level="medium",
            reason_tags=("hand_item_contact",),
            evidence=(evidence,),
        )

        payload = event.to_dict()

        self.assertEqual(payload["camera_id"], "camera-1")
        self.assertEqual(payload["evidence"][0]["evidence_boxes"]["hand"], [80.0, 150.0, 120.0, 190.0])
        self.assertEqual(body_pose.to_dict()["skeleton_edges"], [[0, 1]])
        self.assertEqual(tracklet.start_frame_id, 1)
        self.assertEqual(person_box.center, (60.0, 120.0))

    def test_detection_box_rejects_invalid_bbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "bbox"):
            DetectionBox(
                box_id="bad",
                frame_id=0,
                category="item",
                bbox=(10, 10, 5, 20),
                score=0.5,
            )


if __name__ == "__main__":
    unittest.main()
