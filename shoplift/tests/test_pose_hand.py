from __future__ import annotations

import unittest

from shoplift.core.types import DetectionBox, FrameMeta, Tracklet
from shoplift.vision.pose_hand import HandRegionExtractor, build_person_poses, extract_hand_regions


class PoseHandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameMeta(
            frame_id=5,
            timestamp_ms=165,
            camera_id="camera-pose",
            width=640,
            height=480,
        )
        self.person_box = DetectionBox(
            box_id="person-box-11",
            frame_id=self.frame.frame_id,
            timestamp_ms=self.frame.timestamp_ms,
            category="person",
            bbox=(100, 120, 260, 420),
            score=0.9,
            track_id="person-11",
        )
        self.tracklet = Tracklet(
            track_id="person-11",
            category="person",
            boxes=(self.person_box,),
        )

    def test_wrist_and_elbow_keypoints_derive_bound_hand_regions(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.1 for _ in range(17)]
        keypoints[7] = [130.0, 220.0]
        keypoints[8] = [230.0, 220.0]
        keypoints[9] = [120.0, 270.0]
        keypoints[10] = [250.0, 270.0]
        scores[7] = 0.8
        scores[8] = 0.7
        scores[9] = 0.95
        scores[10] = 0.9

        hands = extract_hand_regions(
            self.frame,
            [keypoints],
            [scores],
            [self.tracklet],
            min_keypoint_score=0.2,
        )

        self.assertEqual({hand.side for hand in hands}, {"left", "right"})
        self.assertEqual({hand.person_track_id for hand in hands}, {"person-11"})
        self.assertEqual({hand.hand_track_id for hand in hands}, {"hand-person-11-left", "hand-person-11-right"})
        self.assertTrue(all(len(hand.source_keypoints) == 2 for hand in hands))
        self.assertTrue(hands[0].bbox[0] <= keypoints[9][0] <= hands[0].bbox[2])
        self.assertEqual({hand.metadata["crop_strategy"] for hand in hands}, {"forearm_guided_axis_aligned"})

    def test_forearm_guided_crop_extends_center_past_wrist(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.0 for _ in range(17)]
        keypoints[8] = [100.0, 100.0]
        keypoints[10] = [140.0, 100.0]
        scores[8] = 0.9
        scores[10] = 0.9
        poses = build_person_poses([keypoints], [scores], [self.tracklet])

        hands = HandRegionExtractor(
            min_keypoint_score=0.2,
            forearm_extension_ratio=0.5,
            hand_side_from_forearm=1.0,
        ).extract(self.frame, poses)

        self.assertEqual(len(hands), 1)
        hand = hands[0]
        self.assertEqual(hand.side, "right")
        self.assertEqual(hand.metadata["crop_center"], (160.0, 100.0))
        self.assertEqual(hand.metadata["forearm_vector"], (40.0, 0.0))
        self.assertEqual(hand.metadata["forearm_length"], 40.0)
        self.assertEqual(hand.metadata["crop_side_length"], 40.0)
        self.assertEqual(hand.bbox, (140.0, 80.0, 180.0, 120.0))

    def test_single_pose_score_is_used_for_forearm_keypoints(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        keypoints[7] = [100.0, 100.0]
        keypoints[9] = [100.0, 130.0]
        poses = build_person_poses([keypoints], [[0.8]], [self.tracklet])

        hands = HandRegionExtractor(min_keypoint_score=0.2).extract(self.frame, poses)

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].metadata["crop_strategy"], "forearm_guided_axis_aligned")
        self.assertEqual(hands[0].metadata["elbow_score"], 0.8)

    def test_low_confidence_wrist_is_filtered(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.0 for _ in range(17)]
        keypoints[9] = [120.0, 270.0]
        keypoints[10] = [250.0, 270.0]
        scores[9] = 0.19
        scores[10] = 0.9

        hands = extract_hand_regions(
            self.frame,
            [keypoints],
            [scores],
            [self.tracklet],
            min_keypoint_score=0.2,
        )

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].side, "right")
        self.assertEqual(hands[0].person_track_id, "person-11")

    def test_build_person_poses_falls_back_to_index_track_ids(self) -> None:
        keypoints = [[[0.0, 0.0, 0.5] for _ in range(17)]]
        poses = build_person_poses(keypoints)

        self.assertEqual(poses[0].person_track_id, "person-0")
        self.assertIsNone(poses[0].person_bbox)
        self.assertEqual(poses[0].keypoints[0], (0.0, 0.0))

    def test_extractor_uses_person_bbox_when_elbow_is_low_confidence(self) -> None:
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.0 for _ in range(17)]
        keypoints[9] = [105.0, 125.0]
        scores[9] = 0.9
        poses = build_person_poses([keypoints], [scores], [self.tracklet])

        hands = HandRegionExtractor(min_keypoint_score=0.2).extract(self.frame, poses)

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].side, "left")
        self.assertGreaterEqual(hands[0].bbox[0], 0.0)
        self.assertEqual(len(hands[0].source_keypoints), 1)
        self.assertEqual(hands[0].metadata["crop_strategy"], "wrist_fallback_axis_aligned")


if __name__ == "__main__":
    unittest.main()
