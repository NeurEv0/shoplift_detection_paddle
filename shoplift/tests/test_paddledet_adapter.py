from __future__ import annotations

import unittest
from collections import defaultdict

from shoplift.adapters import PaddleDetectionAdapter, SHOPLIFT_CLASS_ID_TO_CATEGORY
from shoplift.core.types import FrameMeta


class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class PaddleDetectionAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameMeta(
            frame_id=12,
            timestamp_ms=400,
            camera_id="camera-a",
            width=640,
            height=480,
        )

    def test_convert_detection_rows(self) -> None:
        adapter = PaddleDetectionAdapter(class_id_to_category=SHOPLIFT_CLASS_ID_TO_CATEGORY)
        result = {
            "boxes": FakeArray(
                [
                    [0, 0.91, 10, 20, 50, 80],
                    [3, 0.76, 100, 120, 180, 260],
                ]
            ),
            "boxes_num": [2],
        }

        detections = adapter.convert_detection_result(result, self.frame)

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].category, "item")
        self.assertEqual(detections[0].bbox, (10.0, 20.0, 50.0, 80.0))
        self.assertEqual(detections[1].category, "bag")
        self.assertEqual(detections[1].attributes["source_category"], "backpack")

    def test_convert_detection_dict_rows(self) -> None:
        adapter = PaddleDetectionAdapter()
        result = [
            {
                "track_id": "external-1",
                "label": "person",
                "score": 0.88,
                "bbox": [12, 24, 30, 40],
                "bbox_format": "xywh",
                "occluded": True,
            }
        ]

        detections = adapter.convert_detection_result(result, self.frame)

        self.assertEqual(detections[0].track_id, "external-1")
        self.assertEqual(detections[0].bbox, (12.0, 24.0, 42.0, 64.0))
        self.assertTrue(detections[0].attributes["occluded"])

    def test_convert_detection_dict_with_string_class(self) -> None:
        adapter = PaddleDetectionAdapter()
        result = [{"class": "handbag", "score": 0.8, "bbox": [1, 2, 3, 4]}]

        detections = adapter.convert_detection_result(result, self.frame)

        self.assertEqual(detections[0].category, "bag")

    def test_convert_pipeline_mot_rows(self) -> None:
        adapter = PaddleDetectionAdapter()
        mot_result = {"boxes": [[7, 0, 0.93, 100, 120, 220, 360]]}

        tracklets = adapter.convert_mot_result(mot_result, self.frame)

        self.assertEqual(len(tracklets), 1)
        self.assertEqual(tracklets[0].track_id, "person-7")
        self.assertEqual(tracklets[0].boxes[0].track_id, "person-7")
        self.assertEqual(tracklets[0].boxes[0].bbox, (100.0, 120.0, 220.0, 360.0))

    def test_convert_sde_online_tlwh_output(self) -> None:
        adapter = PaddleDetectionAdapter()
        sde_output = {
            "online_tlwhs": [[10, 20, 30, 40]],
            "online_scores": [0.8],
            "online_ids": [2],
        }

        tracklets = adapter.convert_mot_result(sde_output, self.frame)

        self.assertEqual(tracklets[0].track_id, "person-2")
        self.assertEqual(tracklets[0].boxes[0].bbox, (10.0, 20.0, 40.0, 60.0))

    def test_convert_sde_online_tuple_with_class_mappings(self) -> None:
        adapter = PaddleDetectionAdapter()
        sde_output = [
            defaultdict(list, {0: [[10, 20, 30, 40], [100, 120, 20, 60]]}),
            defaultdict(list, {0: [0.8, 0.7]}),
            defaultdict(list, {0: [2, 3]}),
        ]

        tracklets = adapter.convert_mot_result(sde_output, self.frame)

        self.assertEqual([tracklet.track_id for tracklet in tracklets], ["person-2", "person-3"])
        self.assertEqual(tracklets[1].boxes[0].bbox, (100.0, 120.0, 120.0, 180.0))

    def test_convert_keypoints_to_hand_regions(self) -> None:
        adapter = PaddleDetectionAdapter()
        mot_result = {"boxes": [[7, 0, 0.93, 100, 120, 220, 360]]}
        person_tracks = adapter.convert_mot_result(mot_result, self.frame)
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.1 for _ in range(17)]
        keypoints[7] = [130.0, 220.0]
        keypoints[8] = [190.0, 220.0]
        keypoints[9] = [120.0, 260.0]
        keypoints[10] = [210.0, 260.0]
        scores[7] = 0.7
        scores[8] = 0.8
        scores[9] = 0.9
        scores[10] = 0.95
        kpt_result = {"keypoint": [[keypoints], [scores]]}

        hands = adapter.convert_keypoint_result(kpt_result, self.frame, person_tracks)

        self.assertEqual({hand.side for hand in hands}, {"left", "right"})
        self.assertEqual(hands[0].person_track_id, "person-7")
        self.assertTrue(hands[0].bbox[0] < keypoints[9][0] < hands[0].bbox[2])
        self.assertEqual(hands[0].metadata["source"], "ppdet_keypoint")

    def test_convert_keypoints_to_body_pose_evidence(self) -> None:
        adapter = PaddleDetectionAdapter()
        mot_result = {"boxes": [[7, 0, 0.93, 100, 120, 220, 360]]}
        person_tracks = adapter.convert_mot_result(mot_result, self.frame)
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.1 for _ in range(17)]
        keypoints[5] = [130.0, 180.0]
        keypoints[7] = [140.0, 220.0]
        keypoints[9] = [145.0, 260.0]
        scores[5] = 0.8
        scores[7] = 0.7
        scores[9] = 0.9
        kpt_result = {"keypoint": [[keypoints], [scores]]}

        body_poses = adapter.convert_body_pose_result(kpt_result, self.frame, person_tracks)

        self.assertEqual(len(body_poses), 1)
        self.assertEqual(body_poses[0].person_track_id, "person-7")
        self.assertEqual(body_poses[0].metadata["visible_keypoint_count"], 3)
        self.assertIn((5, 7), body_poses[0].skeleton_edges)

    def test_convert_two_person_keypoints_without_score_field(self) -> None:
        adapter = PaddleDetectionAdapter()
        person_one = [[0.0, 0.0, 0.1] for _ in range(17)]
        person_two = [[0.0, 0.0, 0.1] for _ in range(17)]
        person_one[9] = [120.0, 260.0, 0.9]
        person_two[10] = [300.0, 260.0, 0.9]

        hands = adapter.convert_keypoint_result({"keypoint": [person_one, person_two]}, self.frame)

        self.assertEqual(len(hands), 2)
        self.assertEqual({hand.person_track_id for hand in hands}, {"person-0", "person-1"})

    def test_convert_pphuman_result(self) -> None:
        adapter = PaddleDetectionAdapter()
        keypoints = [[0.0, 0.0] for _ in range(17)]
        scores = [0.1 for _ in range(17)]
        keypoints[9] = [120.0, 260.0]
        keypoints[10] = [210.0, 260.0]
        scores[9] = 0.9
        scores[10] = 0.95
        pphuman_result = {
            "mot": {"boxes": [[7, 0, 0.93, 100, 120, 220, 360]]},
            "kpt": {"keypoint": [[keypoints], [scores]]},
        }

        frame_result = adapter.convert_pphuman_result(pphuman_result, self.frame)

        self.assertEqual(frame_result.metadata["track_count"], 1)
        self.assertEqual(frame_result.metadata["body_pose_count"], 1)
        self.assertEqual(frame_result.metadata["hand_region_count"], 2)
        self.assertEqual(frame_result.to_dict()["body_poses"][0]["person_track_id"], "person-7")
        self.assertEqual(frame_result.to_dict()["person_tracks"][0]["track_id"], "person-7")


if __name__ == "__main__":
    unittest.main()
