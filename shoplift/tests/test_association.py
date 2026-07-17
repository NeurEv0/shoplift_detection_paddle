from __future__ import annotations

import unittest

from shoplift.core.types import BodyPose, DetectionBox, HandRegion, RelationEvidence, Tracklet
from shoplift.tracking.association import (
    AssociationConfig,
    AssociationFrame,
    ContainerEntryDetector,
    DisappearanceAfterEntryDetector,
    HandItemContactAssociator,
    ItemFollowPersonAssociator,
    PoseItemContactAssociator,
    ShopliftingRelationAssociator,
    TrackedDetection,
)


def _box(
    box_id: str,
    frame_id: int,
    category: str,
    bbox: tuple[float, float, float, float],
    *,
    track_id: str | None = None,
    score: float = 0.9,
    attributes: dict[str, object] | None = None,
) -> DetectionBox:
    return DetectionBox(
        box_id=box_id,
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        category=category,
        bbox=bbox,
        score=score,
        track_id=track_id,
        attributes=attributes or {},
    )


def _hand(frame_id: int, bbox: tuple[float, float, float, float]) -> HandRegion:
    return HandRegion(
        hand_track_id="hand-1-right",
        person_track_id="person-1",
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        side="right",
        bbox=bbox,
        score=0.9,
    )


def _pose(frame_id: int, wrist: tuple[float, float]) -> BodyPose:
    keypoints = [(0.0, 0.0) for _ in range(17)]
    scores = [0.0 for _ in range(17)]
    keypoints[10] = wrist
    scores[10] = 0.9
    return BodyPose(
        pose_id="pose-person-1",
        person_track_id="person-1",
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        keypoints=tuple(keypoints),
        scores=tuple(scores),
        score=0.9,
        keypoint_names=tuple(str(index) for index in range(17)),
        skeleton_edges=((8, 10),),
        bbox=(50, 50, 180, 220),
        metadata={"min_keypoint_score": 0.2},
    )


def _person(frame_id: int, track_id: str, bbox: tuple[float, float, float, float]) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        category="person",
        boxes=(_box(f"{track_id}-box-{frame_id}", frame_id, "person", bbox, track_id=track_id),),
    )


def _tracked_item(frame_id: int, bbox: tuple[float, float, float, float]) -> TrackedDetection:
    item = _box(f"item-box-{frame_id}", frame_id, "item", bbox, track_id="item-1")
    return TrackedDetection(detection=item, track_id="item-1")


class AssociationTest(unittest.TestCase):
    def test_hand_item_contact_requires_continuous_stability(self) -> None:
        associator = HandItemContactAssociator(AssociationConfig(min_contact_frames=3))

        first = associator.update(
            frame_id=1,
            timestamp_ms=33,
            hands=(_hand(1, (100, 100, 130, 130)),),
            items=(_tracked_item(1, (126, 104, 150, 126)),),
        )
        second = associator.update(
            frame_id=2,
            timestamp_ms=66,
            hands=(_hand(2, (105, 100, 135, 130)),),
            items=(_tracked_item(2, (131, 104, 155, 126)),),
        )
        third = associator.update(
            frame_id=3,
            timestamp_ms=99,
            hands=(_hand(3, (110, 100, 140, 130)),),
            items=(_tracked_item(3, (136, 104, 160, 126)),),
        )

        self.assertEqual(first, ())
        self.assertEqual(second, ())
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0].relation_type, "hand_item_contact")
        self.assertIn("temporal_consistent", third[0].reason_tags)
        self.assertIn("motion_aligned", third[0].reason_tags)

    def test_pose_item_contact_uses_wrist_keypoints_without_hand_roi(self) -> None:
        associator = PoseItemContactAssociator(AssociationConfig(min_contact_frames=2))

        first = associator.update(
            frame_id=1,
            timestamp_ms=33,
            body_poses=(_pose(1, (130, 116)),),
            items=(_tracked_item(1, (126, 104, 150, 126)),),
        )
        second = associator.update(
            frame_id=2,
            timestamp_ms=66,
            body_poses=(_pose(2, (135, 116)),),
            items=(_tracked_item(2, (131, 104, 155, 126)),),
        )

        self.assertEqual(first, ())
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].relation_type, "hand_item_contact")
        self.assertIn("pose_only", second[0].reason_tags)
        self.assertEqual(second[0].metadata["contact_source"], "body_pose")
        self.assertIn("wrist", second[0].evidence_boxes)

    def test_relation_associator_uses_pose_contact_when_hand_roi_absent(self) -> None:
        associator = ShopliftingRelationAssociator(AssociationConfig(min_contact_frames=1))
        frame = AssociationFrame(
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
            person_tracks=(_person(1, "person-1", (50, 50, 180, 220)),),
            body_poses=(_pose(1, (130, 116)),),
            hand_regions=(),
            items=(_box("item-box-1", 1, "item", (126, 104, 150, 126), track_id="item-1"),),
        )

        result = associator.update(frame)

        contact = [relation for relation in result.relations if relation.relation_type == "hand_item_contact"]
        self.assertEqual(len(contact), 1)
        self.assertIn("pose_only", contact[0].reason_tags)

    def test_item_follow_person_handles_short_missing_gap(self) -> None:
        associator = ItemFollowPersonAssociator(AssociationConfig(max_missing_frames=2))
        person = _person(1, "person-1", (50, 50, 180, 220))
        item = _tracked_item(1, (120, 120, 150, 150))
        contact = RelationEvidence(
            relation_type="hand_item_contact",
            frame_id=1,
            timestamp_ms=33,
            score=0.9,
            reason_tags=("temporal_consistent",),
            person_track_id="person-1",
            hand_track_id="hand-1-right",
            item_track_id="item-1",
        )

        assigned = associator.update(
            frame_id=1,
            timestamp_ms=33,
            person_tracks=(person,),
            items=(item,),
            contact_evidence=(contact,),
        )
        gap = associator.update(
            frame_id=2,
            timestamp_ms=66,
            person_tracks=(person,),
            items=(),
            contact_evidence=(),
        )

        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0].person_track_id, "person-1")
        self.assertEqual(len(gap), 1)
        self.assertIn("gap_filled", gap[0].reason_tags)

    def test_item_follow_person_skips_ambiguous_people(self) -> None:
        associator = ItemFollowPersonAssociator()
        item = _tracked_item(1, (120, 120, 150, 150))
        person_a = _person(1, "person-a", (70, 70, 190, 220))
        person_b = _person(1, "person-b", (70, 70, 190, 220))

        evidences = associator.update(
            frame_id=1,
            timestamp_ms=33,
            person_tracks=(person_a, person_b),
            items=(item,),
        )

        self.assertEqual(evidences, ())

    def test_container_entry_detects_private_and_normal_containers(self) -> None:
        detector = ContainerEntryDetector(AssociationConfig(min_entry_frames=2))
        item_1 = _tracked_item(1, (120, 120, 145, 145))
        item_2 = _tracked_item(2, (121, 121, 146, 146))
        bag_1 = _box("bag-1", 1, "bag", (100, 100, 180, 190), track_id="bag-1")
        bag_2 = _box("bag-2", 2, "bag", (100, 100, 180, 190), track_id="bag-1")
        basket_1 = _box(
            "basket-1",
            3,
            "basket",
            (100, 100, 180, 190),
            track_id="basket-1",
            attributes={"is_normal_container": True},
        )
        basket_2 = _box(
            "basket-2",
            4,
            "basket",
            (100, 100, 180, 190),
            track_id="basket-1",
            attributes={"is_normal_container": True},
        )

        self.assertEqual(
            detector.update(
                frame_id=1,
                timestamp_ms=33,
                items=(item_1,),
                containers=(bag_1,),
                person_by_item={"item-1": "person-1"},
            ),
            (),
        )
        private_entry = detector.update(
            frame_id=2,
            timestamp_ms=66,
            items=(item_2,),
            containers=(bag_2,),
            person_by_item={"item-1": "person-1"},
        )
        detector.update(
            frame_id=3,
            timestamp_ms=99,
            items=(_tracked_item(3, (120, 120, 145, 145)),),
            containers=(basket_1,),
            person_by_item={"item-1": "person-1"},
        )
        normal_entry = detector.update(
            frame_id=4,
            timestamp_ms=132,
            items=(_tracked_item(4, (120, 120, 145, 145)),),
            containers=(basket_2,),
            person_by_item={"item-1": "person-1"},
        )

        self.assertEqual(len(private_entry), 1)
        self.assertIn("entered_private_container", private_entry[0].reason_tags)
        self.assertEqual(len(normal_entry), 1)
        self.assertIn("entered_normal_container", normal_entry[0].reason_tags)
        self.assertLessEqual(normal_entry[0].score, 0.45)

    def test_disappearance_after_entry_requires_threshold_and_exempts_normal_container(self) -> None:
        detector = DisappearanceAfterEntryDetector(
            AssociationConfig(disappeared_after_entry_frames=2)
        )
        private_entry = RelationEvidence(
            relation_type="item_enter_container",
            frame_id=1,
            timestamp_ms=33,
            score=0.85,
            reason_tags=("entered_private_container", "entry_temporal_consistent"),
            person_track_id="person-1",
            item_track_id="item-1",
            container_track_id="bag-1",
            metadata={"is_normal_container": False, "container_kind": "private"},
        )
        normal_entry = RelationEvidence(
            relation_type="item_enter_container",
            frame_id=10,
            timestamp_ms=330,
            score=0.4,
            reason_tags=("entered_normal_container", "entry_temporal_consistent"),
            person_track_id="person-1",
            item_track_id="item-2",
            container_track_id="basket-1",
            metadata={"is_normal_container": True, "container_kind": "normal"},
        )

        detector.update(
            frame_id=1,
            timestamp_ms=33,
            visible_item_ids=("item-1",),
            entry_evidence=(private_entry,),
        )
        self.assertEqual(detector.update(frame_id=2, timestamp_ms=66, visible_item_ids=()), ())
        disappeared = detector.update(frame_id=3, timestamp_ms=99, visible_item_ids=())

        detector.update(
            frame_id=10,
            timestamp_ms=330,
            visible_item_ids=("item-2",),
            entry_evidence=(normal_entry,),
        )
        detector.update(frame_id=11, timestamp_ms=363, visible_item_ids=())
        normal_disappeared = detector.update(frame_id=12, timestamp_ms=396, visible_item_ids=())

        self.assertEqual(len(disappeared), 1)
        self.assertIn("item_disappeared", disappeared[0].reason_tags)
        self.assertIn("after_private_container_entry", disappeared[0].reason_tags)
        self.assertEqual(len(normal_disappeared), 1)
        self.assertIn("normal_container_exempted", normal_disappeared[0].reason_tags)
        self.assertLessEqual(normal_disappeared[0].score, 0.25)


if __name__ == "__main__":
    unittest.main()
