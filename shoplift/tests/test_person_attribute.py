from __future__ import annotations

import unittest

from shoplift.core.types import DetectionBox, FrameMeta, HandRegion, Tracklet
from shoplift.vision.person_attribute import (
    PersonAttributePostProcessor,
    ProxyItemRegionBuilder,
    RuleBasedPersonAttributeEstimator,
)


class PersonAttributeTest(unittest.TestCase):
    def test_visibility_not_judgable_forces_uncertain_hand_state(self) -> None:
        frame, track, _ = self.fixture()
        attribute = PersonAttributePostProcessor().build_attribute(
            frame=frame,
            person_track=track,
            raw={
                "left_hand_state": ("holding_product", 0.82),
                "left_hand_visibility": ("not_judgable", 0.91),
                "right_hand_state": ("empty", 0.7),
                "right_hand_visibility": ("clear", 0.8),
                "body_orientation": ("side", 0.6),
                "occlusion_level": ("light", 0.7),
            },
        )

        self.assertEqual(attribute.left_hand_state.label, "uncertain")
        self.assertEqual(attribute.left_hand_visibility.label, "not_judgable")

    def test_holding_product_hand_roi_generates_proxy_item_detection(self) -> None:
        frame, track, hand = self.fixture()
        attribute = PersonAttributePostProcessor().build_attribute(
            frame=frame,
            person_track=track,
            raw={
                "left_hand_state": ("holding_product", 0.9),
                "left_hand_visibility": ("clear", 0.85),
                "right_hand_state": ("empty", 0.7),
                "right_hand_visibility": ("clear", 0.8),
                "body_orientation": ("front", 0.6),
                "occlusion_level": ("none", 0.7),
            },
        )

        regions = ProxyItemRegionBuilder().build(
            frame=frame,
            person_attributes=(attribute,),
            hand_regions=(hand,),
        )

        self.assertEqual(len(regions), 1)
        proxy = regions[0]
        self.assertFalse(proxy.is_precise_item_bbox)
        self.assertEqual(proxy.proxy_bbox, hand.bbox)
        detection = proxy.to_detection_box()
        self.assertEqual(detection.category, "item")
        self.assertEqual(detection.track_id, "proxy-item-person-1-left")
        self.assertTrue(detection.attributes["is_proxy_item_region"])

    def test_rule_based_estimator_is_conservative(self) -> None:
        frame, track, hand = self.fixture()
        attributes = RuleBasedPersonAttributeEstimator().estimate(
            frame=frame,
            person_tracks=(track,),
            hand_regions=(hand,),
        )

        self.assertEqual(attributes[0].left_hand_state.label, "empty")
        self.assertEqual(attributes[0].right_hand_state.label, "uncertain")

    def fixture(self):
        frame = FrameMeta(
            frame_id=1,
            timestamp_ms=33,
            camera_id="camera-1",
            width=100,
            height=100,
        )
        person_box = DetectionBox(
            box_id="person-box",
            frame_id=1,
            timestamp_ms=33,
            category="person",
            bbox=(10, 10, 70, 90),
            score=0.9,
            track_id="person-1",
        )
        track = Tracklet(track_id="person-1", category="person", boxes=(person_box,))
        hand = HandRegion(
            hand_track_id="hand-person-1-left",
            person_track_id="person-1",
            frame_id=1,
            timestamp_ms=33,
            side="left",
            bbox=(20, 45, 35, 60),
            score=0.8,
        )
        return frame, track, hand


if __name__ == "__main__":
    unittest.main()
