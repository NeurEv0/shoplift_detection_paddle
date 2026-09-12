"""Unit tests for the nested private-container-into-public concealment detector.

v1.3 语义: bag 跟踪+主人绑定; 主人累计在篮停留 >= owner_dwell_frames 触发;
每 (锚, owner) 一次; 容器仅 source=basket.
"""

from __future__ import annotations

import unittest

from shoplift.core.types import DetectionBox, HandRegion
from shoplift.events.event_engine import ShopliftingEventEngine
from shoplift.events.nested_concealment import (
    NestedConcealmentConfig,
    NestedConcealmentDetector,
    TAG_DWELL,
    TAG_ENTRY,
    TAG_HIDDEN,
)
from shoplift.tracking.association import AssociationFrame

BASKET_BOX = (1175.0, 521.0, 1254.0, 615.0)   # 店内篮(口线 y1=521, source basket)
MOUTH = 521.0 + 20.0                          # margin 20 → 541
CARRY_BAG = (1190.0, 400.0, 1230.0, 500.0)    # 携带中(底边 500 < 541)
PLACED_BAG = (1195.0, 530.0, 1230.0, 562.0)   # 放入(底边 562 >= 541)
HAND_BOX = (1200.0, 540.0, 1220.0, 560.0)     # 手在篮内


def _det(box_id: str, frame_id: int, category: str, bbox, source: str, score: float = 0.8):
    return DetectionBox(
        box_id=box_id,
        frame_id=frame_id,
        category=category,
        bbox=bbox,
        score=score,
        timestamp_ms=frame_id * 40,
        attributes={"source_category": source},
    )


def _hand(person: str, frame_id: int, box, side: str = "right"):
    return HandRegion(
        hand_track_id=f"hand-{person}-{frame_id}",
        person_track_id=person,
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        side=side,
        bbox=box,
        score=0.9,
    )


def _frame(frame_id: int, containers=(), hands=()):
    return AssociationFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        camera_id="test-cam",
        person_tracks=(),
        body_poses=(),
        hand_regions=tuple(hands),
        items=(),
        containers=tuple(containers),
        extension_regions=(),
    )


class NestedConcealmentDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NestedConcealmentConfig(owner_dwell_frames=15)
        self.detector = NestedConcealmentDetector(self.config)

    def _feed(self, owner: str, dwell_runs: tuple[tuple[int, int], ...]):
        """携带(手近袋绑定 owner) → 放入 → 隐藏武装 → owner 在篮停留 runs."""
        n_carry = 40
        for fr in range(0, n_carry):
            bag = _det(f"bag-{fr}", fr, "bag", CARRY_BAG, "backpack")
            basket = _det("basket-1", fr, "basket", BASKET_BOX, "basket")
            hand = _hand(owner, fr, (1208.0, 430.0, 1216.0, 470.0))  # 手近包
            self.detector.update(_frame(fr, containers=(basket, bag), hands=(hand,)))
        # 放入 f40-42 (底边过口线), owner 手仍在包旁
        for fr in range(n_carry, n_carry + 3):
            bag = _det(f"bag-{fr}", fr, "bag", PLACED_BAG, "backpack")
            basket = _det("basket-1", fr, "basket", BASKET_BOX, "basket")
            hand = _hand(owner, fr, (1212.0, 500.0, 1220.0, 545.0))
            self.detector.update(_frame(fr, containers=(basket, bag), hands=(hand,)))
        # 包消失 f43+ → hide_confirm=2 → armed f45
        for fr in range(n_carry + 3, n_carry + 6):
            basket = _det("basket-1", fr, "basket", BASKET_BOX, "basket")
            self.detector.update(_frame(fr, containers=(basket,)))
        # owner 停留
        for (start, end) in dwell_runs:
            for fr in range(start, end + 1):
                hand = _hand(owner, fr, HAND_BOX)
                basket = _det("basket-1", fr, "basket", BASKET_BOX, "basket")
                events = self.detector.update(_frame(fr, containers=(basket,), hands=(hand,)))
                if events:
                    return events
        return ()

    def test_owner_dwell_fires(self) -> None:
        events = self._feed("p1", ((200, 214),))  # 15 帧 >= owner_dwell_frames=15
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.person_track_id, "p1")
        self.assertEqual(event.event_type, "private_container_concealment")
        self.assertEqual(event.risk_level, "high")
        self.assertIn(TAG_ENTRY, event.reason_tags)
        self.assertIn(TAG_HIDDEN, event.reason_tags)
        self.assertIn(TAG_DWELL, event.reason_tags)
        self.assertEqual(event.metadata["nested"]["owner"], "p1")

    def test_non_owner_dwell_no_fire(self) -> None:
        """他人(非 owner)伸手入篮不计 → 不触发."""
        self._feed("p1", ())
        events = ()
        for fr in range(200, 230):
            hand = _hand("p9", fr, HAND_BOX)
            basket = _det("basket-1", fr, "basket", BASKET_BOX, "basket")
            events = self.detector.update(_frame(fr, containers=(basket,), hands=(hand,)))
            if events:
                break
        self.assertEqual(events, ())

    def test_short_dwell_no_fire(self) -> None:
        events = self._feed("p1", ((200, 210),))  # 11 帧 < 15
        self.assertEqual(events, ())

    def test_cart_source_not_matched(self) -> None:
        """source=cart 的购物车不进入窗口."""
        for fr in range(0, 40):
            bag = _det(f"bag-{fr}", fr, "bag", CARRY_BAG, "backpack")
            cart = _det("cart-1", fr, "basket", BASKET_BOX, "cart")
            hand = _hand("p1", fr, (1208.0, 430.0, 1216.0, 470.0))
            self.detector.update(_frame(fr, containers=(cart, bag), hands=(hand,)))
        for fr in range(40, 46):
            bag = _det(f"bag-{fr}", fr, "bag", PLACED_BAG, "backpack")
            cart = _det("cart-1", fr, "basket", BASKET_BOX, "cart")
            self.detector.update(_frame(fr, containers=(cart, bag)))
        for fr in range(46, 50):
            cart = _det("cart-1", fr, "basket", BASKET_BOX, "cart")
            self.detector.update(_frame(fr, containers=(cart,)))
        events = ()
        for fr in range(200, 240):
            hand = _hand("p1", fr, HAND_BOX)
            cart = _det("cart-1", fr, "basket", BASKET_BOX, "cart")
            events = self.detector.update(_frame(fr, containers=(cart,), hands=(hand,)))
            if events:
                break
        self.assertEqual(events, ())

    def test_visible_bag_never_arms(self) -> None:
        """包一直可见(不消失) → 即使停留也不触发."""
        events = ()
        for fr in range(0, 60):
            bag = _det(f"bag-{fr}", fr, "bag", PLACED_BAG, "backpack")
            basket = _det("basket-1", fr, "basket", BASKET_BOX, "basket")
            hand = _hand("p1", fr, HAND_BOX)
            events = self.detector.update(_frame(fr, containers=(basket, bag), hands=(hand,)))
            if events:
                break
        self.assertEqual(events, ())


class NestedConcealmentEngineTest(unittest.TestCase):
    """引擎级集成: process_frame 合并探测器事件并通过 RuleValidator 保持 HIGH."""

    def test_engine_merges_nested_event(self) -> None:
        engine = ShopliftingEventEngine(
            nested_concealment_config=NestedConcealmentConfig(owner_dwell_frames=15)
        )
        n_carry = 40
        fired = []
        for fr in range(0, 260):
            containers = [_det("basket-1", fr, "basket", BASKET_BOX, "basket")]
            hands = []
            if fr < n_carry:
                bag = _det(f"bag-{fr}", fr, "bag", CARRY_BAG, "backpack")
                containers.append(bag)
                hands.append(_hand("p1", fr, (1208.0, 430.0, 1216.0, 470.0)))
            elif fr < n_carry + 3:
                bag = _det(f"bag-{fr}", fr, "bag", PLACED_BAG, "backpack")
                containers.append(bag)
                hands.append(_hand("p1", fr, (1212.0, 500.0, 1220.0, 545.0)))
            elif fr >= 200:
                hands.append(_hand("p1", fr, HAND_BOX))
            result = engine.process_frame(_frame(fr, containers=containers, hands=hands))
            for event in result.events:
                if event.event_type == "private_container_concealment":
                    fired.append(event)
        self.assertEqual(len(fired), 1)
        event = fired[0]
        self.assertEqual(event.person_track_id, "p1")
        self.assertEqual(event.risk_level, "high")


if __name__ == "__main__":
    unittest.main()
