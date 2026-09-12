"""M1: container_kind is resolved from ALL evidence as the item's final
destination (temporal rule), not from the first container in the evidence list.

Key scenarios (from the D03 hi384 v2 run):
- person-472 real theft, v2 model: private entry @14413 then a *spurious*
  normal basket entry @14421 (old model kept product too long) -> M1 sees a
  normal entry AFTER the private one -> downgraded to "normal".
  With the new models (product ends ~14407) that later normal entry disappears
  -> the private entry stands alone -> "private" (high risk).
- person-422 false positive: transient private entry @13379, item later enters
  a normal basket @13399 -> "normal".
- person-69: private bag entry @1823 after a weak normal entry @1820 -> the
  private one is latest -> "private".
"""

from shoplift.rules.risk_score import _container_kind_from_snapshot


class _FakeEvidence:
    def __init__(self, relation_type, frame_id, metadata=None):
        self.relation_type = relation_type
        self.frame_id = frame_id
        self.metadata = metadata or {}


class _FakeSnapshot:
    def __init__(self, evidence, metadata=None, reason_tags=()):
        self.evidence = evidence
        self.metadata = metadata or {}
        self.reason_tags = tuple(reason_tags)


def _enter(frame_id, kind):
    return _FakeEvidence("item_enter_container", frame_id,
                         {"container_kind": kind})


def _disappear(entry_frame_id, kind, report_frame=9000):
    return _FakeEvidence(
        "item_disappeared_after_entry", report_frame,
        {"container_kind": kind, "entry_frame_id": entry_frame_id})


def _snapshot(*evidence):
    return _FakeSnapshot(list(evidence))


def test_v2_scenario_later_normal_entry_downgrades():
    # person-472 with the OLD (v2) model: private entry @14413, then a later
    # spurious normal basket entry @14421 -> M1 treats as normal shopping.
    snap = _snapshot(
        _enter(14421, "normal"),
        _disappear(14413, "private"),
    )
    assert _container_kind_from_snapshot(snap) == "normal"


def test_private_entry_with_no_later_normal_is_private():
    # Person-472 with the new models: product ends ~14407, proxy stops there,
    # no later normal-container entry -> private.
    snap = _snapshot(
        _disappear(14406, "private"),
    )
    assert _container_kind_from_snapshot(snap) == "private"


def test_normal_entry_after_private_downgrades_to_normal():
    # Person-422: transient private entry @13379, later normal basket @13399.
    snap = _snapshot(
        _enter(13399, "normal"),
        _disappear(13379, "private"),
    )
    assert _container_kind_from_snapshot(snap) == "normal"


def test_normal_before_private_keeps_private():
    # Person-69: weak normal entry @1820, private bag entry @1823 (latest).
    snap = _snapshot(
        _disappear(1820, "normal"),
        _enter(1823, "private"),
    )
    assert _container_kind_from_snapshot(snap) == "private"


def test_bag_kind_maps_to_private():
    snap = _snapshot(_disappear(14406, "bag"))
    assert _container_kind_from_snapshot(snap) == "private"


def test_special_latest_is_special():
    snap = _snapshot(_enter(14406, "special"))
    assert _container_kind_from_snapshot(snap) == "special"


def test_normal_after_special_downgrades_to_normal():
    snap = _snapshot(
        _enter(14406, "special"),
        _enter(14420, "normal"),
    )
    assert _container_kind_from_snapshot(snap) == "normal"


def test_pure_normal_stays_normal():
    snap = _snapshot(_enter(100, "normal"))
    assert _container_kind_from_snapshot(snap) == "normal"


def test_no_container_falls_back_to_tags():
    snap = _FakeSnapshot([], reason_tags=("entered_private_container",))
    assert _container_kind_from_snapshot(snap) == "private"
