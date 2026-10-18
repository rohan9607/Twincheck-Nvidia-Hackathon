from pathlib import Path

import numpy as np
import pytest

from twincheck_core import (Action, Comparator, Detection, GraspState, Mode, Pose2D, StampBuffer, apply_homography,
                            check_post_place, check_pre_pick, fuse_poses, load_config, pick_threshold, plan_grasp,
                            plan_resync, threshold_sweep, wilson_interval, yaw_diff_deg)
from twincheck_core.checks import CheckPoint

CFG = load_config(Path(__file__).resolve().parents[2] / "config" / "twincheck.yaml")


# ---------- geometry ----------
@pytest.mark.parametrize("a,b,sym,exp", [
    (0, 90, 360, 90), (0, 90, 90, 0), (0, 89, 90, 1), (10, 350, 360, 20),
    (0, 179, 180, 1), (0, 123, 0, 0), (-45, 45, 90, 0),
])
def test_yaw_diff_symmetry(a, b, sym, exp):
    assert yaw_diff_deg(a, b, sym) == pytest.approx(exp)


def test_homography_identity_and_scale():
    H = np.diag([2.0, 3.0, 1.0])
    assert apply_homography(H, 10, 10) == pytest.approx((20, 30))


def test_fuse_cube_wraps_correctly():
    p = fuse_poses([Pose2D(0, 0, 1), Pose2D(2, 0, 89)], symmetry_deg=90)
    assert p.x_mm == pytest.approx(1)
    assert min(p.yaw_deg, 90 - p.yaw_deg) < 1.5  # near 0/90, not 45


# ---------- checks ----------
def test_pre_pick_pass_and_shift():
    pick = CFG.pick_points["slot_1"]
    ok = [Detection("cube", Pose2D(pick.x_mm + 3, pick.y_mm, 2))]
    assert check_pre_pick(pick, "cube", ok, CFG.thresholds, 90).passed
    shifted = [Detection("cube", Pose2D(pick.x_mm + 30, pick.y_mm, 0))]
    out = check_pre_pick(pick, "cube", shifted, CFG.thresholds, 90)
    assert not out.passed and out.position_error_mm == pytest.approx(30)


def test_pre_pick_missing_object():
    out = check_pre_pick(CFG.pick_points["slot_1"], "cube", [], CFG.thresholds, 90)
    assert not out.passed and "no cube" in out.reason


def test_pre_pick_ignores_low_confidence_and_other_class():
    pick = CFG.pick_points["slot_1"]
    dets = [Detection("cube", pick, confidence=0.1), Detection("lid", pick)]
    assert not check_pre_pick(pick, "cube", dets, CFG.thresholds, 90).passed


def test_post_place_right_wrong_dropped_never_picked():
    b = CFG.bins["bin_cubes"]
    pick = CFG.pick_points["slot_1"]
    assert check_post_place("cube", b, [Detection("cube", b.center)], CFG.thresholds, CFG.bins).passed
    wrong = check_post_place("cube", b, [Detection("cube", CFG.bins["bin_lids"].center)], CFG.thresholds, CFG.bins)
    assert not wrong.passed and "wrong bin" in wrong.reason
    stayed = check_post_place("cube", b, [Detection("cube", pick)], CFG.thresholds, CFG.bins, pick)
    assert not stayed.passed and "never left" in stayed.reason
    lost = check_post_place("cube", b, [], CFG.thresholds, CFG.bins, pick)
    assert not lost.passed and "dropped" in lost.reason


# ---------- comparator ----------
def test_watch_mode_alerts_and_escalates():
    c = Comparator(CFG, Mode.WATCH)
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    d1 = c.on_pre_pick([])
    assert d1.action == Action.ALERT and not d1.urgent
    d2 = c.on_grasp_closed(GraspState.EMPTY)
    assert d2.action == Action.ALERT and d2.urgent


def test_guide_mode_retry_then_stop():
    c = Comparator(CFG, Mode.GUIDE)
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    assert c.on_grasp_closed(GraspState.EMPTY).action == Action.RETRY
    assert c.on_grasp_closed(GraspState.EMPTY).action == Action.STOP


def test_guide_bad_place_stops():
    c = Comparator(CFG, Mode.GUIDE)
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    d = c.on_place_done([])
    assert d.outcome.check_point == CheckPoint.POST_PLACE and d.action == Action.STOP


def test_unknown_grasp_is_failure():
    c = Comparator(CFG, Mode.WATCH)
    c.start_trial(1, "lid", CFG.pick_points["slot_2"])
    assert not c.on_grasp_closed(GraspState.UNKNOWN).outcome.passed


# ---------- resync ----------
def test_resync_matches_new_and_missing():
    twin = {"a": ("cube", Pose2D(0, 0)), "b": ("cube", Pose2D(500, 500))}
    dets = [Detection("cube", Pose2D(5, 0)), Detection("lid", Pose2D(100, 100))]
    ups, missing = plan_resync(twin, dets)
    shifts = {u.twin_id: u.shift_mm for u in ups}
    assert shifts["a"] == pytest.approx(5)
    assert any(u.twin_id.startswith("new_") and u.cls == "lid" for u in ups)
    assert missing == ["b"]


# ---------- timesync ----------
def test_stamp_buffer_nearest_and_tolerance():
    buf = StampBuffer()
    for t in (0, 33, 66, 100):
        buf.push(t * 1_000_000, f"f{t}")
    m = buf.nearest(70_000_000, tolerance_ms=10)
    assert m.item == "f66" and m.gap_ms == pytest.approx(4)
    assert buf.nearest(500_000_000, tolerance_ms=10) is None
    buf.push(50_000_000, "late")
    assert buf.nearest(50_000_000, 1).item == "late"


# ---------- grasp ----------
def test_grasp_across_narrow_side_and_fit():
    g = plan_grasp(Pose2D(0, 0, 30), CFG.objects["vial"], max_opening_mm=40)
    assert g.fits and g.pose.yaw_deg == pytest.approx(120)
    too_big = plan_grasp(Pose2D(0, 0, 0), CFG.objects["vial"], max_opening_mm=15)
    assert not too_big.fits


# ---------- stats ----------
def test_wilson_matches_doc_table():
    lo, hi = wilson_interval(18, 20)
    assert 0.68 < lo < 0.72 and 0.96 < hi < 0.98
    lo, hi = wilson_interval(90, 100)
    assert 0.82 < lo < 0.84 and 0.94 < hi < 0.96


def test_threshold_sweep_picks_full_recall():
    rows = threshold_sweep([25, 30, 40], [2, 3, 12], [5, 10, 15, 20, 28])
    best = pick_threshold(rows)
    assert best["threshold"] == 20 and best["false_alarms"] == 0
