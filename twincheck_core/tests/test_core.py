import math
from pathlib import Path

import numpy as np
import pytest

from twincheck_core import (Action, Comparator, Detection, GraspState, Mode, Pose2D, StampBuffer, apply_homography,
                            check_post_place, check_pre_pick, fuse_poses, load_config, pick_threshold, plan_grasp,
                            plan_resync, threshold_sweep, wilson_interval, yaw_diff_deg)
from twincheck_core.cell import CellSpec, OutsideWorkspace, check_row, interpolate
from twincheck_core.checks import CheckPoint, grasp_from_table
from twincheck_core.pickloop import Job, Need, PickLoop, View

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


def test_pre_pick_screw_seen_head_for_tip_passes():
    # the detector's box has no head/tip direction: a screw found 180 deg around is the same screw
    pick = CFG.pick_points["slot_2"]
    flipped = [Detection("screw", Pose2D(pick.x_mm, pick.y_mm, pick.yaw_deg + 178))]
    assert check_pre_pick(pick, "screw", flipped, CFG.thresholds, CFG.symmetry("screw")).passed


def test_pre_pick_missing_object():
    out = check_pre_pick(CFG.pick_points["slot_1"], "cube", [], CFG.thresholds, 90)
    assert not out.passed and "no cube" in out.reason


def test_pre_pick_ignores_low_confidence_and_other_class():
    pick = CFG.pick_points["slot_1"]
    dets = [Detection("cube", pick, confidence=0.1), Detection("screw", pick)]
    assert not check_pre_pick(pick, "cube", dets, CFG.thresholds, 90).passed


def test_post_place_right_wrong_dropped_never_picked():
    b = CFG.bins["bin_cubes"]
    pick = CFG.pick_points["slot_1"]
    assert check_post_place("cube", b, [Detection("cube", b.center)], CFG.thresholds, CFG.bins).passed
    wrong = check_post_place("cube", b, [Detection("cube", CFG.bins["bin_screws"].center)], CFG.thresholds, CFG.bins)
    assert not wrong.passed and "wrong bin" in wrong.reason
    stayed = check_post_place("cube", b, [Detection("cube", pick)], CFG.thresholds, CFG.bins, pick)
    assert not stayed.passed and "never left" in stayed.reason
    lost = check_post_place("cube", b, [], CFG.thresholds, CFG.bins, pick)
    assert not lost.passed and "dropped" in lost.reason


def test_post_place_ignores_parts_already_in_the_bin():
    b, screws = CFG.bins["bin_cubes"], CFG.bins["bin_screws"]
    pick = CFG.pick_points["slot_1"]
    old = Detection("cube", Pose2D(b.center.x_mm + 8, b.center.y_mm))       # from an earlier pick (pocket is 35 mm)
    before = [old, Detection("cube", pick)]
    th = CFG.thresholds
    # dropped on the way: only the old cube is in the bin -> must fail
    lost = check_post_place("cube", b, [old], th, CFG.bins, pick, before)
    assert not lost.passed and "dropped" in lost.reason
    # wrong bin while an old cube sits in the right one -> must fail
    wrong = check_post_place("cube", b, [old, Detection("cube", screws.center)], th, CFG.bins, pick, before)
    assert not wrong.passed and "wrong bin" in wrong.reason
    # the new cube arrived -> pass, and it is the new one that is reported
    ok = check_post_place("cube", b, [old, Detection("cube", b.center)], th, CFG.bins, pick, before)
    assert ok.passed and ok.observed.pose == b.center
    # a fixed program drops on the same spot: new cube 3 mm from the old one, both seen -> pass
    piled = check_post_place("cube", b, [old, Detection("cube", Pose2D(old.pose.x_mm + 3, old.pose.y_mm))], th,
                             CFG.bins, pick, before)
    assert piled.passed
    # new cube hidden by / merged with the old one: count did not rise -> fail (false alarm, not a miss)
    stacked = check_post_place("cube", b, [Detection("cube", Pose2D(old.pose.x_mm + 3, old.pose.y_mm))], th,
                               CFG.bins, pick, before)
    assert not stacked.passed


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


def test_guide_retry_keeps_budget_across_pre_picks():
    # Guide re-sends PRE_PICK for the same trial when it retries: the budget must not reset
    c = Comparator(CFG, Mode.GUIDE)
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    assert c.on_grasp_closed(GraspState.EMPTY).action == Action.RETRY
    moved = Pose2D(CFG.pick_points["slot_1"].x_mm + 25, CFG.pick_points["slot_1"].y_mm)
    c.start_trial(1, "cube", moved)                         # the retry, aimed at the fresh detection
    assert c.on_pre_pick([Detection("cube", moved)]).outcome.passed
    assert c.on_grasp_closed(GraspState.EMPTY).action == Action.STOP
    c.start_trial(2, "cube", CFG.pick_points["slot_1"])     # a new trial starts fresh
    assert c.on_grasp_closed(GraspState.EMPTY).action == Action.RETRY


def test_guide_bad_place_stops():
    c = Comparator(CFG, Mode.GUIDE)
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    d = c.on_place_done([])
    assert d.outcome.check_point == CheckPoint.POST_PLACE and d.action == Action.STOP
    c = Comparator(CFG, Mode.GUIDE, stop_on_bad_place=False)       # the cell: alert and go on
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    d = c.on_place_done([])
    assert not d.outcome.passed and d.action == Action.ALERT


def test_post_place_counts_any_new_part_in_the_target_bin():
    screws, cubes = CFG.bins["bin_screws"], CFG.bins["bin_cubes"]
    pick = CFG.pick_points["slot_2"]
    th = CFG.thresholds
    seen_as_cubes = [Detection("cube", Pose2D(screws.center.x_mm - 5, screws.center.y_mm - 10)),
                     Detection("cube", Pose2D(screws.center.x_mm + 9, screws.center.y_mm - 5))]
    ok = check_post_place("screw", screws, seen_as_cubes, th, CFG.bins, pick, [Detection("screw", pick)])
    assert ok.passed and "seen as cube" in ok.reason
    # a cube already in the cube bin does not make a lost screw pass, and the wrong bin still fails
    lost = check_post_place("screw", screws, [Detection("cube", cubes.center)], th, CFG.bins, pick,
                            [Detection("cube", cubes.center)])
    assert not lost.passed
    wrong = check_post_place("screw", screws, [Detection("screw", cubes.center)], th, CFG.bins, pick, [])
    assert not wrong.passed and "wrong bin" in wrong.reason


def test_comparator_post_place_uses_pre_pick_scene():
    c = Comparator(CFG, Mode.WATCH)
    b = CFG.bins["bin_cubes"]
    old = Detection("cube", b.center)
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    c.on_pre_pick([old, Detection("cube", CFG.pick_points["slot_1"])])
    assert not c.on_place_done([old]).outcome.passed      # only the old cube in the bin: this pick was lost


def test_comparator_scene_snapshot_guards_against_flicker():
    c = Comparator(CFG, Mode.WATCH)
    b = CFG.bins["bin_cubes"]
    old, part = Detection("cube", b.center), Detection("cube", CFG.pick_points["slot_1"])
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    # the check frame missed the old cube, the fuller nearby frame has it
    c.on_pre_pick([part], scene=[old, part])
    assert not c.on_place_done([old]).outcome.passed      # lost part must not pass on the old cube


def test_late_check_is_judged_against_its_own_trial():
    # post-place of trial 1 finishes after trial 2 (a vial) has started: it must still be about the cube
    c = Comparator(CFG, Mode.WATCH)
    cubes, rework = CFG.bins["bin_cubes"], CFG.bins["bin_rework"]
    c.start_trial(1, "cube", CFG.pick_points["slot_3"])
    c.on_pre_pick([Detection("cube", CFG.pick_points["slot_3"])], trial_id=1)
    c.start_trial(2, "vial", CFG.pick_points["slot_1"])
    c.on_pre_pick([Detection("vial", CFG.pick_points["slot_1"])], trial_id=2)
    late = c.on_place_done([Detection("cube", rework.center), Detection("vial", rework.center)], trial_id=1)
    assert late.trial_id == 1 and not late.outcome.passed and "wrong bin" in late.outcome.reason
    assert c.on_place_done([Detection("cube", cubes.center)], trial_id=1).outcome.passed


def test_is_active_only_for_the_started_trial():
    c = Comparator(CFG, Mode.WATCH)
    assert not c.is_active(1)          # grasp/place before any pre-pick: skip, don't crash
    c.start_trial(1, "cube", CFG.pick_points["slot_1"])
    assert c.is_active(1) and not c.is_active(0) and not c.is_active(2)


def test_unknown_grasp_is_failure():
    c = Comparator(CFG, Mode.WATCH)
    c.start_trial(1, "screw", CFG.pick_points["slot_2"])
    assert not c.on_grasp_closed(GraspState.UNKNOWN).outcome.passed


# ---------- resync ----------
def test_resync_matches_new_and_missing():
    twin = {"a": ("cube", Pose2D(0, 0)), "b": ("cube", Pose2D(500, 500))}
    dets = [Detection("cube", Pose2D(5, 0)), Detection("screw", Pose2D(100, 100))]
    ups, missing = plan_resync(twin, dets)
    shifts = {u.twin_id: u.shift_mm for u in ups}
    assert shifts["a"] == pytest.approx(5)
    assert any(u.twin_id.startswith("new_") and u.cls == "screw" for u in ups)
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
    assert [m.item for m in buf.within(50_000_000, 20)] == ["f33", "late", "f66"]


def test_stamp_buffer_restarts_after_clock_step_back():
    buf = StampBuffer(clock_step_ms=500)
    for t in (1000, 1033, 1066):
        buf.push(t * 1_000_000, f"before{t}")
    buf.push(400 * 1_000_000, "after")          # clock jumped back ~0.7 s
    assert buf.clock_steps == 1 and len(buf) == 1
    assert buf.nearest(1033 * 1_000_000, 100) is None   # never match an event to a pre-jump frame


# ---------- grasp ----------
def test_grasp_across_narrow_side_and_fit():
    g = plan_grasp(Pose2D(0, 0, 30), CFG.objects["vial"], max_opening_mm=40)
    assert g.fits and g.pose.yaw_deg == pytest.approx(120)
    too_big = plan_grasp(Pose2D(0, 0, 0), CFG.objects["vial"], max_opening_mm=15)
    assert not too_big.fits


def test_real_jaws_take_small_cube_and_screw_only():
    jaw, slack = CFG.raw["gripper"]["max_opening_mm"], CFG.raw["gripper"]["clearance_mm"]
    fits = {name: plan_grasp(Pose2D(0, 0), spec, jaw, slack).fits for name, spec in CFG.objects.items()}
    assert fits == {"cube": True, "screw": True, "cube_16": False, "vial": False, "cap": False}


# ---------- cell (organizers' robot) ----------
SPEC = CellSpec.from_config(CFG)


def test_check_row_keeps_targets_inside_our_limits():
    z = SPEC.z_above_surface(50)
    check_row(SPEC, SPEC.down(400, 0, z, 0, 0))
    for bad in (SPEC.down(400, 400, z, 0, 0),            # 566 mm from the base axis: controller fault
                SPEC.down(100, 0, z, 0, 0),              # outside the cell's TCP box
                SPEC.down(400, 0, SPEC.surface_z, 0, 0),  # fingertips into the table
                SPEC.down(400, 0, float("nan"), 0, 0)):
        with pytest.raises(OutsideWorkspace):
            check_row(SPEC, bad)


def test_interpolate_limits_steps_and_takes_short_way_round():
    start = [400, 0, -250, 179.0, 0, 0, 0]
    rows = interpolate(SPEC, start, [400, 100, -250, -180.0, 0, 0, 1])
    assert len(rows) == 7 and rows[-1][:3] == [400, 100, -250] and rows[-1][6] == 1
    assert all(abs(r[3]) >= 179 for r in rows)                    # 1 degree, not 359
    steps = [math.dist(a[:3], b[:3]) for a, b in zip([start] + rows, rows)]
    assert max(steps) <= SPEC.max_step_mm + 1e-9


def _rows_ok(rows):
    for r in rows:
        check_row(SPEC, r)
    return rows


def test_pickloop_guide_aims_at_seen_part_and_places_it():
    nominal = CFG.pick_points["slot_1"]
    seen = Pose2D(nominal.x_mm + 30, nominal.y_mm - 20, 20)       # nudged: a blind program would miss it
    loop = PickLoop(CFG, "guide", [Job("cube", nominal)], grasp_by="wrist")
    _rows_ok(loop.step(View(tcp=None)))                            # mock: no tcp yet -> start from park
    assert loop.needs == Need.SCENE
    rows = _rows_ok(loop.step(View(tcp=None, detections=[Detection("cube", seen)])))
    low = min(rows, key=lambda r: r[2])
    assert math.hypot(low[0] - seen.x_mm, low[1] - seen.y_mm) < 1e-6 and low[2] == SPEC.z_above_surface(2.5)
    assert any(r[6] == 1 for r in rows) and loop.needs == Need.GRASP
    rows = _rows_ok(loop.step(View(tcp=None, grasp=GraspState.HOLDING)))
    b = CFG.bins["bin_cubes"].center
    opened = next(i for i, r in enumerate(rows) if r[6] == 0)
    assert math.hypot(rows[opened][0] - b.x_mm, rows[opened][1] - b.y_mm) < 1e-6
    assert loop.needs == Need.SCENE
    loop.step(View(tcp=None, detections=[Detection("cube", b)]))
    assert loop.done
    ev = loop.drain_events()
    assert [e.decision.outcome.passed for e in ev if e.kind == "decision"] == [True, True, True]
    assert ev[-1].kind == "done"


def test_pickloop_watch_stays_blind_and_alerts():
    nominal = CFG.pick_points["slot_1"]
    loop = PickLoop(CFG, "watch", [Job("cube", nominal)], grasp_by="wrist")
    loop.step(View(tcp=None))
    rows = loop.step(View(tcp=None, detections=[Detection("cube", Pose2D(nominal.x_mm + 30, nominal.y_mm))]))
    low = min(rows, key=lambda r: r[2])
    assert (low[0], low[1]) == (nominal.x_mm, nominal.y_mm)       # reaches for the taught point anyway
    d = [e.decision for e in loop.drain_events() if e.kind == "decision"][0]
    assert not d.outcome.passed and d.action == Action.ALERT


def test_pickloop_guide_retries_empty_grasp_once_then_stops():
    nominal = CFG.pick_points["slot_2"]
    loop = PickLoop(CFG, "guide", [Job("cube", nominal), Job("cube", nominal)], grasp_by="wrist")
    scene = [Detection("cube", nominal)]
    loop.step(View(tcp=None))
    loop.step(View(tcp=None, detections=scene))
    loop.step(View(tcp=None, grasp=GraspState.EMPTY))             # retry: drop, park, look again
    assert loop.needs == Need.SCENE and not loop.done
    rows = loop.step(View(tcp=None, detections=scene))
    lifted = rows[-1]
    rows = loop.step(View(tcp=None, grasp=GraspState.EMPTY))      # second miss: stop and hold
    assert loop.done and rows[-1][6] == 0
    # the jaws open only after setting down at the pick point, never at lift height (a held part would drop)
    opened = next(r for r in rows if r[6] == 0)
    assert opened[:2] == lifted[:2] and opened[2] == SPEC.z_above_surface(SPEC.tip_clearance_mm)
    _rows_ok(rows)
    ev = loop.drain_events()
    assert [e.kind for e in ev].count("trial") == 1 and ev[-1].kind == "stop"
    assert loop.step(View(tcp=rows[-1][:6], gripper=0)) == [rows[-1]]   # done: holds still


def test_pickloop_refuses_unreachable_target_and_holds():
    loop = PickLoop(CFG, "baseline", [Job("cube", Pose2D(600, 300))], grasp_by="wrist")   # 671 mm from the base axis
    loop.step(View(tcp=None))
    rows = _rows_ok(loop.step(View(tcp=None, detections=[])))
    assert loop.done and "outside our workspace" in loop.drain_events()[-1].note
    assert rows


def test_pickloop_skips_parts_that_do_not_fit():
    loop = PickLoop(CFG, "guide", [Job("vial", CFG.pick_points["slot_1"]), Job("cube", CFG.pick_points["slot_2"])], grasp_by="wrist")
    loop.step(View(tcp=None))
    dets = [Detection("vial", CFG.pick_points["slot_1"]), Detection("cube", CFG.pick_points["slot_2"])]
    loop.step(View(tcp=None, detections=dets))
    assert loop.needs == Need.GRASP                                 # went on to the cube
    assert any(e.kind == "skip" and "jaws open to 12" in e.note for e in loop.drain_events())


class _SlowArm:
    """The real cell as the dry run recorded it (Oct 23): the fingertip heads straight for the LATEST row at
    about 55 mm/s, and the gripper follows its command at once, wherever the arm is."""

    def __init__(self, start, mm_per_row=5.5, floor_z=None):
        self.pos, self.r, self.g, self.v, self.floor_z = list(start[:3]), start[5], 0.0, mm_per_row, floor_z

    def play(self, rows):
        log = []
        for row in rows:
            d = math.dist(self.pos, row[:3])
            f = 1.0 if d <= self.v else self.v / d
            self.pos = [p + (q - p) * f for p, q in zip(self.pos, row[:3])]
            if self.floor_z is not None:                  # a part under the fingers
                self.pos[2] = max(self.pos[2], self.floor_z)
            self.r, self.g = row[5], row[6]
            log.append((list(self.pos), self.g))
        return log

    def tcp(self):
        return self.pos + [-180.0, 0.0, self.r]


@pytest.mark.parametrize("grasp_by", ["wrist", "table"])
def test_pickloop_waits_for_the_slow_real_arm(grasp_by):
    nominal = CFG.pick_points["slot_1"]
    loop = PickLoop(CFG, "guide", [Job("cube", nominal)], grasp_by=grasp_by)
    arm = _SlowArm(SPEC.down(*SPEC.park, 0.0, 0.0))
    grip = SPEC.z_above_surface(SPEC.tip_clearance_mm)
    b = CFG.bins["bin_cubes"].center
    closed_at, opened_at, looked_at = None, None, []
    for _ in range(2000):
        need = loop.needs
        view = View(tcp=arm.tcp(), gripper=arm.g)
        if need == Need.SCENE:
            looked_at.append(list(arm.pos))
            held = loop.phase.value == "grasped"                    # table grasp check: the cube is gone
            view.detections = [] if held else [Detection("cube", b if loop.placed_id else nominal)]
        elif need == Need.GRASP:
            view.grasp = GraspState.HOLDING
        before = arm.g
        log = arm.play(_rows_ok(loop.step(view)))
        for pos, g in log:
            if g != before and closed_at is None and g == 1:
                closed_at = pos
            if g != before and g == 0 and closed_at is not None and opened_at is None:
                opened_at = pos
            before = g
        if loop.done and not loop.needs:
            break
    assert loop.done
    # the jaws close only at the grasp height over the part, not on the way down (11 cm up on Oct 23)
    assert math.dist(closed_at, [nominal.x_mm, nominal.y_mm, grip]) <= SPEC.arrive_tol_mm
    assert math.hypot(opened_at[0] - b.x_mm, opened_at[1] - b.y_mm) <= SPEC.arrive_tol_mm
    # every camera look happens with the arm parked, out of the table camera's view
    assert looked_at and all(math.dist(p, SPEC.park) <= SPEC.arrive_tol_mm for p in looked_at)
    ev = loop.drain_events()
    assert [e.decision.outcome.passed for e in ev if e.kind == "decision"] == [True, True, True]


def test_pickloop_stops_when_the_arm_is_blocked():
    nominal = CFG.pick_points["slot_1"]
    loop = PickLoop(CFG, "watch", [Job("cube", nominal)], grasp_by="wrist")
    arm = _SlowArm(SPEC.down(*SPEC.park, 0.0, 0.0), floor_z=SPEC.z_above_surface(10))   # fingers on the cube
    for _ in range(2000):
        view = View(tcp=arm.tcp(), gripper=arm.g)
        if loop.needs == Need.SCENE:
            view.detections = [Detection("cube", nominal)]
        arm.play(_rows_ok(loop.step(view)))
        if loop.done:
            break
    assert loop.done and arm.g == 0                        # never closed on top of the part
    ev = loop.drain_events()
    assert ev[-1].kind == "stop" and "did not reach" in ev[-1].note


def test_grasp_from_table_part_still_there_is_empty():
    at = Pose2D(380, -100)
    th = CFG.thresholds
    assert grasp_from_table(at, "cube", [Detection("cube", Pose2D(381, -99))], th) == GraspState.EMPTY
    assert grasp_from_table(at, "cube", [Detection("cube", Pose2D(400, -100))], th) == GraspState.EMPTY  # knocked
    assert grasp_from_table(at, "cube", [Detection("screw", at), Detection("cube", Pose2D(380, 100))],
                            th) == GraspState.HOLDING          # other parts elsewhere don't count
    assert grasp_from_table(at, "cube", [], th) == GraspState.HOLDING


def test_pickloop_table_grasp_check_parks_looks_and_places():
    nominal = CFG.pick_points["slot_1"]
    loop = PickLoop(CFG, "guide", [Job("cube", nominal)], grasp_by="table")
    loop.step(View(tcp=None))
    rows = _rows_ok(loop.step(View(tcp=None, detections=[Detection("cube", nominal)])))
    assert loop.needs == Need.SCENE                                 # not the wrist: the table camera
    assert rows[-1][:3] == list(SPEC.park)                          # looked from the park pose
    rows = _rows_ok(loop.step(View(tcp=None, detections=[])))       # the cube left the table: held
    b = CFG.bins["bin_cubes"].center
    opened = next(r for r in rows if r[6] == 0)
    assert math.hypot(opened[0] - b.x_mm, opened[1] - b.y_mm) < 1e-6
    loop.step(View(tcp=None, detections=[Detection("cube", b)]))
    ev = loop.drain_events()
    assert loop.done and [e.decision.outcome.passed for e in ev if e.kind == "decision"] == [True, True, True]


def test_pickloop_table_grasp_check_retries_from_the_same_scene():
    nominal = CFG.pick_points["slot_1"]
    moved = Pose2D(nominal.x_mm + 4, nominal.y_mm - 3, 10)          # nudged by the failed grasp
    loop = PickLoop(CFG, "guide", [Job("cube", nominal)], grasp_by="table")
    loop.step(View(tcp=None))
    loop.step(View(tcp=None, detections=[Detection("cube", nominal)]))
    rows = _rows_ok(loop.step(View(tcp=None, detections=[Detection("cube", moved)])))   # still on the table
    assert rows[0][6] == 0 and rows[0][:3] == list(SPEC.park)       # opens where it is: nothing is held
    low = min(rows, key=lambda r: r[2])                             # and goes straight back for it
    assert (low[0], low[1]) == (moved.x_mm, moved.y_mm) and loop.needs == Need.SCENE
    rows = _rows_ok(loop.step(View(tcp=None, detections=[Detection("cube", moved)])))   # missed again
    ev = loop.drain_events()
    assert loop.done and ev[-1].kind == "stop" and all(r[2] > SPEC.z_above_surface(50) for r in rows)
    assert [e.kind for e in ev].count("trial") == 1


def test_bin_pocket_at_only_inside_the_pocket():
    b = CFG.bins["bin_cubes"]
    half = CFG.raw["cell"]["bin_pocket_mm"] / 2
    assert CFG.bin_pocket_at(b.center) is b
    assert CFG.bin_pocket_at(Pose2D(b.center.x_mm + half - 1, b.center.y_mm - half + 1)) is b
    assert CFG.bin_pocket_at(Pose2D(b.center.x_mm + half + 2, b.center.y_mm)) is None     # the wall, not the pocket
    assert CFG.bin_pocket_at(CFG.pick_points["slot_1"]) is None


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


# ---------- the operator's placement file ----------
def test_placement_gives_screw_head_and_only_used_bins():
    import random
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cell_policy"))
    from make_placement import layout
    centres, _ = layout(CFG, None, 0)
    heads, _ = layout(CFG, None, 0, heads=True)
    (cx, cy, yaw), (hx, hy, _) = centres["screw"][0], heads["screw"][0]
    half = CFG.objects["screw"].length_mm / 2
    assert (hx, hy) == (round(cx - half * math.cos(math.radians(yaw))), round(cy - half * math.sin(math.radians(yaw))))
    assert heads["cube"] == centres["cube"]                      # only screws have a head
    used = {CFG.objects[j["cls"]].target_bin for j in CFG.raw["cell"]["sequence"]}
    assert len(heads["bin"]) == len(used)
    nudged, _ = layout(CFG, random.Random(3), 30, heads=True)    # nudged screws: head half a length behind
    sx, sy, syaw = nudged["screw"][0]
    assert abs(math.hypot(sx - cx, sy - cy)) <= 30 + half + 1
