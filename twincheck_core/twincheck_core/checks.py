"""The three TwinCheck checks: pre-pick, post-grasp, post-place.

Each check compares an expected value with an observed one and returns a
CheckOutcome. These functions are pure (no ROS, no I/O) so they are unit
tested and shared by the ROS nodes, the twin trial runner and the analysis.
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from enum import Enum

from .config import BinSpec, Thresholds
from .geometry import Pose2D, yaw_diff_deg


class CheckPoint(str, Enum):
    PRE_PICK = "pre_pick"
    POST_GRASP = "post_grasp"
    POST_PLACE = "post_place"


class GraspState(str, Enum):
    HOLDING = "holding"
    EMPTY = "empty"
    MISALIGNED = "misaligned"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Detection:
    cls: str
    pose: Pose2D
    confidence: float = 1.0
    camera: str = ""
    obj_id: str = ""


@dataclass(frozen=True)
class CheckOutcome:
    check_point: CheckPoint
    passed: bool
    reason: str
    position_error_mm: float | None = None
    yaw_error_deg: float | None = None
    observed: Detection | None = None


def nearest_detection(target: Pose2D, detections: list[Detection], cls: str | None,
                      radius_mm: float, min_confidence: float = 0.0) -> Detection | None:
    """Closest detection of the class within radius (the object this check is about)."""
    best, best_d = None, radius_mm
    for d in detections:
        if cls is not None and d.cls != cls:
            continue
        if d.confidence < min_confidence:
            continue
        dist = d.pose.distance_to(target)
        if dist <= best_d:
            best, best_d = d, dist
    return best


def check_pre_pick(expected_pick: Pose2D, cls: str, detections: list[Detection], th: Thresholds,
                   symmetry_deg: float) -> CheckOutcome:
    """Is the object actually where the robot is about to grab?"""
    obs = nearest_detection(expected_pick, detections, cls, th.match_radius_mm, th.min_confidence)
    if obs is None:
        return CheckOutcome(CheckPoint.PRE_PICK, False,
                            f"no {cls} within {th.match_radius_mm:.0f} mm of the pick point")
    d = obs.pose.distance_to(expected_pick)
    a = yaw_diff_deg(obs.pose.yaw_deg, expected_pick.yaw_deg, symmetry_deg)
    if d > th.pre_pick_mm or a > th.pre_pick_deg:
        return CheckOutcome(CheckPoint.PRE_PICK, False,
                            f"{cls} is off the pick point by {d:.0f} mm, {a:.0f} deg; this pick will likely miss",
                            d, a, obs)
    return CheckOutcome(CheckPoint.PRE_PICK, True, "object on the pick point", d, a, obs)


def check_post_grasp(observed: GraspState, expected: GraspState = GraspState.HOLDING) -> CheckOutcome:
    """Is the object really in the gripper? UNKNOWN fails: a missed failure costs more than a false alarm."""
    if observed == expected:
        return CheckOutcome(CheckPoint.POST_GRASP, True, "object held")
    reasons = {
        GraspState.EMPTY: "missed grasp: gripper is empty",
        GraspState.MISALIGNED: "object held badly: misaligned in the gripper",
        GraspState.UNKNOWN: "grasp state unknown: treating as a failure",
    }
    return CheckOutcome(CheckPoint.POST_GRASP, False, reasons.get(observed, f"grasp state {observed.value}"))


def grasp_from_table(picked_at: Pose2D, cls: str, detections: list[Detection], th: Thresholds) -> GraspState:
    """Grasp state from the table camera, with the arm parked after lifting (the stand-in until a wrist check
    is proven). A part of this class still near where we grasped means the jaws came up empty, whether it
    stayed put or was knocked a little; none there means it is in the jaws. A detector miss reads as held,
    so the post-place check is the backstop for that."""
    if nearest_detection(picked_at, detections, cls, th.match_radius_mm, th.min_confidence) is not None:
        return GraspState.EMPTY
    return GraspState.HOLDING


def newest(now: list[Detection], before: list[Detection]) -> Detection:
    """The detection least explained by the earlier scene: farthest from any old part of its class."""
    def gap(d: Detection) -> float:
        return min((b.pose.distance_to(d.pose) for b in before if b.cls == d.cls), default=math.inf)
    return max(now, key=gap)


def check_post_place(cls: str, target_bin: BinSpec, detections: list[Detection], th: Thresholds,
                     all_bins: dict[str, BinSpec] | None = None,
                     pick_point: Pose2D | None = None,
                     before: list[Detection] | None = None) -> CheckOutcome:
    """Did the object land in the right bin?

    Looks for a `cls` that arrived in the target bin. If none, says where it went:
    another bin, still at the pick point (never picked), or nowhere visible (dropped).

    Bins fill up during a run, so "is there a vial in the rework bin?" is true for every
    pick after the first. With `before` (the scene at pre-pick), a place counts only if the
    number of `cls` in the region went UP. Parts dropped on the same spot pile up, so we
    count instead of matching positions. If the detector sees a pile as fewer parts, the
    count does not rise and the check fails: a false alarm, never a missed failure.
    Without `before`, any `cls` in the bin counts (an empty-bin start).

    In the target bin a new part of any class counts too: we know what we carried, and a part half hidden
    in a pocket is often seen as the other class (a screw leaning in its bin read as cubes in the twin).
    """
    old = before or []

    def arrived(region, any_cls: bool = False) -> list[Detection]:
        def ours(d: Detection) -> bool:
            return (any_cls or d.cls == cls) and d.confidence >= th.min_confidence and region(d.pose)
        now, was = [d for d in detections if ours(d)], [d for d in old if ours(d)]
        return now if len(now) > len(was) else []

    for any_cls in (False, True):
        in_target = arrived(target_bin.contains, any_cls)
        if in_target:
            best = newest(in_target, old)
            err = best.pose.distance_to(target_bin.center)
            seen = f" (seen as {best.cls})" if best.cls != cls else ""
            return CheckOutcome(CheckPoint.POST_PLACE, True, f"{cls} in {target_bin.name}{seen}", err, None, best)

    near_margin = arrived(lambda p: target_bin.contains(p, th.post_place_mm) and not target_bin.contains(p))
    if near_margin:
        d = newest(near_margin, old)
        return CheckOutcome(CheckPoint.POST_PLACE, False, f"{cls} on the edge of {target_bin.name}",
                            d.pose.distance_to(target_bin.center), None, d)

    for other in (all_bins or {}).values():
        if other.name == target_bin.name:
            continue
        wrong = arrived(other.contains)
        if wrong:
            return CheckOutcome(CheckPoint.POST_PLACE, False,
                                f"{cls} in the wrong bin ({other.name}, expected {target_bin.name})",
                                newest(wrong, old).pose.distance_to(target_bin.center), None, newest(wrong, old))

    if pick_point is not None:   # full scene: a part that never moved is "old" by definition
        left = nearest_detection(pick_point, detections, cls, th.match_radius_mm, th.min_confidence)
        if left is not None:
            return CheckOutcome(CheckPoint.POST_PLACE, False, f"{cls} never left the pick area", None, None, left)

    return CheckOutcome(CheckPoint.POST_PLACE, False, f"{cls} not found in {target_bin.name}: dropped or lost")
