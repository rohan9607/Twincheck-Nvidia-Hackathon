"""Grasp planning for Guide mode: where to close the two fingers.

A two-finger gripper closes along one axis. We grasp across the object's
narrow side, so the finger axis is perpendicular to the object's long axis.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ObjectSpec
from .geometry import Pose2D, wrap_deg


@dataclass(frozen=True)
class Grasp:
    pose: Pose2D          # gripper center and finger-closing direction, table frame
    opening_mm: float     # how wide the object is across the fingers
    fits: bool            # does it fit in the jaws?
    reason: str = ""


def plan_grasp(obj_pose: Pose2D, spec: ObjectSpec, max_opening_mm: float, clearance_mm: float = 4.0) -> Grasp:
    """Close across the narrow side; refuse objects wider than the open jaws."""
    narrow = min(spec.width_mm, spec.length_mm) if spec.length_mm else spec.width_mm
    # finger-closing axis = object yaw + 90 (across the long axis); round objects: any angle
    close_yaw = 0.0 if spec.symmetry_deg <= 0 else wrap_deg(obj_pose.yaw_deg + 90.0)
    needed = narrow + clearance_mm
    if narrow and needed > max_opening_mm:
        return Grasp(Pose2D(obj_pose.x_mm, obj_pose.y_mm, close_yaw), narrow, False,
                     f"{spec.name} needs {needed:.0f} mm, jaws open to {max_opening_mm:.0f} mm")
    return Grasp(Pose2D(obj_pose.x_mm, obj_pose.y_mm, close_yaw), narrow, True)
