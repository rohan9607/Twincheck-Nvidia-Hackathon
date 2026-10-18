"""Resync: after each pick, move every twin object to where the cameras saw it.

Without this the twin's guesses drift away from reality pick after pick and
start raising false alarms (the problem Millard et al., IROS 2014, describe).
We also log how far each object had to move: a growing correction is an early
warning that calibration or the twin's physics is off.
"""
from __future__ import annotations

from dataclasses import dataclass

from .checks import Detection
from .geometry import Pose2D


@dataclass(frozen=True)
class ResyncUpdate:
    twin_id: str
    cls: str
    old: Pose2D | None
    new: Pose2D
    shift_mm: float | None


def plan_resync(twin_objects: dict[str, tuple[str, Pose2D]], detections: list[Detection],
                match_radius_mm: float = 80.0) -> tuple[list[ResyncUpdate], list[str]]:
    """Greedy nearest-neighbour matching of twin objects to detections, per class.

    twin_objects: {twin_id: (cls, pose)}
    Returns (updates, missing_twin_ids). Detections with no twin match become new
    objects with ids "new_<n>"; twin objects nobody saw are reported as missing.
    """
    pairs = []
    for tid, (cls, pose) in twin_objects.items():
        for i, d in enumerate(detections):
            if d.cls == cls:
                dist = pose.distance_to(d.pose)
                if dist <= match_radius_mm:
                    pairs.append((dist, tid, i))
    pairs.sort()

    used_t, used_d, updates = set(), set(), []
    for dist, tid, i in pairs:
        if tid in used_t or i in used_d:
            continue
        used_t.add(tid)
        used_d.add(i)
        cls, old = twin_objects[tid]
        updates.append(ResyncUpdate(tid, cls, old, detections[i].pose, dist))

    n = 0
    for i, d in enumerate(detections):
        if i not in used_d:
            updates.append(ResyncUpdate(f"new_{n}", d.cls, None, d.pose, None))
            n += 1
    missing = [tid for tid in twin_objects if tid not in used_t]
    return updates, missing
