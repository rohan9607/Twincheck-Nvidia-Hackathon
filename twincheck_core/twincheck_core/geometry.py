"""2D table-frame geometry.

All poses are in the `table` frame: millimeters for position, degrees for yaw.
Objects lying on a flat table only need (x, y, yaw), so we never use full 3D here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x_mm: float
    y_mm: float
    yaw_deg: float = 0.0

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x_mm - other.x_mm, self.y_mm - other.y_mm)


def wrap_deg(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def yaw_diff_deg(a: float, b: float, symmetry_deg: float = 360.0) -> float:
    """Smallest absolute yaw difference, respecting rotational symmetry.

    symmetry_deg is the angle after which the object looks identical:
      360 -> no symmetry, 180 -> a lying vial or a rectangle, 90 -> a cube,
      0   -> round from above (a standing vial, a lid): yaw never matters.
    """
    if symmetry_deg <= 0:
        return 0.0
    d = abs(a - b) % symmetry_deg
    return min(d, symmetry_deg - d)


def canonical_yaw(yaw: float, symmetry_deg: float = 360.0) -> float:
    """Map a yaw into [0, symmetry) so symmetric poses get one label."""
    if symmetry_deg <= 0:
        return 0.0
    return yaw % symmetry_deg


def apply_homography(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Map a pixel (u, v) to table (x_mm, y_mm) with a 3x3 homography."""
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def pixel_yaw_to_table(H: np.ndarray, u: float, v: float, yaw_px_deg: float, step_px: float = 20.0) -> float:
    """Convert an image-space angle at (u, v) to a table-frame angle.

    A homography can rotate and shear, so map a short segment along the
    image angle and measure its direction on the table.
    """
    r = math.radians(yaw_px_deg)
    x0, y0 = apply_homography(H, u, v)
    x1, y1 = apply_homography(H, u + step_px * math.cos(r), v + step_px * math.sin(r))
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def fuse_poses(poses: list[Pose2D], weights: list[float] | None = None, symmetry_deg: float = 360.0) -> Pose2D:
    """Fuse the same object seen by several cameras into one pose.

    Positions: weighted mean. Yaw: circular mean in the symmetry-reduced space,
    so a cube seen at 1 deg and 89 deg fuses near 0/90, not 45.
    """
    if not poses:
        raise ValueError("no poses to fuse")
    w = np.ones(len(poses)) if weights is None else np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(poses))
    w = w / w.sum()
    x = float(np.dot(w, [p.x_mm for p in poses]))
    y = float(np.dot(w, [p.y_mm for p in poses]))
    if symmetry_deg <= 0:
        return Pose2D(x, y, 0.0)
    k = 360.0 / symmetry_deg  # stretch the period to a full circle
    ang = np.radians([p.yaw_deg * k for p in poses])
    s, c = float(np.dot(w, np.sin(ang))), float(np.dot(w, np.cos(ang)))
    yaw = math.degrees(math.atan2(s, c)) / k
    return Pose2D(x, y, canonical_yaw(yaw, symmetry_deg))
