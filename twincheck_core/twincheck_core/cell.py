"""The organizers' cell: frames, our workspace guard, and fingertip paths for top-down picks.

The cell plays rows [x, y, z, w, p, r, gripper] of the fingertip (UTOOL1) in user frame UF0, one row
per 0.1 s step. UF0 x, y are the robot base's (= our table frame); z_uf0 = z_base - 330 mm, so the
work surface is at about -335 mm. w, p, r = -180, 0, R points the fingers down and R turns them.
All numbers come from config/twincheck.yaml `cell:`; units are mm and degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config
from .geometry import wrap_deg

Row = list[float]   # [x, y, z, w, p, r, gripper], UF0 mm / deg, gripper 0 open / 1 closed


@dataclass(frozen=True)
class CellSpec:
    uf0_z_offset_mm: float
    surface_z_base_mm: float
    reach_mm: float
    reach_margin_mm: float
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    w_deg: float
    p_deg: float
    finger_yaw_offset_deg: float
    max_step_mm: float
    max_step_deg: float
    settle_rows: int
    grip_rows: int
    hover_mm: float
    transit_mm: float
    tip_clearance_mm: float
    release_above_rim_mm: float
    bin_height_mm: float
    park: tuple[float, float, float]
    park_r_deg: float
    max_wrist_turn_deg: float
    preferred_r_deg: float | None
    r_window_deg: tuple[float, float] | None
    max_r_tilt_deg: float
    arrive_tol_mm: float
    arrive_tol_deg: float
    arrive_timeout_rows: int
    wait_chunk_rows: int

    @classmethod
    def from_config(cls, cfg: Config) -> "CellSpec":
        c = cfg.raw["cell"]
        b = c["tcp_box_mm"]
        w, p = c["fingers_down_wp_deg"]
        return cls(float(c["uf0_z_offset_mm"]), float(c["surface_z_base_mm"]), float(c["reach_mm"]),
                   float(c["reach_margin_mm"]), (tuple(b["x"]), tuple(b["y"]), tuple(b["z"])), float(w), float(p),
                   float(c["finger_yaw_offset_deg"]), float(c["max_step_mm"]), float(c["max_step_deg"]),
                   int(c["settle_rows"]), int(c["grip_rows"]), float(c["hover_mm"]), float(c["transit_mm"]),
                   float(c["tip_clearance_mm"]), float(c["release_above_rim_mm"]), float(c["bin_height_mm"]),
                   tuple(float(v) for v in c["park_tcp_mm"]), float(c.get("park_r_deg", 0.0)),
                   float(c.get("max_wrist_turn_deg", 1e9)),
                   None if c.get("preferred_r_deg") is None else float(c["preferred_r_deg"]),
                   None if c.get("r_window_deg") is None else tuple(float(v) for v in c["r_window_deg"]),
                   float(c.get("max_r_tilt_deg", 0.0)), float(c["arrive_tol_mm"]),
                   float(c["arrive_tol_deg"]), round(float(c["arrive_timeout_s"]) / float(c["step_s"])),
                   int(c["wait_chunk_rows"]))

    def arrived(self, tcp: list[float], goal: Row) -> bool:
        """The measured fingertip pose (obs tcp_pose, UF0) is at the goal row's position and R."""
        return (math.dist(tcp[:3], goal[:3]) <= self.arrive_tol_mm
                and abs(wrap_deg(tcp[5] - goal[5])) <= self.arrive_tol_deg)

    @property
    def surface_z(self) -> float:
        """The work surface in UF0 (mm)."""
        return self.surface_z_base_mm + self.uf0_z_offset_mm

    def z_above_surface(self, mm: float) -> float:
        return self.surface_z + mm

    def r_for_close_yaw(self, close_yaw_deg: float) -> float:
        """Tool R that makes the fingers close along close_yaw (table frame)."""
        return wrap_deg(close_yaw_deg - self.finger_yaw_offset_deg)

    def r_near_preferred(self, r: float, symmetry_deg: float, max_tilt_deg: float | None = None) -> float:
        """Of the tool R values that grasp the same part (r plus whole turns of its symmetry), the one closest
        to preferred_r_deg. On the real cell all 3 controller faults of Oct 28 came at R 95-99, with the air hoses
        hanging beside the fingers; all 17 good grasps were at R 107-179."""
        if self.preferred_r_deg is None or not symmetry_deg:
            return r
        k = round(wrap_deg(self.preferred_r_deg - r) / symmetry_deg)
        best = wrap_deg(r + k * symmetry_deg)
        if self.r_window_deg is None:
            return best
        # Inside r_window_deg there has been no fault in about 40 real grasps (Oct 28 and 30); the faults came
        # at R 95-99 and at R 191-194. A candidate inside wins; else, if tilting the jaws by at most
        # max_r_tilt_deg reaches the window, tilt (a 10 mm cube 5 deg off is 10.8 mm across: the jaws open 12).
        lo, hi = self.r_window_deg
        tilt = self.max_r_tilt_deg if max_tilt_deg is None else max_tilt_deg
        cands = [wrap_deg(r + j * symmetry_deg) for j in range(-4, 5)]
        inside = [c for c in cands if lo <= c % 360 <= hi]
        if inside:
            return min(inside, key=lambda c: abs(wrap_deg(c - self.preferred_r_deg)))
        for c in sorted(cands, key=lambda c: min(abs(wrap_deg(c - lo)), abs(wrap_deg(c - hi)))):
            edge = lo if abs(wrap_deg(c - lo)) <= abs(wrap_deg(c - hi)) else hi
            if abs(wrap_deg(c - edge)) <= tilt:
                return wrap_deg(edge)
            break
        return best

    def down(self, x: float, y: float, z: float, r: float, gripper: float) -> Row:
        return [x, y, z, self.w_deg, self.p_deg, r, gripper]


class OutsideWorkspace(ValueError):
    pass


def check_row(spec: CellSpec, row: Row) -> None:
    """Raise OutsideWorkspace if the cell would reject or fault on this row, or it would hit the table.
    Our limits sit inside the cell's: never rely on the cell to stop a bad target."""
    if len(row) != 7 or not all(math.isfinite(v) for v in row):
        raise OutsideWorkspace(f"bad row {row}")
    for k, (lo, hi) in enumerate(spec.box):
        if not lo <= row[k] <= hi:
            raise OutsideWorkspace(f"{'xyz'[k]}={row[k]:.0f} mm outside [{lo:.0f}, {hi:.0f}]")
    reach = math.hypot(row[0], row[1])
    if reach > spec.reach_mm - spec.reach_margin_mm:
        raise OutsideWorkspace(f"{reach:.0f} mm from the base axis (limit {spec.reach_mm - spec.reach_margin_mm:.0f})")
    if row[2] < spec.z_above_surface(spec.tip_clearance_mm) - 1e-6:
        raise OutsideWorkspace(f"z={row[2]:.1f} mm is below the grasp height (table at {spec.surface_z:.1f})")


def interpolate(spec: CellSpec, start: Row, goal: Row) -> list[Row]:
    """Rows from start to goal (goal included), each step at most max_step_mm and max_step_deg.
    Angles take the short way round (w = 179 to -180 is 1 degree, not 359). The gripper value is the goal's."""
    dist = math.dist(start[:3], goal[:3])
    dang = [wrap_deg(g - s) for s, g in zip(start[3:6], goal[3:6])]
    n = max(1, math.ceil(dist / spec.max_step_mm), math.ceil(max(abs(a) for a in dang) / spec.max_step_deg))
    rows = []
    for i in range(1, n + 1):
        f = i / n
        pos = [s + (g - s) * f for s, g in zip(start[:3], goal[:3])]
        ang = [wrap_deg(s + a * f) for s, a in zip(start[3:6], dang)]
        rows.append(pos + ang + [goal[6]])
    return rows
