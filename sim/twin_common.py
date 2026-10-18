"""Helpers shared by every Isaac Sim script. Run these scripts with Isaac Sim's python:

    ./python.sh sim/<script>.py --help          (Isaac Sim install)
    /isaac-sim/python.sh sim/<script>.py ...    (Isaac Sim container on the Spark / Brev)

Written for Isaac Sim 4.5 and 5.x (isaacsim.* modules). Older omni.isaac.* names are
tried as a fallback.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "twincheck_core"))


def load_sim_config(path: str | None = None) -> dict:
    with open(path or ROOT / "sim" / "sim_config.yaml") as f:
        return yaml.safe_load(f)


def start_app(headless: bool = True):
    try:
        from isaacsim import SimulationApp
    except ImportError:  # Isaac Sim 4.0 and older
        from omni.isaac.kit import SimulationApp
    return SimulationApp({"headless": headless, "width": 1280, "height": 720})


def open_stage(usd_path: str):
    import omni.usd
    ctx = omni.usd.get_context()
    ok = ctx.open_stage(usd_path)
    if not ok:
        raise RuntimeError(f"could not open {usd_path}")
    return ctx.get_stage()


def meters_per_unit(stage) -> float:
    from pxr import UsdGeom
    return UsdGeom.GetStageMetersPerUnit(stage)


def set_pose_mm(stage, prim_path: str, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float = 0.0,
                roll_deg: float = 0.0, pitch_deg: float = 0.0):
    """Place a prim in world coordinates given in mm and degrees."""
    from pxr import Gf, UsdGeom
    mpu = meters_per_unit(stage)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise KeyError(f"no prim at {prim_path}")
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(x_mm / 1000 / mpu, y_mm / 1000 / mpu, z_mm / 1000 / mpu))
    api.SetRotate(Gf.Vec3f(roll_deg, pitch_deg, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def get_pose_mm(stage, prim_path: str) -> tuple[float, float, float, float]:
    """World (x_mm, y_mm, z_mm, yaw_deg) of a prim."""
    from pxr import Usd, UsdGeom
    mpu = meters_per_unit(stage)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = xf.ExtractTranslation()
    m = np.array(xf.ExtractRotationMatrix())
    yaw = math.degrees(math.atan2(m[0][1], m[0][0]))  # USD row-vector convention
    return t[0] * mpu * 1000, t[1] * mpu * 1000, t[2] * mpu * 1000, yaw


def world_bbox_mm(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds (min, max) in mm."""
    from pxr import Usd, UsdGeom
    mpu = meters_per_unit(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return np.array(r.GetMin()) * mpu * 1000, np.array(r.GetMax()) * mpu * 1000


def duplicate_prim(stage, src: str, dst: str):
    import omni.usd
    from pxr import Sdf
    Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path(src), stage.GetRootLayer(), Sdf.Path(dst))
    return omni.usd.get_context().get_stage().GetPrimAtPath(dst)


def project_points(points_world_m: np.ndarray, view: np.ndarray, proj: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project Nx3 world points (stage units) to pixels using Replicator camera_params.

    USD/Replicator matrices use row vectors: clip = [p, 1] @ view @ proj.
    """
    n = points_world_m.shape[0]
    hom = np.hstack([points_world_m, np.ones((n, 1))])
    clip = hom @ view @ proj
    ndc = clip[:, :3] / clip[:, 3:4]
    u = (ndc[:, 0] + 1) * 0.5 * width
    v = (1 - ndc[:, 1]) * 0.5 * height
    return np.stack([u, v], axis=1)


def obb_corners_mm(x: float, y: float, z: float, yaw_deg: float, length_mm: float, width_mm: float) -> np.ndarray:
    """Top-face rectangle corners (4x3, mm) of an object lying on the table."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    hl, hw = length_mm / 2, width_mm / 2
    local = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
    return np.array([[x + a * c - b * s, y + a * s + b * c, z] for a, b in local])


def import_articulation():
    try:
        from isaacsim.core.prims import SingleArticulation as Articulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation
    return Articulation
