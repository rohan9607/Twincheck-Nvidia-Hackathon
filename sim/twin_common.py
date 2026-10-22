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
    """Place a prim (parented to /World) in world coordinates given in mm and degrees.

    Replaces its transform ops with translate, rotateZ(yaw), rotateY(pitch), rotateX(roll): the starter's own
    op order for cubes (translate, rotateZ) and lying screws (translate, rotateZ, rotateY). XformCommonAPI
    can't edit that stack, so it is rebuilt here.
    """
    from pxr import Gf, UsdGeom
    mpu = meters_per_unit(stage)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise KeyError(f"no prim at {prim_path}")
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(x_mm / 1000 / mpu, y_mm / 1000 / mpu, z_mm / 1000 / mpu))
    xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddRotateYOp().Set(float(pitch_deg))
    xf.AddRotateXOp().Set(float(roll_deg))


def nudge_mm(stage, prim_path: str, dx_mm: float, dy_mm: float):
    """Shift a prim sideways in world x/y, keeping its orientation (a lying screw stays lying)."""
    from pxr import Gf, UsdGeom
    mpu = meters_per_unit(stage)
    for op in UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(op.Get() + Gf.Vec3d(dx_mm / 1000 / mpu, dy_mm / 1000 / mpu, 0.0))
            return
    raise KeyError(f"{prim_path} has no translate op")


def place_part(stage, prim_path: str, origin: str, spec: dict, x_mm: float, y_mm: float, yaw_deg: float,
               surface_z_mm: float):
    """Put a part on the table so the centre of its footprint is at (x, y), yaw in degrees.

    origin (sim_config.yaml `origins`): where the starter's prim origin sits on the part
      bottom_centre    cube: centre of the bottom face
      head_face_lying  screw: centre of the head face, lying on its side, tip along +yaw (cell.add_screws)
    spec: the object's entry in config/twincheck.yaml (length_mm, height_mm).
    """
    if origin == "bottom_centre":
        set_pose_mm(stage, prim_path, x_mm, y_mm, surface_z_mm, yaw_deg)
    elif origin == "head_face_lying":
        d = spec["length_mm"] / 2
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        set_pose_mm(stage, prim_path, x_mm - d * c, y_mm - d * s, surface_z_mm + spec["height_mm"] / 2,
                    yaw_deg, pitch_deg=-90.0)
    else:
        raise ValueError(f"unknown origin {origin!r} for {prim_path}")


def part_pose_mm(stage, prim_path: str, origin: str, spec: dict) -> tuple[float, float, float]:
    """(x, y, yaw) of a part's footprint centre: the inverse of place_part."""
    from pxr import Usd, UsdGeom
    mpu = meters_per_unit(stage)
    m = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    x, y = t[0] * mpu * 1000, t[1] * mpu * 1000
    if origin == "bottom_centre":
        return x, y, math.degrees(math.atan2(m[0][1], m[0][0]))  # local +x in world (row vectors)
    if origin == "head_face_lying":
        dx, dy = -m[2][0], -m[2][1]  # the shank runs along local -z, from the head to the tip
        n = math.hypot(dx, dy) or 1.0
        d = spec["length_mm"] / 2
        return x + d * dx / n, y + d * dy / n, math.degrees(math.atan2(dy, dx))
    raise ValueError(f"unknown origin {origin!r} for {prim_path}")


def footprint_mm(spec: dict) -> tuple[float, float]:
    """(length, width) of a part's outline seen from above. A lying screw is as wide as its head (height_mm)."""
    return spec["length_mm"], max(spec["width_mm"], spec.get("height_mm", spec["width_mm"]))


def local_matrix(stage, prim_path: str):
    from pxr import Usd, UsdGeom
    return UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).GetLocalTransformation(Usd.TimeCode.Default())


def jitter_prim(stage, prim_path: str, home, dx_mm: float, dy_mm: float, dz_mm: float, angle_deg: float, axis):
    """Set a prim (a camera) to its home local matrix, shifted in its parent frame and turned by a small angle
    about `axis` in its own frame. Keeps the home orientation (a camera's downward tilt) intact."""
    from pxr import Gf, UsdGeom
    mpu = meters_per_unit(stage)
    turn = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), angle_deg))
    m = turn * home  # USD row vectors: the turn is applied in the prim's own frame
    m.SetTranslateOnly(home.ExtractTranslation() + Gf.Vec3d(dx_mm, dy_mm, dz_mm) / 1000 / mpu)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


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


def scene_copies(stage, src: str) -> list[str]:
    """Every copy of a part the scene already has: /World/SortCube0 -> [/World/SortCube0, /World/SortCube1, ...].
    Generators must park or label each one, or an unlabelled part stays in every frame."""
    import re
    prim = stage.GetPrimAtPath(src)
    base = re.sub(r"\d+$", "", prim.GetName())
    return sorted(str(c.GetPath()) for c in prim.GetParent().GetChildren() if re.fullmatch(re.escape(base) + r"\d+", c.GetName()))


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
