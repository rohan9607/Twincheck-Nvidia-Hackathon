"""Step 1 with the twin: find out what is inside it.

Prints and saves (sim/out/twin_report.json):
  - every prim with its type (robot, cameras, objects, lights, physics)
  - each camera's resolution, focal length and world pose
  - each object's size (to fill config/twincheck.yaml) and whether it fits the jaws
  - articulation joints (names and limits) for the robot and fingers

    /isaac-sim/python.sh sim/explore_twin.py --usd /workspace/fanuc_er4ia_sim/scene.usd
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import load_sim_config, start_app  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", help="twin USD file (default: sim_config.yaml usd_path)")
    ap.add_argument("--max-prims", type=int, default=600)
    args = ap.parse_args()
    cfg = load_sim_config()
    app = start_app(headless=True)

    from pxr import UsdGeom, UsdLux, UsdPhysics
    from twin_common import get_pose_mm, meters_per_unit, open_stage, world_bbox_mm

    stage = open_stage(args.usd or cfg["usd_path"])
    report = {"meters_per_unit": meters_per_unit(stage), "up_axis": UsdGeom.GetStageUpAxis(stage),
              "prims": [], "cameras": [], "lights": [], "articulations": [], "joints": [], "rigid_bodies": []}

    for i, prim in enumerate(stage.Traverse()):
        path = str(prim.GetPath())
        if i < args.max_prims:
            report["prims"].append({"path": path, "type": prim.GetTypeName()})
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            report["cameras"].append({"path": path, "focal_length": cam.GetFocalLengthAttr().Get(),
                                      "h_aperture": cam.GetHorizontalApertureAttr().Get(),
                                      "v_aperture": cam.GetVerticalApertureAttr().Get(),
                                      "pose_mm_yaw": get_pose_mm(stage, path)})
        if prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName():
            report["lights"].append(path)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            report["articulations"].append(path)
        if prim.IsA(UsdPhysics.Joint):
            j = {"path": path, "type": prim.GetTypeName()}
            for lim in ("physics:lowerLimit", "physics:upperLimit"):
                a = prim.GetAttribute(lim)
                if a and a.HasValue():
                    j[lim.split(":")[1]] = a.Get()
            report["joints"].append(j)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            lo, hi = world_bbox_mm(stage, path)
            size = (hi - lo).round(1).tolist()
            report["rigid_bodies"].append({"path": path, "size_mm": size, "center_mm": ((lo + hi) / 2).round(1).tolist()})

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    (out / "twin_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nmeters per unit: {report['meters_per_unit']}  up axis: {report['up_axis']}")
    print(f"\nCAMERAS ({len(report['cameras'])})")
    for c in report["cameras"]:
        print(f"  {c['path']}  focal={c['focal_length']}  pose(mm,yaw)={[round(v, 1) for v in c['pose_mm_yaw']]}")
    print(f"\nARTICULATIONS: {report['articulations']}")
    print(f"\nJOINTS ({len(report['joints'])})")
    for j in report["joints"]:
        print(f"  {j['path']}  {j.get('lowerLimit')}..{j.get('upperLimit')}")
    print(f"\nRIGID BODIES ({len(report['rigid_bodies'])}): size in mm")
    for b in report["rigid_bodies"]:
        print(f"  {b['path']}  size={b['size_mm']}  center={b['center_mm']}")
    print(f"\nLIGHTS: {report['lights']}")
    print(f"\nFull report: {out / 'twin_report.json'}")
    print("Next: fill sim/sim_config.yaml and the TODO sizes in config/twincheck.yaml, then run sim/finger_fit_test.py")
    app.close()


if __name__ == "__main__":
    main()
