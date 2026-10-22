"""Build the TwinCheck sorting scene with the organizers' starter (v0.5), without editing it.

Sets the starter's scene_config to our scenario (config/twincheck.yaml): the rollout's cubes and M6x30
screws on our pick points, our three 50 mm bins, no vials or caps (they don't fit the jaws). Then runs the
starter's own build_scene.py with the remaining arguments.

    /workspace/isaac/python.sh sim/build_twincheck_scene.py --starter /workspace/fanuc_er4ia_sim \
        --headless --save /workspace/twincheck_scene.usd [--shoot /workspace/shots/scene1]

The starter defines add_screws() but never calls it and has no screw dimensions; both are supplied here.
A screw's prim origin is its head face; we place it so the middle of its length is on the pick point (the same
rule as twin_common.place_part and the detector's box centre).
"""
import argparse
import math
import runpy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# M6 x 30 countersunk socket screw, ISO 10642 (metres)
SCREW_HEAD_D = 0.012
SCREW_HEAD_H = 0.0033
SCREW_SHANK_D = 0.006
SCREW_LENGTH = 0.030
SCREW_MASS = 0.007
BIN_MATS = {"bin_cubes": "plastic_red", "bin_screws": "bin_yellow", "bin_rework": "plastic_blue"}


def scenario(cfg):
    """Cubes [(x, y, colour)], cube yaws [deg], screws [(x_head, y_head, yaw)] in metres, from cell.sequence."""
    cubes, yaws, screws = [], [], []
    for job in cfg["cell"]["sequence"]:
        p = cfg["pick_points"][job["pick_point"]]
        x, y, yaw = p["x_mm"] / 1000, p["y_mm"] / 1000, float(p.get("yaw_deg", 0.0))
        if job["cls"] == "cube":
            cubes.append((x, y, "yellow"))
            yaws.append(yaw)
        elif job["cls"] == "screw":
            # tip points along +yaw from the head; the middle of the screw is half its length from the head face
            d = SCREW_LENGTH / 2
            screws.append((x - d * math.cos(math.radians(yaw)), y - d * math.sin(math.radians(yaw)), yaw))
        else:
            raise SystemExit(f"sequence class {job['cls']!r} is not in the sorting scene (cube, screw)")
    return cubes, yaws, screws


def bins(cfg, depth_mm=25.0):
    """Starter SORT_BINS entries for our bins (outer size from the config, 35 mm pocket, 55 mm tall)."""
    h = cfg["cell"]["bin_height_mm"] / 1000
    out = {}
    for name, b in cfg["bins"].items():
        outer = (2 * b["half_x_mm"] / 1000, 2 * b["half_y_mm"] / 1000)
        out[name] = {"xy": (b["center"]["x_mm"] / 1000, b["center"]["y_mm"] / 1000), "outer": outer,
                     "pocket": 0.035, "pocket_r": 0.003, "wall_h": h, "floor_t": h - depth_mm / 1000,
                     "mat": BIN_MATS.get(name, "plastic_blue"), "release_z": h + 0.02}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starter", default="/workspace/fanuc_er4ia_sim", help="starter's fanuc_er4ia_sim folder")
    ap.add_argument("--config", default=str(ROOT / "config" / "twincheck.yaml"))
    args, rest = ap.parse_known_args()
    starter = Path(args.starter).resolve()
    cfg = yaml.safe_load(Path(args.config).read_text())

    sys.path.insert(0, str(starter))
    import scene_config as C  # the starter's module; build_scene.py imports this same object

    cubes, yaws, screws = scenario(cfg)
    C.SORT_CUBES, C.SORT_CUBE_YAWS_DEG = cubes, yaws
    C.SORT_VIALS, C.SORT_CAPS, C.CAP_ORIENTATIONS = [], [], []
    C.SCREWS = screws
    C.SCREW_HEAD_D, C.SCREW_HEAD_H, C.SCREW_SHANK_D = SCREW_HEAD_D, SCREW_HEAD_H, SCREW_SHANK_D
    C.SCREW_LENGTH, C.SCREW_MASS = SCREW_LENGTH, SCREW_MASS
    C.SHOW_BINS, C.SORT_BINS = True, bins(cfg)

    # cell.py imports Omniverse modules, which only load once SimulationApp is running: patch it right after
    # build_scene.py starts the app.
    import isaacsim
    start_app = isaacsim.SimulationApp

    def start_app_and_patch(*a, **k):
        app = start_app(*a, **k)
        import cell
        add_objects = cell.add_sort_objects

        def add_objects_and_screws(stage):
            add_objects(stage)
            cell.add_screws(stage)

        cell.add_sort_objects = add_objects_and_screws
        return app

    isaacsim.SimulationApp = start_app_and_patch
    print(f"[twincheck] cubes={cubes} screws(head xy, yaw)={screws} bins={list(C.SORT_BINS)}")

    sys.argv = [str(starter / "build_scene.py")] + rest
    runpy.run_path(str(starter / "build_scene.py"), run_name="__main__")


if __name__ == "__main__":
    main()
