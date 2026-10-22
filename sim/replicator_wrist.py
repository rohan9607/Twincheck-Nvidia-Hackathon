"""Generate wrist-camera images for the grasp check: holding / empty / misaligned.

Output is the YOLO classification layout:
    out/train/holding/*.jpg  out/train/empty/*.jpg  out/train/misaligned/*.jpg  (+ val/)
It also saves out/empty_reference.png: the closed, empty gripper, used by the
reference_diff method (copy it to models/wrist_empty_ref.png).

    /isaac-sim/python.sh sim/replicator_wrist.py --n 1500 --seed 3 --out data/wrist_v1_seed3_n1500
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import ROOT, load_sim_config, start_app  # noqa: E402

STATES = ["holding", "empty", "misaligned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="total images, split evenly across the 3 states")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    cfg = load_sim_config()
    import yaml
    proj = yaml.safe_load(open(ROOT / "config" / "twincheck.yaml"))
    rng = random.Random(args.seed)

    app = start_app(headless=True)
    import omni.replicator.core as rep
    from PIL import Image
    from twin_common import get_pose_mm, import_articulation, open_stage, set_pose_mm

    stage = open_stage(cfg["usd_path"])
    try:
        from isaacsim.core.api import World
    except ImportError:
        from omni.isaac.core import World
    world = World(stage_units_in_meters=1.0)
    Articulation = import_articulation()
    robot = Articulation(cfg["robot"]["articulation"])
    world.reset()
    robot.initialize()
    finger_idx = [robot.dof_names.index(j) for j in cfg["robot"]["finger_joints"]]

    W, H = cfg["cameras"]["resolution"]
    rp = rep.create.render_product(cfg["cameras"]["wrist"], (W, H))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    lights = cfg.get("lights") or []

    out = Path(args.out)
    for split in ("train", "val"):
        for s in STATES:
            (out / split / s).mkdir(parents=True, exist_ok=True)

    classes = list(cfg["objects"].keys())
    park = lambda p: set_pose_mm(stage, p, 5000, 5000, -1000)  # noqa: E731

    def set_fingers(value):
        """value: one joint value for every finger joint, or a list (one per joint, e.g. jaw_l, jaw_r)."""
        q = robot.get_joint_positions()
        for i, v in zip(finger_idx, value if isinstance(value, list) else [value] * len(finger_idx)):
            q[i] = v
        robot.set_joint_positions(q)

    def render():
        for _ in range(3):
            world.step(render=True)
        rep.orchestrator.step(rt_subframes=8)
        return np.asarray(rgb.get_data())[..., :3]

    # Empty reference: fingers closed, no object.
    for p in cfg["objects"].values():
        park(p)
    set_fingers(cfg["robot"]["finger_closed"])
    Image.fromarray(render()).save(out / "empty_reference.png")

    per_state = args.n // len(STATES)
    for state in STATES:
        for k in range(per_state):
            for p in cfg["objects"].values():
                park(p)
            tx, ty, tz, tyaw = get_pose_mm(stage, cfg["robot"]["tool_link"])
            cls = rng.choice(classes)
            spec = proj["objects"][cls]
            if state == "empty":
                set_fingers(cfg["robot"]["finger_closed"])
            else:
                # object centered between the fingers (holding) or shifted/tilted (misaligned)
                dz = -rng.uniform(5, 15)
                off = (0.0, 0.0, 0.0)
                tilt = 0.0
                if state == "misaligned":
                    off = (rng.uniform(-12, 12), rng.uniform(-6, 6), rng.uniform(-10, 5))
                    tilt = rng.choice([-1, 1]) * rng.uniform(15, 45)
                set_pose_mm(stage, cfg["objects"][cls], tx + off[0], ty + off[1], tz + dz + off[2],
                            tyaw + 90 + rng.uniform(-5, 5), roll_deg=tilt)
                # close until the fingers touch the object's narrow side
                ratio = min(1.0, spec["width_mm"] / proj["gripper"]["max_opening_mm"])
                open_v, closed_v = cfg["robot"]["finger_open"], cfg["robot"]["finger_closed"]
                if not isinstance(open_v, list):
                    open_v, closed_v = [open_v], [closed_v]
                set_fingers([c + (o - c) * ratio for o, c in zip(open_v, closed_v)])
            for lp in lights:
                a = stage.GetPrimAtPath(lp).GetAttribute("inputs:intensity")
                if a:
                    a.Set(rng.uniform(500, 1800))
            img = render().astype(np.float32) + np.random.normal(0, rng.uniform(0, 4), (H, W, 3))
            split = "val" if rng.random() < args.val_frac else "train"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out / split / state / f"{state}_{k:05d}.jpg")
        print(f"{state}: {per_state} images")

    print(f"done: {out}\nCopy {out / 'empty_reference.png'} to models/wrist_empty_ref.png for reference_diff.")
    app.close()


if __name__ == "__main__":
    main()
