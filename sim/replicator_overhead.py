"""Generate labeled overhead training images from the twin (YOLO oriented-box format).

Per frame it randomizes: object count, type, position and yaw on the table,
distractor clutter, light intensity and color, camera pose jitter, then renders
each overhead camera and writes one YOLO-OBB label line per object:
    class_id x1 y1 x2 y2 x3 y3 x4 y4     (corners normalized to 0..1)

    /isaac-sim/python.sh sim/replicator_overhead.py --n 500 --seed 42 --out data/overhead_v1_seed42_n500
    /isaac-sim/python.sh sim/replicator_overhead.py --n 2000 --seed 7 --out data/overhead_v2_seed7_n2000 --clutter

Check 20 images with perception/draw_labels.py BEFORE rendering thousands.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import ROOT, load_sim_config, start_app  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="frames per camera")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", nargs="+", default=["table"])
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--clutter", action="store_true", help="allow objects close together / touching")
    ap.add_argument("--lighting", choices=["mild", "wide"], default="mild")
    ap.add_argument("--cam-jitter-mm", type=float, default=15.0)
    ap.add_argument("--cam-jitter-deg", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_sim_config()
    import yaml
    proj = yaml.safe_load(open(ROOT / "config" / "twincheck.yaml"))
    classes = list(cfg["objects"].keys())
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    app = start_app(headless=True)
    import omni.replicator.core as rep
    from twin_common import (duplicate_prim, footprint_mm, jitter_prim, local_matrix, meters_per_unit, obb_corners_mm,
                             open_stage, place_part, project_points, scene_copies, set_pose_mm)

    stage = open_stage(cfg["usd_path"])
    mpu = meters_per_unit(stage)
    W, H = cfg["cameras"]["resolution"]
    tbl = cfg["table"]
    rep.orchestrator.set_capture_on_play(False)

    # A pool of object copies: the scene's own copies plus duplicates up to max_objects, parked far away when unused.
    pool = {}
    for cls, src in cfg["objects"].items():
        own = scene_copies(stage, src)
        pool[cls] = own + [str(duplicate_prim(stage, src, f"{src}_copy{i}").GetPath())
                           for i in range(len(own), args.max_objects)]
    park = lambda p: set_pose_mm(stage, p, 5000, 5000, -1000)  # noqa: E731

    # Lights: the scene's own lights, varied around their authored intensity
    light_paths = cfg.get("lights") or []
    if not light_paths:
        dome = rep.create.light(light_type="Dome", intensity=1000)
        light_paths = [str(dome.get_output_prims()["prims"][0].GetPath())]
    light_home = {}
    for lp in light_paths:
        a = stage.GetPrimAtPath(lp).GetAttribute("inputs:intensity")
        if a and a.Get() is not None:
            light_home[lp] = (a, float(a.Get()))

    # Cameras: remember the calibrated pose so jitter is always around it
    cams = {}
    for cam in args.cameras:
        path = cfg["cameras"][cam]
        rp = rep.create.render_product(path, (W, H))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        par = rep.AnnotatorRegistry.get_annotator("camera_params")
        rgb.attach(rp)
        par.attach(rp)
        cams[cam] = {"path": path, "home": local_matrix(stage, path), "rgb": rgb, "params": par}

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    from PIL import Image
    meta = []
    for f in range(args.n):
        # ---- randomize objects
        for paths in pool.values():
            for p in paths:
                park(p)
        placed = []
        for _ in range(rng.randint(1, args.max_objects)):
            cls = rng.choice(classes)
            free = [p for p in pool[cls] if p not in [q[1] for q in placed]]
            if not free:
                continue
            spec = proj["objects"][cls]
            for _try in range(30):
                x = rng.uniform(*tbl["x_range_mm"])
                y = rng.uniform(*tbl["y_range_mm"])
                min_gap = 5 if args.clutter else max(footprint_mm(spec)) + 15
                if all(np.hypot(x - px, y - py) > min_gap for _, _, px, py, _ in placed):
                    break
            yaw = rng.uniform(0, 360)
            place_part(stage, free[0], cfg["origins"][cls], spec, x, y, yaw, tbl["surface_z_mm"])
            placed.append((cls, free[0], x, y, yaw))

        # ---- randomize light
        lo, hi = (0.6, 1.4) if args.lighting == "mild" else (0.25, 2.0)
        for a, home in light_home.values():
            a.Set(home * rng.uniform(lo, hi))
        for lp in light_paths:
            a = stage.GetPrimAtPath(lp).GetAttribute("inputs:colorTemperature")
            if a:
                a.Set(rng.uniform(4000, 6500))

        split = "val" if rng.random() < args.val_frac else "train"
        for cam, c in cams.items():
            axis = np.random.normal(size=3)
            jitter_prim(stage, c["path"], c["home"], *(rng.gauss(0, args.cam_jitter_mm / 2) for _ in range(3)),
                        rng.gauss(0, args.cam_jitter_deg), axis / np.linalg.norm(axis))
        rep.orchestrator.step(rt_subframes=8)

        for cam, c in cams.items():
            img = c["rgb"].get_data()
            params = c["params"].get_data()
            view = np.array(params["cameraViewTransform"]).reshape(4, 4)
            projm = np.array(params["cameraProjection"]).reshape(4, 4)
            # sensor noise and slight blur, so the model does not learn the renderer's perfection
            arr = np.asarray(img)[..., :3].astype(np.float32)
            arr += np.random.normal(0, rng.uniform(0, 4), arr.shape)
            frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            stem = f"{cam}_{f:06d}"
            frame.save(out / "images" / split / f"{stem}.jpg", quality=92)

            lines = []
            for cls, _, x, y, yaw in placed:
                spec = proj["objects"][cls]
                corners = obb_corners_mm(x, y, tbl["surface_z_mm"] + spec.get("height_mm", spec["width_mm"]),
                                         yaw, *footprint_mm(spec))
                px = project_points(corners / 1000 / mpu, view, projm, W, H)
                if (px[:, 0].max() < 0 or px[:, 0].min() > W or px[:, 1].max() < 0 or px[:, 1].min() > H):
                    continue  # outside this camera's view
                norm = np.clip(px / [W, H], 0, 1).flatten()
                lines.append(f"{classes.index(cls)} " + " ".join(f"{v:.6f}" for v in norm))
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            meta.append({"image": f"{split}/{stem}.jpg", "camera": cam, "objects": [
                {"cls": cl, "x_mm": x, "y_mm": y, "yaw_deg": yw} for cl, _, x, y, yw in placed]})

        if f % 50 == 0:
            print(f"frame {f}/{args.n}", flush=True)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    (out / "meta.json").write_text(json.dumps({"seed": args.seed, "args": vars(args), "frames": meta}))
    print(f"done: {out}  ({args.n} frames x {len(cams)} cameras)")
    app.close()


if __name__ == "__main__":
    main()
