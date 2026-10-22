"""Generate labeled overhead training images from the twin (YOLO oriented-box format).

Per frame it randomizes: object count, type, position and yaw on the table, cube colour, distractor shapes,
the arm's pose, light intensity and color, camera pose jitter; then renders each overhead camera and writes
one YOLO-OBB label line per object:
    class_id x1 y1 x2 y2 x3 y3 x4 y4     (corners normalized to 0..1)

    # v1: one arm pose, yellow cubes, no distractors
    /workspace/isaac/python.sh sim/replicator_overhead.py --n 500 --seed 42 --out /workspace/data/overhead_v1_seed42_n500
    # v2: random arm poses, three cube colours, clutter, distractors, wide lighting
    /workspace/isaac/python.sh sim/replicator_overhead.py --n 2000 --seed 7 --out /workspace/data/overhead_v2_seed7_n2000 \
        --arm random --colours red yellow blue --max-objects 6 --clutter --distractors 3 --lighting wide

A frame where the arm hides more than (1 - --min-visible) of any part is rendered again with another arm pose:
a label on a part the camera can't see teaches the detector to see things that aren't there.
Check 20 images with perception/draw_labels.py BEFORE rendering thousands.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import ROOT, load_sim_config, start_app  # noqa: E402

# distractor colours: none of the parts' colours (cubes red/yellow/blue, screws black)
DISTRACTOR_RGB = [(0.2, 0.6, 0.25), (0.9, 0.45, 0.1), (0.5, 0.3, 0.6), (0.9, 0.9, 0.88), (0.55, 0.55, 0.55),
                  (0.45, 0.3, 0.18), (0.1, 0.5, 0.5)]
TOOL_OFFSET_M = (0.293, 0.0, 0.0)  # J6 origin -> fingertip along the tool axis (FAQ, UTOOL1)


def overlap(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    """Do two convex 2D polygons (Nx2, mm) come closer than margin? Separating-axis test."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.hypot(*e) or 1.0)
            pa, pb = a @ n, b @ n
            if pa.max() + margin < pb.min() or pb.max() + margin < pa.min():
                return False
    return True


def random_arm_q(rng, capture_deg, fk_tool, surface_m):
    """Joint angles (deg, J1..J6): near the starter's capture pose, or random with the fingertip above the table."""
    if rng.random() < 0.4:
        return [capture_deg[f"J{i}"] + rng.gauss(0, 8) for i in range(1, 7)]
    for _ in range(50):
        q = [rng.uniform(-120, 60), rng.uniform(-10, 50), rng.uniform(-50, 30), rng.uniform(-90, 90),
             rng.uniform(-120, -10), rng.uniform(-180, 180)]
        _, tip = fk_tool(np.radians(q), TOOL_OFFSET_M)
        if tip[2] > surface_m + 0.04:
            return q
    return [capture_deg[f"J{i}"] for i in range(1, 7)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="frames per camera")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", nargs="+", default=["table"])
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--clutter", action="store_true", help="half the frames pack parts close together (not touching)")
    ap.add_argument("--lighting", choices=["mild", "wide"], default="mild")
    ap.add_argument("--cam-jitter-mm", type=float, default=15.0)
    ap.add_argument("--cam-jitter-deg", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--arm", choices=["fixed", "random"], default="fixed", help="random: a new arm pose per frame")
    ap.add_argument("--colours", nargs="+", default=["yellow"], help="cube colours (the cube USD's colour variants)")
    ap.add_argument("--distractors", type=int, default=0, help="up to this many non-part shapes per frame")
    ap.add_argument("--min-visible", type=float, default=0.6, help="share of each part the arm may not hide")
    ap.add_argument("--starter", default="/workspace/fanuc_er4ia_sim", help="starter folder (fk.py, scene_config.py)")
    args = ap.parse_args()

    cfg = load_sim_config()
    import yaml
    proj = yaml.safe_load(open(ROOT / "config" / "twincheck.yaml"))
    classes = list(cfg["objects"].keys())
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    app = start_app(headless=True)
    import omni.replicator.core as rep
    from PIL import Image, ImageDraw
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    from twin_common import (duplicate_prim, footprint_mm, jitter_prim, local_matrix, meters_per_unit, obb_corners_mm,
                             open_stage, place_part, project_points, scene_copies, set_pose_mm)

    stage = open_stage(cfg["usd_path"])
    mpu = meters_per_unit(stage)
    W, H = cfg["cameras"]["resolution"]
    tbl = cfg["table"]
    surface_mm = tbl["surface_z_mm"]
    rep.orchestrator.set_capture_on_play(False)

    # A pool of object copies: the scene's own copies plus duplicates up to max_objects, parked far away when unused.
    pool = {}
    for cls, src in cfg["objects"].items():
        own = scene_copies(stage, src)
        pool[cls] = own + [str(duplicate_prim(stage, src, f"{src}_copy{i}").GetPath())
                           for i in range(len(own), args.max_objects)]
    park = lambda p: set_pose_mm(stage, p, 5000, 5000, -1000)  # noqa: E731

    # Distractors: plain shapes the detector must ignore
    distractors = []
    for i in range(args.distractors):
        # unit-sized (1 x 1 x 1 stage units, centred), sized per frame by a scale op
        if i % 2:
            g = UsdGeom.Cylinder.Define(stage, f"/World/Distractor{i}")
            g.CreateRadiusAttr(0.5)
            g.CreateHeightAttr(1.0)
            g.CreateAxisAttr("Z")
        else:
            g = UsdGeom.Cube.Define(stage, f"/World/Distractor{i}")
            g.CreateSizeAttr(1.0)
        g.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
        g.CreateDisplayColorAttr([Gf.Vec3f(0.5)])
        distractors.append(g)
        park(str(g.GetPath()))

    # The arm: posed through physics, so the parts must not be simulated (they would fall from where we put them)
    robot = None
    if args.arm == "random":
        for paths in pool.values():
            for p in paths:
                prim = stage.GetPrimAtPath(p)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        sys.path.insert(0, args.starter)
        from fk import tool_pose
        import scene_config as starter_cfg
        world = World(stage_units_in_meters=1.0)
        world.reset()
        robot = SingleArticulation(prim_path=cfg["robot"]["articulation"], name="er4ia")
        robot.initialize()
        dof = list(robot.dof_names)
        jaw_open, jaw_closed = cfg["robot"]["finger_open"], cfg["robot"]["finger_closed"]

        def pose_arm():
            q = np.array(robot.get_joint_positions(), dtype=float)
            deg = random_arm_q(rng, starter_cfg.CAPTURE_POSE_DEG, tool_pose, (surface_mm - 900.0) / 1000)
            for i, d in enumerate(deg):
                q[dof.index(f"J{i + 1}")] = math.radians(d)
            jaws = jaw_open if rng.random() < 0.5 else jaw_closed
            for j, v in zip(cfg["robot"]["finger_joints"], jaws):
                q[dof.index(j)] = v
            robot.set_joint_positions(q)
            robot.apply_action(ArticulationAction(joint_positions=q))
            for _ in range(3):
                world.step(render=False)
            return [round(d, 1) for d in deg]

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
        c = {"path": path, "home": local_matrix(stage, path)}
        for name in ("rgb", "camera_params") + (("instance_id_segmentation",) if robot else ()):
            ann = rep.AnnotatorRegistry.get_annotator(name, **({"init_params": {"colorize": False}}
                                                                if name == "instance_id_segmentation" else {}))
            ann.attach(rp)
            c[name] = ann
        cams[cam] = c

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def set_cube_colour(part, colour):
        """The saved scene flattens cube.usda, losing its colour variant set; its Red/Yellow/Blue materials are
        still under <cube>/Visual/Looks, so rebind every bound mesh to the chosen one."""
        visual = stage.GetPrimAtPath(f"{part}/Visual")
        mat = UsdShade.Material(stage.GetPrimAtPath(f"{part}/Visual/Looks/{colour.capitalize()}"))
        if not mat:
            raise KeyError(f"no {colour} material under {visual.GetPath()}/Looks")
        rgb = next(c.GetAttribute("inputs:diffuseColor").Get() for c in Usd.PrimRange(mat.GetPrim())
                   if c.GetAttribute("inputs:diffuseColor"))
        for prim in Usd.PrimRange(visual):
            rel = prim.GetRelationship("material:binding")
            if rel and rel.GetTargets():
                UsdShade.MaterialBindingAPI(prim).Bind(mat)
                UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([rgb])

    def outline_px(cls, x, y, yaw, view, projm):
        spec = proj["objects"][cls]
        corners = obb_corners_mm(x, y, surface_mm + spec.get("height_mm", spec["width_mm"]), yaw, *footprint_mm(spec))
        return project_points(corners / 1000 / mpu, view, projm, W, H)

    def hidden_share(seg, px):
        """Share of a part's outline (pixels) covered by the robot in the instance segmentation."""
        x0, y0 = np.floor(px.min(axis=0)).astype(int)
        x1, y1 = np.ceil(px.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        mask = Image.new("L", (x1 - x0 + 1, y1 - y0 + 1))
        ImageDraw.Draw(mask).polygon([(u - x0, v - y0) for u, v in px], fill=1)
        inside = np.asarray(mask, bool)
        return float(seg[y0:y1 + 1, x0:x1 + 1][inside].mean()) if inside.any() else 0.0

    meta, rerolls = [], 0
    for f in range(args.n):
        # ---- parts
        for paths in pool.values():
            for p in paths:
                park(p)
        placed, outlines = [], []
        margin = 1.0 if (args.clutter and rng.random() < 0.5) else 15.0
        for _ in range(rng.randint(1, args.max_objects)):
            cls = rng.choice(classes)
            free = [p for p in pool[cls] if p not in [q[1] for q in placed]]
            if not free:
                continue
            spec = proj["objects"][cls]
            for _try in range(40):
                x, y = rng.uniform(*tbl["x_range_mm"]), rng.uniform(*tbl["y_range_mm"])
                yaw = rng.uniform(0, 360)
                poly = obb_corners_mm(x, y, 0, yaw, *footprint_mm(spec))[:, :2]
                if not any(overlap(poly, o, margin) for o in outlines):
                    break
            else:
                continue
            place_part(stage, free[0], cfg["origins"][cls], spec, x, y, yaw, surface_mm)
            if cls == "cube":
                set_cube_colour(free[0], rng.choice(args.colours))
            placed.append((cls, free[0], x, y, yaw))
            outlines.append(poly)

        # ---- distractors, clear of the parts
        for g in distractors:
            park(str(g.GetPath()))
        for g in distractors[:rng.randint(0, len(distractors))]:
            if isinstance(g, UsdGeom.Cube):
                sx, sy, sz = rng.uniform(20, 45), rng.uniform(5, 25), rng.uniform(3, 20)  # flat or long: not a cube
            else:
                r, h = rng.uniform(5, 15), rng.uniform(5, 40)
                sx = sy = 2 * r
                sz = h
            for _try in range(40):
                x, y = rng.uniform(200, 520), rng.uniform(-220, 190)
                yaw = rng.uniform(0, 360)
                poly = obb_corners_mm(x, y, 0, yaw, sx, sy)[:, :2]
                if not any(overlap(poly, o, 10.0) for o in outlines):
                    break
            else:
                continue
            path = str(g.GetPath())
            set_pose_mm(stage, path, x, y, surface_mm + sz / 2, yaw)
            UsdGeom.Xformable(g.GetPrim()).AddScaleOp().Set(Gf.Vec3f(sx, sy, sz) / 1000 / mpu)
            g.GetDisplayColorAttr().Set([Gf.Vec3f(*rng.choice(DISTRACTOR_RGB))])
            outlines.append(poly)

        # ---- light
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

        # ---- render; with a random arm, pose it again until it hides no part too much
        arm_deg = None
        for attempt in range(6):
            if robot is not None:
                arm_deg = pose_arm()
            rep.orchestrator.step(rt_subframes=8)
            if robot is None:
                break
            worst = 0.0
            for cam, c in cams.items():
                params = c["camera_params"].get_data()
                view = np.array(params["cameraViewTransform"]).reshape(4, 4)
                projm = np.array(params["cameraProjection"]).reshape(4, 4)
                seg = c["instance_id_segmentation"].get_data()
                arm_ids = [int(k) for k, v in seg["info"]["idToLabels"].items()
                           if str(v).startswith(cfg["robot"]["articulation"])]
                arm_mask = np.isin(seg["data"], arm_ids)
                for cls, _, x, y, yaw in placed:
                    worst = max(worst, hidden_share(arm_mask, outline_px(cls, x, y, yaw, view, projm)))
            if worst <= 1.0 - args.min_visible:
                break
            rerolls += 1
        else:
            print(f"frame {f}: the arm hides a part in every pose tried; keeping the last one", flush=True)

        for cam, c in cams.items():
            params = c["camera_params"].get_data()
            view = np.array(params["cameraViewTransform"]).reshape(4, 4)
            projm = np.array(params["cameraProjection"]).reshape(4, 4)
            # sensor noise, so the model does not learn the renderer's perfection
            arr = np.asarray(c["rgb"].get_data())[..., :3].astype(np.float32)
            arr += np.random.normal(0, rng.uniform(0, 4), arr.shape)
            frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            stem = f"{cam}_{f:06d}"
            frame.save(out / "images" / split / f"{stem}.jpg", quality=92)

            lines = []
            for cls, _, x, y, yaw in placed:
                px = outline_px(cls, x, y, yaw, view, projm)
                if (px[:, 0].max() < 0 or px[:, 0].min() > W or px[:, 1].max() < 0 or px[:, 1].min() > H):
                    continue  # outside this camera's view
                norm = np.clip(px / [W, H], 0, 1).flatten()
                lines.append(f"{classes.index(cls)} " + " ".join(f"{v:.6f}" for v in norm))
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            meta.append({"image": f"{split}/{stem}.jpg", "camera": cam, "arm_deg": arm_deg, "objects": [
                {"cls": cl, "x_mm": x, "y_mm": y, "yaw_deg": yw} for cl, _, x, y, yw in placed]})

        if f % 50 == 0:
            print(f"frame {f}/{args.n}  (arm re-posed {rerolls} times so far)", flush=True)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    (out / "meta.json").write_text(json.dumps({"seed": args.seed, "args": vars(args), "frames": meta}))
    print(f"done: {out}  ({args.n} frames x {len(cams)} cameras, arm re-posed {rerolls} times)")
    app.close()


if __name__ == "__main__":
    main()
