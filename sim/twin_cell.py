"""Run cell_policy in the twin the way the organizers' cell runs it, so a grasp is proven before a real slot.

The organizers' mock (run.py --mock) only checks formats: it has no kinematics and synthetic images. This
plays the same loop in Isaac Sim: render the table and wrist cameras, call Policy.act(obs) exactly as the
client does, and play every returned row as one 0.1 s step (starter IK to the fingertip target, the gripper
after the row). The rollout ends when the policy is done and holds, or after --max-rows.

    /workspace/isaac/python.sh sim/twin_cell.py --mode guide --rollouts 10 --nudge-mm 30 --seed 1 \
        --model runs/obb/overhead_v2/weights/best.pt --client /workspace/client --session twin_cell_guide

Same layouts as make_placement.py (same seed and nudge -> the same rollouts the operator sets up). Scoring is
the operator's: a rollout succeeds when every part ends in its target bin. Ground truth is also written at
each check the policy makes (what the table camera saw: true part poses; at the wrist check: what is lifted),
so caught and missed failures can be counted against the policy's own decisions.

Twin, not real: the arm follows targets exactly unless --lag-rows (the real arm is about 8 rows behind), and
cell.finger_yaw_offset_deg is honoured by construction here (R -> closing direction), so it can't be measured
in the twin: measure it on the real cell (wrist image at R = 0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import ROOT, load_sim_config, open_stage, start_app  # noqa: E402

PHYSICS_HZ = 60
ROW_S = 0.1
UF0_Z_MM = -330.0


def wpr_matrix(w: float, p: float, r: float) -> np.ndarray:
    """FANUC w, p, r (deg; fixed x, y, z angles) -> rotation matrix Rz(r) Ry(p) Rx(w)."""
    w, p, r = np.radians([w, p, r])
    cx, sx, cy, sy, cz, sz = math.cos(w), math.sin(w), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_wpr(m: np.ndarray) -> tuple[float, float, float]:
    p = -math.asin(max(-1.0, min(1.0, m[2, 0])))
    w = math.atan2(m[2, 1], m[2, 2])
    r = math.atan2(m[1, 0], m[0, 0])
    return math.degrees(w), math.degrees(p), math.degrees(r)


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class LazyImages(Mapping):
    """obs["images"]: renders a camera only when the policy reads it, and records ground truth at that moment."""

    def __init__(self, render, names, on_read):
        self.render, self.names, self.on_read, self.cache = render, list(names), on_read, {}

    def __getitem__(self, k):
        if k not in self.names:
            raise KeyError(k)
        if k not in self.cache:
            self.cache[k] = self.render(k)
            self.on_read(k, self.cache[k])
        return self.cache[k]

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "watch", "guide"], default="guide")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--nudge-mm", type=float, default=0.0, help="as make_placement.py (0 = nominal layout)")
    ap.add_argument("--seed", type=int, default=1, help="as make_placement.py")
    ap.add_argument("--model", default=None, help="table detector (default: config models.detector)")
    ap.add_argument("--client", default="/workspace/client", help="organizers' client package (calibration/)")
    ap.add_argument("--session", default=None, help="trial database tag (default twin_cell_<mode>)")
    ap.add_argument("--lag-rows", type=int, default=0, help="arm follows targets this many rows late (real: ~8)")
    ap.add_argument("--max-rows", type=int, default=1500, help="rows per rollout before the operator stops it")
    ap.add_argument("--hold-rows", type=int, default=20, help="end the rollout after the policy is done this long")
    ap.add_argument("--jpeg", type=int, default=90, help="JPEG quality applied to images (0 = raw)")
    ap.add_argument("--frames-every", type=int, default=0, help="save a table frame every N rows (0 = off)")
    ap.add_argument("--wrist-oracle", action="store_true",
                    help="TWIN ONLY: answer the wrist check from the twin's truth (lifted or not), to test the rest "
                         "of the loop before a wrist method exists. Numbers from such runs say nothing about the check")
    ap.add_argument("--save-wrist", action="store_true", help="save every wrist image served, named by the truth")
    ap.add_argument("--debug", action="store_true", help="log every row (target, tip, jaws, parts) to <session>_rows.jsonl")
    ap.add_argument("--wrist-frames", action="store_true", help="with --frames-every, save the wrist view too")
    ap.add_argument("--starter", default="/workspace/fanuc_er4ia_sim")
    ap.add_argument("--out", default=str(ROOT / "runs" / "twin_cell"))
    args = ap.parse_args()

    session = args.session or f"twin_cell_{args.mode}"
    os.environ["TWINCHECK_MODE"] = args.mode
    os.environ["TWINCHECK_SESSION"] = session
    os.environ["TWINCHECK_ENV"] = "twin"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth_path = out / f"{session}_truth.jsonl"
    dbg = open(out / f"{session}_rows.jsonl", "w") if args.debug else None

    scfg = load_sim_config()
    cfg_raw = yaml.safe_load((ROOT / "config" / "twincheck.yaml").read_text())
    app = start_app(headless=True)
    try:
        import cv2
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Usd, UsdGeom, UsdPhysics

        from twin_common import part_pose_mm, place_part, scene_copies

        sys.path.insert(0, args.starter)
        import cell as starter_cell
        import fk
        import ik
        import scene_config as C

        sys.path[:0] = [str(Path(args.client).resolve()), str(ROOT / "cell_policy")]
        sys.path.insert(0, str(ROOT / "twincheck_core"))
        from make_placement import layout
        from twincheck_core import load_config
        from twincheck_policy import Policy

        cfg = load_config()
        cell = cfg_raw["cell"]
        yaw_offset = float(cell["finger_yaw_offset_deg"])

        stage = open_stage(scfg["usd_path"])
        world = World(stage_units_in_meters=1.0, physics_dt=1 / PHYSICS_HZ, rendering_dt=1 / PHYSICS_HZ)
        starter_cell.enable_ccd(world)
        rep.orchestrator.set_capture_on_play(False)

        # Parts: the scene's copies per class, assigned to the sequence in order
        jobs = cell["sequence"]
        pool = {cls: scene_copies(stage, src) for cls, src in scfg["objects"].items()}
        parts = []  # (cls, prim path)
        used = {cls: 0 for cls in pool}
        for job in jobs:
            cls = job["cls"]
            if used[cls] >= len(pool[cls]):
                raise SystemExit(f"the scene has {len(pool[cls])} {cls}(s); the sequence needs more")
            parts.append((cls, pool[cls][used[cls]]))
            used[cls] += 1
        for cls, paths in pool.items():  # copies the sequence doesn't use go far away
            for p in paths[used[cls]:]:
                place_part(stage, p, scfg["origins"][cls], cfg_raw["objects"][cls], 5000, 5000, 0, -1000)
        surface_mm = scfg["table"]["surface_z_mm"]
        base_z_mm = C.TABLE_TOP_Z * 1000  # robot base in the world

        def set_kinematic(on: bool):
            for _, p in parts:
                UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p)).CreateKinematicEnabledAttr().Set(on)

        cams = {}
        res = {"table": tuple(scfg["cameras"]["resolution"]), "wrist": (1280, 720)}
        for name in ("table", "wrist"):
            rp = rep.create.render_product(scfg["cameras"][name], res[name])
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach(rp)
            cams[name] = ann

        world.reset()
        arm = SingleArticulation(prim_path=scfg["robot"]["articulation"], name="er4ia")
        arm.initialize()
        dof = list(arm.dof_names)
        arm_idx = [dof.index(f"J{i}") for i in range(1, 7)]
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.array(C.TOOL_AXIS) * starter_cell.tip_offset())
        tool_axis, spread_axis = np.array(C.TOOL_AXIS), np.array(C.FINGER_SPREAD_AXIS)

        def tcp_of(q6) -> list[float]:
            """Fingertip [x, y, z, w, p, r] in UF0. The tool frame: z along the fingers, x the closing direction
            turned back by finger_yaw_offset (so R = closing yaw - offset, as twincheck_core.cell assumes)."""
            rmat, tip = fk.tool_pose(q6, tool)
            z = rmat @ tool_axis
            x = rot_z(-yaw_offset) @ (rmat @ spread_axis)
            x = x - z * (x @ z)
            x /= np.linalg.norm(x)
            m = np.column_stack([x, np.cross(z, x), z])
            w, p, r = matrix_wpr(m)
            return [tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, w, p, r]

        def jaw_full(value):
            q = np.array(arm.get_joint_positions(), dtype=float)
            for j, v in starter_cell.jaw_targets(value).items():
                q[dof.index(j)] = v
            return q

        state = {"q": None, "jaw": C.JAW_OPEN, "closed": 0.0, "fault": None}

        def command(q6, jaw, steps):
            q_from = np.array(state["q"])
            for k in range(1, steps + 1):
                full = jaw_full(jaw)
                full[arm_idx] = q_from + (np.array(q6) - q_from) * k / steps
                arm.apply_action(ArticulationAction(joint_positions=full))
                world.step(render=False)
            state["q"] = np.array(q6)

        def part_states():
            out_ = []
            for cls, p in parts:
                x, y, yaw = part_pose_mm(stage, p, scfg["origins"][cls], cfg_raw["objects"][cls])
                m = UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                z = m.ExtractTranslation()[2] * 1000 - surface_mm
                out_.append({"cls": cls, "prim": p, "x_mm": round(x, 1), "y_mm": round(y, 1),
                             "yaw_deg": round(yaw, 1), "z_above_surface_mm": round(z, 1)})
            return out_

        def jaw_value_for_close(q6):
            """The pneumatic jaws close until they meet the part. A position drive stands in for that force
            limit: seat on a cube's width (the starter's CUBE_JAW_HALF_TARGET), close fully otherwise."""
            _, tip = fk.tool_pose(q6, tool)
            near = min(part_states(), key=lambda s: math.hypot(s["x_mm"] - tip[0] * 1000, s["y_mm"] - tip[1] * 1000))
            if near["cls"] == "cube" and math.hypot(near["x_mm"] - tip[0] * 1000, near["y_mm"] - tip[1] * 1000) < 12:
                return C.CUBE_JAW_HALF_TARGET
            return C.JAW_CLOSED

        def render(name):
            rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
            rgb = np.asarray(cams[name].get_data())[..., :3].copy()
            if args.jpeg:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, args.jpeg])
                rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            return rgb

        def in_bin(s, bin_name):
            b = cfg_raw["bins"][bin_name]
            return (abs(s["x_mm"] - b["center"]["x_mm"]) <= b["half_x_mm"]
                    and abs(s["y_mm"] - b["center"]["y_mm"]) <= b["half_y_mm"]
                    and s["z_above_surface_mm"] < cell["bin_height_mm"])

        def bin_of(s):
            return next((n for n in cfg_raw["bins"] if in_bin(s, n)), None)

        # Home: the starter's home tip, fingers down
        q_home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q_home, ok = ik.solve(q_home, np.array([0.30, 0.0, 0.24]), [0.0, 0.0, -1.0], limits, tool_offset=tool)
        if not ok:
            raise RuntimeError("home IK failed")

        policy = Policy(args.model)
        if args.wrist_oracle:
            from twincheck_core import GraspState

            class Oracle:
                def check(self, rgb):
                    return GraspState.HOLDING if any(s["z_above_surface_mm"] > 20 for s in part_states())                         else GraspState.EMPTY

            policy.wrist = Oracle()
        rng = random.Random(args.seed)
        summary = []
        for k in range(1, args.rollouts + 1):
            objects, notes = layout(cfg, rng if args.nudge_mm else None, args.nudge_mm)
            # ---- reset: parts on their layout, arm home, jaws open
            set_kinematic(True)
            idx_cls = {cls: 0 for cls in pool}
            for cls, p in parts:
                x, y, yaw = objects[cls][idx_cls[cls]]
                idx_cls[cls] += 1
                place_part(stage, p, scfg["origins"][cls], cfg_raw["objects"][cls], x, y, yaw, surface_mm)
            full = jaw_full(C.JAW_OPEN)
            full[arm_idx] = q_home
            arm.set_joint_positions(full)
            state.update(q=q_home.copy(), jaw=C.JAW_OPEN, closed=0.0, fault=None)
            command(q_home, C.JAW_OPEN, 30)
            set_kinematic(False)
            command(q_home, C.JAW_OPEN, 30)
            start = part_states()
            policy.reset()
            trial = {"trial_id": k, "layout": "different" if args.nudge_mm else "same"}
            truth = open(truth_path, "a")

            def record(cam, img, k=k):
                states = part_states()
                rec = {"rollout": k, "camera": cam, "row": rows_done, "parts": states,
                       "trial": policy.loop.trial_id if policy.loop else None}
                if cam == "wrist":
                    rec["lifted"] = [s["cls"] for s in states if s["z_above_surface_mm"] > 20]
                    if args.save_wrist:
                        wdir = out / "wrist" / session
                        wdir.mkdir(parents=True, exist_ok=True)
                        tag = "_".join(rec["lifted"]) or "empty"
                        cv2.imwrite(str(wdir / f"r{k:02d}_row{rows_done:04d}_{tag}.jpg"),
                                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                truth.write(json.dumps(rec) + "\n")
                truth.flush()

            lag = deque()
            rows_done, held, t0 = 0, 0, time.time()
            while rows_done < args.max_rows:
                joints_deg = list(np.degrees(state["q"])) + [state["closed"]]
                obs = {"images": LazyImages(render, ["table", "wrist"], record), "joints": joints_deg,
                       "tcp_pose": tcp_of(state["q"]), "trial": trial}
                rows = policy.act(obs)
                for row in rows:
                    lag.append(row)
                    target = lag.popleft() if len(lag) > args.lag_rows else None
                    if target is not None:
                        x, y, z, w, p, r, g = target
                        pos = np.array([x, y, z - UF0_Z_MM]) / 1000
                        m = wpr_matrix(w, p, r)
                        approach = m[:, 2]
                        spread = rot_z(yaw_offset) @ m[:, 0]
                        cur = fk.tool_pose(state["q"], tool)[0] @ spread_axis
                        if cur @ spread < 0:  # two-finger jaws: either sign closes the same way; take the near one
                            spread = -spread
                        q, ok = ik.solve(state["q"], pos, approach, limits, tool_offset=tool, spread_dir=spread)
                        if not ok:
                            q, ok = ik.solve(state["q"], pos, approach, limits, tool_offset=tool, spread_dir=spread,
                                             iters=500, damping=0.02)
                        if not ok or np.linalg.norm(fk.tool_pose(q, tool)[1] - pos) > 0.002:
                            state["fault"] = f"IK failed at row {rows_done}: {[round(v, 1) for v in target]}"
                            break
                        command(q, state["jaw"], int(PHYSICS_HZ * ROW_S))
                        if (g >= 0.5) != (state["closed"] >= 0.5):  # the gripper acts after the arm reaches the row
                            state["jaw"] = jaw_value_for_close(q) if g >= 0.5 else C.JAW_OPEN
                            state["closed"] = 1.0 if g >= 0.5 else 0.0
                    else:
                        command(state["q"], state["jaw"], int(PHYSICS_HZ * ROW_S))
                    rows_done += 1
                    if args.debug:
                        jp = arm.get_joint_positions()
                        dbg.write(json.dumps({
                            "rollout": k, "row": rows_done, "target": [round(v, 1) for v in row],
                            "tip": [round(v, 1) for v in tcp_of(state["q"])[:3]],
                            "tip_measured": [round(v, 1) for v in tcp_of(np.array(jp)[arm_idx])[:3]],
                            "jaws_mm": [round(float(jp[dof.index(j)]) * 1000, 2) for j in ("jaw_l", "jaw_r")],
                            "jaw_cmd": round(state["jaw"] * 1000, 2),
                            "parts": [(s["cls"], s["x_mm"], s["y_mm"], s["z_above_surface_mm"]) for s in part_states()]}) + "\n")
                    if args.frames_every and rows_done % args.frames_every == 0:
                        fdir = out / "frames" / session / f"r{k:02d}"
                        fdir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(fdir / f"{rows_done:05d}.jpg"), cv2.cvtColor(render("table"), cv2.COLOR_RGB2BGR))
                        if args.wrist_frames:
                            cv2.imwrite(str(fdir / f"{rows_done:05d}_wrist.jpg"), cv2.cvtColor(render("wrist"), cv2.COLOR_RGB2BGR))
                if state["fault"]:
                    break
                if policy.loop is not None and policy.loop.done and not lag:
                    held += len(rows)
                    if held >= args.hold_rows:
                        break
            end = part_states()
            placed = []
            for (cls, p), s in zip(parts, end):
                want = cfg_raw["objects"][cls]["target_bin"]
                placed.append({"cls": cls, "want": want, "got": bin_of(s)})
            ok_all = state["fault"] is None and all(pp["got"] == pp["want"] for pp in placed)
            rec = {"rollout": k, "mode": args.mode, "success": ok_all, "rows": rows_done,
                   "time_s": round(rows_done * ROW_S, 1), "fault": state["fault"], "notes": notes, "parts": placed,
                   "start": start, "end": end, "wall_s": round(time.time() - t0, 1)}
            truth.write(json.dumps({"rollout": k, "result": rec}) + "\n")
            truth.close()
            summary.append(rec)
            print(f"rollout {k}: {'SUCCESS' if ok_all else 'FAIL'} in {rec['time_s']} s "
                  f"({rec['wall_s']} s wall) {[(pp['cls'], pp['got']) for pp in placed]}"
                  + (f" FAULT {state['fault']}" if state["fault"] else ""), flush=True)

        n_ok = sum(r["success"] for r in summary)
        print(f"done: {n_ok}/{len(summary)} rollouts succeeded ({args.mode}); truth in {truth_path}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
