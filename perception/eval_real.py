"""Score the overhead detector on the REAL test set, in the units that matter.

Standard mAP says little about whether a pick will land. This reports:
  - recall: share of labeled real objects the model found
  - position error (mm) after the homography, and yaw error (degrees, symmetry-aware)
  - per-class numbers, and the misses, saved as images to look at

    python perception/eval_real.py --model runs/obb/train/weights/best.pt --data data/real_test_v1 \
        --camera table --out runs/eval/real_v1

The real test set is YOLO-OBB (export from CVAT). It must NEVER be used for training.
Acceptance gate (from the execution plan): recall >= 95%, median position <= 5 mm, median yaw <= 5 deg.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "twincheck_core"), str(ROOT / "ros2_ws/src/twincheck_perception")]

from twincheck_core import Pose2D, load_config, yaw_diff_deg  # noqa: E402
from twincheck_core.geometry import apply_homography, pixel_yaw_to_table  # noqa: E402
from twincheck_perception.infer import CameraCalib, OverheadDetector  # noqa: E402


def label_poses(lines, names, w, h, calib, symmetry):
    out = []
    for parts in lines:
        c = names[int(parts[0])]
        pts = np.array(parts[1:9], float).reshape(4, 2) * [w, h]
        cx, cy = pts.mean(axis=0)
        e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
        long = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        ang = math.degrees(math.atan2(long[1], long[0]))
        u, v = calib.undistort_px(cx, cy)
        x, y = apply_homography(calib.H, u, v)
        out.append((c, Pose2D(x, y, pixel_yaw_to_table(calib.H, u, v, ang)), pts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="real test folder with data.yaml, images/, labels/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--camera", default="table", help="which calibration to apply")
    ap.add_argument("--match-mm", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config()
    sym = {n: o.symmetry_deg for n, o in cfg.objects.items()}
    cam = cfg.raw["cameras"][args.camera]
    calib = CameraCalib.load(args.camera, cam["homography"], cam.get("intrinsics"))
    det = OverheadDetector(args.model, sym, conf=0.25)
    root = Path(args.data)
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    out = Path(args.out)
    (out / "misses").mkdir(parents=True, exist_ok=True)

    per_cls: dict[str, dict] = {}
    pos_err, yaw_err = [], []
    imgs = sorted((root / "images" / args.split).glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {root / 'images' / args.split}")
    for p in imgs:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        lab = root / "labels" / args.split / (p.stem + ".txt")
        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        truth = label_poses(lines, names, w, h, calib, sym)
        found = det.detect(img, calib)
        used = set()
        missed_here = False
        for cls, pose, pts in truth:
            s = per_cls.setdefault(cls, {"n": 0, "found": 0, "pos": [], "yaw": []})
            s["n"] += 1
            cands = [(d.pose.distance_to(pose), i) for i, d in enumerate(found) if d.cls == cls and i not in used]
            cands = [c for c in cands if c[0] <= args.match_mm]
            if not cands:
                missed_here = True
                cv2.polylines(img, [pts.astype(np.int32)], True, (0, 0, 255), 2)
                continue
            d_mm, i = min(cands)
            used.add(i)
            a = yaw_diff_deg(found[i].pose.yaw_deg, pose.yaw_deg, sym.get(cls, 360))
            s["found"] += 1
            s["pos"].append(d_mm)
            s["yaw"].append(a)
            pos_err.append(d_mm)
            yaw_err.append(a)
        if missed_here:
            cv2.imwrite(str(out / "misses" / p.name), img)

    n = sum(s["n"] for s in per_cls.values())
    f = sum(s["found"] for s in per_cls.values())
    summary = {
        "images": len(imgs), "objects": n, "recall": f / n if n else 0.0,
        "median_pos_mm": float(np.median(pos_err)) if pos_err else None,
        "p90_pos_mm": float(np.percentile(pos_err, 90)) if pos_err else None,
        "median_yaw_deg": float(np.median(yaw_err)) if yaw_err else None,
        "per_class": {c: {"recall": s["found"] / s["n"], "median_pos_mm": float(np.median(s["pos"])) if s["pos"] else None,
                          "median_yaw_deg": float(np.median(s["yaw"])) if s["yaw"] else None, "n": s["n"]}
                      for c, s in per_cls.items()},
    }
    gate = (summary["recall"] >= 0.95 and (summary["median_pos_mm"] or 99) <= 5 and (summary["median_yaw_deg"] or 99) <= 5)
    summary["passes_gate"] = gate
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nGATE: {'PASS' if gate else 'NOT YET'} (recall>=95%, median pos<=5 mm, median yaw<=5 deg)")
    print(f"misses saved to {out / 'misses'}: look at each and ask which randomization would have covered it")


if __name__ == "__main__":
    main()
