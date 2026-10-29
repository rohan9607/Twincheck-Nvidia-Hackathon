"""Score the cell perception path on twin renders against the twin's ground truth, in mm.

Runs exactly what the policy runs on the real cell: our YOLO detector, then the organizers' calibration
(pixel_to_base, cut at surface + class height). Ground truth is the generator's meta.json (part centres in
the robot base frame). Render the test set with the camera at its calibrated pose:

    /workspace/isaac/python.sh sim/replicator_overhead.py --n 50 --seed 7 --val-frac 0 \
        --cam-jitter-mm 0 --cam-jitter-deg 0 --out /workspace/data/twin_mm_test_seed7_n50
    python cell_policy/eval_twin.py --data <that folder> --model models/overhead_v1_n500.pt \
        --client <client package folder>

Twin numbers only: they show our pipeline and the organizers' calibration agree with the twin camera.
The G3 gate (recall >= 95 %, median <= 5 mm, median yaw <= 5 deg) is on REAL images.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "twincheck_core"), str(Path(__file__).resolve().parent)]

from twincheck_core import Pose2D, load_config, plan_resync, yaw_diff_deg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="generator output folder (meta.json + images/)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--client", required=True, help="the organizers' client package (has calibration/)")
    ap.add_argument("--camera", default="table")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-mm", type=float, default=30.0, help="farther than this counts as a miss")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.client).resolve()))
    from cell_perception import TableDetector

    cfg = load_config()
    det = TableDetector(cfg, args.model, conf=args.conf)
    data = Path(args.data)
    frames = [f for f in json.loads((data / "meta.json").read_text())["frames"] if f["camera"] == args.camera]

    err = defaultdict(list)  # cls -> [(pos_mm, yaw_deg, dx, dy)]
    truth_n, extra = defaultdict(int), defaultdict(int)
    for f in frames:
        rgb = cv2.cvtColor(cv2.imread(str(data / "images" / f["image"])), cv2.COLOR_BGR2RGB)
        truth = {f"t{i}": (o["cls"], Pose2D(o["x_mm"], o["y_mm"], o["yaw_deg"])) for i, o in enumerate(f["objects"])}
        for cls, _ in truth.values():
            truth_n[cls] += 1
        updates, _missing = plan_resync(truth, det.detect(rgb), match_radius_mm=args.match_mm)
        for u in updates:
            if u.old is None:
                extra[u.cls] += 1
                continue
            sym = cfg.objects[u.cls].symmetry_deg if u.cls in cfg.objects else 90.0   # a bin: square
            err[u.cls].append((u.shift_mm, abs(yaw_diff_deg(u.new.yaw_deg, u.old.yaw_deg, sym)),
                               u.new.x_mm - u.old.x_mm, u.new.y_mm - u.old.y_mm))

    print(f"{len(frames)} {args.camera} frames, model {Path(args.model).name}, match within {args.match_mm} mm")
    print(f"{'class':8} {'found':>9} {'recall':>7} {'extra':>5} {'pos med':>8} {'pos p95':>8} "
          f"{'bias x,y (mm)':>15} {'yaw med':>8} {'yaw p95':>8}")
    for cls in sorted(truth_n):
        e = np.array(err[cls]) if err[cls] else np.zeros((0, 4))
        found = len(e)
        row = f"{cls:8} {found:>4}/{truth_n[cls]:<4} {found / truth_n[cls]:>7.1%} {extra[cls]:>5}"
        if found:
            row += (f" {np.median(e[:, 0]):>8.1f} {np.percentile(e[:, 0], 95):>8.1f} "
                    f"{np.mean(e[:, 2]):>7.1f},{np.mean(e[:, 3]):>6.1f} "
                    f"{np.median(e[:, 1]):>8.1f} {np.percentile(e[:, 1], 95):>8.1f}")
        print(row)
    print("pos = distance between detected and true centre (mm); bias = mean detected - true; yaw symmetry-aware")


if __name__ == "__main__":
    main()
