"""Check calibration at the start of every real-arm session (5 marked spots).

Put an object on each printed spot of the position template, take a photo, and
compare where the detector + homography puts it with the spot's known position.

    python calib/verify.py --image session/verify.jpg --spots calib/template_spots.csv \
        --camera table --model models/overhead_obb.pt

spots CSV: spot,x_mm,y_mm   Logs mean and max error; over 5 mm means recalibrate.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "twincheck_core"), str(ROOT / "ros2_ws/src/twincheck_perception")]

from twincheck_core import load_config  # noqa: E402
from twincheck_perception.infer import CameraCalib, OverheadDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--spots", required=True)
    ap.add_argument("--camera", default="table")
    ap.add_argument("--model", help="detector weights (default: config models.detector)")
    args = ap.parse_args()

    cfg = load_config()
    cam = cfg.raw["cameras"][args.camera]
    calib = CameraCalib.load(args.camera, cam["homography"], cam.get("intrinsics"))
    det = OverheadDetector(args.model or cfg.raw["models"]["detector"], {n: o.symmetry_deg for n, o in cfg.objects.items()})
    dets = det.detect(cv2.imread(args.image), calib)
    with open(args.spots) as f:
        spots = [(r["spot"], float(r["x_mm"]), float(r["y_mm"])) for r in csv.DictReader(f)]

    errs = []
    print(f"{'spot':<8}{'true (mm)':>18}{'seen (mm)':>20}{'error':>10}")
    for name, x, y in spots:
        if not dets:
            print(f"{name:<8} nothing detected")
            continue
        d = min(dets, key=lambda d: np.hypot(d.pose.x_mm - x, d.pose.y_mm - y))
        e = float(np.hypot(d.pose.x_mm - x, d.pose.y_mm - y))
        errs.append(e)
        print(f"{name:<8}{f'({x:.0f}, {y:.0f})':>18}{f'({d.pose.x_mm:.1f}, {d.pose.y_mm:.1f})':>20}{e:>8.1f} mm")
    if errs:
        verdict = "OK" if max(errs) <= 5 else "RECALIBRATE"
        print(f"\nmean {np.mean(errs):.1f} mm, max {np.max(errs):.1f} mm -> {verdict}")


if __name__ == "__main__":
    main()
