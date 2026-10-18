"""Map overhead-camera pixels to robot coordinates on the table (the homography).

1. Lay 4-6 ArUco markers flat on the table, spread across the work area.
2. With the plant manager and mentor: jog the arm's tool tip to each marker
   center and read X, Y (mm, robot base frame) off the controller.
   Write them to a CSV:   marker_id,x_mm,y_mm
3. Take one photo with each overhead camera (markers visible, arm out of the way).
4. Run once per camera:

    python calib/homography.py --image calib/raw/overhead_a_markers.jpg --points calib/marker_points.csv \
        --intrinsics calib/out/overhead_a_K.npz --out calib/out/overhead_a_H.npy

It prints the error at every marker (leave-one-out, so the number is honest) and
saves an overlay image with the robot-frame grid drawn on the photo.
Since the markers are measured in robot coordinates, "table" x/y == robot_base x/y.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def detect_markers(img, dictionary=cv2.aruco.DICT_4X4_50):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.ravel(), corners)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--points", required=True, help="CSV: marker_id,x_mm,y_mm (robot frame)")
    ap.add_argument("--intrinsics", help="npz from intrinsics.py; undistorts before fitting")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    K = dist = None
    if args.intrinsics:
        z = np.load(args.intrinsics)
        K, dist = z["K"], z["dist"]
        img = cv2.undistort(img, K, dist)
    found = detect_markers(img)
    with open(args.points) as f:
        robot = {int(r["marker_id"]): (float(r["x_mm"]), float(r["y_mm"])) for r in csv.DictReader(f)}
    ids = sorted(set(found) & set(robot))
    missing = sorted(set(robot) - set(found))
    if missing:
        print(f"markers not seen in this image: {missing}")
    if len(ids) < 4:
        raise SystemExit(f"need 4+ markers seen and measured, have {len(ids)}")

    px = np.array([found[i] for i in ids], np.float64)
    mm = np.array([robot[i] for i in ids], np.float64)
    H, _ = cv2.findHomography(px, mm, 0)

    # Leave-one-out error: fit without marker i, predict marker i.
    print("\nmarker   error (leave-one-out)")
    errs = []
    for k, i in enumerate(ids):
        if len(ids) < 5:
            break
        keep = [j for j in range(len(ids)) if j != k]
        Hk, _ = cv2.findHomography(px[keep], mm[keep], 0)
        p = Hk @ np.array([*px[k], 1.0])
        e = float(np.hypot(*(p[:2] / p[2] - mm[k])))
        errs.append(e)
        print(f"  id {i:<4} {e:6.2f} mm")
    if errs:
        print(f"mean {np.mean(errs):.2f} mm, max {np.max(errs):.2f} mm  (aim for under 3-5 mm)")
    else:
        print("use 5+ markers to get an honest error estimate")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, H)

    # Overlay: draw a 50 mm robot-frame grid on the image
    Hinv = np.linalg.inv(H)
    over = img.copy()
    xs = np.arange(np.floor(mm[:, 0].min() / 50) * 50 - 50, mm[:, 0].max() + 100, 50)
    ys = np.arange(np.floor(mm[:, 1].min() / 50) * 50 - 50, mm[:, 1].max() + 100, 50)
    for x in xs:
        pts = np.array([[x, y, 1] for y in ys]) @ Hinv.T
        cv2.polylines(over, [(pts[:, :2] / pts[:, 2:]).astype(np.int32)], False, (0, 255, 255), 1)
    for y in ys:
        pts = np.array([[x, y, 1] for x in xs]) @ Hinv.T
        cv2.polylines(over, [(pts[:, :2] / pts[:, 2:]).astype(np.int32)], False, (0, 255, 255), 1)
    for i, p in zip(ids, px):
        cv2.circle(over, tuple(int(v) for v in p), 6, (0, 0, 255), 2)
        cv2.putText(over, str(i), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out.with_suffix(".overlay.jpg")), over)
    print(f"saved {out} and {out.with_suffix('.overlay.jpg')}")


if __name__ == "__main__":
    main()
