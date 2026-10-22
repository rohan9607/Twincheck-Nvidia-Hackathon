"""Lens calibration (intrinsics) from checkerboard photos.

Take 15-20 photos of the printed checkerboard with the camera you are calibrating:
tilt it, move it to the corners and edges of the image, fill different parts of
the frame. Then:

    python calib/intrinsics.py --images "calib/raw/table/*.jpg" --out calib/out/table_K.npz --square-mm 25

Saves K (camera matrix) and dist (distortion), prints the reprojection error
(below 0.5 px is good, above 1 px means retake blurry or poorly spread photos),
and writes an undistorted preview so you can see straight lines are straight.
Works on a laptop webcam too: practice with --webcam 0 before going to CSI.
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def capture_webcam(index: int, n: int, folder: Path, cols: int, rows: int):
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(index)
    saved = 0
    print("SPACE = save frame when the board is found, Q = done")
    while saved < n:
        ok, frame = cap.read()
        if not ok:
            break
        # FAST_CHECK bails out quickly when no board is in view, so the preview stays live
        found, corners = cv2.findChessboardCorners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (cols, rows),
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                                                   + cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        cv2.drawChessboardCorners(view, (cols, rows), corners, found)
        cv2.putText(view, f"saved {saved}/{n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("calibration", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found:
            cv2.imwrite(str(folder / f"cb_{saved:02d}.jpg"), frame)
            saved += 1
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return sorted(str(p) for p in folder.glob("*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help='glob, e.g. "calib/raw/table/*.jpg"')
    ap.add_argument("--webcam", type=int, help="capture from this camera index instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0)
    args = ap.parse_args()

    files = capture_webcam(args.webcam, 20, Path(args.out).with_suffix("") / "captures", args.cols, args.rows) \
        if args.webcam is not None else sorted(glob.glob(args.images))
    if len(files) < 8:
        raise SystemExit(f"need at least 8 images, found {len(files)}")

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    obj_pts, img_pts, size, used = [], [], None, []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for f in files:
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))
        if not found:
            print(f"  board not found: {f}")
            continue
        obj_pts.append(objp)
        img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit))
        used.append(f)
    if len(used) < 8:
        raise SystemExit(f"board found in only {len(used)} images; retake photos")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, K=K, dist=dist, image_size=np.array(size), rms=rms)
    print(f"\nused {len(used)}/{len(files)} images, reprojection error {rms:.3f} px "
          f"({'good' if rms < 0.5 else 'ok' if rms < 1 else 'retake photos'})")
    print(f"K =\n{K.round(2)}\ndist = {dist.ravel().round(4)}\nsaved {out}")
    sample = cv2.imread(used[0])
    cv2.imwrite(str(out.with_name(out.stem + "_undistorted_preview.jpg")),
                np.hstack([sample, cv2.undistort(sample, K, dist)]))


if __name__ == "__main__":
    main()
