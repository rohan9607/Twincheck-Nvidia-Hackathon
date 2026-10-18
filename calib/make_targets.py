"""Print-ready calibration targets.

  checkerboard.png - 9x6 inner corners, 25 mm squares (print at 100% scale, tape flat on card)
  aruco_markers.png - 6 ArUco markers (DICT_4X4_50, ids 0-5), 40 mm each, cut out and lay on the table

    python calib/make_targets.py --out calib/targets
Measure the printed square and marker size with a ruler: printers scale. Use the
measured values in intrinsics.py --square-mm and homography.py.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DPI = 300
MM = DPI / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib/targets")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25)
    ap.add_argument("--marker-mm", type=float, default=40)
    ap.add_argument("--markers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sq = int(round(args.square_mm * MM))
    board = np.full(((args.rows + 1) * sq, (args.cols + 1) * sq), 255, np.uint8)
    for r in range(args.rows + 1):
        for c in range(args.cols + 1):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
    board = cv2.copyMakeBorder(board, sq, sq, sq, sq, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(out / "checkerboard.png"), board)

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = int(round(args.marker_mm * MM))
    pad = int(round(10 * MM))
    sheet = np.full((2 * (m + 2 * pad), 3 * (m + 2 * pad)), 255, np.uint8)
    for i in range(args.markers):
        img = cv2.aruco.generateImageMarker(d, i, m)
        r, c = divmod(i, 3)
        y, x = r * (m + 2 * pad) + pad, c * (m + 2 * pad) + pad
        sheet[y:y + m, x:x + m] = img
        cv2.putText(sheet, f"id {i}", (x, y + m + pad // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.imwrite(str(out / "aruco_markers.png"), sheet)
    print(f"wrote {out}/checkerboard.png and {out}/aruco_markers.png at {DPI} dpi; print at 100% scale")


if __name__ == "__main__":
    main()
