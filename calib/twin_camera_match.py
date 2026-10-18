"""Make the twin's cameras see what the real cameras see.

Two jobs:
  1. --print-usd-lens: turn real intrinsics (K) into USD camera settings
     (focal length and aperture) to paste into the twin camera prim.
  2. --overlay: blend a real photo with a twin render from the same camera so you
     can see if table edges and fixtures line up. Misalignment here becomes
     sim-to-real error later, so fix it now.

    python calib/twin_camera_match.py --print-usd-lens calib/out/overhead_a_K.npz
    python calib/twin_camera_match.py --overlay real.jpg twin_render.jpg --out calib/out/overhead_a_match.jpg
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-usd-lens", metavar="K_NPZ")
    ap.add_argument("--h-aperture-mm", type=float, default=20.955, help="USD horizontal aperture to keep fixed")
    ap.add_argument("--overlay", nargs=2, metavar=("REAL", "TWIN"))
    ap.add_argument("--out", default="overlay.jpg")
    args = ap.parse_args()

    if args.print_usd_lens:
        z = np.load(args.print_usd_lens)
        K, (w, h) = z["K"], z["image_size"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        ha = args.h_aperture_mm
        focal = fx * ha / w
        va = ha * h / w * (fx / fy)
        print(f"image {w}x{h}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        print("Set on the twin camera prim:")
        print(f"  focalLength         = {focal:.4f}")
        print(f"  horizontalAperture  = {ha:.4f}")
        print(f"  verticalAperture    = {va:.4f}")
        print(f"  horizontalApertureOffset = {(cx - w / 2) / w * ha:.4f}")
        print(f"  verticalApertureOffset   = {(cy - h / 2) / h * va:.4f}")
        print("Lens distortion is not modeled by the USD pinhole camera: undistort real images (the detector node does).")

    if args.overlay:
        real, twin = cv2.imread(args.overlay[0]), cv2.imread(args.overlay[1])
        twin = cv2.resize(twin, (real.shape[1], real.shape[0]))
        edges = cv2.Canny(cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY), 60, 160)
        blend = cv2.addWeighted(real, 0.6, twin, 0.4, 0)
        blend[edges > 0] = (0, 0, 255)
        cv2.imwrite(args.out, blend)
        print(f"saved {args.out}: red lines are twin edges drawn on the real photo; they should sit on the real edges")


if __name__ == "__main__":
    main()
