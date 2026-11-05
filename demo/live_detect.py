"""Live demo: the sim-trained detector on real cell frames (and on twin renders), in a full-screen window.

Each part gets its position in robot millimetres (the organizers' calibration) and its distance from the spot a
taught program would reach. The jaws leave about 3 mm, so anything farther is a blind miss.

    python demo/live_detect.py                 # real frames (demo/assets/real), then twin renders
    python demo/live_detect.py --client D:/twincheck/PhysicalAI-Hackathon-2026

    python demo/live_detect.py --save out/      # no window: write the annotated images (backup)

Keys: space / right = next, left = previous, a = auto-play, f = full screen, q / Esc = quit.
Real frames are shown for testing only; the model was trained on twin renders alone.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "twincheck_core"))
sys.path.insert(0, str(REPO / "cell_policy"))

from twincheck_core import load_config  # noqa: E402

COLOURS = {"cube": (59, 68, 217), "screw": (20, 140, 240), "bin": (91, 158, 46)}   # BGR
JAW_SLACK_MM = 3.0
W, H = 1920, 1080
IW, IH = 1440, 810            # the camera image; the side panel takes the rest


def taught_spots(cfg) -> list[tuple[str, float, float]]:
    pts = cfg.raw["pick_points"]
    return [(j["cls"], pts[j["pick_point"]]["x_mm"], pts[j["pick_point"]]["y_mm"]) for j in cfg.raw["cell"]["sequence"]]


def banner(img, text, y, scale=0.9, colour=(255, 255, 255), bg=(30, 34, 38), x=0, w=None):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (x, y - th - 16), (x + (w or tw + 32), y + 12), bg, -1)
    cv2.putText(img, text, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default=str(REPO.parent / "PhysicalAI-Hackathon-2026"),
                    help="the organizers' client package (for calibration/geometry.py)")
    ap.add_argument("--model", default=str(REPO / "models" / "overhead_v4_n3000.pt"))
    ap.add_argument("--seconds", type=float, default=4.0, help="auto-play interval")
    ap.add_argument("--save", help="write every annotated image to this folder instead of opening a window (backup)")
    args = ap.parse_args()
    sys.path.insert(0, args.client)
    from calibration.geometry import SURFACE_Z, project   # noqa: E402
    from cell_perception import TableDetector              # noqa: E402

    cfg = load_config()
    det = TableDetector(cfg, args.model)
    spots = taught_spots(cfg)
    images = sorted((HERE / "assets" / "real").glob("*.jpg")) + sorted((HERE / "assets" / "twin").glob("*.jpg"))
    min_conf = cfg.raw["thresholds"]["min_confidence"]                            # the policy's own cut
    for _ in range(3):                                                             # warm up the GPU
        det.model.predict(cv2.imread(str(images[0])), verbose=False)

    win = "TwinCheck live detector"
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    else:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 900)
    i, auto, full, shown = 0, False, False, 0.0
    while True:
        path = images[i % len(images)]
        real = path.parent.name == "real"
        bgr = cv2.imread(str(path))
        t0 = time.perf_counter()
        res = det.model.predict(bgr, conf=det.conf, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        found = [d for d in det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))       # same model, robot mm
                 if d.confidence >= min_conf]
        bins = [d.pose for d in found if d.cls == "bin"]
        view = cv2.resize(bgr, (IW, IH))                                             # camera image, left
        sx, sy = IW / bgr.shape[1], IH / bgr.shape[0]
        if res.obb is not None:                                                      # the model's oriented boxes
            for poly, c, s in zip(res.obb.xyxyxyxy.cpu().numpy(), res.obb.cls.cpu().numpy(), res.obb.conf.cpu().numpy()):
                if s >= min_conf:
                    pts = (poly * [sx, sy]).astype(np.int32)
                    cv2.polylines(view, [pts], True, COLOURS.get(res.names[int(c)], (255, 255, 255)), 3, cv2.LINE_AA)
        if real:                                                                     # taught spots: red X
            for cls, x, y in spots:
                u, v = project(det.cam, [[x / 1000, y / 1000, SURFACE_Z]])[0]
                u, v = int(u * IW / det.cam["size"][0]), int(v * IH / det.cam["size"][1])
                cv2.drawMarker(view, (u, v), (40, 40, 220), cv2.MARKER_TILTED_CROSS, 30, 4, cv2.LINE_AA)

        rows, offs = [], []                                                          # side panel, two lines each
        for d in sorted(found, key=lambda d: (d.cls != "cube", d.cls != "screw", d.pose.y_mm)):
            head = f"{d.cls} {d.confidence:.2f}   ({d.pose.x_mm:.0f}, {d.pose.y_mm:.0f}) mm  {d.pose.yaw_deg:.0f} deg"
            note, bad = "", False
            if d.cls != "bin" and any(d.pose.distance_to(b) < 25 for b in bins):
                note = "inside a bin"
            elif real and d.cls != "bin":
                same = [math.hypot(d.pose.x_mm - x, d.pose.y_mm - y) for c, x, y in spots if c == d.cls]
                if same and min(same) < 60:
                    offs.append(min(same))
                    bad = min(same) > JAW_SLACK_MM
                    note = f"{min(same):.1f} mm from its taught spot" + (": blind miss" if bad else ": ok")
            rows.append((head, note, bad, COLOURS.get(d.cls, (255, 255, 255))))

        img = np.full((H, W, 3), (30, 34, 38), np.uint8)
        img[110:110 + IH, :IW] = view
        title = (f"REAL cell camera (test only)  |  {path.stem.replace('_', ' ')} of the official Guide run, before the first pick"
                 if real else "ISAAC SIM twin render  |  thin lines = labels computed from the 3D scene, thick = the model")
        banner(img, title, 44, 0.95, w=W)
        banner(img, f"YOLO11n-OBB, trained on 3,000 twin renders and 0 real images  |  {ms:.0f} ms on this laptop GPU",
               92, 0.75, colour=(210, 230, 255), bg=(50, 56, 62), w=W)
        cv2.putText(img, f"What the model found ({len(rows)})", (IW + 22, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y = 196
        for head, note, bad, col in rows[:14]:
            cv2.circle(img, (IW + 30, y - 8), 8, col, -1)
            cv2.putText(img, head, (IW + 48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if note:
                cv2.putText(img, note, (IW + 48, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (110, 120, 255) if bad else (190, 200, 205), 1, cv2.LINE_AA)
            y += 62 if note else 38
        if real and offs:
            msg = (f"{sum(o > JAW_SLACK_MM for o in offs)} of {len(offs)} parts are {min(offs):.0f}-{max(offs):.0f} mm "
                   f"from their taught spots (red X); the jaws allow about 3 mm.")
            msg2 = "A taught program reaches the X and misses. TwinCheck aims at the box the camera found."
        else:
            msg = "Every image is randomized: part pose and colour, bins anywhere, lighting, clutter, camera jitter."
            msg2 = "The labels come free from the 3D scene, so no one hand-labels anything."
        cv2.putText(img, msg, (20, 110 + IH + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, msg2, (20, 110 + IH + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190, 200, 205), 2, cv2.LINE_AA)
        banner(img, "[space] next   [left] back   [a] auto-play   [f] full screen   [q] quit", H - 10, 0.55,
               colour=(160, 165, 170), bg=(30, 34, 38), w=W)
        if args.save:
            cv2.imwrite(str(Path(args.save) / f"{path.parent.name}_{path.stem}.jpg"), img)
            print(path.parent.name, path.stem, f"{ms:.0f} ms", [r[0] + "  " + r[1] for r in rows])
            i += 1
            if i == len(images):
                return
            continue
        cv2.imshow(win, img)
        shown = time.time()

        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if k in (ord(" "), 83, ord("d")):
                i += 1; break
            if k in (81, ord("b")):
                i -= 1; break
            if k == ord("a"):
                auto = not auto
            if k == ord("f"):
                full = not full
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            if auto and time.time() - shown > args.seconds:
                i += 1; break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return


if __name__ == "__main__":
    main()
