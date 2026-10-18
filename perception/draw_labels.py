"""Draw YOLO-OBB labels on images: check synthetic labels BEFORE rendering thousands.

    python perception/draw_labels.py --data data/overhead_v1_seed42_n500 --n 20 --out data/check_v1

Open the output folder and look: every object should have a tight box on its
edges, with the right class name. A wrong box here means a wrong model later.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

COLORS = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (220, 60, 220), (255, 255, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder with data.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data)
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    imgs = sorted((root / "images" / args.split).glob("*.jpg"))
    random.Random(args.seed).shuffle(imgs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    empty = 0
    for p in imgs[:args.n]:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        lab = root / "labels" / args.split / (p.stem + ".txt")
        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        if not lines:
            empty += 1
        for parts in lines:
            c = int(parts[0])
            pts = (np.array(parts[1:9], float).reshape(4, 2) * [w, h]).astype(np.int32)
            col = COLORS[c % len(COLORS)]
            cv2.polylines(img, [pts], True, col, 2)
            cv2.putText(img, str(names[c]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite(str(out / p.name), img)
    print(f"wrote {min(args.n, len(imgs))} images to {out} ({empty} with no labels)")


if __name__ == "__main__":
    main()
