"""Compare wrist grasp-check methods on the same labeled images.

Folder layout: data/wrist_real_test/{holding,empty,misaligned}/*.jpg
(real wrist photos from CSI; or a twin val split for a first look)

    python perception/eval_wrist.py --data data/wrist_real_test --method reference_diff --reference models/wrist_empty_ref.png
    python perception/eval_wrist.py --data data/wrist_real_test --method classifier --classifier runs/classify/train/weights/best.pt
    python perception/eval_wrist.py --data data/wrist_real_test --method cosmos

The number that matters most: "holding" said on an EMPTY grasp (a missed failure).
Acceptance gate: >= 95% correct and zero empty-called-holding.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "twincheck_core"), str(ROOT / "ros2_ws/src/twincheck_perception")]

from twincheck_core import load_config  # noqa: E402
from twincheck_perception.infer import WristChecker  # noqa: E402

STATES = ["holding", "empty", "misaligned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", choices=["reference_diff", "classifier", "cosmos"], required=True)
    ap.add_argument("--reference")
    ap.add_argument("--classifier")
    ap.add_argument("--roi", type=int, nargs=4, help="x y w h of the finger gap")
    ap.add_argument("--diff-threshold", type=float, default=18.0)
    ap.add_argument("--object-name", default="vial")
    args = ap.parse_args()

    cfg = load_config()
    cos = cfg.raw["models"]["cosmos"]
    chk = WristChecker(args.method, reference_path=args.reference or cfg.raw["models"]["wrist_reference"],
                       roi=tuple(args.roi) if args.roi else None, diff_threshold=args.diff_threshold,
                       classifier_path=args.classifier or cfg.raw["models"]["wrist_classifier"],
                       cosmos_base_url=cos["base_url"], cosmos_model=cos["model"],
                       cosmos_api_key_env=cos["api_key_env"], object_name=args.object_name)
    confusion = Counter()
    for truth in STATES:
        for p in sorted((Path(args.data) / truth).glob("*.jpg")):
            pred, _, _ = chk.check(cv2.imread(str(p)))
            confusion[(truth, pred.value)] += 1
    total = sum(confusion.values())
    if not total:
        raise SystemExit("no images found")
    correct = sum(v for (t, p), v in confusion.items() if t == p)
    # Reference diff only separates empty vs not-empty; count holding/misaligned as "not empty" for it
    if args.method == "reference_diff":
        correct = sum(v for (t, p), v in confusion.items() if (t == "empty") == (p == "empty"))
    empty_as_holding = confusion[("empty", "holding")] + confusion[("empty", "misaligned")]

    cols = STATES + ["unknown"]
    header = "truth / pred"
    print(f"\n{header:<14}" + "".join(f"{c:>12}" for c in cols))
    for t in STATES:
        print(f"{t:<14}" + "".join(f"{confusion[(t, c)]:>12}" for c in cols))
    acc = correct / total
    print(f"\naccuracy {acc:.1%} ({correct}/{total});  empty grasps called held: {empty_as_holding}")
    print(f"GATE: {'PASS' if acc >= 0.95 and empty_as_holding == 0 else 'NOT YET'}")
    print(json.dumps({"method": args.method, "accuracy": acc, "empty_called_held": empty_as_holding}))


if __name__ == "__main__":
    main()
