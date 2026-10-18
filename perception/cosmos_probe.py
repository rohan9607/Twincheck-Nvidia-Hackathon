"""Quick check of a vision-language model on a few wrist images before wiring it in.

Organizers recommend Cosmos Nano on the Spark (tested on this exact setup).
Point it at the local server on the Spark, or at build.nvidia.com:

    export NVIDIA_API_KEY=...            # only for build.nvidia.com; never commit it
    python perception/cosmos_probe.py --images "data/wrist_real_test/*/*.jpg" \
        --base-url http://localhost:8000/v1 --model <model id from the Spark guide>

Prints each answer and the time per image (it must be fast enough to answer
before the arm reaches the bin).
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "twincheck_core"), str(ROOT / "ros2_ws/src/twincheck_perception")]
from twincheck_perception.infer import WristChecker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--object-name", default="vial")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    chk = WristChecker("cosmos", cosmos_base_url=args.base_url, cosmos_model=args.model, object_name=args.object_name)
    times = []
    for p in sorted(glob.glob(args.images))[:args.limit]:
        t0 = time.time()
        state, conf, detail = chk.check(cv2.imread(p))
        times.append(time.time() - t0)
        print(f"{Path(p).parent.name:>11} / {Path(p).name:<28} -> {state.value:<11} {detail}  ({times[-1]:.2f}s)")
    if times:
        print(f"\nmean {sum(times) / len(times):.2f}s per image")


if __name__ == "__main__":
    main()
