"""Try the whole check logic on any laptop, with no ROS, no twin and no GPU.

Generates fake trials (objects near the pick point, some shifted, some failed
grasps, some dropped), runs them through the same Comparator the ROS node uses,
and prints what TwinCheck would say. Good for learning how the checks behave and
for tuning thresholds before the twin is ready.

    python scripts/simulate_checks.py --n 200 --shift-rate 0.2 --drop-rate 0.1 --mode watch
"""
import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "twincheck_core"))
from twincheck_core import (Comparator, Detection, GraspState, Mode, Pose2D, load_config,  # noqa: E402
                            rate_with_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["watch", "guide"], default="watch")
    ap.add_argument("--noise-mm", type=float, default=2.0, help="detector position noise")
    ap.add_argument("--shift-rate", type=float, default=0.2, help="share of trials with the part moved 15-40 mm")
    ap.add_argument("--empty-rate", type=float, default=0.1, help="share with a missed grasp")
    ap.add_argument("--drop-rate", type=float, default=0.05, help="share dropped on the way to the bin")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    rng = random.Random(args.seed)
    comp = Comparator(cfg, Mode(args.mode))
    caught = failures = false_alarms = good = 0
    for t in range(1, args.n + 1):
        cls = rng.choice(list(cfg.objects))
        slot = rng.choice(list(cfg.pick_points))
        pick = cfg.pick_points[slot]
        comp.start_trial(t, cls, pick)
        shifted = rng.random() < args.shift_rate
        d = rng.uniform(15, 40) if shifted else 0.0
        a = rng.uniform(0, 6.283)
        true_pose = Pose2D(pick.x_mm + d * math.cos(a), pick.y_mm + d * math.sin(a), pick.yaw_deg)
        seen = Pose2D(true_pose.x_mm + rng.gauss(0, args.noise_mm), true_pose.y_mm + rng.gauss(0, args.noise_mm),
                      true_pose.yaw_deg + rng.gauss(0, 2))
        decisions = [comp.on_pre_pick([Detection(cls, seen, 0.9)])]
        # what actually happens: a shifted part is missed when the shift is larger than the jaw slack
        missed = (shifted and d > 12) or rng.random() < args.empty_rate
        decisions.append(comp.on_grasp_closed(GraspState.EMPTY if missed else GraspState.HOLDING))
        dropped = (not missed) and rng.random() < args.drop_rate
        bin_c = cfg.bins[cfg.objects[cls].target_bin].center
        if missed:
            place_dets = [Detection(cls, true_pose, 0.9)]
        elif dropped:
            place_dets = []
        else:
            place_dets = [Detection(cls, Pose2D(bin_c.x_mm + rng.gauss(0, 8), bin_c.y_mm + rng.gauss(0, 8)), 0.9)]
        decisions.append(comp.on_place_done(place_dets))

        success = not missed and not dropped
        flagged = any(not x.outcome.passed for x in decisions)
        if success:
            good += 1
            false_alarms += flagged
        else:
            failures += 1
            caught += flagged
        if args.verbose or (flagged and t <= 10):
            for x in decisions:
                if not x.outcome.passed:
                    print(f"trial {t:3d} {x.outcome.check_point.value:<10} {x.action.value:<6} {x.outcome.reason}")

    print(f"\n{args.n} simulated trials ({args.mode} mode)")
    print(f"failures caught: {rate_with_interval(caught, failures)}")
    print(f"false alarms:    {rate_with_interval(false_alarms, good)}")


if __name__ == "__main__":
    main()
