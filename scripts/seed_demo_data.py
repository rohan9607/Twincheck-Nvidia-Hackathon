"""Fill the database with fake trials so the dashboard and results scripts can be
built and tested before the twin or the arm is ready. Tagged session 'demo';
remove with --clear.

    python scripts/seed_demo_data.py            # add ~180 demo trials
    python scripts/seed_demo_data.py --clear    # delete them again
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "twincheck_core"))
from twincheck_core import Comparator, Detection, GraspState, Mode, Pose2D, load_config  # noqa: E402

DSN = os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    db = psycopg.connect(DSN, autocommit=True)
    if args.clear:
        n = db.execute("DELETE FROM trials WHERE session_tag='demo'").rowcount
        print(f"deleted {n} demo trials")
        return

    cfg = load_config()
    rng = random.Random(args.seed)
    plan = [("twin", "baseline", 60), ("twin", "watch", 60), ("real", "baseline", 30), ("real", "watch", 30)]
    for env, mode, n in plan:
        comp = Comparator(cfg, Mode.WATCH)
        for i in range(n):
            cls = rng.choice(list(cfg.objects))
            slot = rng.choice(list(cfg.pick_points))
            pick = cfg.pick_points[slot]
            induced = rng.choice([None, None, None, "nudged_part", "empty_grasp"]) if mode == "watch" else None
            shift = rng.uniform(20, 40) if (induced == "nudged_part" or (mode == "baseline" and rng.random() < 0.3)) else 0
            a = rng.uniform(0, 2 * math.pi)
            noise = 2 if env == "twin" else 4
            obs = Pose2D(pick.x_mm + shift * math.cos(a) + rng.gauss(0, noise), pick.y_mm + shift * math.sin(a) + rng.gauss(0, noise))
            missed = shift > 12 or induced == "empty_grasp"
            success = not missed
            tid = db.execute("""INSERT INTO trials (env, mode, ladder_rung, object_class, induced_failure, pick_success,
                                session_tag, pick_point, seed) VALUES (%s,%s,'random_pos',%s,%s,%s,'demo',%s,%s)
                                RETURNING trial_id""", (env, mode, cls, induced, success, slot, args.seed)).fetchone()[0]
            comp.start_trial(tid, cls, pick)
            bin_c = cfg.bins[cfg.objects[cls].target_bin].center
            outs = [comp.on_pre_pick([Detection(cls, obs)]),
                    comp.on_grasp_closed(GraspState.EMPTY if missed else GraspState.HOLDING),
                    comp.on_place_done([] if missed else [Detection(cls, Pose2D(bin_c.x_mm + rng.gauss(0, 6), bin_c.y_mm))])]
            for d in outs:
                o = d.outcome
                db.execute("""INSERT INTO checks (trial_id, check_point, passed, position_err_mm, yaw_err_deg, reason,
                              action, urgent, sync_gap_ms, expected_source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'config')""",
                           (tid, o.check_point.value, o.passed, o.position_error_mm, o.yaw_error_deg, o.reason,
                            d.action.value, d.urgent, rng.uniform(5, 40)))
    print("seeded demo trials (session_tag='demo')")


if __name__ == "__main__":
    main()
