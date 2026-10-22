"""Every number for the video, the summary and the finals, straight from the trial log.

    python scripts/results.py                         # all tables
    python scripts/results.py --env real --csv runs/results_real.csv
    python scripts/results.py --sweep --env twin      # threshold sweep (caught vs false alarms)

Rates always come with a 95% Wilson interval: with 20-30 real trials, a rate
without its range overstates what we know.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "twincheck_core"))
from twincheck_core import pick_threshold, rate_with_interval, threshold_sweep  # noqa: E402

DSN = os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck")


def q(db, sql, params=()):
    cur = db.execute(sql, params)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def table(title, rows, cols):
    print(f"\n{title}\n" + "-" * len(title))
    if not rows:
        print("  (no data yet)")
        return
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["twin", "real"])
    ap.add_argument("--session")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--csv", help="also write the per-trial summary here")
    args = ap.parse_args()

    import psycopg
    db = psycopg.connect(DSN)
    where, params = ["true"], []
    if args.env:
        where.append("env = %s")
        params.append(args.env)
    if args.session:
        where.append("session_tag = %s")
        params.append(args.session)
    w = " AND ".join(where)
    rows = q(db, f"SELECT * FROM trial_summary WHERE {w} ORDER BY trial_id", params)

    # 1. Pick success: baseline vs TwinCheck modes
    out = []
    for key in sorted({(r["env"], r["mode"], r["ladder_rung"]) for r in rows}):
        g = [r for r in rows if (r["env"], r["mode"], r["ladder_rung"]) == key and r["pick_success"] is not None
             and r["induced_failure"] is None]
        if g:
            ok = sum(r["pick_success"] for r in g)
            out.append({"env": key[0], "mode": key[1], "rung": key[2], "pick success": rate_with_interval(ok, len(g))})
    table("Pick success (normal trials, no induced failure)", out, ["env", "mode", "rung", "pick success"])

    # 2. Failures caught: real failures that TwinCheck flagged. An induced failure is always a real one
    #    for a fixed program; in Guide it only counts if the pick still failed (else Guide recovered it, table 2b)
    def failed(r):
        return r["pick_success"] is False or (r["induced_failure"] is not None and r["mode"] != "guide")

    out = []
    for key in sorted({(r["env"], r["mode"]) for r in rows if r["mode"] != "baseline"}):
        fails = [r for r in rows if (r["env"], r["mode"]) == key and failed(r)]
        for kind in sorted({r["induced_failure"] or "natural" for r in fails}):
            g = [r for r in fails if (r["induced_failure"] or "natural") == kind]
            out.append({"env": key[0], "mode": key[1], "failure": kind,
                        "caught": rate_with_interval(sum(bool(r["flagged"]) for r in g), len(g))})
    table("Failures caught by TwinCheck", out, ["env", "mode", "failure", "caught"])

    # 2b. Guide recovery: induced failures that still ended with the part in the right bin
    out = []
    for key in sorted({(r["env"], r["induced_failure"]) for r in rows if r["mode"] == "guide" and r["induced_failure"]}):
        g = [r for r in rows if r["mode"] == "guide" and (r["env"], r["induced_failure"]) == key
             and r["pick_success"] is not None]
        if g:
            out.append({"env": key[0], "failure": key[1], "retried": sum(bool(r["retried"]) for r in g),
                        "recovered": rate_with_interval(sum(bool(r["pick_success"]) for r in g), len(g))})
    table("Guide: induced failures recovered", out, ["env", "failure", "retried", "recovered"])

    # 3. False alarms: good picks that TwinCheck flagged anyway
    out = []
    for key in sorted({(r["env"], r["mode"]) for r in rows if r["mode"] != "baseline"}):
        good = [r for r in rows if (r["env"], r["mode"]) == key and r["pick_success"] and not r["induced_failure"]]
        if good:
            out.append({"env": key[0], "mode": key[1],
                        "false alarms": rate_with_interval(sum(bool(r["flagged"]) for r in good), len(good))})
    table("False alarms (good picks flagged)", out, ["env", "mode", "false alarms"])

    # 4. Which check caught what
    out = []
    for kind in sorted({r["induced_failure"] for r in rows if r["induced_failure"]}):
        g = [r for r in rows if r["induced_failure"] == kind]
        out.append({"failure": kind, "n": len(g),
                    "pre_pick": sum(bool(r["flagged_pre_pick"]) for r in g),
                    "post_grasp": sum(bool(r["flagged_post_grasp"]) for r in g),
                    "post_place": sum(bool(r["flagged_post_place"]) for r in g)})
    table("Which check caught each induced failure", out, ["failure", "n", "pre_pick", "post_grasp", "post_place"])

    # 5. Drift: how far resync had to move objects
    drift = [r["resync_max_shift_mm"] for r in rows if r["resync_max_shift_mm"] is not None]
    if drift:
        drift.sort()
        print(f"\nResync shift: median {drift[len(drift) // 2]:.1f} mm, max {drift[-1]:.1f} mm over {len(drift)} picks")

    # 6. Training runs
    runs = q(db, "SELECT run_id, model, dataset, gpu, runtime_min, credits_used, real_recall, real_pos_err_mm "
                 "FROM training_runs ORDER BY run_id")
    table("Training runs", runs, ["run_id", "model", "dataset", "gpu", "runtime_min", "credits_used", "real_recall",
                                  "real_pos_err_mm"])
    credits = q(db, "SELECT coalesce(sum(credits_used),0) AS c FROM compute_sessions")[0]["c"] + \
        sum((r["credits_used"] or 0) for r in runs)
    print(f"\nBrev credits logged so far: {credits:.1f} of the ~150 budget")

    if args.sweep:
        fail = [r["pre_pick_err_mm"] for r in rows if r["induced_failure"] == "nudged_part" and r["pre_pick_err_mm"] is not None]
        ok = [r["pre_pick_err_mm"] for r in rows if r["pick_success"] and not r["induced_failure"] and r["pre_pick_err_mm"] is not None]
        sweep = threshold_sweep(fail, ok, [2, 4, 6, 8, 10, 12, 15, 20, 25, 30])
        table("Pre-pick threshold sweep (mm)", sweep, ["threshold", "caught", "failures", "false_alarms", "good_picks"])
        best = pick_threshold(sweep)
        print(f"\nRecommended pre_pick_mm: {best['threshold'] if best else 'none catches every failure yet'}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["trial_id"])
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nper-trial CSV: {args.csv}")


if __name__ == "__main__":
    main()
