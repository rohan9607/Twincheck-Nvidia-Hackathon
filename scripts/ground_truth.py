"""The Recorder's tool during a real-arm session: enter what actually happened.

Runs in a terminal next to the arm. For each new trial the logger creates, it asks:
  did the pick work? (y/n), which failure did we cause (enter = none), notes.

    python scripts/ground_truth.py --session oct29_numbers

Type 'u' to fix the previous trial, 'q' to quit. Answers are what a PERSON saw,
never what TwinCheck said.
"""
import argparse
import os
import time

import psycopg

DSN = os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@localhost:5432/twincheck")
FAILS = {"": None, "e": "empty_grasp", "r": "removed_mid_grasp", "n": "nudged_part"}


def ask(db, tid):
    row = db.execute("SELECT object_class, mode, pick_point FROM trials WHERE trial_id=%s", (tid,)).fetchone()
    print(f"\ntrial {tid}: {row[0]} at {row[2]} ({row[1]})")
    while True:
        a = input("  pick worked? [y/n/u=fix previous/q]: ").strip().lower()
        if a in ("y", "n", "u", "q"):
            break
    if a in ("u", "q"):
        return a
    f = input("  failure we caused? [enter=none, e=empty grasp, r=removed mid-grasp, n=nudged]: ").strip().lower()
    notes = input("  notes (enter to skip): ").strip()
    db.execute("UPDATE trials SET pick_success=%s, induced_failure=%s, notes=coalesce(nullif(%s,''), notes) WHERE trial_id=%s",
               (a == "y", FAILS.get(f), notes, tid))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="the session_tag used in the launch command")
    args = ap.parse_args()
    db = psycopg.connect(DSN, autocommit=True)
    done = []
    print("waiting for trials... (Ctrl+C to stop)")
    while True:
        rows = db.execute("SELECT trial_id FROM trials WHERE session_tag=%s AND pick_success IS NULL ORDER BY trial_id",
                          (args.session,)).fetchall()
        for (tid,) in rows:
            r = ask(db, tid)
            if r == "q":
                return
            if r == "u" and done:
                ask(db, done[-1])
                ask(db, tid)
            done.append(tid)
        time.sleep(1)


if __name__ == "__main__":
    main()
