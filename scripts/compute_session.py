"""Log GPU sessions so the Brev budget never surprises us.

    python scripts/compute_session.py start --machine brev:render-1 --gpu "RTX PRO 6000" --purpose "render v2" --owner rohan
    python scripts/compute_session.py stop --id 7 --credits 6.5
    python scripts/compute_session.py list

Reminder printed on every start: STOP THE BREV INSTANCE WHEN YOU ARE DONE.
"""
import argparse
import os

import psycopg

DSN = os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck")
BUDGET = 150.0
WARN_AT = 75.0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--machine", required=True)
    s.add_argument("--gpu", default="")
    s.add_argument("--purpose", default="")
    s.add_argument("--owner", default="")
    t = sub.add_parser("stop")
    t.add_argument("--id", type=int, required=True)
    t.add_argument("--credits", type=float, default=0.0)
    sub.add_parser("list")
    args = ap.parse_args()

    db = psycopg.connect(DSN, autocommit=True)
    if args.cmd == "start":
        sid = db.execute("INSERT INTO compute_sessions (machine, gpu, purpose, owner) VALUES (%s,%s,%s,%s) RETURNING session_id",
                         (args.machine, args.gpu, args.purpose, args.owner)).fetchone()[0]
        print(f"session {sid} started. STOP THE INSTANCE WHEN YOU ARE DONE, then: compute_session.py stop --id {sid} --credits N")
    elif args.cmd == "stop":
        db.execute("UPDATE compute_sessions SET stopped_at=now(), credits_used=%s WHERE session_id=%s", (args.credits, args.id))
    total = db.execute("SELECT coalesce(sum(credits_used),0) FROM compute_sessions").fetchone()[0]
    if args.cmd == "list":
        for r in db.execute("SELECT session_id, started_at, stopped_at, machine, purpose, credits_used, owner "
                            "FROM compute_sessions ORDER BY session_id"):
            print(r)
        open_ = db.execute("SELECT session_id, machine FROM compute_sessions WHERE stopped_at IS NULL").fetchall()
        if open_:
            print(f"\nSTILL OPEN (is the instance stopped?): {open_}")
    print(f"credits used: {total:.1f} / {BUDGET:.0f}" + ("   <- over the 75-credit check-in: ask in #compute-questions"
                                                         if total > WARN_AT else ""))


if __name__ == "__main__":
    main()
