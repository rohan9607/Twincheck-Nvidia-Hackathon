"""TwinCheck dashboard API: live alerts, results and images from the trial database.

Other teams share the Spark and can reach open ports, so EVERY request needs the
token from TWINCHECK_DASH_TOKEN (header "Authorization: Bearer <token>" or ?token=).

    TWINCHECK_DASH_TOKEN=$(openssl rand -hex 16) uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8090
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "twincheck_core"))
from twincheck_core import wilson_interval  # noqa: E402

DSN = os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck")
TOKEN = os.environ.get("TWINCHECK_DASH_TOKEN", "")
IMAGE_ROOT = Path(os.environ.get("TWINCHECK_IMAGE_ROOT", ROOT)).resolve()

app = FastAPI(title="TwinCheck dashboard")


def auth(request: Request, token: str | None = Query(default=None)):
    if not TOKEN:
        raise HTTPException(500, "set TWINCHECK_DASH_TOKEN before starting the dashboard")
    given = token or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(given, TOKEN):
        raise HTTPException(401, "bad or missing token")


def rows(sql: str, params=()) -> list[dict]:
    with psycopg.connect(DSN) as db:
        cur = db.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "lo": lo, "hi": hi}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/summary", dependencies=[Depends(auth)])
def summary(env: str | None = None, session: str | None = None):
    where, p = ["true"], []
    if env:
        where.append("env = %s")
        p.append(env)
    if session:
        where.append("session_tag = %s")
        p.append(session)
    t = rows(f"SELECT * FROM trial_summary WHERE {' AND '.join(where)}", p)
    groups = {}
    for r in t:
        g = groups.setdefault(f"{r['env']}|{r['mode']}", {"env": r["env"], "mode": r["mode"], "trials": 0,
                                                            "_ok": [0, 0], "_caught": [0, 0], "_false": [0, 0]})
        g["trials"] += 1
        if r["pick_success"] is not None and not r["induced_failure"]:
            g["_ok"][1] += 1
            g["_ok"][0] += bool(r["pick_success"])
        if r["induced_failure"] or r["pick_success"] is False:
            g["_caught"][1] += 1
            g["_caught"][0] += bool(r["flagged"])
        if r["pick_success"] and not r["induced_failure"]:
            g["_false"][1] += 1
            g["_false"][0] += bool(r["flagged"])
    out = []
    for g in groups.values():
        out.append({"env": g["env"], "mode": g["mode"], "trials": g["trials"], "pick_success": rate(*g["_ok"]),
                    "failures_caught": rate(*g["_caught"]), "false_alarms": rate(*g["_false"])})
    return sorted(out, key=lambda x: (x["env"], x["mode"]))


@app.get("/api/trials", dependencies=[Depends(auth)])
def trials(limit: int = 100):
    return rows("SELECT trial_id, started_at, env, mode, ladder_rung, object_class, induced_failure, pick_success, "
                "flagged, retried, resync_max_shift_mm, session_tag FROM trial_summary ORDER BY trial_id DESC LIMIT %s",
                (limit,))


@app.get("/api/checks", dependencies=[Depends(auth)])
def checks(limit: int = 50, failed_only: bool = False):
    f = "WHERE NOT c.passed" if failed_only else ""
    return rows(f"SELECT c.*, t.object_class, t.env, t.mode FROM checks c JOIN trials t USING (trial_id) {f} "
                "ORDER BY c.check_id DESC LIMIT %s", (limit,))


@app.get("/api/training", dependencies=[Depends(auth)])
def training():
    return rows("SELECT * FROM training_runs ORDER BY run_id")


@app.get("/api/stream", dependencies=[Depends(auth)])
async def stream():
    """Server-sent events: every new check as it is logged."""
    async def gen():
        last = (rows("SELECT coalesce(max(check_id), 0) AS m FROM checks")[0]["m"])
        while True:
            new = rows("SELECT c.*, t.object_class FROM checks c JOIN trials t USING (trial_id) "
                       "WHERE c.check_id > %s ORDER BY c.check_id", (last,))
            for r in new:
                last = r["check_id"]
                yield f"data: {json.dumps(r, default=str)}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/image", dependencies=[Depends(auth)])
def image(path: str):
    p = (IMAGE_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if IMAGE_ROOT not in p.parents and p != IMAGE_ROOT:
        raise HTTPException(403, "outside the image folder")
    if not p.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(p)


web = ROOT / "dashboard" / "web" / "dist"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
