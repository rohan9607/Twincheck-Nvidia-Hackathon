"""TwinCheck as a policy for the organizers' cell client (github.com/SIDCLab/PhysicalAI-Hackathon-2026).

The client calls act(obs); we return fingertip rows ("tcp" actions). The pick-and-place logic and the
three checks are twincheck_core.PickLoop; this file only turns camera images into what the loop asked
for, splits rows into chunks the cell accepts, and logs every trial and check.

The cell moves the real arm only during an approved, operator-supervised slot, and enforces its own
limits; PickLoop refuses any target outside our tighter ones (config/twincheck.yaml cell:).

Environment:
  TWINCHECK_MODE     baseline | watch | guide (default guide)
  TWINCHECK_SESSION  session tag for the trial database and the JSONL log (default cell_<date>)
  TWINCHECK_ENV      real (default) | twin: the trials.env column
  TWINCHECK_DSN      Postgres URL; trials and checks go to the database when set (as scripts/results.py reads)
  TWINCHECK_WRIST    reference_diff (default) | classifier | cosmos
  TWINCHECK_CONFIG   another config file
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "twincheck_core", REPO / "ros2_ws" / "src" / "twincheck_perception"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from twincheck_core import Job, Need, PickLoop, View, load_config  # noqa: E402

log = logging.getLogger("twincheck.policy")
CFG = load_config()
CELL = CFG.raw["cell"]
MAX_CHUNK = 100   # rows per act() the cell accepts


class TrialLog:
    """One JSONL line per event (always), and trials/checks rows in Postgres (when TWINCHECK_DSN is set)."""

    def __init__(self, session: str, env: str, mode: str):
        self.session, self.env, self.mode = session, env, mode
        out = REPO / "runs" / "cell"
        out.mkdir(parents=True, exist_ok=True)
        self.jsonl = out / f"{session}.jsonl"
        self.db, self.db_ids = None, {}
        dsn = os.environ.get("TWINCHECK_DSN")
        if dsn:
            try:
                import psycopg
                self.db = psycopg.connect(dsn, autocommit=True)
            except Exception as e:  # noqa: BLE001  the arm must not stop because the database is down
                log.warning("trial database unavailable (%s); logging to %s only", type(e).__name__, self.jsonl)

    def write(self, rollout: dict, ev) -> None:
        d = ev.decision
        rec = {"t": dt.datetime.now().isoformat(timespec="milliseconds"), "session": self.session, "mode": self.mode,
               "rollout": rollout.get("trial_id"), "layout": rollout.get("layout"), "kind": ev.kind,
               "trial": ev.trial_id, "cls": ev.cls, "note": ev.note}
        if ev.aim:
            rec["aim"] = [round(ev.aim.x_mm, 1), round(ev.aim.y_mm, 1), round(ev.aim.yaw_deg, 1)]
        if d:
            o = d.outcome
            rec |= {"check": o.check_point.value, "passed": o.passed, "reason": o.reason, "action": d.action.value,
                    "urgent": d.urgent, "pos_err_mm": o.position_error_mm, "yaw_err_deg": o.yaw_error_deg}
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        level = logging.WARNING if (d and not d.outcome.passed) or ev.kind in ("stop", "skip") else logging.INFO
        log.log(level, "trial %s %s %s", ev.trial_id, ev.kind, rec.get("reason") or ev.note)
        if self.db is not None:
            try:
                self._db(rollout, ev)
            except Exception as e:  # noqa: BLE001
                log.warning("trial database write failed: %s", e)

    def _db(self, rollout: dict, ev) -> None:
        if ev.kind == "trial":
            note = f"cell rollout {rollout.get('trial_id')} layout {rollout.get('layout')}"
            self.db_ids[ev.trial_id] = self.db.execute(
                """INSERT INTO trials (env, mode, ladder_rung, object_class, session_tag, ros_trial_id, pick_point, notes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING trial_id""",
                (self.env, self.mode, "cell", ev.cls, self.session, ev.trial_id,
                 f"{ev.aim.x_mm:.0f},{ev.aim.y_mm:.0f}", note)).fetchone()[0]
        elif ev.kind == "decision" and ev.trial_id in self.db_ids:
            d, o = ev.decision, ev.decision.outcome
            tid = self.db_ids[ev.trial_id]
            self.db.execute(
                """INSERT INTO checks (trial_id, check_point, passed, position_err_mm, yaw_err_deg, reason, action,
                                       urgent, expected_source) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (tid, o.check_point.value, o.passed, o.position_error_mm, o.yaw_error_deg, o.reason, d.action.value,
                 d.urgent, "cell_policy"))
            if d.action.value == "retry":
                self.db.execute("UPDATE trials SET retried = true WHERE trial_id = %s", (tid,))


class Policy:
    CAMERAS = ["table", "wrist"]
    IMAGE_SIZE = tuple(CELL["image_size"])
    IMAGE_FORMAT = CELL["image_format"]
    ACTION = "tcp"
    PREFETCH = False
    DEPTH: list[str] = []
    DT = CELL["step_s"]

    def __init__(self, checkpoint: str | None = None):
        self.mode = os.environ.get("TWINCHECK_MODE", "guide")
        if self.mode not in ("baseline", "watch", "guide"):
            raise ValueError(f"TWINCHECK_MODE must be baseline, watch or guide, not {self.mode!r}")
        session = os.environ.get("TWINCHECK_SESSION") or f"cell_{dt.date.today():%Y%m%d}"
        self.trials = TrialLog(session, os.environ.get("TWINCHECK_ENV", "real"), self.mode)
        self.jobs = [Job(j["cls"], CFG.pick_points[j["pick_point"]]) for j in CELL["sequence"]]
        self.table = self._make(lambda: self._table(checkpoint), "table detector")
        self.wrist = self._make(lambda: self._wrist(), "wrist check")
        self.next_trial = 1
        self.loop: PickLoop | None = None
        self.pending: list[list[float]] = []
        self.rollout: dict = {}
        log.info("TwinCheck %s mode, session %s, %d jobs per rollout", self.mode, session, len(self.jobs))

    @staticmethod
    def _make(build, what: str):
        try:
            return build()
        except Exception as e:  # noqa: BLE001  without it the checks see nothing and fail, which is safe
            log.warning("no %s (%s: %s): its checks will fail", what, type(e).__name__, e)
            return None

    @staticmethod
    def _table(checkpoint: str | None):
        from cell_perception import TableDetector
        path = Path(checkpoint) if checkpoint else REPO / CFG.raw["models"]["detector"]
        if not path.exists():
            raise FileNotFoundError(path)
        return TableDetector(CFG, path)

    @staticmethod
    def _wrist():
        from cell_perception import WristCheck
        return WristCheck(CFG, os.environ.get("TWINCHECK_WRIST", "reference_diff"), REPO)

    def reset(self) -> None:
        if self.loop is not None:
            self.next_trial = self.loop.next_id
        self.loop = PickLoop(CFG, self.mode, self.jobs, first_trial_id=self.next_trial,
                             max_picks=int(CELL["max_picks_per_rollout"]))
        self.pending = []
        self.rollout = {}

    def act(self, obs: dict):
        if self.loop is None:
            self.reset()
        self.rollout = obs.get("trial") or self.rollout
        if not self.pending:
            view = View(tcp=obs.get("tcp_pose"), gripper=float(obs["joints"][6]))
            if self.loop.needs == Need.SCENE:
                view.detections = self.table.detect(obs["images"]["table"]) if self.table else []
            elif self.loop.needs == Need.GRASP and self.wrist:
                view.grasp = self.wrist.check(obs["images"]["wrist"])
            self.pending = self.loop.step(view)
            for ev in self.loop.drain_events():
                self.trials.write(self.rollout, ev)
        chunk, self.pending = self.pending[:MAX_CHUNK], self.pending[MAX_CHUNK:]
        return chunk
