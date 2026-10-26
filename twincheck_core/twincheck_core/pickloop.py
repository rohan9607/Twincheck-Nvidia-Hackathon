"""One rollout on the organizers' cell: pick-and-place with TwinCheck's three checks, as rows for act(obs).

The policy calls step(view) every time the cell asks for actions. step returns rows to play; when it
needs to look (the table camera before a pick and after a place, the wrist camera after a grasp) it parks
the arm, returns those rows, and sets `needs`. The policy then fills that part of the next view.

    SCAN (park, table camera) -> pre-pick check -> PICK (hover, descend, close, lift)
      -> post-grasp check -> PLACE (carry high, release over the bin)
      -> SCAN: post-place check, and the same scene is the next job's pre-pick scene

The post-grasp check (config cell.grasp_check) looks with the wrist camera ("wrist"), or parks and looks
with the table camera for the part where it was ("table"); a table retry re-aims from that same scene.

Every leg ends at an arrival gate: with feedback (view.tcp), the next leg, a gripper change and a camera
look wait until the measured fingertip is at the leg's goal. The real arm moves at about 55 mm/s and heads
straight for the latest target, so open-loop rows cut corners onto the parts and closed the jaws in the air
(dry run, Oct 23). Without feedback (the mock's first step, unit tests) the gates are skipped.

Modes (the same comparison the twin runs):
  baseline  blind fixed program: reaches for the configured pick points; checks are logged only
  watch     the same blind motion; failed checks are alerts
  guide     aims at the part the camera sees; a failed grasp retries once from a fresh scan;
            a failure with no retries left stops the rollout (the arm parks and holds); a bad place
            stops it too unless cell.stop_on_bad_place is false (then it is an alert, and the next part goes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .cell import CellSpec, OutsideWorkspace, Row, check_row, interpolate
from .checks import Detection, GraspState, grasp_from_table
from .comparator import Action, Comparator, Decision, Mode
from .config import Config
from .geometry import Pose2D
from .grasp import plan_grasp


class Need(str, Enum):
    NOTHING = "nothing"
    SCENE = "scene"     # table-camera detections, table frame mm
    GRASP = "grasp"     # wrist-camera grasp state


@dataclass
class View:
    """What the policy saw for this step. Only the part `needs` asked for has to be filled."""
    tcp: list[float] | None                     # obs["tcp_pose"], UF0; None before the first command (mock)
    gripper: float = 0.0
    detections: list[Detection] | None = None
    grasp: GraspState | None = None


@dataclass(frozen=True)
class Job:
    cls: str
    nominal: Pose2D     # where the blind program reaches


@dataclass
class Event:
    kind: str           # "trial" (a pick starts), "decision", "skip", "stop", "done"
    trial_id: int
    cls: str = ""
    aim: Pose2D | None = None
    decision: Decision | None = None
    note: str = ""


@dataclass(frozen=True)
class _Arrive:
    """Gate in the row queue: play nothing after it until the arm is at `row` (pose, R and gripper)."""
    row: Row


class _Phase(str, Enum):
    START = "start"
    SCAN = "scan"           # parked; the next view has a scene
    GRASPED = "grasped"     # lifted; the next view has the wrist grasp state
    DONE = "done"


@dataclass
class PickLoop:
    cfg: Config
    mode: str                       # baseline | watch | guide
    jobs: list[Job]
    first_trial_id: int = 1
    max_picks: int = 6
    grasp_by: str | None = None     # wrist | table; None = config cell.grasp_check
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self.spec = CellSpec.from_config(self.cfg)
        self.grasp_by = self.grasp_by or self.cfg.raw["cell"].get("grasp_check", "wrist")
        if self.grasp_by not in ("wrist", "table"):
            raise ValueError(f"cell.grasp_check must be wrist or table, not {self.grasp_by!r}")
        self._grasp_at: Pose2D | None = None
        # baseline runs the checks like watch; only what we do with a failure differs
        self.comparator = Comparator(self.cfg, Mode.GUIDE if self.mode == "guide" else Mode.WATCH,
                                     stop_on_bad_place=bool(self.cfg.raw["cell"].get("stop_on_bad_place", True)))
        self.phase = _Phase.START
        self._needs = Need.NOTHING
        self.last: Row | None = None        # the last row queued (where the arm will end up)
        self._queue: list = []              # rows and _Arrive gates not yet returned
        self._gate: _Arrive | None = None   # waiting for this arrival
        self._waited = 0                    # rows held at the current gate
        self.job_i = 0
        self.picks = 0
        self.next_id = self.first_trial_id
        self.trial_id: int | None = None   # the pick in progress
        self.placed_id: int | None = None  # placed, waiting for its post-place check

    # ------------------------------------------------------------------ public
    @property
    def done(self) -> bool:
        return self.phase == _Phase.DONE

    @property
    def needs(self) -> Need:
        """What the next view must carry. Only once every queued row has played and the arm has arrived."""
        return self._needs if not self._queue and self._gate is None else Need.NOTHING

    def step(self, view: View) -> list[Row]:
        """Rows to play next (never empty: when done, hold where the arm is)."""
        if self.last is None:
            g = 1.0 if view.gripper >= 0.5 else 0.0
            self.last = (list(view.tcp[:6]) + [g]) if view.tcp else self.spec.down(*self.spec.park, 0.0, g)
        if self._gate is not None:
            if not self._at(view, self._gate.row):
                self._waited += self.spec.wait_chunk_rows
                if self._waited <= self.spec.arrive_timeout_rows:
                    return [list(self._gate.row) for _ in range(self.spec.wait_chunk_rows)]
                goal, tcp = [round(v, 1) for v in self._gate.row[:3]], [round(v, 1) for v in view.tcp[:3]]
                self._queue, self.last, self._gate = [], list(self._gate.row), None
                return self._release(view, self._finish(
                    "stop", f"the arm did not reach {goal} (it is at {tcp}): blocked, e.g. resting on a part"))
            self._gate = None
        if not self._queue:
            try:
                self._queue = self._advance(view)
            except OutsideWorkspace as e:
                self._queue = self._finish("stop", f"refused a target outside our workspace: {e}")
        return self._release(view, [])

    def _at(self, view: View, row: Row) -> bool:
        return (view.tcp is None or self.spec.arrived(view.tcp, row)) and (view.gripper >= 0.5) == (row[6] >= 0.5)

    def _release(self, view: View, rows: list) -> list[Row]:
        """Queued rows up to the next arrival gate. Without feedback the gates pass straight away."""
        self._queue = rows + self._queue
        out: list[Row] = []
        while self._queue:
            item = self._queue.pop(0)
            if isinstance(item, _Arrive):
                if view.tcp is None:
                    continue
                self._gate, self._waited = item, 0
                break
            out.append(item)
        return out or [list(self._gate.row if self._gate else self.last)]

    def drain_events(self) -> list[Event]:
        out, self.events = self.events, []
        return out

    # ------------------------------------------------------------------ state machine
    def _advance(self, view: View) -> list[Row]:
        if self.phase == _Phase.DONE:
            return []
        if self.phase == _Phase.START:
            return self._park()
        if self.phase == _Phase.SCAN:
            dets = view.detections or []
            if self.placed_id is not None:
                self._log(self.comparator.on_place_done(dets, trial_id=self.placed_id), self.placed_id)
                last = self.events[-1].decision
                self.placed_id = None
                if last.action == Action.STOP:
                    return self._finish("stop", "post-place failed")
            return self._pre_pick(dets)
        if self.phase == _Phase.GRASPED:
            table = self.grasp_by == "table"
            if table:
                cls = self.comparator.trials[self.trial_id].cls
                state = grasp_from_table(self._grasp_at, cls, view.detections or [], self.cfg.thresholds)
            else:
                state = view.grasp or GraspState.UNKNOWN
            d = self.comparator.on_grasp_closed(state, trial_id=self.trial_id)
            self._log(d, self.trial_id)
            # the table camera saw the part still on the table, so nothing is held: open where we are.
            # From the wrist alone we can't be sure, so set it down where it was picked first.
            release = self._open_here() if table else self._release_here()
            if d.action == Action.RETRY:          # guide: look again (table: this scene), same trial
                return release + (self._pre_pick(view.detections or []) if table else self._park())
            if d.action == Action.STOP:
                return self._finish("stop", "grasp failed with no retries left", release)
            return self._place()
        return []

    def _pre_pick(self, dets: list[Detection]) -> list[Row]:
        retry = self.trial_id is not None          # a guide retry of the pick in progress
        if not retry:
            if self.job_i >= len(self.jobs) or self.picks >= self.max_picks:
                return self._finish("done", f"{self.picks} picks")
            self.job_i += 1
            self.trial_id, self.next_id = self.next_id, self.next_id + 1
        job = self.jobs[self.job_i - 1]
        tid = self.trial_id
        aim = self._aim(job, dets)
        self.comparator.start_trial(tid, job.cls, aim)
        if not retry:
            self.events.append(Event("trial", tid, job.cls, aim))
        d = self.comparator.on_pre_pick(dets, trial_id=tid)
        self._log(d, tid)
        if d.action == Action.STOP:
            return self._finish("stop", "part not found with no retries left")
        if d.action == Action.RETRY:              # guide: nothing to aim at; look again
            return self._park()
        grasp = plan_grasp(aim, self.cfg.objects[job.cls], self.cfg.raw["gripper"]["max_opening_mm"],
                           self.cfg.raw["gripper"]["clearance_mm"])
        if not grasp.fits:
            self.events.append(Event("skip", tid, job.cls, aim, note=grasp.reason))
            self.trial_id = None
            return self._pre_pick(dets)
        self.picks += 1
        return self._pick(grasp.pose)

    def _aim(self, job: Job, dets: list[Detection]) -> Pose2D:
        """Blind modes reach for the nominal point. Guide reaches for the seen part of this class that is
        closest to it and not already in a bin."""
        if self.mode != "guide":
            return job.nominal
        loose = [d for d in dets if d.cls == job.cls and d.confidence >= self.cfg.thresholds.min_confidence
                 and not any(b.contains(d.pose) for b in self.cfg.bins.values())]
        if not loose:
            return job.nominal
        return min(loose, key=lambda d: d.pose.distance_to(job.nominal)).pose

    # ------------------------------------------------------------------ motions
    def _go(self, x: float, y: float, z: float, r: float, gripper: float) -> list:
        goal = self.spec.down(x, y, z, r, gripper)
        check_row(self.spec, goal)
        rows = interpolate(self.spec, self.last, goal)
        for row in rows:
            check_row(self.spec, row)
        self.last = rows[-1]
        return rows + [_Arrive(list(self.last))]

    def _hold(self, n: int, gripper: float | None = None) -> list:
        if gripper is None:
            return [list(self.last) for _ in range(n)]
        self.last = self.last[:6] + [gripper]
        return [list(self.last) for _ in range(n)] + [_Arrive(list(self.last))]

    def _up_to(self, z: float) -> list[Row]:
        """Straight up first, so a carried part never sweeps through a bin wall."""
        if self.last[2] >= z:
            return []
        return self._go(self.last[0], self.last[1], z, self.last[5], self.last[6])

    def _to_park(self) -> list:
        s = self.spec
        x, y, z = s.park
        return (self._up_to(s.z_above_surface(s.transit_mm)) + self._go(x, y, z, 0.0, self.last[6])
                + self._hold(s.settle_rows))

    def _park(self) -> list:
        rows = self._to_park()
        self.phase, self._needs = _Phase.SCAN, Need.SCENE
        return rows

    def _open_here(self) -> list:
        return self._hold(self.spec.grip_rows, 0.0)

    def _pick(self, at: Pose2D) -> list[Row]:
        s = self.spec
        r = s.r_for_close_yaw(at.yaw_deg)
        hover, grip = s.z_above_surface(s.hover_mm), s.z_above_surface(s.tip_clearance_mm)
        rows = (self._up_to(s.z_above_surface(s.transit_mm))
                + self._go(at.x_mm, at.y_mm, s.z_above_surface(s.transit_mm), r, 0.0)
                + self._go(at.x_mm, at.y_mm, hover, r, 0.0)
                + self._go(at.x_mm, at.y_mm, grip, r, 0.0)
                + self._hold(s.settle_rows)
                + self._hold(s.grip_rows, 1.0)
                + self._go(at.x_mm, at.y_mm, s.z_above_surface(s.transit_mm), r, 1.0)
                + self._hold(s.settle_rows))
        self._grasp_at = at
        if self.grasp_by == "table":              # park out of the table camera's view and look
            rows += self._to_park()
            self.phase, self._needs = _Phase.GRASPED, Need.SCENE
        else:
            self.phase, self._needs = _Phase.GRASPED, Need.GRASP
        return rows

    def _place(self) -> list[Row]:
        s = self.spec
        b = self.cfg.bins[self.comparator.trials[self.trial_id].target_bin].center
        release = s.z_above_surface(s.bin_height_mm + s.release_above_rim_mm)
        rows = (self._go(b.x_mm, b.y_mm, s.z_above_surface(s.transit_mm), self.last[5], 1.0)
                + self._go(b.x_mm, b.y_mm, release, self.last[5], 1.0)
                + self._hold(s.settle_rows)
                + self._hold(s.grip_rows, 0.0))
        self.placed_id, self.trial_id = self.trial_id, None
        return rows + self._park()

    def _release_here(self) -> list[Row]:
        """Set whatever may be held back down where it was picked, open, and rise again. Opening at lift
        height would drop a held part about 10 cm onto the table and scatter it (seen in the twin)."""
        s = self.spec
        x, y, _, _, _, r, g = self.last
        return (self._go(x, y, s.z_above_surface(s.tip_clearance_mm), r, g)
                + self._hold(s.settle_rows)
                + self._hold(s.grip_rows, 0.0)
                + self._up_to(s.z_above_surface(s.hover_mm)))

    # ------------------------------------------------------------------ bookkeeping
    def _log(self, d: Decision, tid: int) -> None:
        self.events.append(Event("decision", tid, self.comparator.trials[tid].cls, decision=d))

    def _finish(self, kind: str, note: str, rows: list[Row] | None = None) -> list[Row]:
        self.events.append(Event(kind, self.trial_id or 0, note=note))
        self.phase, self._needs = _Phase.DONE, Need.NOTHING
        rows = rows or []
        try:
            rows += self._park()
        except OutsideWorkspace:
            pass
        self.phase, self._needs = _Phase.DONE, Need.NOTHING
        return rows
