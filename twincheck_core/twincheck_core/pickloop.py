"""One rollout on the organizers' cell: pick-and-place with TwinCheck's three checks, as rows for act(obs).

The policy calls step(view) every time the cell asks for actions. step returns rows to play; when it
needs to look (the table camera before a pick and after a place, the wrist camera after a grasp) it parks
the arm, returns those rows, and sets `needs`. The policy then fills that part of the next view.

    SCAN (park, table camera) -> pre-pick check -> PICK (hover, descend, close, lift)
      -> wrist camera: post-grasp check -> PLACE (carry high, release over the bin)
      -> SCAN: post-place check, and the same scene is the next job's pre-pick scene

Modes (the same comparison the twin runs):
  baseline  blind fixed program: reaches for the configured pick points; checks are logged only
  watch     the same blind motion; failed checks are alerts
  guide     aims at the part the camera sees; a failed grasp retries once from a fresh scan;
            a bad place, or a failure with no retries left, stops the rollout (the arm parks and holds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .cell import CellSpec, OutsideWorkspace, Row, check_row, interpolate
from .checks import Detection, GraspState
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
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self.spec = CellSpec.from_config(self.cfg)
        # baseline runs the checks like watch; only what we do with a failure differs
        self.comparator = Comparator(self.cfg, Mode.GUIDE if self.mode == "guide" else Mode.WATCH)
        self.phase = _Phase.START
        self.needs = Need.NOTHING
        self.last: Row | None = None
        self.job_i = 0
        self.picks = 0
        self.next_id = self.first_trial_id
        self.trial_id: int | None = None   # the pick in progress
        self.placed_id: int | None = None  # placed, waiting for its post-place check

    # ------------------------------------------------------------------ public
    @property
    def done(self) -> bool:
        return self.phase == _Phase.DONE

    def step(self, view: View) -> list[Row]:
        """Rows to play next (never empty: when done, hold where the arm is)."""
        if self.last is None:
            g = 1.0 if view.gripper >= 0.5 else 0.0
            self.last = (list(view.tcp[:6]) + [g]) if view.tcp else self.spec.down(*self.spec.park, 0.0, g)
        try:
            rows = self._advance(view)
        except OutsideWorkspace as e:
            rows = self._finish("stop", f"refused a target outside our workspace: {e}")
        return rows or [list(self.last)]

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
            d = self.comparator.on_grasp_closed(view.grasp or GraspState.UNKNOWN, trial_id=self.trial_id)
            self._log(d, self.trial_id)
            if d.action == Action.RETRY:          # guide: drop whatever is held, look again, same trial
                return self._release_here() + self._park()
            if d.action == Action.STOP:
                return self._finish("stop", "grasp failed with no retries left", self._release_here())
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
    def _go(self, x: float, y: float, z: float, r: float, gripper: float) -> list[Row]:
        goal = self.spec.down(x, y, z, r, gripper)
        check_row(self.spec, goal)
        rows = interpolate(self.spec, self.last, goal)
        for row in rows:
            check_row(self.spec, row)
        self.last = rows[-1]
        return rows

    def _hold(self, n: int, gripper: float | None = None) -> list[Row]:
        if gripper is not None:
            self.last = self.last[:6] + [gripper]
        return [list(self.last) for _ in range(n)]

    def _up_to(self, z: float) -> list[Row]:
        """Straight up first, so a carried part never sweeps through a bin wall."""
        if self.last[2] >= z:
            return []
        return self._go(self.last[0], self.last[1], z, self.last[5], self.last[6])

    def _park(self) -> list[Row]:
        s = self.spec
        x, y, z = s.park
        rows = self._up_to(s.z_above_surface(s.transit_mm)) + self._go(x, y, z, 0.0, self.last[6])
        self.phase, self.needs = _Phase.SCAN, Need.SCENE
        return rows + self._hold(s.settle_rows)

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
        self.phase, self.needs = _Phase.GRASPED, Need.GRASP
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
        return self._hold(self.spec.grip_rows, 0.0)

    # ------------------------------------------------------------------ bookkeeping
    def _log(self, d: Decision, tid: int) -> None:
        self.events.append(Event("decision", tid, self.comparator.trials[tid].cls, decision=d))

    def _finish(self, kind: str, note: str, rows: list[Row] | None = None) -> list[Row]:
        self.events.append(Event(kind, self.trial_id or 0, note=note))
        self.phase, self.needs = _Phase.DONE, Need.NOTHING
        rows = rows or []
        try:
            rows += self._park()
        except OutsideWorkspace:
            pass
        self.phase, self.needs = _Phase.DONE, Need.NOTHING
        return rows
