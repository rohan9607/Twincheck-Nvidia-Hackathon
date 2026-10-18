"""Per-trial state machine that turns robot events + observations into decisions.

Event order for one pick:  PRE_PICK -> GRASP_CLOSED -> PLACE_DONE
At each event the matching check runs. In Watch mode a failure is an ALERT.
In Guide mode a failure is RETRY (up to max_retries), then STOP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .checks import (CheckOutcome, CheckPoint, Detection, GraspState, check_post_grasp, check_post_place,
                     check_pre_pick)
from .config import Config
from .geometry import Pose2D


class Mode(str, Enum):
    WATCH = "watch"
    GUIDE = "guide"


class Action(str, Enum):
    NONE = "none"
    ALERT = "alert"
    RETRY = "retry"
    STOP = "stop"


@dataclass
class Decision:
    trial_id: int
    outcome: CheckOutcome
    action: Action
    urgent: bool = False


@dataclass
class TrialState:
    trial_id: int
    cls: str
    pick_point: Pose2D
    target_bin: str
    failures_in_row: int = 0
    retries: int = 0
    outcomes: list[CheckOutcome] = field(default_factory=list)


class Comparator:
    def __init__(self, cfg: Config, mode: Mode = Mode.WATCH):
        self.cfg = cfg
        self.mode = mode
        self.trial: TrialState | None = None

    def start_trial(self, trial_id: int, cls: str, pick_point: Pose2D, target_bin: str | None = None) -> None:
        bin_name = target_bin or self.cfg.objects[cls].target_bin
        self.trial = TrialState(trial_id, cls, pick_point, bin_name)

    def _require_trial(self) -> TrialState:
        if self.trial is None:
            raise RuntimeError("no active trial; call start_trial first")
        return self.trial

    def on_pre_pick(self, detections: list[Detection]) -> Decision:
        t = self._require_trial()
        out = check_pre_pick(t.pick_point, t.cls, detections, self.cfg.thresholds, self.cfg.symmetry(t.cls))
        return self._decide(out)

    def on_grasp_closed(self, grasp: GraspState) -> Decision:
        return self._decide(check_post_grasp(grasp))

    def on_place_done(self, detections: list[Detection]) -> Decision:
        t = self._require_trial()
        out = check_post_place(t.cls, self.cfg.bins[t.target_bin], detections, self.cfg.thresholds,
                               self.cfg.bins, t.pick_point)
        return self._decide(out)

    def _decide(self, out: CheckOutcome) -> Decision:
        t = self._require_trial()
        t.outcomes.append(out)
        if out.passed:
            t.failures_in_row = 0
            return Decision(t.trial_id, out, Action.NONE)

        t.failures_in_row += 1
        urgent = t.failures_in_row >= 2
        if self.mode == Mode.WATCH:
            return Decision(t.trial_id, out, Action.ALERT, urgent)

        # Guide mode: a pre-pick miss is corrected by re-planning to the observed pose.
        # A failed grasp is retried once from a fresh detection. A bad placement, or
        # any failure after the retry budget, stops the arm for the operator.
        if out.check_point == CheckPoint.POST_PLACE:
            return Decision(t.trial_id, out, Action.STOP, True)
        if t.retries < self.cfg.thresholds.max_retries:
            t.retries += 1
            return Decision(t.trial_id, out, Action.RETRY, urgent)
        return Decision(t.trial_id, out, Action.STOP, True)
