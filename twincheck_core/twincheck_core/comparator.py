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
    before: list[Detection] | None = None   # scene at pre-pick: tells this pick's part from parts already in bins
    outcomes: list[CheckOutcome] = field(default_factory=list)


class Comparator:
    """Trials are kept by id: a check that finishes late (post-place waits for the scene to
    settle) must be judged against ITS trial, even if the next pick has already started."""

    KEEP_TRIALS = 20

    def __init__(self, cfg: Config, mode: Mode = Mode.WATCH):
        self.cfg = cfg
        self.mode = mode
        self.trials: dict[int, TrialState] = {}
        self.trial: TrialState | None = None   # the latest started trial

    def start_trial(self, trial_id: int, cls: str, pick_point: Pose2D, target_bin: str | None = None) -> None:
        """A PRE_PICK for a trial already running is a Guide retry: aim at the new pick point
        but keep the retry and failure counts, or the retry budget would never run out."""
        if trial_id in self.trials:
            self.trials[trial_id].pick_point = pick_point
            self.trial = self.trials[trial_id]
            return
        bin_name = target_bin or self.cfg.objects[cls].target_bin
        self.trial = self.trials[trial_id] = TrialState(trial_id, cls, pick_point, bin_name)
        while len(self.trials) > self.KEEP_TRIALS:
            del self.trials[min(self.trials)]

    def is_active(self, trial_id: int) -> bool:
        """True if this trial's PRE_PICK started it. Grasp/place events for any other trial
        (a missed pre-pick, a key pressed out of order) must be skipped, not checked against the wrong pick."""
        return trial_id in self.trials

    def _require_trial(self, trial_id: int | None) -> TrialState:
        t = self.trial if trial_id is None else self.trials.get(trial_id)
        if t is None:
            raise RuntimeError(f"no trial {trial_id if trial_id is not None else ''} started; call start_trial first")
        return t

    def on_pre_pick(self, detections: list[Detection], scene: list[Detection] | None = None,
                    trial_id: int | None = None) -> Decision:
        """scene: the bins' contents to compare the place against. Pass the frame with the most
        detections around this moment: a part missed here would let a lost part count as placed."""
        t = self._require_trial(trial_id)
        t.before = list(scene if scene is not None else detections)
        out = check_pre_pick(t.pick_point, t.cls, detections, self.cfg.thresholds, self.cfg.symmetry(t.cls))
        return self._decide(out, t)

    def on_grasp_closed(self, grasp: GraspState, trial_id: int | None = None) -> Decision:
        return self._decide(check_post_grasp(grasp), self._require_trial(trial_id))

    def on_place_done(self, detections: list[Detection], trial_id: int | None = None) -> Decision:
        t = self._require_trial(trial_id)
        out = check_post_place(t.cls, self.cfg.bins[t.target_bin], detections, self.cfg.thresholds,
                               self.cfg.bins, t.pick_point, t.before)
        return self._decide(out, t)

    def _decide(self, out: CheckOutcome, t: TrialState) -> Decision:
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
