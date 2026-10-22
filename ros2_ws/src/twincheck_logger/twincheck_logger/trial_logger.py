"""Trial logger: every event, check and resync goes into Postgres.

Trials are created at PRE_PICK with the environment, mode and run-sheet info.
Ground truth (did the pick actually work?) comes from the twin runner
automatically, or from the Recorder on the real arm via scripts/ground_truth.py.
"""
import math
import os

import psycopg
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from twincheck_msgs.msg import CheckResult, ResyncReport, RobotEvent


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


class TrialLogger(Node):
    def __init__(self):
        super().__init__("trial_logger")
        self.declare_parameter("dsn", os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck"))
        self.declare_parameter("env", "twin")               # twin | real
        self.declare_parameter("mode", "watch")             # baseline | watch | guide
        self.declare_parameter("ladder_rung", "fixed_pose")
        self.declare_parameter("model_version", "")
        self.declare_parameter("session_tag", "")
        self.db = psycopg.connect(self.get_parameter("dsn").value, autocommit=True)
        self.trial_db_id: dict[int, int] = {}   # ROS trial_id -> trials.trial_id
        self.create_subscription(RobotEvent, "/robot/event", self.on_event, 20)
        self.create_subscription(CheckResult, "/check/result", self.on_check, 20)
        self.create_subscription(ResyncReport, "/twin/resync_report", self.on_resync, 20)
        self.get_logger().info("logging to Postgres")

    def p(self, name):
        return self.get_parameter(name).value

    def on_event(self, ev: RobotEvent):
        if ev.event != RobotEvent.PRE_PICK or ev.trial_id in self.trial_db_id:
            return
        row = self.db.execute(
            """INSERT INTO trials (env, mode, ladder_rung, object_class, model_version, threshold_mm, session_tag,
                                   ros_trial_id, pick_point)
               VALUES (%s,%s,%s,%s,%s,NULL,%s,%s,%s) RETURNING trial_id""",
            (self.p("env"), self.p("mode"), self.p("ladder_rung"), ev.object_cls or "unknown",
             self.p("model_version"), self.p("session_tag"), ev.trial_id, ev.pick_point)).fetchone()
        self.trial_db_id[ev.trial_id] = row[0]

    def on_check(self, r: CheckResult):
        tid = self.trial_db_id.get(r.trial_id)
        if tid is None:
            self.get_logger().warn(f"check for unknown trial {r.trial_id}; skipped")
            return
        self.db.execute(
            """INSERT INTO checks (trial_id, check_point, passed, position_err_mm, yaw_err_deg, reason, action, urgent,
                                   expected_img, observed_img, sync_gap_ms, expected_source)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (tid, r.check_point, r.passed, nan_to_none(r.position_error_mm), nan_to_none(r.yaw_error_deg), r.reason,
             r.action, r.urgent, r.expected_img or None, r.observed_img or None, nan_to_none(r.sync_gap_ms),
             r.expected_source))
        if r.action == "retry":
            self.db.execute("UPDATE trials SET retried = true WHERE trial_id = %s", (tid,))

    def on_resync(self, rep: ResyncReport):
        tid = self.trial_db_id.get(rep.trial_id)
        shifts = [s for s in rep.shift_mm if not math.isnan(s)]
        if tid is None or not shifts:
            return
        self.db.execute("UPDATE trials SET resync_max_shift_mm = %s WHERE trial_id = %s", (max(shifts), tid))


def main():
    rclpy.init()
    node = TrialLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C or launch shutdown
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
