"""Comparator node: the heart of TwinCheck.

On each robot event it runs the matching check:
  PRE_PICK     -> overhead objects at that moment vs the pick point
  GRASP_CLOSED -> wrist grasp state for that trial vs "holding"
  PLACE_DONE   -> overhead objects after the scene settles vs the target bin
It saves the observed (and, if the shadow twin rendered one, expected) image,
publishes /check/result, and decides alert / retry / stop via twincheck_core.
"""
import math
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from twincheck_core import Comparator, Mode, StampBuffer, load_config
from twincheck_core.checks import GraspState
from twincheck_msgs.msg import CheckResult, ExpectedState, GraspState as GraspStateMsg, ObjectArray, RobotEvent

from .conv import detections, grasp_state, stamp_ns


class ComparatorNode(Node):
    def __init__(self):
        super().__init__("comparator_node")
        self.declare_parameter("config", "")
        self.declare_parameter("mode", "watch")
        self.declare_parameter("place_settle_ms", 600.0)
        self.declare_parameter("grasp_timeout_ms", 3000.0)
        self.declare_parameter("image_dir", "runs/images")
        self.declare_parameter("default_object", "cube")
        self.declare_parameter("default_pick_point", "slot_1")

        self.cfg = load_config(self.get_parameter("config").value or None)
        self.mode = Mode(self.get_parameter("mode").value)
        self.comp = Comparator(self.cfg, self.mode)
        self.tol_ms = self.cfg.thresholds.sync_tolerance_ms
        self.bridge = CvBridge()
        self.img_dir = Path(self.get_parameter("image_dir").value)
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.objects: StampBuffer = StampBuffer(120)
        self.overhead_imgs: StampBuffer = StampBuffer(60)
        self.wrist_imgs: StampBuffer = StampBuffer(60)
        self.expected: dict[tuple[int, str], ExpectedState] = {}
        self.waiting: list[tuple[str, int, RobotEvent]] = []   # (kind, due_ns, event)
        self.grasp_msgs: list[GraspStateMsg] = []

        self.create_subscription(ObjectArray, "/perception/objects", lambda m: self.objects.push(stamp_ns(m.header), m), 20)
        self.create_subscription(GraspStateMsg, "/perception/grasp_state", lambda m: self.grasp_msgs.append(m), 10)
        self.create_subscription(ExpectedState, "/twin/expected", self.on_expected, 10)
        self.create_subscription(RobotEvent, "/robot/event", self.on_event, 10)
        cams = self.cfg.raw["cameras"]
        self.create_subscription(Image, cams["overhead_a"]["topic"], lambda m: self.overhead_imgs.push(stamp_ns(m.header), m), 2)
        self.create_subscription(Image, cams["wrist"]["topic"], lambda m: self.wrist_imgs.push(stamp_ns(m.header), m), 2)
        self.pub = self.create_publisher(CheckResult, "/check/result", 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f"comparator in {self.mode.value} mode")

    # ---------------------------------------------------------------- inputs
    def on_expected(self, m: ExpectedState):
        self.expected[(m.trial_id, m.check_point)] = m

    def on_event(self, ev: RobotEvent):
        now = stamp_ns(ev.header)
        if ev.event == RobotEvent.PRE_PICK:
            cls = ev.object_cls or self.get_parameter("default_object").value
            pp_name = ev.pick_point or self.get_parameter("default_pick_point").value
            self.comp.start_trial(ev.trial_id, cls, self.cfg.pick_points[pp_name], ev.target_bin or None)
            self.waiting.append(("pre_pick", now, ev))
        elif ev.event == RobotEvent.GRASP_CLOSED:
            self.waiting.append(("post_grasp", now + int(self.get_parameter("grasp_timeout_ms").value * 1e6), ev))
        elif ev.event == RobotEvent.PLACE_DONE:
            self.waiting.append(("post_place", now + int(self.get_parameter("place_settle_ms").value * 1e6), ev))

    # ---------------------------------------------------------------- checks
    def tick(self):
        now_ns = self.get_clock().now().nanoseconds
        still = []
        for kind, due_ns, ev in self.waiting:
            done = False
            if kind == "pre_pick":
                done = self.run_pre_pick(ev)
            elif kind == "post_grasp":
                done = self.run_post_grasp(ev, timed_out=now_ns >= due_ns)
            elif kind == "post_place":
                done = self.run_post_place(ev, due_ns) if now_ns >= due_ns else False
            if not done:
                still.append((kind, due_ns, ev))
        self.waiting = still

    def run_pre_pick(self, ev) -> bool:
        m = self.objects.nearest(stamp_ns(ev.header), self.tol_ms)
        if m is None:
            return False
        decision = self.comp.on_pre_pick(detections(m.item))
        self.publish(ev, decision, m.gap_ms, self.overhead_imgs, stamp_ns(ev.header))
        return True

    def run_post_grasp(self, ev, timed_out: bool) -> bool:
        t_ev = stamp_ns(ev.header)
        mine = [g for g in self.grasp_msgs if f"trial {ev.trial_id}:" in g.detail and stamp_ns(g.header) >= t_ev]
        if mine:
            g = mine[0]
            self.grasp_msgs.remove(g)
            decision = self.comp.on_grasp_closed(grasp_state(g))
            self.publish(ev, decision, (stamp_ns(g.header) - t_ev) / 1e6, self.wrist_imgs, stamp_ns(g.header),
                         extra=f" [{g.method}: {g.detail}]")
            return True
        if timed_out:  # no wrist answer in time: count it as unknown (a failure), never as success
            decision = self.comp.on_grasp_closed(GraspState.UNKNOWN)
            self.publish(ev, decision, math.nan, self.wrist_imgs, t_ev, extra=" [no wrist result in time]")
            return True
        return False

    def run_post_place(self, ev, due_ns) -> bool:
        m = self.objects.latest_after(due_ns)
        if m is None:
            return False
        decision = self.comp.on_place_done(detections(m.item))
        self.publish(ev, decision, m.gap_ms, self.overhead_imgs, m.stamp_ns)
        return True

    # ---------------------------------------------------------------- output
    def save_image(self, buf: StampBuffer, stamp: int, name: str) -> str:
        m = buf.nearest(stamp, 500)
        if m is None:
            return ""
        path = self.img_dir / name
        cv2.imwrite(str(path), self.bridge.imgmsg_to_cv2(m.item, "bgr8"))
        return str(path)

    def publish(self, ev, decision, gap_ms, img_buf, img_stamp, extra=""):
        out = decision.outcome
        cp = out.check_point.value
        exp = self.expected.get((ev.trial_id, cp))
        reason = out.reason + extra
        if exp is not None and exp.source != "config" and not exp.object_expected and cp == "pre_pick":
            reason += " | the twin also predicts this grasp misses"
        msg = CheckResult()
        msg.header.stamp = ev.header.stamp
        msg.header.frame_id = "table"
        msg.trial_id = ev.trial_id
        msg.check_point = cp
        msg.passed = out.passed
        msg.position_error_mm = out.position_error_mm if out.position_error_mm is not None else math.nan
        msg.yaw_error_deg = out.yaw_error_deg if out.yaw_error_deg is not None else math.nan
        msg.reason = reason
        msg.action = decision.action.value
        msg.urgent = decision.urgent
        msg.sync_gap_ms = float(gap_ms)
        msg.expected_source = exp.source if exp else "config"
        msg.expected_img = exp.expected_img if exp else ""
        msg.observed_img = self.save_image(img_buf, img_stamp, f"trial{ev.trial_id:04d}_{cp}.jpg")
        self.pub.publish(msg)
        level = self.get_logger().info if out.passed else self.get_logger().warn
        level(f"trial {ev.trial_id} {cp}: {'PASS' if out.passed else 'FAIL'} - {reason} -> {decision.action.value}")


def main():
    rclpy.init()
    node = ComparatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
