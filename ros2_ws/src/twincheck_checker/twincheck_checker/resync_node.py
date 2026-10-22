"""Resync node: after every post-place check, snap the twin to reality.

Reads the twin's current object poses (/twin/objects, published by the shadow
twin), matches them to what the cameras see now, and sends the corrected poses
to the twin (/twin/set_poses). Publishes a ResyncReport so the logger records
how far things had to move (drift).
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from twincheck_core import StampBuffer, plan_resync
from twincheck_msgs.msg import CheckResult, DetectedObject, ObjectArray, ResyncReport

from .conv import detections, stamp_ns


class ResyncNode(Node):
    def __init__(self):
        super().__init__("resync_node")
        self.declare_parameter("match_radius_mm", 80.0)
        self.declare_parameter("settle_ms", 300.0)
        self.objects: StampBuffer = StampBuffer(60)
        self.twin: ObjectArray | None = None
        self.pending: list[tuple[int, int]] = []
        self.create_subscription(ObjectArray, "/perception/objects", lambda m: self.objects.push(stamp_ns(m.header), m), 20)
        self.create_subscription(ObjectArray, "/twin/objects", self.on_twin, 10)
        self.create_subscription(CheckResult, "/check/result", self.on_result, 10)
        self.set_pub = self.create_publisher(ObjectArray, "/twin/set_poses", 10)
        self.report_pub = self.create_publisher(ResyncReport, "/twin/resync_report", 10)
        self.create_timer(0.1, self.tick)

    def on_twin(self, m: ObjectArray):
        self.twin = m

    def on_result(self, r: CheckResult):
        if r.check_point == "post_place":
            due = stamp_ns(r.header) + int(self.get_parameter("settle_ms").value * 1e6)
            self.pending.append((r.trial_id, due))

    def tick(self):
        still = []
        for trial_id, due in self.pending:
            m = self.objects.latest_after(due)
            if m is None:
                still.append((trial_id, due))
                continue
            self.resync(trial_id, m.item)
        self.pending = still

    def resync(self, trial_id: int, observed: ObjectArray):
        twin_objs = {}
        if self.twin is not None:
            twin_objs = {o.id: (o.cls, detections(ObjectArray(objects=[o]))[0].pose) for o in self.twin.objects}
        updates, missing = plan_resync(twin_objs, detections(observed), self.get_parameter("match_radius_mm").value)

        out = ObjectArray()
        out.header = observed.header
        for u in updates:
            out.objects.append(DetectedObject(id=u.twin_id, cls=u.cls, x_mm=u.new.x_mm, y_mm=u.new.y_mm,
                                              yaw_deg=u.new.yaw_deg, confidence=1.0, camera="resync"))
        self.set_pub.publish(out)

        rep = ResyncReport()
        rep.header = observed.header
        rep.trial_id = trial_id
        rep.twin_ids = [u.twin_id for u in updates]
        rep.classes = [u.cls for u in updates]
        rep.shift_mm = [u.shift_mm if u.shift_mm is not None else math.nan for u in updates]
        rep.missing_twin_ids = missing
        self.report_pub.publish(rep)
        moved = [s for s in rep.shift_mm if not math.isnan(s)]
        if moved:
            self.get_logger().info(f"trial {trial_id}: resynced {len(updates)} objects, max shift {max(moved):.1f} mm")


def main():
    rclpy.init()
    node = ResyncNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C or launch shutdown
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
