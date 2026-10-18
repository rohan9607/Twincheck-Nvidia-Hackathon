"""Config-based expectations: what should be true at each check point.

This is the always-available baseline answer key. When the Isaac Sim shadow
twin is running it publishes physics-based expectations on the same topic with
source="physics"; the comparator then mentions the twin's prediction.
"""
import rclpy
from rclpy.node import Node

from twincheck_core import load_config
from twincheck_msgs.msg import DetectedObject, ExpectedState, GraspState, RobotEvent

EVENT_TO_CHECK = {RobotEvent.PRE_PICK: "pre_pick", RobotEvent.GRASP_CLOSED: "post_grasp",
                  RobotEvent.PLACE_DONE: "post_place"}


class ExpectationNode(Node):
    def __init__(self):
        super().__init__("expectation_node")
        self.declare_parameter("config", "")
        self.cfg = load_config(self.get_parameter("config").value or None)
        self.pub = self.create_publisher(ExpectedState, "/twin/expected", 10)
        self.create_subscription(RobotEvent, "/robot/event", self.on_event, 10)

    def on_event(self, ev: RobotEvent):
        cp = EVENT_TO_CHECK.get(ev.event)
        if cp is None:
            return
        cls = ev.object_cls or "cube"
        bin_name = ev.target_bin or self.cfg.objects[cls].target_bin
        m = ExpectedState()
        m.header = ev.header
        m.trial_id = ev.trial_id
        m.check_point = cp
        m.object_expected = True
        m.expected_grasp = GraspState.HOLDING
        m.target_bin = bin_name
        m.source = "config"
        if cp == "pre_pick" and ev.pick_point in self.cfg.pick_points:
            p = self.cfg.pick_points[ev.pick_point]
        else:
            p = self.cfg.bins[bin_name].center
        m.expected_object = DetectedObject(id="expected", cls=cls, x_mm=p.x_mm, y_mm=p.y_mm, yaw_deg=p.yaw_deg,
                                           confidence=1.0, camera="config")
        self.pub.publish(m)


def main():
    rclpy.init()
    node = ExpectationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
