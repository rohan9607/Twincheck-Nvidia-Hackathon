"""Alert node: turns failed checks into something a person notices.

Prints a clear banner, publishes /alerts (plain text for the dashboard and any
other consumer) and, in Guide mode, forwards RETRY / STOP to /guide/command.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from twincheck_msgs.msg import CheckResult


class AlertNode(Node):
    def __init__(self):
        super().__init__("alert_node")
        self.declare_parameter("bell", True)
        self.alert_pub = self.create_publisher(String, "/alerts", 10)
        self.guide_pub = self.create_publisher(String, "/guide/command", 10)
        self.create_subscription(CheckResult, "/check/result", self.on_result, 10)

    def on_result(self, r: CheckResult):
        if r.passed:
            return
        level = "URGENT" if r.urgent else "ALERT"
        text = f"[{level}] trial {r.trial_id} {r.check_point}: {r.reason} -> {r.action.upper()}"
        if r.observed_img:
            text += f" (image: {r.observed_img})"
        bell = "\a" if self.get_parameter("bell").value else ""
        print(f"{bell}\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)
        self.alert_pub.publish(String(data=text))
        if r.action in ("retry", "stop"):
            self.guide_pub.publish(String(data=f"{r.action}:{r.trial_id}:{r.check_point}"))


def main():
    rclpy.init()
    node = AlertNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
