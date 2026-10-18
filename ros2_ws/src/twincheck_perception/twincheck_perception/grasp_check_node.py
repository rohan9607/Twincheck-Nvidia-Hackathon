"""Wrist grasp-check node.

Keeps a short buffer of wrist frames. When the robot reports GRASP_CLOSED, it
takes the frame closest to (just after) that moment and decides whether the
object is held. The gripper has no part-present sensor, so this is the only
evidence of a successful grasp.
"""
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from twincheck_core import StampBuffer, load_config
from twincheck_msgs.msg import GraspState as GraspStateMsg
from twincheck_msgs.msg import RobotEvent

from .infer import WristChecker

STATE_TO_MSG = {"holding": GraspStateMsg.HOLDING, "empty": GraspStateMsg.EMPTY,
                "misaligned": GraspStateMsg.MISALIGNED, "unknown": GraspStateMsg.UNKNOWN}


class GraspCheckNode(Node):
    def __init__(self):
        super().__init__("grasp_check_node")
        self.declare_parameter("config", "")
        self.declare_parameter("method", "reference_diff")      # reference_diff | classifier | cosmos
        self.declare_parameter("roi", [0, 0, 0, 0])               # x, y, w, h of the finger gap; zeros = full image
        self.declare_parameter("diff_threshold", 18.0)
        self.declare_parameter("settle_ms", 150.0)                # wait for the fingers to finish closing

        cfg = load_config(self.get_parameter("config").value or None)
        roi = list(self.get_parameter("roi").value)
        cos = cfg.raw["models"]["cosmos"]
        self.checker_kwargs = dict(
            reference_path=cfg.raw["models"]["wrist_reference"],
            roi=tuple(roi) if any(roi) else None,
            diff_threshold=self.get_parameter("diff_threshold").value,
            classifier_path=cfg.raw["models"]["wrist_classifier"],
            cosmos_base_url=cos["base_url"], cosmos_model=cos["model"], cosmos_api_key_env=cos["api_key_env"])
        self.method = self.get_parameter("method").value
        self.checkers: dict[str, WristChecker] = {}
        self.tolerance_ms = cfg.thresholds.sync_tolerance_ms
        self.bridge = CvBridge()
        self.frames: StampBuffer = StampBuffer(maxlen=90)

        self.create_subscription(Image, cfg.raw["cameras"]["wrist"]["topic"], self.on_image, 5)
        self.create_subscription(RobotEvent, "/robot/event", self.on_event, 10)
        self.pub = self.create_publisher(GraspStateMsg, "/perception/grasp_state", 10)
        self.pending: list[tuple[int, object]] = []
        self.create_timer(0.05, self.process_pending)

    def checker_for(self, obj: str) -> WristChecker:
        key = f"{self.method}:{obj}"
        if key not in self.checkers:
            self.checkers[key] = WristChecker(self.method, object_name=obj or "object", **self.checker_kwargs)
        return self.checkers[key]

    def on_image(self, msg: Image):
        s = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.frames.push(s, msg)

    def on_event(self, ev: RobotEvent):
        if ev.event != RobotEvent.GRASP_CLOSED:
            return
        s = ev.header.stamp.sec * 1_000_000_000 + ev.header.stamp.nanosec
        self.pending.append((s + int(self.get_parameter("settle_ms").value * 1e6), ev))

    def process_pending(self):
        still = []
        for target_ns, ev in self.pending:
            m = self.frames.latest_after(target_ns)
            if m is None:
                still.append((target_ns, ev))
                continue
            state, conf, detail = self.checker_for(ev.object_cls).check(self.bridge.imgmsg_to_cv2(m.item, "bgr8"))
            out = GraspStateMsg()
            out.header = m.item.header
            out.state = STATE_TO_MSG[state.value]
            out.confidence = conf
            out.method = self.method
            out.detail = f"trial {ev.trial_id}: {detail}"
            self.pub.publish(out)
        self.pending = still


def main():
    rclpy.init()
    node = GraspCheckNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
