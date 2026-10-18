"""Robot event source: tells the checker WHEN each check should run.

Three sources, chosen by the `source` parameter:
  joint_states - read-only: watch /joint_states (from the FANUC driver in read-only
                 use, or from Isaac Sim) and emit an event when the arm stops at a
                 known waypoint from config `waypoints`
  manual       - an operator presses keys in `twincheck_trigger` (fallback if we
                 cannot read the controller at all)
  guide        - Guide mode publishes its own events; this node only numbers trials

The run sheet (YAML list) says which object each trial is about, so trials line
up with the trial log.
"""
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from twincheck_core import load_config
from twincheck_msgs.msg import RobotEvent


class RobotStateNode(Node):
    def __init__(self):
        super().__init__("robot_state_node")
        self.declare_parameter("config", "")
        self.declare_parameter("source", "joint_states")       # joint_states | manual | guide
        self.declare_parameter("run_sheet", "")                # YAML list of {object_cls, pick_point, target_bin}
        self.declare_parameter("first_trial_id", 1)

        cfg = load_config(self.get_parameter("config").value or None)
        wp = cfg.raw.get("waypoints", {})
        self.tol = float(wp.get("tolerance_rad", 0.02))
        self.still = float(wp.get("still_velocity_rad_s", 0.01))
        self.waypoints = {ev: {name: list(q) for name, q in (wp.get(ev) or {}).items()}
                          for ev in ("pre_pick", "grasp_closed", "place_done")}
        self.trial_id = self.get_parameter("first_trial_id").value - 1
        self.run_sheet = []
        rs = self.get_parameter("run_sheet").value
        if rs:
            with open(rs) as f:
                self.run_sheet = yaml.safe_load(f) or []
        self.last_event = None

        self.pub = self.create_publisher(RobotEvent, "/robot/event", 10)
        src = self.get_parameter("source").value
        if src == "joint_states":
            self.create_subscription(JointState, "/joint_states", self.on_joints, 20)
        self.create_subscription(String, "/robot/manual_event", self.on_manual, 10)
        self.get_logger().info(f"robot events from {src}; run sheet has {len(self.run_sheet)} rows")

    def trial_info(self) -> dict:
        idx = self.trial_id - self.get_parameter("first_trial_id").value
        if 0 <= idx < len(self.run_sheet):
            return self.run_sheet[idx]
        return {}

    def emit(self, event: str, stamp=None, source="joint_states"):
        if event == RobotEvent.PRE_PICK:
            self.trial_id += 1  # a new pick begins at its pre-pick moment
        info = self.trial_info()
        msg = RobotEvent()
        msg.header.stamp = stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_base"
        msg.event = event
        msg.trial_id = self.trial_id
        msg.object_cls = info.get("object_cls", "")
        msg.pick_point = info.get("pick_point", "")
        msg.target_bin = info.get("target_bin", "")
        msg.source = source
        self.pub.publish(msg)
        self.last_event = event
        self.get_logger().info(f"trial {self.trial_id}: {event} ({msg.object_cls or '?'})")

    def on_joints(self, js: JointState):
        if js.velocity and max(abs(v) for v in js.velocity) > self.still:
            return
        q = list(js.position)
        for event, named in self.waypoints.items():
            for _, target in named.items():
                if len(target) == len(q) and max(abs(a - b) for a, b in zip(q, target)) <= self.tol:
                    if event != self.last_event:  # fire once per arrival
                        self.emit(event, js.header.stamp)
                    return

    def on_manual(self, msg: String):
        ev = msg.data.strip()
        if ev in (RobotEvent.PRE_PICK, RobotEvent.GRASP_CLOSED, RobotEvent.PLACE_DONE):
            self.emit(ev, source="manual")


def main():
    rclpy.init()
    node = RobotStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def trigger_main():
    """Keyboard trigger for manual mode: p = pre_pick, g = grasp_closed, d = place_done, q = quit."""
    rclpy.init()
    node = Node("twincheck_trigger")
    pub = node.create_publisher(String, "/robot/manual_event", 10)
    keys = {"p": RobotEvent.PRE_PICK, "g": RobotEvent.GRASP_CLOSED, "d": RobotEvent.PLACE_DONE}
    print("p = pre_pick, g = grasp_closed, d = place_done, q = quit (press Enter after each)")
    try:
        while rclpy.ok():
            k = input("> ").strip().lower()
            if k == "q":
                break
            if k in keys:
                pub.publish(String(data=keys[k]))
                rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()
