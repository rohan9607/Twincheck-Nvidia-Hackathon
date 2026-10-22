"""Guide mode: vision-guided pick and sort with retry.

SAFETY FIRST
  - Starts DISARMED: it plans and logs every move but sends nothing to the arm.
  - Arm it only with the plant manager and mentor present:  armed:=true
  - Every target is clamped to config safety.workspace_limits_mm; anything outside is refused.
  - Velocity and acceleration are scaled down (config safety.*_scaling).
  - A STOP from the comparator halts the loop until an operator publishes "resume" on /guide/command.

It assumes the overhead calibration maps pixels straight to robot_base x/y in mm
(markers touched with the arm's tool tip), so table x/y == robot_base x/y and
heights are given as z_mm parameters.

The FANUC ROS 2 driver, MoveIt group name, tool link and gripper I/O service are
parameters: confirm them with Dr. Zeng before arming.
"""
import math
import threading
import time

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool

from twincheck_core import Pose2D, StampBuffer, load_config, plan_grasp
from twincheck_core.checks import nearest_detection
from twincheck_msgs.msg import CheckResult, ObjectArray, RobotEvent


def quat_top_down(yaw_deg: float):
    """Tool pointing straight down (roll = pi), rotated by yaw about z. Returns (x, y, z, w)."""
    r, y = math.pi, math.radians(yaw_deg)
    cr, sr, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(y / 2), math.sin(y / 2)
    return (sr * cy, sr * sy, cr * sy, cr * cy)


class GuideNode(Node):
    def __init__(self):
        super().__init__("guide_node")
        for name, default in [("config", ""), ("armed", False), ("run_sheet", ""), ("planning_group", "manipulator"),
                              ("tool_link", "tool0"), ("base_frame", "base_link"), ("gripper_service", "/gripper/set"),
                              ("table_z_mm", 0.0), ("approach_z_mm", 120.0), ("grasp_z_mm", 15.0),
                              ("place_z_mm", 80.0), ("search_radius_mm", 120.0), ("first_trial_id", 1),
                              ("part_wait_s", 10.0)]:
            self.declare_parameter(name, default)
        self.cfg = load_config(self.p("config") or None)
        self.safety = self.cfg.raw["safety"]
        self.armed = bool(self.p("armed"))
        self.objects: StampBuffer = StampBuffer(30)
        self.results: dict[tuple[int, str], CheckResult] = {}
        self.stopped = False
        self.lock = threading.Lock()

        self.create_subscription(ObjectArray, "/perception/objects", self.on_objects, 10)
        self.create_subscription(CheckResult, "/check/result", self.on_result, 10)
        self.create_subscription(String, "/guide/command", self.on_command, 10)
        self.event_pub = self.create_publisher(RobotEvent, "/robot/event", 10)
        self.gripper = self.create_client(SetBool, self.p("gripper_service"))

        self.moveit = None
        if self.armed:
            self.get_logger().warn("GUIDE MODE IS ARMED: the arm will move. Plant manager and mentor must be present.")
            from moveit.planning import MoveItPy  # only needed when armed
            self.moveit = MoveItPy(node_name="twincheck_moveit")
            self.arm = self.moveit.get_planning_component(self.p("planning_group"))
        else:
            self.get_logger().info("guide mode DISARMED (dry run): moves are planned and logged, not executed")

        with open(self.p("run_sheet")) as f:
            self.sheet = yaml.safe_load(f) or []
        threading.Thread(target=self.run, daemon=True).start()

    def p(self, name):
        return self.get_parameter(name).value

    # ---------------------------------------------------------------- inputs
    def on_objects(self, m: ObjectArray):
        self.objects.push(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec, m)

    def on_result(self, r: CheckResult):
        with self.lock:
            self.results[(r.trial_id, r.check_point)] = r

    def on_command(self, m: String):
        cmd = m.data.split(":")[0]
        if cmd == "stop":
            self.stopped = True
            self.get_logger().error(f"STOP received ({m.data}); waiting for 'resume' on /guide/command")
        elif cmd == "resume":
            self.stopped = False

    # ---------------------------------------------------------------- helpers
    def emit(self, event, trial_id, row, target: Pose2D | None = None):
        ev = RobotEvent()
        ev.header.stamp = self.get_clock().now().to_msg()
        ev.header.frame_id = "robot_base"
        ev.event, ev.trial_id, ev.source = event, trial_id, "guide"
        ev.object_cls, ev.pick_point, ev.target_bin = row["object_cls"], row.get("pick_point", ""), row.get("target_bin", "")
        if target is not None:   # the pre-pick check compares against where we will actually grab
            ev.has_target = True
            ev.target_x_mm, ev.target_y_mm, ev.target_yaw_deg = target.x_mm, target.y_mm, target.yaw_deg
        self.event_pub.publish(ev)

    def wait_result(self, trial_id, cp, timeout=8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                r = self.results.get((trial_id, cp))
            if r is not None:
                return r
            time.sleep(0.05)
        return None

    def latest_detections(self):
        from twincheck_core import Detection
        if not len(self.objects):
            return []
        m = self.objects.nearest(self.get_clock().now().nanoseconds, 1000)
        if m is None:
            return []
        return [Detection(o.cls, Pose2D(o.x_mm, o.y_mm, o.yaw_deg), o.confidence, o.camera, o.id) for o in m.item.objects]

    def wait_for_part(self, cls, area):
        """The next part may still be arriving (or the cameras starting): look for up to part_wait_s."""
        t0 = time.time()
        while rclpy.ok():
            dets = self.latest_detections()
            target = nearest_detection(area, dets, cls, self.p("search_radius_mm")) if area else \
                next((d for d in dets if d.cls == cls), None)
            if target is not None or time.time() - t0 > self.p("part_wait_s"):
                return target
            time.sleep(0.1)
        return None

    def within_limits(self, x, y, z) -> bool:
        lim = self.safety["workspace_limits_mm"]
        return lim["x_min"] <= x <= lim["x_max"] and lim["y_min"] <= y <= lim["y_max"] and z >= lim["z_min"]

    def move(self, x, y, z, yaw) -> bool:
        if not self.within_limits(x, y, z):
            self.get_logger().error(f"refused move to ({x:.0f}, {y:.0f}, {z:.0f}) mm: outside workspace limits")
            return False
        self.get_logger().info(f"{'MOVE' if self.armed else 'DRY-RUN move'} to x={x:.0f} y={y:.0f} z={z:.0f} yaw={yaw:.0f}")
        if not self.armed:
            time.sleep(0.2)
            return True
        ps = PoseStamped()
        ps.header.frame_id = self.p("base_frame")
        ps.pose.position.x, ps.pose.position.y = x / 1000.0, y / 1000.0
        ps.pose.position.z = (self.p("table_z_mm") + z) / 1000.0
        qx, qy, qz, qw = quat_top_down(yaw)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=ps, pose_link=self.p("tool_link"))
        from moveit.planning import PlanRequestParameters
        params = PlanRequestParameters(self.moveit, "")
        params.max_velocity_scaling_factor = self.safety["max_velocity_scaling"]
        params.max_acceleration_scaling_factor = self.safety["max_acceleration_scaling"]
        plan = self.arm.plan(single_plan_parameters=params)
        if not plan:
            self.get_logger().error("planning failed")
            return False
        self.moveit.execute(plan.trajectory, controllers=[])
        return True

    def set_gripper(self, close: bool):
        self.get_logger().info(f"{'' if self.armed else 'DRY-RUN '}gripper {'close' if close else 'open'}")
        if not self.armed:
            return
        if not self.gripper.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"gripper service {self.p('gripper_service')} not available")
        fut = self.gripper.call_async(SetBool.Request(data=close))
        while not fut.done():
            time.sleep(0.02)
        time.sleep(0.4)  # pneumatic jaws need a moment

    # ---------------------------------------------------------------- main loop
    def run(self):
        time.sleep(1.0)
        trial_id = self.p("first_trial_id") - 1
        for row in self.sheet:
            trial_id += 1
            cls = row["object_cls"]
            spec = self.cfg.objects[cls]
            bin_c = self.cfg.bins[row.get("target_bin") or spec.target_bin].center
            area = self.cfg.pick_points.get(row.get("pick_point", ""), None)
            attempts = 0
            while rclpy.ok():
                while self.stopped:
                    time.sleep(0.2)
                target = self.wait_for_part(cls, area)
                if target is None:
                    self.get_logger().warn(f"trial {trial_id}: no {cls} visible after {self.p('part_wait_s'):.0f} s; skipping")
                    break
                g = plan_grasp(target.pose, spec, self.cfg.raw["gripper"]["max_opening_mm"],
                               self.cfg.raw["gripper"]["clearance_mm"])
                if not g.fits:
                    self.get_logger().error(f"trial {trial_id}: {g.reason}; skipping")
                    break
                self.emit(RobotEvent.PRE_PICK, trial_id, row, target.pose)
                self.wait_result(trial_id, "pre_pick", 3.0)
                gx, gy, gyaw = g.pose.x_mm, g.pose.y_mm, g.pose.yaw_deg
                self.set_gripper(False)
                ok = self.move(gx, gy, self.p("approach_z_mm"), gyaw) and self.move(gx, gy, self.p("grasp_z_mm"), gyaw)
                if not ok:
                    break
                self.set_gripper(True)
                self.emit(RobotEvent.GRASP_CLOSED, trial_id, row)
                self.move(gx, gy, self.p("approach_z_mm"), gyaw)
                r = self.wait_result(trial_id, "post_grasp")
                if r is not None and not r.passed:
                    self.set_gripper(False)
                    attempts += 1
                    with self.lock:
                        self.results.pop((trial_id, "post_grasp"), None)
                        self.results.pop((trial_id, "pre_pick"), None)
                    if r.action == "retry" and attempts <= self.cfg.thresholds.max_retries:
                        self.get_logger().warn(f"trial {trial_id}: retrying from a fresh detection")
                        continue
                    break
                self.move(bin_c.x_mm, bin_c.y_mm, self.p("place_z_mm"), 0.0)
                self.set_gripper(False)
                self.emit(RobotEvent.PLACE_DONE, trial_id, row)
                self.wait_result(trial_id, "post_place")
                break
        self.get_logger().info("run sheet finished")


def main():
    rclpy.init()
    node = GuideNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C or launch shutdown
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
