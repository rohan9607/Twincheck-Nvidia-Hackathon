"""Fake world: stands in for the robot, the cameras and the shadow twin, so the
whole ROS pipeline can run on a laptop with no Isaac Sim and no arm.

It publishes what the perception nodes would (/perception/objects at 15 Hz and a
/perception/grasp_state after every GRASP_CLOSED) and /twin/objects (what resync last told the twin, plus a small drift). Each
trial follows a scenario chosen at random, and the true outcome is written to the
trial row, like the twin runner and the Recorder do.

Scenarios (the induced ones use the schema's names):
  none              part on the pick point, picked and placed in its bin
  nudged_part       part moved 20-40 mm before the grab; a blind program grabs air
  empty_grasp       part on the pick point but the fingers close on nothing
  removed_mid_grasp grasp looks fine, part is lost on the way to the bin
  wrong_bin         natural failure: part lands in another bin
  nudged_empty      Guide only: part moved AND the first grasp misses (retry path)

Watch mode (the fake world presses the event keys itself):
  ros2 launch twincheck_bringup twincheck.launch.py env:=twin mode:=watch events:=manual perception:=false \
      sim_time:=false session_tag:=fake run_sheet:=<sheet>
  ros2 run twincheck_checker fake_world --ros-args -p driver:=watch -p run_sheet:=<sheet> -p session_tag:=fake

Guide mode, disarmed (guide_node drives, the fake world reacts):
  ros2 launch ... mode:=guide perception:=false sim_time:=false session_tag:=fake_guide run_sheet:=<sheet>
  ros2 run twincheck_checker fake_world --ros-args -p driver:=guide -p run_sheet:=<sheet> -p session_tag:=fake_guide

clock (default false): also publish /clock for sim_time:=true runs. A subscriber's sim clock
can lag under load and stamp events with stale times; keep it off unless testing exactly that.
keep_in_bins (default true): placed parts stay in their bins, as on the real cell, so
resync has objects to track across picks; false empties the bins after every trial.
"""
import json
import math
import os
import random
import threading
import time

import rclpy
import yaml
from builtin_interfaces.msg import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from twincheck_core import load_config
from twincheck_msgs.msg import DetectedObject, GraspState, ObjectArray, RobotEvent

SCENARIOS = ["none", "nudged_part", "empty_grasp", "removed_mid_grasp", "wrong_bin", "nudged_empty"]
SCHEMA_FAILURE = {"nudged_part": "nudged_part", "empty_grasp": "empty_grasp", "nudged_empty": "empty_grasp",
                  "removed_mid_grasp": "removed_mid_grasp"}


def wall_stamp() -> Time:
    ns = time.time_ns()
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class FakeWorld(Node):
    def __init__(self):
        super().__init__("fake_world")
        for name, default in [("config", ""), ("driver", "watch"), ("run_sheet", ""), ("n", 0), ("seed", 0),
                              ("scenarios", "none,nudged_part,empty_grasp,removed_mid_grasp,wrong_bin"),
                              ("weights", "5,2,1,1,1"), ("noise_mm", 1.5), ("twin_drift_mm", 2.0),
                              ("session_tag", ""), ("dsn", os.environ.get("TWINCHECK_DSN", "")),
                              ("step_s", 0.6), ("clock", False), ("keep_in_bins", True)]:
            self.declare_parameter(name, default)
        self.cfg = load_config(self.p("config") or None)
        self.rng = random.Random(self.p("seed"))
        with open(self.p("run_sheet")) as f:
            self.sheet = yaml.safe_load(f) or []
        self.n = self.p("n") or len(self.sheet)
        self.scenarios = [s.strip() for s in self.p("scenarios").split(",")]
        self.weights = [float(w) for w in self.p("weights").split(",")]
        bad = [s for s in self.scenarios if s not in SCENARIOS]
        if bad or len(self.weights) != len(self.scenarios):
            raise ValueError(f"bad scenarios/weights: {bad or self.weights}")

        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.scene: dict[str, tuple[str, float, float, float]] = {}   # id -> (cls, x, y, yaw) on the table
        self.twin: dict[str, tuple[str, float, float, float]] = {}    # what resync last told the twin
        self.trials: dict[int, dict] = {}                              # ros trial id -> plan and outcome
        self.next_obj = 0

        self.obj_pub = self.create_publisher(ObjectArray, "/perception/objects", 10)
        self.grasp_pub = self.create_publisher(GraspState, "/perception/grasp_state", 10)
        self.twin_pub = self.create_publisher(ObjectArray, "/twin/objects", 10)
        self.manual_pub = self.create_publisher(String, "/robot/manual_event", 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.create_subscription(RobotEvent, "/robot/event", self.on_event, 10)
        self.create_subscription(ObjectArray, "/twin/set_poses", self.on_set_poses, 10)
        # timers run on the wall clock so they keep ticking even under use_sim_time
        if self.p("clock"):
            threading.Thread(target=self.clock_loop, daemon=True).start()
        threading.Thread(target=self.sensor_loop, daemon=True).start()

        self.db = None
        if self.p("dsn") and self.p("session_tag"):
            import psycopg
            self.db = psycopg.connect(self.p("dsn"), autocommit=True)
        else:
            self.get_logger().warn("no dsn or session_tag: ground truth will only be printed")

    def p(self, name):
        return self.get_parameter(name).value

    # ---------------------------------------------------------------- sensors
    def clock_loop(self):
        while not self.stopping.is_set() and rclpy.ok():
            self.clock_pub.publish(Clock(clock=wall_stamp()))
            time.sleep(0.01)

    def sensor_loop(self):
        k = 0
        while not self.stopping.is_set() and rclpy.ok():
            noise = self.p("noise_mm")
            arr = ObjectArray()
            arr.header.stamp = wall_stamp()
            arr.header.frame_id = "table"
            with self.lock:
                for oid, (cls, x, y, yaw) in self.scene.items():
                    arr.objects.append(DetectedObject(
                        id=oid, cls=cls, x_mm=x + self.rng.gauss(0, noise), y_mm=y + self.rng.gauss(0, noise),
                        yaw_deg=yaw + self.rng.gauss(0, 1.0), confidence=0.9, camera="table"))
                twin = list(self.twin.items())
            self.obj_pub.publish(arr)
            if k % 3 == 0:   # the twin's belief, a little off: the drift resync has to correct
                t = ObjectArray()
                t.header.stamp = arr.header.stamp
                t.header.frame_id = "table"
                d = self.p("twin_drift_mm")
                for oid, (cls, x, y, yaw) in twin:
                    t.objects.append(DetectedObject(id=oid, cls=cls, x_mm=x + d, y_mm=y, yaw_deg=yaw,
                                                    confidence=1.0, camera="twin"))
                self.twin_pub.publish(t)
            k += 1
            time.sleep(1 / 15)

    def on_set_poses(self, m: ObjectArray):
        with self.lock:
            self.twin = {o.id: (o.cls, o.x_mm, o.y_mm, o.yaw_deg) for o in m.objects}

    # ---------------------------------------------------------------- scene
    def row(self, trial_id: int) -> dict:
        r = self.sheet[(trial_id - 1) % len(self.sheet)]
        cls = r["object_cls"]
        return {"cls": cls, "pick_point": r.get("pick_point", ""),
                "bin": r.get("target_bin") or self.cfg.objects[cls].target_bin}

    def plan_trial(self, trial_id: int) -> dict:
        """Choose the scenario and put the part on the table near its pick point."""
        r = self.row(trial_id)
        scen = self.rng.choices(self.scenarios, self.weights)[0]
        pp = self.cfg.pick_points[r["pick_point"]]
        x, y = pp.x_mm, pp.y_mm
        if scen in ("nudged_part", "nudged_empty"):
            dist = self.rng.uniform(20, 40)
            ang = self.rng.uniform(0, 2 * math.pi)
            x, y = x + dist * math.cos(ang), y + dist * math.sin(ang)
        oid = f"part_{self.next_obj}"
        self.next_obj += 1
        with self.lock:
            self.scene[oid] = (r["cls"], x, y, pp.yaw_deg)
        plan = {**r, "scenario": scen, "obj": oid, "grasps": 0, "held": False, "success": False}
        self.trials[trial_id] = plan
        self.get_logger().info(f"trial {trial_id}: {r['cls']} at {r['pick_point']}, scenario {scen}")
        return plan

    def grasp_outcome(self, plan: dict) -> bool:
        """Does this grasp close on the part?"""
        plan["grasps"] += 1
        scen = plan["scenario"]
        if scen == "empty_grasp":
            return self.p("driver") == "guide" and plan["grasps"] > 1   # a retry from fresh detection works
        if scen == "nudged_part":
            return self.p("driver") == "guide"     # guide grabs where the part is; a blind program misses
        if scen == "nudged_empty":
            return plan["grasps"] > 1
        return True

    def place(self, plan: dict):
        with self.lock:
            if not plan["held"]:
                return   # nothing in the gripper; the part (if any) stays where it was
            plan["held"] = False
            if plan["scenario"] == "removed_mid_grasp":
                return   # lost on the way
            target = plan["bin"]
            if plan["scenario"] == "wrong_bin":
                target = next(b for b in self.cfg.bins if b != plan["bin"])
            c = self.cfg.bins[target].center
            self.scene[plan["obj"]] = (plan["cls"], c.x_mm + self.rng.uniform(-15, 15),
                                       c.y_mm + self.rng.uniform(-15, 15), self.rng.uniform(0, 180))
            plan["success"] = target == plan["bin"]
            plan["binned"] = True

    def finish(self, trial_id: int):
        plan = self.trials[trial_id]
        with self.lock:
            # a part left on the tray is cleared by the operator; parts in bins pile up like on the real cell
            if not (self.p("keep_in_bins") and plan.get("binned")):
                self.scene.pop(plan["obj"], None)
        failure = SCHEMA_FAILURE.get(plan["scenario"])
        self.get_logger().info(f"trial {trial_id} truth: success={plan['success']} ({plan['scenario']})")
        if self.db is not None:
            self.db.execute("""UPDATE trials SET pick_success=%s, induced_failure=%s, notes=%s
                               WHERE session_tag=%s AND ros_trial_id=%s""",
                            (plan["success"], failure, json.dumps({"fake_scenario": plan["scenario"]}),
                             self.p("session_tag"), trial_id))

    # ---------------------------------------------------------------- events
    def on_event(self, ev: RobotEvent):
        plan = self.trials.get(ev.trial_id)
        if plan is None:
            return
        if ev.event == RobotEvent.GRASP_CLOSED:
            ok = self.grasp_outcome(plan)
            with self.lock:
                if ok:
                    plan["held"] = True
                    self.scene.pop(plan["obj"], None)   # part is in the gripper, off the table
            threading.Timer(0.2, self.publish_grasp, args=(ev.trial_id, ok)).start()
        elif ev.event == RobotEvent.PLACE_DONE and self.p("driver") == "guide":
            self.place(plan)
            plan["placed"] = True   # run_guide puts the next part on the tray right away
            threading.Timer(1.5, self.finish, args=(ev.trial_id,)).start()

    def publish_grasp(self, trial_id: int, held: bool):
        m = GraspState()
        m.header.stamp = wall_stamp()
        m.state = GraspState.HOLDING if held else GraspState.EMPTY
        m.confidence = 0.9
        m.method = "fake"
        m.detail = f"trial {trial_id}: fake world says {'holding' if held else 'empty'}"
        self.grasp_pub.publish(m)

    # ---------------------------------------------------------------- drivers
    def wait_for_pipeline(self, pubs, timeout_s=30.0):
        """A message sent before DDS discovery matches a subscriber is silently dropped,
        so wait until every topic we drive has its listener."""
        t0 = time.time()
        while any(p.get_subscription_count() == 0 for p in pubs):
            if time.time() - t0 > timeout_s:
                missing = [p.topic_name for p in pubs if p.get_subscription_count() == 0]
                raise RuntimeError(f"nobody listening on {missing}; is the launch running with the same ROS_DOMAIN_ID?")
            time.sleep(0.1)
        time.sleep(0.5)

    def run_watch(self):
        """The robot's own blind program: same moves every time, events from the key presses."""
        step = self.p("step_s")
        self.wait_for_pipeline([self.manual_pub, self.obj_pub, self.grasp_pub])
        for trial_id in range(1, self.n + 1):
            plan = self.plan_trial(trial_id)
            time.sleep(step)
            self.manual_pub.publish(String(data="pre_pick"))
            time.sleep(step)
            self.manual_pub.publish(String(data="grasp_closed"))
            time.sleep(step + 0.4)   # wrist answer + carry to the bin
            self.place(plan)
            time.sleep(0.2)
            self.manual_pub.publish(String(data="place_done"))
            time.sleep(1.5)          # post-place settle and resync
            self.finish(trial_id)
        self.get_logger().info("fake world: all trials done")

    def run_guide(self):
        """guide_node drives; keep a part on the tray for each trial it starts."""
        self.wait_for_pipeline([self.obj_pub, self.grasp_pub])
        for trial_id in range(1, self.n + 1):
            # guide reads detections as soon as the previous post-place check is in
            self.plan_trial(trial_id)
            t0 = time.time()
            while rclpy.ok() and not self.trials[trial_id].get("placed"):
                if time.time() - t0 > 30:
                    self.get_logger().error(f"guide never placed trial {trial_id}; moving on")
                    self.finish(trial_id)
                    break
                time.sleep(0.05)
        time.sleep(2.0)   # last finish()
        self.get_logger().info("fake world: all trials done")


def main():
    rclpy.init()
    node = FakeWorld()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        (node.run_guide if node.p("driver") == "guide" else node.run_watch)()
        time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.stopping.set()
        time.sleep(0.2)
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
