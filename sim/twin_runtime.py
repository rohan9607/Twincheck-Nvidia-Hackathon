"""The twin at runtime, in one of two roles.

--role trials   Run scripted pick-and-sort trials INSIDE the twin (no real robot):
                moves the arm through config waypoints, closes the fingers, induces
                failures on purpose, streams cameras + /joint_states over ROS 2 so the
                normal TwinCheck nodes check every pick, and records ground truth.
                This is how we get 100 trials per condition cheaply.

--role shadow   Run beside the REAL arm: mirror its /joint_states, keep twin objects
                where the cameras last saw them (/twin/set_poses from resync_node),
                publish /twin/objects, and on every PRE_PICK publish the twin's
                prediction of whether the grasp will capture the object.

    /isaac-sim/python.sh sim/twin_runtime.py --role trials --rung random_pos --n 100 --seed 1 --induce none
    /isaac-sim/python.sh sim/twin_runtime.py --role trials --rung random_pos --n 100 --seed 1 --induce empty_grasp
    /isaac-sim/python.sh sim/twin_runtime.py --role shadow

Run the TwinCheck nodes at the same time, with the same ROS_DOMAIN_ID:
  trials role: ros2 launch twincheck_bringup twincheck.launch.py env:=twin mode:=watch events:=manual
               (this script sends the pick events itself, so the waypoint detector must be off)
  shadow role: ros2 launch twincheck_bringup twincheck.launch.py env:=real mode:=watch
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin_common import ROOT, load_sim_config, start_app  # noqa: E402

INDUCE = ["none", "empty_grasp", "removed_mid_grasp", "nudged_part"]
RUNGS = ["fixed_pose", "random_pos", "random_rot", "second_object", "clutter", "lighting"]


def enable_ros_bridge():
    from omni.isaac.core.utils.extensions import enable_extension  # works in 4.x and 5.x (deprecated alias)
    for ext in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
        try:
            enable_extension(ext)
            return ext
        except Exception:
            continue
    raise RuntimeError("could not enable the Isaac Sim ROS 2 bridge extension")


def build_ros_graph(cfg, bridge_ext: str):
    """OmniGraph that publishes /clock, /joint_states and the three camera topics."""
    import omni.graph.core as og
    ns = "isaacsim.ros2.bridge" if bridge_ext.startswith("isaacsim") else "omni.isaac.ros2_bridge"
    core = "isaacsim.core.nodes" if bridge_ext.startswith("isaacsim") else "omni.isaac.core_nodes"
    W, H = cfg["cameras"]["resolution"]
    nodes = [("tick", "omni.graph.action.OnPlaybackTick"),
             ("time", f"{core}.IsaacReadSimulationTime"),
             ("clock", f"{ns}.ROS2PublishClock"),
             ("joints", f"{ns}.ROS2PublishJointState")]
    values = [("joints.inputs:targetPrim", cfg["robot"]["articulation"]), ("joints.inputs:topicName", "joint_states")]
    conns = [("tick.outputs:tick", "clock.inputs:execIn"), ("time.outputs:simulationTime", "clock.inputs:timeStamp"),
             ("tick.outputs:tick", "joints.inputs:execIn"), ("time.outputs:simulationTime", "joints.inputs:timeStamp")]
    for cam in ("table", "side", "wrist"):
        nodes += [(f"rp_{cam}", f"{core}.IsaacCreateRenderProduct"), (f"pub_{cam}", f"{ns}.ROS2CameraHelper")]
        values += [(f"rp_{cam}.inputs:cameraPrim", cfg["cameras"][cam]), (f"rp_{cam}.inputs:width", W),
                   (f"rp_{cam}.inputs:height", H), (f"pub_{cam}.inputs:topicName", f"cam/{cam}/image_raw"),
                   (f"pub_{cam}.inputs:type", "rgb"), (f"pub_{cam}.inputs:frameId", f"cam_{cam}")]
        conns += [("tick.outputs:tick", f"rp_{cam}.inputs:execIn"),
                  (f"rp_{cam}.outputs:execOut", f"pub_{cam}.inputs:execIn"),
                  (f"rp_{cam}.outputs:renderProductPath", f"pub_{cam}.inputs:renderProductPath")]
    og.Controller.edit({"graph_path": "/TwinCheckROS", "evaluator_name": "execution"},
                       {og.Controller.Keys.CREATE_NODES: nodes, og.Controller.Keys.SET_VALUES: values,
                        og.Controller.Keys.CONNECT: conns})


class Twin:
    def __init__(self, cfg, proj, publish_sensors: bool):
        """publish_sensors: True in the trials role (the twin IS the robot and cameras).
        False in the shadow role: the real arm and real cameras publish those topics."""
        from twin_common import import_articulation, open_stage
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World
        self.cfg, self.proj = cfg, proj
        self.stage = open_stage(cfg["usd_path"])
        self.bridge = enable_ros_bridge()
        if publish_sensors:
            build_ros_graph(cfg, self.bridge)
        self.world = World(stage_units_in_meters=1.0)
        Articulation = import_articulation()
        self.robot = Articulation(cfg["robot"]["articulation"])
        self.world.reset()
        self.robot.initialize()
        self.finger_idx = [self.robot.dof_names.index(j) for j in cfg["robot"]["finger_joints"]]
        self.arm_idx = [i for i in range(len(self.robot.dof_names)) if i not in self.finger_idx]

    def step(self, n=1):
        for _ in range(n):
            self.world.step(render=True)

    def arm_q(self):
        q = self.robot.get_joint_positions()
        return [float(q[i]) for i in self.arm_idx]

    def move_joints(self, target, seconds=1.5, hz=60):
        """Straight-line joint interpolation with position targets (physics follows)."""
        q0 = np.array(self.arm_q())
        q1 = np.array(target)
        for k in range(1, int(seconds * hz) + 1):
            q = q0 + (q1 - q0) * (k / (seconds * hz))
            full = self.robot.get_joint_positions()
            for j, i in enumerate(self.arm_idx):
                full[i] = q[j]
            self.robot.apply_action(self._action(full))
            self.step()

    def fingers(self, closed: bool, settle_steps=30):
        v = self.cfg["robot"]["finger_closed" if closed else "finger_open"]
        full = self.robot.get_joint_positions()
        for i, q in zip(self.finger_idx, v if isinstance(v, list) else [v] * len(self.finger_idx)):
            full[i] = q
        self.robot.apply_action(self._action(full))
        self.step(settle_steps)

    @staticmethod
    def _action(q):
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction
        return ArticulationAction(joint_positions=q)


# --------------------------------------------------------------------------- trials role
def run_trials(args, cfg, proj):
    import psycopg
    import rclpy
    from std_msgs.msg import String
    from twin_common import nudge_mm, part_pose_mm, place_part, set_pose_mm

    tw = Twin(cfg, proj, publish_sensors=True)
    rclpy.init()
    node = rclpy.create_node("twin_trials")
    manual = node.create_publisher(String, "/robot/manual_event", 10)
    rng = random.Random(args.seed)
    wp = proj["waypoints"]
    if not wp.get("pre_pick") or not wp.get("place_done"):
        raise SystemExit("fill waypoints.pre_pick / place_done (joint values) in config/twincheck.yaml first")
    tbl = cfg["table"]
    dsn = os.environ.get("TWINCHECK_DSN", "postgresql://twincheck:twincheck@127.0.0.1:5432/twincheck")
    db = psycopg.connect(dsn, autocommit=True)
    classes = list(cfg["objects"].keys()) if args.rung in ("second_object", "clutter") else [args.object]
    # An event sent before DDS discovery matches robot_state_node is silently dropped (found on the laptop run)
    t0 = time.time()
    while manual.get_subscription_count() == 0:
        if time.time() - t0 > 30:
            raise SystemExit("nobody on /robot/manual_event: launch twincheck with events:=manual, same ROS_DOMAIN_ID")
        rclpy.spin_once(node, timeout_sec=0.1)
    tw.world.play()

    for t in range(args.n):
        cls = rng.choice(classes)
        slot = rng.choice(list(wp["pre_pick"].keys()))
        pick = proj["pick_points"][slot]
        spec = proj["objects"][cls]
        # Object placement per ladder rung
        dx = dy = dyaw = 0.0
        if args.rung != "fixed_pose":
            dx, dy = rng.uniform(-args.spread_mm, args.spread_mm), rng.uniform(-args.spread_mm, args.spread_mm)
        if args.rung in ("random_rot", "second_object", "clutter", "lighting"):
            dyaw = rng.uniform(0, 360)
        obj = cfg["objects"][cls]
        place_part(tw.stage, obj, cfg["origins"][cls], spec, pick["x_mm"] + dx, pick["y_mm"] + dy,
                   pick.get("yaw_deg", 0) + dyaw, tbl["surface_z_mm"])
        tw.step(20)

        bin_name = spec["target_bin"]
        place_q = wp["place_done"].get(bin_name) or next(iter(wp["place_done"].values()))
        tw.fingers(False)
        tw.move_joints(wp["pre_pick"][slot])
        manual.publish(String(data="pre_pick"))
        tw.step(10)

        if args.induce == "nudged_part":   # someone bumps the part just before the grab
            nudge_mm(tw.stage, obj, rng.choice([-1, 1]) * rng.uniform(20, 40), 0.0)
        if args.induce == "empty_grasp":   # the part is gone before the fingers close
            set_pose_mm(tw.stage, obj, 5000, 5000, -1000)
        tw.fingers(True)
        manual.publish(String(data="grasp_closed"))
        if args.induce == "removed_mid_grasp":
            tw.step(5)
            set_pose_mm(tw.stage, obj, 5000, 5000, -1000)
        tw.step(15)

        tw.move_joints(place_q)
        tw.fingers(False)
        manual.publish(String(data="place_done"))
        tw.step(40)

        # Ground truth from the twin itself: is the object inside its target bin now?
        x, y, _ = part_pose_mm(tw.stage, obj, cfg["origins"][cls], spec)
        b = proj["bins"][bin_name]
        success = abs(x - b["center"]["x_mm"]) <= b["half_x_mm"] and abs(y - b["center"]["y_mm"]) <= b["half_y_mm"]
        time.sleep(0.5)  # let the logger create the trial row
        db.execute("""UPDATE trials SET pick_success=%s, induced_failure=%s, seed=%s, ladder_rung=%s,
                      object_class=%s, notes=%s
                      WHERE trial_id = (SELECT max(trial_id) FROM trials WHERE env='twin')""",
                   (success, None if args.induce == "none" else args.induce, args.seed, args.rung, cls,
                    json.dumps({"dx": dx, "dy": dy, "dyaw": dyaw, "slot": slot})))
        print(f"trial {t + 1}/{args.n}: {cls} rung={args.rung} induce={args.induce} success={success}", flush=True)
        set_pose_mm(tw.stage, obj, 5000, 5000, -1000)
        rclpy.spin_once(node, timeout_sec=0.0)

    tw.world.stop()
    rclpy.shutdown()


# --------------------------------------------------------------------------- shadow role
def run_shadow(args, cfg, proj):
    import rclpy
    from sensor_msgs.msg import JointState
    from twin_common import part_pose_mm, place_part
    from twincheck_msgs.msg import DetectedObject, ExpectedState, GraspState, ObjectArray, RobotEvent

    tw = Twin(cfg, proj, publish_sensors=False)
    rclpy.init()
    node = rclpy.create_node("shadow_twin")
    real_q = {"q": None}
    id_to_prim = {}
    pool = {cls: [p] for cls, p in cfg["objects"].items()}

    node.create_subscription(JointState, args.real_joint_topic, lambda m: real_q.__setitem__("q", list(m.position)), 10)

    def on_set(msg: ObjectArray):
        used = set()
        for o in msg.objects:
            prim = id_to_prim.get(o.id)
            if prim is None:
                free = [p for p in pool.get(o.cls, []) if p not in id_to_prim.values() and p not in used]
                if not free:
                    continue
                prim = free[0]
                id_to_prim[o.id] = prim
            used.add(prim)
            spec = proj["objects"][o.cls]
            place_part(tw.stage, prim, cfg["origins"][o.cls], spec, o.x_mm, o.y_mm, o.yaw_deg, cfg["table"]["surface_z_mm"])
    node.create_subscription(ObjectArray, "/twin/set_poses", on_set, 10)

    exp_pub = node.create_publisher(ExpectedState, "/twin/expected", 10)
    obj_pub = node.create_publisher(ObjectArray, "/twin/objects", 10)

    def predict_capture(ev: RobotEvent) -> bool:
        """Would two fingers closing at the pick point capture the nearest object of this class?"""
        if ev.pick_point not in proj["pick_points"]:
            return True
        pp = proj["pick_points"][ev.pick_point]
        spec = proj["objects"].get(ev.object_cls)
        if spec is None:
            return True
        best = None
        for oid, prim in id_to_prim.items():
            cls = next(c for c, ps in pool.items() if prim in ps)
            x, y, yaw = part_pose_mm(tw.stage, prim, cfg["origins"][cls], proj["objects"][cls])
            d = math.hypot(x - pp["x_mm"], y - pp["y_mm"])
            if best is None or d < best[0]:
                best = (d, x, y, yaw)
        if best is None:
            return False
        _, x, y, yaw = best
        # Express the object center in the gripper frame (finger closing axis = pick yaw + 90)
        a = math.radians(pp.get("yaw_deg", 0) + 90)
        rx, ry = x - pp["x_mm"], y - pp["y_mm"]
        across = rx * math.cos(a) + ry * math.sin(a)      # along the closing axis
        along = -rx * math.sin(a) + ry * math.cos(a)      # along the finger face
        jaw = proj["gripper"]["max_opening_mm"]
        finger_face = args.finger_face_mm
        narrow = min(spec["width_mm"], spec["length_mm"] or spec["width_mm"])
        return abs(across) <= (jaw - narrow) / 2 and abs(along) <= finger_face / 2

    def on_event(ev: RobotEvent):
        if ev.event != RobotEvent.PRE_PICK:
            return
        m = ExpectedState()
        m.header = ev.header
        m.trial_id = ev.trial_id
        m.check_point = "pre_pick"
        m.object_expected = predict_capture(ev)
        m.expected_grasp = GraspState.HOLDING if m.object_expected else GraspState.EMPTY
        m.source = "twin"
        pp = proj["pick_points"].get(ev.pick_point, {"x_mm": 0, "y_mm": 0})
        m.expected_object = DetectedObject(id="expected", cls=ev.object_cls, x_mm=float(pp["x_mm"]),
                                           y_mm=float(pp["y_mm"]), yaw_deg=float(pp.get("yaw_deg", 0)),
                                           confidence=1.0, camera="twin")
        exp_pub.publish(m)
    node.create_subscription(RobotEvent, "/robot/event", on_event, 10)

    tw.world.play()
    print("shadow twin running; Ctrl+C to stop", flush=True)
    last_pub = 0.0
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.0)
            if real_q["q"] is not None and len(real_q["q"]) >= len(tw.arm_idx):
                full = tw.robot.get_joint_positions()
                for j, i in enumerate(tw.arm_idx):
                    full[i] = real_q["q"][j]
                tw.robot.set_joint_positions(full)  # mirror, not simulate, the real arm
            tw.step()
            if time.time() - last_pub > 0.2:
                arr = ObjectArray()
                arr.header.stamp = node.get_clock().now().to_msg()
                arr.header.frame_id = "table"
                for oid, prim in id_to_prim.items():
                    cls = next(c for c, ps in pool.items() if prim in ps)
                    x, y, yaw = part_pose_mm(tw.stage, prim, cfg["origins"][cls], proj["objects"][cls])
                    arr.objects.append(DetectedObject(id=oid, cls=cls, x_mm=x, y_mm=y, yaw_deg=yaw, confidence=1.0, camera="twin"))
                obj_pub.publish(arr)
                last_pub = time.time()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["trials", "shadow"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rung", choices=RUNGS, default="fixed_pose")
    ap.add_argument("--induce", choices=INDUCE, default="none")
    ap.add_argument("--object", default="cube", help="object for single-object rungs")
    ap.add_argument("--spread-mm", type=float, default=30.0, help="random position range around the pick point")
    ap.add_argument("--real-joint-topic", default="/joint_states", help="the real arm's joint states (read-only)")
    ap.add_argument("--finger-face-mm", type=float, default=20.0, help="TODO: measure the finger pad length")
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    cfg = load_sim_config()
    import yaml
    proj = yaml.safe_load(open(ROOT / "config" / "twincheck.yaml"))
    app = start_app(headless=args.headless)
    try:
        (run_trials if args.role == "trials" else run_shadow)(args, cfg, proj)
    finally:
        app.close()


if __name__ == "__main__":
    main()
