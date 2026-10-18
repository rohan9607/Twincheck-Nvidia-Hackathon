"""Bring up the whole TwinCheck pipeline.

Examples
  Twin, Watch mode:   ros2 launch twincheck_bringup twincheck.launch.py env:=twin mode:=watch
  Real arm, Watch:    ros2 launch twincheck_bringup twincheck.launch.py env:=real mode:=watch events:=joint_states
  Real arm, manual:   ros2 launch twincheck_bringup twincheck.launch.py env:=real mode:=watch events:=manual
                      (then run: ros2 run twincheck_checker twincheck_trigger)
  Baseline logging:   ros2 launch twincheck_bringup twincheck.launch.py env:=real mode:=baseline
  Guide, dry run:     ros2 launch twincheck_bringup twincheck.launch.py env:=twin mode:=guide
  Guide, ARMED:       ... mode:=guide armed:=true   (plant manager + mentor present only)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    share = get_package_share_directory("twincheck_bringup")
    config = lc("config") or os.environ.get("TWINCHECK_CONFIG", "")
    run_sheet = lc("run_sheet") or os.path.join(share, "config", "run_sheet_example.yaml")
    env, mode = lc("env"), lc("mode")
    use_sim_time = env == "twin"
    common = {"config": config, "use_sim_time": use_sim_time}
    # Baseline runs log trials and checks exactly like Watch, so the numbers line up;
    # the robot keeps its own program and nobody acts on the alerts.
    checker_mode = "guide" if mode == "guide" else "watch"
    events = "guide" if mode == "guide" else lc("events")

    nodes = [
        Node(package="twincheck_perception", executable="detector_node", parameters=[common]),
        Node(package="twincheck_perception", executable="grasp_check_node",
             parameters=[common | {"method": lc("wrist_method")}]),
        Node(package="twincheck_checker", executable="robot_state_node",
             parameters=[common | {"source": events, "run_sheet": run_sheet}]),
        Node(package="twincheck_checker", executable="expectation_node", parameters=[common]),
        Node(package="twincheck_checker", executable="comparator_node",
             parameters=[common | {"mode": checker_mode, "image_dir": lc("image_dir")}]),
        Node(package="twincheck_checker", executable="resync_node", parameters=[common]),
        Node(package="twincheck_checker", executable="alert_node", parameters=[common], output="screen"),
        Node(package="twincheck_logger", executable="trial_logger",
             parameters=[common | {"env": env, "mode": mode, "ladder_rung": lc("rung"),
                                   "model_version": lc("model_version"), "session_tag": lc("session_tag")}]),
    ]
    if mode == "guide":
        nodes.append(Node(package="twincheck_guide", executable="guide_node", output="screen",
                          parameters=[common | {"armed": lc("armed").lower() == "true", "run_sheet": run_sheet}]))
    return nodes


def generate_launch_description():
    args = [
        DeclareLaunchArgument("env", default_value="twin", description="twin | real"),
        DeclareLaunchArgument("mode", default_value="watch", description="baseline | watch | guide"),
        DeclareLaunchArgument("events", default_value="joint_states", description="joint_states | manual"),
        DeclareLaunchArgument("wrist_method", default_value="reference_diff", description="reference_diff | classifier | cosmos"),
        DeclareLaunchArgument("armed", default_value="false", description="Guide only: true lets the arm move"),
        DeclareLaunchArgument("config", default_value=""),
        DeclareLaunchArgument("run_sheet", default_value=""),
        DeclareLaunchArgument("rung", default_value="fixed_pose"),
        DeclareLaunchArgument("model_version", default_value=""),
        DeclareLaunchArgument("session_tag", default_value=""),
        DeclareLaunchArgument("image_dir", default_value="runs/images"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_nodes)])
