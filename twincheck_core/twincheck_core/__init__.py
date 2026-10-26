"""TwinCheck core: pure-Python check logic shared by ROS 2 nodes, the twin and analysis."""
from .cell import CellSpec, OutsideWorkspace, check_row, interpolate
from .checks import (CheckOutcome, CheckPoint, Detection, GraspState, check_post_grasp, check_post_place,
                     check_pre_pick, grasp_from_table)
from .comparator import Action, Comparator, Decision, Mode
from .config import BinSpec, Config, ObjectSpec, Thresholds, load_config
from .geometry import Pose2D, apply_homography, fuse_poses, yaw_diff_deg
from .grasp import Grasp, plan_grasp
from .pickloop import Event, Job, Need, PickLoop, View
from .resync import ResyncUpdate, plan_resync
from .stats import pick_threshold, rate_with_interval, threshold_sweep, wilson_interval
from .timesync import StampBuffer

__all__ = [
    "CellSpec",
    "OutsideWorkspace",
    "check_row",
    "interpolate",
    "Event",
    "Job",
    "Need",
    "PickLoop",
    "View",
    "CheckOutcome",
    "CheckPoint",
    "Detection",
    "GraspState",
    "check_post_grasp",
    "check_post_place",
    "check_pre_pick",
    "grasp_from_table",
    "Action",
    "Comparator",
    "Decision",
    "Mode",
    "BinSpec",
    "Config",
    "ObjectSpec",
    "Thresholds",
    "load_config",
    "Pose2D",
    "apply_homography",
    "fuse_poses",
    "yaw_diff_deg",
    "Grasp",
    "plan_grasp",
    "ResyncUpdate",
    "plan_resync",
    "pick_threshold",
    "rate_with_interval",
    "threshold_sweep",
    "wilson_interval",
    "StampBuffer",
]
__version__ = "0.1.0"
