"""Conversions between ROS messages and twincheck_core types."""
from twincheck_core import Detection, GraspState, Pose2D
from twincheck_msgs.msg import GraspState as GraspStateMsg
from twincheck_msgs.msg import ObjectArray

MSG_TO_STATE = {GraspStateMsg.HOLDING: GraspState.HOLDING, GraspStateMsg.EMPTY: GraspState.EMPTY,
                GraspStateMsg.MISALIGNED: GraspState.MISALIGNED, GraspStateMsg.UNKNOWN: GraspState.UNKNOWN}


def stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def detections(msg: ObjectArray) -> list[Detection]:
    return [Detection(o.cls, Pose2D(o.x_mm, o.y_mm, o.yaw_deg), o.confidence, o.camera, o.id) for o in msg.objects]


def grasp_state(msg: GraspStateMsg) -> GraspState:
    return MSG_TO_STATE.get(msg.state, GraspState.UNKNOWN)
