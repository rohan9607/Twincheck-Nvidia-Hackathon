"""Overhead detector node.

Subscribes to both overhead cameras, detects objects, maps them to the table
frame and fuses the two views. Publishes /perception/objects (ObjectArray),
stamped with the capture time of the newest image used.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from twincheck_core import load_config
from twincheck_msgs.msg import DetectedObject, ObjectArray

from .infer import CameraCalib, OverheadDetector, fuse_cameras


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        self.declare_parameter("config", "")
        self.declare_parameter("cameras", ["table"])
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("fuse_radius_mm", 25.0)
        self.declare_parameter("max_pair_gap_ms", 100.0)

        cfg = load_config(self.get_parameter("config").value or None)
        self.symmetry = {n: o.symmetry_deg for n, o in cfg.objects.items()}
        self.detector = OverheadDetector(cfg.raw["models"]["detector"], self.symmetry,
                                         conf=self.get_parameter("conf").value)
        self.bridge = CvBridge()
        self.latest: dict[str, tuple[int, list]] = {}
        self.calibs: dict[str, CameraCalib] = {}
        for cam in self.get_parameter("cameras").value:
            c = cfg.raw["cameras"][cam]
            self.calibs[cam] = CameraCalib.load(cam, c["homography"], c.get("intrinsics"))
            self.create_subscription(Image, c["topic"], lambda m, cam=cam: self.on_image(cam, m), 2)
        self.pub = self.create_publisher(ObjectArray, "/perception/objects", 10)
        self.get_logger().info(f"detector ready on {list(self.calibs)}")

    def on_image(self, cam: str, msg: Image):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest[cam] = (stamp_ns, self.detector.detect(img, self.calibs[cam]))

        # Fuse with the other camera only if its frame is close in time.
        max_gap = self.get_parameter("max_pair_gap_ms").value * 1e6
        dets = []
        for c, (s, d) in self.latest.items():
            if abs(s - stamp_ns) <= max_gap:
                dets.extend(d)
        fused = fuse_cameras(dets, self.symmetry, self.get_parameter("fuse_radius_mm").value)

        out = ObjectArray()
        out.header = msg.header
        out.header.frame_id = "table"
        for d in fused:
            out.objects.append(DetectedObject(id=d.obj_id, cls=d.cls, x_mm=d.pose.x_mm, y_mm=d.pose.y_mm,
                                              yaw_deg=d.pose.yaw_deg, confidence=d.confidence, camera=d.camera))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C or launch shutdown
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
