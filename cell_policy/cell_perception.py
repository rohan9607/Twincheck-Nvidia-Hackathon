"""Perception for the organizers' cell: table-camera detections in table mm, and the wrist grasp check.

Pixels map to the table with the organizers' calibration (client package, calibration/geometry.py:
pixel_to_base), so our own homography is not needed on the real cell. A part is seen by its top face,
so each class's pixel ray is cut at the surface plus that class's height.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from twincheck_core import Config, Detection, GraspState, Pose2D
from twincheck_core.geometry import canonical_yaw

log = logging.getLogger("twincheck.perception")


class TableDetector:
    """YOLO oriented boxes on the table image -> Detection in the table frame (mm, deg)."""

    def __init__(self, cfg: Config, model_path: str | Path, conf: float = 0.25):
        from calibration.geometry import load, pixel_to_base  # the client package (run.py's folder)
        from ultralytics import YOLO

        self.cam = load()["table"]
        self.to_base = pixel_to_base
        self.model = YOLO(str(model_path), task="obb")
        self.cfg = cfg
        self.conf = conf
        self.surface_m = cfg.raw["cell"]["surface_z_base_mm"] / 1000.0

    def _table_mm(self, u: float, v: float, z_m: float) -> tuple[float, float]:
        p = self.to_base(self.cam, (u, v), z=z_m)
        return float(p[0]) * 1000.0, float(p[1]) * 1000.0

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h, w = rgb.shape[:2]
        sx, sy = self.cam["size"][0] / w, self.cam["size"][1] / h    # calibration is for the native 1920 x 1080
        res = self.model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=self.conf, verbose=False)[0]
        out: list[Detection] = []
        if res.obb is None:
            return out
        for (cx, cy, bw, bh, r), c, s in zip(res.obb.xywhr.cpu().numpy(), res.obb.cls.cpu().numpy(),
                                             res.obb.conf.cpu().numpy()):
            cls = res.names[int(c)]
            spec = self.cfg.objects.get(cls)
            if spec is None:
                continue
            z = self.surface_m + spec.height_mm / 1000.0
            ang = float(r) + (math.pi / 2 if bh > bw else 0.0)          # long axis in the image
            u, v = float(cx) * sx, float(cy) * sy
            x, y = self._table_mm(u, v, z)
            x2, y2 = self._table_mm(u + 20 * math.cos(ang), v + 20 * math.sin(ang), z)
            yaw = canonical_yaw(math.degrees(math.atan2(y2 - y, x2 - x)), spec.symmetry_deg)
            out.append(Detection(cls, Pose2D(x, y, yaw), float(s), "table"))
        return out


class WristCheck:
    """HOLDING / EMPTY / MISALIGNED from the wrist image (twincheck_perception.infer.WristChecker)."""

    def __init__(self, cfg: Config, method: str, repo: Path):
        from twincheck_perception.infer import WristChecker

        m = cfg.raw["models"]
        self.checker = WristChecker(method, reference_path=str(repo / m["wrist_reference"]),
                                    classifier_path=str(repo / m["wrist_classifier"]),
                                    cosmos_base_url=m["cosmos"]["base_url"], cosmos_model=m["cosmos"]["model"],
                                    cosmos_api_key_env=m["cosmos"]["api_key_env"])

    def check(self, rgb: np.ndarray) -> GraspState:
        state, conf, why = self.checker.check(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log.info("wrist: %s (%.2f, %s)", state.value, conf, why)
        return state
