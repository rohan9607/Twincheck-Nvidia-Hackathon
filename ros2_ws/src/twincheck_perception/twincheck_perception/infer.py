"""Inference helpers with no ROS dependency, so eval scripts reuse the exact node code.

- OverheadDetector: YOLO oriented boxes -> table-frame poses (undistort + homography)
- fuse_cameras: merge detections of the same object from both overhead cameras
- WristChecker: is the object between the fingers? (reference diff, classifier or Cosmos VLM)
"""
from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from twincheck_core import Detection, GraspState, Pose2D, fuse_poses
from twincheck_core.geometry import apply_homography, canonical_yaw, pixel_yaw_to_table


@dataclass
class CameraCalib:
    name: str
    H: np.ndarray                  # undistorted pixel -> table mm
    K: np.ndarray | None = None
    dist: np.ndarray | None = None

    @classmethod
    def load(cls, name: str, homography_path: str, intrinsics_path: str | None) -> "CameraCalib":
        H = np.load(homography_path)
        K = dist = None
        if intrinsics_path and Path(intrinsics_path).exists():
            z = np.load(intrinsics_path)
            K, dist = z["K"], z["dist"]
        return cls(name, H, K, dist)

    def undistort_px(self, u: float, v: float) -> tuple[float, float]:
        if self.K is None:
            return u, v
        pts = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.K, self.dist, P=self.K)
        return float(pts[0, 0, 0]), float(pts[0, 0, 1])


class OverheadDetector:
    """Wraps an Ultralytics OBB model (.pt or TensorRT .engine)."""

    def __init__(self, model_path: str, class_symmetry: dict[str, float], conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # imported lazily so tests run without it
        self.model = YOLO(model_path, task="obb")
        self.sym = class_symmetry
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_bgr: np.ndarray, calib: CameraCalib) -> list[Detection]:
        res = self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out: list[Detection] = []
        if res.obb is None:
            return out
        names = res.names
        for (cx, cy, w, h, r), c, s in zip(res.obb.xywhr.cpu().numpy(), res.obb.cls.cpu().numpy(),
                                           res.obb.conf.cpu().numpy()):
            cls = names[int(c)]
            # long-axis angle in the image
            ang = math.degrees(r) + (90.0 if h > w else 0.0)
            u, v = calib.undistort_px(float(cx), float(cy))
            x, y = apply_homography(calib.H, u, v)
            yaw = pixel_yaw_to_table(calib.H, u, v, ang)
            out.append(Detection(cls, Pose2D(x, y, canonical_yaw(yaw, self.sym.get(cls, 360))), float(s), calib.name))
        return out


def fuse_cameras(dets: list[Detection], symmetry: dict[str, float], radius_mm: float = 25.0) -> list[Detection]:
    """Group same-class detections closer than radius (same object, different camera) and fuse them."""
    remaining = sorted(dets, key=lambda d: -d.confidence)
    fused: list[Detection] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed] + [d for d in remaining if d.cls == seed.cls and d.pose.distance_to(seed.pose) <= radius_mm]
        remaining = [d for d in remaining if d not in group]
        pose = fuse_poses([g.pose for g in group], [g.confidence for g in group], symmetry.get(seed.cls, 360))
        cams = "+".join(sorted({g.camera for g in group}))
        fused.append(Detection(seed.cls, pose, max(g.confidence for g in group), cams, f"obj_{len(fused)}"))
    return fused


# ---------------------------------------------------------------- wrist check
class WristChecker:
    """Decide HOLDING / EMPTY / MISALIGNED from one wrist image.

    method:
      reference_diff - compare the region between the fingers with an image of the
                       empty closed gripper (rendered in the twin or shot once at CSI)
      classifier     - a YOLO classification model trained on twin renders
      cosmos         - ask a vision-language model (Cosmos Nano on the Spark, or build.nvidia.com)
    """

    def __init__(self, method: str, *, reference_path: str | None = None, roi: tuple[int, int, int, int] | None = None,
                 diff_threshold: float = 18.0, classifier_path: str | None = None,
                 cosmos_base_url: str | None = None, cosmos_model: str | None = None,
                 cosmos_api_key_env: str = "NVIDIA_API_KEY", object_name: str = "object"):
        self.method = method
        self.roi = roi  # x, y, w, h of the finger gap in the wrist image
        self.diff_threshold = diff_threshold
        self.object_name = object_name
        if method == "reference_diff":
            if not reference_path or not Path(reference_path).exists():
                raise FileNotFoundError(f"wrist reference image not found: {reference_path}")
            self.ref = cv2.cvtColor(cv2.imread(reference_path), cv2.COLOR_BGR2GRAY)
        elif method == "classifier":
            from ultralytics import YOLO
            self.model = YOLO(classifier_path, task="classify")
        elif method == "cosmos":
            self.base_url = (cosmos_base_url or "").rstrip("/")
            self.model_id = cosmos_model
            self.api_key = os.environ.get(cosmos_api_key_env, "")
        else:
            raise ValueError(f"unknown wrist method {method}")

    def _crop(self, img: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return img
        x, y, w, h = self.roi
        return img[y:y + h, x:x + w]

    def check(self, image_bgr: np.ndarray) -> tuple[GraspState, float, str]:
        if self.method == "reference_diff":
            return self._reference_diff(image_bgr)
        if self.method == "classifier":
            return self._classifier(image_bgr)
        return self._cosmos(image_bgr)

    def _reference_diff(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref = cv2.resize(self.ref, (gray.shape[1], gray.shape[0]))
        a, b = self._crop(gray), self._crop(ref)
        score = float(np.mean(cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0))))
        if score > self.diff_threshold:
            return GraspState.HOLDING, min(1.0, score / (2 * self.diff_threshold)), f"diff={score:.1f}"
        return GraspState.EMPTY, min(1.0, 1 - score / self.diff_threshold), f"diff={score:.1f}"

    def _classifier(self, img):
        r = self.model.predict(self._crop(img), verbose=False)[0]
        name = r.names[int(r.probs.top1)]
        conf = float(r.probs.top1conf)
        try:
            return GraspState(name), conf, f"classifier={name}"
        except ValueError:
            return GraspState.UNKNOWN, conf, f"classifier label {name}"

    def _cosmos(self, img):
        ok, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = (f"This image is from a camera on a robot wrist looking at a two-finger gripper that has just closed. "
                  f"Is a {self.object_name} held between the fingers? Answer with exactly one word: "
                  f"HOLDING, EMPTY, or MISALIGNED (held but tilted or only partly gripped).")
        body = {"model": self.model_id, "max_tokens": 10, "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = json.load(resp)["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:  # network or model error: never pretend we saw a grasp
            return GraspState.UNKNOWN, 0.0, f"cosmos error: {e}"
        for s in (GraspState.MISALIGNED, GraspState.EMPTY, GraspState.HOLDING):
            if s.name in text:
                return s, 0.8, f"cosmos: {text[:60]}"
        return GraspState.UNKNOWN, 0.0, f"cosmos unclear: {text[:60]}"
