"""Load the single project config (config/twincheck.yaml).

Every threshold, object, bin and waypoint lives in that one file so the twin,
the real arm and the tests all agree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .geometry import Pose2D


@dataclass
class ObjectSpec:
    name: str
    symmetry_deg: float
    target_bin: str
    width_mm: float = 0.0
    length_mm: float = 0.0


@dataclass
class BinSpec:
    name: str
    center: Pose2D
    half_x_mm: float
    half_y_mm: float

    def contains(self, p: Pose2D, margin_mm: float = 0.0) -> bool:
        return (abs(p.x_mm - self.center.x_mm) <= self.half_x_mm + margin_mm
                and abs(p.y_mm - self.center.y_mm) <= self.half_y_mm + margin_mm)


@dataclass
class Thresholds:
    pre_pick_mm: float = 10.0
    pre_pick_deg: float = 10.0
    post_place_mm: float = 15.0
    match_radius_mm: float = 60.0
    min_confidence: float = 0.5
    sync_tolerance_ms: float = 100.0
    max_retries: int = 1


@dataclass
class Config:
    objects: dict[str, ObjectSpec]
    bins: dict[str, BinSpec]
    thresholds: Thresholds
    pick_points: dict[str, Pose2D] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def symmetry(self, cls: str) -> float:
        spec = self.objects.get(cls)
        return spec.symmetry_deg if spec else 360.0


def _pose(d: dict) -> Pose2D:
    return Pose2D(float(d["x_mm"]), float(d["y_mm"]), float(d.get("yaw_deg", 0.0)))


def load_config(path: str | os.PathLike | None = None) -> Config:
    if path is None:
        path = os.environ.get("TWINCHECK_CONFIG") or Path(__file__).resolve().parents[2] / "config" / "twincheck.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    objects = {
        name: ObjectSpec(name=name, symmetry_deg=float(o.get("symmetry_deg", 360)), target_bin=o["target_bin"],
                         width_mm=float(o.get("width_mm", 0)), length_mm=float(o.get("length_mm", 0)))
        for name, o in raw["objects"].items()
    }
    bins = {
        name: BinSpec(name=name, center=_pose(b["center"]), half_x_mm=float(b["half_x_mm"]),
                      half_y_mm=float(b["half_y_mm"]))
        for name, b in raw["bins"].items()
    }
    th = Thresholds(**raw.get("thresholds", {}))
    picks = {name: _pose(p) for name, p in (raw.get("pick_points") or {}).items()}
    return Config(objects=objects, bins=bins, thresholds=th, pick_points=picks, raw=raw)
