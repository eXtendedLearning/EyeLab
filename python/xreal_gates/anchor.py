"""Offline lock test: how still does a target stay in the world, frame by frame?

For every snapshot of an XR run (G4 bursts, G5 locks), estimate the target pose in
the camera, chain it with the 6DoF head pose at receipt and the vendor's
RGB-camera-from-head pose, and express the target centre in world (tracking)
coordinates. A rigidly placed target should give the same point every time; the
spread is what a camera-locked overlay would jitter or jump by.

Frame chain (the same as the G5 gate, ``GateController.ApplyLock``):

- OpenCV camera (x right, y down, z forward) -> Unity camera: ``F = diag(1, -1, 1)``
- Unity camera -> head: ``device_xr.rgb_camera_pose_from_head`` (run.json)
- head -> world: ``pose.csv`` head pose (``update`` rows) nearest the frame's receipt
  time. Receipt, not exposure: during head motion the error grows with the
  unmeasured capture latency, so compare still phases.
"""

from __future__ import annotations

import bisect
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .eyecam import EyeCamera
from .targets import GateTarget

F = np.diag([1.0, -1.0, 1.0])


def qrot(q, v) -> np.ndarray:
    """Rotate v by the unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z], float)
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def qmat(q) -> np.ndarray:
    return np.column_stack([qrot(q, e) for e in np.eye(3)])


@dataclass
class Lock:
    t: float
    step: str
    snapshot: str
    centre_world: np.ndarray     # m
    rot_world: np.ndarray        # 3x3, target frame (Unity-flipped) in world
    head: np.ndarray             # m
    dist_m: float
    rms_px: float
    n_markers: int
    n_points: int


def _rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def analyse(run_dir: Path, target: GateTarget, camera: EyeCamera) -> List[Lock]:
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    head_cam = info.get("device_xr", {}).get("rgb_camera_pose_from_head", {})
    if not head_cam.get("ok"):
        raise ValueError(f"{run_dir.name}: no rgb_camera_pose_from_head in run.json (not an XR run?)")
    t_hc, q_hc = np.array(head_cam["position_m"], float), head_cam["rotation_xyzw"]
    received = {r["cam_ts"]: float(r["t_unity"]) for r in _rows(run_dir / "camera.csv") if r["t_unity"]}
    pose = [r for r in _rows(run_dir / "pose.csv") if r["phase"] == "update" and r["t_unity"]]
    pose_t = [float(r["t_unity"]) for r in pose]
    steps = [(float(e["t_unity"]), e["result"]) for e in _rows(run_dir / "events.csv")
             if e["event"] == "step" and e["t_unity"]]
    centre_local = np.mean(np.array(target.overlay()["points"], float), axis=0)

    def step_at(t: float) -> str:
        s = "before_protocol"
        for t0, name in steps:
            if t >= t0:
                s = name
        return s

    locks = []
    for path in sorted((run_dir / "snapshots").glob("*.pgm")):
        t = received.get(path.stem)
        if t is None or not pose:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        est = target.estimate(gray, camera.K, camera.dist)
        if not est.ok:
            continue
        R, _ = cv2.Rodrigues(est.rvec)
        p_cam = F @ (R @ centre_local + np.ravel(est.tvec))
        r = pose[min(bisect.bisect_left(pose_t, t), len(pose) - 1)]
        q_h = [float(r[k]) for k in ("rot_x", "rot_y", "rot_z", "rot_w")]
        p_h = np.array([float(r[k]) for k in ("pos_x", "pos_y", "pos_z")])
        centre_world = qrot(q_h, qrot(q_hc, p_cam) + t_hc) + p_h
        rot_world = qmat(q_h) @ qmat(q_hc) @ (F @ R @ F)
        locks.append(Lock(t, step_at(t), path.name, centre_world, rot_world, p_h,
                          float(np.linalg.norm(est.tvec)), est.rms_px, est.n_markers, est.n_points))
    return locks


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def report(locks: List[Lock]) -> List[str]:
    if not locks:
        return ["no snapshot gave an accepted target pose"]
    W = np.array([lk.centre_world for lk in locks])
    ref = np.median(W, axis=0)
    dev = np.linalg.norm(W - ref, axis=1) * 1000.0
    ref_rot = locks[int(np.argmin(dev))].rot_world
    lines = [f"{len(locks)} locks; target centre in world (median) {np.round(ref, 3).tolist()} m; "
             f"deviation from it: median {np.median(dev):.1f} mm, p90 {np.percentile(dev, 90):.1f} mm, "
             f"max {dev.max():.1f} mm"]
    groups: Dict[str, List[int]] = OrderedDict()
    for i, lk in enumerate(locks):
        groups.setdefault(lk.step, []).append(i)
    for step, idx in groups.items():
        w = W[idx]
        within = np.linalg.norm(w - np.median(w, axis=0), axis=1) * 1000.0
        ang = [_angle_deg(ref_rot, locks[i].rot_world) for i in idx]
        lines.append(f"  {step:18s} n={len(idx):3d} dist {np.median([locks[i].dist_m for i in idx]):.2f} m "
                     f"rms {np.median([locks[i].rms_px for i in idx]):.2f} px  from run median {np.median(dev[idx]):6.1f} mm"
                     f"  spread within step {np.median(within):5.1f} (max {within.max():5.1f}) mm"
                     f"  rotation {np.median(ang):4.1f} deg")
    return lines


def summary_numbers(locks: List[Lock]) -> Optional[dict]:
    if not locks:
        return None
    W = np.array([lk.centre_world for lk in locks])
    dev = np.linalg.norm(W - np.median(W, axis=0), axis=1) * 1000.0
    return {"n": len(locks), "median_mm": float(np.median(dev)), "p90_mm": float(np.percentile(dev, 90))}
