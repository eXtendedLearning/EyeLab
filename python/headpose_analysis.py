#!/usr/bin/env python3
"""What the 64 bytes behind ``NRHeadPoseGetPose`` actually are - Stage S3b.

Pure maths on a recorded CSV: no ctypes, no hardware, runs anywhere. The split
mirrors ``imu_analysis`` for S3.

Static analysis fixed the *size* of the pose payload (16 float32) and nothing
else. Rather than assume a convention, this module tests the candidates
against the data and reports which one the numbers support:

* **row-major 4x4**, translation in the 4th column -> indices 3, 7, 11, with
  (0, 0, 0, 1) in 12..15;
* **column-major 4x4** (the OpenGL memory order), translation in 12, 13, 14,
  with zeros in 3, 7, 11;
* **quaternion + position**, if the first four floats are consistently unit
  norm.

For whichever layout fits, the implied 3x3 block is checked for orthonormality
(unit rows, mutual orthogonality, determinant +1) - a rotation matrix passes,
a coincidence does not.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

#: How close to 0 / 1 a float has to be to count as fixed.
TOL = 1e-3

#: Rotation-block checks pass within this.
ORTHO_TOL = 1e-2

ROW_MAJOR = {"name": "row-major 4x4 (translation in the 4th column)",
             "rotation": (0, 1, 2, 4, 5, 6, 8, 9, 10),
             "translation": (3, 7, 11), "fixed": (12, 13, 14)}
COL_MAJOR = {"name": "column-major 4x4 (translation in the last 3 slots)",
             "rotation": (0, 1, 2, 4, 5, 6, 8, 9, 10),
             "translation": (12, 13, 14), "fixed": (3, 7, 11)}


def load_csv(path: Path) -> dict[str, list]:
    cols: dict[str, list] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                cols.setdefault(key, []).append(float(value) if value else math.nan)
    return cols


def _matrix(cols: dict[str, list]) -> np.ndarray:
    """The 16 pose floats as ``(n, 16)``."""
    return np.array([cols[f"m{i}"] for i in range(16)], dtype=float).T


def timing(cols: dict[str, list]) -> dict:
    """Delivered rate and jitter, from the device clock and the host clock."""
    out: dict = {}
    for label, key, scale in (("hmd", "hmd_ns", 1e-6), ("host", "host_ns", 1e-6)):
        values = np.array(cols.get(key, []), dtype=float)
        if values.size < 3:
            continue
        dt = np.diff(values) * scale
        dt = dt[np.isfinite(dt)]
        if dt.size == 0:
            continue
        out[label] = {
            "rate_hz": round(1000.0 / float(np.median(dt)), 2) if np.median(dt) else None,
            "dt_median_ms": round(float(np.median(dt)), 3),
            "dt_std_ms": round(float(np.std(dt)), 3),
            "dt_max_ms": round(float(np.max(dt)), 3),
            "non_monotonic": int(np.sum(dt <= 0)),
        }
    stale = cols.get("hmd_ns")
    if stale:
        unique = len(set(stale))
        out["distinct_device_timestamps"] = unique
        out["repeated_device_timestamps"] = len(stale) - unique
    return out


def _orthonormal(block: np.ndarray) -> dict:
    """Is this 3x3 a rotation? Reports the three tests, not just a verdict."""
    norms = np.linalg.norm(block, axis=1)
    gram = block @ block.T
    off = gram - np.diag(np.diag(gram))
    det = float(np.linalg.det(block))
    return {
        "row_norms": [round(float(n), 5) for n in norms],
        "max_offdiagonal": round(float(np.max(np.abs(off))), 5),
        "determinant": round(det, 5),
        "is_rotation": bool(np.all(np.abs(norms - 1) < ORTHO_TOL)
                            and np.max(np.abs(off)) < ORTHO_TOL
                            and abs(det - 1) < ORTHO_TOL),
    }


def _layout_fit(mats: np.ndarray, layout: dict) -> dict:
    fixed = mats[:, list(layout["fixed"])]
    last = mats[:, 15]
    fit = {
        "layout": layout["name"],
        "fixed_slots_max_abs": round(float(np.max(np.abs(fixed))), 6),
        "last_slot_mean": round(float(np.mean(last)), 6),
        "fixed_slots_are_zero": bool(np.max(np.abs(fixed)) < TOL),
        "last_slot_is_one": bool(np.all(np.abs(last - 1.0) < TOL)),
    }
    block = mats[0, list(layout["rotation"])].reshape(3, 3)
    fit["rotation_block"] = _orthonormal(block)
    translation = mats[:, list(layout["translation"])]
    fit["translation_first"] = [round(float(v), 5) for v in translation[0]]
    fit["translation_std_m"] = [round(float(v), 5) for v in np.std(translation, axis=0)]
    fit["translation_range_m"] = [round(float(v), 5)
                                  for v in (np.max(translation, axis=0)
                                            - np.min(translation, axis=0))]
    fit["consistent"] = bool(fit["fixed_slots_are_zero"] and fit["last_slot_is_one"]
                             and fit["rotation_block"]["is_rotation"])
    return fit


def _quaternion_fit(mats: np.ndarray) -> dict:
    norms = np.linalg.norm(mats[:, 0:4], axis=1)
    return {
        "first_four_norm_mean": round(float(np.mean(norms)), 5),
        "first_four_norm_std": round(float(np.std(norms)), 5),
        "consistent": bool(np.all(np.abs(norms - 1.0) < ORTHO_TOL)),
    }


def interpret(mats: np.ndarray) -> dict:
    """Which layout the recorded floats support. May legitimately be none."""
    fits = [_layout_fit(mats, ROW_MAJOR), _layout_fit(mats, COL_MAJOR)]
    quat = _quaternion_fit(mats)
    winners = [f["layout"] for f in fits if f["consistent"]]
    if quat["consistent"] and not winners:
        verdict = "first four floats are unit norm: a quaternion, not a 4x4"
    elif len(winners) == 1:
        verdict = winners[0]
    elif winners:
        verdict = "ambiguous: " + " and ".join(winners)
    else:
        verdict = "no candidate layout fits; the 64 bytes are something else"
    return {"verdict": verdict, "candidates": fits, "quaternion": quat}


def reasons(cols: dict[str, list]) -> dict:
    out: dict[str, int] = {}
    for value in cols.get("tracking_reason", []):
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out


def analyse(cols: dict[str, list]) -> dict:
    """Everything the CLI prints. Empty dict for an empty recording."""
    if not cols.get("host_ns"):
        return {"error": "no samples"}
    mats = _matrix(cols)
    result = {"samples": int(mats.shape[0]), "timing": timing(cols),
              "tracking_reasons": reasons(cols), "pose": interpret(mats)}
    moved = float(np.max(np.std(mats, axis=0)))
    result["max_slot_std"] = round(moved, 6)
    result["pose_is_static"] = bool(moved < 1e-7)
    return result
