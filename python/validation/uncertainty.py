"""Ground-truth uncertainty from the measuring instrument (GUM, JCGM 100:2008).

What the plates' ground truth depends on, and what it does not:

- Uniform print scale (s_x = s_y) cancels in every on-face registration
  metric: markers and reference points come from the same print, so a scaled
  plate gives the same image (the pose just sits at a scaled depth). It only
  matters for absolute lengths (B3 steps, distances).
- Anisotropy (s_x != s_y) does not cancel: a point at distance R from the
  face anchor moves by up to R * |s_x - s_y| / 2 relative to the markers.
- The fold angle of plate L does not change the 2D face ground truth, but it
  is part of the 3D model EyeLab registers to: an error d_phi moves the
  points of face V by up to d * d_phi (d = height above the fold).

An instrument quoted as "+-a" is taken as a rectangular distribution,
standard uncertainty a / sqrt(3) (GUM 4.3.7). Expanded uncertainty U = k u
with k = 2 (~95 %). The test uncertainty ratio against the accuracy goal is
TUR = goal / U; 4 or more is the usual requirement (ANSI/NCSL Z540.3).
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, Optional, Sequence

GOAL_MM = 1.0           # accuracy the validation must be able to resolve
K_COVERAGE = 2.0
MIN_TUR = 4.0
PRINT_PLACEMENT_MM = 0.02   # printer dot placement on a 600 dpi raster
PAPER_FLATNESS_MM = 0.05    # paper bonded to a rigid flat board


def rect_u(half_width: float) -> float:
    """Standard uncertainty of a +-half_width rectangular distribution."""
    return float(half_width) / math.sqrt(3.0)


def reading_u(readings: Sequence[float], accuracy_mm: float) -> float:
    """Standard uncertainty of the mean of repeated readings (type A + type B)."""
    u_b = rect_u(accuracy_mm)
    if len(readings) > 1:
        u_a = statistics.stdev(readings) / math.sqrt(len(readings))
    else:
        u_a = 0.0
    return math.hypot(u_a, u_b)


def expanded(u: float) -> float:
    return K_COVERAGE * u


def tur(u_mm: float, goal_mm: float = GOAL_MM) -> float:
    """Test uncertainty ratio of a measurement with standard uncertainty ``u_mm``."""
    U = expanded(u_mm)
    return float("inf") if U <= 0 else goal_mm / U


def lever_mm(target) -> Dict[str, float]:
    """Largest in-face distance of a reference point or marker centre from the
    face anchor (``r``), and largest height of a face-V point above the fold
    (``fold_arm``, plate L only)."""
    import numpy as np
    r = 0.0
    for pid in target.ids():
        r = max(r, float(np.hypot(*target.face_coords([pid])[0])))
    for m in target.markers:
        r = max(r, float(np.hypot(*target.to_face(m.face, m.center)[0])))
    arm = 0.0
    if "V" in target.faces:
        arm = max((float(target.face_coords([p])[0][1]) for p in target.face_ids("V")), default=0.0)
    return {"r": 1000.0 * r, "fold_arm": 1000.0 * abs(arm)}


def point_uncertainty(target, u_bar_x_mm: float, u_bar_y_mm: float,
                      bar_x_mm: float, bar_y_mm: float,
                      u_fold_deg: Optional[float] = None) -> dict:
    """Standard uncertainty (mm) of a reference point, as seen by the on-face
    registration metrics, with the budget that produced it."""
    rx, ry = u_bar_x_mm / bar_x_mm, u_bar_y_mm / bar_y_mm
    u_aniso = math.hypot(rx, ry)                  # u(s_x - s_y)
    lev = lever_mm(target)
    budget = {
        "anisotropy_mm": 0.5 * lev["r"] * u_aniso,
        "print_placement_mm": PRINT_PLACEMENT_MM,
        "paper_flatness_mm": PAPER_FLATNESS_MM,
    }
    if u_fold_deg is not None and lev["fold_arm"] > 0:
        budget["fold_angle_mm"] = lev["fold_arm"] * math.radians(u_fold_deg)
    u = math.sqrt(sum(v * v for v in budget.values()))
    return {"u_mm": u, "U_mm": expanded(u), "tur": tur(u), "budget": budget,
            "u_scale_rel": 0.5 * u_aniso, "u_aniso_rel": u_aniso, "lever_mm": lev}


def step_uncertainty(accuracy_mm: float) -> float:
    """Standard uncertainty of a B3 step: difference of two positions, each set
    with the instrument (independent settings: u * sqrt(2))."""
    return math.sqrt(2.0) * rect_u(accuracy_mm)


def verdict(u_mm: float, goal_mm: float = GOAL_MM) -> str:
    t = tur(u_mm, goal_mm)
    if t >= MIN_TUR:
        return f"OK: TUR {t:.1f} >= {MIN_TUR:g} for a {goal_mm:g} mm goal"
    return (f"INSUFFICIENT: TUR {t:.1f} < {MIN_TUR:g} for a {goal_mm:g} mm goal "
            f"(U = {expanded(u_mm):.2f} mm; need U <= {goal_mm / MIN_TUR:.2f} mm)")
