"""Live setup assistant: how far the current view is from the planned condition.

The camera already sees the plate, so the viewing angle, distance and
centring are read from the plate pose (the raw solve, ``measured_pose``), not
from an inclinometer and a tape. This is only a positioning aid: the
analysis never uses these numbers as ground truth (it detects the plate on its
own), and the geometry actually achieved is logged per frame, so the report
can use it instead of the nominal condition.

Quantities (plate model frames, see ``plates``):

- ``distance_mm``: optical centre to the model origin (plate A: page centre;
  plate L: centre of the fold line). The protocol measures from the front
  glass with +-10 mm tolerance; the few mm difference is inside it.
- ``angle_deg`` (plate A): line of sight to the centre vs the plate normal.
- ``elevation_deg`` (plate L): line of sight below the plane of face H
  (90 = looking straight down on H, 45 = along the bisector).
- centring: image offset of the centre from the principal point, as a
  fraction of the half-width.

With a device IMU (camera built into a Surface) the same angles relative to
gravity are shown next to them; on a level plate the two must agree, which
also checks the table is level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

TOLERANCES = {"distance_mm": 10.0, "angle_deg": 2.0, "elevation_deg": 2.0}
LOOSE = {"distance_mm": 30.0, "angle_deg": 5.0, "elevation_deg": 5.0}   # B3: "~" values
CENTRE_TOL = 0.15


@dataclass
class Reading:
    distance_mm: float
    angle_deg: float
    elevation_deg: float
    centre_px: tuple
    centre_offset: float          # |offset| / half image width


@dataclass
class Row:
    name: str
    value: float
    nominal: Optional[float]
    tol: Optional[float]

    @property
    def ok(self) -> Optional[bool]:
        if self.nominal is None or not math.isfinite(self.value):
            return None
        return abs(self.value - self.nominal) <= self.tol


def measure(target, rvec, tvec, K, image_size) -> Reading:
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    t = np.asarray(tvec, float).ravel()
    d = float(np.linalg.norm(t))
    los = t / d                                          # camera -> plate centre
    face_main = "F" if "F" in target.faces else "H"
    n = R @ target.faces[face_main].normal               # plate normal, camera frame
    angle = math.degrees(math.acos(min(1.0, abs(float(los @ n)))))
    elevation = math.degrees(math.asin(float(np.clip(-(los @ n), -1.0, 1.0))))
    uv = (K @ t)[:2] / t[2] if t[2] > 0 else np.array([np.nan, np.nan])
    w = float(image_size[0])
    off = float(np.linalg.norm(uv - np.asarray(K)[:2, 2])) / (w / 2.0)
    return Reading(1000.0 * d, angle, elevation, (float(uv[0]), float(uv[1])), off)


def rows_for(condition: dict, r: Reading) -> List[Row]:
    loose = condition.get("test") == "B3"
    tol = LOOSE if loose else TOLERANCES
    rows = [Row("distance_mm", r.distance_mm, _num(condition.get("distance_mm")), tol["distance_mm"])]
    if "elevation_deg" in condition:
        rows.append(Row("elevation_deg", r.elevation_deg, _num(condition["elevation_deg"]),
                        tol["elevation_deg"]))
    else:
        rows.append(Row("angle_deg", r.angle_deg, _num(condition.get("angle_deg")), tol["angle_deg"]))
    rows.append(Row("centre_offset", r.centre_offset, 0.0, CENTRE_TOL))
    return rows


def all_ok(rows: List[Row]) -> bool:
    return all(row.ok is not False for row in rows) and any(row.ok for row in rows)


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_LABEL = {"distance_mm": ("distance", "mm", "{:.0f}"), "angle_deg": ("angle", "deg", "{:.1f}"),
          "elevation_deg": ("elevation", "deg", "{:.1f}"), "centre_offset": ("centring", "", "{:.2f}")}


def draw_panel(img, rows: List[Row], imu: Optional[dict] = None, imu_key: Optional[str] = None,
               origin=(10, 45)) -> None:
    """Draw the setup panel in place (BGR image)."""
    x, y = origin
    lines = []
    for row in rows:
        label, unit, fmt = _LABEL.get(row.name, (row.name, "", "{:.2f}"))
        val = fmt.format(row.value) if math.isfinite(row.value) else "--"
        target = "" if row.nominal is None else f"  target {fmt.format(row.nominal)} +-{fmt.format(row.tol)}"
        colour = (160, 160, 160) if row.ok is None else ((60, 200, 60) if row.ok else (40, 40, 230))
        lines.append((f"{label:10s} {val:>7s} {unit:3s}{target}", colour))
    if imu:
        val = imu.get(imu_key) if imu_key else None
        txt = f"IMU {imu.get('camera')}: {imu_key.split('_')[0] if imu_key else ''} " + (
            f"{val:.1f} deg" if isinstance(val, float) else "--")
        txt += f" | roll {imu.get('roll_deg', float('nan')):+.1f} | {'still' if imu.get('still') else 'MOVING'}"
        lines.append((txt, (255, 200, 80) if imu.get("still") else (40, 40, 230)))
    h = 22 * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - 18), (x + 470, y - 18 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (txt, colour) in enumerate(lines):
        cv2.putText(img, txt, (x, y + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def imu_key_for(condition: dict) -> str:
    """IMU quantity comparable with the condition (valid for a level plate)."""
    return "elevation_deg" if "elevation_deg" in condition else "axis_from_down_deg"
