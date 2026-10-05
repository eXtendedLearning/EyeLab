#!/usr/bin/env python3
"""Stage S3 analysis — what an IMU recording says, with no hardware attached.

Pure functions over a sample table (``xreal_imu.CSV_COLUMNS``). Nothing here
assumes units or axis conventions; it *measures* the evidence for them:

* **Timing** from the device timestamps: rate, interval jitter, gaps and
  non-monotonic samples. Delivery latency cannot be measured without a shared
  clock, but its *jitter* can: the spread of ``host_ns - t_ns`` around its
  trend is how unevenly samples reach the process (clock drift is removed by
  a linear fit before taking the spread).
* **At rest**: gyro bias and noise per axis, the mean accelerometer vector,
  its magnitude and therefore its unit (``|a| ~ 9.81`` means m/s^2, ``~ 1``
  means g), and which body axis gravity lies along.
* **Over a known turn**: the bias-corrected gyro integral per axis. A 90 deg
  turn integrates to ~1.571 if the unit is rad/s and ~90 if it is deg/s; the
  dominant axis and its sign map the turn onto the device frame.

Thresholds used to *label* results are named constants with the reason for
their value; nothing here decides pass/fail for a stage.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

STANDARD_GRAVITY = 9.80665

#: |a| at rest within this relative band of 1 g in some unit names that unit.
#: 10 % absorbs sensor scale error and local gravity (9.78-9.83 m/s^2).
GRAVITY_BAND = 0.10

#: An interval longer than this multiple of the median counts as a gap.
GAP_FACTOR = 1.5

#: Seconds at the start of a turn recording assumed still, for the gyro bias.
STILL_LEAD_S = 2.0

#: A turn integral within this relative band of the expected angle names the unit.
TURN_BAND = 0.25

AXES = ("x", "y", "z")


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    for k in ("host_ns", "t_ns", "t_dev_ns"):
        if k in cols:
            cols[k] = np.array([int(r[k]) for r in rows], dtype=np.int64)
    return cols


def _vec(cols: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return np.column_stack([cols[prefix + a] for a in AXES])


def timing(cols: dict[str, np.ndarray], clock: str = "t_ns") -> dict[str, Any]:
    t = cols[clock].astype(np.int64)
    n = len(t)
    out: dict[str, Any] = {"clock": clock, "samples": int(n)}
    if n < 3:
        return out
    dt = np.diff(t)
    positive = dt[dt > 0]
    out["non_monotonic"] = int(np.sum(dt <= 0))
    if positive.size == 0:
        return out
    median = float(np.median(positive))
    span_s = (t[-1] - t[0]) / 1e9
    gaps = positive[positive > GAP_FACTOR * median]
    out.update(
        span_s=round(span_s, 3),
        rate_hz=round((n - 1) / span_s, 2) if span_s > 0 else None,
        dt_median_ms=round(median / 1e6, 4),
        dt_std_ms=round(float(np.std(positive)) / 1e6, 4),
        dt_p99_ms=round(float(np.percentile(positive, 99)) / 1e6, 4),
        dt_max_ms=round(float(positive.max()) / 1e6, 3),
        gaps=int(gaps.size),
        missing_estimate=int(np.sum(np.round(gaps / median) - 1)),
    )
    return out


def delivery_jitter(cols: dict[str, np.ndarray]) -> dict[str, Any]:
    """Spread of host arrival time against device time, drift removed."""
    t = cols["t_ns"].astype(np.float64)
    host = cols["host_ns"].astype(np.float64)
    if len(t) < 3 or np.ptp(t) == 0:
        return {}
    ts = (t - t[0]) / 1e9
    lag_ms = (host - t) / 1e6
    slope, intercept = np.polyfit(ts, lag_ms, 1)
    resid = lag_ms - (slope * ts + intercept)
    return {
        "clock_drift_ppm": round(float(slope) * 1e3, 2),   # ms per s -> ppm
        "jitter_std_ms": round(float(np.std(resid)), 4),
        "jitter_p99_ms": round(float(np.percentile(np.abs(resid), 99)), 4),
        "jitter_max_ms": round(float(np.abs(resid).max()), 3),
    }


def infer_accel_unit(magnitude: float) -> Optional[str]:
    for unit, g in (("m/s^2", STANDARD_GRAVITY), ("g", 1.0)):
        if abs(magnitude - g) <= GRAVITY_BAND * g:
            return unit
    return None


def rest(cols: dict[str, np.ndarray]) -> dict[str, Any]:
    gyro, acc = _vec(cols, "g"), _vec(cols, "a")
    mean_acc = acc.mean(axis=0)
    magnitude = float(np.linalg.norm(mean_acc))
    dominant = int(np.argmax(np.abs(mean_acc)))
    up_sign = "+" if mean_acc[dominant] > 0 else "-"
    return {
        "gyro_bias": [round(float(v), 6) for v in gyro.mean(axis=0)],
        "gyro_std": [round(float(v), 6) for v in gyro.std(axis=0)],
        "accel_mean": [round(float(v), 5) for v in mean_acc],
        "accel_std": [round(float(v), 5) for v in acc.std(axis=0)],
        "accel_magnitude": round(magnitude, 5),
        "accel_unit": infer_accel_unit(magnitude),
        # Accelerometers report specific force: at rest it points *up*.
        "gravity_reaction_axis": f"{up_sign}{AXES[dominant]}",
        "gravity_alignment": round(abs(float(mean_acc[dominant])) / magnitude, 4)
        if magnitude > 0 else None,
    }


def turn(cols: dict[str, np.ndarray], expected_deg: float = 90.0,
         still_lead_s: float = STILL_LEAD_S) -> dict[str, Any]:
    """Integrate the bias-corrected gyro over a recording of one known turn."""
    t = cols["t_ns"].astype(np.float64) / 1e9
    gyro = _vec(cols, "g")
    if len(t) < 3:
        return {}
    still = t - t[0] < still_lead_s
    if still.sum() < 10:
        return {"error": f"fewer than 10 samples in the first {still_lead_s} s"}
    bias = gyro[still].mean(axis=0)
    dt = np.diff(t, prepend=t[0])
    angle = ((gyro - bias) * dt[:, None]).sum(axis=0)
    dominant = int(np.argmax(np.abs(angle)))
    value = float(angle[dominant])
    unit = None
    for name, expected in (("rad/s", math.radians(expected_deg)), ("deg/s", expected_deg)):
        if abs(abs(value) - expected) <= TURN_BAND * expected:
            unit = name
    share = abs(value) / float(np.linalg.norm(angle)) if np.linalg.norm(angle) else 0.0
    return {
        "bias_used": [round(float(v), 6) for v in bias],
        "integral": [round(float(v), 4) for v in angle],
        "dominant_axis": f"{'+' if value > 0 else '-'}{AXES[dominant]}",
        "dominant_share": round(share, 3),
        "expected_deg": expected_deg,
        "gyro_unit": unit,
        "degrees_if_rad_s": round(math.degrees(value), 1),
    }


def analyse(cols: dict[str, np.ndarray], mode: str = "rest",
            expected_deg: float = 90.0) -> dict[str, Any]:
    if not cols:
        return {"error": "no samples"}
    out: dict[str, Any] = {
        "timing": timing(cols, "t_ns"),
        "timing_device_clock": timing(cols, "t_dev_ns"),
        "delivery": delivery_jitter(cols),
    }
    if mode == "rest":
        out["rest"] = rest(cols)
    else:
        out["turn"] = turn(cols, expected_deg)
    return out
