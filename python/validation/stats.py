"""Descriptive statistics (THEORY §9.2: report RMSE with median, p95 and max).

Frames of one static session are not independent repetitions of the
measurement, so no per-frame confidence interval is reported. Between-pose
variability comes from repositioned sessions (``report.aggregate``), where the
session is the unit.
"""
from __future__ import annotations

import numpy as np


def summarize(values) -> dict:
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n": 0}
    return {"n": n, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else float("nan"),
            "rmse": float(np.sqrt(np.mean(v ** 2))), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def across_units(values) -> dict:
    """Mean and SD of per-session values (the repositioned-session unit)."""
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": len(v), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}
