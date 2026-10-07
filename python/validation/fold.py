"""Fold angle of plate L measured from images instead of a protractor.

Model: the two faces share the fold line (model X axis). Face H is the plane
z = 0; face V is face H rotated about X by phi = 180 deg - interior angle
(``plates._face_frame``). Each reference point's 2D position on its face comes
from the print (and the measured scale bars), so the only unknowns of a frame
are the plate pose (6) and the interior angle (1). They are fitted jointly to
the independently detected saddle points of both faces (``detect.locate_points``,
the same detector the analysis uses), minimising reprojection error.

A 1 deg change moves a point 64 mm above the fold by ~1.1 mm, i.e. a few
pixels at 0.5 m, while saddles are located to a few hundredths of a pixel, so
a single frame already constrains the angle to well under 0.1 deg. Bias comes
mostly from the intrinsics; viewing the plate from several directions during
the capture averages part of it out (see ``estimate_session``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

MIN_POINTS_PER_FACE = 4
K_BIAS_FLOOR_DEG = 0.1   # floor for the intrinsics-driven bias in the session result


@dataclass
class FoldFit:
    interior_deg: float
    sd_deg: float          # from the fit's covariance (this frame)
    rms_px: float
    n_points: int


def face_model(target, ids, interior_deg: float) -> np.ndarray:
    """(N, 3) model points for ``ids`` with face V folded to ``interior_deg``."""
    phi = math.radians(180.0 - interior_deg)
    ey_v = np.array([0.0, math.cos(phi), math.sin(phi)])
    uv = target.face_coords(ids)
    out = np.zeros((len(ids), 3))
    for k, pid in enumerate(ids):
        u, v = uv[k]
        out[k] = (u, v, 0.0) if target.point_face[pid] == "H" else u * np.array([1.0, 0, 0]) + v * ey_v
    return out


def fit_frame(target, detected: Dict[str, np.ndarray], K, dist,
              interior0: float = 90.0) -> Optional[FoldFit]:
    """Joint pose + interior-angle fit for one frame (``detected``: raw px)."""
    from scipy.optimize import least_squares
    ids = [i for i in target.ids() if i in detected]
    per_face = {f: sum(target.point_face[i] == f for i in ids) for f in ("H", "V")}
    if min(per_face.values()) < MIN_POINTS_PER_FACE:
        return None
    img = np.array([detected[i] for i in ids], float).reshape(-1, 2)
    K = np.asarray(K, float)
    dist = np.asarray(dist, float)
    ok, rvec, tvec = cv2.solvePnP(face_model(target, ids, interior0), img, K, dist,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None

    def resid(x):
        X = face_model(target, ids, x[6])
        proj = cv2.projectPoints(X, x[:3], x[3:6], K, dist)[0].reshape(-1, 2)
        return (proj - img).ravel()

    x0 = np.concatenate([np.ravel(rvec), np.ravel(tvec), [interior0]])
    sol = least_squares(resid, x0, x_scale=np.r_[np.full(3, 0.01), np.full(3, 0.001), 0.1],
                        method="lm")
    if not sol.success:
        return None
    r = sol.fun
    dof = max(1, len(r) - 7)
    s2 = float(r @ r) / dof
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        sd = float(math.sqrt(max(cov[6, 6], 0.0)))
    except np.linalg.LinAlgError:
        sd = float("nan")
    return FoldFit(float(sol.x[6]), sd, float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2 * 2))), len(ids))


def estimate_session(session, target, every: int = 1, search_radius: float = 5.0) -> dict:
    """Interior angle from every k-th stored image of a session.

    Returns the mean, the spread across frames, and a standard uncertainty
    ``u_deg`` = sqrt((sd / sqrt(n))^2 + K_BIAS_FLOOR_DEG^2). The floor is
    deliberate: frames of one session share the intrinsics, so their scatter
    does not show a calibration bias.
    """
    from .detect import locate_points
    K, dist = session.camera_matrix, session.dist_coeffs
    fits: List[FoldFit] = []
    recs = [r for r in session.records if r.frame is not None]
    for rec in recs[::max(1, every)]:
        loc = locate_points(session.read_gray(rec), target, K, dist, search_radius)
        f = fit_frame(target, loc.points, K, dist)
        if f is not None:
            fits.append(f)
    if not fits:
        return {"n_frames": 0, "interior_deg": float("nan"), "u_deg": float("nan")}
    a = np.array([f.interior_deg for f in fits])
    sd = float(a.std(ddof=1)) if len(a) > 1 else float(fits[0].sd_deg)
    return {"n_frames": len(fits), "interior_deg": float(a.mean()), "median_deg": float(np.median(a)),
            "sd_frames_deg": sd, "median_fit_sd_deg": float(np.median([f.sd_deg for f in fits])),
            "median_rms_px": float(np.median([f.rms_px for f in fits])),
            "u_deg": float(math.hypot(sd / math.sqrt(len(a)), K_BIAS_FLOOR_DEG))}
