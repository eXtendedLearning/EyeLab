"""Validation metrics.

- detection: measured / displayed / coasted rates, lock states;
- pose jitter: raw (``measured``) and drawn (``pose``) poses, mm and deg;
- overlay registration error: EyeLab's projected point vs the independently
  detected printed point, in px and in mm on the face plane;
- overlay jitter: per-node spread of where EyeLab puts a point on the face;
- step error: known jig translation vs estimated translation.

Lengths are metres internally and converted to mm only in ``*_mm`` outputs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera import apply_h, undistort
from .detect import apply_overrides, locate_points
from .target import M_TO_MM

NAN = float("nan")


# ── pose ──
def rvecs_to_matrices(rvecs) -> np.ndarray:
    return np.stack([cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0] for r in rvecs])


def chordal_mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rotation_angle_deg(Ra, Rb) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_jitter(rvecs, tvecs) -> dict:
    """Dispersion of a static pose. ``tvecs`` in metres; outputs in mm / deg."""
    t = np.asarray(tvecs, float).reshape(-1, 3) * M_TO_MM
    if len(t) < 2:
        raise ValueError("need at least 2 posed frames")
    Rs = rvecs_to_matrices(rvecs)
    Rm = chordal_mean_rotation(Rs)
    return {"n": len(t), "t_mean_mm": t.mean(axis=0), "t_sd_xyz_mm": t.std(axis=0, ddof=1),
            "t_radial_dev_mm": np.linalg.norm(t - t.mean(axis=0), axis=1),
            "rot_dev_deg": np.array([rotation_angle_deg(Rm, R) for R in Rs])}


def detection_stats(records) -> dict:
    n = len(records)
    if n == 0:
        return {"n_frames": 0}
    measured = [r for r in records if r.measured is not None]
    shown = [r for r in records if r.pose is not None]
    coasted = [r for r in shown if r.pose.coasted]
    return {"n_frames": n, "measured_rate": len(measured) / n, "displayed_rate": len(shown) / n,
            "coasted_rate": len(coasted) / n,
            "lock_states": dict(Counter(str(r.lock_state) for r in records)),
            "markers_in_measured_pose": {str(k): v for k, v in
                                         sorted(Counter(r.n_markers_measured for r in measured).items())}}


def step_error(session_a, session_b, known_step_mm: float, which: str = "measured") -> dict:
    """Estimated minus true displacement between two static sessions of the same
    target moved by a jig-known pure translation."""
    ta = session_a.poses(which)[1] * M_TO_MM
    tb = session_b.poses(which)[1] * M_TO_MM
    est = float(np.linalg.norm(tb.mean(axis=0) - ta.mean(axis=0)))
    return {"which": which, "known_mm": known_step_mm, "estimated_mm": est,
            "error_mm": est - known_step_mm, "n_a": len(ta), "n_b": len(tb)}


def projected_jitter_px(records, which: str = "projected") -> np.ndarray:
    """Radial deviation (px) of each node's projection from its session mean."""
    by: Dict[str, List[np.ndarray]] = {}
    for r in records:
        for k, v in getattr(r, which).items():
            by.setdefault(k, []).append(v)
    dev = [np.linalg.norm(np.array(v) - np.mean(v, axis=0), axis=1) for v in by.values() if len(v) > 1]
    return np.concatenate(dev) if dev else np.zeros(0)


# ── overlay registration ──
@dataclass
class RegistrationRow:
    frame: int
    point_id: str
    face: str
    lock_state: str
    coasted: bool
    n_markers: int          # markers in this frame's measured pose
    detected: bool
    seed_offset_px: float
    err_px: float           # |EyeLab drawn pose projection - detected|, float
    err_px_raw: float       # same for the raw measured pose
    err_px_drawn: float     # same for the integer pixels actually drawn
    err_mm: float           # on the face plane
    err_mm_raw: float
    err_du_mm: float        # signed EyeLab - truth, face axes
    err_dv_mm: float
    eyelab_u_mm: float      # where EyeLab puts the point on the face
    eyelab_v_mm: float


FIELDS = tuple(RegistrationRow.__dataclass_fields__)


def _to_face(H, uv_raw, K, dist) -> np.ndarray:
    return apply_h(np.linalg.inv(H), undistort([uv_raw], K, dist))[0]


def registration_errors(session, target, search_radius: float = 5.0, every: int = 1,
                        subpix_win: int = 3, overrides: Optional[Dict[int, dict]] = None
                        ) -> Tuple[List[RegistrationRow], dict]:
    """``overrides``: reviewed ground truth of this session, ``{frame index: {point id:
    raw px or None}}`` (``gt_review.load_overrides``); it replaces the automatic saddles
    of those frames before the face homographies are fitted."""
    K, dist = session.camera_matrix, session.dist_coeffs
    rows: List[RegistrationRow] = []
    qa = {"homography_residual_px": [], "frames_analysed": 0, "frames_no_marker": 0,
          "points_overridden": 0}
    posed = [r for r in session.records
             if r.frame is not None and (r.pose is not None or r.measured is not None)]
    for rec in posed[::max(1, every)]:
        gray = session.read_gray(rec)
        loc = locate_points(gray, target, K, dist, search_radius, subpix_win)
        if overrides and rec.index in overrides:
            apply_overrides(loc, target, K, dist, overrides[rec.index])
            qa["points_overridden"] += len(overrides[rec.index])
        qa["frames_analysed"] += 1
        if not loc.markers:
            qa["frames_no_marker"] += 1
        for res in loc.residual_px.values():
            qa["homography_residual_px"].extend(res.tolist())
        for pid in target.ids():
            face = target.point_face[pid]
            det = loc.points.get(pid)
            H = loc.homographies.get(face)
            truth = target.face_coords([pid])[0]

            def px(d):
                return float(np.linalg.norm(det - d[pid])) if det is not None and pid in d else NAN

            def on_face(d):
                if H is None or pid not in d:
                    return None
                return _to_face(H, d[pid], K, dist)

            e = on_face(rec.projected)
            e_raw = on_face(rec.projected_raw)
            d_mm = (e - truth) * M_TO_MM if e is not None and det is not None else (NAN, NAN)
            rows.append(RegistrationRow(
                rec.index, pid, face, str(rec.lock_state),
                bool(rec.pose.coasted) if rec.pose else False, rec.n_markers_measured,
                det is not None, loc.seed_offset_px.get(pid, NAN),
                px(rec.projected), px(rec.projected_raw), px(rec.drawn),
                float(np.hypot(*d_mm)),
                float(np.linalg.norm(e_raw - truth) * M_TO_MM) if e_raw is not None and det is not None else NAN,
                float(d_mm[0]), float(d_mm[1]),
                float(e[0] * M_TO_MM) if e is not None else NAN,
                float(e[1] * M_TO_MM) if e is not None else NAN))
    return rows, qa


def overlay_jitter_mm(rows: List[RegistrationRow]) -> np.ndarray:
    """Radial deviation (mm) of EyeLab's on-face position of each node from its mean."""
    by: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        if np.isfinite(r.eyelab_u_mm):
            by.setdefault(r.point_id, []).append((r.eyelab_u_mm, r.eyelab_v_mm))
    dev = [np.linalg.norm(np.array(v) - np.mean(v, axis=0), axis=1) for v in by.values() if len(v) > 1]
    return np.concatenate(dev) if dev else np.zeros(0)


def per_node(rows: List[RegistrationRow]) -> List[dict]:
    """Bias (mean error vector) and spread per node, for systematic-error maps."""
    out = []
    ids = sorted({r.point_id for r in rows}, key=lambda k: (len(k), k))
    for pid in ids:
        rs = [r for r in rows if r.point_id == pid and np.isfinite(r.err_mm)]
        if not rs:
            continue
        du = np.array([r.err_du_mm for r in rs])
        dv = np.array([r.err_dv_mm for r in rs])
        e = np.array([r.err_mm for r in rs])
        out.append({"point_id": pid, "face": rs[0].face, "n": len(rs),
                    "bias_du_mm": float(du.mean()), "bias_dv_mm": float(dv.mean()),
                    "bias_mm": float(np.hypot(du.mean(), dv.mean())),
                    "rmse_mm": float(np.sqrt(np.mean(e ** 2))), "p95_mm": float(np.percentile(e, 95))})
    return out


