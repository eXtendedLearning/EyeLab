"""T6.11: SPAAM taps on the Beam Pro, and the per-eye eye-display model fitted from them.

Each tap pairs a crosshair fixed in one eye's display with a real point: the centre of the ChArUco board's middle square,
whose room position comes from the trusted board poses of the two seconds before the tap, carried into the room as the
overlay is (``b3a.target_in_room``). With the head pose at the tap, the point is known in the head frame, then in the
frame of the SDK's eye for that display (``display_left_from_head`` / ``display_right_from_head`` in run.json).

The crosshair sits along (u, v, 1) in that frame, u = tan(x), v = tan(y). The SDK's model predicts that the point is seen
there when P_x / P_z = u and P_y / P_z = v. The eye-display model fitted here (Cutolo et al. 2020: the eye and display as
an off-axis pinhole with the eye's nodal point as its centre) has:

* the true eye centre C in the SDK eye's frame (3);
* the display's virtual image plane at z = D, as w = 1 / D (1);
* the display's angular scale s and its centre offset (px, py) in tan units (3).

A display pixel (u, v) is the point X = D (s u + px, s v + py, 1). The wearer aligns when C, X and P are collinear:

    X / D = w C + (1 - w C_z) (P - C) / (P_z - C_z)   (x and y)

so u = (X_x / D - px) / s. With C = 0, s = 1, p = 0 this is the SDK's model at any D. A lateral C gives an angular error
C_x (1 / d - 1 / D): zero at the display's own distance, growing at arm's length, opposite for the two eyes when the
eye spacing is wrong. That is T6.10 run 1.

A plain DLT (3x4, 11 parameters) is fitted too when there are 6 taps or more, as the textbook SPAAM, and its centre of
projection is reported as a cross-check.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .beampro import load_csv

BOARD_CENTRE = np.array([0.0625, 0.0875, 0.0])   # middle square of board25 (marker 8), board frame, metres
WINDOW_S = 2.0
NAMES = ["cx_mm", "cy_mm", "cz_mm", "inv_D_per_m", "scale", "px_deg", "py_deg"]


def pose_from_meta(s: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """'ok; p x y z; q x y z w' (Unity-handed, metres) as (R, t, ok)."""
    from .b3a import quat_matrix
    try:
        parts = {seg.strip().split(" ")[0]: [float(v) for v in seg.strip().split(" ")[1:]] for seg in s.split(";")[1:]}
        return quat_matrix(*parts["q"]), np.array(parts["p"]), s.startswith("ok")
    except (KeyError, ValueError, TypeError, AttributeError):
        return np.eye(3), np.zeros(3), False


def predict(params: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Display coordinates (u, v) = tan of the crosshair direction at which points ``P`` (N x 3, SDK eye frame) are
    seen aligned, for the model ``params`` (C in metres, w = 1/D, s, px, py in tan units)."""
    C, w, s, p = params[:3], params[3], params[4], params[5:7]
    P = np.atleast_2d(P)
    lam = (P[:, :2] - C[:2]) / (P[:, 2:3] - C[2])
    xd = w * C[:2] + (1 - w * C[2]) * lam
    return (xd - p) / s


def _unpack(x: np.ndarray, fixed_w: Optional[float]) -> np.ndarray:
    if fixed_w is None:
        return x
    return np.concatenate([x[:3], [fixed_w], x[3:]])


def fit_model(P: np.ndarray, uv: np.ndarray, fixed_w: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Least squares on the display coordinates. Returns (params, rms of the angular residual in degrees)."""
    from scipy.optimize import least_squares
    x0 = np.array([0, 0, 0, 0.25, 1, 0, 0], float)
    if fixed_w is not None:
        x0 = np.delete(x0, 3)

    def res(x):
        return (predict(_unpack(x, fixed_w), P) - uv).ravel()
    sol = least_squares(res, x0, x_scale="jac")
    params = _unpack(sol.x, fixed_w)
    r = (predict(params, P) - uv)
    return params, float(np.degrees(np.sqrt(np.mean(np.sum(r * r, axis=1)))))


def dlt_centre(P: np.ndarray, uv: np.ndarray) -> Optional[np.ndarray]:
    """Textbook SPAAM: the 3x4 G with (u, v, 1) ~ G (P, 1), and its centre of projection (G's null space)."""
    if len(P) < 6:
        return None
    A = []
    for (X, Y, Z), (u, v) in zip(P, uv):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    G = np.linalg.svd(np.array(A))[2][-1].reshape(3, 4)
    c = np.linalg.svd(G)[2][-1]
    return c[:3] / c[3] if abs(c[3]) > 1e-12 else None


def tap_points(run: Path, set_dir: Optional[Path] = None) -> Tuple[List[dict], Dict[str, str]]:
    """Each good tap with the real point in the SDK eye's frame. Returns (taps, notes)."""
    from . import b3a
    from .beampro import load_json
    run = Path(run)
    meta = load_json(run)
    rgb = b3a.rgb_from_head(meta)[:2]
    eyes = {"L": pose_from_meta(meta.get("display_left_from_head", "")),
            "R": pose_from_meta(meta.get("display_right_from_head", ""))}
    events = load_csv(run / "events.csv")
    bad = {e["detail"].split(":")[-1] for e in events if e.get("event") == "tap_bad"}
    heads = {h["seq"]: h for h in load_csv(run / "head.csv") if h.get("ok") == "1"}
    live = [r for r in load_csv(run / "live.csv")
            if r.get("accepted") == "1" and r["pose_ok"] == "1" and r.get("target") == "board25" and r["seq"] in heads]
    cu = b3a.F_CV @ BOARD_CENTRE
    room = []
    for r in live:
        rv, tv = b3a._pose(r)
        R, t = b3a.target_in_room(rv, tv, b3a.head_unity(heads[r["seq"]]), rgb)
        room.append((int(r["cam_ts"]), t + R @ cu, float(np.linalg.norm(np.asarray(tv, float)))))
    taps = []
    for row in load_csv(run / "spaam.csv"):
        if row["tap"] in bad:
            continue
        tn = int(row["mono_ns"])
        near = [(p, d) for ts, p, d in room if tn - WINDOW_S * 1e9 <= ts <= tn]
        if len(near) < 3:
            taps.append({"tap": row["tap"], "step": row["step"], "eye": row["eye"], "ok": False, "n": len(near)})
            continue
        pts = np.array([p for p, _ in near])
        Pw = np.median(pts, axis=0)
        RH = b3a.quat_matrix(*[float(row[k]) for k in ("head_qx", "head_qy", "head_qz", "head_qw")])
        tH = np.array([float(row[k]) for k in ("head_px", "head_py", "head_pz")])
        Ph = RH.T @ (Pw - tH)
        RE, tE, _ = eyes[row["eye"]]
        Pe = RE.T @ (Ph - tE)
        uv = np.array([math.tan(math.radians(float(row["deg_x"]))), math.tan(math.radians(float(row["deg_y"])))])
        taps.append({"tap": row["tap"], "step": row["step"], "eye": row["eye"], "ok": True, "n": len(near),
                     "spread_mm": float(np.linalg.norm(pts.std(axis=0)) * 1000), "P": Pe, "uv": uv,
                     "dist_mm": float(np.linalg.norm(Pe) * 1000)})
    notes = {"eye_L": meta.get("display_left_from_head", "?"), "eye_R": meta.get("display_right_from_head", "?"),
             "bad": ", ".join(sorted(bad, key=int)) if bad else "none"}
    return taps, notes


def _fmt(params: np.ndarray) -> str:
    c = params[:3] * 1000
    D = 1 / params[3] if params[3] > 1e-6 else float("inf")
    return (f"eye centre ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}) mm from the SDK's, display at {D:.2f} m, scale "
            f"{params[4]:.3f}, centre ({math.degrees(params[5]):+.2f}, {math.degrees(params[6]):+.2f}) deg")


def spaam_report(run: Path) -> List[str]:
    from .beampro import load_json
    run = Path(run)
    meta = load_json(run)
    if meta.get("variant") != "spaam":
        return []
    out = [f"   spaam: {meta.get('spaam', '?')}",
           f"   SDK eyes: left {meta.get('display_left_from_head', '?')}; right {meta.get('display_right_from_head', '?')}",
           f"   XR eye positions: {meta.get('xr_eye_positions', '?')}; stereo separation {meta.get('camera_stereo_separation', '?')}",
           f"   projections: left {meta.get('display_left_projection', '?')[:80]}...",
           "   eye check: " + "; ".join(e["detail"] for e in load_csv(run / "events.csv") if e.get("event") == "rating")]
    if not (run / "spaam.csv").exists():
        return out + ["   no taps"]
    taps, notes = tap_points(run)
    good = [t for t in taps if t["ok"]]
    out.append(f"   taps: {len(taps)} (bad, dropped: {notes['bad']}); with the board seen in the 2 s before: {len(good)}")
    eyes_fit = {}
    for eye in ("L", "R"):
        sel = [t for t in good if t["eye"] == eye]
        if len(sel) < 3:
            out.append(f"   {eye}: {len(sel)} taps, too few")
            continue
        P = np.array([t["P"] for t in sel])
        uv = np.array([t["uv"] for t in sel])
        sdk = P[:, :2] / P[:, 2:3]
        e = sdk - uv   # the SDK draws P at sdk; the wearer sees P at uv: + = the drawing right of / above the real point
        out.append(f"   {eye} eye, {len(sel)} taps: where the SDK would draw the point against where it is (mm at the target; "
                   f"+x = drawing right of it, +y = above)")
        for cm in sorted({t['step'].split('_')[0] for t in sel}):
            idx = [i for i, t in enumerate(sel) if t["step"].startswith(cm + "_")]
            mm = np.array([e[i] * P[i, 2] * 1000 for i in idx])
            out.append(f"      {cm}: {len(idx)} taps at {np.median([sel[i]['dist_mm'] for i in idx]):4.0f} mm: "
                       f"mean ({mm[:, 0].mean():+6.1f}, {mm[:, 1].mean():+6.1f}) mm, spread {np.linalg.norm(mm.std(axis=0)):5.1f} mm; "
                       f"board point spread {np.median([sel[i]['spread_mm'] for i in idx]):.1f} mm")
        out.append(f"      SDK model rms {math.degrees(float(np.sqrt(np.mean(np.sum(e * e, axis=1))))):.2f} deg")
        fits = []
        if len(sel) >= 7:
            fits.append(("free display distance", fit_model(P, uv)))
        if len(sel) >= 6:
            fits.append(("display at 4 m", fit_model(P, uv, fixed_w=0.25)))
        for name, (params, rms) in fits:
            out.append(f"      fit, {name}: {_fmt(params)}; rms {rms:.2f} deg")
        if fits:
            eyes_fit[eye] = fits[-1][0]
        c = dlt_centre(P, uv)
        if c is not None:
            out.append(f"      DLT centre of projection: ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, {c[2] * 1000:+.1f}) mm from the SDK eye")
    if len(eyes_fit) == 2:
        RL, tL, _ = pose_from_meta(meta.get("display_left_from_head", ""))
        RR, tR, _ = pose_from_meta(meta.get("display_right_from_head", ""))
        cl, cr = tL + RL @ eyes_fit["L"][:3], tR + RR @ eyes_fit["R"][:3]
        out.append(f"   eye spacing: SDK {np.linalg.norm(tL - tR) * 1000:.1f} mm; fitted (display at 4 m) "
                   f"{np.linalg.norm(cl - cr) * 1000:.1f} mm")
    return out
