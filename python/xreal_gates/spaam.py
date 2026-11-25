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

BOARD_CENTRE = np.array([0.0625, 0.0875, 0.0])   # middle square of board25 (marker 8) at the nominal 25 mm, board frame, metres
#: The Eye camera's pose in the head on this One Pro + Eye, as every run from 0.4.012 logs it. T6.11 run 1 (variant
#: spaam, 0.4.014) did not log it, and its first analysis fell back to identity: a 1.1 deg pitch and a 28 mm lever arm
#: left out, which made its taps disagree with T6.12's by about 17 mm (2026-11-23).
BEAMPRO_RGB_FROM_HEAD = "ok; p 0.00127746048 0.00535731856 0.0281137042; q -0.00971239 -0.005232626 -0.00308899628 0.9999344"


def board_centre(meta: dict) -> np.ndarray:
    """The tap point in the board frame (metres, OpenCV), as the run's set placed it: from 0.4.016 the device logs
    ``board_centre_m`` (marker 8's centre at the printed size); earlier runs posed the board at the nominal size."""
    try:
        return np.array([float(v) for v in str(meta["board_centre_m"]).split()[:3]])
    except (KeyError, ValueError):
        return BOARD_CENTRE.copy()


def board_scale(meta: dict) -> float:
    """The factor that re-poses a run's board25 at the board's true printed size. A run's set posed it at some square
    size, which the logged tap point shows (marker 8's centre is 2.5 squares across); poses scale with the square. Runs
    before 0.4.017 posed it at 25 mm (nominal, to 0.4.015) or 25.97 mm (0.4.016), against 34.17 mm printed (2026-11-24),
    so their real points sat 1.37x or 1.32x too near the camera."""
    from gui_common import CHARUCO_SQUARE_PRINTED_M
    used = float(board_centre(meta)[0]) / 2.5
    return CHARUCO_SQUARE_PRINTED_M / used if used > 0 else 1.0


def rgb_meta(meta: dict) -> dict:
    """``meta`` with the device's ``rgb_from_head`` filled in when the run did not log it (T6.11 run 1)."""
    return meta if "rgb_from_head" in meta else dict(meta, rgb_from_head=BEAMPRO_RGB_FROM_HEAD.replace("ok;", "ok (filled in);", 1))
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


SDK = np.array([0, 0, 0, 0.25, 1, 0, 0], float)
# which parameters a model frees: "eye" = the eye centre only (the SDK's display kept), "display" = the display's scale
# and centre only (the SDK's eye kept; the device's model from 0.4.017), "full" = eye, scale and centre, "free_d" = all
# seven (the display distance too; needs more depth range than T6.11 run 1 had)
MODELS = {"eye": [0, 1, 2], "display": [4, 5, 6], "full": [0, 1, 2, 4, 5, 6], "free_d": [0, 1, 2, 3, 4, 5, 6]}
#: The model the device fits and draws (EyeModel.Free). T6.13 (2026-11-24): with the board at its true size the SDK's eye
#: positions are right and a fixed angle and scale per eye is left; this predicts a fresh tap, in the same wearing or
#: another, to about 3 mm (the eye centre alone 5 mm, the SDK 13 mm).
DEVICE_MODEL = "display"


def fit_model(P: np.ndarray, uv: np.ndarray, fixed_w: Optional[float] = None, model: str = "") -> Tuple[np.ndarray, float]:
    """Least squares on the display coordinates. Returns (params, rms of the angular residual in degrees).
    ``model`` picks the free parameters (``MODELS``); by default "full" with ``fixed_w`` and "free_d" without."""
    from scipy.optimize import least_squares
    model = model or ("full" if fixed_w is not None else "free_d")
    free = MODELS[model]
    base = SDK.copy()
    if fixed_w is not None:
        base[3] = fixed_w

    def unpack(x):
        q = base.copy()
        q[free] = x
        return q

    def res(x):
        return (predict(unpack(x), P) - uv).ravel()
    sol = least_squares(res, base[free], x_scale="jac")
    params = unpack(sol.x)
    r = (predict(params, P) - uv)
    return params, float(np.degrees(np.sqrt(np.mean(np.sum(r * r, axis=1)))))


def leave_one_out(P: np.ndarray, uv: np.ndarray, model: str, fixed_w: float = 0.25) -> Optional[np.ndarray]:
    """Each tap predicted by the model fitted on the others: the error (mm at the target, x and y) a fresh tap would
    have. This is the calibration's quality on data it has not seen, which an in-sample rms overstates."""
    k = len(MODELS[model])
    if len(P) * 2 - 2 < k + 2:
        return None
    out = []
    for i in range(len(P)):
        m = np.arange(len(P)) != i
        params, _ = fit_model(P[m], uv[m], fixed_w=fixed_w, model=model)
        out.append((predict(params, P[i:i + 1])[0] - uv[i]) * P[i, 2] * 1000)
    return np.array(out)


def trusted_board(r: dict) -> bool:
    """The device's trust, or the overlay variants' board rule (12+ markers, rms <= 15 px, in front of the camera),
    which T6.11 run 1 did not apply on the device: its 60 cm taps had 16-marker poses marked untrusted."""
    if r.get("pose_ok") != "1" or r.get("target") != "board25":
        return False
    if r.get("accepted") == "1":
        return True
    try:
        return int(r["board_markers"]) >= 12 and float(r["rms_px"]) <= 15 and float(r["t2"]) > 0
    except (KeyError, ValueError):
        return False


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


def repose_set(set_dir: Optional[Path] = None) -> dict:
    """The camera and board a run's board poses are re-solved with (``tap_points``): a frame set's K, dist, board25 (at
    the printed size since 0.4.017) and calibration name, the key of the device's tap store."""
    from . import b3a
    s = b3a.load_set(set_dir or b3a.DEFAULT_SET)
    if s is None:
        raise FileNotFoundError(f"no set.json in {set_dir or b3a.DEFAULT_SET}")
    t = b3a.target_of(s, "board25")
    return {"K": np.asarray(s["K"], float).reshape(3, 3), "dist": np.asarray(s["dist"], float),
            "board": dict(zip(t["ids"], np.asarray(t["corners"], float).reshape(len(t["ids"]), 4, 3))),
            "camera": str(s.get("calibration", "?"))}


def _resolve(rows: List[dict], run: Path, rs: dict, k: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Each row's board pose re-solved from its logged corners with ``rs``'s camera and board, starting from the logged
    pose scaled to the printed size so the same planar solution is kept."""
    import cv2
    from . import b3a
    want = {r["seq"] for r in rows}
    corners = b3a._corners_by_frame([c for c in load_csv(run / "live_corners.csv") if c["seq"] in want], "seq")
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for r in rows:
        cs = corners.get(r["seq"], {})
        ids = [i for i in cs if i in rs["board"]]
        if len(ids) < 2:
            continue
        rv, tv = b3a._pose(r)
        ok, rv2, tv2 = cv2.solvePnP(np.concatenate([rs["board"][i] for i in ids]), np.concatenate([cs[i] for i in ids]),
                                    rs["K"], rs["dist"], np.asarray(rv, float).reshape(3, 1).copy(),
                                    (np.asarray(tv, float) * k).reshape(3, 1).copy(), useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            out[r["seq"]] = (rv2.reshape(3), tv2.reshape(3))
    return out


def tap_points(run: Path, set_dir: Optional[Path] = None, repose: Optional[dict] = None) -> Tuple[List[dict], Dict[str, str]]:
    """Each good tap with the real point in the SDK eye's frame, the board re-posed at its true printed size
    (``board_scale``). With ``repose`` (``repose_set``), each board pose is re-solved from the run's logged corners
    with that camera model instead: T6.12 run 3 (2026-11-25) found that taps posed under two camera models differ by
    the models' principal points, about 0.5 deg. Returns (taps, notes)."""
    from . import b3a
    from .beampro import load_json
    run = Path(run)
    meta = load_json(run)
    rgb = b3a.rgb_from_head(rgb_meta(meta))[:2]
    eyes = {"L": pose_from_meta(meta.get("display_left_from_head", "")),
            "R": pose_from_meta(meta.get("display_right_from_head", ""))}
    events = load_csv(run / "events.csv")
    bad = {e["detail"].split(":")[-1] for e in events if e.get("event") == "tap_bad"}
    heads = {h["seq"]: h for h in load_csv(run / "head.csv") if h.get("ok") == "1"}
    live = [r for r in load_csv(run / "live.csv") if trusted_board(r) and r["seq"] in heads]
    k = board_scale(meta)
    cu = b3a.F_CV @ (board_centre(meta) * k)
    solved = _resolve(live, run, repose, k) if repose is not None else None
    room = []
    for r in live:
        if solved is not None:
            if r["seq"] not in solved:
                continue
            rv, tv = solved[r["seq"]]
        else:
            rv, tv = b3a._pose(r)
            tv = np.asarray(tv, float) * k
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
        taps.append({"tap": row["tap"], "step": row["step"], "eye": row["eye"], "ok": True, "n": len(near), "run": run.name,
                     "spread_mm": float(np.linalg.norm(pts.std(axis=0)) * 1000), "P": Pe, "uv": uv,
                     "dist_mm": float(np.linalg.norm(Pe) * 1000)})
    notes = {"eye_L": meta.get("display_left_from_head", "?"), "eye_R": meta.get("display_right_from_head", "?"),
             "bad": ", ".join(sorted(bad, key=int)) if bad else "none",
             "rgb": "logged" if "rgb_from_head" in meta else "not logged: the device's value from later runs",
             "centre": ("logged by the device" if "board_centre_m" in meta else "nominal 25 mm board")
             + (f"; re-posed at the printed size (x{k:.3f})" if abs(k - 1) > 1e-6 and repose is None else "")
             + (f"; poses re-solved from the corners with {repose['camera']}" if repose is not None else "")}
    return taps, notes


def _fmt(params: np.ndarray) -> str:
    c = params[:3] * 1000
    D = 1 / params[3] if params[3] > 1e-6 else float("inf")
    return (f"eye centre ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}) mm from the SDK's, display at {D:.2f} m, scale "
            f"{params[4]:.3f}, centre ({math.degrees(params[5]):+.2f}, {math.degrees(params[6]):+.2f}) deg")


def eye_lines(sel: List[dict], eye: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """One eye's taps: the SDK model's error by depth, each model's fit with its leave-one-out error (the quality of
    the calibration on a tap it has not seen), and the DLT centre. Returns (lines, each model's params)."""
    out: List[str] = []
    if len(sel) < 3:
        return [f"   {eye}: {len(sel)} taps, too few"], {}
    P = np.array([t["P"] for t in sel])
    uv = np.array([t["uv"] for t in sel])
    sdk = P[:, :2] / P[:, 2:3]
    e = sdk - uv   # the SDK draws P at sdk; the wearer sees P at uv: + = the drawing right of / above the real point
    out.append(f"   {eye} eye, {len(sel)} taps: where the SDK draws the point against where it is (mm at the target; "
               f"+x = drawing right of it, +y = above)")
    for cm in sorted({t['step'].split('_')[0] for t in sel}, key=lambda s: int(s[1:]) if s[1:].isdigit() else 0):
        idx = [i for i, t in enumerate(sel) if t["step"].startswith(cm + "_")]
        mm = np.array([e[i] * P[i, 2] * 1000 for i in idx])
        out.append(f"      {cm:5s} {len(idx):2d} taps at {np.median([sel[i]['dist_mm'] for i in idx]):4.0f} mm: "
                   f"mean ({mm[:, 0].mean():+6.1f}, {mm[:, 1].mean():+6.1f}) mm, spread {np.linalg.norm(mm.std(axis=0)):5.1f} mm; "
                   f"board point spread {np.median([sel[i]['spread_mm'] for i in idx]):.1f} mm")
    sdk_mm = np.linalg.norm(e * P[:, 2:3] * 1000, axis=1)
    out.append(f"      SDK model: rms {math.degrees(float(np.sqrt(np.mean(np.sum(e * e, axis=1))))):.2f} deg, "
               f"median {np.median(sdk_mm):.1f} mm at the target")
    fits: Dict[str, np.ndarray] = {}
    for model, label in (("display", "display scale and centre (the device's)"), ("eye", "eye position only"),
                         ("full", "eye, scale, centre")):
        if len(sel) * 2 < len(MODELS[model]) + 2:
            continue
        params, rms = fit_model(P, uv, fixed_w=0.25, model=model)
        loo = leave_one_out(P, uv, model)
        loo_s = (f"; leave-one-out median {np.median(np.linalg.norm(loo, axis=1)):.1f} mm, "
                 f"max {np.max(np.linalg.norm(loo, axis=1)):.1f} mm") if loo is not None else ""
        out.append(f"      fit {label} (display at 4 m): {_fmt(params)}; rms {rms:.2f} deg{loo_s}")
        fits[model] = params
    c = dlt_centre(P, uv)
    if c is not None:
        out.append(f"      DLT centre of projection: ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, {c[2] * 1000:+.1f}) mm from the SDK eye")
    return out, fits


def spacing_line(meta: dict, fits: Dict[str, np.ndarray]) -> List[str]:
    if len(fits) != 2:
        return []
    RL, tL, _ = pose_from_meta(meta.get("display_left_from_head", ""))
    RR, tR, _ = pose_from_meta(meta.get("display_right_from_head", ""))
    if any("eye" not in fits[e] for e in fits):
        return []
    cl, cr = tL + RL @ fits["L"]["eye"][:3], tR + RR @ fits["R"]["eye"][:3]
    return [f"   eye spacing: SDK {np.linalg.norm(tL - tR) * 1000:.1f} mm; fitted (eye position only) {np.linalg.norm(cl - cr) * 1000:.1f} mm"]


def spaam_report(run: Path) -> List[str]:
    from .beampro import load_json
    run = Path(run)
    meta = load_json(run)
    if meta.get("variant") not in ("spaam", "eyecal", "eyecheck"):
        return []
    out = [f"   spaam: {meta.get('spaam', '?')}",
           f"   SDK eyes: left {meta.get('display_left_from_head', '?')}; right {meta.get('display_right_from_head', '?')}",
           f"   XR eye positions: {meta.get('xr_eye_positions', '?')}; stereo separation {meta.get('camera_stereo_separation', '?')}",
           f"   projections: left {meta.get('display_left_projection', '?')[:80]}...",
           "   eye check: " + "; ".join(e["detail"] for e in load_csv(run / "events.csv") if e.get("event") == "rating")]
    for k in ("eye_taps_earlier", "eye_all_at_start", "eye_L_round", "eye_R_round", "eye_L_all", "eye_R_all", "eye_fit", "eye_model_saved"):
        if meta.get(k):
            out.append(f"   device {k}: {meta[k]}")
    modes = [e["detail"] for e in load_csv(run / "events.csv") if e.get("event") in ("eyemode", "eyemode_unavailable")]
    if modes:
        out.append("   model switches: " + ", ".join(modes))
    if not (run / "spaam.csv").exists():
        return out + ["   no taps"]
    taps, notes = tap_points(run)
    dev = {r["tap"]: r for r in load_csv(run / "spaam.csv") if r.get("pe_z")}
    diffs = [float(np.linalg.norm(t["P"] - np.array([float(dev[t["tap"]][k]) for k in ("pe_x", "pe_y", "pe_z")])) * 1000)
             for t in taps if t["ok"] and t["tap"] in dev]
    if diffs:
        k = board_scale(meta)
        why = f" (Python re-posed the board x{k:.3f}, the device did not)" if abs(k - 1) > 1e-6 else ""
        out.append(f"   device tap points against Python: {len(diffs)} taps, median {np.median(diffs):.2f} mm, max {max(diffs):.2f} mm{why}")
    good = [t for t in taps if t["ok"]]
    out.append(f"   taps: {len(taps)} (bad, dropped: {notes['bad']}); with the board seen in the 2 s before: {len(good)}")
    out.append(f"   tap point: rgb_from_head {notes['rgb']}; board centre {notes['centre']}")
    fits = {}
    for eye in ("L", "R"):
        lines, params = eye_lines([t for t in good if t["eye"] == eye], eye)
        out += lines
        if params:
            fits[eye] = params
    return out + spacing_line(meta, fits)


TAPS_HEADER = "wearing,camera,eye,step,px,py,pz,u,v"


def run_camera(meta: dict) -> str:
    """The camera calibration a run's poses used (the set's ``calibration``), the key of the device's tap store."""
    import re
    m = re.search(r"calibration ([^;]+);", str(meta.get("set", "")))
    return m.group(1).strip() if m else "?"


def export_taps(runs: List[Path], camera: Optional[str] = None, repose: Optional[dict] = None) -> List[str]:
    """The good taps of ``runs`` as the device's ``calibration/eye_taps.csv`` (EyeModel.cs): one wearing per run, each
    tap with the camera calibration its point was measured with (the device fits only taps of its current one).
    ``repose`` (``repose_set``) re-solves every run's poses with that set's camera and keys the taps with it; that is
    the way to carry taps over to a new calibration. ``camera`` only relabels the key. 0.4.017 relabelled taps posed
    under the Surface calibration as the Beam Pro's, on the tape check's 0.5 % agreement; but a tape checks the scale,
    not the principal point, and the two differ by 5 px, about 0.5 deg (T6.12 run 3)."""
    from .beampro import load_json
    lines = [TAPS_HEADER]
    for run in runs:
        cam = repose["camera"] if repose is not None else camera or run_camera(load_json(Path(run)))
        for t in tap_points(Path(run), repose=repose)[0]:
            if t["ok"]:
                lines.append(",".join([Path(run).name, cam, t["eye"], t["step"]] + [repr(float(x)) for x in (*t["P"], *t["uv"])]))
    return lines


def pooled_report(runs: List[Path], repose: Optional[dict] = None) -> List[str]:
    """Several SPAAM sessions: each one's fit of the device's model (does it repeat?), then all taps pooled per eye.
    ``repose`` re-solves every run's poses with one camera model (``tap_points``)."""
    from .beampro import load_json
    out = [f"== SPAAM pooled over {len(runs)} runs" + (f", poses re-solved with {repose['camera']}" if repose else "")]
    every: List[dict] = []
    meta = {}
    for run in runs:
        meta = load_json(Path(run))
        taps = [t for t in tap_points(Path(run), repose=repose)[0] if t["ok"]]
        every += taps
        per = []
        for eye in ("L", "R"):
            sel = [t for t in taps if t["eye"] == eye]
            if len(sel) >= 3:
                params, _ = fit_model(np.array([t["P"] for t in sel]), np.array([t["uv"] for t in sel]), fixed_w=0.25,
                                      model=DEVICE_MODEL)
                per.append(f"{eye} centre ({math.degrees(params[5]):+.2f}, {math.degrees(params[6]):+.2f}) deg, "
                           f"scale {params[4]:.3f} from {len(sel)} taps")
        out.append(f"   {Path(run).name}: " + "; ".join(per))
    fits = {}
    for eye in ("L", "R"):
        lines, params = eye_lines([t for t in every if t["eye"] == eye], eye)
        out += lines
        if params:
            fits[eye] = params
    return out + spacing_line(meta, fits)
