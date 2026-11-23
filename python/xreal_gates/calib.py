"""Eye intrinsic calibration from a gate calibration run (``-gate calib``).

The run saves an upright PGM every 0.5 s while the operator moves the 25 mm
ChArUco board of ``calibrate.py`` through the Eye's view. Here: detect the board
in every snapshot, pick a spread of views (image position x board size), and
calibrate with OpenCV's standard 5-coefficient model and with the rational
8-coefficient model. The rational model is used only when it lowers the RMS by
more than 15 %: the Eye's lens is wide (about 93 degrees across), but a richer
model fitted to few views mostly fits noise.

Output: an OpenCV YAML in the ``calibrate.save_calibration`` format (readable by
``calibrate.read_calibration``) plus a JSON report next to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .eyecam import EYE_SIZE, VENDOR_FX_FY_CX_CY, vendor_camera

MIN_CORNERS = 8
MAX_VIEWS = 60
GRID = (8, 6)               # coverage cells across the image
RATIONAL_GAIN = 0.85        # rational model only if rms_rational < 0.85 * rms_standard
#: Refuse (write nothing) below this image coverage: a board seen in one place gives a calibration
#: that reprojects well and is wrong (G1 of 16 Oct: 8 % coverage, RMS 0.37 px, fx 4344 px).
MIN_COVERAGE = 0.35
#: Refuse a focal length this far from the vendor's (relative); a wrong one scales every pose.
MAX_FOCAL_DEVIATION = 0.25


@dataclass
class View:
    path: Path
    corners: np.ndarray
    ids: np.ndarray
    centre: np.ndarray
    area: float


@dataclass
class CalibResult:
    yaml_path: Path
    report_path: Path
    rms_px: float
    model: str
    views_used: int
    views_found: int
    snapshots: int
    coverage: float
    K: np.ndarray
    dist: np.ndarray
    notes: List[str] = field(default_factory=list)


def latest_calib_run(root: Path) -> Optional[Path]:
    runs = sorted(p for p in Path(root).glob("*_Calib_*") if p.is_dir() and (p / "snapshots").is_dir())
    return runs[-1] if runs else None


def _board():
    from calibrate import BOARD_COLS, BOARD_ROWS, make_charuco_board
    from gui_common import CHARUCO_MARKER_PRINTED_M, CHARUCO_SQUARE_PRINTED_M
    board = make_charuco_board(BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_PRINTED_M, CHARUCO_MARKER_PRINTED_M)
    return board, (BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_PRINTED_M, CHARUCO_MARKER_PRINTED_M)


def find_views(run_dir: Path, board) -> Tuple[List[View], int]:
    det = cv2.aruco.CharucoDetector(board)
    views, n = [], 0
    for path in sorted((Path(run_dir) / "snapshots").glob("*.pgm")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or (gray.shape[1], gray.shape[0]) != EYE_SIZE:
            continue
        n += 1
        cc, ci, _, _ = det.detectBoard(gray)
        if ci is None or len(ci) < MIN_CORNERS:
            continue
        pts = cc.reshape(-1, 2)
        area = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))
        views.append(View(path, cc, ci, pts.mean(axis=0), area))
    return views, n


def select_views(views: List[View], max_views: int = MAX_VIEWS) -> List[View]:
    """Round-robin over (image cell, size class) bins, most corners first in each bin."""
    if len(views) <= max_views:
        return list(views)
    w, h = EYE_SIZE
    areas = np.array([v.area for v in views])
    edges = np.quantile(areas, [1 / 3, 2 / 3])
    bins = {}
    for v in views:
        key = (min(int(v.centre[0] / w * 4), 3), min(int(v.centre[1] / h * 3), 2), int(np.searchsorted(edges, v.area)))
        bins.setdefault(key, []).append(v)
    for b in bins.values():
        b.sort(key=lambda v: -len(v.ids))
    chosen, depth = [], 0
    while len(chosen) < max_views:
        added = False
        for b in bins.values():
            if depth < len(b) and len(chosen) < max_views:
                chosen.append(b[depth])
                added = True
        if not added:
            break
        depth += 1
    return chosen


def coverage(views: List[View]) -> float:
    w, h = EYE_SIZE
    cells = set()
    for v in views:
        for x, y in v.corners.reshape(-1, 2):
            cells.add((min(int(x / w * GRID[0]), GRID[0] - 1), min(int(y / h * GRID[1]), GRID[1] - 1)))
    return len(cells) / (GRID[0] * GRID[1])


def _calibrate(views: List[View], board, flags: int):
    rms, K, dist, _, _, _, _, per_view = cv2.aruco.calibrateCameraCharucoExtended(
        [v.corners for v in views], [v.ids for v in views], board, EYE_SIZE, None, None, flags=flags)
    return float(rms), K, dist.ravel(), np.ravel(per_view)


def calibrate_run(run_dir: Path, out_yaml: Optional[Path] = None, max_views: int = MAX_VIEWS) -> CalibResult:
    from calibrate import MIN_FRAMES, save_calibration

    run_dir = Path(run_dir)
    board, (cols, rows, square, marker) = _board()
    views, n = find_views(run_dir, board)
    notes = []
    if len(views) < MIN_FRAMES:
        raise ValueError(f"{run_dir.name}: board found in {len(views)} of {n} snapshots; need {MIN_FRAMES}")
    chosen = select_views(views, max_views)
    cov = coverage(chosen)
    if cov < MIN_COVERAGE:
        raise ValueError(f"the board covered only {cov:.0%} of the image (need {MIN_COVERAGE:.0%}); "
                         "move it into every corner and edge of the picture")
    std = _calibrate(chosen, board, 0)
    rat = _calibrate(chosen, board, cv2.CALIB_RATIONAL_MODEL)
    use_rational = rat[0] < RATIONAL_GAIN * std[0]
    rms, K, dist, per_view = rat if use_rational else std
    model = "rational (k1-k6, p1, p2)" if use_rational else "standard (k1, k2, p1, p2, k3)"
    vk = vendor_camera().K
    for name, got, ref in (("fx", K[0, 0], vk[0, 0]), ("fy", K[1, 1], vk[1, 1])):
        if abs(got - ref) / ref > MAX_FOCAL_DEVIATION:
            raise ValueError(f"implausible result: {name} {got:.0f} px against the vendor's {ref:.0f} px "
                             f"(RMS {rms:.2f} px, coverage {cov:.0%}); not written")
    if cov < 0.6:
        notes.append(f"only {cov:.0%} of the image covered: move the board into the corners and edges")
    worst = sorted(zip(per_view, chosen), key=lambda t: -t[0])[:3]

    out_yaml = Path(out_yaml) if out_yaml else run_dir / "eye_calibration.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(str(out_yaml), K, dist.reshape(1, -1), EYE_SIZE, rms, cols, rows, square, marker)
    vendor = vendor_camera()
    report = {
        "run": run_dir.name, "snapshots": n, "views_found": len(views), "views_used": len(chosen),
        "coverage": round(cov, 3), "model": model,
        "rms_px": {"used": rms, "standard": std[0], "rational": rat[0]},
        "K": K.tolist(), "dist": dist.tolist(),
        "vendor": {"fx_fy_cx_cy_stored_rows": list(VENDOR_FX_FY_CX_CY), "K_upright": vendor.K.tolist()},
        "delta_vs_vendor": {"fx": float(K[0, 0] - vendor.K[0, 0]), "fy": float(K[1, 1] - vendor.K[1, 1]),
                            "cx": float(K[0, 2] - vendor.K[0, 2]), "cy": float(K[1, 2] - vendor.K[1, 2])},
        "worst_views": [{"snapshot": v.path.name, "rms_px": float(e)} for e, v in worst],
        "board": {"cols": cols, "rows": rows, "square_m": square, "marker_m": marker},
        "notes": notes,
    }
    report_path = out_yaml.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return CalibResult(out_yaml, report_path, rms, model, len(chosen), len(views), n, cov, K, dist, notes)


def describe(r: CalibResult) -> List[str]:
    v = vendor_camera().K
    lines = [
        f"board found in {r.views_found} of {r.snapshots} snapshots, {r.views_used} views used, "
        f"image coverage {r.coverage:.0%}",
        f"model {r.model}: RMS {r.rms_px:.3f} px",
        f"fx {r.K[0, 0]:.1f}  fy {r.K[1, 1]:.1f}  cx {r.K[0, 2]:.1f}  cy {r.K[1, 2]:.1f}   "
        f"(vendor {v[0, 0]:.1f} {v[1, 1]:.1f} {v[0, 2]:.1f} {v[1, 2]:.1f})",
        "dist " + " ".join(f"{d:+.4f}" for d in r.dist),
        f"written {r.yaml_path}",
    ]
    return lines + [f"NOTE: {n}" for n in r.notes]


# ---------------------------------------------------------------- T6.13: the Eye on the Beam Pro

#: Tape distances of the T6.13 check steps (camera lens to the board's surface), millimetres.
TAPE_STEPS = {"t50": 500.0, "t100": 1000.0}
#: Accept a Beam Pro calibration only if the tape check's scale (tape change over pose distance change) is this close to 1.
MAX_TAPE_SCALE_ERROR = 0.03
#: Where the accepted Beam Pro Eye calibration goes: a camera profile like the GUI's, plus the JSON the frame set reads.
BEAMPRO_PROFILE = Path(__file__).resolve().parents[1] / "config" / "cameras" / "beampro_eye.yaml"


def _frame_steps(run_dir: Path) -> dict:
    from .beampro import load_csv
    return {int(r["seq"]): r["step"] for r in load_csv(Path(run_dir) / "live.csv")}


def _views_in(paths: List[Path], board) -> List[View]:
    det = cv2.aruco.CharucoDetector(board)
    views = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or (gray.shape[1], gray.shape[0]) != EYE_SIZE:
            continue
        cc, ci, _, _ = det.detectBoard(gray)
        if ci is None or len(ci) < MIN_CORNERS:
            continue
        pts = cc.reshape(-1, 2)
        views.append(View(path, cc, ci, pts.mean(axis=0),
                          float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))))
    return views


def plane_distance_mm(views: List[View], board, K: np.ndarray, dist: np.ndarray) -> Optional[float]:
    """Median distance from the camera centre to the board's plane over ``views``, millimetres."""
    out = []
    for v in views:
        obj, img = board.matchImagePoints(v.corners, v.ids)
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist)
        if ok:
            R = cv2.Rodrigues(rv)[0]
            out.append(abs(float(R[:, 2] @ tv.ravel())) * 1000)
    return float(np.median(out)) if out else None


def tape_check(steps: dict, board, K: np.ndarray, dist: np.ndarray) -> dict:
    """Distances of the tape steps under (K, dist) and the scale they imply: (1000 - 500) / (d100 - d50). The scale does
    not depend on where the lens sits against the camera centre; the offset (tape - pose) is that distance."""
    d = {name: plane_distance_mm(steps.get(name, []), board, K, dist) for name in TAPE_STEPS}
    out = {"distances_mm": d}
    if all(v is not None for v in d.values()):
        a, b = d["t50"], d["t100"]
        out["scale"] = (TAPE_STEPS["t100"] - TAPE_STEPS["t50"]) / (b - a) if b != a else None
        out["lens_offset_mm"] = {n: TAPE_STEPS[n] - d[n] for n in TAPE_STEPS}
    return out


def calibrate_beampro(run_dir: Path, set_K: Optional[np.ndarray] = None, set_dist: Optional[np.ndarray] = None,
                      max_views: int = MAX_VIEWS) -> dict:
    """T6.13: the Eye's intrinsics on the Beam Pro from a ``camcal`` run's saved frames (move steps ``m_*``), checked
    against the tape steps (t50, t100). Returns the report; ``accepted`` says whether the checks passed.

    The vendor's intrinsics are not a reference here: T6.12's tape check found them (and the Surface calibration) 25-28 %
    short in focal length for the Beam Pro's stream, so the usual vendor deviation check would refuse the right answer."""
    from calibrate import MIN_FRAMES
    run_dir = Path(run_dir)
    board, (cols, rows, square, marker) = _board()
    steps_of = _frame_steps(run_dir)
    by_step: dict = {}
    for path in sorted((run_dir / "frames").glob("*.pgm")):
        try:
            seq = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        by_step.setdefault(steps_of.get(seq, "?"), []).append(path)
    move = [p for s, ps in by_step.items() if s.startswith("m_") for p in ps]
    views = _views_in(move, board)
    tape = {name: _views_in(by_step.get(name, []), board) for name in TAPE_STEPS}
    if len(views) < MIN_FRAMES:
        raise ValueError(f"{run_dir.name}: board found in {len(views)} of {len(move)} move frames; need {MIN_FRAMES}")
    chosen = select_views(views, max_views)
    cov = coverage(chosen)
    std = _calibrate(chosen, board, 0)
    rat = _calibrate(chosen, board, cv2.CALIB_RATIONAL_MODEL)
    use_rational = rat[0] < RATIONAL_GAIN * std[0]
    rms, K, dist, _ = rat if use_rational else std
    check = tape_check(tape, board, K, dist)
    report = {
        "run": run_dir.name, "frames": {s: len(p) for s, p in sorted(by_step.items())},
        "views_found": len(views), "views_used": len(chosen), "coverage": round(cov, 3),
        "model": "rational (k1-k6, p1, p2)" if use_rational else "standard (k1, k2, p1, p2, k3)",
        "rms_px": {"used": rms, "standard": std[0], "rational": rat[0]},
        "K": K.tolist(), "dist": dist.tolist(), "size": list(EYE_SIZE),
        "board": {"cols": cols, "rows": rows, "square_m": square, "marker_m": marker},
        "tape_views": {n: len(v) for n, v in tape.items()}, "tape": check,
        "vendor": vendor_camera().K.tolist(),
    }
    if set_K is not None:
        report["tape_with_set"] = tape_check(tape, board, np.asarray(set_K, float).reshape(3, 3),
                                             np.zeros(5) if set_dist is None else np.asarray(set_dist, float))
    notes = []
    if cov < MIN_COVERAGE:
        notes.append(f"only {cov:.0%} of the image covered (need {MIN_COVERAGE:.0%})")
    sc = check.get("scale")
    if sc is None:
        notes.append("no tape check (the board was not seen in t50 and t100)")
    elif abs(sc - 1) > MAX_TAPE_SCALE_ERROR:
        notes.append(f"tape scale {sc:.3f}: off by more than {MAX_TAPE_SCALE_ERROR:.0%}")
    report["notes"] = notes
    report["accepted"] = not notes
    return report


def write_beampro(report: dict, out_yaml: Path = BEAMPRO_PROFILE) -> Tuple[Path, Path]:
    """The calibration as a camera profile (``calibrate.save_calibration`` YAML) and its JSON report beside it; the
    JSON's ``K`` and ``dist`` are what ``b3a.eye_calibration`` and ``b3a-set`` read."""
    from calibrate import save_calibration
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    b = report["board"]
    save_calibration(str(out_yaml), np.asarray(report["K"]), np.asarray(report["dist"]).reshape(1, -1), EYE_SIZE,
                     report["rms_px"]["used"], b["cols"], b["rows"], b["square_m"], b["marker_m"])
    js = out_yaml.with_suffix(".json")
    js.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_yaml, js


def _tape_line(name: str, chk: dict) -> str:
    d = chk["distances_mm"]
    if chk.get("scale") is None:
        return f"   tape check, {name}: distances {d}"
    off = chk["lens_offset_mm"]
    return (f"   tape check, {name}: t50 {d['t50']:.1f} mm, t100 {d['t100']:.1f} mm; scale {chk['scale']:.3f}; "
            f"lens to camera centre {off['t50']:.1f} / {off['t100']:.1f} mm")


def describe_beampro(r: dict) -> List[str]:
    K = np.asarray(r["K"])
    out = [f"{r['run']}: {r['views_used']} of {r['views_found']} views, coverage {r['coverage']:.0%}, {r['model']}, "
           f"RMS {r['rms_px']['used']:.3f} px",
           f"   fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}; dist {np.round(r['dist'][:5], 4).tolist()}",
           _tape_line("new calibration", r["tape"])]
    if r.get("tape_with_set"):
        out.append(_tape_line("the set's calibration", r["tape_with_set"]))
    out.append("   ACCEPTED" if r["accepted"] else "   NOT ACCEPTED: " + "; ".join(r["notes"]))
    return out
