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
    from gui_common import CHARUCO_MARKER_M, CHARUCO_SQUARE_M
    board = make_charuco_board(BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M)
    return board, (BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M)


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
