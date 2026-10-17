"""The three physical targets the G5 locked-grid gate can lock onto.

Each target answers two questions: where is it in this camera image (pose in the
OpenCV camera frame of the upright image), and what should the glasses draw on
it (overlay segments and points in the target's own frame, metres). Every
dimension comes from its single definition elsewhere:

- ``plateA`` / ``plateL``: the validation plates (``validation/plates.py`` via the
  target JSON in ``config/targets/``, the caliper-measured one when present).
  Pose from the plate's ArUco marker corners plus the printed saddle points that
  the validation detector (``validation.detect.locate_points``) finds around them.
- ``board25``: the 5 x 7 ChArUco calibration board of ``calibrate.py``
  (25 mm squares, 19 mm markers, ``gui_common``), pose from its ChArUco corners.

Target frames are right-handed (plates: Z out of the printed surface; board:
OpenCV's ChArUco frame). The Unity side converts with ``F = diag(1, -1, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
TARGETS_DIR = PYTHON_DIR / "config" / "targets"

#: Acceptance (per frame). A pose with a larger reprojection RMS, or a planar
#: pose whose two IPPE solutions fit almost equally well, is not used.
MAX_RMS_PX = 2.0
MAX_AMBIGUITY = 0.8          # best/second-best reprojection error of the planar solutions
MIN_BOARD_CORNERS = 6
MIN_PLATE_SADDLES = 6
MIN_PLATE_MARKERS = 2

TARGET_NAMES = ("plateA", "plateL", "board25")


@dataclass
class PoseEstimate:
    ok: bool
    reason: str = ""
    rvec: Optional[np.ndarray] = None   # OpenCV camera frame, upright image
    tvec: Optional[np.ndarray] = None   # metres
    rms_px: float = float("nan")
    ambiguity: float = float("nan")
    n_markers: int = 0
    n_points: int = 0
    marker_ids: List[int] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {"ok": self.ok, "reason": self.reason, "rms_px": _num(self.rms_px),
             "ambiguity": _num(self.ambiguity), "n_markers": self.n_markers,
             "n_points": self.n_points, "marker_ids": list(self.marker_ids)}
        if self.rvec is not None and self.tvec is not None:
            R, _ = cv2.Rodrigues(np.asarray(self.rvec, float))
            d["rvec"] = [float(v) for v in np.ravel(self.rvec)]
            d["tvec"] = [float(v) for v in np.ravel(self.tvec)]
            d["R"] = [float(v) for v in R.ravel()]          # row-major
        return d


def _num(v: float):
    return None if v is None or not np.isfinite(v) else float(v)


def solve_pose(obj: np.ndarray, img: np.ndarray, K, dist, planar: bool):
    """(rvec, tvec, rms_px, ambiguity). Planar: IPPE with its two solutions, then LM refinement."""
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    ambiguity = float("nan")
    if planar:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not n:
            return None
        rvec, tvec = rvecs[0], tvecs[0]
        e = np.ravel(errs) if errs is not None else np.zeros(0)
        if n > 1 and len(e) > 1 and e[1] > 0:
            ambiguity = float(e[0] / e[1])
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return rvec, tvec, rms, ambiguity


class GateTarget:
    name = ""
    label = ""

    def estimate(self, gray: np.ndarray, K, dist) -> PoseEstimate:
        raise NotImplementedError

    def overlay(self) -> dict:
        """``{"segments": [[x0,y0,z0,x1,y1,z1], ...], "points": [[x,y,z], ...]}`` in metres."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.label


def _accept(est: PoseEstimate, planar: bool, strong: bool) -> PoseEstimate:
    finite = (np.isfinite(est.rms_px) and np.all(np.isfinite(est.rvec)) and np.all(np.isfinite(est.tvec))
              and float(np.ravel(est.tvec)[2]) > 0)
    if not finite:
        est.ok, est.reason = False, "degenerate solution"
    elif est.rms_px > MAX_RMS_PX:
        est.ok, est.reason = False, f"rms {est.rms_px:.2f} px > {MAX_RMS_PX}"
    elif planar and not strong and np.isfinite(est.ambiguity) and est.ambiguity > MAX_AMBIGUITY:
        est.ok, est.reason = False, f"ambiguous planar pose ({est.ambiguity:.2f})"
    else:
        est.ok, est.reason = True, "ok"
    return est


# ---------------------------------------------------------------- plates

class PlateTarget(GateTarget):
    def __init__(self, layout_key: str, target_path: Optional[Path] = None):
        from validation.plates import layout_from_provenance
        from validation.target import load_target

        self.name = f"plate{layout_key}"
        path = target_path or _plate_target_path(layout_key)
        self.target = load_target(path)
        self.layout = layout_from_provenance(self.target.provenance)
        self.source = str(path)
        self.label = f"plate {layout_key} ({Path(path).name})"
        self.planar = len(self.target.faces) == 1

    def estimate(self, gray, K, dist) -> PoseEstimate:
        from validation.detect import locate_points

        loc = locate_points(gray, self.target, K, dist)
        seen = [m for m in self.target.markers if m.id in loc.markers]
        if not seen:
            return PoseEstimate(False, "no plate marker")
        obj = [m.corners() for m in seen]
        img = [loc.markers[m.id] for m in seen]
        pids = [p for p in loc.points if p in self.target.points]
        if pids:
            obj.append(self.target.array(pids))
            img.append(np.array([loc.points[p] for p in pids]))
        obj, img = np.vstack(obj), np.vstack(img)
        faces_seen = {m.face for m in seen} | {self.target.point_face[p] for p in pids}
        planar = len(faces_seen) == 1
        solved = solve_pose(obj, img, K, dist, planar)
        if solved is None:
            return PoseEstimate(False, "PnP failed", n_markers=len(seen), n_points=len(pids))
        rvec, tvec, rms, amb = solved
        est = PoseEstimate(True, "", rvec, tvec, rms, amb, len(seen), len(pids), [m.id for m in seen])
        # Saddles seeded from a single 20 mm marker are often wrong (validation B1b, 15 Oct: seed
        # offsets up to the search radius), so a plate pose needs at least two markers.
        if len(seen) < MIN_PLATE_MARKERS:
            est.ok, est.reason = False, f"{len(seen)} marker(s), need {MIN_PLATE_MARKERS}"
            return est
        return _accept(est, planar, strong=len(pids) >= MIN_PLATE_SADDLES)

    def overlay(self) -> dict:
        t, segs = self.target, []
        for a, b in t.edges:
            pa, pb = t.points.get(str(a)), t.points.get(str(b))
            if pa is not None and pb is not None:
                segs.append(_seg(pa, pb))
        for m in t.markers:
            c = m.corners()
            segs += [_seg(c[i], c[(i + 1) % 4]) for i in range(4)]
        for pf in self.layout.faces:
            x0, y0, x1, y1 = pf.region
            corners = [self._page_to_model(pf.key, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            segs += [_seg(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        return {"segments": segs, "points": [_v(t.points[i]) for i in t.ids()]}

    def _page_to_model(self, face_key: str, px: float, py: float) -> np.ndarray:
        """Page mm to model metres, the mapping of ``validation.plates.build_target``."""
        pf, f = self.layout.face(face_key), self.target.faces[face_key]
        sx = float(self.target.provenance.get("scale_x", 1.0))
        sy = float(self.target.provenance.get("scale_y", 1.0))
        ax, ay = pf.anchor
        return f.origin + sx * (px - ax) / 1000.0 * f.ex + sy * (ay - py) / 1000.0 * f.ey


def _plate_target_path(layout_key: str) -> Path:
    measured = TARGETS_DIR / f"plate_{layout_key}.measured.target.json"
    return measured if measured.exists() else TARGETS_DIR / f"plate_{layout_key}.target.json"


# ---------------------------------------------------------------- ChArUco board

class BoardTarget(GateTarget):
    name = "board25"

    def __init__(self):
        from calibrate import BOARD_COLS, BOARD_ROWS, make_charuco_board
        from gui_common import CHARUCO_MARKER_M, CHARUCO_SQUARE_M

        self.cols, self.rows, self.square = BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M
        self.board = make_charuco_board(BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.label = (f"ChArUco {BOARD_COLS}x{BOARD_ROWS}, {CHARUCO_SQUARE_M * 1000:.0f} mm squares, "
                      f"{CHARUCO_MARKER_M * 1000:.0f} mm markers")
        self.source = "calibrate.make_charuco_board"

    def estimate(self, gray, K, dist) -> PoseEstimate:
        cc, ci, mc, mi = self.detector.detectBoard(gray)
        n_markers = 0 if mi is None else len(mi)
        ids = [] if mi is None else [int(i) for i in np.ravel(mi)]
        if ci is None or len(ci) < MIN_BOARD_CORNERS:
            return PoseEstimate(False, f"{0 if ci is None else len(ci)} board corners", n_markers=n_markers,
                                n_points=0 if ci is None else len(ci), marker_ids=ids)
        obj, img = self.board.matchImagePoints(cc, ci)
        solved = solve_pose(obj, img, K, dist, planar=True)
        if solved is None:
            return PoseEstimate(False, "PnP failed", n_markers=n_markers, n_points=len(ci), marker_ids=ids)
        rvec, tvec, rms, amb = solved
        est = PoseEstimate(True, "", rvec, tvec, rms, amb, n_markers, len(ci), ids)
        return _accept(est, planar=True, strong=len(ci) >= 12)

    def overlay(self) -> dict:
        s, W, H = self.square, self.cols * self.square, self.rows * self.square
        segs = [_seg((i * s, 0, 0), (i * s, H, 0)) for i in range(self.cols + 1)]
        segs += [_seg((0, j * s, 0), (W, j * s, 0)) for j in range(self.rows + 1)]
        pts = [_v(p) for p in np.asarray(self.board.getChessboardCorners(), float).reshape(-1, 3)]
        return {"segments": segs, "points": pts}


def _v(p) -> List[float]:
    return [float(c) for c in np.asarray(p, float).ravel()[:3]]


def _seg(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return _v(a) + _v(b)


def make_target(name: str) -> GateTarget:
    if name == "plateA":
        return PlateTarget("A")
    if name == "plateL":
        return PlateTarget("L")
    if name == "board25":
        return BoardTarget()
    raise ValueError(f"unknown target {name!r}; expected one of {', '.join(TARGET_NAMES)}")


def overlay_counts(overlay: Dict[str, list]) -> str:
    return f"{len(overlay['segments'])} segments, {len(overlay['points'])} points"
