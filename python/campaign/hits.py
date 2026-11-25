"""Hit positions on a campaign sheet from a scan, a phone photo or an Eye frame (T7.0 §6 C6).

1. The frame markers (ids 30-33) and the sheet-ID marker give a first homography page mm -> image.
2. The border saddles (or A40's grid) are refined to their X-junctions; marker corners are refined by ArUco's
   sub-pixel step. A least-squares homography over all of them is the sheet's frame in the image.
3. The page is rectified at ``ppmm`` pixels per mm. Printed ink (the sheet's own render, dilated) is masked, and
   every remaining dark blob is a candidate dot; its centroid is intensity-weighted.
4. Dots are assigned to the sheet's targets by minimum total distance (Hungarian), at most ``max_mm`` away.

Errors are reported in the model frame (x right, y up on the page), mm: ``dx = hit_x - nom_x``,
``dy = -(hit_y - nom_y)`` in page coordinates.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import sheets as S

SHEETS_DIR = Path(__file__).resolve().parent.parent / "config" / "campaign" / "sheets"


@dataclass
class Hit:
    n: int
    nom: Tuple[float, float]           # page mm
    hit: Optional[Tuple[float, float]]  # page mm, None = no dot assigned
    area_mm2: float = float("nan")

    @property
    def err(self) -> Optional[Tuple[float, float]]:
        if self.hit is None:
            return None
        return (self.hit[0] - self.nom[0], -(self.hit[1] - self.nom[1]))


@dataclass
class Result:
    sheet: str
    source: str
    hits: List[Hit]
    extra_dots: List[Tuple[float, float]]
    fit_points: int
    fit_rms_mm: float
    mm_per_px: float                     # image resolution on the sheet (median)
    notes: List[str] = field(default_factory=list)


def load_sheet(name_or_number) -> S.Sheet:
    """Rebuild a sheet from its JSON (the definition the app and the print used)."""
    if isinstance(name_or_number, int):
        idx = json.loads((SHEETS_DIR / "index.json").read_text(encoding="utf-8"))
        name = next(s["name"] for s in idx["sheets"] if s["number"] == name_or_number and s["kind"] != "A40")
    else:
        name = name_or_number
    d = json.loads((SHEETS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    sh = S.sheet_a40() if d["kind"] == "A40" else S.sheet(d["kind"], d["number"])
    if S.digest(sh) != d["digest"]:
        raise ValueError(f"{name}: the sheet code no longer reproduces the printed sheet (digest)")
    return sh


def _detect(gray, dictionary):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), p)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def sheet_id(gray) -> Optional[int]:
    got = _detect(gray, S.ID_DICT)
    return min(got) if got else None


def _marker_page_corners(m: S.Marker) -> np.ndarray:
    h = m.size / 2
    return np.array([[m.x - h, m.y - h], [m.x + h, m.y - h], [m.x + h, m.y + h], [m.x - h, m.y + h]])


def _project(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def fit_frame(gray, sh: S.Sheet) -> Tuple[np.ndarray, int, float, float, List[str]]:
    """Homography page mm -> image px, number of points, RMS residual (mm), mm per px."""
    notes = []
    src, dst = [], []
    for dic in (S.FRAME_DICT, S.ID_DICT):
        got = _detect(gray, dic)
        for m in sh.markers:
            if m.dictionary == dic and m.id in got:
                src.append(_marker_page_corners(m))
                dst.append(got[m.id])
    if len(src) < 2:
        raise RuntimeError(f"{sh.name}: only {len(src)} markers found")
    src, dst = np.vstack(src), np.vstack(dst)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    # local scale (px per mm) at the page centre
    c = np.array([[S.CENTRE[0], S.CENTRE[1]], [S.CENTRE[0] + 1, S.CENTRE[1]]])
    px_per_mm = float(np.linalg.norm(np.diff(_project(H, c), axis=0)))
    sad_src, sad_dst = [], []
    for s in sh.saddles:
        guess = _project(H, np.array([[s.x, s.y]]))[0]
        w = int(max(3, min(25, 0.35 * s.square * px_per_mm)))
        if not (w <= guess[0] < gray.shape[1] - w and w <= guess[1] < gray.shape[0] - w):
            continue
        pt = guess.reshape(1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(gray, pt, (w, w), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
        if np.linalg.norm(pt.reshape(2) - guess) < 0.25 * s.square * px_per_mm:
            sad_src.append([s.x, s.y])
            sad_dst.append(pt.reshape(2))
    if sad_src:
        src = np.vstack([src, np.array(sad_src)])
        dst = np.vstack([dst, np.array(sad_dst)])
    else:
        notes.append("no saddle refined")
    H, mask = cv2.findHomography(src, dst, cv2.LMEDS)
    Hinv = np.linalg.inv(H)
    res = _project(Hinv, dst) - src
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    px_per_mm = float(np.linalg.norm(np.diff(_project(H, c), axis=0)))
    return H, len(src), rms, 1.0 / px_per_mm, notes


def rectify(gray, H, ppmm: float) -> np.ndarray:
    Sm = np.diag([ppmm, ppmm, 1.0])
    Hc = H @ np.linalg.inv(Sm)
    size = (int(round(S.PAGE[0] * ppmm)), int(round(S.PAGE[1] * ppmm)))
    return cv2.warpPerspective(gray, Hc, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=255)


def _dilate(mask, mm: float, ppmm: float):
    k = max(3, int(round(2 * mm * ppmm)) | 1)
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def label_mask(sh: S.Sheet, ppmm: float, shape) -> np.ndarray:
    """A box over each printed target number, wide enough for any font.

    The glyphs' extent depends on the OpenCV that printed the sheet (Hershey in 4.x, another font in 5.x), so the
    exact render does not mask the printed labels: 0.5.000's PDFs came from OpenCV 5 and, analysed under 4.13, an
    unmarked P sheet gave 8 false hits. A dot inside the box is lost (reported as missing), never mistaken.
    """
    m = np.zeros(shape, bool)
    for t in sh.targets:
        lab = S._target_label(t)
        x0, x1 = lab.x - 0.8, lab.x + 1.4 * lab.h * len(lab.s) + 1.0
        y0, y1 = lab.y - 1.35 * lab.h - 0.8, lab.y + 0.4 * lab.h + 0.6
        m[max(0, int(y0 * ppmm)):int(np.ceil(y1 * ppmm)), max(0, int(x0 * ppmm)):int(np.ceil(x1 * ppmm))] = True
    return m


def find_dots(rect, sh: S.Sheet, ppmm: float, mm_per_px: float = 0.05, min_d_mm: float = 0.3,
              max_d_mm: float = 4.0):
    """Dark blobs that are not printed ink: (x_mm, y_mm, area_mm2).

    The frame's ink (markers, saddles, bars, text: all at least 15 mm from any target) is masked generously,
    by max(2 mm, 3 source pixels); printed crosses on P sheets only by 0.4 mm + 1.5 source pixels, so a dot in or
    near a cross's open centre survives.
    """
    frame_ink = _dilate(S.render(sh, ppmm, targets=False) < 128, max(2.0, 3 * mm_per_px), ppmm)
    ink = frame_ink
    if sh.kind == "P":
        cross = (S.render(sh, ppmm, targets=True) < 128) & ~(S.render(sh, ppmm, targets=False) < 128)
        ink = ink | _dilate(cross, 0.4 + 1.5 * mm_per_px, ppmm)
        ink = ink | label_mask(sh, ppmm, ink.shape)
    work = np.zeros_like(ink)
    x0, y0, x1, y1 = (int(v * ppmm) for v in (52.0, 22.0, 245.0, 188.0))   # inside the frame
    work[y0:y1, x0:x1] = True
    bg = cv2.medianBlur(rect, max(3, int(round(6 * ppmm)) | 1))
    diff = cv2.subtract(bg, rect).astype(np.float32)
    vals = diff[work & ~ink]
    thr = max(25.0, float(np.percentile(vals, 50) + 6 * (np.percentile(vals, 84) - np.percentile(vals, 50)) + 1))
    cand = ((diff > thr) & work & ~ink).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    out = []
    amin = math.pi * (min_d_mm / 2) ** 2 * ppmm ** 2
    amax = math.pi * (max_d_mm / 2) ** 2 * ppmm ** 2
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (amin <= a <= amax):
            continue
        ys, xs = np.nonzero(lab == i)
        w = diff[ys, xs]
        out.append((float((xs * w).sum() / w.sum()) / ppmm, float((ys * w).sum() / w.sum()) / ppmm,
                    float(a) / ppmm ** 2))
    return out


def assign(sh: S.Sheet, dots, max_mm: float):
    from scipy.optimize import linear_sum_assignment
    T = np.array([[t.x, t.y] for t in sh.targets])
    hits = [Hit(t.n, (t.x, t.y), None) for t in sh.targets]
    if not dots:
        return hits, []
    D = np.array([[d[0], d[1]] for d in dots])
    cost = np.linalg.norm(T[:, None, :] - D[None, :, :], axis=2)
    big = 1e6
    r, c = linear_sum_assignment(np.where(cost <= max_mm, cost, big))
    used = set()
    for i, j in zip(r, c):
        if cost[i, j] <= max_mm:
            hits[i].hit = (dots[j][0], dots[j][1])
            hits[i].area_mm2 = dots[j][2]
            used.add(j)
    extra = [(d[0], d[1]) for j, d in enumerate(dots) if j not in used]
    return hits, extra


def analyse(image_path, sheet: Optional[str] = None, ppmm: float = 10.0, max_mm: float = 40.0,
            calib: Optional[str] = None) -> Tuple[Result, np.ndarray]:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if calib:
        fs = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
        K, dist = fs.getNode("camera_matrix").mat(), fs.getNode("dist_coeffs").mat()
        gray = cv2.undistort(gray, K, dist)
    if sheet is None:
        sid = sheet_id(gray)
        if sid is None:
            raise RuntimeError("no sheet-ID marker found: pass --sheet")
        sh = load_sheet(sid)
    else:
        sh = load_sheet(sheet)
    H, npts, rms, mmpx, notes = fit_frame(gray, sh)
    rect = rectify(gray, H, ppmm)
    dots = find_dots(rect, sh, ppmm, mmpx)
    hits, extra = assign(sh, dots, max_mm)
    return Result(sh.name, str(image_path), hits, extra, npts, rms, mmpx, notes), rect


def write_csv(res: Result, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "source", "target", "nom_x_mm", "nom_y_mm", "hit_x_mm", "hit_y_mm", "dx_mm", "dy_mm",
                    "err_mm", "dot_area_mm2", "fit_points", "fit_rms_mm", "mm_per_px"])
        for h in res.hits:
            e = h.err
            w.writerow([res.sheet, res.source, h.n, f"{h.nom[0]:.2f}", f"{h.nom[1]:.2f}",
                        "" if h.hit is None else f"{h.hit[0]:.3f}", "" if h.hit is None else f"{h.hit[1]:.3f}",
                        "" if e is None else f"{e[0]:.3f}", "" if e is None else f"{e[1]:.3f}",
                        "" if e is None else f"{math.hypot(*e):.3f}",
                        "" if h.hit is None else f"{h.area_mm2:.2f}", res.fit_points, f"{res.fit_rms_mm:.3f}",
                        f"{res.mm_per_px:.4f}"])


def stats(errs: List[Tuple[float, float]]) -> Dict[str, float]:
    if not errs:
        return {"n": 0}
    E = np.array(errs)
    r = np.hypot(E[:, 0], E[:, 1])
    return {"n": len(E), "bias_x_mm": float(E[:, 0].mean()), "bias_y_mm": float(E[:, 1].mean()),
            "sd_x_mm": float(E[:, 0].std(ddof=1)) if len(E) > 1 else float("nan"),
            "sd_y_mm": float(E[:, 1].std(ddof=1)) if len(E) > 1 else float("nan"),
            "median_mm": float(np.median(r)), "rmse_mm": float(np.sqrt(np.mean(r ** 2))),
            "r95_mm": float(np.percentile(r, 95)), "max_mm": float(r.max())}


def annotate(rect, res: Result, ppmm: float, gain: float = 1.0) -> np.ndarray:
    """Rectified page with nominal crosses (blue), hits (red) and the error vector."""
    out = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
    P = lambda x, y: (int(round(x * ppmm)), int(round(y * ppmm)))
    for h in res.hits:
        c = P(*h.nom)
        cv2.drawMarker(out, c, (255, 80, 0), cv2.MARKER_CROSS, int(6 * ppmm), 2)
        cv2.putText(out, str(h.n), (c[0] + 15, c[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 0), 2)
        if h.hit is not None:
            e = (h.hit[0] - h.nom[0], h.hit[1] - h.nom[1])
            cv2.circle(out, P(*h.hit), int(1.5 * ppmm), (0, 0, 255), 2)
            cv2.line(out, c, P(h.nom[0] + gain * e[0], h.nom[1] + gain * e[1]), (0, 0, 255), 2)
            cv2.putText(out, f"{math.hypot(*e):.1f}", (c[0] + 15, c[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)
    for d in res.extra_dots:
        cv2.circle(out, P(*d), int(2 * ppmm), (0, 160, 255), 2)
    return out
