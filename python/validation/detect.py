"""Independent localisation of the printed reference points in a raw frame.

The analysis never uses EyeLab's output to find the ground truth (that was the
failure of the first draft: seeded by EyeLab's projection, a large EyeLab error
made the detector lock onto a checker-square corner and report a small one).
Instead, per face:

1. detect the plate's ArUco markers with the analysis' own detector;
2. homography face-plane -> undistorted image from that face's marker corners
   (a face whose markers are not visible is seeded from a PnP solve on the
   markers that are, e.g. the grazing face of the folded plate);
3. predict every saddle, search a small radius around the prediction, accept
   only a sub-pixel point that is a true X-junction (4 alternating sectors);
4. refit the homography with the accepted saddles and repeat, so points far
   from a visible marker are reached by growth rather than extrapolation.

The final per-face homography is fitted to the detected saddles; it maps any
image point to the face plane (the mm conversion in ``metrics``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from .camera import apply_h, distort, fit_homography, undistort

_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)


def detect_saddle(gray: np.ndarray, seed_uv, search_radius: float, win: int = 3,
                  sigma: float = 1.5) -> Optional[np.ndarray]:
    """Strongest saddle (-det Hessian) within ``search_radius`` of the seed,
    refined by cornerSubPix; None if nothing, or if refinement leaves the radius."""
    h, w = gray.shape[:2]
    u, v = float(seed_uv[0]), float(seed_uv[1])
    r = int(np.ceil(search_radius)) + 2 * win
    x0, x1 = max(int(u) - r, 0), min(int(u) + r + 1, w)
    y0, y1 = max(int(v) - r, 0), min(int(v) + r + 1, h)
    if x1 - x0 < 2 * win + 3 or y1 - y0 < 2 * win + 3:
        return None
    g = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (0, 0), sigma)
    ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    resp = -(ixx * iyy - ixy * ixy)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    resp[(xx - u) ** 2 + (yy - v) ** 2 > search_radius ** 2] = -np.inf
    if not np.isfinite(resp).any() or resp.max() <= 0:
        return None
    cy, cx = np.unravel_index(int(np.argmax(resp)), resp.shape)
    pt = np.array([[[x0 + cx, y0 + cy]]], dtype=np.float32)
    cv2.cornerSubPix(gray, pt, (win, win), (-1, -1), _CRITERIA)
    p = pt[0, 0].astype(float)
    if np.hypot(p[0] - u, p[1] - v) > search_radius:
        return None
    return p


def is_x_junction(gray: np.ndarray, pt, rho: float, n: int = 64,
                  min_contrast: float = 30.0) -> bool:
    """Ring test: an X-junction shows 4 dark/bright transitions and half dark
    (exactly, under any local affine view); an L-corner or an edge shows 2."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    mx = (pt[0] + rho * np.cos(a)).astype(np.float32).reshape(1, -1)
    my = (pt[1] + rho * np.sin(a)).astype(np.float32).reshape(1, -1)
    s = cv2.remap(gray.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE).ravel()
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < min_contrast:
        return False
    b = s < 0.5 * (lo + hi)
    transitions = int(np.count_nonzero(b != np.roll(b, 1)))
    return transitions == 4 and 0.3 <= b.mean() <= 0.7


def detect_markers(gray: np.ndarray, dictionary_id: int) -> Dict[int, np.ndarray]:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary_id), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


@dataclass
class Located:
    points: Dict[str, np.ndarray] = field(default_factory=dict)       # raw px
    seed_offset_px: Dict[str, float] = field(default_factory=dict)
    homographies: Dict[str, np.ndarray] = field(default_factory=dict)  # face m -> undist px
    residual_px: Dict[str, np.ndarray] = field(default_factory=dict)   # per face, saddles
    markers: Dict[int, np.ndarray] = field(default_factory=dict)


def locate_points(gray: np.ndarray, target, K, dist, search_radius: float = 5.0,
                  win: int = 3, max_rounds: int = 8) -> Located:
    K = np.asarray(K, float)
    dist = np.asarray(dist, float)
    h, w = gray.shape[:2]
    square_m = float(target.provenance.get("square_mm", 15.0)) / 1000.0
    loc = Located(markers=detect_markers(gray, target.aruco_dictionary_id()))
    seen = [m for m in target.markers if m.id in loc.markers]
    if not seen:
        return loc
    pnp = _pnp_from_markers(seen, loc.markers, K, dist)
    for face in target.faces:
        ids = target.face_ids(face)
        uv_face = dict(zip(ids, target.face_coords(ids)))
        m_src, m_dst = [], []
        for m in target.markers:
            if m.face == face and m.id in loc.markers:
                m_src.append(target.to_face(face, m.corners()))
                m_dst.append(undistort(loc.markers[m.id], K, dist))
        if m_src:
            m_src, m_dst = np.vstack(m_src), np.vstack(m_dst)
            H = fit_homography(m_src, m_dst)
        elif pnp is not None:
            m_src, m_dst = np.zeros((0, 2)), np.zeros((0, 2))
            pred = cv2.projectPoints(target.array(ids), pnp[0], pnp[1], K, dist)[0].reshape(-1, 2)
            H = fit_homography(np.array([uv_face[i] for i in ids]), undistort(pred, K, dist))
        else:
            continue
        found: Dict[str, np.ndarray] = {}
        for _ in range(max_rounds):
            if H is None:
                break
            new = 0
            for pid in ids:
                if pid in found:
                    continue
                p = uv_face[pid]
                q = apply_h(H, np.array([p, p + [square_m, 0], p + [0, square_m]]))
                sq_px = float(min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[0])))
                if sq_px < 2 * win + 1:
                    continue
                seed = distort(q[:1], K, dist)[0]
                if not (0 <= seed[0] < w and 0 <= seed[1] < h):
                    continue
                r = min(search_radius, 0.3 * sq_px)
                pt = detect_saddle(gray, seed, r, win, sigma=float(np.clip(sq_px / 10, 1.0, 2.0)))
                if pt is None or not is_x_junction(gray, pt, float(np.clip(0.4 * sq_px, 2.5, 12.0))):
                    continue
                found[pid] = pt
                loc.seed_offset_px[pid] = float(np.linalg.norm(pt - seed))
                new += 1
            if new == 0:
                break
            f_ids = list(found)
            s_src = np.array([uv_face[i] for i in f_ids])
            s_dst = undistort(np.array([found[i] for i in f_ids]), K, dist)
            H = fit_homography(np.vstack([m_src, s_src]), np.vstack([m_dst, s_dst]))
        if not found:
            continue
        f_ids = list(found)
        s_src = np.array([uv_face[i] for i in f_ids])
        s_dst = undistort(np.array([found[i] for i in f_ids]), K, dist)
        H_final = fit_homography(s_src, s_dst)
        if H_final is None:
            H_final = fit_homography(np.vstack([m_src, s_src]), np.vstack([m_dst, s_dst]))
        if H_final is None:
            continue
        loc.points.update(found)
        loc.homographies[face] = H_final
        loc.residual_px[face] = np.linalg.norm(apply_h(H_final, s_src) - s_dst, axis=1)
    return loc


def _pnp_from_markers(markers, detected, K, dist):
    """Plate pose from all visible markers (seeding only; not EyeLab's pose)."""
    obj = np.vstack([m.corners() for m in markers])
    img = np.vstack([detected[m.id] for m in markers])
    try:
        if len({tuple(np.round(m.normal, 6)) for m in markers}) == 1:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            return (rvecs[0], tvecs[0]) if n else None
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        return (rvec, tvec) if ok else None
    except cv2.error:
        return None


def apply_overrides(loc: Located, target, K, dist, overrides) -> Located:
    """Replace automatic saddles with reviewed ones (``gt_review``; ``None`` removes a
    point) and refit each touched face's homography from the resulting saddles, as
    ``locate_points`` does for its final fit. ``loc`` is changed in place."""
    touched = set()
    for pid, uv in (overrides or {}).items():
        if pid not in target.point_face:
            continue
        touched.add(target.point_face[pid])
        if uv is None:
            loc.points.pop(pid, None)
        else:
            loc.points[pid] = np.asarray(uv, float)
        loc.seed_offset_px.pop(pid, None)
    for face in touched:
        ids = [p for p in target.face_ids(face) if p in loc.points]
        src = target.face_coords(ids) if ids else np.zeros((0, 2))
        dst = undistort(np.array([loc.points[p] for p in ids]), K, dist) if ids else np.zeros((0, 2))
        H = fit_homography(src, dst) if len(ids) >= 4 else None
        if H is None:
            loc.homographies.pop(face, None)
            loc.residual_px.pop(face, None)
            continue
        loc.homographies[face] = H
        loc.residual_px[face] = np.linalg.norm(apply_h(H, src) - dst, axis=1)
    return loc
