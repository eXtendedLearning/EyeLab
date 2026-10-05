"""Pixel-geometry helpers: (un)distortion and planar homographies."""
from __future__ import annotations

import cv2
import numpy as np


def undistort(pts, K, dist) -> np.ndarray:
    """Raw (distorted) pixels -> ideal pinhole pixels, same K."""
    p = np.asarray(pts, float).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    return cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 2)


def distort(pts, K, dist) -> np.ndarray:
    """Ideal pinhole pixels -> raw (distorted) pixels, same K."""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    xn = (p - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    obj = np.column_stack([xn, np.ones(len(p))])
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)
    return out.reshape(-1, 2)


def fit_homography(src, dst):
    """Least-squares (DLT + LM) homography src (N,2) -> dst (N,2); None if degenerate."""
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    if len(src) < 4 or np.linalg.matrix_rank(src - src.mean(0), tol=1e-9) < 2:
        return None
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def apply_h(H, pts) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]
