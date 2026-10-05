"""Synthetic scenes with exact ground truth, to verify the analysis and the
whole capture chain.

``render_scene`` warps each face of the printed page into a pinhole camera
(no distortion). ``generate_session`` fakes an EyeLab pose (true + bias +
noise) to check that the analysis recovers a known error;
``test_capture_e2e`` instead feeds rendered frames to the real
``ArucoPipeline``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .plates import PageLayout
from .recorder import SessionRecorder
from .render import render_page
from .target import Target


def look_at(eye, center, up) -> tuple:
    """World->camera (rvec, tvec) for a camera at ``eye`` looking at ``center``;
    ``up`` appears as image-up."""
    z = np.asarray(center, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel(), -R @ np.asarray(eye, float)


def view_pose(target: Target, distance_m: float, tilt_deg: float = 0.0, pan_deg: float = 0.0):
    """Camera facing the target's points. Planar plate: along +Z, image-up = +Y.
    Folded plate: from the bisector of its two faces, image-up = +Z. ``tilt``
    rotates the viewpoint about X, ``pan`` about the image-up axis."""
    c = target.array(target.ids()).mean(axis=0)
    normals = [f.normal for f in target.faces.values()]
    d = np.sum(normals, axis=0)
    d /= np.linalg.norm(d)
    up = np.array([0.0, 1.0, 0.0]) if len(normals) == 1 else np.array([0.0, 0.0, 1.0])
    Rt = cv2.Rodrigues(np.array([np.radians(tilt_deg), 0, 0]))[0]
    Rp = cv2.Rodrigues(up * np.radians(pan_deg))[0]
    d = Rp @ Rt @ d
    return look_at(c + distance_m * d, c, Rp @ Rt @ up)


def render_scene(layout: PageLayout, target: Target, K, rvec, tvec, image_size=(1280, 720),
                 ppm: float = 12.0, background: int = 128) -> np.ndarray:
    K = np.asarray(K, float)
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    t = np.asarray(tvec, float).ravel()
    W, H = image_size
    out = np.full((H, W), float(background), np.float32)
    for pf in layout.faces:
        f = target.faces[pf.key]
        o_c = R @ f.origin + t
        if np.dot(R @ f.normal, o_c) >= 0:          # back-facing
            continue
        ax, ay = pf.anchor
        A = np.array([[1 / (1000 * ppm), 0, (0.5 / ppm - ax) / 1000],
                      [0, -1 / (1000 * ppm), (ay - 0.5 / ppm) / 1000], [0, 0, 1]])
        Hm = K @ np.column_stack([R @ f.ex, R @ f.ey, o_c]) @ A
        page = render_page(layout, ppm, annotate=False, face=pf.key).astype(np.float32)
        scale = K[0, 0] / (o_c[2] * 1000.0)          # image px per mm, approx.
        sigma = 0.5 * ppm / scale
        if sigma > 0.3:
            page = cv2.GaussianBlur(page, (0, 0), sigma)
        mask = np.zeros(page.shape, np.float32)
        x0, y0, x1, y1 = pf.region
        mask[int(round(y0 * ppm)):int(round(y1 * ppm)), int(round(x0 * ppm)):int(round(x1 * ppm))] = 1
        warped = cv2.warpPerspective(page, Hm, (W, H), flags=cv2.INTER_LINEAR)
        m = cv2.warpPerspective(mask, Hm, (W, H), flags=cv2.INTER_LINEAR)
        out = m * warped + (1 - m) * out
    return out


def generate_session(out_dir, layout: PageLayout, target: Target, K, rvec, tvec,
                     n_frames: int = 10, rot_noise_deg: float = 0.0, trans_noise_mm: float = 0.0,
                     bias_mm=(0.0, 0.0, 0.0), pixel_noise: float = 1.0, image_size=(1280, 720),
                     seed: int = 0) -> Path:
    """Session whose 'EyeLab' pose is the true pose + bias + noise (camera frame, mm)."""
    rng = np.random.default_rng(seed)
    K = np.asarray(K, float)
    dist = np.zeros(5)
    clean = render_scene(layout, target, K, rvec, tvec, image_size)
    R_true = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    ids = target.ids()
    obj = target.array(ids)
    mids = target.marker_ids()
    cond = {"rot_noise_deg": rot_noise_deg, "trans_noise_mm": trans_noise_mm,
            "bias_mm": list(bias_mm), "pixel_noise": pixel_noise}
    with SessionRecorder(out_dir, K, dist, image_size, "synthetic", cond) as rec:
        for _ in range(n_frames):
            img = np.clip(clean + rng.normal(0, pixel_noise, clean.shape), 0, 255).astype(np.uint8)
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            dR = cv2.Rodrigues(a * np.radians(rng.normal(0, rot_noise_deg)))[0]
            r_est = cv2.Rodrigues(dR @ R_true)[0].ravel()
            t_est = (np.asarray(tvec, float).ravel()
                     + (np.asarray(bias_mm, float) + rng.normal(0, trans_noise_mm, 3)) / 1000.0)
            proj = cv2.projectPoints(obj, r_est, t_est, K, dist)[0].reshape(-1, 2)
            pts = dict(zip(ids, proj))
            pose = {"rvec": r_est, "tvec": t_est, "marker_ids": mids}
            rec.record(img, pose=pose, measured=pose, lock_state="locked", markers_seen=mids,
                       projected=pts, projected_raw=pts, drawn={k: np.round(v) for k, v in pts.items()})
    return Path(out_dir)
