#!/usr/bin/env python3
"""
Pure wireframe-projection math for the AR overlay.

This module is deliberately free of any UI / tkinter dependency: it takes
geometry nodes plus a camera pose (or a node->world transform) and returns
2D pixel coordinates. Callers do their own drawing. That keeps the projection
logic reusable by both the desktop GUI (``eyelab_gui.py``) and the future
off-device EyeLab Service (ADR-001, Step C), which has no display to draw to.

Functions
    project_nodes      — UNV nodes -> image pixels (board-pose or registration path).
    wireframe_segments — node pixels + edge list -> drawable pixel segment pairs.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

Pixel = tuple[int, int]
Segment = tuple[Pixel, Pixel]

_ZERO_VEC = np.zeros((3, 1), dtype=np.float32)


def project_nodes(
    nodes: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    rvec: np.ndarray = _ZERO_VEC,
    tvec: np.ndarray = _ZERO_VEC,
    node_transform: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
) -> dict[int, Pixel]:
    """Project UNV geometry nodes onto the image plane.

    Args:
        nodes: list of ``{"id", "x", "y", "z"}`` dicts in the UNV/model frame (metres).
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: distortion coefficients, or None.
        rvec, tvec: camera pose used by ``cv2.projectPoints`` (default identity).
        node_transform: optional callable mapping a (3,) UNV point to a (3,) world
            point (e.g. ``SpatialRegistration.transform_point``). When given, each
            node is mapped to world first and then projected with ``rvec``/``tvec``
            (left at identity for the registration path, where points are already
            expressed in the camera frame). Nodes for which the transform returns
            ``None`` are skipped. When omitted, nodes are projected directly — the
            board-pose path, where ``rvec``/``tvec`` carry the detected board pose.

    Returns:
        ``{node_id: (px, py)}`` for every node that projected successfully.
    """
    ids: list[int] = []
    pts: list[list[float]] = []
    for n in nodes:
        p = np.array([n["x"], n["y"], n["z"]], dtype=np.float64)
        if node_transform is not None:
            world = node_transform(p)
            if world is None:
                continue
            p = np.asarray(world, dtype=np.float64).reshape(3)
        ids.append(int(n["id"]))
        pts.append([float(p[0]), float(p[1]), float(p[2])])

    if not pts:
        return {}

    obj = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 3)
    image_pts, _ = cv2.projectPoints(
        obj,
        np.asarray(rvec, dtype=np.float32),
        np.asarray(tvec, dtype=np.float32),
        camera_matrix,
        dist_coeffs,
    )
    image_pts = image_pts.reshape(-1, 2)
    return {
        nid: (int(round(image_pts[i, 0])), int(round(image_pts[i, 1])))
        for i, nid in enumerate(ids)
    }


def wireframe_segments(
    node_px: dict[int, Pixel],
    edges: list,
) -> list[Segment]:
    """Build drawable pixel segments from projected nodes and an edge list.

    Each edge is a 2-tuple/list of node IDs. Edges whose endpoints did not both
    project are dropped.
    """
    segments: list[Segment] = []
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        if a in node_px and b in node_px:
            segments.append((node_px[a], node_px[b]))
    return segments
