"""Rasterise a plate layout: used by the printable PDF and by the synthetic renderer.

Every edge is placed by rounding its own page-mm coordinate to the pixel grid
(no accumulated error): at 600 dpi the quantisation is <= 0.021 mm, far below
caliper uncertainty. Pixel ``i`` covers page mm ``[i/ppm, (i+1)/ppm)``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from .plates import PageLayout


def _px(v_mm: float, ppm: float) -> int:
    return int(round(v_mm * ppm))


def _fill(img, x0, y0, x1, y1, ppm, value=0):
    img[_px(y0, ppm):_px(y1, ppm), _px(x0, ppm):_px(x1, ppm)] = value


def marker_bits(dictionary: str, marker_id: int) -> np.ndarray:
    """(n+2)x(n+2) cell matrix (border included), 0 = black, 255 = white."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    cells = d.markerSize + 2
    return cv2.aruco.generateImageMarker(d, int(marker_id), cells, borderBits=1)


def layout_digest(layout: PageLayout) -> str:
    return hashlib.sha256(json.dumps(asdict(layout), sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def _text(img, s, x_mm, y_mm, h_mm, ppm):
    scale = h_mm * ppm / 22.0
    cv2.putText(img, s, (_px(x_mm, ppm), _px(y_mm, ppm)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0,
                max(1, int(round(0.12 * h_mm * ppm))), cv2.LINE_AA)


def render_page(layout: PageLayout, ppm: float, annotate: bool = True,
                face: Optional[str] = None) -> np.ndarray:
    """uint8 page raster (255 white). ``face`` renders that face's features only."""
    W, H = layout.page_mm
    img = np.full((_px(H, ppm), _px(W, ppm)), 255, np.uint8)
    s = layout.square_mm
    for sd in layout.saddles:
        if face is None or sd.face == face:
            _fill(img, sd.x - s, sd.y - s, sd.x, sd.y, ppm)
            _fill(img, sd.x, sd.y, sd.x + s, sd.y + s, ppm)
    for m in layout.markers:
        if face is not None and m.face != face:
            continue
        bits = marker_bits(layout.dictionary, m.id)
        n = bits.shape[0]
        cell = m.size_mm / n
        x0, y0 = m.x - m.size_mm / 2, m.y - m.size_mm / 2
        for r in range(n):
            for c in range(n):
                if bits[r, c] == 0:
                    _fill(img, x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell,
                          y0 + (r + 1) * cell, ppm)
    if not annotate:
        return img
    lw = 0.35
    for b in layout.bars:
        if b.name == "x":
            _fill(img, b.x, b.y, b.x + b.length_mm, b.y + lw, ppm)
            for k in range(int(b.length_mm // 10) + 1):
                t = 4.0 if k in (0, int(b.length_mm // 10)) else 2.0
                _fill(img, b.x + 10 * k, b.y - t, b.x + 10 * k + lw, b.y, ppm)
            _text(img, f"X {b.length_mm:.0f} mm. Both bars: caliper from one edge of the first "
                       f"tick to the SAME edge of the last tick", b.x, b.y + 5.5, 2.6, ppm)
        else:
            _fill(img, b.x, b.y, b.x + lw, b.y + b.length_mm, ppm)
            for k in range(int(b.length_mm // 10) + 1):
                t = 4.0 if k in (0, int(b.length_mm // 10)) else 2.0
                _fill(img, b.x - t, b.y + 10 * k, b.x, b.y + 10 * k + lw, ppm)
            _text(img, "Y", b.x - 3.5, b.y - 2.0, 2.6, ppm)
            _text(img, f"{b.length_mm:.0f}", b.x - 5.5, b.y + b.length_mm + 5.0, 2.6, ppm)
    for sd in layout.saddles:
        _text(img, str(sd.id), sd.x + 16.0, sd.y - 16.5, 3.0, ppm)
    for m in layout.markers:
        _text(img, f"id {m.id}", m.x - m.size_mm / 2, m.y + m.size_mm / 2 + 4.5, 2.6, ppm)
    if layout.fold_y is not None:
        for x0 in (2.0, layout.page_mm[0] - 9.0):
            _fill(img, x0, layout.fold_y - lw / 2, x0 + 7.0, layout.fold_y + lw / 2, ppm)
    ids = ",".join(str(m.id) for m in layout.markers)
    sizes = {m.size_mm for m in layout.markers}
    head = (f"EyeLab validation {layout.title} | {layout.dictionary} ids {ids}, "
            f"{'/'.join(f'{v:g}' for v in sorted(sizes))} mm | layout {layout_digest(layout)}")
    _text(img, head, 45.0, 9.0, 3.0, ppm)
    _text(img, "Print A4 landscape at 100 % (actual size, no fit/shrink). Measure both "
               "scale bars with calipers.", 45.0, 14.5, 2.6, ppm)
    return img
