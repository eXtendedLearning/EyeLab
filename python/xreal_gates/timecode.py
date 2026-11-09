"""Optical timecode for the camera-to-app latency (T5.1 Phase 1). ``python -m xreal_gates timecode``.

The Surface's own screen shows one large ArUco marker (DICT_6X6_1000) whose id is a draw counter
mod 1000, redrawn as fast as the window allows (about once per display refresh). Every draw is logged
with its QPC time (``time.perf_counter()``, which on Windows is the raw QueryPerformanceCounter in
seconds; Unity logs ``Stopwatch.GetTimestamp()`` on the same clock) to
``GateRuns/timecode/timecode_<stamp>.csv``.

The gate looks at the screen with the Eye; the pose service, run with ``--target timecode``, reads the
id in every frame it is sent (:class:`TimecodeTarget`). ``python -m xreal_gates stages RUN`` then pairs
each frame's id with the draw that showed it: frame arrival minus draw time is the camera-to-app
latency. It includes the Surface panel's own presentation delay (Windows composition and scan-out,
typically one to two refreshes), so it is an upper bound on exposure-to-arrival; the optional phone
slow-motion film of glasses and screen (T5.1) calibrates that part.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .targets import GateTarget, PoseEstimate

DICT = cv2.aruco.DICT_6X6_1000
IDS = 1000
WINDOW = "EyeLab timecode"


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(DICT)


def marker_image(marker_id: int, size: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(_dictionary(), int(marker_id) % IDS, size)


def frame_for(counter: int, width: int, height: int, img: Optional[np.ndarray] = None) -> np.ndarray:
    """White canvas with the marker for ``counter`` centred (quiet zone included) and a small text clock."""
    if img is None or img.shape != (height, width):
        img = np.empty((height, width), np.uint8)
    img.fill(255)
    side = int(min(width, height) * 0.8)
    y0, x0 = (height - side) // 2, (width - side) // 2
    img[y0:y0 + side, x0:x0 + side] = marker_image(counter % IDS, side)
    cv2.putText(img, f"{counter % IDS:03d}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return img


def run_display(out_dir: Path, seconds: float = 0.0, width: int = 1280, height: int = 720) -> Path:
    """Fullscreen timecode until Esc/q (or ``seconds``); returns the CSV path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"timecode_{datetime.now():%Y%m%d_%H%M%S}.csv"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:  # stay in front of the gate player's window (OpenCV 4.6+; older builds ignore it)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except (AttributeError, cv2.error):
        pass
    canvas = None
    t_end = time.perf_counter() + seconds if seconds > 0 else float("inf")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qpc_draw", "qpc_shown", "counter", "id"])
        counter = 0
        print(f"timecode: {path}; Esc or q ends")
        while time.perf_counter() < t_end:
            canvas = frame_for(counter, width, height, canvas)
            t0 = time.perf_counter()
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            t1 = time.perf_counter()
            w.writerow([f"{t0:.6f}", f"{t1:.6f}", counter, counter % IDS])
            counter += 1
            if key in (27, ord("q")):
                break
    cv2.destroyWindow(WINDOW)
    return path


class TimecodeTarget(GateTarget):
    """Reads the timecode id in a frame. Never a pose (``ok`` is False); the id is in ``marker_ids``."""

    name = "timecode"
    label = "Surface-screen timecode (DICT_6X6_1000 draw counter)"
    source = "xreal_gates.timecode"

    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(_dictionary(), cv2.aruco.DetectorParameters())

    def estimate(self, gray, K, dist) -> PoseEstimate:
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return PoseEstimate(False, "no timecode", n_markers=0)
        found = [int(i) for i in np.ravel(ids)]
        sizes = [cv2.contourArea(np.asarray(c, np.float32).reshape(-1, 2)) for c in corners]
        best = found[int(np.argmax(sizes))]
        est = PoseEstimate(False, f"timecode {best}", n_markers=1, marker_ids=[best])
        est.set_image_points(np.asarray(corners[int(np.argmax(sizes))]).reshape(-1, 2))
        return est

    def overlay(self) -> dict:
        return {"segments": [], "points": []}


def load_draws(paths: List[Path]) -> List[Tuple[float, float, int]]:
    """(qpc_draw, qpc_shown, id) of every draw in the given timecode CSVs, in time order."""
    out = []
    for p in paths:
        with Path(p).open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                out.append((float(r["qpc_draw"]), float(r["qpc_shown"]), int(r["id"])))
    out.sort()
    return out


def draw_time(draws: List[Tuple[float, float, int]], index: Dict[int, List[int]], marker_id: int,
              before: float, horizon: float = 5.0) -> Optional[Tuple[float, float]]:
    """(draw, shown) QPC of the latest draw of ``marker_id`` before ``before`` (within ``horizon`` s)."""
    best = None
    for i in index.get(marker_id, ()):
        t0, t1, _ = draws[i]
        if t0 <= before and before - t0 <= horizon:
            best = (t0, t1)
    return best


def index_draws(draws: List[Tuple[float, float, int]]) -> Dict[int, List[int]]:
    idx: Dict[int, List[int]] = {}
    for i, (_, _, mid) in enumerate(draws):
        idx.setdefault(mid, []).append(i)
    return idx
