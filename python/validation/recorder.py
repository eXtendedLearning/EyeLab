"""Session recorder (schema 2). Plain values in, JSON lines + PNG frames out.

Pose convention: ``rvec``/``tvec`` map MODEL coordinates (the target / UNV
frame, metres) into CAMERA coordinates, i.e. ``cv2.solvePnP`` output, which is
what ``ArucoPipeline`` returns. ``pose`` is what EyeLab draws (after the pose
lock / Kalman filter, possibly coasted); ``measured`` is the same frame's raw
solve. ``projected`` / ``projected_raw`` are float pixel projections of the
target points under ``pose`` / ``measured`` (raw, distorted image pixels);
``drawn`` are the integer pixels ``overlay.project_nodes`` actually draws.

PNG encoding runs on a writer thread so a capture loop is not blocked by it.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np

SCHEMA_VERSION = 2


def _pose(p: Optional[dict]) -> Optional[dict]:
    if p is None:
        return None
    out = {"rvec": np.asarray(p["rvec"], float).ravel().tolist(),
           "tvec": np.asarray(p["tvec"], float).ravel().tolist(),
           "marker_ids": [int(m) for m in p.get("marker_ids", [])]}
    for k in ("coasted", "rms_reproj_px"):
        if p.get(k) is not None:
            out[k] = p[k] if k == "coasted" else float(p[k])
    return out


def _pts(d: Optional[Dict[object, Sequence[float]]]) -> dict:
    return {str(k): [float(v[0]), float(v[1])] for k, v in (d or {}).items()}


class SessionRecorder:
    def __init__(self, out_dir, camera_matrix, dist_coeffs, image_size: Sequence[int],
                 platform: str, condition: Optional[dict] = None, extra_meta: Optional[dict] = None,
                 frame_ext: str = ".png", queue_size: int = 64):
        self.dir = Path(out_dir)
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.frame_ext = frame_ext
        meta = {
            "schema_version": SCHEMA_VERSION, "platform": platform, "length_unit": "m",
            "image_size": [int(image_size[0]), int(image_size[1])],
            "camera_matrix": np.asarray(camera_matrix, float).reshape(3, 3).tolist(),
            "dist_coeffs": np.asarray(dist_coeffs, float).ravel().tolist(),
            "condition": condition or {}, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta.update(extra_meta or {})
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._log = open(self.dir / "log.jsonl", "w", encoding="utf-8")
        self._idx = 0
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._error: Optional[BaseException] = None
        self._writer = threading.Thread(target=self._write_frames, daemon=True)
        self._writer.start()

    def _write_frames(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, frame = item
            try:
                if not cv2.imwrite(str(path), frame):
                    raise IOError(f"could not write {path}")
            except BaseException as e:  # surfaced on the next record()/close()
                self._error = e

    def record(self, frame: Optional[np.ndarray], timestamp: Optional[float] = None,
               pose: Optional[dict] = None, measured: Optional[dict] = None,
               lock_state: Optional[str] = None, lock_reject: Optional[str] = None,
               markers_seen: Iterable[int] = (), projected=None, projected_raw=None, drawn=None,
               timings_ms: Optional[dict] = None, extra: Optional[dict] = None) -> int:
        """Log one frame. ``pose``/``measured``: dicts with rvec, tvec, marker_ids
        (+ coasted / rms_reproj_px), or None. ``extra``: JSON-safe per-frame
        side data (setup geometry, IMU), stored under ``"extra"``."""
        if self._error is not None:
            raise self._error
        idx = self._idx
        name = None
        if frame is not None:
            name = f"{idx:06d}{self.frame_ext}"
            self._q.put((self.frames_dir / name, np.array(frame, copy=True)))
        entry = {"i": idx, "t": float(time.perf_counter() if timestamp is None else timestamp),
                 "frame": name, "lock_state": lock_state, "lock_reject": lock_reject,
                 "pose": _pose(pose), "measured": _pose(measured),
                 "markers_seen": [int(m) for m in markers_seen],
                 "projected": _pts(projected), "projected_raw": _pts(projected_raw),
                 "drawn": _pts(drawn), "timings_ms": timings_ms or {}}
        if extra:
            entry["extra"] = extra
        self._log.write(json.dumps(entry) + "\n")
        self._idx += 1
        return idx

    def close(self) -> None:
        if self._log.closed:
            return
        self._q.put(None)
        self._writer.join()
        self._log.close()
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
