"""Loader for sessions written by ``recorder.SessionRecorder`` (schema 2)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .recorder import SCHEMA_VERSION


@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray            # metres
    marker_ids: Tuple[int, ...]
    coasted: bool = False
    rms_reproj_px: Optional[float] = None


@dataclass
class FrameRecord:
    index: int
    timestamp: float
    frame: Optional[str]
    pose: Optional[Pose]        # what EyeLab draws
    measured: Optional[Pose]    # raw solve of this frame
    lock_state: Optional[str]
    markers_seen: Tuple[int, ...]
    projected: Dict[str, np.ndarray]
    projected_raw: Dict[str, np.ndarray]
    drawn: Dict[str, np.ndarray]
    timings_ms: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # setup geometry / IMU (recorder ``extra``)
    source: dict = field(default_factory=dict)  # source frame identity (session_id, sequence)

    @property
    def n_markers_measured(self) -> int:
        return len(self.measured.marker_ids) if self.measured else 0


@dataclass
class Session:
    path: Path
    meta: dict
    records: List[FrameRecord]

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.asarray(self.meta["camera_matrix"], float)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.meta["dist_coeffs"], float)

    def read_gray(self, rec: FrameRecord) -> np.ndarray:
        img = cv2.imread(str(self.path / "frames" / rec.frame), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.path / "frames" / rec.frame)
        return img

    def poses(self, which: str = "measured"):
        """(rvecs, tvecs) of frames that have that pose; ``which``: measured|pose."""
        ps = [getattr(r, which) for r in self.records if getattr(r, which) is not None]
        return (np.array([p.rvec for p in ps]).reshape(-1, 3),
                np.array([p.tvec for p in ps]).reshape(-1, 3))


def _pose(d) -> Optional[Pose]:
    if not d:
        return None
    return Pose(np.asarray(d["rvec"], float), np.asarray(d["tvec"], float),
                tuple(d.get("marker_ids", [])), bool(d.get("coasted", False)),
                d.get("rms_reproj_px"))


def _pts(d) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, float) for k, v in (d or {}).items()}


def load_session(path) -> Session:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {meta.get('schema_version')}")
    if meta.get("length_unit") != "m":
        raise ValueError(f"{path}: length_unit must be 'm'")
    records = []
    with open(path / "log.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            records.append(FrameRecord(
                int(e["i"]), float(e["t"]), e.get("frame"), _pose(e.get("pose")),
                _pose(e.get("measured")), e.get("lock_state"), tuple(e.get("markers_seen", [])),
                _pts(e.get("projected")), _pts(e.get("projected_raw")), _pts(e.get("drawn")),
                e.get("timings_ms", {}), e.get("extra", {}) or {},
                e.get("source", {}) or {}))
    return Session(path, meta, records)
