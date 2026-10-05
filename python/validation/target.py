"""Ground-truth target: reference points, faces and ArUco markers of a plate.

One file defines everything that must agree physically: the points the
analysis measures, the markers EyeLab registers to, and the planar faces they
lie on. The EyeLab marker config and the UNV geometry are *derived* from it
(``export_marker_config``, ``export_unv``), never edited separately.

All coordinates are metres, in the target's model frame (= the UNV frame
EyeLab registers to). Each face is planar; ``face_coords`` gives a point's 2D
position on its face, which is what the per-face homographies in ``metrics``
use as ground truth. That 2D ground truth comes from the print alone, so it
does not depend on how accurately a folded plate was assembled.

JSON schema ``eyelab-validation-target/2``::

    {"schema": ..., "name": "plate_A", "length_unit": "m",
     "uncertainty_mm": 0.1, "aruco_dictionary": "DICT_4X4_50",
     "provenance": {...},
     "faces":   {"F": {"origin": [x,y,z], "ex": [..], "ey": [..]}},
     "markers": [{"id": 30, "face": "F", "center": [..], "normal": [..],
                  "roll_deg": 0.0, "size_m": 0.02}],
     "points":  {"1": {"xyz": [..], "face": "F"}},
     "edges":   [[1, 2], ...]}
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCHEMA = "eyelab-validation-target/2"
M_TO_MM = 1000.0


@dataclass
class Face:
    origin: np.ndarray  # (3,) m
    ex: np.ndarray      # (3,) unit, face "right"
    ey: np.ndarray      # (3,) unit, face "up"

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.ex, self.ey)
        return n / np.linalg.norm(n)


@dataclass
class Marker:
    id: int
    face: str
    center: np.ndarray  # (3,) m
    normal: np.ndarray  # (3,) unit
    roll_deg: float
    size_m: float

    def corners(self) -> np.ndarray:
        """(4, 3) corners in OpenCV order TL, TR, BR, BL (EyeLab's definition)."""
        from registration import marker_object_corners
        return np.asarray(marker_object_corners(self.center, self.normal, self.roll_deg,
                                                self.size_m), float)


@dataclass
class Target:
    name: str
    faces: Dict[str, Face]
    markers: List[Marker]
    points: Dict[str, np.ndarray]           # id -> (3,) m
    point_face: Dict[str, str]
    edges: List[Tuple[int, int]] = field(default_factory=list)
    uncertainty_mm: float = float("nan")
    aruco_dictionary: str = "DICT_4X4_50"
    provenance: dict = field(default_factory=dict)

    # ── queries ──
    def ids(self) -> List[str]:
        return sorted(self.points, key=lambda k: (0, int(k), "") if k.isdigit() else (1, 0, k))

    def array(self, ids) -> np.ndarray:
        return np.array([self.points[i] for i in ids], dtype=float).reshape(-1, 3)

    def face_ids(self, face: str) -> List[str]:
        return [i for i in self.ids() if self.point_face[i] == face]

    def to_face(self, face: str, xyz) -> np.ndarray:
        """(N, 3) model-frame points -> (N, 2) face coordinates, metres."""
        f = self.faces[face]
        d = np.asarray(xyz, float).reshape(-1, 3) - f.origin
        return np.column_stack([d @ f.ex, d @ f.ey])

    def face_coords(self, ids) -> np.ndarray:
        ids = list(ids)
        out = np.zeros((len(ids), 2))
        for k, i in enumerate(ids):
            out[k] = self.to_face(self.point_face[i], self.points[i])[0]
        return out

    def marker_ids(self) -> List[int]:
        return [m.id for m in self.markers]

    def marker(self, marker_id: int) -> Marker:
        for m in self.markers:
            if m.id == int(marker_id):
                return m
        raise KeyError(marker_id)

    def nodes(self) -> List[dict]:
        """Points as EyeLab geometry nodes (``{"id","x","y","z"}``, metres)."""
        return [{"id": int(i), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for i, p in ((i, self.points[i]) for i in self.ids())]

    def aruco_dictionary_id(self) -> int:
        import cv2
        return int(getattr(cv2.aruco, self.aruco_dictionary))

    # ── EyeLab-side exports ──
    def marker_correspondences(self):
        from registration import MarkerCorrespondence
        return [MarkerCorrespondence(marker_id=m.id, unv_position=m.center, normal=m.normal,
                                     roll_deg=m.roll_deg, marker_size_mm=m.size_m * M_TO_MM,
                                     description=f"{self.name} face {m.face}")
                for m in self.markers]

    def export_marker_config(self, path) -> None:
        """EyeLab marker config (Markers tab / ``read_marker_config``)."""
        from registration import save_marker_config
        sizes = {round(m.size_m * M_TO_MM, 6) for m in self.markers}
        default = sizes.pop() if len(sizes) == 1 else None
        save_marker_config(str(path), self.marker_correspondences(), default_marker_size_mm=default)

    def export_unv(self, path) -> None:
        """UNV geometry (SI, metres) of the reference points, same node IDs."""
        from unv_writer import write_unv
        write_unv(path, self.nodes(), [list(e) for e in self.edges],
                  description=f"EyeLab validation target {self.name}")


# ── JSON I/O ──
def _v(x) -> List[float]:
    return [float(c) for c in np.asarray(x, float).ravel()]


def target_to_dict(t: Target) -> dict:
    return {
        "schema": SCHEMA, "name": t.name, "length_unit": "m",
        "uncertainty_mm": t.uncertainty_mm, "aruco_dictionary": t.aruco_dictionary,
        "provenance": t.provenance,
        "faces": {k: {"origin": _v(f.origin), "ex": _v(f.ex), "ey": _v(f.ey)}
                  for k, f in t.faces.items()},
        "markers": [{"id": m.id, "face": m.face, "center": _v(m.center), "normal": _v(m.normal),
                     "roll_deg": m.roll_deg, "size_m": m.size_m} for m in t.markers],
        "points": {i: {"xyz": _v(t.points[i]), "face": t.point_face[i]} for i in t.ids()},
        "edges": [[int(a), int(b)] for a, b in t.edges],
    }


def save_target(t: Target, path) -> None:
    Path(path).write_text(json.dumps(target_to_dict(t), indent=2), encoding="utf-8")


def load_target(path) -> Target:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {d.get('schema')!r}")
    if d.get("length_unit") != "m":
        raise ValueError(f"{path}: length_unit must be 'm'")
    faces = {k: Face(np.asarray(f["origin"], float), np.asarray(f["ex"], float),
                     np.asarray(f["ey"], float)) for k, f in d["faces"].items()}
    markers = [Marker(int(m["id"]), m["face"], np.asarray(m["center"], float),
                      np.asarray(m["normal"], float), float(m["roll_deg"]), float(m["size_m"]))
               for m in d["markers"]]
    pts = {str(k): np.asarray(v["xyz"], float) for k, v in d["points"].items()}
    pf = {str(k): v["face"] for k, v in d["points"].items()}
    return Target(d["name"], faces, markers, pts, pf, [tuple(e) for e in d.get("edges", [])],
                  float(d.get("uncertainty_mm", float("nan"))),
                  d.get("aruco_dictionary", "DICT_4X4_50"), d.get("provenance", {}))


def file_digest(path) -> str:
    """Short content hash: ties a session to the exact target / calibration file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


target_digest = file_digest


def roll_for_axes(normal, x_axis) -> float:
    """Roll (deg) that makes EyeLab's marker x-axis equal ``x_axis`` for ``normal``."""
    from registration import marker_axes_from_normal
    x0, y0, n = marker_axes_from_normal(normal, 0.0)
    x = np.asarray(x_axis, float)
    return float(np.degrees(np.arctan2(np.dot(np.cross(x0, x), n), np.dot(x0, x))))
