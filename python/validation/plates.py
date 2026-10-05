"""Plate layouts: the single definition of every printed validation target.

A layout lives in PAGE millimetres (A4 landscape, origin top-left, y down),
because that is what the printer reproduces. ``build_target`` maps it into the
model frame (metres) and applies the caliper measurements of the printed scale
bars and, for the folded plate, the measured fold angle. Nothing else in the
package hard-codes a physical position.

Model frames (right-handed, Z out of the printed surface when flat):

- ``A`` (planar): X page-right, Y page-up, origin at the page centre.
- ``L`` (folded): the page is folded along its horizontal centre line. The
  bottom half lies flat (face ``H``, normal +Z); the top half stands up
  (face ``V``, normal -Y at 90 deg). X runs along the fold, origin at its
  centre. It reproduces the two-face flange case with non-coplanar markers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .target import Face, Marker, Target, roll_for_axes

A4_LANDSCAPE_MM = (297.0, 210.0)
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])


@dataclass
class PageFace:
    key: str
    region: Tuple[float, float, float, float]  # x0, y0, x1, y1 page mm (render mask)
    anchor: Tuple[float, float]                # page mm that maps to the model origin
    folded: bool = False                       # True: rotates about X by the fold angle


@dataclass
class PageMarker:
    id: int
    face: str
    x: float
    y: float
    size_mm: float


@dataclass
class PageSaddle:
    id: int
    face: str
    x: float
    y: float


@dataclass
class ScaleBar:
    name: str       # "x" or "y": which page axis it measures
    x: float        # start, page mm
    y: float
    length_mm: float


@dataclass
class PageLayout:
    name: str
    title: str
    faces: List[PageFace]
    markers: List[PageMarker]
    saddles: List[PageSaddle]
    bars: List[ScaleBar]
    edges: List[Tuple[int, int]]
    square_mm: float = 15.0
    page_mm: Tuple[float, float] = A4_LANDSCAPE_MM
    fold_y: Optional[float] = None
    dictionary: str = "DICT_4X4_50"
    notes: List[str] = field(default_factory=list)
    args: Dict[str, float] = field(default_factory=dict)  # factory kwargs, to rebuild it

    def face(self, key: str) -> PageFace:
        return next(f for f in self.faces if f.key == key)


def _grid_edges(ids: List[List[int]]) -> List[Tuple[int, int]]:
    edges = []
    for r, row in enumerate(ids):
        for c, pid in enumerate(row):
            if c + 1 < len(row):
                edges.append((pid, row[c + 1]))
            if r + 1 < len(ids):
                edges.append((pid, ids[r + 1][c]))
    return edges


def plate_a(marker_size_mm: float = 20.0, pitch_mm: float = 40.0, nx: int = 5, ny: int = 4,
            marker_ids=(30, 31, 32)) -> PageLayout:
    """Planar plate: nx x ny saddle grid, three markers in an asymmetric L."""
    W, H = A4_LANDSCAPE_MM
    cx, cy = W / 2, H / 2
    xs = [cx + (i - (nx - 1) / 2) * pitch_mm for i in range(nx)]
    ys = [cy + (j - (ny - 1) / 2) * pitch_mm for j in range(ny)]
    saddles, grid, n = [], [], 1
    for y in ys:
        row = []
        for x in xs:
            saddles.append(PageSaddle(n, "F", x, y))
            row.append(n)
            n += 1
        grid.append(row)
    mx, my = 118.0, 70.0
    markers = [PageMarker(marker_ids[0], "F", cx - mx, cy - my, marker_size_mm),
               PageMarker(marker_ids[1], "F", cx + mx, cy - my, marker_size_mm),
               PageMarker(marker_ids[2], "F", cx - mx, cy + my, marker_size_mm)]
    bars = [ScaleBar("x", cx - 100.0, 196.0, 200.0), ScaleBar("y", 14.0, cy - 75.0, 150.0)]
    return PageLayout("A", "planar plate A", [PageFace("F", (0, 0, W, H), (cx, cy))],
                      markers, saddles, bars, _grid_edges(grid),
                      notes=["Glue flat on a rigid board (glass, aluminium, 5 mm+ foam board)."],
                      args={"marker_size_mm": marker_size_mm, "pitch_mm": pitch_mm, "nx": nx, "ny": ny,
                            "marker_ids": list(marker_ids)})


def plate_l(marker_size_mm: float = 20.0, pitch_mm: float = 40.0, nx: int = 5,
            marker_ids=(40, 41, 42, 43)) -> PageLayout:
    """Folded two-face plate: 5x2 saddles and two markers on each face."""
    W, H = A4_LANDSCAPE_MM
    cx, fy = W / 2, H / 2
    xs = [cx + (i - (nx - 1) / 2) * pitch_mm for i in range(nx)]
    saddles, grid_h, grid_v, n = [], [], [], 1
    for d in (28.0, 64.0):                 # face H: distance from fold
        row = []
        for x in xs:
            saddles.append(PageSaddle(n, "H", x, fy + d))
            row.append(n)
            n += 1
        grid_h.append(row)
    for d in (28.0, 64.0):                 # face V: height above fold
        row = []
        for x in xs:
            saddles.append(PageSaddle(n, "V", x, fy - d))
            row.append(n)
            n += 1
        grid_v.append(row)
    mx, md = 118.0, 46.0
    markers = [PageMarker(marker_ids[0], "H", cx - mx, fy + md, marker_size_mm),
               PageMarker(marker_ids[1], "H", cx + mx, fy + md, marker_size_mm),
               PageMarker(marker_ids[2], "V", cx - mx, fy - md, marker_size_mm),
               PageMarker(marker_ids[3], "V", cx + mx, fy - md, marker_size_mm)]
    bars = [ScaleBar("x", cx - 100.0, 197.0, 200.0), ScaleBar("y", 287.0, fy + 10.0, 75.0)]
    edges = _grid_edges(grid_h) + _grid_edges(grid_v) + [
        (grid_h[0][c], grid_v[0][c]) for c in range(nx)]
    faces = [PageFace("H", (0, fy, W, H), (cx, fy)),
             PageFace("V", (0, 0, W, fy), (cx, fy), folded=True)]
    return PageLayout("L", "folded plate L", faces, markers, saddles, bars, edges, fold_y=fy,
                      notes=["Valley fold along the edge ticks: printed side on the inside of the L.",
                             "Bond each half to a rigid 90 deg bracket; fold line on the bracket edge."],
                      args={"marker_size_mm": marker_size_mm, "pitch_mm": pitch_mm, "nx": nx,
                            "marker_ids": list(marker_ids)})


LAYOUTS = {"A": plate_a, "L": plate_l}


def layout_from_provenance(prov: dict) -> PageLayout:
    """Rebuild the page layout a target was generated from."""
    return LAYOUTS[prov["layout"]](**prov.get("layout_args", {}))


def _face_frame(pf: PageFace, fold_interior_deg: float) -> Face:
    if not pf.folded:
        return Face(np.zeros(3), X_AXIS.copy(), Y_AXIS.copy())
    phi = np.radians(180.0 - fold_interior_deg)   # rotation of the upper half about X
    return Face(np.zeros(3), X_AXIS.copy(), np.array([0.0, np.cos(phi), np.sin(phi)]))


def build_target(layout: PageLayout, scale_x: float = 1.0, scale_y: float = 1.0,
                 fold_interior_deg: float = 90.0, uncertainty_mm: float = float("nan"),
                 measured: bool = False) -> Target:
    """Map a page layout to a model-frame target (metres).

    ``scale_x``/``scale_y`` are measured/nominal ratios of the printed scale
    bars; they rescale page positions about each face anchor, which is what a
    printer's axis-dependent scale error does. Marker edge length uses their
    mean (anisotropy of ~0.1 % changes a 20 mm edge by 0.02 mm).
    """
    faces = {pf.key: _face_frame(pf, fold_interior_deg) for pf in layout.faces}

    def to_model(face_key: str, px: float, py: float) -> np.ndarray:
        pf, f = layout.face(face_key), faces[face_key]
        ax, ay = pf.anchor
        u = scale_x * (px - ax) / 1000.0
        v = scale_y * (ay - py) / 1000.0
        return f.origin + u * f.ex + v * f.ey

    points = {str(s.id): to_model(s.face, s.x, s.y) for s in layout.saddles}
    point_face = {str(s.id): s.face for s in layout.saddles}
    markers = []
    for m in layout.markers:
        f = faces[m.face]
        markers.append(Marker(m.id, m.face, to_model(m.face, m.x, m.y), f.normal,
                              roll_for_axes(f.normal, f.ex),
                              0.5 * (scale_x + scale_y) * m.size_mm / 1000.0))
    prov = {"layout": layout.name, "layout_args": dict(layout.args), "measured": bool(measured),
            "scale_x": scale_x,
            "scale_y": scale_y, "square_mm": layout.square_mm,
            "bars_nominal_mm": {b.name: b.length_mm for b in layout.bars}}
    if layout.fold_y is not None:
        prov["fold_interior_deg"] = fold_interior_deg
    return Target(f"plate_{layout.name}", faces, markers, points, point_face,
                  list(layout.edges), uncertainty_mm, layout.dictionary, prov)
