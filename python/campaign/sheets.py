"""Campaign sheets (T7.0 §5): one marker frame F40 on every page, layouts A40, M, P, T.

Page millimetres, A4 landscape, origin top-left, y down (what the printer reproduces). The model frame is
the validation plates' one: origin at the page centre, X right, Y up, Z out of the paper, metres.

- F40: markers 30 (top left), 31 (top right), 32 (bottom left), 33 (bottom right), 40 mm, DICT_4X4_50,
  centres 28 mm from the page edges; X scale bar 150 mm at the bottom, Y scale bar 90 mm on the left.
- A40: F40 + plate A's 5x4 saddle grid (40 mm pitch, 12 mm squares). Validation layout "A40".
- M (AR), P (printed control), T (practice): F40 + 8 border saddles (8 mm squares, scan/photo reference)
  + a sheet-ID marker (DICT_6X6_250, 20 mm, id = sheet number). Targets: 12 (T: 6), seeded, at least
  30 mm apart and 15 mm from every printed feature. P prints them as numbered crosshairs; M and T do not.

Every feature has a keep-out rectangle; ``check`` refuses a page whose rectangles overlap or leave the
printable area. Text extents come from cv2.getTextSize, so the check sees what is drawn.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PAGE = (297.0, 210.0)
CENTRE = (PAGE[0] / 2, PAGE[1] / 2)
PRINT_MARGIN = 5.0          # mm, no feature (keep-out included) closer to the paper edge
FRAME_DICT = "DICT_4X4_50"
ID_DICT = "DICT_6X6_250"
FRAME_IDS = (30, 31, 32, 33)
FRAME_MARKER_MM = 40.0
FRAME_INSET = 28.0          # marker centre from the page edges
ID_MARKER_MM = 20.0
BORDER_SQUARE = 8.0
GRID_SQUARE = 12.0
TARGET_CLEAR = 15.0         # mm from any printed feature
TARGET_SPACING = 30.0       # mm between targets
WORK = (60.0, 40.0, 237.0, 170.0)   # x0, y0, x1, y1: where targets may lie
LINE = 0.35
SEED_BASE = 20261125


@dataclass
class Marker:
    id: int
    x: float
    y: float
    size: float
    dictionary: str = FRAME_DICT


@dataclass
class Saddle:
    id: int
    x: float
    y: float
    square: float


@dataclass
class Bar:
    name: str      # "x" or "y"
    x: float
    y: float
    length: float


@dataclass
class Text:
    s: str
    x: float       # baseline left, mm
    y: float
    h: float


@dataclass
class Target:
    n: int         # presentation order, 1-based (printed number on P sheets)
    x: float
    y: float


@dataclass
class Sheet:
    kind: str                  # A40 | M | P | T
    number: int                # sheet number (= ID marker id); 0 for A40
    markers: List[Marker]
    saddles: List[Saddle]
    bars: List[Bar]
    texts: List[Text] = field(default_factory=list)
    targets: List[Target] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def name(self) -> str:
        return "A40" if self.kind == "A40" else f"{self.kind}{self.number:03d}"


# ------------------------------------------------------------------ text metrics
def _text_scale(h_mm: float, ppm: float) -> Tuple[float, int]:
    return h_mm * ppm / 22.0, max(1, int(round(0.12 * h_mm * ppm)))


def text_box(t: Text, ppm: float = 600 / 25.4) -> Tuple[float, float, float, float]:
    scale, th = _text_scale(t.h, ppm)
    (w, h), base = cv2.getTextSize(t.s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    return (t.x, t.y - h / ppm, t.x + w / ppm, t.y + base / ppm)


# ------------------------------------------------------------------ the frame
def frame() -> Tuple[List[Marker], List[Bar], List[Text]]:
    W, H = PAGE
    a = FRAME_INSET
    markers = [Marker(FRAME_IDS[0], a, a, FRAME_MARKER_MM), Marker(FRAME_IDS[1], W - a, a, FRAME_MARKER_MM),
               Marker(FRAME_IDS[2], a, H - a, FRAME_MARKER_MM), Marker(FRAME_IDS[3], W - a, H - a, FRAME_MARKER_MM)]
    bars = [Bar("x", CENTRE[0] - 75.0, 196.0, 150.0), Bar("y", 50.0, 60.0, 90.0)]
    texts = [Text("Bars X 150 mm (bottom), Y 90 mm (left): caliper from one edge of the first tick to the SAME edge"
                  " of the last", 73.5, 202.5, 2.2)]
    return markers, bars, texts


def _header(kind: str, number: int, extra: str) -> List[Text]:
    label = "A40" if kind == "A40" else f"{kind}{number:03d}"
    return [Text(f"EyeLab campaign 0.5  {label}  {extra}", 56.0, 9.0, 3.0),
            Text("Print A4 landscape, 100 % actual size, no fit. Ids 30-33 4x4_50 40 mm.", 56.0, 14.0, 2.4)]


def border_saddles() -> List[Saddle]:
    pts = [(100.0, 28.0), (148.5, 28.0), (197.0, 28.0), (100.0, 178.0), (197.0, 178.0),
           (28.0, 80.0), (28.0, 130.0), (269.0, 80.0)]
    return [Saddle(i + 1, x, y, BORDER_SQUARE) for i, (x, y) in enumerate(pts)]


ID_MARKER_POS = (269.0, 130.0)


def grid_saddles() -> List[Saddle]:
    xs = [CENTRE[0] + (i - 2) * 40.0 for i in range(5)]
    ys = [CENTRE[1] + (j - 1.5) * 40.0 for j in range(4)]
    return [Saddle(1 + j * 5 + i, x, y, GRID_SQUARE) for j, y in enumerate(ys) for i, x in enumerate(xs)]


# ------------------------------------------------------------------ keep-out boxes
def boxes(sh: Sheet, with_targets: bool = True) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    out = []
    for m in sh.markers:
        cells = 6 if m.dictionary == FRAME_DICT else 8
        h = m.size / 2 + m.size / cells
        out.append((f"marker {m.id}", (m.x - h, m.y - h, m.x + h, m.y + h)))
    for s in sh.saddles:
        h = s.square + 0.5
        out.append((f"saddle {s.id}", (s.x - h, s.y - h, s.x + h, s.y + h)))
    for b in sh.bars:
        if b.name == "x":
            out.append((f"bar x", (b.x - 0.5, b.y - 4.5, b.x + b.length + 0.5, b.y + LINE + 0.5)))
        else:
            out.append((f"bar y", (b.x - 0.5, b.y - 0.5, b.x + 4.5, b.y + b.length + 0.5)))
    for t in sh.texts:
        x0, y0, x1, y1 = text_box(t)
        out.append((f"text '{t.s[:16]}'", (x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3)))
    if with_targets and sh.kind == "P":
        for t in sh.targets:
            out.append((f"target {t.n}", (t.x - 8.5, t.y - 8.5, t.x + 8.5, t.y + 8.5)))
            out.append((f"target {t.n} label", text_box(_target_label(t))))
    return out


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _rect_dist(p, r) -> float:
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def check(sh: Sheet) -> List[str]:
    """Problems on the page: overlapping keep-outs, features off the printable area, bad targets."""
    errs = []
    bx = boxes(sh)
    W, H = PAGE
    for name, r in bx:
        if r[0] < PRINT_MARGIN - 4.0 or r[1] < PRINT_MARGIN - 4.0 or r[2] > W - PRINT_MARGIN + 4.0 \
                or r[3] > H - PRINT_MARGIN + 4.0:
            errs.append(f"{sh.name}: {name} outside the page {tuple(round(v, 1) for v in r)}")
    for i, (na, a) in enumerate(bx):
        for nb, b in bx[i + 1:]:
            same = na.split(" ")[:2] == nb.split(" ")[:2] and na.startswith("target")
            if not same and _overlap(a, b):
                errs.append(f"{sh.name}: {na} overlaps {nb}")
    # printed ink (not keep-out) must stay printable
    for m in sh.markers:
        if min(m.x - m.size / 2, m.y - m.size / 2, W - m.x - m.size / 2, H - m.y - m.size / 2) < PRINT_MARGIN:
            errs.append(f"{sh.name}: marker {m.id} ink within {PRINT_MARGIN} mm of the edge")
    feats = [r for _, r in boxes(sh, with_targets=False)]
    for t in sh.targets:
        d = min(_rect_dist((t.x, t.y), r) for r in feats)
        if d < TARGET_CLEAR - 1e-9:
            errs.append(f"{sh.name}: target {t.n} {d:.1f} mm from a printed feature")
    for i, a in enumerate(sh.targets):
        for b in sh.targets[i + 1:]:
            if math.hypot(a.x - b.x, a.y - b.y) < TARGET_SPACING - 1e-9:
                errs.append(f"{sh.name}: targets {a.n} and {b.n} closer than {TARGET_SPACING} mm")
    return errs


# ------------------------------------------------------------------ targets
def make_targets(base: Sheet, n: int, seed: int) -> List[Target]:
    rng = np.random.default_rng(seed)
    feats = [r for _, r in boxes(base, with_targets=False)]
    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError(f"cannot place {n} targets (seed {seed})")
        p = (float(rng.uniform(WORK[0], WORK[2])), float(rng.uniform(WORK[1], WORK[3])))
        p = (round(p[0], 1), round(p[1], 1))
        if min(_rect_dist(p, r) for r in feats) < TARGET_CLEAR:
            continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < TARGET_SPACING for q in pts):
            continue
        pts.append(p)
    return [Target(i + 1, x, y) for i, (x, y) in enumerate(pts)]   # generation order is random: it is the order


def _target_label(t: Target) -> Text:
    return Text(str(t.n), t.x + 3.0, t.y - 3.0, 3.0)


# ------------------------------------------------------------------ sheets
KIND_TEXT = {"M": "AR - targets hidden", "P": "PRINTED - hit the numbered crosses in order",
             "T": "PRACTICE - targets hidden"}
N_TARGETS = {"M": 12, "P": 12, "T": 6}


def sheet_a40() -> Sheet:
    markers, bars, texts = frame()
    sh = Sheet("A40", 0, markers, grid_saddles(), bars, texts + _header("A40", 0, "plate A40: 5x4 saddles, 40 mm pitch"))
    return sh


def sheet(kind: str, number: int) -> Sheet:
    markers, bars, texts = frame()
    markers = markers + [Marker(number, ID_MARKER_POS[0], ID_MARKER_POS[1], ID_MARKER_MM, ID_DICT)]
    sh = Sheet(kind, number, markers, border_saddles(), bars, texts + _header(kind, number, KIND_TEXT[kind]))
    sh.seed = SEED_BASE * 1000 + number
    sh.targets = make_targets(sh, N_TARGETS[kind], sh.seed)
    return sh


def digest(sh: Sheet) -> str:
    return hashlib.sha256(json.dumps(asdict(sh), sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ rendering
def _px(v: float, ppm: float) -> int:
    return int(round(v * ppm))


def _fill(img, x0, y0, x1, y1, ppm):
    img[_px(y0, ppm):_px(y1, ppm), _px(x0, ppm):_px(x1, ppm)] = 0


def _marker(img, m: Marker, ppm):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, m.dictionary))
    n = d.markerSize + 2
    bits = cv2.aruco.generateImageMarker(d, int(m.id), n, borderBits=1)
    cell = m.size / n
    x0, y0 = m.x - m.size / 2, m.y - m.size / 2
    for r in range(n):
        for c in range(n):
            if bits[r, c] == 0:
                _fill(img, x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell, ppm)


def _text(img, t: Text, ppm):
    scale, th = _text_scale(t.h, ppm)
    cv2.putText(img, t.s, (_px(t.x, ppm), _px(t.y, ppm)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)


def render(sh: Sheet, ppm: float = 600 / 25.4, targets: Optional[bool] = None) -> np.ndarray:
    """uint8 page (255 white). ``targets``: draw the targets (default: only on P sheets)."""
    W, H = PAGE
    img = np.full((_px(H, ppm), _px(W, ppm)), 255, np.uint8)
    for s in sh.saddles:
        q = s.square
        _fill(img, s.x - q, s.y - q, s.x, s.y, ppm)
        _fill(img, s.x, s.y, s.x + q, s.y + q, ppm)
    for m in sh.markers:
        _marker(img, m, ppm)
    for b in sh.bars:
        n = int(b.length // 10)
        if b.name == "x":
            _fill(img, b.x, b.y, b.x + b.length, b.y + LINE, ppm)
            for k in range(n + 1):
                t = 4.0 if k in (0, n) else 2.0
                _fill(img, b.x + 10 * k, b.y - t, b.x + 10 * k + LINE, b.y, ppm)
        else:
            _fill(img, b.x, b.y, b.x + LINE, b.y + b.length, ppm)
            for k in range(n + 1):
                t = 4.0 if k in (0, n) else 2.0
                _fill(img, b.x + LINE, b.y + 10 * k, b.x + LINE + t, b.y + 10 * k + LINE, ppm)
    for t in sh.texts:
        _text(img, t, ppm)
    if targets if targets is not None else sh.kind == "P":
        for t in sh.targets:
            g, a = 1.5, 8.0   # open centre, arm end (mm)
            for s in (-1, 1):
                x0, x1 = sorted((t.x + s * g, t.x + s * a))
                _fill(img, x0, t.y - LINE / 2, x1, t.y + LINE / 2, ppm)
                y0, y1 = sorted((t.y + s * g, t.y + s * a))
                _fill(img, t.x - LINE / 2, y0, t.x + LINE / 2, y1, ppm)
            _text(img, _target_label(t), ppm)
    return img


# ------------------------------------------------------------------ export
def to_model(x: float, y: float) -> List[float]:
    """Page mm -> model frame metres (origin page centre, X right, Y up, Z out of the paper)."""
    return [round((x - CENTRE[0]) / 1000.0, 6), round((CENTRE[1] - y) / 1000.0, 6), 0.0]


def sheet_json(sh: Sheet) -> dict:
    return {
        "schema": "eyelab.campaign.sheet/1",
        "name": sh.name, "kind": sh.kind, "number": sh.number, "seed": sh.seed, "digest": digest(sh),
        "page_mm": list(PAGE), "model_origin_page_mm": list(CENTRE),
        "frame": {"dictionary": FRAME_DICT, "ids": list(FRAME_IDS), "size_mm": FRAME_MARKER_MM,
                  "target_file": "plate_A40.target.json"},
        "id_marker": next(({"dictionary": m.dictionary, "id": m.id, "size_mm": m.size, "page_mm": [m.x, m.y]}
                           for m in sh.markers if m.dictionary == ID_DICT), None),
        "markers": [{"id": m.id, "dictionary": m.dictionary, "size_mm": m.size, "page_mm": [m.x, m.y],
                     "model_m": to_model(m.x, m.y)} for m in sh.markers],
        "saddles": [{"id": s.id, "square_mm": s.square, "page_mm": [s.x, s.y], "model_m": to_model(s.x, s.y)}
                    for s in sh.saddles],
        "bars_mm": {b.name: b.length for b in sh.bars},
        "targets": [{"n": t.n, "page_mm": [t.x, t.y], "model_m": to_model(t.x, t.y)} for t in sh.targets],
    }


def write_pdf(pages: List[np.ndarray], path: Path, dpi: int = 600) -> None:
    from PIL import Image
    ims = [Image.fromarray(np.where(p >= 128, 255, 0).astype(np.uint8)).convert("1") for p in pages]
    ims[0].save(str(path), "PDF", resolution=float(dpi), save_all=True, append_images=ims[1:])


# Numbering: T 1-10, P 11-40, M 41-100 (ID marker id = number, DICT_6X6_250).
PLAN = {"T": range(1, 11), "P": range(11, 41), "M": range(41, 101)}


def build_all(plan: Dict[str, range] = PLAN) -> Tuple[Sheet, List[Sheet]]:
    sheets = [sheet(k, n) for k, ns in plan.items() for n in ns]
    return sheet_a40(), sheets
