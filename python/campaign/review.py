"""Manual review of the hits on one image (T7.0 §14.5): what the person decided, kept apart from the automatic result.

The automatic analysis can take a fleck of dirt for a dot or miss a faint one. A review records, per target:

- ``confirmed``: the automatic dot is right;
- ``pinned``: the dot is where the person put it (a detected candidate, or a click refined to the dot's centre);
- ``absent``: there is no dot for this target (an automatic "dot" there was dirt);
- no entry: not reviewed, the automatic result stands.

Reviews live next to the results, ``results/<stem>.review.json``, keyed to the source file's version. Every automatic
analysis of that version applies the review (``store.analyse`` calls :func:`apply`), so re-analysing never loses it;
a new version of the file (a rescan) does not inherit it and says so. Nothing here depends on tkinter.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCHEMA = "eyelab.campaign.review/1"
STATUSES = ("confirmed", "pinned", "absent")
SAME_DOT_MM = 0.3      # an automatic extra dot this close to a final hit is that hit
REVIEWABLE = ("scan", "photo")


def review_path(result_json: Path) -> Path:
    p = Path(result_json)
    return p.with_name(p.name[:-len(".json")] + ".review.json" if p.name.endswith(".json") else p.name + ".review.json")


def load(result_json: Path) -> Optional[dict]:
    p = review_path(result_json)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return d if d.get("schema") == SCHEMA else None


def save(result_json: Path, key: str, version: str, sheet: str, decisions: Dict[int, dict], note: str = "") -> Path:
    """``decisions``: target -> {"status": confirmed|pinned|absent, "hit": [x_mm, y_mm] or None}. Page mm."""
    clean = {}
    for n, d in decisions.items():
        st = d.get("status")
        if st not in STATUSES:
            continue
        hit = d.get("hit")
        if st == "absent":
            hit = None
        elif hit is None:
            raise ValueError(f"target {n}: {st} needs a position")
        clean[str(int(n))] = {"status": st, "hit": None if hit is None else [round(float(hit[0]), 4),
                                                                              round(float(hit[1]), 4)]}
    p = review_path(result_json)
    p.write_text(json.dumps({"schema": SCHEMA, "key": key, "source_version": version, "sheet": sheet,
                             "reviewed": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note, "targets": clean},
                            indent=1), encoding="utf-8")
    return p


def apply(result_json: Path, version: str, res) -> Optional[dict]:
    """Overrides ``res`` (a ``hits.Result``) in place with the review of this file version; returns what it did.

    None when there is no review. ``{"stale": True}`` when the review is for another version of the file (not applied).
    """
    rv = load(result_json)
    if rv is None:
        return None
    if rv.get("source_version") != version:
        res.notes.insert(0, f"manual review of {rv.get('reviewed')} is for an older version of this file: not applied")
        return {"stale": True, "reviewed": rv.get("reviewed")}
    if rv.get("sheet") and rv["sheet"] != res.sheet:
        res.notes.insert(0, f"manual review is for sheet {rv['sheet']}, the image is {res.sheet}: not applied")
        return {"stale": True, "reviewed": rv.get("reviewed")}
    counts = {s: 0 for s in STATUSES}
    removed: List[Tuple[float, float]] = []
    for h in res.hits:
        d = rv["targets"].get(str(h.n))
        if not d:
            continue
        counts[d["status"]] += 1
        if d["status"] == "absent":
            if h.hit is not None:
                removed.append(h.hit)
            h.hit = None
            h.area_mm2 = float("nan")
        elif d["status"] == "pinned":
            if h.hit is not None and math.dist(h.hit, d["hit"]) > SAME_DOT_MM:
                removed.append(h.hit)   # the automatic dot was wrong; it is not a hit of this target
            h.hit = (float(d["hit"][0]), float(d["hit"][1]))
        # confirmed: the automatic dot stands; if it moved since (new rule), keep the confirmed position
        elif h.hit is None or math.dist(h.hit, d["hit"]) > SAME_DOT_MM:
            h.hit = (float(d["hit"][0]), float(d["hit"][1]))
    finals = [h.hit for h in res.hits if h.hit is not None]
    res.extra_dots = [e for e in res.extra_dots if all(math.dist(e, f) > SAME_DOT_MM for f in finals)]
    res.notes.insert(0, "manual review " + rv.get("reviewed", "") + ": " +
                     ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    return {"reviewed": rv.get("reviewed"), **counts, "rejected_dots": len(removed), "note": rv.get("note", "")}


def refine_click(rect: np.ndarray, x_mm: float, y_mm: float, ppmm: float, r_mm: float = 1.2) -> Tuple[float, float]:
    """The centre of the dark spot nearest a click (darkness-weighted centroid within ``r_mm``); the click itself when
    nothing there is darker than the paper."""
    h, w = rect.shape
    cx, cy = x_mm * ppmm, y_mm * ppmm
    r = max(2, int(math.ceil(r_mm * ppmm)))
    x0, x1 = max(0, int(cx) - 3 * r), min(w, int(cx) + 3 * r + 1)
    y0, y1 = max(0, int(cy) - 3 * r), min(h, int(cy) + 3 * r + 1)
    if x1 <= x0 or y1 <= y0:
        return x_mm, y_mm
    patch = rect[y0:y1, x0:x1].astype(np.float32)
    paper = float(np.percentile(patch, 75))
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    dark = np.clip(paper - patch, 0, None) * inside
    noise = float(np.std(patch[patch >= np.percentile(patch, 50)])) if patch.size > 4 else 0.0
    dark[dark < 3 * noise + 8] = 0
    if dark.sum() <= 0:
        return x_mm, y_mm
    return float((xs * dark).sum() / dark.sum()) / ppmm, float((ys * dark).sum() / dark.sum()) / ppmm
