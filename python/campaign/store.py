"""The campaign data folder (T7.0 §4.2): one place for scans, photos and Beam Pro runs, analysed as they arrive.

Layout under the root (default ``python/campaign_runs/``, gitignored)::

    scans/            flatbed scans of used sheets: PNG or TIFF, 600 dpi, greyscale (final numbers)
    photos/           phone or Beam Pro camera photos of used sheets (live estimate); optional photos/camera.yaml
                      (camera_matrix, dist_coeffs) undistorts them first
    runs/             Beam Pro run folders as pulled (…/BeamProRuns/<run>/); a campaign trial's review-look Eye frames
                      give the Eye estimate of its sheet
    sheets_used.csv   pulled from files/campaign/ on the Beam Pro (optional: run.json carries the same row)
    results/          written here: per source a .json summary, a .csv of hits and a _check.jpg

The sheet of an image is read from its sheet-ID marker; a sheet name in the file or folder name (``M041``) wins.
Nothing here depends on tkinter (``campaign.gui`` is a thin view on it).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "campaign_runs"
SUBDIRS = ("scans", "photos", "runs", "results")
IMAGE_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".pgm"}
REFUSED_EXT = {".pdf": "PDF: save the scan as PNG or TIFF (lossless) instead",
               ".heic": "HEIC: set the phone to save JPEG, or convert to PNG"}
SHEET_RE = re.compile(r"(?<![A-Za-z0-9])([MPT]\d{3})(?![0-9])")
EYE_FRAMES_MAX = 10   # review-look frames analysed per trial (evenly spread, most markers first)
EYE_CAMERA = Path(__file__).resolve().parent.parent / "config" / "cameras" / "beampro_eye.json"
# Above this the Eye cannot see a 1 mm pen dot: the first real review looks (2026-11-26, sheet at about 48 cm) were at
# 0.67-0.77 mm/px and found 0-1 of 12 dots. The synthetic Eye test holds to 0.6 mm at 0.45 mm/px; a review look at
# about 20 cm (sheet filling the frame) gives about 0.3 mm/px.
EYE_MAX_MM_PER_PX = 0.45
SCHEMA = "eyelab.campaign.result/1"
FILE_SETTLE_S = 2.0    # a file that arrived more recently may still be being written
RUN_SETTLE_S = 10.0    # a run folder is pulled file by file (live.csv can arrive before its frames)
RETRY_S = 300.0        # a failed analysis is tried again after this (the source may have been incomplete)


def _stamp(p: Path) -> float:
    """When the file or folder arrived here: adb pull and copy keep the device's mtime, ctime is local."""
    s = p.stat()
    return max(s.st_mtime, s.st_ctime)


@dataclass
class Source:
    key: str            # path relative to the root, forward slashes (unique id)
    kind: str           # scan | photo | eye
    path: Path          # image, or the run folder for kind eye
    mtime: float        # local arrival: newest of mtime and ctime (adb pull and copies keep the old mtime)
    sheet_hint: Optional[str] = None
    refused: Optional[str] = None
    version: str = ""   # changes whenever the source does (for a run: arrival and the number of saved frames)

    @property
    def settle_s(self) -> float:
        return RUN_SETTLE_S if self.kind == "eye" else FILE_SETTLE_S


def ensure_layout(root: Path) -> Path:
    root = Path(root)
    for d in SUBDIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text("EyeLab campaign 0.5 data folder, watched by campaign.bat. Layout under the root "
                          + __doc__.split("Layout under the root", 1)[1].split("Nothing here", 1)[0].strip() + "\n",
                          encoding="utf-8")
    return root


def _hint(path: Path, root: Path) -> Optional[str]:
    rel = path.relative_to(root).as_posix()
    found = SHEET_RE.findall(rel)
    return found[-1] if found else None


def _result_stem(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)


def read_run_meta(run_dir: Path) -> Dict[str, str]:
    p = run_dir / "run.json"
    if not p.exists():
        return {}
    try:
        return {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in
                json.loads(p.read_text(encoding="utf-8")).items()}
    except (ValueError, OSError):
        return {}


def _run_sheet(meta: Dict[str, str]) -> Optional[str]:
    m = SHEET_RE.search(meta.get("campaign_sheet", ""))
    return m.group(1) if m else None


def _find(base: Path, name: str) -> List[Path]:
    """Files called ``name`` under ``base``, not descending into frames/ (thousands of Eye frames per run)."""
    import os
    out = []
    for d, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x != "frames"]
        if name in files:
            out.append(Path(d) / name)
    return out


def discover(root: Path) -> List[Source]:
    """Every source under scans/, photos/ and runs/ (campaign trials only)."""
    root = Path(root)
    out: List[Source] = []
    for sub, kind in (("scans", "scan"), ("photos", "photo")):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in IMAGE_EXT or ext in REFUSED_EXT:
                st = _stamp(p)
                out.append(Source(p.relative_to(root).as_posix(), kind, p, st, _hint(p, root),
                                  REFUSED_EXT.get(ext), f"{st:.3f}/{p.stat().st_size}"))
    runs = root / "runs"
    if runs.exists():
        for live in sorted(_find(runs, "live.csv")):
            d = live.parent
            meta = read_run_meta(d)
            if meta.get("variant") != "mark" and "campaign_sheet" not in meta:
                continue   # not a campaign trial, or not finished (run.json is written at the end)
            fr = d / "frames"
            parts = [live, d / "run.json", d] + ([fr] if fr.exists() else [])
            st = max(_stamp(x) for x in parts)
            nfr = sum(1 for _ in os.scandir(fr)) if fr.exists() else 0
            out.append(Source(d.relative_to(root).as_posix(), "eye", d, st, _run_sheet(meta),
                              version=f"{st:.3f}/{nfr}"))
    return out


def result_path(root: Path, src: Source) -> Path:
    return Path(root) / "results" / (_result_stem(src.key) + ".json")


def is_current(root: Path, src: Source) -> bool:
    p = result_path(root, src)
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return False
    if r.get("schema") != SCHEMA or r.get("source_version") != src.version:
        return False
    if "error" in r and not src.refused and time.time() - r.get("analysed_epoch", 0) > RETRY_S:
        return False   # failed: try again (an adb pull may have been incomplete)
    return True


def pending(root: Path, settle_s: Optional[float] = None) -> List[Source]:
    """Sources without a current result, once they have stopped arriving (``settle_s``, default per kind)."""
    now = time.time()
    return [s for s in discover(root) if not is_current(root, s)
            and now - s.mtime >= (s.settle_s if settle_s is None else settle_s)]


# ------------------------------------------------------------------ analysis
def _review_frames(run_dir: Path, limit: int = EYE_FRAMES_MAX) -> List[Path]:
    rows = []
    with open(run_dir / "live.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("step") or "").startswith("review") and r.get("saved"):
                p = run_dir / r["saved"]
                if p.exists():
                    try:
                        rows.append((int(r.get("markers") or 0), int(r["seq"]), p))
                    except ValueError:
                        continue
    if not rows:
        return []
    best = max(r[0] for r in rows)
    good = sorted((r for r in rows if r[0] >= best - 1), key=lambda r: r[1])
    if len(good) <= limit:
        return [r[2] for r in good]
    idx = np.linspace(0, len(good) - 1, limit).round().astype(int)
    return [good[i][2] for i in idx]


def _summary(hits, extra, fit_points, fit_rms, mm_per_px, notes, frames=None) -> dict:
    from . import hits as Hm
    errs = [h.err for h in hits if h.err is not None]
    return {"targets": len(hits), "found": len(errs), "missing": [h.n for h in hits if h.hit is None],
            "extra_dots": len(extra), "fit_points": fit_points, "fit_rms_mm": fit_rms, "mm_per_px": mm_per_px,
            "stats": Hm.stats(errs), "notes": notes, **({"frames": frames} if frames is not None else {})}


def analyse(root: Path, src: Source) -> dict:
    """Analyse one source, write its results (json, csv, check image) and return the json."""
    import cv2
    from . import hits as Hm
    root = Path(root)
    out = result_path(root, src)
    base = {"schema": SCHEMA, "key": src.key, "kind": src.kind, "source_mtime": src.mtime,
            "source_version": src.version, "analysed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysed_epoch": time.time()}
    try:
        if src.refused:
            raise RuntimeError(src.refused)
        if src.kind == "eye":
            r = _analyse_eye(root, src, out)
        else:
            calib = root / "photos" / "camera.yaml"
            res, rect = Hm.analyse(src.path, src.sheet_hint, calib=str(calib) if src.kind == "photo" and
                                   calib.exists() else None)
            from . import review as Rv
            manual = Rv.apply(out, src.version, res)   # the person's review of this file version, if any
            Hm.write_csv(res, out.with_suffix(".csv"))
            cv2.imwrite(str(out.with_name(out.stem + "_check.jpg")), Hm.annotate(rect, res, 10.0),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            r = {"sheet": res.sheet, **_summary(res.hits, res.extra_dots, res.fit_points, res.fit_rms_mm,
                                                res.mm_per_px, res.notes)}
            if manual:
                r["manual"] = manual
        base.update(r)
    except Exception as e:   # recorded, not retried until the file changes
        base.update({"sheet": src.sheet_hint, "error": f"{type(e).__name__}: {e}"})
    out.write_text(json.dumps(base, indent=1), encoding="utf-8")
    return base


def _analyse_eye(root: Path, src: Source, out: Path) -> dict:
    """Eye estimate: each review-look frame analysed, each target's dot the median over the frames that found it."""
    import cv2
    from . import hits as Hm
    if not src.sheet_hint:
        raise RuntimeError("run.json has no campaign_sheet")
    frames = _review_frames(src.path)
    if not frames:
        row = read_run_meta(src.path).get("campaign_result", "").split(",")
        if len(row) > 10 and row[10] == "1":
            raise RuntimeError(f"trial aborted after {row[7]} targets: no review look")
        raise RuntimeError("no review-look frames saved in this run")
    per: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    fits, last, last_rect, notes = [], None, None, []
    calib = str(EYE_CAMERA) if EYE_CAMERA.exists() else None
    for fp in frames:
        try:
            res, rect = Hm.analyse(fp, src.sheet_hint, calib=calib)
        except Exception as e:
            notes.append(f"{fp.name}: {e}")
            continue
        fits.append((res.fit_points, res.fit_rms_mm, res.mm_per_px))
        for h in res.hits:
            if h.hit is not None:
                per[h.n].append(h.hit)
        last, last_rect = res, rect
    if last is None:
        raise RuntimeError("no review frame could be fitted: " + "; ".join(notes[:3]))
    used = len(fits)
    for h in last.hits:
        pts = per.get(h.n, [])
        h.hit = tuple(np.median(np.array(pts), axis=0)) if len(pts) * 2 >= used else None
    last.source = f"{src.key} ({used} review frames)"
    Hm.write_csv(last, out.with_suffix(".csv"))
    cv2.imwrite(str(out.with_name(out.stem + "_check.jpg")), Hm.annotate(last_rect, last, 10.0),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    F = np.array(fits)
    mmpx = float(np.median(F[:, 2]))
    if calib is None:
        notes.insert(0, f"not undistorted: {EYE_CAMERA.name} missing")
    out = {"sheet": last.sheet, **_summary(last.hits, [], int(np.median(F[:, 0])), float(np.median(F[:, 1])),
                                           mmpx, notes[:5], frames=used)}
    if mmpx > EYE_MAX_MM_PER_PX:
        out["too_far"] = True
        out["notes"].insert(0, f"too far for dots: {mmpx:.2f} mm/px at the review look (needs <= "
                               f"{EYE_MAX_MM_PER_PX}); look at the sheet from about 20 cm, sheet filling the frame")
    return out


# ------------------------------------------------------------------ overview
def load_results(root: Path) -> List[dict]:
    out = []
    keys = {s.key for s in discover(root)}
    for p in sorted((Path(root) / "results").glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if r.get("schema") == SCHEMA and r.get("key") in keys:   # results of deleted sources are ignored
            r["_json"] = str(p)
            out.append(r)
    return out


USED_FIELDS = ("sheet", "participant", "condition", "display", "protocol", "run", "bad_marks", "aborted",
               "local_time")


def trials(root: Path) -> Dict[str, dict]:
    """sheet -> trial row (participant, condition, display, …): sheets_used*.csv first, then the runs' run.json."""
    root = Path(root)
    out: Dict[str, dict] = {}
    for p in sorted(root.glob("sheets_used*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("sheet"):
                    out[r["sheet"]] = {k: r.get(k, "") for k in USED_FIELDS}
    runs = root / "runs"
    if runs.exists():
        for rj in sorted(_find(runs, "run.json")):
            row = read_run_meta(rj.parent).get("campaign_result")
            if not row:
                continue
            vals = next(csv.reader([row]))
            head = "sheet,participant,condition,display,protocol,run,sheet_digest,targets_shown,next_presses," \
                   "bad_marks,aborted,local_time".split(",")
            d = dict(zip(head, vals))
            out.setdefault(d["sheet"], {k: d.get(k, "") for k in USED_FIELDS})
    return out


def by_sheet(results: List[dict], trial_map: Dict[str, dict]) -> List[dict]:
    """One row per sheet: its trial and, per source kind, the latest result's median and R95."""
    rows: Dict[str, dict] = {}
    for r in sorted(results, key=lambda r: r.get("analysed", "")):
        sh = r.get("sheet")
        if not sh or "error" in r:
            continue
        if r.get("too_far"):
            rows.setdefault(sh, {"sheet": sh, **{k: v for k, v in trial_map.get(sh, {}).items() if k != "sheet"}})
            rows[sh][r["kind"]] = {"too_far": True, "mm_per_px": r.get("mm_per_px")}
            continue
        row = rows.setdefault(sh, {"sheet": sh, **{k: v for k, v in trial_map.get(sh, {}).items() if k != "sheet"}})
        st = r.get("stats", {})
        row[r["kind"]] = {"found": r.get("found"), "targets": r.get("targets"), "median": st.get("median_mm"),
                          "r95": st.get("r95_mm"), "bias": (st.get("bias_x_mm"), st.get("bias_y_mm"))}
    return [rows[k] for k in sorted(rows)]


def pooled(root: Path, results: List[dict], trial_map: Dict[str, dict]) -> List[dict]:
    """Per participant, condition, display and source kind: every hit pooled (the latest result per sheet+kind)."""
    from . import hits as Hm
    latest: Dict[Tuple[str, str], dict] = {}
    for r in sorted(results, key=lambda r: r.get("analysed", "")):
        if r.get("sheet") and "error" not in r and not r.get("too_far"):
            latest[(r["sheet"], r["kind"])] = r
    groups: Dict[tuple, List[Tuple[float, float]]] = defaultdict(list)
    sheets: Dict[tuple, set] = defaultdict(set)
    for (sh, kind), r in latest.items():
        t = trial_map.get(sh, {})
        key = (t.get("participant", "?") or "?", t.get("condition", sh[0]) or sh[0], t.get("display", ""), kind)
        csv_path = Path(r["_json"]).with_suffix(".csv")
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for h in csv.DictReader(f):
                if h["dx_mm"]:
                    groups[key].append((float(h["dx_mm"]), float(h["dy_mm"])))
        sheets[key].add(sh)
    out = []
    for k in sorted(groups):
        st = Hm.stats(groups[k])
        out.append({"participant": k[0], "condition": k[1], "display": k[2], "kind": k[3],
                    "sheets": len(sheets[k]), **st})
    return out


def fmt(v, nd=2) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"
