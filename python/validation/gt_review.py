"""Manual review of the analysis' ground truth (the saddles found by ``detect``).

Two walks over recorded sessions, both driven by ``gt_review_gui``:

- **audit**: a reproducible random sample of (session, frame, point). Each item is
  accepted, corrected or marked invalid. It measures how often the automatic
  ground truth is wrong without changing the analysis.
- **missing**: every point the analysis did not find (or found far from its face
  plane), over all sessions. The proposal is the point's position in the session's
  other frames, snapped in this one.
- **session**: every point of one session in frame order. A decision is carried to
  the following frames (the scene is static) and re-snapped there; a later frame is
  shown again only where the carried point no longer snaps in place (``strict``:
  also where the automatic point disagrees with it).

An audit proposes the reviewed truth where a session review exists, the automatic
point otherwise, so the same tool also spot-checks a finished session review.

A click is a seed: it is snapped to the nearest X-junction (sub-pixel) unless the
user turns snapping off. Decisions are appended to a CSV, last row per
(session, frame, point) wins, so a review can be stopped and resumed. Session and
missing-point decisions go to ``session.csv``, audits to ``audit.csv``, both in the
review folder (``review_dir``); ``load_overrides`` merges them for the analysis
(``metrics.registration_errors``). Pixel
coordinates are raw (distorted) image pixels, OpenCV convention (pixel centres on
integers). No tkinter here: everything is testable headless.
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .camera import apply_h, undistort
from .detect import Located, detect_saddle, is_x_junction, locate_points
from .plates import build_target, layout_from_provenance
from .session import ArchivedSession, check_meta, load_session, parse_records
from .target import Target, load_target

Key = Tuple[str, int, str]          # (session id, frame index i, point id)
UV = Tuple[float, float]


# ── recorded campaign, folder or zip ──
class RunSource:
    """Read-only view of a campaign folder (``validation_runs/``) or a zip of it."""

    def __init__(self, path):
        self.path = Path(path)
        self._zip: Optional[zipfile.ZipFile] = None
        if self.path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.path)
            names = self._zip.namelist()
            camp = min((n for n in names if n.endswith("campaign.json")), key=len)
            self._prefix = camp[: -len("campaign.json")]
            self._names = set(names)
        elif not (self.path / "campaign.json").exists():
            raise FileNotFoundError(f"{self.path}: no campaign.json")

    def _read(self, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(self._prefix + rel)
        return (self.path / rel).read_bytes()

    def campaign(self) -> dict:
        return json.loads(self._read("campaign.json"))

    def session_ids(self) -> List[str]:
        if self._zip is not None:
            rels = [n[len(self._prefix):] for n in self._names if n.startswith(self._prefix)]
        else:
            rels = [p.relative_to(self.path).as_posix() for p in self.path.glob("sessions/*/meta.json")]
        out = []
        for r in rels:
            parts = r.split("/")
            if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "meta.json" \
                    and not parts[1].startswith("_"):
                out.append(parts[1])
        return sorted(out)

    def meta(self, sid: str) -> dict:
        return json.loads(self._read(f"sessions/{sid}/meta.json"))

    def log(self, sid: str) -> List[dict]:
        text = self._read(f"sessions/{sid}/log.jsonl").decode("utf-8")
        return [json.loads(line) for line in io.StringIO(text) if line.strip()]

    def load_session(self, sid: str):
        """The session as ``session.Session`` (frames read from the zip if needed)."""
        if self._zip is None:
            return load_session(self.path / "sessions" / sid)
        meta = self.meta(sid)
        check_meta(meta, f"{self.path.name}:{sid}")
        text = self._read(f"sessions/{sid}/log.jsonl").decode("utf-8")
        return ArchivedSession(Path("sessions") / sid, meta, parse_records(text.splitlines()),
                               reader=lambda name: self.frame_gray(sid, name))

    def frame_gray(self, sid: str, name: str) -> np.ndarray:
        buf = np.frombuffer(self._read(f"sessions/{sid}/frames/{name}"), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"{sid}/frames/{name}")
        return img


def review_dir(source_path) -> Path:
    """Where a source's review lives: next to a zip in a folder of the same name
    (``imports/x.zip`` -> ``imports/x/gt_review``), inside a campaign folder otherwise."""
    p = Path(source_path)
    return (p.parent / p.stem if p.suffix.lower() == ".zip" else p) / "gt_review"


def reports_dir(source_path) -> Path:
    """Re-analysis with the reviewed ground truth (``reanalyse``)."""
    p = Path(source_path)
    return p.parent / p.stem / "reports" if p.suffix.lower() == ".zip" else p / "reports_reviewed"


def target_for(source: RunSource, meta: dict, config_dir) -> Target:
    """The measured target of the session's plate, rebuilt from the caliper readings
    in ``campaign.json`` (the ``*.measured.target.json`` may only exist on the capture
    machine). Without readings, the nominal target."""
    plate = (meta.get("condition") or {}).get("plate", "A")
    nominal = load_target(Path(config_dir) / "targets" / f"plate_{plate}.target.json")
    rd = (source.campaign().get("plates", {}).get(plate) or {}).get("readings")
    if not rd:
        return nominal
    bars = nominal.provenance["bars_nominal_mm"]
    return build_target(layout_from_provenance(nominal.provenance),
                        float(np.mean(rd["x_mm"])) / bars["x"], float(np.mean(rd["y_mm"])) / bars["y"],
                        float(rd.get("fold_deg") or 90.0), float(rd.get("uncertainty_mm") or math.nan),
                        measured=True)


# ── per-frame data ──
@dataclass
class FrameView:
    sid: str
    i: int
    name: str
    gray: np.ndarray
    loc: Located
    eyelab: Dict[str, np.ndarray]      # EyeLab's projected points (raw px)
    K: np.ndarray
    dist: np.ndarray

    def auto(self, pid: str) -> Optional[np.ndarray]:
        p = self.loc.points.get(pid)
        return None if p is None else np.asarray(p, float)


@dataclass
class SessionInfo:
    sid: str
    meta: dict
    target: Target
    frames: List[Tuple[int, str]]      # (log index i, frame file) of saved frames
    eyelab: Dict[int, Dict[str, np.ndarray]]

    @property
    def condition(self) -> dict:
        return self.meta.get("condition") or {}

    @property
    def point_ids(self) -> List[str]:
        return self.target.ids()


class Review:
    """Shared data access: sessions, frames (cached), auto detections, snapping."""

    def __init__(self, source: RunSource, config_dir, cache_frames: int = 64):
        self.source = source
        self.config_dir = Path(config_dir)
        self._sessions: Dict[str, SessionInfo] = {}
        self._frames: Dict[Tuple[str, int], FrameView] = {}
        self._cache_frames = cache_frames

    def session(self, sid: str) -> SessionInfo:
        if sid not in self._sessions:
            meta = self.source.meta(sid)
            log = self.source.log(sid)
            frames = [(int(e["i"]), e["frame"]) for e in log if e.get("frame")]
            eyelab = {int(e["i"]): {k: np.asarray(v, float) for k, v in (e.get("projected") or {}).items()}
                      for e in log if e.get("frame")}
            self._sessions[sid] = SessionInfo(sid, meta, target_for(self.source, meta, self.config_dir),
                                              frames, eyelab)
        return self._sessions[sid]

    def frame(self, sid: str, i: int) -> FrameView:
        key = (sid, i)
        if key not in self._frames:
            s = self.session(sid)
            name = dict(s.frames)[i]
            gray = self.source.frame_gray(sid, name)
            K = np.asarray(s.meta["camera_matrix"], float)
            dist = np.asarray(s.meta["dist_coeffs"], float)
            loc = locate_points(gray, s.target, K, dist)
            if len(self._frames) >= self._cache_frames:
                self._frames.pop(next(iter(self._frames)))
            self._frames[key] = FrameView(sid, i, name, gray, loc, s.eyelab.get(i, {}), K, dist)
        return self._frames[key]

    def square_px(self, fv: FrameView, pid: str) -> float:
        """Approximate printed-square size (px) at this point, from the face homography."""
        t = self.session(fv.sid).target
        face = t.point_face[pid]
        H = fv.loc.homographies.get(face)
        if H is None:
            return 12.0
        sq = float(t.provenance.get("square_mm", 15.0)) / 1000.0
        p = t.face_coords([pid])[0]
        q = apply_h(H, np.array([p, p + [sq, 0], p + [0, sq]]))
        return float(min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[0])))

    def residual_px(self, fv: FrameView, pid: str) -> float:
        """Distance (undistorted px) of the auto point from its face homography."""
        t = self.session(fv.sid).target
        H = fv.loc.homographies.get(t.point_face[pid])
        a = fv.auto(pid)
        if H is None or a is None:
            return math.nan
        pred = apply_h(H, t.face_coords([pid]))[0]
        return float(np.linalg.norm(pred - undistort([a], fv.K, fv.dist)[0]))

    def snap(self, fv: FrameView, pid: str, uv, radius: float = 3.0) -> Optional[np.ndarray]:
        """Nearest true X-junction to ``uv`` (sub-pixel), or None."""
        sq = self.square_px(fv, pid)
        pt = detect_saddle(fv.gray, uv, radius, 3, sigma=float(np.clip(sq / 10, 1.0, 2.0)))
        if pt is None or not is_x_junction(fv.gray, pt, float(np.clip(0.4 * sq, 2.5, 12.0))):
            return None
        return pt


# ── decisions ──
FIELDS = ("session", "frame", "point_id", "mode", "status", "ref_kind", "ref_u", "ref_v", "auto_u",
          "auto_v", "final_u", "final_v", "snapped", "d_px", "test", "seed", "time")
STATUSES = ("ok", "corrected", "invalid", "carried")


class DecisionStore:
    """Append-only CSV; the last row of a key wins."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows: Dict[Key, dict] = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows[(r["session"], int(r["frame"]), r["point_id"])] = r

    def get(self, key: Key) -> Optional[dict]:
        return self.rows.get(key)

    def put_many(self, rows: List[dict]) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, FIELDS)
            if new:
                w.writeheader()
            for r in rows:
                r = {k: r.get(k, "") for k in FIELDS}
                w.writerow(r)
                self.rows[(r["session"], int(r["frame"]), str(r["point_id"]))] = {k: str(v) for k, v in r.items()}

    def put(self, row: dict) -> None:
        self.put_many([row])


def status_for(item: "Item", final: Optional[np.ndarray]) -> str:
    """ok = the judged point (``item.ref``) confirmed, corrected = moved (or there was
    none), invalid = not judgeable."""
    if final is None:
        return "invalid"
    if item.ref is not None and np.linalg.norm(np.asarray(final) - item.ref) < 1e-3:
        return "ok"
    return "corrected"


def make_row(item: "Item", mode: str, status: str, final: Optional[np.ndarray], snapped: bool,
             seed="") -> dict:
    a, ref = item.auto, item.ref
    d = float(np.linalg.norm(final - ref)) if final is not None and ref is not None else ""
    f = lambda v, k: f"{v[k]:.3f}" if v is not None else ""
    return {"session": item.sid, "frame": item.i, "point_id": item.pid, "mode": mode,
            "status": status, "ref_kind": item.ref_kind, "ref_u": f(ref, 0), "ref_v": f(ref, 1),
            "auto_u": f(a, 0), "auto_v": f(a, 1), "final_u": f(final, 0),
            "final_v": f(final, 1), "snapped": int(bool(snapped)),
            "d_px": f"{d:.3f}" if d != "" else "", "test": item.test, "seed": seed,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


# ── walks ──
@dataclass
class Item:
    sid: str
    i: int
    pid: str
    test: str
    auto: Optional[np.ndarray]
    proposal: Optional[np.ndarray]      # what Space would confirm
    note: str = ""
    ref: Optional[np.ndarray] = None    # the point being judged (d_px is measured from it)
    ref_kind: str = "auto"              # "auto" or "reviewed" (truth from a session review)

    def __post_init__(self):
        if self.ref is None and self.ref_kind == "auto":
            self.ref = self.auto


@dataclass
class _Walk:
    review: Review
    store: DecisionStore
    history: List[Key] = field(default_factory=list)
    pos: int = -1                       # index into history of the shown item

    mode = ""

    def item(self, key: Key, proposal=None, note="") -> Item:
        sid, i, pid = key
        fv = self.review.frame(sid, i)
        a = fv.auto(pid)
        return Item(sid, i, pid, str(self.review.session(sid).condition.get("test", "")), a,
                    a if proposal is None else np.asarray(proposal, float), note)

    def back(self) -> Optional[Item]:
        if self.pos > 0:
            self.pos -= 1
            return self._reshow(self.history[self.pos])
        return None

    def _reshow(self, key: Key) -> Item:
        prev = self.store.get(key)
        it = self.item(key)
        if prev and prev.get("final_u"):
            it.proposal = np.array([float(prev["final_u"]), float(prev["final_v"])])
            it.note = f"previous decision: {prev['status']}"
        return it


class AuditWalk(_Walk):
    """Random stratified sample over sessions; same seed, same sample."""

    mode = "audit"

    def __init__(self, review: Review, store: DecisionStore, sids: List[str], n: int, seed: int,
                 truth: Optional[DecisionStore] = None):
        super().__init__(review, store)
        self.seed = seed
        self.truth = truth
        self.keys = audit_keys(review, sids, n, seed)

    def item(self, key: Key, proposal=None, note="") -> Item:
        it = super().item(key, proposal, note)
        t = self.truth.get(key) if self.truth is not None else None
        if t is not None:
            final = np.array([float(t["final_u"]), float(t["final_v"])]) if t.get("final_u") else None
            it.ref, it.ref_kind = final, "reviewed"
            if proposal is None:
                it.proposal = final
            it.note = it.note or f"proposal from session review ({t['status']})"
        return it

    def total(self) -> int:
        return len(self.keys)

    def done(self) -> int:
        return sum(1 for k in self.keys if self.store.get(k))

    def next(self) -> Optional[Item]:
        if self.pos + 1 < len(self.history):
            self.pos += 1
            return self._reshow(self.history[self.pos])
        for k in self.keys:
            if not self.store.get(k) and k not in self.history:
                self.history.append(k)
                self.pos = len(self.history) - 1
                return self.item(k)
        return None

    def decide(self, it: Item, status: str, final, snapped: bool) -> None:
        self.store.put(make_row(it, self.mode, status, final, snapped, self.seed))


def audit_keys(review: Review, sids: List[str], n: int, seed: int) -> List[Key]:
    """``n`` distinct (session, frame, point) spread evenly over ``sids``, shuffled."""
    rng = np.random.default_rng(seed)
    sids = sorted(sids)
    base, extra = divmod(n, len(sids))
    bonus = set(rng.choice(len(sids), size=extra, replace=False).tolist()) if extra else set()
    keys: List[Key] = []
    for j, sid in enumerate(sids):
        s = review.session(sid)
        pool = [(i, pid) for i, _ in s.frames for pid in s.point_ids]
        k = min(base + (1 if j in bonus else 0), len(pool))
        for idx in rng.choice(len(pool), size=k, replace=False):
            i, pid = pool[int(idx)]
            keys.append((sid, int(i), pid))
    order = rng.permutation(len(keys))
    return [keys[int(o)] for o in order]


class SessionWalk(_Walk):
    """All points of one session in frame order, with carry-over of decisions."""

    mode = "session"

    def __init__(self, review: Review, store: DecisionStore, sid: str, tol_px: float = 0.5,
                 show_all: bool = False, strict: bool = False):
        super().__init__(review, store)
        self.sid = sid
        self.tol = tol_px
        self.show_all = show_all
        self.strict = strict
        self.auto_disagree = 0
        s = review.session(sid)
        self.keys = [(sid, i, pid) for i, _ in s.frames for pid in s.point_ids]
        self.carried: Dict[str, Optional[np.ndarray]] = {}
        self._cursor = 0
        self.auto_carried = 0

    def total(self) -> int:
        return len(self.keys)

    def done(self) -> int:
        return sum(1 for k in self.keys if self.store.get(k))

    def _remember(self, pid: str, row: dict) -> None:
        if row["status"] == "invalid" or not row.get("final_u"):
            self.carried[pid] = None
        else:
            self.carried[pid] = np.array([float(row["final_u"]), float(row["final_v"])])

    def next(self) -> Optional[Item]:
        if self.pos + 1 < len(self.history):
            self.pos += 1
            return self._reshow(self.history[self.pos])
        pending: List[dict] = []
        try:
            while self._cursor < len(self.keys):
                key = self.keys[self._cursor]
                self._cursor += 1
                _, i, pid = key
                prev = self.store.get(key)
                if prev:                                   # resume: replay the decision
                    self._remember(pid, prev)
                    continue
                if pid not in self.carried:                # first time this point is seen
                    return self._show(key)
                fv = self.review.frame(self.sid, i)
                last = self.carried[pid]
                cand = None if last is None else self.review.snap(fv, pid, last)
                a = fv.auto(pid)
                if last is None:                           # invalid so far: ask again only if auto exists
                    if a is None and not self.show_all:
                        it = self.item(key)
                        pending.append(make_row(it, self.mode, "carried", None, False))
                        self.carried[pid] = None
                        continue
                    return self._show(key)
                stable = cand is not None and np.linalg.norm(cand - last) <= self.tol
                agree = cand is not None and a is not None and np.linalg.norm(cand - a) <= self.tol
                if stable and (agree or not self.strict) and not self.show_all:
                    it = self.item(key)
                    pending.append(make_row(it, self.mode, "carried", cand, True))
                    self.carried[pid] = cand
                    self.auto_carried += 1
                    self.auto_disagree += int(not agree)
                    continue
                if cand is None:
                    note = "carried point did not snap here"
                elif not stable:
                    note = f"carried point snapped {np.linalg.norm(cand - last):.2f} px away"
                elif a is None:
                    note = "no automatic point"
                else:
                    note = f"automatic point {np.linalg.norm(cand - a):.2f} px from carried"
                return self._show(key, cand if cand is not None else last, note)
            return None
        finally:
            self.store.put_many(pending)

    def _show(self, key: Key, proposal=None, note="") -> Item:
        self.history.append(key)
        self.pos = len(self.history) - 1
        return self.item(key, proposal, note)

    def decide(self, it: Item, status: str, final, snapped: bool) -> None:
        row = make_row(it, self.mode, status, final, snapped)
        self.store.put(row)
        self._remember(it.pid, row)


class MissingWalk(_Walk):
    """Points without an automatic truth (or one farther than ``max_residual_px`` from
    its face plane), across sessions, skipping keys already decided anywhere."""

    mode = "missing"

    def __init__(self, review: Review, store: DecisionStore, scan: List[dict],
                 also_decided: Optional[DecisionStore] = None, max_residual_px: float = math.inf):
        super().__init__(review, store)
        self.other = also_decided
        self.typical: Dict[Tuple[str, str], np.ndarray] = {}
        by: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        for r in scan:
            if r["auto_u"]:
                by.setdefault((r["session"], r["point_id"]), []).append((float(r["auto_u"]), float(r["auto_v"])))
        if also_decided is not None or store is not None:     # reviewed truth beats auto
            rev: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
            for st in (store, also_decided):
                for (sid, _, pid), r in (st.rows.items() if st is not None else []):
                    if r.get("final_u"):
                        rev.setdefault((sid, pid), []).append((float(r["final_u"]), float(r["final_v"])))
            by.update(rev)
        self.typical = {k: np.median(np.array(v), axis=0) for k, v in by.items()}
        self.keys = [(r["session"], int(r["frame"]), r["point_id"]) for r in scan
                     if not r["auto_u"] or (r["residual_px"] and float(r["residual_px"]) > max_residual_px)]

    def _decided(self, k: Key) -> bool:
        return bool(self.store.get(k) or (self.other is not None and self.other.get(k)))

    def total(self) -> int:
        return len(self.keys)

    def done(self) -> int:
        return sum(1 for k in self.keys if self._decided(k))

    def item(self, key: Key, proposal=None, note="") -> Item:
        it = super().item(key, proposal, note)
        if proposal is None:
            sid, i, pid = key
            fv = self.review.frame(sid, i)
            typ = self.typical.get((sid, pid))
            radius = float(np.clip(0.35 * self.review.square_px(fv, pid), 3.0, 8.0))
            snapped = None if typ is None else self.review.snap(fv, pid, typ, radius=radius)
            if snapped is not None:
                it.proposal, it.note = snapped, "proposal: this point in the other frames, snapped"
            elif it.auto is None:
                e = fv.eyelab.get(pid)
                it.proposal = typ if typ is not None else e
                it.note = "proposal not snapped: check it"
        return it

    def next(self) -> Optional[Item]:
        if self.pos + 1 < len(self.history):
            self.pos += 1
            return self._reshow(self.history[self.pos])
        for k in self.keys:
            if not self._decided(k) and k not in self.history:
                self.history.append(k)
                self.pos = len(self.history) - 1
                return self.item(k)
        return None

    def decide(self, it: Item, status: str, final, snapped: bool) -> None:
        self.store.put(make_row(it, self.mode, status, final, snapped))


SCAN_FIELDS = ("session", "frame", "point_id", "auto_u", "auto_v", "residual_px")


def scan(review: Review, sids: List[str], path, progress=print) -> List[dict]:
    """Automatic truth of every saved frame of ``sids`` (cached in ``path``; sessions
    already in the cache are not scanned again)."""
    path = Path(path)
    rows: List[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    have = {r["session"] for r in rows}
    todo = [s for s in sids if s not in have]
    for j, sid in enumerate(todo, 1):
        s = review.session(sid)
        new = []
        for i, _ in s.frames:
            fv = review.frame(sid, i)
            for pid in s.point_ids:
                a = fv.auto(pid)
                res = review.residual_px(fv, pid)
                new.append({"session": sid, "frame": i, "point_id": pid,
                            "auto_u": f"{a[0]:.3f}" if a is not None else "",
                            "auto_v": f"{a[1]:.3f}" if a is not None else "",
                            "residual_px": "" if math.isnan(res) else f"{res:.3f}"})
        path.parent.mkdir(parents=True, exist_ok=True)
        first = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, SCAN_FIELDS)
            if first:
                w.writeheader()
            w.writerows(new)
        rows += [{k: str(v) for k, v in r.items()} for r in new]
        progress(f"  scanned {sid} ({j}/{len(todo)})")
    wanted = set(sids)
    return [r for r in rows if r["session"] in wanted]


def load_overrides(folder) -> Dict[str, Dict[int, Dict[str, Optional[np.ndarray]]]]:
    """Reviewed truth of a review folder, ``{session: {frame: {point: raw px or None}}}``.
    ``session.csv`` first, then ``audit.csv`` (an audit is a later check, it wins)."""
    out: Dict[str, Dict[int, Dict[str, Optional[np.ndarray]]]] = {}
    for name in ("session.csv", "audit.csv"):
        f = Path(folder) / name
        if not f.exists():
            continue
        for (sid, i, pid), uv in iter_overrides(DecisionStore(f)):
            out.setdefault(sid, {}).setdefault(i, {})[pid] = uv
    return out


def reanalyse(source: RunSource, review: Review, folder, out_dir, tests: List[str],
              every: int = 1, plots: bool = True, progress=print) -> Path:
    """Analyse the source's sessions of ``tests`` with the reviewed ground truth."""
    from .report import aggregate, analyze_session
    ov = load_overrides(folder)
    out_dir = Path(out_dir)
    for test in tests:
        sids = [s for s in source.session_ids() if s.split("_")[0] == test]
        sums = []
        for j, sid in enumerate(sids, 1):
            sm = analyze_session(source.load_session(sid), review.session(sid).target,
                                 out_dir / test / f"sessions_{sid}", every=every, plots=plots,
                                 overrides=ov.get(sid))
            sums.append(sm)
            r = sm["registration"]
            progress(f"  {sid} ({j}/{len(sids)}): err_mm rmse {r['err_mm'].get('rmse', math.nan):.3f} "
                     f"p95 {r['err_mm'].get('p95', math.nan):.3f}, reviewed points "
                     f"{r['qa']['ground_truth_points_reviewed']}")
        if sums:
            aggregate(sums, out_dir / test)
    return out_dir


# ── summary ──
def upper_bound_95(k: int, n: int) -> float:
    """One-sided 95 % Clopper-Pearson upper bound of a proportion k/n."""
    if n == 0:
        return math.nan
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.95, k + 1, n - k))
    except ImportError:                                   # exact for k = 0
        return 1.0 - 0.05 ** (1.0 / n) if k == 0 else math.nan


def summarize(store: DecisionStore, tol_px: float = 0.5) -> List[dict]:
    """Per (mode, test, judged point kind): counts, failure rate of the judged point
    (automatic or reviewed) and its one-sided 95 % bound. A failure is a final truth
    farther than ``tol_px`` from the judged point, or a truth where there was none."""
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in store.rows.values():
        groups.setdefault((r["mode"], r["test"], r.get("ref_kind") or "auto"), []).append(r)
    out = []
    for (mode, test, kind), rows in sorted(groups.items()):
        judged = [r for r in rows if r["status"] != "invalid" and r.get("final_u")]
        fail = [r for r in judged if not r["d_px"] or float(r["d_px"]) > tol_px]
        d = [float(r["d_px"]) for r in fail if r["d_px"]]
        n, k = len(judged), len(fail)
        out.append({"mode": mode, "test": test, "ref": kind, "decided": len(rows), "judged": n,
                    "ok": sum(r["status"] == "ok" for r in rows),
                    "carried": sum(r["status"] == "carried" for r in rows),
                    "corrected": sum(r["status"] == "corrected" for r in rows),
                    "invalid": sum(r["status"] == "invalid" for r in rows),
                    "failures": k, "failure_rate": k / n if n else math.nan,
                    "failure_rate_u95": upper_bound_95(k, n),
                    "fail_d_px_median": float(np.median(d)) if d else math.nan})
    return out


def iter_overrides(store: DecisionStore) -> Iterator[Tuple[Key, Optional[np.ndarray]]]:
    """Final truth per key (None = invalid), for the analysis to use instead of auto."""
    for key, r in store.rows.items():
        if r["status"] == "invalid":
            yield key, None
        elif r["final_u"]:
            yield key, np.array([float(r["final_u"]), float(r["final_v"])])
