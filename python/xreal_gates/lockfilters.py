"""Lock filters, offline: the three policies of the G5 gate (``LockAnchor.cs``) and how to compare them.

The gate runs ``latest``, ``euro`` (One Euro, Casiez et al. 2012) and ``hysteresis`` (anchor with
hysteresis) side by side on every camera pose and logs each one's output as a ``filters`` event,
so all three are compared on identical measurements. Here:

- ``measurements(run, target)``: the world-frame measurements of a run, from its ``filters`` events
  (0.3.003 and later) or, for older runs, rebuilt from ``pose_service.csv`` + ``pose.csv`` +
  ``run.json`` with the gate's frame chain (view offset not included).
- Python ports of the three filters and of the shown pose's per-frame glide (same constants as the
  C#), to replay any run with other parameters.
- ``compare(...)``: per protocol step and filter, the **jitter** (path speed of the shown plate
  corners, mm/s, 30 fps: how far the drawing travels per second while the target is still) and the
  **error** (median distance of the shown corners from a non-causal reference: the median of all
  measurements within +-1 s, mm). ``rev3`` is the 0.3.002 behaviour for reference: ``latest``
  applied directly, without the per-frame glide.
  In a still phase the reference is the best available stand-in for the truth; shake is what the
  wearer sees as jitter, error includes lag and any offset the filter holds.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .anchor import F, qrot

CONFIRM_M, AGREE_M, AGREE_DEG = 0.10, 0.03, 3.0
NAMES = ("latest", "euro", "hysteresis")
COLUMNS = ("rev3",) + NAMES


# ---------------------------------------------------------------- quaternion helpers (x, y, z, w)

def q_norm(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def q_angle_deg(a, b) -> float:
    d = abs(float(np.dot(q_norm(a), q_norm(b))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def q_slerp(a, b, t: float):
    a, b = q_norm(a), q_norm(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return q_norm(a + t * (b - a))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def alpha(dt: float, cutoff_hz: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-3))
    return 1.0 / (1.0 + tau / max(dt, 1e-3))


# ---------------------------------------------------------------- filters (ports of LockAnchor.cs)

class LockFilter:
    name = ""
    render_tau = 0.06

    def __init__(self):
        self.has = False
        self.p = np.zeros(3)
        self.q = np.array([0, 0, 0, 1.0])
        self.pending = None
        self.applied = 0

    def add(self, p, q, t, w=1.0) -> str:
        p, q = np.asarray(p, float), q_norm(q)
        if not self.has:
            self.reset(p, q, t)
            return "First"
        jump = float(np.linalg.norm(p - self.p))
        if jump > CONFIRM_M:
            if self.pending is not None and np.linalg.norm(p - self.pending[0]) <= AGREE_M \
                    and q_angle_deg(q, self.pending[1]) <= AGREE_DEG:
                self.reset(p, q, t)
                return "Confirmed"
            self.pending = (p, q)
            return "Pending"
        self.pending = None
        return self.update(p, q, t, min(max(w, 0.05), 1.0), jump, q_angle_deg(q, self.q))

    def reset(self, p, q, t):
        self.has, self.p, self.q, self.pending = True, np.asarray(p, float), q_norm(q), None
        self.applied += 1

    def update(self, p, q, t, w, jump, jump_deg) -> str:
        raise NotImplementedError


class Latest(LockFilter):
    name, render_tau = "latest", 0.06

    def update(self, p, q, t, w, jump, jump_deg):
        if jump <= 0.015 and jump_deg <= 1.5:
            self.p, self.q = 0.5 * (self.p + p), q_slerp(self.q, q, 0.5)
            kind = "Smoothed"
        else:
            self.p, self.q = p, q
            kind = "Replaced"
        self.applied += 1
        return kind


class Euro(LockFilter):
    name, render_tau = "euro", 0.05

    def __init__(self, min_cutoff=0.1, beta=2.0, beta_rot=0.015, dcutoff=1.0):
        super().__init__()
        self.min_cutoff, self.beta, self.beta_rot, self.dcutoff = min_cutoff, beta, beta_rot, dcutoff
        self.t = 0.0
        self.v = np.zeros(3)
        self.w = 0.0

    def reset(self, p, q, t):
        super().reset(p, q, t)
        self.t, self.v, self.w = t, np.zeros(3), 0.0

    def update(self, p, q, t, w, jump, jump_deg):
        dt = max(t - self.t, 1e-3)
        self.t = t
        ad = alpha(dt, self.dcutoff)
        self.v = self.v + ad * ((p - self.p) / dt - self.v)
        self.w = self.w + ad * (jump_deg / dt - self.w)
        a = alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(self.v))) * w
        ar = alpha(dt, self.min_cutoff + self.beta_rot * self.w) * w
        self.p = self.p + a * (p - self.p)
        self.q = q_slerp(self.q, q, ar)
        self.applied += 1
        return "Filtered"


class Hysteresis(LockFilter):
    name, render_tau = "hysteresis", 0.3

    def __init__(self, threshold_m=0.005, threshold_deg=0.6, outside_count=3, tau=0.6):
        super().__init__()
        self.th_m, self.th_deg, self.n_out, self.tau = threshold_m, threshold_deg, outside_count, tau
        self.ep, self.eq, self.t, self.out = np.zeros(3), np.array([0, 0, 0, 1.0]), 0.0, 0

    def reset(self, p, q, t):
        super().reset(p, q, t)
        self.ep, self.eq, self.t, self.out = np.asarray(p, float), q_norm(q), t, 0

    def update(self, p, q, t, w, jump, jump_deg):
        dt = max(t - self.t, 1e-3)
        self.t = t
        a = (1.0 - math.exp(-dt / self.tau)) * w
        self.ep = self.ep + a * (p - self.ep)
        self.eq = q_slerp(self.eq, q, a)
        outside = np.linalg.norm(self.ep - self.p) > self.th_m or q_angle_deg(self.eq, self.q) > self.th_deg
        self.out = self.out + 1 if outside else 0
        if self.out >= self.n_out:
            self.p, self.q, self.out = self.ep.copy(), self.eq.copy(), 0
            self.applied += 1
            return "Corrected"
        return "Held"


def make_filters(**params) -> Dict[str, LockFilter]:
    return {"latest": Latest(), "euro": Euro(**params.get("euro", {})),
            "hysteresis": Hysteresis(**params.get("hysteresis", {}))}


# ---------------------------------------------------------------- data

@dataclass
class Meas:
    t: float
    step: str
    p: np.ndarray
    q: np.ndarray
    w: float = 1.0
    logged: Dict[str, tuple] = field(default_factory=dict)   # filter outputs logged by the gate


def _rows(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _kv(detail: str) -> dict:
    return dict(part.split("=", 1) for part in detail.split() if "=" in part)


def _pq(text: str):
    v = [float(x) for x in text.split(",")]
    return np.array(v[:3]), np.array(v[3:7])


def _steps(events):
    steps = [(float(e["t_unity"]), e["result"]) for e in events if e["event"] == "step" and e["t_unity"]]

    def at(t):
        s = "before_protocol"
        for t0, name in steps:
            if t >= t0:
                s = name
        return s
    return at


def measurements(run_dir: Path, target: str) -> List[Meas]:
    run_dir = Path(run_dir)
    events = _rows(run_dir / "events.csv")
    step_at = _steps(events)
    out = []
    for e in events:
        if e["event"] == "filters" and e["result"] == target:
            d = _kv(e["detail"])
            t = float(d["t"])
            p, q = _pq(d["meas"])
            logged = {n: _pq(d[n]) for n in NAMES if n in d}
            out.append(Meas(t, step_at(t), p, q, float(d.get("w", 1.0)), logged))
    if out:
        return out
    return _rebuild(run_dir, target, events, step_at)


def _rebuild(run_dir, target, events, step_at) -> List[Meas]:
    """Older runs: camera pose (pose_service.csv) + head pose at send (pose.csv) + RGB-from-head (run.json)."""
    svc = {}
    for r in _rows(run_dir / "pose_service.csv"):
        if r["ok"] == "1" and r.get("target", target) == target:
            svc[int(r["seq"])] = r
    pose = [r for r in _rows(run_dir / "pose.csv") if r["phase"] == "update" and r["t_unity"]]
    pt = [float(r["t_unity"]) for r in pose]
    hc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["device_xr"]["rgb_camera_pose_from_head"]
    t_hc, q_hc = np.array(hc["position_m"], float), hc["rotation_xyzw"]
    import cv2
    out = []
    for e in events:
        if e["event"] != "lock" or "jump_mm" not in e["detail"]:
            continue
        d = _kv(e["detail"])
        if d.get("target", target) != target or int(d["seq"]) not in svc:
            continue
        r = svc[int(d["seq"])]
        t = float(e["t_unity"]) - float(d.get("rt_ms", 0)) / 1000.0
        R = cv2.Rodrigues(np.array([float(r["rx"]), float(r["ry"]), float(r["rz"])]))[0]
        tv = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
        h = pose[min(bisect.bisect_left(pt, t), len(pose) - 1)]
        qh = [float(h[k]) for k in ("rot_x", "rot_y", "rot_z", "rot_w")]
        ph = np.array([float(h[k]) for k in ("pos_x", "pos_y", "pos_z")])
        p = qrot(qh, qrot(q_hc, F @ tv) + t_hc) + ph
        Rw = _qmat(qh) @ _qmat(q_hc) @ (F @ R @ F)
        out.append(Meas(t, step_at(t), p, _quat(Rw)))
    return out


def _qmat(q):
    return np.column_stack([qrot(q, e) for e in np.eye(3)])


def _quat(R) -> np.ndarray:
    """Rotation matrix to (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return q_norm([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q_norm(q)


def corners(target: str) -> np.ndarray:
    """Centre and bounding-box corners of the target's overlay points, in the Unity-flipped target frame."""
    from .targets import make_target
    pts = np.array(make_target(target).overlay()["points"], float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = (lo + hi) / 2
    cs = [c, [lo[0], lo[1], c[2]], [hi[0], lo[1], c[2]], [hi[0], hi[1], c[2]], [lo[0], hi[1], c[2]]]
    return np.array([F @ np.asarray(x, float) for x in cs])


def _apply(p, q, local) -> np.ndarray:
    return np.array([np.asarray(p) + qrot(q, x) for x in local])


# ---------------------------------------------------------------- comparison

@dataclass
class Score:
    jitter_mm_s: float
    error_mm: float
    n: int


def simulate(meas: List[Meas], local: np.ndarray, fps: float = 30.0, **params):
    """Replays the measurements through fresh filters; returns {name: (frame_times, shown_points)} and kinds."""
    filters = make_filters(**params)
    filters["rev3"] = Latest()
    filters["rev3"].render_tau = 1e-6          # 0.3.002: every correction shown as a step
    shown = {n: None for n in COLUMNS}
    out = {n: ([], []) for n in COLUMNS}
    kinds = {n: [] for n in COLUMNS}
    if not meas:
        return out, kinds
    t, i, dt = meas[0].t, 0, 1.0 / fps
    while t <= meas[-1].t + 1.0:
        while i < len(meas) and meas[i].t <= t:
            m = meas[i]
            for n, f in filters.items():
                kinds[n].append((m.t, f.add(m.p, m.q, m.t, m.w)))
            i += 1
        for n, f in filters.items():
            if not f.has:
                continue
            if shown[n] is None or np.linalg.norm(shown[n][0] - f.p) > 0.2:
                shown[n] = (f.p.copy(), f.q.copy())
            else:
                k = 1.0 - math.exp(-dt / f.render_tau)
                shown[n] = (shown[n][0] + k * (f.p - shown[n][0]), q_slerp(shown[n][1], f.q, k))
            out[n][0].append(t)
            out[n][1].append(_apply(shown[n][0], shown[n][1], local))
        t += dt
    return {n: (np.array(a), np.array(b)) for n, (a, b) in out.items()}, kinds


def compare(meas: List[Meas], target: str, steps: Optional[List[str]] = None, **params) -> "OrderedDict[str, Dict[str, Score]]":
    local = corners(target)
    sim, kinds = simulate(meas, local, **params)
    mt = np.array([m.t for m in meas])
    mpts = np.array([_apply(m.p, m.q, local) for m in meas])
    result: "OrderedDict[str, Dict[str, Score]]" = OrderedDict()
    order = list(OrderedDict.fromkeys(m.step for m in meas))
    for step in (steps or order):
        idx = [k for k, m in enumerate(meas) if m.step == step]
        if len(idx) < 5:
            continue
        t0, t1 = meas[idx[0]].t, meas[idx[-1]].t
        scores = {}
        for n in COLUMNS:
            ft, fp = sim[n]
            sel = (ft >= t0) & (ft <= t1)
            if sel.sum() < 3:
                continue
            P = fp[sel]
            path = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(axis=1).sum()
            jitter = float(path / max(ft[sel][-1] - ft[sel][0], 1e-6)) * 1000
            errs = []
            for tt, pts in zip(ft[sel], P):
                near = (mt >= tt - 1.0) & (mt <= tt + 1.0)
                if near.sum() >= 3:
                    ref = np.median(mpts[near], axis=0)
                    errs.append(np.linalg.norm(pts - ref, axis=1).mean())
            scores[n] = Score(jitter, float(np.median(errs)) * 1000 if errs else float("nan"), len(idx))
        result[step] = scores
    return result


def ratings(run_dir: Path) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for e in _rows(Path(run_dir) / "events.csv"):
        if e["event"] == "rating":
            out.setdefault(_kv(e["detail"]).get("method", "?"), []).append(int(e["result"]))
    return out


def report(run_dir: Path, target: str, **params) -> List[str]:
    meas = measurements(run_dir, target)
    if not meas:
        return [f"{Path(run_dir).name}: no measurements for {target}"]
    src = "filters events" if meas[0].logged else "rebuilt from pose_service.csv"
    lines = [f"{Path(run_dir).name} [{target}]: {len(meas)} measurements ({src}); "
             f"jitter = path speed of the shown corners (mm/s), error = distance from the +-1 s median of the "
             f"measurements (mm); rev3 = 0.3.002 (latest, no glide)"]
    lines.append(f"  {'step':16s} " + "  ".join(f"{n:>21s}" for n in COLUMNS))
    for step, sc in compare(meas, target, **params).items():
        cells = []
        for n in COLUMNS:
            s = sc.get(n)
            cells.append(f"{'-':>21s}" if s is None else f"jit {s.jitter_mm_s:6.1f} err {s.error_mm:5.1f}".rjust(21))
        lines.append(f"  {step:16s} " + "  ".join(cells))
    res = compare(meas, target, **params)
    still = [sc for step, sc in res.items() if step.endswith("_still")]
    if still:
        lines.append("  all *_still blocks (mean): " + "   ".join(
            f"{n} jitter {np.mean([sc[n].jitter_mm_s for sc in still if n in sc]):.1f} mm/s "
            f"error {np.nanmean([sc[n].error_mm for sc in still if n in sc]):.1f} mm" for n in COLUMNS))
    r = ratings(run_dir)
    if r:
        lines.append("  ratings (1-5): " + ", ".join(f"{n} {np.mean(v):.1f} ({v})" for n, v in sorted(r.items())))
    return lines
