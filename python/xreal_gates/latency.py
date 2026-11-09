"""How old is an Eye frame when the gate pairs it with a head pose? ``python -m xreal_gates latency RUN``.

Temporal calibration from consistency (the same idea as camera-IMU time-offset estimation): a fixed
target must have the same world position in every lock. If each frame is paired with the head pose
of the wrong instant, every head turn misplaces it, so consecutive locks disagree while turning.
For trial delays L the target is re-chained with the head pose L earlier than the pairing under test
and the median lock-to-lock jump is measured, separately while turning and while still; the minimum
while turning estimates the error of that pairing.

Three pairings are scanned:

``logged``
    What the run did. 0.3.004 and earlier: the head pose at sending (age 0). 0.3.005-0.3.006: a fixed
    ``cam_latency`` (160 ms). 0.3.007 and later: each lock's own ``age_ms`` (T5.1 Phase 0). The best L
    is then the residual on top of what ran, and should be near 0.
``service``
    ``base + queue age`` with the queue age from the pose service's log: ``t_host - cam_ts`` above its
    minimum within +-20 s (non-causal, sent frames only). Guide section 23.
``unity``
    The estimator 0.3.007 runs inside Unity, rebuilt from ``camera.csv`` (every arrived frame):
    ``D = t_arrived - cam_ts``, ``D_min`` its causal 20 s minimum, and the frame paired with the head
    pose ``base + (D - D_min) + (t_send - t_arrived)`` before sending. On a 0.3.007 run with the right
    base it reproduces ``logged`` exactly.

The age is not constant (6 Nov, guide section 23): the SDK decode queue grows inside every stream
window, so a frame is between 0 and about 0.45 s older than the freshest frame of its window.
"""

from __future__ import annotations

import bisect
import csv
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .anchor import F, qrot

WINDOW_S = 20.0


def _kv(detail: str) -> dict:
    return dict(p.split("=", 1) for p in detail.split() if "=" in p)


def _rows(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def queue_age(service_rows: List[dict], window_s: float = WINDOW_S) -> dict:
    """Per sequence: ``t_host - cam_ts`` above its minimum within +-window_s (s), the decode-queue age.

    The rolling minimum is the freshest frame's offset, which also absorbs slow drift between the
    host clock and the device clock (about 20 ms over 5 min on 6 Nov).
    """
    rows = [r for r in service_rows if r.get("cam_ts")]
    if not rows:
        return {}
    th = np.array([float(r["t_host"]) for r in rows])
    d = th - np.array([int(r["cam_ts"]) / 1e9 for r in rows])
    out = {}
    for i, r in enumerate(rows):
        env = d[(th > th[i] - window_s) & (th < th[i] + window_s)].min()
        out[int(r["seq"])] = float(d[i] - env)
    return out


class CausalQueueAge:
    """Running ``D - min(D over the last window_s)``; the same algorithm as Unity's ``CameraProbe``.

    ``D = t_arrived - cam_ts * 1e-9`` per frame. A monotonic deque keeps the minimum in O(1) amortised.
    The minimum is carried across stream restarts (a restart's first frame is the freshest).
    """

    def __init__(self, window_s: float = WINDOW_S):
        self.window_s = window_s
        self._q: deque = deque()  # (t, D), D increasing from front to back

    def add(self, t: float, cam_ts_ns: int) -> Tuple[float, float]:
        """Returns (D, queue age) for this frame, both in s."""
        d = t - cam_ts_ns * 1e-9
        while self._q and self._q[-1][1] >= d:
            self._q.pop()
        self._q.append((t, d))
        while self._q[0][0] < t - self.window_s:
            self._q.popleft()
        return d, d - self._q[0][1]


def causal_arrivals(camera_rows: Iterable[dict], window_s: float = WINDOW_S) -> Dict[int, Tuple[float, float]]:
    """cam_ts -> (arrival time t_unity, causal queue age s) for every unique frame of camera.csv."""
    est = CausalQueueAge(window_s)
    out: Dict[int, Tuple[float, float]] = {}
    for r in camera_rows:
        if not r.get("cam_ts") or r.get("dup_ts") == "1":
            continue
        ts = int(r["cam_ts"])
        if ts in out:
            continue
        t = float(r["t_unity"])
        out[ts] = (t, est.add(t, ts)[1])
    return out


class _Lock:
    __slots__ = ("t_send", "seq", "cam_ts", "tv", "age_logged")

    def __init__(self, t_send, seq, cam_ts, tv, age_logged):
        self.t_send, self.seq, self.cam_ts, self.tv, self.age_logged = t_send, seq, cam_ts, tv, age_logged


def load(run_dir: Path):
    """Locks, head interpolator, chain constants and the run's logged fixed latency (ms, or None)."""
    run_dir = Path(run_dir)
    events = _rows(run_dir / "events.csv")
    service = _rows(run_dir / "pose_service.csv")
    svc = {int(r["seq"]): r for r in service if r["ok"] == "1"}
    pose = [r for r in _rows(run_dir / "pose.csv") if r["phase"] == "update" and r["t_unity"]]
    pt = np.array([float(r["t_unity"]) for r in pose])
    pp = np.array([[float(r[k]) for k in ("pos_x", "pos_y", "pos_z")] for r in pose])
    pq = np.array([[float(r[k]) for k in ("rot_x", "rot_y", "rot_z", "rot_w")] for r in pose])
    hc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["device_xr"]["rgb_camera_pose_from_head"]
    t_hc, q_hc = np.array(hc["position_m"], float), hc["rotation_xyzw"]
    fixed = next((float(e["result"]) for e in events if e["event"] == "cam_latency"), 0.0)

    locks: List[_Lock] = []
    for e in events:
        if e["event"] != "lock" or "world=" not in e["detail"]:
            continue
        d = _kv(e["detail"])
        seq = int(d["seq"])
        if seq not in svc:
            continue
        r = svc[seq]
        # 0.3.007 logs the exact send time; before, the lock time minus the round trip (late by up to
        # one Unity frame, since the reply is picked up in the next Update).
        t_send = float(d["t_send"]) if "t_send" in d else float(e["t_unity"]) - float(d.get("rt_ms", 0)) / 1000.0
        age = float(d["age_ms"]) / 1000.0 if "age_ms" in d else fixed / 1000.0
        locks.append(_Lock(t_send, seq, int(d.get("cam_ts") or r["cam_ts"] or 0),
                           np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]), age))

    def head(t):
        i = min(max(bisect.bisect_left(pt, t), 1), len(pt) - 1)
        a = float(np.clip((t - pt[i - 1]) / max(pt[i] - pt[i - 1], 1e-6), 0, 1))
        q = pq[i - 1] * (1 - a) + pq[i] * a * np.sign(float(np.dot(pq[i - 1], pq[i])) or 1.0)
        return pp[i - 1] * (1 - a) + pp[i] * a, q / np.linalg.norm(q)

    per_lock_logged = any(e["event"] == "lock" and "age_ms=" in e["detail"] for e in events)
    return dict(locks=locks, head=head, t_hc=t_hc, q_hc=q_hc, service=service,
                camera=_rows(run_dir / "camera.csv"), fixed_ms=None if per_lock_logged else fixed)


def pairing_times(run: dict, mode: str) -> Optional[np.ndarray]:
    """Per lock, the instant (Unity realtime s) whose head pose the pairing uses, before the trial L.

    ``logged`` is what ran; ``service`` and ``unity`` are with base 0 (the scan supplies the base).
    """
    locks = run["locks"]
    if mode == "logged":
        return np.array([k.t_send - k.age_logged for k in locks])
    if mode == "service":
        qa = queue_age(run["service"])
        if not qa:
            return None
        return np.array([k.t_send - qa.get(k.seq, 0.0) for k in locks])
    if mode == "unity":
        arr = causal_arrivals(run["camera"])
        if not arr:
            return None
        out = []
        for k in locks:
            if k.cam_ts not in arr:
                return None
            t_arr, q = arr[k.cam_ts]
            out.append(t_arr - q)  # = t_send - (queue age + hold)
        return np.array(out)
    raise ValueError(mode)


def scan(run: dict, base_t: np.ndarray, delays_ms=range(0, 301, 10), turning_deg_s: float = 3.0,
         still_m_s: float = 0.03):
    """List of (L ms, median jump while turning mm, while still mm, n turning) for pairing ``base_t - L``.

    Still = turning below ``turning_deg_s`` and the head moving below ``still_m_s`` (0.3.008: the 6DoF
    protocol now slides the head, which is neither turning nor still).
    """
    locks, head = run["locks"], run["head"]

    def turn_rate(t):
        _, q1 = head(t)
        _, q0 = head(t - 0.1)
        return np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))) / 0.1

    def lin_speed(t):
        return float(np.linalg.norm(head(t)[0] - head(t - 0.1)[0])) / 0.1

    T = np.array([k.t_send for k in locks])
    S = np.array([turn_rate(t) for t in T])
    V = np.array([lin_speed(t) for t in T])
    close = np.diff(T) < 0.3
    mv = close & (S[1:] > turning_deg_s)
    st = close & (S[1:] <= turning_deg_s) & (V[1:] <= still_m_s)
    out = []
    for L in delays_ms:
        W = []
        for k, tb in zip(locks, base_t):
            p, q = head(tb - L / 1000.0)
            W.append(qrot(q, qrot(run["q_hc"], F @ k.tv) + run["t_hc"]) + p)
        d = np.linalg.norm(np.diff(np.array(W), axis=0), axis=1) * 1000
        out.append((int(L), float(np.median(d[mv])) if mv.any() else float("nan"),
                    float(np.median(d[st])) if st.any() else float("nan"), int(mv.sum())))
    return out


def estimate(run_dir: Path, delays_ms=range(0, 301, 10), turning_deg_s: float = 3.0, per_frame: bool = False,
             mode: Optional[str] = None):
    """Scan rows and the logged fixed latency in ms (None when the run logged a per-lock age).

    ``mode`` is ``logged`` (default), ``service`` (``per_frame=True``) or ``unity``; see the module text.
    """
    run = load(run_dir)
    mode = mode or ("service" if per_frame else "logged")
    if len(run["locks"]) < 10:
        return [], run["fixed_ms"]
    base_t = pairing_times(run, mode)
    if base_t is None:
        return [], run["fixed_ms"]
    return scan(run, base_t, delays_ms, turning_deg_s), run["fixed_ms"]


def _best(rows):
    return min((r for r in rows if not np.isnan(r[1])), key=lambda r: r[1], default=None)


def report(run_dir: Path) -> List[str]:
    run = load(run_dir)
    name = Path(run_dir).name
    if len(run["locks"]) < 10:
        return [f"{name}: too few locks for a latency estimate"]
    fixed = run["fixed_ms"]
    rows = scan(run, pairing_times(run, "logged"))
    what = f"logged cam_latency {fixed:.0f} ms" if fixed is not None else "logged per-frame age (age_ms per lock)"
    lines = [f"{name}: {what}; extra delay tried on top of it ->"]
    for L, mv, st, n in rows[::2]:
        lines.append(f"  +{L:3d} ms: lock-to-lock jump while turning {mv:5.1f} mm (n={n}), still {st:4.1f} mm")
    best = _best(rows)
    if best:
        lines.append(f"  as run: turning {rows[0][1]:.1f} mm, still {rows[0][2]:.1f} mm; best: +{best[0]} ms "
                     f"(turning {best[1]:.1f} mm)"
                     + (f" -> frame age at sending about {fixed + best[0]:.0f} ms" if fixed is not None
                        else " = residual on top of the per-frame age"))
    if fixed is None:
        ages = np.array([k.age_logged for k in run["locks"]]) * 1000
        lines.append(f"  logged age: median {np.median(ages):.0f} ms, p10 {np.percentile(ages, 10):.0f}, "
                     f"p90 {np.percentile(ages, 90):.0f}")
    for mode, label in (("service", "service queue age (t_host - cam_ts, +-20 s min)"),
                        ("unity", "Unity estimator (camera.csv, causal 20 s min, + hold)")):
        base_t = pairing_times(run, mode)
        if base_t is None:
            continue
        pf = scan(run, base_t)
        pb = _best(pf)
        if not pb:
            continue
        lines.append(f"  per-frame, {label}:")
        lines.append("    " + "  ".join(f"{L}:{mv:.1f}" for L, mv, _, _ in pf[::4]) + "  (base ms: turning mm)")
        lines.append(f"    best base {pb[0]} ms -> turning {pb[1]:.1f} mm, still {pb[2]:.1f} mm")
    qa = np.array(list(queue_age(run["service"]).values())) * 1000
    if qa.size:
        lines.append(f"  service queue age: median {np.median(qa):.0f} ms, p90 {np.percentile(qa, 90):.0f} ms")
    return lines
