"""How old is an Eye frame when the gate pairs it with a head pose? ``python -m xreal_gates latency RUN``.

Temporal calibration from consistency (the same idea as camera-IMU time-offset estimation): a fixed
target must have the same world position in every lock. If each frame is paired with the head pose
of the wrong instant, every head turn misplaces it, so consecutive locks disagree while turning.
For trial delays L the target is re-chained with the head pose L before the frame was sent and the
median lock-to-lock jump is measured, separately while turning and while still; the minimum while
turning estimates the frame age at sending.

Runs of 0.3.004 and earlier paired frames with the head pose at sending: L is the whole frame age
(22 Oct - 2 Nov, five runs: 140-200 ms). Runs of 0.3.005 and later already use ``cam_latency``
(logged); the estimate is then the residual on top of it and should be near 0.
"""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .anchor import F, qrot


def _kv(detail: str) -> dict:
    return dict(p.split("=", 1) for p in detail.split() if "=" in p)


def _rows(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def estimate(run_dir: Path, delays_ms=range(0, 301, 10), turning_deg_s: float = 3.0):
    """List of (delay_ms, median jump while turning mm, while still mm, n turning) and the logged latency."""
    run_dir = Path(run_dir)
    events = _rows(run_dir / "events.csv")
    svc = {int(r["seq"]): r for r in _rows(run_dir / "pose_service.csv") if r["ok"] == "1"}
    pose = [r for r in _rows(run_dir / "pose.csv") if r["phase"] == "update" and r["t_unity"]]
    pt = np.array([float(r["t_unity"]) for r in pose])
    pp = np.array([[float(r[k]) for k in ("pos_x", "pos_y", "pos_z")] for r in pose])
    pq = np.array([[float(r[k]) for k in ("rot_x", "rot_y", "rot_z", "rot_w")] for r in pose])
    hc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["device_xr"]["rgb_camera_pose_from_head"]
    t_hc, q_hc = np.array(hc["position_m"], float), hc["rotation_xyzw"]
    logged = next((float(e["result"]) for e in events if e["event"] == "cam_latency"), 0.0)

    locks: List[Tuple[float, np.ndarray]] = []
    for e in events:
        if e["event"] != "lock" or "world=" not in e["detail"]:
            continue
        d = _kv(e["detail"])
        seq = int(d["seq"])
        if seq not in svc:
            continue
        r = svc[seq]
        t_send = float(e["t_unity"]) - float(d.get("rt_ms", 0)) / 1000.0
        locks.append((t_send, np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])))

    def head(t):
        i = min(max(bisect.bisect_left(pt, t), 1), len(pt) - 1)
        a = float(np.clip((t - pt[i - 1]) / max(pt[i] - pt[i - 1], 1e-6), 0, 1))
        return pp[i - 1] * (1 - a) + pp[i] * a, (pq[i] if a > 0.5 else pq[i - 1])

    def turn_rate(t):
        _, q1 = head(t)
        _, q0 = head(t - 0.1)
        return np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))) / 0.1

    out = []
    if len(locks) < 10:
        return out, logged
    T = np.array([t for t, _ in locks])
    S = np.array([turn_rate(t) for t in T])
    for L in delays_ms:
        W = []
        for t, tv in locks:
            p, q = head(t - logged / 1000.0 - L / 1000.0)
            W.append(qrot(q, qrot(q_hc, F @ tv) + t_hc) + p)
        W = np.array(W)
        d = np.linalg.norm(np.diff(W, axis=0), axis=1) * 1000
        close = np.diff(T) < 0.3
        mv, st = close & (S[1:] > turning_deg_s), close & (S[1:] <= turning_deg_s)
        out.append((int(L), float(np.median(d[mv])) if mv.any() else float("nan"),
                    float(np.median(d[st])) if st.any() else float("nan"), int(mv.sum())))
    return out, logged


def report(run_dir: Path) -> List[str]:
    rows, logged = estimate(run_dir)
    name = Path(run_dir).name
    if not rows:
        return [f"{name}: too few locks for a latency estimate"]
    best: Optional[tuple] = min((r for r in rows if not np.isnan(r[1])), key=lambda r: r[1], default=None)
    lines = [f"{name}: logged cam_latency {logged:.0f} ms; extra delay tried on top of it ->"]
    for L, mv, st, n in rows[::2]:
        lines.append(f"  +{L:3d} ms: lock-to-lock jump while turning {mv:5.1f} mm (n={n}), still {st:4.1f} mm")
    if best:
        lines.append(f"  best: +{best[0]} ms (turning {best[1]:.1f} mm) -> frame age at sending about {logged + best[0]:.0f} ms")
    return lines
