"""Blind time and camera health of a locking run (T5.1 Phase 2). ``python -m xreal_gates blind RUN``.

Blind time is the part of a protocol block spent in gaps between consecutive locks longer than
``GAP_S`` (1 s); a block's start and end count as gap edges, so a block with no lock is blind for
its whole length. Blocks are grouped by the suffix of their step id: ``still``, ``move``, ``rate``
(compare runs, ``cmpN_<method>_still`` ...) and the standard G5 steps by name.

Camera health, from ``perf.csv`` and the ``stream`` events: received camera fps per fifth of the
run, the CPU clock (``cpu_performance_pct``), and the stream windows (how many, why each ended,
median length). With the decoder falling behind, windows end on ``lag`` and get short; each
restart costs about 2 s without locks (guide §23, §25).

Works on any run with ``events.csv`` lock and step events: Surface gate runs, and Beam Pro runs
that keep the same log format.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GAP_S = 1.0
KINDS = ("still", "move", "rate")
# Standard G5 steps (GateProtocol.G5) that are holds or moves. "acquire" is left out: it is blind until
# the first lock by definition (guide §25's table counts the compare still blocks only).
NAMED = {"static": "still", "back_to_a": "still", "live": "still", "move_head": "move"}


def _rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def kind_of(step_id: str) -> Optional[str]:
    """``still`` / ``move`` / ``rate`` for a step id, or None for steps that are neither."""
    for k in KINDS:
        if step_id.endswith("_" + k) or step_id == k:
            return k
    return NAMED.get(step_id)


def blind_fractions(events: List[dict], gap_s: float = GAP_S) -> Dict[str, Tuple[float, float]]:
    """Per block kind: (blind seconds, total seconds)."""
    ev = [e for e in events if _float(e.get("t_unity")) is not None]
    if not ev:
        return {}
    end = max(float(e["t_unity"]) for e in ev)
    steps = [(float(e["t_unity"]), e["result"]) for e in ev if e["event"] == "step"]
    locks = sorted(float(e["t_unity"]) for e in ev if e["event"] == "lock" and "world=" in e.get("detail", ""))
    out: Dict[str, List[float]] = {}
    for i, (t0, sid) in enumerate(steps):
        t1 = steps[i + 1][0] if i + 1 < len(steps) else end
        kind = kind_of(sid)
        if kind is None or t1 <= t0:
            continue
        pts = [t0] + [t for t in locks if t0 < t < t1] + [t1]
        blind = sum(b - a for a, b in zip(pts, pts[1:]) if b - a > gap_s)
        acc = out.setdefault(kind, [0.0, 0.0])
        acc[0] += blind
        acc[1] += t1 - t0
    return {k: (v[0], v[1]) for k, v in out.items()}


def camera_health(run_dir: Path) -> dict:
    perf = _rows(Path(run_dir) / "perf.csv")
    fps = [v for v in (_float(r.get("cam_fps")) for r in perf) if v is not None]
    clk = [v for v in (_float(r.get("cpu_performance_pct")) for r in perf) if v is not None]

    def fifths(v):
        n = len(v)
        return [round(statistics.median(v[i * n // 5:(i + 1) * n // 5]), 1) for i in range(5)] if n >= 5 else []

    stream = [e for e in _rows(Path(run_dir) / "events.csv") if e.get("event") == "stream"]
    on = [v for v in (_float(e.get("detail", "").split("on_s=")[1].split()[0]) if "on_s=" in e.get("detail", "") else None
                      for e in stream) if v is not None]
    return {"cam_fps_fifths": fifths(fps), "clock_fifths": fifths(clk),
            "windows": len(stream), "ends": dict(Counter(e["result"] for e in stream)),
            "median_window_s": statistics.median(on) if on else None}


def report(run_dir: Path) -> List[str]:
    run_dir = Path(run_dir)
    fr = blind_fractions(_rows(run_dir / "events.csv"))
    if not fr:
        return [f"{run_dir.name}: no protocol steps or locks in events.csv"]
    parts = [f"{k} {100 * b / t:.0f} % of {t:.0f} s" for k, (b, t) in fr.items() if t > 0]
    lines = [f"{run_dir.name}: time in lock gaps > {GAP_S:.0f} s: " + ", ".join(parts)]
    h = camera_health(run_dir)
    if h["cam_fps_fifths"]:
        lines.append(f"  camera fps by fifth of the run: {h['cam_fps_fifths']}; CPU clock %: {h['clock_fifths']}")
    if h["windows"]:
        mw = h["median_window_s"]
        lines.append(f"  stream windows: {h['windows']}, ended by {h['ends']}, median length "
                     + (f"{mw:.1f} s" if mw is not None else "-"))
    return lines


__all__ = ["report", "blind_fractions", "camera_health", "kind_of"]
