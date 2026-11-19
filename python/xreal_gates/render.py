"""T6.5 render gate: frame pacing and Jack's ratings for each rendering setting.

The Beam Pro run ``<stamp>_render`` switches the rendering settings live (``RenderProbe.cs``) and writes every
frame's XR display counters to ``render.csv``; Jack's taps are ``rating,<setting>:<answer>`` rows in
``events.csv``. For each setting the report gives the Unity frame rate, how many frames took longer than one
display refresh, how often the compositor showed a frame more than once (present count), motion-to-photon,
the extended, early and teared frame counters, and the last rating.

The question it answers: is the flicker frame pacing (fps below the refresh rate, frames shown twice) or
aliasing (hard edges, fixed by MSAA or SMAA)? If the head-locked setting is steady while the room-fixed one
flickers, it is motion; if a setting that reaches the refresh rate is steady, it is pacing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .beampro import load_csv, quantile

FILE = "render.csv"


def _f(v: str) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def ratings(run: Path) -> Dict[str, List[str]]:
    """Every rating tap per setting, in order (the last one counts)."""
    out: Dict[str, List[str]] = {}
    for r in load_csv(Path(run) / "events.csv"):
        if r.get("event") != "rating":
            continue
        setting, _, answer = (r.get("detail") or "").partition(":")
        out.setdefault(setting, []).append(answer)
    return out


def setting_stats(rows: List[dict], refresh_hz: float = 60.0) -> dict:
    """Frame statistics of one setting's rows of render.csv."""
    t = [int(r["mono_ns"]) for r in rows if r.get("mono_ns")]
    dt = [x for x in (_f(r.get("dt_ms")) for r in rows) if x is not None]
    present = [int(float(r["present"])) for r in rows if _f(r.get("present")) is not None and float(r["present"]) >= 0]
    m2p = [x for x in (_f(r.get("m2p_ms")) for r in rows) if x is not None]
    span = (t[-1] - t[0]) / 1e9 if len(t) > 1 else 0.0
    refresh_ms = 1000.0 / refresh_hz

    def delta(col: str) -> Optional[float]:
        v = [x for x in (_f(r.get(col)) for r in rows) if x is not None]
        return v[-1] - v[0] if len(v) > 1 else None

    return {
        "frames": len(rows),
        "fps": (len(t) - 1) / span if span > 0 else float("nan"),
        "dt_median": quantile(dt, 0.5),
        "dt_p95": quantile(dt, 0.95),
        "over_refresh": sum(1 for x in dt if x > refresh_ms * 1.25) / len(dt) if dt else float("nan"),
        "present_1": sum(1 for p in present if p == 1) / len(present) if present else float("nan"),
        "present_2plus": sum(1 for p in present if p >= 2) / len(present) if present else float("nan"),
        "m2p": quantile(m2p, 0.5),
        "extended": delta("extended"),
        "early": delta("early"),
        "teared": delta("teared"),
        "scale": rows[-1].get("scale", "") if rows else "",
        "msaa": rows[-1].get("msaa", "") if rows else "",
    }


def render_report(run: Path, refresh_hz: float = 60.0) -> List[str]:
    rows = load_csv(Path(run) / FILE)
    if not rows:
        return ["   render: no render.csv"]
    order: List[str] = []
    by: Dict[str, List[dict]] = {}
    for r in rows:
        s = r.get("setting", "")
        if s not in by:
            order.append(s)
            by[s] = []
        by[s].append(r)
    rated = ratings(run)
    out = [f"   render: {len(rows)} frames, {len(order)} settings; refresh taken as {refresh_hz:.0f} Hz"]
    out.append("   setting       fps   dt med/p95 ms  >1 refresh  shown 1x/2x+  m2p ms  ext/early/tear  scale msaa  rating")

    def n(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:.0f}"

    for s in order:
        st = setting_stats(by[s], refresh_hz)
        rating = rated.get(s, [])
        out.append(
            f"   {s:12s} {st['fps']:5.1f}   {st['dt_median']:5.1f}/{st['dt_p95']:5.1f}    {st['over_refresh'] * 100:5.1f} %"
            f"    {st['present_1'] * 100:4.0f}/{st['present_2plus'] * 100:3.0f} %   {st['m2p']:5.1f}"
            f"   {n(st['extended'])}/{n(st['early'])}/{n(st['teared'])}"
            f"     {_f(st['scale']) or 0:.2f} {st['msaa']:>4s}  {rating[-1] if rating else 'not rated'}"
            + (f" (taps: {', '.join(rating)})" if len(rating) > 1 else ""))
    for r in load_csv(Path(run) / "events.csv"):
        if r.get("event") == "setting_error":
            out.append(f"   SETTING ERROR {r.get('detail')}")
    return out
