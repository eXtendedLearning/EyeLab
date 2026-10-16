"""One line per gate run: what ran, how it ended, and the numbers that decide G1-G3.

Stdlib only. Lag growth is (receipt time - cam_ts) relative to the run's first frame,
assuming ``cam_ts`` in nanoseconds on one device clock (verified on G1 data).

Since 0.2.007 also: seconds at full CPU clock after Enter XR (until the clock stays below
150 % of nominal for 3 s), the rate at which the SDK *received* Eye frames over USB (native
``rgb recv`` counter; compare with the decoded camera fps), the hottest thermal zone, and the
G4 camera bursts (count and median time to the first frame).
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunSummary:
    run: str
    build: str
    gate: str
    label: str
    result: str
    duration_s: float
    camera_frames: int
    camera_fps_median: Optional[float]
    lag_end_s: Optional[float]
    tracking_ready_after_s: Optional[float]
    clock_median_pct: Optional[float]
    clock_min_pct: Optional[float]
    full_clock_s: Optional[float] = None
    full_clock_all: bool = False
    recv_fps: Optional[float] = None
    temp_max_c: Optional[float] = None
    bursts: int = 0
    burst_first_median_s: Optional[float] = None

    def row(self) -> str:
        def f(v, fmt):
            return "-" if v is None else format(v, fmt)
        return (f"{self.run:36s} {self.build:20s} {self.result:18s} {self.duration_s:6.0f}s "
                f"cam {self.camera_frames:5d} fr {f(self.camera_fps_median, '5.1f')} fps lag {f(self.lag_end_s, '6.1f')}s "
                f"ready {f(self.tracking_ready_after_s, '5.1f')}s clock {f(self.clock_median_pct, '4.0f')}%"
                f"/{f(self.clock_min_pct, '3.0f')}% "
                f"full {'all' if self.full_clock_all else f(self.full_clock_s, '4.0f') + 's':>5s} "
                f"recv {f(self.recv_fps, '4.1f')} temp {f(self.temp_max_c, '4.1f')}C "
                f"bursts {self.bursts:3d} first {f(self.burst_first_median_s, '4.2f')}s")


HEADER = ("run                                  build                result             duration "
          "camera                          tracking     cpu clock median/min  "
          "full-clock-after-XR  usb-recv-fps  hottest-zone  G4-bursts")

#: Below this CPU clock (% of nominal) for 3 consecutive 1 s samples counts as throttled.
THROTTLED_PCT = 150.0
_RGB_RECV = re.compile(r"^(?P<t>[0-9.]+)\s.*rgb recv, (?P<n>\d+),")


def _rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(text) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def summarize(run_dir: Path) -> RunSummary:
    info = {}
    if (run_dir / "run.json").exists():
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    args = info.get("args", {})
    build = info.get("build", {}).get("utc_build", "?")

    events = _rows(run_dir / "events.csv")
    times = [t for t in (_float(e.get("t_unity")) for e in events) if t is not None]
    duration = (max(times) - min(times)) if times else 0.0
    result = next((e["result"] for e in reversed(events) if e.get("event") == "gate_result"), "none")
    ready_wait = next((_float(e["t_unity"]) for e in events
                       if e.get("event") == "tracking_ready" and e.get("result") == "WAIT"), None)
    ready_at = next((_float(e["t_unity"]) for e in events
                     if e.get("event") == "tracking_ready" and e.get("result") == "READY"), None)
    ready_after = (ready_at - ready_wait) if ready_wait is not None and ready_at is not None else None

    cam = _rows(run_dir / "camera.csv")
    fps_med = lag_end = None
    if len(cam) > 2:
        t = [_float(r["t_unity"]) for r in cam]
        ts = [int(r["cam_ts"]) / 1e9 for r in cam]
        per_second = {}
        for ti in t:
            per_second[int(ti)] = per_second.get(int(ti), 0) + 1
        full = sorted(per_second.items())[1:-1]  # drop partial first/last second
        if full:
            fps_med = statistics.median(c for _, c in full)
        lag_end = (t[-1] - ts[-1]) - (t[0] - ts[0])

    perf = _rows(run_dir / "perf.csv")
    clock = [v for v in (_float(r.get("cpu_performance_pct")) for r in perf) if v is not None]

    xr_at = next((_float(e["t_unity"]) for e in events
                  if e.get("event") == "enter_xr" and e.get("result") == "REQUESTED"), None)
    full_s, full_all = _full_clock(perf, xr_at)
    temps = [v for v in (_float(r.get("temp_max_c")) for r in perf) if v is not None]

    bursts = [e for e in events if e.get("event") == "burst" and e.get("result") in ("OK", "SHORT", "NO_FRAMES")]
    firsts = [v for v in (_float(_detail(e.get("detail", ""), "first_s")) for e in bursts) if v is not None]

    return RunSummary(
        run=run_dir.name,
        build=build,
        gate=args.get("gate", "?"),
        label=args.get("label", "?"),
        result=result,
        duration_s=duration,
        camera_frames=len(cam),
        camera_fps_median=fps_med,
        lag_end_s=lag_end,
        tracking_ready_after_s=ready_after,
        clock_median_pct=statistics.median(clock) if clock else None,
        clock_min_pct=min(clock) if clock else None,
        full_clock_s=full_s,
        full_clock_all=full_all,
        recv_fps=_recv_fps(run_dir / "native.log"),
        temp_max_c=max(temps) if temps else None,
        bursts=len(bursts),
        burst_first_median_s=statistics.median(firsts) if firsts else None,
    )


def _detail(text: str, key: str) -> Optional[str]:
    """Value of ``key=value`` in an event detail string."""
    for part in text.split():
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


def _full_clock(perf: List[dict], xr_at: Optional[float]):
    """Seconds from Enter XR until the clock stays below THROTTLED_PCT for 3 samples.

    Returns (seconds, False), or (None, True) when the run never throttled, or (None, False)
    when there was no XR or no clock data.
    """
    if xr_at is None:
        return None, False
    samples = [(t, c) for t, c in ((_float(r.get("t_unity")), _float(r.get("cpu_performance_pct"))) for r in perf)
               if t is not None and c is not None and t >= xr_at]
    if not samples:
        return None, False
    for i in range(len(samples) - 2):
        if all(c < THROTTLED_PCT for _, c in samples[i:i + 3]):
            return samples[i][0] - xr_at, False
    return None, True


def _recv_fps(native_log: Path) -> Optional[float]:
    """Eye frames received over USB per second, from the SDK's cumulative ``rgb recv`` counter."""
    if not native_log.exists():
        return None
    points = []
    with native_log.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RGB_RECV.match(line)
            if m:
                points.append((float(m.group("t")), int(m.group("n"))))
    if len(points) < 2:
        return None
    # The counter restarts with every capture session; use the longest increasing stretch.
    best, start = None, 0
    for i in range(1, len(points) + 1):
        if i == len(points) or points[i][1] < points[i - 1][1]:
            (t0, n0), (t1, n1) = points[start], points[i - 1]
            if t1 > t0 and (best is None or t1 - t0 > best[0]):
                best = (t1 - t0, (n1 - n0) / (t1 - t0))
            start = i
    return best[1] if best else None
