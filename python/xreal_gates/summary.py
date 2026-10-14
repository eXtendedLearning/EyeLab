"""One line per gate run: what ran, how it ended, and the numbers that decide G1-G3.

Stdlib only. Lag growth is (receipt time - cam_ts) relative to the run's first frame,
assuming ``cam_ts`` in nanoseconds on one device clock (verified on G1 data).
"""

from __future__ import annotations

import csv
import json
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

    def row(self) -> str:
        def f(v, fmt):
            return "-" if v is None else format(v, fmt)
        return (f"{self.run:36s} {self.build:20s} {self.result:18s} {self.duration_s:6.0f}s "
                f"cam {self.camera_frames:5d} fr {f(self.camera_fps_median, '5.1f')} fps lag {f(self.lag_end_s, '6.1f')}s "
                f"ready {f(self.tracking_ready_after_s, '5.1f')}s clock {f(self.clock_median_pct, '4.0f')}%"
                f"/{f(self.clock_min_pct, '3.0f')}%")


HEADER = ("run                                  build                result             duration "
          "camera                          tracking     cpu clock median/min")


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
    )
