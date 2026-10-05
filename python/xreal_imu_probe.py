#!/usr/bin/env python3
"""Stage S3 — record the glasses' IMU and characterise it.

Two protocols, both with the glasses on USB and nothing else running
(``xreal_shutdown.bat`` first if unsure):

**Rest** — timing, bias, noise, accelerometer unit, gravity axis::

    python python/xreal_imu_probe.py

  Glasses flat on a table, arms folded, lenses facing forward, untouched for
  the whole recording (default 30 s). A beep marks start and end.

**Turn** — gyro unit and axis mapping, one known 90 deg rotation::

    python python/xreal_imu_probe.py --turn yaw

  Glasses flat on the table against a straight edge. After the first beep,
  wait ~3 s, rotate them 90 deg to the LEFT (counter-clockwise seen from
  above) about the vertical, keeping them flat; align with the other edge,
  let go and wait for the second beep. ``--turn roll`` / ``--turn pitch`` are
  the same with the glasses tipped 90 deg onto a temple / onto the lenses.

Each run writes the raw samples to ``python/.logs/imu_<timestamp>_<mode>.csv``
and a summary line to ``python/.logs/xreal_imu_probe.jsonl``. ``--analyse CSV``
re-runs the analysis on an existing recording without touching hardware.

Exit status: 0 if samples were recorded and analysed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import imu_analysis
from native_isolation import append_jsonl, emit, run_json_worker

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / ".logs"
LOG_PATH = LOG_DIR / "xreal_imu_probe.jsonl"
_WORKER_FLAG = "--_worker"

DEFAULT_DURATION_S = {"rest": 30.0, "turn": 15.0}

#: API bring-up takes 2 s warm and up to ~16 s cold (S2); leave room for both.
STARTUP_ALLOWANCE_S = 90.0


def _beep(freq: int, ms: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:  # noqa: BLE001 - no sound is not a failure
        pass


def _run_worker(spec_json: str) -> int:
    sys.path.insert(0, str(HERE))
    import xreal_imu

    spec = json.loads(spec_json)
    report = xreal_imu.record(
        duration_s=spec["duration_s"],
        csv_path=Path(spec["csv_path"]),
        entry=spec["entry"],
        directory=Path(spec["dir"]) if spec.get("dir") else None,
        on_started=lambda: _beep(880, 250),
        on_finished=lambda: (_beep(660, 150), _beep(660, 150)),
    )
    emit(report.to_dict())
    return 0


def run_isolated(duration_s: float, csv_path: Path, entry: str,
                 directory: Optional[Path]) -> dict:
    spec = {"duration_s": duration_s, "csv_path": str(csv_path), "entry": entry,
            "dir": str(directory) if directory else None}
    return run_json_worker(
        [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG, json.dumps(spec)],
        duration_s + STARTUP_ALLOWANCE_S, label="S3 IMU recording",
        hang_hint="run xreal_shutdown.bat, replug the glasses and retry",
    )


def _fmt3(values) -> str:
    return "  ".join(f"{v:+.5f}" for v in values)


def _render(payload: dict, analysis: dict, mode: str) -> str:
    lines = ["XREAL IMU - Stage S3", "=" * 20, ""]
    if payload.get("fatal"):
        lines.append(f"FAILED: {payload['fatal']}")
        return "\n".join(lines)
    calls = payload.get("calls") or []
    if calls:
        lines.append("calls:")
        for c in calls:
            outcome = f"error {c['error']}" if c.get("error") else f"-> {c.get('result')}"
            mark = "ok " if c.get("succeeded") else "!! "
            lines.append(f"  {mark}{c['name']} {outcome}  [{c.get('elapsed_ms')} ms]")
        lines.append("")
    if payload.get("glasses_link"):
        lines.append(f"glasses link      {payload['glasses_link']}  (SDK log)")
    lines.append(f"samples           {payload.get('samples', 0)} in {payload.get('duration_s')} s"
                 f"   getter failures {payload.get('getter_failures', 0)}"
                 f"   destroy failures {payload.get('destroy_failures', 0)}")
    if payload.get("callback_exception"):
        lines.append(f"callback error    {payload['callback_exception']}")
    if payload.get("error"):
        lines.append(f"error             {payload['error']}")

    t = analysis.get("timing") or {}
    if t.get("rate_hz"):
        lines += ["", "timing (HMD clock):",
                  f"  rate            {t['rate_hz']} Hz   (median dt {t['dt_median_ms']} ms)",
                  f"  dt jitter       std {t['dt_std_ms']} ms   p99 {t['dt_p99_ms']} ms   max {t['dt_max_ms']} ms",
                  f"  gaps            {t['gaps']}  (~{t['missing_estimate']} samples missing)"
                  f"   non-monotonic {t['non_monotonic']}"]
    d = analysis.get("delivery") or {}
    if d:
        lines.append(f"  delivery        jitter std {d['jitter_std_ms']} ms   p99 {d['jitter_p99_ms']} ms"
                     f"   max {d['jitter_max_ms']} ms   drift {d['clock_drift_ppm']} ppm")
    r = analysis.get("rest")
    if r:
        lines += ["", "at rest (raw units):",
                  f"  gyro bias       {_fmt3(r['gyro_bias'])}",
                  f"  gyro std        {_fmt3(r['gyro_std'])}",
                  f"  accel mean      {_fmt3(r['accel_mean'])}   |a| = {r['accel_magnitude']}",
                  f"  accel unit      {r['accel_unit'] or 'UNKNOWN (|a| matches neither 9.81 nor 1)'}",
                  f"  gravity         reaction along {r['gravity_reaction_axis']}"
                  f"  (alignment {r['gravity_alignment']})"]
    tr = analysis.get("turn")
    if tr:
        if tr.get("error"):
            lines += ["", f"turn: {tr['error']}"]
        else:
            lines += ["", f"turn ({mode}, expected {tr['expected_deg']} deg):",
                      f"  integral        {_fmt3(tr['integral'])}",
                      f"  dominant        {tr['dominant_axis']}  ({tr['dominant_share']:.0%} of the rotation)",
                      f"  gyro unit       {tr['gyro_unit'] or 'UNDECIDED'}"
                      f"   (= {tr['degrees_if_rad_s']} deg if rad/s)"]
    lines += ["", f"verdict           {'OK' if payload.get('ok') else 'incomplete'}"]
    if payload.get("csv_path"):
        lines.append(f"samples at        {payload['csv_path']}")
    lines.append(f"logged to         {LOG_PATH}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == [_WORKER_FLAG]:
        return _run_worker(argv[1])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turn", choices=("yaw", "pitch", "roll"), default=None,
                        help="record one known 90 deg turn instead of rest")
    parser.add_argument("--degrees", type=float, default=90.0, help="size of the turn")
    parser.add_argument("--duration", type=float, default=None, help="seconds")
    parser.add_argument("--entry", choices=("loader", "api"), default="loader")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--analyse", type=Path, default=None,
                        help="analyse an existing CSV; no hardware")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    mode = "turn" if args.turn else "rest"
    label = args.turn or "rest"

    if args.analyse:
        payload = {"ok": True, "csv_path": str(args.analyse), "samples": None,
                   "duration_s": None}
    else:
        duration = args.duration or DEFAULT_DURATION_S[mode]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = LOG_DIR / f"imu_{stamp}_{label}.csv"
        if not args.json:
            print(f"recording {duration:.0f} s ({label}); library start-up takes 2-20 s "
                  f"before the first beep.\n")
        payload = run_isolated(duration, csv_path, args.entry, args.dir)

    analysis: dict = {}
    if payload.get("csv_path"):
        cols = imu_analysis.load_csv(Path(payload["csv_path"]))
        analysis = imu_analysis.analyse(cols, mode, args.degrees)
        if payload.get("samples") is None:
            payload["samples"] = len(cols.get("t_ns", []))
    payload["mode"], payload["turn"] = mode, args.turn
    if payload.get("native_output"):
        from xreal_native import glasses_link_from_log
        payload["glasses_link"] = glasses_link_from_log(payload["native_output"])
    payload["analysis"] = analysis
    ok = bool(payload.get("ok")) and "error" not in analysis
    payload["ok"] = ok
    if not args.analyse:
        append_jsonl(LOG_PATH, payload)

    print(json.dumps(payload, indent=2) if args.json else _render(payload, analysis, label))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
