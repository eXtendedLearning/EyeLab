#!/usr/bin/env python3
"""Stage S3b - licence gate, perception enumeration, and the head-pose recording.

Run these with the glasses on USB and nothing else holding the SDK
(``xreal_shutdown.bat`` first if unsure). None of them needs Nebula.

**1. Gate discriminator** - seconds, no motion, nothing started::

    python python/xreal_headpose_probe.py --gate

  One create in each of three families in a single session: ``imu`` (gated),
  ``grayscale_camera`` (gated) and ``NRHMDCreate`` (ungated control). Their
  three codes together say whether the licensed-feature list is empty or
  merely lacks ``imu``.

**2. Perception enumeration** - seconds, no motion, nothing but the group::

    python python/xreal_headpose_probe.py --perception

  The SDK listing which tracking modes these glasses offer, with the feature
  bits each one supports. This is the layer that produces the handle
  ``NRHeadTracking`` wants, so run it before the pose recording and after any
  failure in it.

**3. Head pose**::

    python python/xreal_headpose_probe.py                     # 20 s
    python python/xreal_headpose_probe.py --perception-id 2   # pin one

  Keep the glasses still for the first few seconds after the first beep so the
  analysis can tell a static pose from a moving one, then move them slowly
  through yaw, pitch and roll before the second beep. A pose that never changes
  is informative too: it means the fusion is not running.

  The beeps come from the PC's default audio device, not from the glasses, and
  the first one only fires **after** the first pose is acquired - so silence
  means either a muted output or a run that never got that far. The printed
  summary is the authority, not the sound.

Each pose run writes samples to ``python/.logs/headpose_<timestamp>.csv`` and a
summary to ``python/.logs/xreal_headpose_probe.jsonl``. ``--analyse CSV``
re-runs the maths on an existing recording without touching hardware.

Exit status: 0 if the level did what it set out to do, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import headpose_analysis
from native_isolation import append_jsonl, emit, run_json_worker
from xreal_result_codes import describe, describe_reason

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / ".logs"
LOG_PATH = LOG_DIR / "xreal_headpose_probe.jsonl"
_WORKER_FLAG = "--_worker"

DEFAULT_DURATION_S = 20.0

#: API bring-up is 2 s warm and up to ~16 s cold (S2); leave room for both.
STARTUP_ALLOWANCE_S = 90.0


def _beep(freq: int, ms: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:  # noqa: BLE001 - no sound is not a failure
        pass


def _run_worker(spec_json: str) -> int:
    sys.path.insert(0, str(HERE))
    import xreal_headpose

    spec = json.loads(spec_json)
    directory = Path(spec["dir"]) if spec.get("dir") else None
    level = spec["level"]
    if level == "gate":
        report = xreal_headpose.probe_gate(entry=spec["entry"], directory=directory)
    elif level == "perception":
        report = xreal_headpose.probe_perception(entry=spec["entry"],
                                                directory=directory)
    elif level == "sweep":
        report = xreal_headpose.sweep_perceptions(entry=spec["entry"],
                                                 directory=directory,
                                                 clock=spec["clock"],
                                                 offset_ns=spec["offset_ns"])
    else:
        report = xreal_headpose.record(
            duration_s=spec["duration_s"],
            csv_path=Path(spec["csv_path"]),
            entry=spec["entry"],
            directory=directory,
            rate_hz=spec["rate_hz"],
            time_candidates=tuple(spec["time_candidates"]),
            perception_id=spec.get("perception_id"),
            clock=spec["clock"],
            offset_ns=spec["offset_ns"],
            on_started=lambda: _beep(880, 250),
            on_finished=lambda: (_beep(660, 150), _beep(660, 150)),
        )
    emit(report.to_dict())
    return 0


def run_isolated(spec: dict, timeout_s: float, label: str) -> dict:
    return run_json_worker(
        [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG, json.dumps(spec)],
        timeout_s, label=label,
        hang_hint="run xreal_shutdown.bat, replug the glasses and retry",
    )


def _render_calls(payload: dict, lines: list) -> None:
    calls = payload.get("calls") or []
    if not calls:
        return
    lines.append("calls:")
    for c in calls:
        if c.get("error"):
            outcome = f"error {c['error']}"
        else:
            outcome = f"-> {describe(c.get('result'))}"
        mark = "ok " if c.get("succeeded") else "!! "
        lines.append(f"  {mark}{c['name']} {outcome}  [{c.get('elapsed_ms')} ms]")
    lines.append("")


def _render_perceptions(payload: dict, lines: list) -> None:
    items = payload.get("perceptions") or []
    if not items:
        return
    lines.append("")
    lines.append("perception group:")
    lines.append("  idx   id      features (mask / set bits)")
    for it in items:
        mask = it.get("features_mask")
        mask_txt = f"{mask:#018x}" if mask is not None else "n/a"
        bits = it.get("features_bits")
        lines.append(f"  {it['index']:<5} {str(it.get('perception_id')):<7} "
                     f"{mask_txt}  {bits if bits else ''}"
                     + (f"   {it['error']}" if it.get("error") else ""))


def _render_sweep(payload: dict) -> str:
    lines = ["XREAL perception sweep - Stage S3b", "=" * 34, ""]
    if payload.get("fatal"):
        return "\n".join(lines + [f"FAILED: {payload['fatal']}"])
    if payload.get("glasses_link"):
        lines.append(f"glasses link      {payload['glasses_link']}  (SDK log)")
    lines.append(f"clock             {payload.get('clock')}")
    lines += ["", "  id  start  clock_ns        features  acquire            "
                  "reason                  pose"]
    for r in payload.get("sweep") or []:
        mask = r.get("features_mask")
        lines.append(
            f"  {str(r.get('perception_id')):<3} "
            f"{'y' if r.get('started') else 'N':<6} "
            f"{str(r.get('clock_ns')):<15} "
            f"{(hex(mask) if mask is not None else '-'):<9} "
            f"{str(r.get('acquire') or '-'):<18} "
            f"{str(r.get('reason') or '-'):<23} "
            f"{'identity' if r.get('identity_pose') else ('moving' if r.get('identity_pose') is False else '-')}")
    started = [r for r in payload.get("sweep") or [] if r.get("started")]
    tracking = [r for r in started if r.get("identity_pose") is False]
    lines += ["", f"summary           {len(started)} of {len(payload.get('sweep') or [])} "
                  f"perceptions started; {len(tracking)} returned a non-identity pose"]
    if payload.get("error"):
        lines.append(f"error             {payload['error']}")
    lines += ["", f"verdict           {'OK' if payload.get('ok') else 'incomplete'}",
              f"logged to         {LOG_PATH}"]
    return "\n".join(lines)


def _render_gate(payload: dict) -> str:
    lines = ["XREAL licence gate - Stage S3b", "=" * 30, ""]
    if payload.get("fatal"):
        return "\n".join(lines + [f"FAILED: {payload['fatal']}"])
    _render_calls(payload, lines)
    if payload.get("glasses_link"):
        lines.append(f"glasses link      {payload['glasses_link']}  (SDK log)")
    lines.append("")
    lines.append("feature            gated on            result")
    for label, gated in (("imu", '"imu"'), ("grayscale_camera", '"grayscale_camera"'),
                         ("hmd", "not gated (control)")):
        code = (payload.get("codes") or {}).get(label)
        lines.append(f"  {label:<17}{gated:<20}{describe(code)}")
    lines += ["", f"verdict           {payload.get('verdict') or 'n/a'}"]
    if payload.get("error"):
        lines.append(f"error             {payload['error']}")
    lines.append(f"logged to         {LOG_PATH}")
    return "\n".join(lines)


def _render_perception(payload: dict) -> str:
    lines = ["XREAL perception group - Stage S3b", "=" * 34, ""]
    if payload.get("fatal"):
        return "\n".join(lines + [f"FAILED: {payload['fatal']}"])
    _render_calls(payload, lines)
    if payload.get("glasses_link"):
        lines.append(f"glasses link      {payload['glasses_link']}  (SDK log)")
    lines.append(f"group handle      {payload.get('group_handle')}")
    _render_perceptions(payload, lines)
    if payload.get("error"):
        lines.append(f"error             {payload['error']}")
    lines += ["", f"verdict           {'OK' if payload.get('ok') else 'incomplete'}",
              f"logged to         {LOG_PATH}"]
    return "\n".join(lines)


def _render_pose(payload: dict, analysis: dict) -> str:
    lines = ["XREAL head pose - Stage S3b", "=" * 27, ""]
    if payload.get("fatal"):
        return "\n".join(lines + [f"FAILED: {payload['fatal']}"])
    _render_calls(payload, lines)
    if payload.get("glasses_link"):
        lines.append(f"glasses link      {payload['glasses_link']}  (SDK log)")
    if payload.get("perception_id") is not None:
        lines.append(f"perception        id {payload['perception_id']}"
                     f"   handle {payload.get('perception_handle')}")
    _render_perceptions(payload, lines)
    if payload.get("perception_clock_ns") is not None:
        mask = payload.get("perception_features_mask")
        lines.append(f"perception clock  {payload['perception_clock_ns']} ns"
                     f"   features {hex(mask) if mask is not None else 'n/a'}")
    if payload.get("tracking_handle") is not None:
        lines.append(f"tracking handle   {payload['tracking_handle']}")
    if payload.get("clock"):
        lines.append(f"clock source      {payload['clock']}"
                     f"   offset {payload.get('offset_ns', 0)} ns")
    if payload.get("acquire_result") is not None:
        lines.append(f"time argument     {payload.get('time_argument')}"
                     f"   acquire {payload['acquire_result']}")
    lines.append(f"samples           {payload.get('samples', 0)} in "
                 f"{payload.get('duration_s')} s"
                 f"   acquire failures {payload.get('acquire_failures', 0)}"
                 f"   getter {payload.get('getter_failures', 0)}"
                 f"   destroy {payload.get('destroy_failures', 0)}")
    if payload.get("error"):
        lines.append(f"error             {payload['error']}")

    t = (analysis.get("timing") or {})
    if t.get("hmd"):
        h = t["hmd"]
        lines += ["", "timing (device clock):",
                  f"  rate            {h['rate_hz']} Hz   (median dt {h['dt_median_ms']} ms)",
                  f"  jitter          std {h['dt_std_ms']} ms   max {h['dt_max_ms']} ms"
                  f"   non-monotonic {h['non_monotonic']}"]
    if t.get("distinct_device_timestamps") is not None:
        lines.append(f"  distinct stamps {t['distinct_device_timestamps']}"
                     f"   repeated {t['repeated_device_timestamps']}")
    if payload.get("tracking_reason_names"):
        lines += ["", f"tracking reasons  {payload['tracking_reason_names']}"]
    elif analysis.get("tracking_reasons"):
        lines += ["", "tracking reasons  " + ", ".join(
            f"{describe_reason(int(k))} x{v}"
            for k, v in analysis["tracking_reasons"].items())]
    pose = analysis.get("pose") or {}
    if pose:
        lines += ["", f"pose payload      {pose['verdict']}"]
        for cand in pose.get("candidates", []):
            rot = cand["rotation_block"]
            lines.append(f"  {cand['layout']}")
            lines.append(f"    fixed slots max |x| {cand['fixed_slots_max_abs']}"
                         f"   last slot mean {cand['last_slot_mean']}")
            lines.append(f"    rotation rows {rot['row_norms']}"
                         f"  offdiag {rot['max_offdiagonal']}"
                         f"  det {rot['determinant']}"
                         f"  -> {'rotation' if rot['is_rotation'] else 'not a rotation'}")
            lines.append(f"    translation first {cand['translation_first']}"
                         f"   std {cand['translation_std_m']}"
                         f"   range {cand['translation_range_m']}")
        q = pose.get("quaternion") or {}
        lines.append(f"  first four floats norm {q.get('first_four_norm_mean')}"
                     f" +/- {q.get('first_four_norm_std')}")
    if analysis.get("pose_is_static"):
        lines.append("  NOTE: every slot is constant across the recording - "
                     "the pose is not being updated")
    lines += ["", f"verdict           {'OK' if payload.get('ok') else 'incomplete'}"]
    if payload.get("csv_path"):
        lines.append(f"samples at        {payload['csv_path']}")
    if payload.get("logged"):
        lines.append(f"logged to         {LOG_PATH}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == [_WORKER_FLAG]:
        return _run_worker(argv[1])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="store_true",
                        help="licence-gate discriminator; nothing started")
    parser.add_argument("--perception", action="store_true",
                        help="enumerate the perception group and stop")
    parser.add_argument("--sweep", action="store_true",
                        help="one pose through every perception in the group")
    parser.add_argument("--clock", choices=("hmd", "zero"), default="hmd",
                        help="where hmd_time_nanos comes from (default: the "
                             "perception's own clock)")
    parser.add_argument("--offset-ms", type=float, default=0.0,
                        help="added to the device clock, e.g. to predict ahead")
    parser.add_argument("--perception-id", type=int, default=None,
                        help="pin one perception instead of taking the first that starts")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--rate", type=float, default=100.0, help="poll target, Hz")
    parser.add_argument("--time-ns", default="0",
                        help="comma-separated hmd_time_nanos values to try in order")
    parser.add_argument("--entry", choices=("loader", "api"), default="loader")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--analyse", type=Path, default=None,
                        help="analyse an existing CSV; no hardware")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.analyse:
        cols = headpose_analysis.load_csv(args.analyse)
        analysis = headpose_analysis.analyse(cols)
        payload = {"ok": "error" not in analysis, "csv_path": str(args.analyse),
                   "samples": analysis.get("samples"), "duration_s": None}
        print(json.dumps({**payload, "analysis": analysis}, indent=2)
              if args.json else _render_pose(payload, analysis))
        return 0 if payload["ok"] else 1

    analysis: dict = {}
    offset_ns = int(args.offset_ms * 1e6)
    if args.gate:
        spec = {"level": "gate", "entry": args.entry,
                "dir": str(args.dir) if args.dir else None}
        payload = run_isolated(spec, STARTUP_ALLOWANCE_S, "S3b gate discriminator")
    elif args.perception:
        spec = {"level": "perception", "entry": args.entry,
                "dir": str(args.dir) if args.dir else None}
        payload = run_isolated(spec, STARTUP_ALLOWANCE_S, "S3b perception enumeration")
    elif args.sweep:
        spec = {"level": "sweep", "entry": args.entry,
                "dir": str(args.dir) if args.dir else None,
                "clock": args.clock, "offset_ns": offset_ns}
        payload = run_isolated(spec, STARTUP_ALLOWANCE_S + 60,
                               "S3b perception sweep")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = LOG_DIR / f"headpose_{stamp}.csv"
        candidates = [int(v) for v in str(args.time_ns).split(",") if v.strip()]
        spec = {"level": "pose", "entry": args.entry,
                "dir": str(args.dir) if args.dir else None,
                "duration_s": args.duration, "csv_path": str(csv_path),
                "rate_hz": args.rate, "time_candidates": candidates or [0],
                "perception_id": args.perception_id,
                "clock": args.clock, "offset_ns": offset_ns}
        if not args.json:
            print(f"recording {args.duration:.0f} s of head pose; library start-up "
                  f"takes 2-20 s before the first beep.\n")
        payload = run_isolated(spec, args.duration + STARTUP_ALLOWANCE_S,
                               "S3b head-pose recording")
        if payload.get("csv_path"):
            analysis = headpose_analysis.analyse(
                headpose_analysis.load_csv(Path(payload["csv_path"])))

    if payload.get("native_output"):
        from xreal_native import glasses_link_from_log
        payload["glasses_link"] = glasses_link_from_log(payload["native_output"])
    payload["analysis"] = analysis
    payload["logged"] = True
    append_jsonl(LOG_PATH, payload)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.gate:
        print(_render_gate(payload))
    elif args.perception:
        print(_render_perception(payload))
    elif args.sweep:
        print(_render_sweep(payload))
    else:
        print(_render_pose(payload, analysis))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
