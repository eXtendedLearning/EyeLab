"""Beam Pro probe runs (T6.0 gate BS, T6.1 gate B1). ``python -m xreal_gates bp RUN [RUN ...]``.

A run is a folder pulled from ``/sdcard/Android/data/it.univpm.eyelab/files/BeamProRuns/`` holding
``run.json`` and, by probe:

- ``inventory``: ``run.json`` carries the Camera2, sensor and display inventory; ``imu.csv`` the IMU
  recording (``type,event_ns,boot_ns,mono_ns,v0,v1,v2,accuracy``).
- ``chain``: ``eye_frames.csv`` (one row per kept Eye frame: ``cam_ts`` next to every clock at arrival).
- ``timecode``: as ``chain`` plus ``draws.csv`` (the marker drawn on the Beam Pro screen) and
  ``frames/*.pgm`` (saved Eye Y planes), decoded here.

Nothing assumes the clock domain of ``cam_ts``: :func:`clock_domain` tries every clock and unit and keeps
the one that gives a positive, stable frame age.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

CLOCKS = ("boot_ns", "mono_ns", "real_ns", "hmd_ns")
UNITS = {"ns": 1.0, "us": 1e3, "ms": 1e6}
SENSOR_NAMES = {1: "accelerometer", 4: "gyroscope", 35: "accelerometer_uncal", 16: "gyroscope_uncal"}


# ----------------------------------------------------------------------------- loading
def load_json(run: Path) -> dict:
    p = Path(run) / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(v: str) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ----------------------------------------------------------------------------- clock domain
@dataclass
class Domain:
    clock: str
    unit: str
    median_ms: float
    iqr_ms: float
    slope_ms_per_min: float
    n: int

    def line(self) -> str:
        return (f"cam_ts on {self.clock} in {self.unit}: age median {self.median_ms:.1f} ms, IQR {self.iqr_ms:.1f} ms, "
                f"drift {self.slope_ms_per_min:+.2f} ms/min over {self.n} frames")


def _slope(ts: Sequence[float], ys: Sequence[float]) -> float:
    n = len(ts)
    if n < 3:
        return 0.0
    mt, my = sum(ts) / n, sum(ys) / n
    den = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den if den > 0 else 0.0


def clock_domain(frames: List[Dict[str, str]]) -> Tuple[Optional[Domain], List[Domain]]:
    """Every (clock, unit) candidate, and the plausible one: age between 0 and 5 s, smallest IQR."""
    cands: List[Domain] = []
    for clock in CLOCKS:
        for unit, f in UNITS.items():
            ages, times = [], []
            for r in frames:
                c, t = _num(r.get(clock, "")), _num(r.get("cam_ts", ""))
                if c is None or t is None or c <= 0 or t <= 0:
                    continue
                ages.append((c - t * f) / 1e6)
                times.append(c / 6e10)  # minutes
            if len(ages) < 3:
                continue
            cands.append(Domain(clock, unit, quantile(ages, 0.5), quantile(ages, 0.75) - quantile(ages, 0.25),
                                _slope(times, ages), len(ages)))
    ok = [d for d in cands if 0.0 < d.median_ms < 5000.0 and d.iqr_ms < 1000.0]
    best = min(ok, key=lambda d: (d.iqr_ms, d.median_ms)) if ok else None
    return best, cands


# ----------------------------------------------------------------------------- chain report
def chain_report(run: Path) -> List[str]:
    frames = load_csv(Path(run) / "eye_frames.csv")
    meta = load_json(run)
    out = [f"== {Path(run).name}  probe {meta.get('probe', '?')}  app {meta.get('app_version', '?')}  "
           f"device {meta.get('device_model', '?')}"]
    if not frames:
        out.append("   no eye_frames.csv rows: did StartRGBCameraDataCapture work? "
                   f"(eye_start_capture={meta.get('eye_start_capture', '?')})")
        return out
    res = {(r.get("res_w"), r.get("res_h"), r.get("plane_w"), r.get("plane_h")) for r in frames}
    out.append("   resolution / Y plane: " + "; ".join(f"{a}x{b} plane {c}x{d}" for a, b, c, d in sorted(res)))
    boots = [_num(r["boot_ns"]) for r in frames if _num(r.get("boot_ns", "")) is not None]
    dur = (boots[-1] - boots[0]) / 1e9 if len(boots) > 1 else float("nan")
    fps = (len(frames) - 1) / dur if dur and dur > 0 else float("nan")
    drained = [int(r.get("drained") or 0) for r in frames]
    out.append(f"   {len(frames)} kept frames over {dur:.1f} s: {fps:.1f} fps kept; drained {sum(drained)} "
               f"(frames with drained>0: {sum(1 for d in drained if d > 0)})")
    cts = [_num(r["cam_ts"]) for r in frames]
    dts = sorted(b - a for a, b in zip(cts, cts[1:]) if a is not None and b is not None and b > a)
    if dts:
        out.append(f"   cam_ts step median {quantile(dts, 0.5):.0f} raw units (p5 {quantile(dts, 0.05):.0f}, "
                   f"p95 {quantile(dts, 0.95):.0f})")
    hmd_ok = sum(1 for r in frames if (_num(r.get("hmd_ns", "")) or 0) > 0)
    out.append(f"   GetHMDTimeNanos non-zero in {hmd_ok}/{len(frames)} rows")
    best, cands = clock_domain(frames)
    if best is None:
        out.append("   cam_ts matches no clock with a plausible age; candidates (median ms):")
        out.extend(f"      {d.clock}/{d.unit}: {d.median_ms:.3g}" for d in cands if d.unit == "ns")
    else:
        out.append("   " + best.line())
        others = [d for d in cands if d is not best and 0.0 < d.median_ms < 5000.0]
        for d in others:
            out.append(f"      also plausible: {d.line()}")
        out.append("   decoder queue: " + ("suspected (age grows)" if best.slope_ms_per_min > 5 else "not seen")
                   + f", drift {best.slope_ms_per_min:+.2f} ms/min")
    return out


# ----------------------------------------------------------------------------- timecode
def read_pgm(path: Path):
    import numpy as np
    data = Path(path).read_bytes()
    parts, i = [], 0
    while len(parts) < 4:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        parts.append(data[i:j])
        i = j
    i += 1
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(data[i:i + w * h], dtype=np.uint8).reshape(h, w)


def decode_id(gray) -> Tuple[Optional[int], str]:
    """Timecode marker id in a Y plane, trying rows as delivered and flipped (the PC stream was bottom-up)."""
    import cv2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
    for name, img in (("as_is", gray), ("flipped", gray[::-1].copy())):
        _, ids, _ = det.detectMarkers(img)
        if ids is not None and len(ids):
            return int(ids.ravel()[0]), name
    return None, ""


def match_draw(draws: List[Tuple[int, int, float]], marker_id: int, arrival_ns: float,
               window_s: float = 5.0) -> Optional[Tuple[int, float]]:
    """The latest draw of ``marker_id`` at or before arrival, within ``window_s``: (counter, set time ns)."""
    best = None
    for counter, did, t in draws:
        if did != marker_id or t > arrival_ns or arrival_ns - t > window_s * 1e9:
            continue
        if best is None or t > best[1]:
            best = (counter, t)
    return best


def timecode_report(run: Path, clock: str = "boot_ns") -> List[str]:
    run = Path(run)
    frames = load_csv(run / "eye_frames.csv")
    draws_rows = load_csv(run / "draws.csv")
    meta = load_json(run)
    out = [f"   timecode host: {meta.get('timecode_host', '?')}, draws {len(draws_rows)}"]
    if not draws_rows:
        return out + ["   no draws.csv"]
    set_col = "set_" + clock
    draws = [(int(r["counter"]), int(r["id"]), float(r[set_col])) for r in draws_rows if _num(r.get(set_col, ""))]
    dd = sorted(b[2] - a[2] for a, b in zip(draws, draws[1:]))
    if dd:
        out.append(f"   draw period median {quantile(dd, 0.5) / 1e6:.2f} ms")
    lat, cam_minus_draw, decoded, orient = [], [], 0, {}
    best, _ = clock_domain(frames)
    for r in frames:
        saved = [s for s in (r.get("saved") or "").split("|") if s.startswith("frames/")]
        if not saved or not (run / saved[0]).exists():
            continue
        mid, how = decode_id(read_pgm(run / saved[0]))
        if mid is None:
            continue
        decoded += 1
        orient[how] = orient.get(how, 0) + 1
        arr = float(r[clock])
        m = match_draw(draws, mid, arr)
        if m is None:
            continue
        lat.append((arr - m[1]) / 1e6)
        if best is not None and best.clock == clock:
            cam_minus_draw.append((float(r["cam_ts"]) * UNITS[best.unit] - m[1]) / 1e6)
    sampled = sum(1 for r in frames if "frames/" in (r.get("saved") or ""))
    out.append(f"   decoded {decoded}/{sampled} saved frames; orientation {orient}")
    if lat:
        out.append(f"   draw -> arrival (upper bound on exposure -> arrival): median {quantile(lat, 0.5):.1f} ms, "
                   f"p10 {quantile(lat, 0.1):.1f}, p90 {quantile(lat, 0.9):.1f}, n {len(lat)}")
    if cam_minus_draw:
        out.append(f"   draw -> cam_ts: median {quantile(cam_minus_draw, 0.5):.1f} ms, min {min(cam_minus_draw):.1f} "
                   "(negative = cam_ts is earlier than the draw it shows: not exposure time, or a clock offset)")
    return out


# ----------------------------------------------------------------------------- inventory
def imu_report(run: Path) -> List[str]:
    rows = load_csv(Path(run) / "imu.csv")
    if not rows:
        return ["   no imu.csv"]
    by: Dict[int, List[Tuple[float, float]]] = {}
    for r in rows:
        t, b = _num(r.get("event_ns", "")), _num(r.get("boot_ns", ""))
        if t is None or b is None:
            continue
        by.setdefault(int(r["type"]), []).append((t, b))
    out = []
    for typ, evs in sorted(by.items()):
        evs.sort()
        dt = [b[0] - a[0] for a, b in zip(evs, evs[1:]) if b[0] > a[0]]
        lag = [(b - t) / 1e6 for t, b in evs]
        rate = 1e9 / quantile(dt, 0.5) if dt else float("nan")
        out.append(f"   IMU {SENSOR_NAMES.get(typ, typ)}: {len(evs)} events, {rate:.1f} Hz (period p95/p5 "
                   f"{quantile(dt, 0.95) / max(quantile(dt, 0.05), 1):.2f}), delivery after event median "
                   f"{quantile(lag, 0.5):.2f} ms, p95 {quantile(lag, 0.95):.2f} ms")
    return out


def inventory_report(run: Path) -> List[str]:
    meta = load_json(run)
    out = [f"== {Path(run).name}  inventory  app {meta.get('app_version', '?')}"]
    dev = meta.get("device") or {}
    if dev:
        out.append(f"   {dev.get('manufacturer')} {dev.get('model')}, Android {dev.get('android_release')} "
                   f"(SDK {dev.get('sdk_int')}), build {dev.get('build_display')}, SoC {dev.get('soc_model')}")
        for p in dev.get("packages", []):
            out.append(f"   package {p.get('package')} {p.get('version_name')} ({p.get('version_code')})")
    cams = meta.get("cameras") or {}
    out.append(f"   camera id list {cams.get('camera_id_list')}; concurrent sets {cams.get('concurrent_camera_ids')}")
    for c in cams.get("cameras", []):
        logical = 18 in (c.get("capabilities") or [])  # REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA
        out.append(f"   cam {c.get('id')} {c.get('facing')}{' LOGICAL' if logical else ''}"
                   f"{'' if c.get('in_id_list') else ' (physical only)'}: physical {c.get('physical_ids')}, "
                   f"sync {c.get('sensor_sync_type')}, ts {c.get('timestamp_source')}, f {c.get('focal_lengths_mm')} mm, "
                   f"K {c.get('lens_intrinsic_calibration')}, t {c.get('lens_pose_translation')} (ref "
                   f"{c.get('lens_pose_reference')}), array {c.get('pixel_array')}")
    for d in (meta.get("displays") or {}).get("displays", []):
        out.append(f"   display {d.get('id')} '{d.get('name')}' {d.get('mode')}, deadline "
                   f"{(d.get('presentation_deadline_ns') or 0) / 1e6:.1f} ms")
    sensors = (meta.get("sensors") or {}).get("sensors", [])
    imu = [s for s in sensors if s.get("type") in (1, 4, 35, 16)]
    for s in imu:
        out.append(f"   sensor {s.get('string_type')}: {s.get('name')} ({s.get('vendor')}), min delay "
                   f"{s.get('min_delay_us')} us, fifo {s.get('fifo_max')}, direct level {s.get('direct_report_max_level')}")
    out.append(f"   {len(sensors)} sensors listed in all")
    if meta.get("inventory_error"):
        out.append(f"   ERROR {meta['inventory_error']}")
    return out + imu_report(run)


def report(run: Path) -> List[str]:
    if not (Path(run) / "run.json").exists():
        return [f"== {Path(run).name}: no run.json here; give a BeamProRuns/<stamp>_<probe> folder"]
    probe = load_json(run).get("probe", "")
    if probe == "inventory":
        return inventory_report(run)
    lines = chain_report(run)
    if probe == "timecode":
        lines += timecode_report(run)
    return lines
