"""Beam Pro probe runs (T6.0 gate BS, T6.1 gate B1, T6.2 gates B2, B1t, BS-2). ``python -m xreal_gates bp RUN ...``.

A run is a folder pulled from ``/sdcard/Android/data/it.univpm.eyelab/files/BeamProRuns/`` holding
``run.json`` and, by probe:

- ``inventory``: ``run.json`` carries the Camera2, sensor and display inventory; ``imu.csv`` the IMU
  recording (``type,event_ns,boot_ns,mono_ns,v0,v1,v2,accuracy``).
- ``chain``: ``eye_frames.csv`` (one row per kept Eye frame: ``cam_ts`` next to every clock at arrival).
- ``timecode``: as ``chain`` plus ``draws.csv`` (the marker drawn on the Beam Pro screen) and
  ``frames/*.pgm`` (saved Eye Y planes), decoded here. From 0.4.002 each id is held 100 ms and ``draws.csv``
  has one row per id change; the latency also uses only the first Eye frame that shows each id.
- ``head``, ``head_eye``, ``head_eye_3dof``: ``head.csv`` (XR head pose and tracking state every Update),
  ``events.csv`` (mode switches, tracking changes, jumps); with the Eye also ``eye_frames.csv`` and
  ``eye_pose.csv`` (native head pose at ``cam_ts`` and at arrival, 4x4 floats in memory order).
- ``stereo``: ``stereo.csv`` (per capture of the hidden logical rear camera: both physical sensor timestamps).

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
    best, _ = clock_domain(frames)
    if best is not None and "set_" + best.clock in draws_rows[0]:
        clock = best.clock   # draws and arrivals on the clock cam_ts is on, so draw -> cam_ts is meaningful
    set_col = "set_" + clock
    draws = [(int(r["counter"]), int(r["id"]), float(r[set_col])) for r in draws_rows if _num(r.get(set_col, ""))]
    dd = sorted(b[2] - a[2] for a, b in zip(draws, draws[1:]))
    if dd:
        out.append(f"   draw period median {quantile(dd, 0.5) / 1e6:.2f} ms")
    lat, cam_minus_draw, decoded, orient = [], [], 0, {}
    first: Dict[int, Tuple[float, float]] = {}   # draw counter -> (arrival - draw, cam_ts - draw) of its first frame
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
        cmd = float("nan")
        if best is not None and best.clock == clock:
            cmd = (float(r["cam_ts"]) * UNITS[best.unit] - m[1]) / 1e6
            cam_minus_draw.append(cmd)
        if m[0] not in first:
            first[m[0]] = ((arr - m[1]) / 1e6, cmd)
    sampled = sum(1 for r in frames if "frames/" in (r.get("saved") or ""))
    out.append(f"   decoded {decoded}/{sampled} saved frames; orientation {orient}")
    if lat:
        out.append(f"   draw -> arrival (upper bound on exposure -> arrival): median {quantile(lat, 0.5):.1f} ms, "
                   f"p10 {quantile(lat, 0.1):.1f}, p90 {quantile(lat, 0.9):.1f}, n {len(lat)}")
    if cam_minus_draw:
        out.append(f"   draw -> cam_ts: median {quantile(cam_minus_draw, 0.5):.1f} ms, min {min(cam_minus_draw):.1f} "
                   "(negative = cam_ts is earlier than the draw it shows: not exposure time, or a clock offset)")
    if first:
        fl = [a for a, _ in first.values()]
        fc = [c for _, c in first.values() if not math.isnan(c)]
        line = (f"   first frame of each id (n {len(fl)}): draw -> arrival median {quantile(fl, 0.5):.1f} ms, "
                f"p10 {quantile(fl, 0.1):.1f}, p90 {quantile(fl, 0.9):.1f}")
        if fc:
            line += (f"; draw -> cam_ts median {quantile(fc, 0.5):.1f} ms, p10 {quantile(fc, 0.1):.1f}, "
                     f"p90 {quantile(fc, 0.9):.1f}")
        out.append(line)
    return out


# ----------------------------------------------------------------------------- head pose (T6.2, gate B2)
def rot_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle between the rotation parts of two 4x4 matrices stored in memory order. trace(A^T B) is the same for
    row- and column-major storage, so the storage convention does not matter."""
    tr = sum(a[r * 4 + c] * b[r * 4 + c] for r in range(3) for c in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def translation(m: Sequence[float]) -> Tuple[float, float, float]:
    """Column-major (translation in 12..14) when m[3], m[7], m[11] are zero, else row-major (3, 7, 11)."""
    if abs(m[3]) + abs(m[7]) + abs(m[11]) < 1e-6:
        return m[12], m[13], m[14]
    return m[3], m[7], m[11]


def quat_angle_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def still_spread(rows: List[Dict[str, str]], window_s: float = 5.0) -> str:
    """Position spread in the window where the head rotated least: how quiet the pose is at rest."""
    pts = [(float(r["mono_ns"]) / 1e9, (float(r["px"]), float(r["py"]), float(r["pz"])),
            (float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))) for r in rows if r.get("tracked") == "1"]
    best = None
    j = 0
    for i in range(len(pts)):
        while j < len(pts) and pts[j][0] - pts[i][0] < window_s:
            j += 1
        if j - i < 20 or pts[j - 1][0] - pts[i][0] < 0.9 * window_s:
            continue
        rot = max(quat_angle_deg(pts[i][2], p[2]) for p in pts[i:j])
        if best is None or rot < best[0]:
            best = (rot, i, j)
    if best is None:
        return ""
    rot, i, j = best
    seg = [p[1] for p in pts[i:j]]
    c = [sum(p[k] for p in seg) / len(seg) for k in range(3)]
    d = [math.dist(p, c) * 1000 for p in seg]
    return (f"stillest {window_s:.0f} s window: rotation within {rot:.2f} deg, position spread median "
            f"{quantile(d, 0.5):.1f} mm, max {max(d):.1f} mm")


def head_report(run: Path) -> List[str]:
    run = Path(run)
    rows = load_csv(run / "head.csv")
    meta = load_json(run)
    out = [f"== {run.name}  probe {meta.get('probe', '?')}  app {meta.get('app_version', '?')}  mode requested "
           f"{meta.get('head_mode_requested', '?')}, at start {meta.get('head_mode_at_start', '?')}, "
           f"at end {meta.get('head_mode_at_end', '?')}"]
    for e in load_csv(run / "events.csv"):
        if e.get("event") in ("cue", "cue_speech", "jump"):
            continue   # cue steps: guided.phase_lines; jumps: listed with their phase (loop) or counted below
        out.append(f"   event {e.get('event')}: {e.get('detail')}")
    jumps = [e.get("detail") for e in load_csv(run / "events.csv") if e.get("event") == "jump"]
    if jumps:
        out.append(f"   {len(jumps)} position jumps > 10 cm in one frame: {', '.join(jumps)}")
    if not rows:
        return out + ["   no head.csv rows"]
    t = [float(r["mono_ns"]) / 1e9 for r in rows]
    dur = t[-1] - t[0]
    tracked = [r.get("tracked") == "1" for r in rows]
    rate = (len(rows) - 1) / dur if dur > 0 else float("nan")
    out.append(f"   {len(rows)} head rows over {dur:.1f} s ({rate:.1f} Hz); tracked {100.0 * sum(tracked) / len(rows):.1f} %; "
               f"tracking types {sorted({r.get('tracking_type', '') for r in rows})}")
    first_tracked = next((i for i, k in enumerate(tracked) if k), None)
    if first_tracked is not None:
        out.append(f"   first tracked row {t[first_tracked] - t[0]:.2f} s after the first row")
    pos = [(float(r["px"]), float(r["py"]), float(r["pz"])) for r, k in zip(rows, tracked) if k]
    if len(pos) > 1:
        path = sum(math.dist(a, b) for a, b in zip(pos, pos[1:]))
        span = [max(p[i] for p in pos) - min(p[i] for p in pos) for i in range(3)]
        steps = [math.dist(a, b) * 1000 for a, b in zip(pos, pos[1:])]
        out.append(f"   position: path {path:.2f} m, span x {span[0]:.2f} y {span[1]:.2f} z {span[2]:.2f} m, "
                   f"step p99 {quantile(steps, 0.99):.1f} mm, max {max(steps):.0f} mm")
    still = still_spread(rows)
    if still:
        out.append("   " + still)
    return out


def eye_pose_report(run: Path) -> List[str]:
    rows = load_csv(Path(run) / "eye_pose.csv")
    if not rows:
        return []
    ok = [r for r in rows if r.get("cam_ok") == "1" and r.get("arr_ok") == "1"]
    out = [f"   head pose at cam_ts and at arrival found for {len(ok)}/{len(rows)} frames "
           f"(at cam_ts {sum(r.get('cam_ok') == '1' for r in rows)}, at arrival {sum(r.get('arr_ok') == '1' for r in rows)})"]
    if not ok:
        return out
    ang, trans = [], []
    for r in ok:
        c = [float(r[f"c{i}"]) for i in range(16)]
        a = [float(r[f"a{i}"]) for i in range(16)]
        ang.append(rot_angle_deg(c, a))
        trans.append(math.dist(translation(c), translation(a)) * 1000)
    out.append(f"   head motion inside the frame age (pose at cam_ts vs at arrival): rotation median "
               f"{quantile(ang, 0.5):.2f} deg, p90 {quantile(ang, 0.9):.2f}, max {max(ang):.2f}; translation median "
               f"{quantile(trans, 0.5):.1f} mm, p90 {quantile(trans, 0.9):.1f}")
    c0 = [float(ok[0][f"c{i}"]) for i in range(16)]
    layout = "column-major (translation in 12..14)" if abs(c0[3]) + abs(c0[7]) + abs(c0[11]) < 1e-6 \
        else "row-major (translation in 3, 7, 11)"
    out.append("   matrix layout guess: " + layout)
    return out


# ----------------------------------------------------------------------------- stereo (T6.2, gate BS-2)
def stereo_report(run: Path) -> List[str]:
    run = Path(run)
    meta = load_json(run)
    out = [f"== {run.name}  {meta.get('probe', 'stereo')}  app {meta.get('app_version', '?')}",
           f"   start: {meta.get('stereo_start', '?')}", f"   stop: {meta.get('stereo_stop', '?')}"]
    rows = load_csv(run / "stereo.csv")
    if not rows:
        return out + ["   no captures"]
    pairs = [(float(r["ts_a"]), float(r["ts_b"])) for r in rows
             if _num(r.get("ts_a", "")) is not None and _num(r.get("ts_b", "")) is not None]
    out.append(f"   {len(rows)} captures, {len(pairs)} with both physical timestamps")
    if len(pairs) > 1:
        off = [(a - b) / 1e6 for a, b in pairs]
        dur = (pairs[-1][0] - pairs[0][0]) / 1e9
        rate = (len(pairs) - 1) / dur if dur > 0 else float("nan")
        out.append(f"   rate {rate:.1f} Hz; timestamp offset a - b median {quantile(off, 0.5):.3f} ms, "
                   f"p5 {quantile(off, 0.05):.3f}, p95 {quantile(off, 0.95):.3f}, max |offset| {max(abs(o) for o in off):.3f}")
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
        logical = 11 in (c.get("capabilities") or [])  # REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA
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
    if probe in ("stereo", "pan"):
        lines = stereo_report(run)
        if probe == "pan":
            from . import guided
            lines += guided.phase_lines(run) + guided.pan_report(run) + imu_report(run)
        return lines
    if probe in ("loop", "sync"):
        from . import guided
        lines = head_report(run) + guided.phase_lines(run)
        lines += guided.loop_report(run) if probe == "loop" else guided.sync_report(run)
        if (Path(run) / "eye_frames.csv").exists():
            lines += chain_report(run)[1:] + eye_pose_report(run)
        return lines
    if probe.startswith("head"):
        lines = head_report(run)
        if (Path(run) / "eye_frames.csv").exists():
            lines += chain_report(run)[1:] + eye_pose_report(run)
        return lines
    lines = chain_report(run)
    if probe == "timecode":
        lines += timecode_report(run)
    return lines
