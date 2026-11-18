"""T6.3 guided runs on the Beam Pro: cue phases, loop closure, motion sync, stereo pan.

The probes log every cue step in ``events.csv`` (``cue,<step id>`` with all clocks), so a run can be cut into
the phases the operator was told to do. Three analyses use that:

* ``loop``: still jitter per still phase, the turn angles read against 90 deg, the loop closure (where the head
  stands at the end against the start, on the same floor mark) and every position jump with its phase.
* ``sync``: when the Eye exposes, relative to ``cam_ts``. The image shift between consecutive Eye frames (phase
  correlation, 320x180) is fitted against the head rotation between the two frame times, taken from the native
  head pose logged every 5 ms (``pose_grid.csv``), for trial offsets ``delta``; the best fit gives exposure =
  ``cam_ts + delta``. Needs no display, unlike the timecode.
* ``pan``: the same against the Beam Pro gyroscope, for each of the two rear sensors of the logical camera; the
  difference of the two offsets is the exposure offset between the sensors.

Units: time in ns (input) and ms (reports); rotation in rad; shifts in thumbnail pixels.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .beampro import load_csv, load_json, quantile, read_pgm

DELTAS_MS = [d * 0.5 for d in range(-400, 401)]   # -200 .. +200 ms in 0.5 ms steps


# ----------------------------------------------------------------------------- phases
def cue_phases(run: Path, clock: str = "mono_ns") -> List[Tuple[str, float, float]]:
    """(step id, start, end) in ``clock`` ns from the ``cue`` events; the last step ends at ``cue,end``."""
    cues = [(e["detail"], float(e[clock])) for e in load_csv(Path(run) / "events.csv")
            if e.get("event") == "cue" and e.get(clock)]
    return [(name, t0, t1) for (name, t0), (_, t1) in zip(cues, cues[1:]) if name != "end"]


def phase_of(phases: Sequence[Tuple[str, float, float]], t: float) -> str:
    for name, t0, t1 in phases:
        if t0 <= t < t1:
            return name
    return "-"


# ----------------------------------------------------------------------------- loop closure (B2)
def yaw_pitch_deg(q: Sequence[float]) -> Tuple[float, float]:
    """Yaw and pitch of the forward axis (0, 0, 1) of a Unity rotation (x, y, z, w): y up, left-handed."""
    x, y, z, w = q
    fx, fy, fz = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
    return math.degrees(math.atan2(fx, fz)), math.degrees(math.asin(max(-1.0, min(1.0, fy))))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def loop_report(run: Path) -> List[str]:
    run = Path(run)
    phases = cue_phases(run)
    rows = [r for r in load_csv(run / "head.csv") if r.get("tracked") == "1"]
    if not phases or not rows:
        return ["   no cue phases or no tracked head rows"]
    t = [float(r["mono_ns"]) for r in rows]
    pos = [(float(r["px"]), float(r["py"]), float(r["pz"])) for r in rows]
    yaw = [yaw_pitch_deg((float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])))[0] for r in rows]

    def idx(t0: float, t1: float) -> List[int]:
        return [i for i, ti in enumerate(t) if t0 <= ti < t1]

    def mean_pos(ii: List[int]) -> Tuple[float, float, float]:
        return tuple(sum(pos[i][k] for i in ii) / len(ii) for k in range(3))

    def yaw_at_end(t0: float, t1: float) -> Optional[float]:
        ii = idx(max(t0, t1 - 1.0e9), t1)   # the last second of a step: the move is done, the head holds
        if not ii:
            return None
        ref = yaw[ii[0]]
        return ref + quantile([_wrap(yaw[i] - ref) for i in ii], 0.5)

    out = ["   phases: " + ", ".join(f"{n} {(b - a) / 1e9:.1f} s" for n, a, b in phases)]
    still = {}
    for name, t0, t1 in phases:
        if not name.startswith("still"):
            continue
        ii = idx(t0 + 2e9, t1)   # skip 2 s settling after the prompt
        if len(ii) < 10:
            continue
        c = mean_pos(ii)
        d = [math.dist(pos[i], c) * 1000 for i in ii]
        yr = [_wrap(yaw[i] - yaw[ii[0]]) for i in ii]
        dur = (t[ii[-1]] - t[ii[0]]) / 1e9
        still[name] = (c, quantile([yaw[i] for i in ii], 0.5))
        out.append(f"   {name}: position spread median {quantile(d, 0.5):.1f} mm, max {max(d):.1f} mm; yaw range "
                   f"{max(yr) - min(yr):.2f} deg; drift {math.dist(pos[ii[0]], pos[ii[-1]]) * 1000 / max(dur, 1e-3):.1f} mm/s")
    prev_yaw = None
    for name, t0, t1 in phases:
        y = yaw_at_end(t0, t1)
        if name.startswith(("turn_", "front_")) and y is not None and prev_yaw is not None:
            out.append(f"   {name}: yaw change {_wrap(y - prev_yaw):+.1f} deg")
        if y is not None:
            prev_yaw = y
    if "still_1" in still and "still_2" in still:
        (a, ya), (b, yb) = still["still_1"], still["still_2"]
        horiz = math.hypot(b[0] - a[0], b[2] - a[2]) * 1000
        out.append(f"   LOOP CLOSURE (still_2 - still_1, same floor mark): horizontal {horiz:.0f} mm, height "
                   f"{(b[1] - a[1]) * 1000:+.0f} mm, yaw {_wrap(yb - ya):+.1f} deg")
    for i in range(1, len(pos)):
        step = math.dist(pos[i], pos[i - 1])
        if step > 0.10:
            out.append(f"   jump {step * 1000:.0f} mm at {(t[i] - phases[0][1]) / 1e9:.1f} s, phase {phase_of(phases, t[i])}")
    return out


# ----------------------------------------------------------------------------- motion offset (B1t, BS-2)
def image_shifts(images: Sequence) -> List[Optional[Tuple[float, float, float]]]:
    """Phase-correlation shift (dx, dy, response) from image k-1 to image k; None for k = 0 or a missing image."""
    import cv2
    import numpy as np
    out: List[Optional[Tuple[float, float, float]]] = [None]
    win = None
    for prev, cur in zip(images, images[1:]):
        if prev is None or cur is None:
            out.append(None)
            continue
        if win is None:
            win = cv2.createHanningWindow((cur.shape[1], cur.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(prev), np.float32(cur), win)
        out.append((dx, dy, resp))
    return out[:len(images)]


def rotvec(r) -> Tuple[float, float, float]:
    """Rotation vector of a 3x3 rotation (nested lists or array)."""
    tr = r[0][0] + r[1][1] + r[2][2]
    ang = math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))
    v = (r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1])
    if ang < 1e-9:
        return (v[0] / 2, v[1] / 2, v[2] / 2)
    s = ang / (2 * math.sin(ang))
    return (v[0] * s, v[1] * s, v[2] * s)


def cumulative_rotation(times: Sequence[float], rots: Sequence) -> List[Tuple[float, float, float]]:
    """Running sum of body-frame rotation increments log(R_i^T R_i+1): differences of it over a short interval are
    the body rotation over that interval (small-angle composition)."""
    acc = [(0.0, 0.0, 0.0)]
    for a, b in zip(rots, rots[1:]):
        rel = [[sum(a[k][i] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]   # a^T b
        w = rotvec(rel)
        acc.append(tuple(acc[-1][k] + w[k] for k in range(3)))
    return acc


def integrate_gyro(times: Sequence[float], rates: Sequence[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    acc = [(0.0, 0.0, 0.0)]
    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]) / 1e9
        w = [(rates[i][k] + rates[i - 1][k]) / 2 for k in range(3)]
        acc.append(tuple(acc[-1][k] + w[k] * dt for k in range(3)))
    return acc


def best_offset(intervals: Sequence[Tuple[float, float]], shifts: Sequence[Tuple[float, float]],
                ref_t: Sequence[float], ref_theta: Sequence[Tuple[float, float, float]],
                deltas_ms: Sequence[float] = DELTAS_MS) -> Optional[dict]:
    """Fit shift = A @ (theta(t1 + d) - theta(t0 + d)) + b for each trial offset d; return the best by R^2.

    ``intervals`` are (t0, t1) frame times in ns on the clock of ``ref_t``; ``shifts`` the image shifts over them."""
    import numpy as np
    if len(shifts) < 20:
        return None
    rt = np.asarray(ref_t, float)
    th = np.asarray(ref_theta, float)
    iv = np.asarray(intervals, float)
    y = np.asarray(shifts, float)
    var = ((y - y.mean(axis=0)) ** 2).sum()
    if var <= 0:
        return None
    scores = []
    for d in deltas_ms:
        a, b = iv[:, 0] + d * 1e6, iv[:, 1] + d * 1e6
        ok = (a >= rt[0]) & (b <= rt[-1])
        if ok.sum() < 20:
            scores.append(float("nan"))
            continue
        x = np.stack([np.interp(b[ok], rt, th[:, k]) - np.interp(a[ok], rt, th[:, k]) for k in range(3)], axis=1)
        x1 = np.hstack([x, np.ones((len(x), 1))])
        coef, *_ = np.linalg.lstsq(x1, y[ok], rcond=None)
        res = y[ok] - x1 @ coef
        v = ((y[ok] - y[ok].mean(axis=0)) ** 2).sum()
        scores.append(1.0 - (res ** 2).sum() / v if v > 0 else float("nan"))
    s = np.asarray(scores)
    if np.all(np.isnan(s)):
        return None
    i = int(np.nanargmax(s))
    best = deltas_ms[i]

    def at(dd: float) -> float:
        j = min(range(len(deltas_ms)), key=lambda k: abs(deltas_ms[k] - dd))
        return float(s[j])

    # half-width of the peak: where R^2 falls 0.02 below the best
    lo = hi = i
    while lo > 0 and not math.isnan(s[lo - 1]) and s[lo - 1] > s[i] - 0.02:
        lo -= 1
    while hi < len(s) - 1 and not math.isnan(s[hi + 1]) and s[hi + 1] > s[i] - 0.02:
        hi += 1
    return {"delta_ms": best, "r2": float(s[i]), "r2_minus10": at(best - 10), "r2_plus10": at(best + 10),
            "r2_zero": at(0.0), "width_ms": deltas_ms[hi] - deltas_ms[lo], "n": len(shifts)}


def _offset_line(label: str, r: Optional[dict]) -> str:
    if r is None:
        return f"   {label}: not enough motion or data"
    return (f"   {label}: best delta {r['delta_ms']:+.1f} ms (R^2 {r['r2']:.3f}; at -10 ms {r['r2_minus10']:.3f}, "
            f"+10 ms {r['r2_plus10']:.3f}, at 0 {r['r2_zero']:.3f}; peak width {r['width_ms']:.1f} ms; {r['n']} frame pairs)")


def _mat3(m: Sequence[float]):
    """Rotation part of a 4x4 stored column-major (the layout T6.2 found for GetHeadPoseAtTime)."""
    return [[m[c * 4 + r] for c in range(3)] for r in range(3)]


def sync_report(run: Path) -> List[str]:
    """B1t by motion: exposure time of the Eye relative to cam_ts, against the native head pose."""
    run = Path(run)
    frames = load_csv(run / "eye_frames.csv")
    grid = [g for g in load_csv(run / "pose_grid.csv") if g.get("ok") == "1"]
    if not frames or len(grid) < 50:
        return [f"   motion sync: {len(frames)} frames, {len(grid)} pose grid rows: not enough"]
    gt = [float(g["hmd_ns"]) for g in grid]
    rots = [_mat3([float(g[f"m{i}"]) for i in range(16)]) for g in grid]
    theta = cumulative_rotation(gt, rots)
    imgs, ts = [], []
    for f in frames:
        name = (f.get("saved") or "").split("|")[-1]
        if not name.startswith("frames/"):
            continue
        p = run / name
        imgs.append(read_pgm(p) if p.exists() else None)
        ts.append(float(f["cam_ts"]))
    sh = image_shifts(imgs)
    iv, ys = [], []
    for k in range(1, len(ts)):
        s = sh[k]
        if s is None or s[2] < 0.05 or not (20e6 < ts[k] - ts[k - 1] < 50e6):
            continue
        iv.append((ts[k - 1], ts[k]))
        ys.append((s[0], s[1]))
    out = [f"   motion sync: {len(imgs)} saved frames, {len(ys)} usable consecutive pairs, pose grid {len(grid)} rows "
           f"({(gt[-1] - gt[0]) / 1e9:.1f} s)"]
    out.append(_offset_line("Eye exposure - cam_ts (head pose)", best_offset(iv, ys, gt, theta)))
    out.append("   (negative delta: the exposure is before cam_ts; arrival age before cam_ts adds to the 130 ms)")
    return out


# ----------------------------------------------------------------------------- stereo pan (BS-2)
def read_thumbs(path: Path, w: int, h: int):
    """Records of stereo_thumbs_<id>.bin: (timestamp ns, image)."""
    import numpy as np
    data = path.read_bytes()
    n = 8 + w * h
    out = []
    for o in range(0, len(data) - n + 1, n):
        ts = struct.unpack(">q", data[o:o + 8])[0]
        out.append((ts, np.frombuffer(data, np.uint8, w * h, o + 8).reshape(h, w)))
    return out


def gyro_series(run: Path) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    rows = [r for r in load_csv(Path(run) / "imu.csv") if r.get("type") == "4"]   # TYPE_GYROSCOPE, rad/s
    rows.sort(key=lambda r: float(r["event_ns"]))
    return [float(r["event_ns"]) for r in rows], [(float(r["v0"]), float(r["v1"]), float(r["v2"])) for r in rows]


def pan_report(run: Path) -> List[str]:
    run = Path(run)
    meta = load_json(run)
    stop = meta.get("stereo_stop") or {}
    if isinstance(stop, str):
        return [f"   pan: stereo stop {stop}"]
    size = str(stop.get("thumb_size", ""))
    if "x" not in size:
        return ["   pan: no thumbnails"]
    w, h = (int(v) for v in size.split("x"))
    gt, gr = gyro_series(run)
    if len(gt) < 100:
        return ["   pan: no gyroscope in imu.csv"]
    theta = integrate_gyro(gt, gr)
    out = [f"   pan: gyroscope {len(gt)} events over {(gt[-1] - gt[0]) / 1e9:.1f} s; thumbnails {w}x{h}"]
    res = {}
    for f in sorted(run.glob("stereo_thumbs_*.bin")):
        pid = f.stem.split("_")[-1]
        recs = read_thumbs(f, w, h)
        sh = image_shifts([img for _, img in recs])
        iv, ys = [], []
        for k in range(1, len(recs)):
            s = sh[k]
            if s is None or s[2] < 0.05 or not (20e6 < recs[k][0] - recs[k - 1][0] < 50e6):
                continue
            iv.append((recs[k - 1][0], recs[k][0]))
            ys.append((s[0], s[1]))
        r = best_offset(iv, ys, gt, theta)
        res[pid] = r
        out.append(_offset_line(f"sensor {pid} exposure - image timestamp (gyro), {len(recs)} images", r))
    ok = [r for r in res.values() if r]
    if len(ok) == 2:
        a, b = ok
        out.append(f"   EXPOSURE OFFSET between the sensors: {a['delta_ms'] - b['delta_ms']:+.1f} ms "
                   f"(peak widths {a['width_ms']:.1f} / {b['width_ms']:.1f} ms)")
    return out


def phase_lines(run: Path) -> List[str]:
    phases = cue_phases(run)
    ev = load_csv(Path(run) / "events.csv")
    speech = next((e.get("detail") for e in ev if e.get("event") == "cue_speech"), "?")
    meta = load_json(run)
    first = meta.get("glasses_first_frame_s"), meta.get("glasses_first_pose_s"), meta.get("glasses_square_placed_s")
    return [f"   guided: {len(phases)} cue steps, speech {speech}",
            f"   glasses: first frame {first[0]} s, first head pose {first[1]} s, square placed {first[2]} s after app start"]
