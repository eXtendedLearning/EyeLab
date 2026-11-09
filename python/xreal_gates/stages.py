"""Per-stage latency of the Eye correction path (T5.1 Phase 1). ``python -m xreal_gates stages RUN``.

Reads ``frame_timing.csv`` (one row per frame sent to the pose service, build 0.3.008 and later). Every
time in it is QPC seconds, the one clock that Unity (``Stopwatch.GetTimestamp()``) and the pose service
(``time.perf_counter()`` on Windows) share, so stages across the process boundary subtract directly.

Stages, in path order (columns ``q_*``):

=================  =========================================================================================
exposure->arrival  ``base + queue age`` as logged (the base is fitted, :mod:`latency`); measured directly
                   when the Eye looks at the Surface timecode (``--target timecode``, :mod:`timecode`):
                   arrival minus the draw that showed the id, which includes the panel's own delay
poll interval      Unity frame start minus the previous one: the vendor texture polls the SDK once per
                   frame, so a decoded frame waits on average half of it before Unity sees it
texture            Unity frame start -> ``FrameArrived`` (acquire, three plane uploads)
hold               arrival -> our copy starts (waiting for the service to be free, the send-rate cap)
copy               the 0.9 MB managed copy of the Y plane
send               WebSocket send of the frame (client side)
to service         send done -> the service has the whole message
parse              received -> detection starts (header, flip, FramePacket)
detection          detection + PnP for every target
reply              detection end -> reply received in Unity (JSON, socket)
apply wait         reply received -> applied to the lock (next Update)
display            applied -> next frame starts (the overlay is drawn in between)
=================  =========================================================================================

``HMD clock``: ``hmd_arrived_ns`` is ``GetHMDTimeNanos`` at each frame's arrival. If ``cam_ts`` is on the
same clock, ``hmd - cam_ts`` is the exact frame age at arrival (and its minimum the true base); if the
clocks differ, the difference is huge or drifts. ``native pose``: the 4x4 that ``GetHeadPoseAtTime``
returns for ``cam_ts`` and for HMD-now, compared with Unity's head pose to find its convention.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anchor import qrot

STAGES: Sequence[Tuple[str, str, str]] = (
    ("texture", "q_frame", "q_arrived"),
    ("hold", "q_arrived", "q_copy0"),
    ("copy", "q_copy0", "q_copy1"),
    ("send", "q_send0", "q_send1"),
    ("to service", "q_send1", "q_recv"),
    ("parse", "q_recv", "q_det0"),
    ("detection", "q_det0", "q_det1"),
    ("reply", "q_det1", "q_reply_recv"),
    ("apply wait", "q_reply_recv", "q_applied"),
    ("display", "q_applied", "q_shown"),
)


def _rows(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(r: dict, k: str) -> float:
    v = r.get(k, "")
    try:
        return float(v) if v not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def _stat(ms: np.ndarray) -> Optional[Tuple[float, float, int]]:
    ms = ms[np.isfinite(ms)]
    if ms.size == 0:
        return None
    return float(np.median(ms)), float(np.percentile(ms, 90)), int(ms.size)


def stage_table(rows: List[dict]) -> List[Tuple[str, Optional[Tuple[float, float, int]]]]:
    """(stage, (median ms, p90 ms, n)) in path order, from frame_timing rows."""
    out = []
    poll = np.array([(_f(r, "q_frame") - _f(r, "q_frame_prev")) * 1000 for r in rows])
    out.append(("poll interval", _stat(poll)))
    for name, a, b in STAGES:
        out.append((name, _stat(np.array([(_f(r, b) - _f(r, a)) * 1000 for r in rows]))))
    age = np.array([_f(r, "age_ms") - _f(r, "hold_ms") for r in rows])
    out.append(("exposure->arrival (base+queue, logged)", _stat(age)))
    send_to_shown = np.array([(_f(r, "q_shown") - _f(r, "q_copy0")) * 1000 for r in rows])
    out.append(("copy start -> shown", _stat(send_to_shown)))
    total = np.array([_f(r, "age_ms") + (_f(r, "q_shown") - _f(r, "q_copy0")) * 1000 for r in rows])
    out.append(("exposure -> shown (total)", _stat(total)))
    return out


def timecode_latency(rows: List[dict], service: List[dict], draws) -> Dict[str, Optional[Tuple[float, float, int]]]:
    """Camera-to-app from the timecode: arrival minus the draw (and minus the shown time) of the id read."""
    from .timecode import draw_time, index_draws

    if not draws:
        return {}
    idx = index_draws(draws)
    ids = {}
    for r in service:
        if r.get("target") == "timecode" and r.get("marker_ids"):
            ids[int(r["seq"])] = int(r["marker_ids"].split()[0])
    d_draw, d_shown = [], []
    for r in rows:
        seq = int(r["seq"])
        if seq not in ids:
            continue
        t_arr = _f(r, "q_arrived")
        hit = draw_time(draws, idx, ids[seq], t_arr)
        if hit is None:
            continue
        d_draw.append((t_arr - hit[0]) * 1000)
        d_shown.append((t_arr - hit[1]) * 1000)
    return {"draw -> arrival": _stat(np.array(d_draw)), "waitKey return -> arrival": _stat(np.array(d_shown)),
            "frames read": (float(len(d_draw)), 0.0, len(ids))}


def hmd_clock(camera: List[dict]) -> Optional[dict]:
    """``hmd_ns - cam_ts`` at arrival over all frames: same clock if small, positive and slowly varying."""
    d = np.array([(int(r["hmd_ns"]) - int(r["cam_ts"])) / 1e9 for r in camera
                  if r.get("hmd_ns") not in (None, "", "0") and r.get("cam_ts")])
    if d.size == 0:
        return None
    return {"n": int(d.size), "min_s": float(d.min()), "median_s": float(np.median(d)), "p90_s": float(np.percentile(d, 90)),
            "same_clock": bool(0.0 < d.min() < 2.0 and np.percentile(d, 90) < 5.0)}


# ---------------------------------------------------------------- native pose convention

def _mat(text: str) -> Optional[np.ndarray]:
    v = [float(x) for x in text.split()] if text else []
    return np.array(v, float).reshape(4, 4) if len(v) == 16 else None


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    """(x, y, z, w) of a proper rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


CONVENTIONS = {
    "row-major": lambda M: M[:3, :3],
    "column-major": lambda M: M[:3, :3].T,
    "row-major, z flipped": lambda M: np.diag([1, 1, -1]) @ M[:3, :3] @ np.diag([1, 1, -1]),
    "column-major, z flipped": lambda M: np.diag([1, 1, -1]) @ M[:3, :3].T @ np.diag([1, 1, -1]),
}


def _angle(qa, qb) -> float:
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))))


def native_pose(rows: List[dict]) -> Optional[dict]:
    """Which reading of ``GetHeadPoseAtTime``'s 4x4 matches Unity's head rotation, and how well.

    Compares the native pose for HMD-now with Unity's head pose logged in the same row; relative
    rotations between rows are used (a fixed frame offset between the two worlds cancels). Then, with the
    best convention, the native pose at ``cam_ts`` against Unity's head pose at the logged capture time.
    """
    pairs = []
    for r in rows:
        M = _mat(r.get("native_now", ""))
        u = r.get("unity_head_now", "").split()
        if M is None or len(u) != 7 or not np.all(np.isfinite(M)):
            continue
        pairs.append((M, np.array([float(x) for x in u[3:]])))
    if len(pairs) < 5:
        return None
    result = {}
    for name, conv in CONVENTIONS.items():
        errs = []
        q0n, q0u = _quat_from_R(conv(pairs[0][0])), pairs[0][1]
        for M, qu in pairs[1:]:
            qn = _quat_from_R(conv(M))
            # relative rotation from the first sample, in each world
            rn = _qmul(_qinv(q0n), qn)
            ru = _qmul(_qinv(q0u), qu)
            errs.append(_angle(rn, ru))
        result[name] = float(np.median(errs))
    best = min(result, key=result.get)
    return {"conventions_deg": result, "best": best, "n": len(pairs)}


def _qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


# ---------------------------------------------------------------- report

def find_timecodes(run_dir: Path, rows: List[dict]) -> List[Path]:
    """Timecode CSVs (``GateRuns/timecode/``) whose draws overlap the run's QPC range."""
    q = np.array([_f(r, "q_arrived") for r in rows])
    q = q[np.isfinite(q)]
    if q.size == 0:
        return []
    out = []
    for folder in (Path(run_dir), Path(run_dir).parent.parent / "timecode"):
        for p in sorted(folder.glob("timecode_*.csv")) if folder.exists() else []:
            r = _rows(p)
            if r and float(r[0]["qpc_draw"]) < q.max() and float(r[-1]["qpc_draw"]) > q.min():
                out.append(p)
    return out


def report(run_dir: Path, timecode: Optional[Sequence[Path]] = None) -> List[str]:
    from .timecode import load_draws

    run_dir = Path(run_dir)
    rows = _rows(run_dir / "frame_timing.csv")
    name = run_dir.name
    if not rows:
        return [f"{name}: no frame_timing.csv (build 0.3.008 or later)"]
    lines = [f"{name}: {len(rows)} frames sent; stage latency in ms (median / p90 / n), QPC clock"]
    for stage, st in stage_table(rows):
        lines.append(f"  {stage:40s} " + ("-" if st is None else f"{st[0]:7.1f} / {st[1]:7.1f}  (n={st[2]})"))
    tcs = list(timecode) if timecode else find_timecodes(run_dir, rows)
    if tcs:
        tl = timecode_latency(rows, _rows(run_dir / "pose_service.csv"), load_draws(tcs))
        lines.append(f"  timecode ({', '.join(p.name for p in tcs)}):")
        for k, st in tl.items():
            if k == "frames read":
                lines.append(f"    frames with a timecode: {int(st[0])} of {st[2]}" if st else "")
            else:
                lines.append(f"    {k:36s} " + ("-" if st is None else f"{st[0]:7.1f} / {st[1]:7.1f}  (n={st[2]})"))
    hc = hmd_clock(_rows(run_dir / "camera.csv"))
    if hc:
        lines.append(f"  HMD clock - cam_ts at arrival: min {hc['min_s'] * 1000:.1f} ms, median {hc['median_s'] * 1000:.1f} ms, "
                     f"p90 {hc['p90_s'] * 1000:.1f} ms (n={hc['n']}) -> "
                     + ("same clock: hmd - cam_ts is the exact frame age" if hc["same_clock"] else "different clocks"))
    else:
        lines.append("  HMD clock: not logged (GetHMDTimeNanos unavailable)")
    npose = native_pose(rows)
    if npose:
        conv = ", ".join(f"{k} {v:.2f} deg" for k, v in npose["conventions_deg"].items())
        lines.append(f"  GetHeadPoseAtTime(now) vs Unity head, relative rotation error: {conv} -> {npose['best']} (n={npose['n']})")
    return lines


__all__ = ["report", "stage_table", "timecode_latency", "hmd_clock", "native_pose", "qrot"]
