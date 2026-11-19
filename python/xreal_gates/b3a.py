"""T6.6, gate B3a (ADR-0004 option A2): ArUco detection and the board pose on the Beam Pro, against Python.

Two halves:

* ``make_set`` writes the frame set the agent pushes to the app's ``files/b3a/`` (``python -m xreal_gates b3a-set``):
  ``set.json`` (dictionary, detector tuning, Eye intrinsics, plate A's marker corners in metres, frame list),
  synthetic 1280x720 frames of plate A at known poses (``truth.json``), and empty frames for the no-marker cost.
* ``detect_report`` reads a ``<stamp>_detect`` run: the bench (each set frame processed 5 times with 1 thread and
  with OpenCV's default threads) and the live part (each Eye frame processed on a worker thread while the glasses
  draw). The parity test runs the Python pipeline (``pose_estimator``: CLAHE, ``ArucoDetector``, the same
  ``LStructureDetector._solve_board_pose``) on the same frames, the bench set and the Eye frames the run saved, and
  compares marker ids, corners and poses.

Times on the device are CLOCK_MONOTONIC, the HMD clock of ``cam_ts``, so ``end_mono_ns - cam_ts`` is the age of a
pose when it is ready, before any filter or rendering.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .beampro import load_csv, quantile

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent
REPO = PYTHON.parent
DEFAULT_SET = PYTHON / "beampro_runs" / "b3a_set"
DEFAULT_MARKERS = PYTHON / "config" / "markers" / "plate_A.markers.json"
# The Eye calibration from the PC gates (2026-10-21, 1280x720, RMS 0.47 px; local, gitignored run folder). It was
# taken on the PC stream; whether it holds on the Beam Pro's stream is checked in B3b, not here: parity and timing
# use the same K on both sides.
DEFAULT_CALIB = REPO / "unity_projects" / "xreal_pc_gates" / "GateRuns" / "runs" / "20261021_113312_Calib_calib" / "eye_calibration.json"
SIZE = (1280, 720)
SCHEMA = "eyelab-b3a/1"


# ---- set ---------------------------------------------------------------------------------------------------------

def board(markers_json: Path = DEFAULT_MARKERS) -> Tuple[List[int], np.ndarray]:
    """Marker ids and object corners (n, 4, 3) in metres, in marker_object_corners order."""
    from registration import marker_object_corners, read_marker_config
    cfg = read_marker_config(str(markers_json))
    ids, corners = [], []
    for m in cfg.markers:
        size_mm = m.marker_size_mm if m.marker_size_mm is not None else cfg.default_marker_size_mm
        ids.append(int(m.marker_id))
        corners.append(marker_object_corners(m.unv_position, m.normal, m.roll_deg, float(size_mm) / 1000.0))
    return ids, np.asarray(corners, dtype=np.float64)


def eye_calibration(path: Path = DEFAULT_CALIB) -> Tuple[np.ndarray, np.ndarray]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(d["K"], dtype=np.float64), np.asarray(d["dist"], dtype=np.float64).reshape(-1)


def tuning_list() -> List[float]:
    from pose_estimator import DEFAULT_ARUCO_DETECTOR_TUNING
    return [float(v) for v in dataclasses.astuple(DEFAULT_ARUCO_DETECTOR_TUNING)]


def _rot(rx_deg: float, ry_deg: float, rz_deg: float = 0.0) -> np.ndarray:
    """Plate rotation in the camera frame: facing the camera (180 deg about x), then the tilts."""
    face = cv2.Rodrigues(np.array([math.pi, 0.0, 0.0]))[0]
    tilt = cv2.Rodrigues(np.radians([rx_deg, 0.0, 0.0]))[0] @ cv2.Rodrigues(np.radians([0.0, ry_deg, 0.0]))[0]
    roll = cv2.Rodrigues(np.radians([0.0, 0.0, rz_deg]))[0]
    return roll @ tilt @ face


def render(K: np.ndarray, dist: np.ndarray, ids: Sequence[int], corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
           seed: int, size: Tuple[int, int] = SIZE) -> np.ndarray:
    """Plate A on a grey, textured background, seen by the Eye at (rvec, tvec); mild blur and noise."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w), 110, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 6)
    pts = corners.reshape(-1, 3)
    lo, hi = pts.min(axis=0) - 0.02, pts.max(axis=0) + 0.02
    plate = np.array([[lo[0], hi[1], 0], [hi[0], hi[1], 0], [hi[0], lo[1], 0], [lo[0], lo[1], 0]], np.float64)
    pp, _ = cv2.projectPoints(plate, rvec, tvec, K, dist)
    cv2.fillConvexPoly(img, np.round(pp.reshape(-1, 2) * 16).astype(np.int32), 235, lineType=cv2.LINE_AA, shift=4)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = 240
    for mid, c in zip(ids, corners):
        marker = cv2.aruco.generateImageMarker(dictionary, int(mid), side, borderBits=1).astype(np.float32)
        marker = marker * (235 - 25) / 255.0 + 25
        dst, _ = cv2.projectPoints(c.astype(np.float64), rvec, tvec, K, dist)
        src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
        H = cv2.getPerspectiveTransform(src, dst.reshape(-1, 2).astype(np.float32))
        warped = cv2.warpPerspective(marker, H, (w, h), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.ones((side, side), np.float32), H, (w, h), flags=cv2.INTER_AREA)
        img = img * (1 - mask) + warped * mask
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 3.0, (h, w))
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def write_pgm(path: Path, img: np.ndarray) -> None:
    h, w = img.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{w} {h}\n255\n".encode("ascii") + np.ascontiguousarray(img).tobytes())


def read_pgm(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    parts, pos = [], 0
    while len(parts) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        parts.append(data[start:pos])
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(data, np.uint8, w * h, pos + 1).reshape(h, w)


POSES = [(d, rx, ry, rz) for d in (0.35, 0.5, 0.7, 1.0) for (rx, ry, rz) in ((0, 0, 0), (25, 0, 5), (0, 35, -5), (20, -25, 10))]


def make_set(out: Path = DEFAULT_SET, calib: Path = DEFAULT_CALIB, markers: Path = DEFAULT_MARKERS,
             empty: int = 4, snapshots: Optional[Path] = None) -> dict:
    """Write set.json, truth.json and frames/*.pgm into ``out`` (replacing an older set)."""
    out = Path(out)
    K, dist = eye_calibration(calib)
    ids, corners = board(markers)
    frames, truth = [], {}
    for i, (d, rx, ry, rz) in enumerate(POSES):
        rvec = cv2.Rodrigues(_rot(rx, ry, rz))[0].reshape(3)
        tvec = np.array([0.03 * ((i % 3) - 1), 0.02 * ((i % 2) * 2 - 1), d])
        name = f"frames/plate_{i:02d}.pgm"
        write_pgm(out / name, render(K, dist, ids, corners, rvec, tvec, seed=i))
        frames.append(name)
        truth[name] = {"rvec": rvec.tolist(), "tvec": tvec.tolist(), "distance_m": d, "tilt_deg": [rx, ry, rz]}
    rng = np.random.default_rng(1000)
    for j in range(empty):
        img = np.clip(110 + cv2.GaussianBlur(rng.normal(0, 40, SIZE[::-1]), (0, 0), 3 + j), 0, 255).astype(np.uint8)
        name = f"frames/empty_{j:02d}.pgm"
        write_pgm(out / name, img)
        frames.append(name)
    if snapshots:
        for p in sorted(Path(snapshots).glob("*.pgm"))[:6]:
            img = read_pgm(p)
            if img.shape == SIZE[::-1]:
                name = f"frames/eye_{p.stem}.pgm"
                write_pgm(out / name, img)
                frames.append(name)
    s = {
        "schema": SCHEMA,
        "board_name": Path(markers).name.split(".")[0],
        "calibration": f"{Path(calib).parent.name}/{Path(calib).name}",
        "dictionary": int(cv2.aruco.DICT_4X4_50),
        "tuning": tuning_list(),
        "K": K.reshape(-1).tolist(),
        "dist": dist.tolist(),
        "ids": ids,
        "corners": corners.reshape(-1).tolist(),
        "frames": frames,
    }
    (out / "set.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return s


def load_set(set_dir: Path = DEFAULT_SET) -> Optional[dict]:
    p = Path(set_dir) / "set.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---- the Python side of the parity test --------------------------------------------------------------------------

class PythonPipeline:
    """CLAHE, detection and the board pose exactly as pose_estimator does them, with the set's tuning."""

    def __init__(self, s: dict):
        from pose_estimator import ArucoDetectorTuning, LStructureDetector, make_detector_parameters
        fields = dataclasses.fields(ArucoDetectorTuning)
        tuning = ArucoDetectorTuning(*[int(v) if f.type in (int, "int") else float(v) for f, v in zip(fields, s["tuning"])])
        self.clip = tuning.clip_limit
        self.clahe = cv2.createCLAHE(clipLimit=round(float(self.clip), 3), tileGridSize=(8, 8))
        dictionary = cv2.aruco.getPredefinedDictionary(int(s["dictionary"]))
        self.detector = cv2.aruco.ArucoDetector(dictionary, make_detector_parameters(tuning))
        obj = np.asarray(s["corners"], np.float32).reshape(-1, 4, 3)
        self.board = cv2.aruco.Board(list(obj), dictionary, np.asarray(s["ids"], np.int32))
        self.K = np.asarray(s["K"], np.float64).reshape(3, 3)
        self.dist = np.asarray(s["dist"], np.float64)
        self._solve = LStructureDetector._solve_board_pose
        self._cls = LStructureDetector

    def process(self, gray: np.ndarray) -> dict:
        enhanced = self.clahe.apply(gray) if self.clip > 0 else gray
        corners, ids, _rej = self.detector.detectMarkers(enhanced)
        out = {"ids": [], "corners": {}, "pose": None}
        if ids is None or len(ids) == 0:
            return out
        out["ids"] = [int(i) for i in ids.flatten()]
        out["corners"] = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
        obj, img = self.board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < 4:
            return out
        rvec, tvec, _amb = self._solve(self._cls, obj, img, self.K, self.dist)
        if rvec is not None:
            out["pose"] = (np.asarray(rvec, np.float64).reshape(3), np.asarray(tvec, np.float64).reshape(3))
        return out


def rotation_deg(r1: Sequence[float], r2: Sequence[float]) -> float:
    R = cv2.Rodrigues(np.asarray(r1, np.float64))[0].T @ cv2.Rodrigues(np.asarray(r2, np.float64))[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def _f(v: Optional[str]) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _corners_by_frame(rows: List[dict], key: str) -> Dict[str, Dict[int, np.ndarray]]:
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for r in rows:
        c = np.array([float(r[f"{a}{k}"]) for k in range(4) for a in ("x", "y")]).reshape(4, 2)
        out.setdefault(r[key], {})[int(r["id"])] = c
    return out


def compare(dev_corners: Dict[int, np.ndarray], dev_pose: Optional[Tuple[np.ndarray, np.ndarray]], py: dict) -> dict:
    """Ids, the largest corner difference (px) and the pose difference (mm, deg) between device and Python."""
    d_ids, p_ids = set(dev_corners), set(py["ids"])
    common = d_ids & p_ids
    corner = max((float(np.abs(dev_corners[i] - py["corners"][i]).max()) for i in common), default=float("nan"))
    t_mm = r_deg = float("nan")
    if dev_pose is not None and py["pose"] is not None:
        t_mm = float(np.linalg.norm(dev_pose[1] - py["pose"][1]) * 1000)
        r_deg = rotation_deg(dev_pose[0], py["pose"][0])
    return {"ids_equal": d_ids == p_ids, "only_device": sorted(d_ids - p_ids), "only_python": sorted(p_ids - d_ids),
            "corner_px": corner, "t_mm": t_mm, "r_deg": r_deg,
            "pose_both": dev_pose is not None and py["pose"] is not None,
            "pose_one": (dev_pose is None) != (py["pose"] is None)}


def _pose(r: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if r.get("pose_ok") != "1":
        return None
    return (np.array([float(r[f"r{k}"]) for k in range(3)]), np.array([float(r[f"t{k}"]) for k in range(3)]))


def _q(xs: List[float]) -> str:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return "     -/     -"
    return f"{quantile(xs, 0.5):6.2f}/{quantile(xs, 0.95):6.2f}"


# ---- reports -----------------------------------------------------------------------------------------------------

def bench_report(run: Path, set_dir: Path = DEFAULT_SET) -> List[str]:
    run, set_dir = Path(run), Path(set_dir)
    rows = load_csv(run / "bench.csv")
    if not rows:
        return ["   bench: no bench.csv"]
    out = [f"   bench: {len(rows)} rows (median/p95 ms over frames x reps)"]
    out.append("   threads    frames  markers>0   clahe          detect         pose           total")
    for tm in sorted({r["threads_req"] for r in rows}, key=lambda v: int(v)):
        sel = [r for r in rows if r["threads_req"] == tm]
        used = sel[0].get("threads", "?")
        frames = len({r["frame"] for r in sel})
        with_m = [r for r in sel if int(r["markers"] or 0) > 0]
        out.append(f"   {('1' if tm == '1' else 'default')+' ('+used+')':10s} {frames:5d}   {len(with_m) / len(sel) * 100:5.0f} %  "
                   f"{_q([_f(r['clahe_ms']) for r in sel])}  {_q([_f(r['detect_ms']) for r in sel])}  "
                   f"{_q([_f(r['pose_ms']) for r in sel])}  {_q([_f(r['total_ms']) for r in sel])}")
        empty = [r for r in sel if "empty" in r["frame"] or "eye_" in r["frame"]]
        if empty:
            out.append(f"     frames without plate A: total {_q([_f(r['total_ms']) for r in empty])} ms")
    s = load_set(set_dir)
    if s is None:
        return out + [f"   parity: no set.json in {set_dir}"]
    truth = json.loads((set_dir / "truth.json").read_text(encoding="utf-8")) if (set_dir / "truth.json").exists() else {}
    py = PythonPipeline(s)
    dev_corners = _corners_by_frame(load_csv(run / "bench_corners.csv"), "frame")
    first = {r["frame"]: r for r in rows if r["threads_req"] == "1" and r["rep"] == "0"}
    res, err_dev, err_py = [], [], []
    for name, r in first.items():
        p = set_dir / name
        if not p.exists():
            continue
        pr = py.process(read_pgm(p))
        c = compare(dev_corners.get(name, {}), _pose(r), pr)
        res.append(c)
        if name in truth:
            t = truth[name]
            for pose, acc in ((_pose(r), err_dev), (pr["pose"], err_py)):
                if pose is not None:
                    acc.append((float(np.linalg.norm(pose[1] - np.asarray(t["tvec"])) * 1000), rotation_deg(pose[0], t["rvec"])))
    out.append(_parity_line("parity (bench)", res))
    for label, acc in (("device", err_dev), ("python", err_py)):
        if acc:
            out.append(f"   truth error {label}: {len(acc)} poses, t median {quantile([a[0] for a in acc], 0.5):.2f} mm, "
                       f"max {max(a[0] for a in acc):.2f} mm; r median {quantile([a[1] for a in acc], 0.5):.3f} deg")
    return out


def _parity_line(label: str, res: List[dict]) -> str:
    if not res:
        return f"   {label}: nothing compared"
    same = sum(1 for c in res if c["ids_equal"])
    corner = [c["corner_px"] for c in res if c["corner_px"] == c["corner_px"]]
    t = [c["t_mm"] for c in res if c["pose_both"]]
    r = [c["r_deg"] for c in res if c["pose_both"]]
    one = sum(1 for c in res if c["pose_one"])
    diff = [f"{c['only_device']}/{c['only_python']}" for c in res if not c["ids_equal"]]
    return (f"   {label}: {len(res)} frames, ids equal {same}/{len(res)}"
            + (f" (device-only/python-only: {', '.join(diff[:4])})" if diff else "")
            + (f"; corners max {max(corner):.4f} px" if corner else "")
            + (f"; pose max {max(t):.3f} mm, {max(r):.4f} deg over {len(t)}" if t else "")
            + (f"; pose on one side only {one}" if one else ""))


def live_report(run: Path, set_dir: Path = DEFAULT_SET, refresh_hz: float = 60.0) -> List[str]:
    run = Path(run)
    rows = load_csv(run / "live.csv")
    if not rows:
        return ["   live: no live.csv"]
    frames = load_csv(run / "frames.csv")
    out = ["   live: per cue step; ms median/p95; age = monotonic now - cam_ts (HMD clock)"]
    out.append("   step       thr  done/s  skipped  markers  pose   total ms       wait ms        age ready ms    unity fps  >1 refresh")
    order: List[str] = []
    for r in rows:
        if r["step"] not in order:
            order.append(r["step"])
    for step in order:
        sel = [r for r in rows if r["step"] == step]
        t = [int(r["end_mono_ns"]) for r in sel]
        span = (t[-1] - t[0]) / 1e9 if len(t) > 1 else float("nan")
        rate = (len(t) - 1) / span if span and span == span and span > 0 else float("nan")
        skipped = int(sel[-1]["skipped"]) - int(sel[0]["skipped"])
        seen = sum(1 for r in sel if int(r["markers"] or 0) > 0) / len(sel)
        posed = sum(1 for r in sel if r["pose_ok"] == "1") / len(sel)
        wait = [(int(r["start_mono_ns"]) - int(r["arrival_mono_ns"])) / 1e6 for r in sel]
        age = [(int(r["end_mono_ns"]) - int(r["cam_ts"])) / 1e6 for r in sel]
        fr = [f for f in frames if f["step"] == step]
        dt = [x for x in (_f(f["dt_ms"]) for f in fr) if x is not None]
        fps = len(dt) / (sum(dt) / 1000) if dt else float("nan")
        over = sum(1 for x in dt if x > 1000 / refresh_hz * 1.25) / len(dt) * 100 if dt else float("nan")
        out.append(f"   {step:10s} {sel[0]['threads']:>3s}  {rate:6.1f}  {skipped:7d}  {seen * 100:5.0f} %  {posed * 100:3.0f} %"
                   f"  {_q([_f(r['total_ms']) for r in sel])}  {_q(wait)}  {_q(age)}  {fps:9.1f}  {over:7.1f} %")
        still = [_pose(r) for r in sel if r["pose_ok"] == "1"]
        if step.endswith("_still") and len(still) > 5:
            tv = np.array([p[1] for p in still]) * 1000
            out.append(f"     still: {len(still)} poses, t std {np.linalg.norm(tv.std(axis=0)):.2f} mm, "
                       f"distance median {np.median(np.linalg.norm(tv, axis=1)):.0f} mm, rms {_q([_f(r['rms_px']) for r in sel])} px")
    s = load_set(set_dir)
    saved = [r for r in rows if r.get("saved")]
    if s is not None and saved:
        py = PythonPipeline(s)
        dev_corners = _corners_by_frame(load_csv(run / "live_corners.csv"), "seq")
        res = []
        for r in saved:
            p = run / r["saved"]
            if p.exists():
                res.append(compare(dev_corners.get(r["seq"], {}), _pose(r), py.process(read_pgm(p))))
        out.append(_parity_line("parity (Eye frames)", res))
    return out


def detect_report(run: Path, set_dir: Path = DEFAULT_SET) -> List[str]:
    from .beampro import load_json
    meta = load_json(run)
    lines = [f"== {Path(run).name}  detect  app {meta.get('app_version', '?')}",
             f"   plugin: {meta.get('plugin', meta.get('plugin_error', '?'))}",
             f"   set: {meta.get('set', '?')}; bench {meta.get('bench_status', '?')}",
             f"   glasses seen: {meta.get('glasses_seen', '?')}; Eye frames {meta.get('eye_frames', '?')}, processed "
             f"{meta.get('processed', '?')}, skipped {meta.get('skipped', '?')}"
             + (f"; WORKER ERROR {meta['worker_error']}" if meta.get("worker_error") else "")]
    return lines + bench_report(run, set_dir) + live_report(run, set_dir)
