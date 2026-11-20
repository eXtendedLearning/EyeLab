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
           seed: int, size: Tuple[int, int] = SIZE, noise: Optional[float] = None) -> np.ndarray:
    """Plate A on a grey, textured background, seen by the Eye at (rvec, tvec); mild blur and noise.

    ``noise`` is the per-pixel sigma in grey levels (default ``SYNTH_NOISE``). The detector's cost depends steeply on
    it, because CLAHE and the adaptive threshold turn fine noise into contours: sigma 3 gave about 65,000 contours
    and 170 ms on the desktop, against about 4,000 in a real Eye frame (B3a run 2026-11-19); 0.7 gives about 1,500.
    Real Eye frames are the cost reference (``eyeplate_*`` frames in the set); the synthetic ones are for parity
    and truth.
    """
    noise = SYNTH_NOISE if noise is None else noise
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
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, noise, (h, w))
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


#: per-pixel noise of the synthetic frames (grey levels); see ``render``
SYNTH_NOISE = 0.7

POSES = [(d, rx, ry, rz) for d in (0.35, 0.5, 0.7, 1.0) for (rx, ry, rz) in ((0, 0, 0), (25, 0, 5), (0, 35, -5), (20, -25, 10))]


def make_set(out: Path = DEFAULT_SET, calib: Path = DEFAULT_CALIB, markers: Path = DEFAULT_MARKERS,
             empty: int = 4, snapshots: Optional[Path] = None, eye_plate: Optional[Path] = None,
             bottom_up: bool = False, eye_plate_count: int = 8) -> dict:
    """Write set.json, truth.json and frames/*.pgm into ``out`` (replacing an older set).

    ``eye_plate``: a folder of full-resolution Eye frames with plate A in view (e.g. a detect run's ``frames``);
    ``eye_plate_count`` of them where a plain detector finds two or more board markers are added as ``eyeplate_*``
    frames, flipped upright when ``bottom_up`` (the Eye stores rows bottom-up; 0.4.006 saved them that way). They
    have no truth; they give the cost and the parity on real frames.
    """
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
    if eye_plate:
        found = []
        for p in sorted(Path(eye_plate).glob("*.pgm")):
            img = read_pgm(p)
            if img.shape != SIZE[::-1]:
                continue
            img = img[::-1].copy() if bottom_up else img
            if len(_detect_ids(img, ids)) >= 2:
                found.append((p, img))
        step = max(1, len(found) // max(1, eye_plate_count))
        for p, img in found[::step][:eye_plate_count]:
            name = f"frames/eyeplate_{p.stem}.pgm"
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


def _detect_ids(img: np.ndarray, board_ids: Sequence[int]) -> List[int]:
    """Board ids found by a plain DICT_4X4_50 detector (to select and check frames; not the parity pipeline)."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    _, found, _ = det.detectMarkers(img)
    return [] if found is None else sorted(set(int(i) for i in found.ravel()) & set(board_ids))


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
        out.append(f"   {('default' if tm == '-1' else tm)+' ('+used+')':10s} {frames:5d}   {len(with_m) / len(sel) * 100:5.0f} %  "
                   f"{_q([_f(r['clahe_ms']) for r in sel])}  {_q([_f(r['detect_ms']) for r in sel])}  "
                   f"{_q([_f(r['pose_ms']) for r in sel])}  {_q([_f(r['total_ms']) for r in sel])}")
        for label, key in (("synthetic plate A", "plate_"), ("real Eye frames with plate A", "eyeplate_")):
            part = [r for r in sel if key in r["frame"] and (key != "plate_" or "eyeplate_" not in r["frame"])]
            if part:
                out.append(f"     {label}: total {_q([_f(r['total_ms']) for r in part])} ms "
                           f"over {len({r['frame'] for r in part})} frames")
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
    res, err_dev, err_py, div = [], [], [], []
    for name, r in first.items():
        p = set_dir / name
        if not p.exists():
            continue
        pr = py.process(read_pgm(p))
        c = compare(dev_corners.get(name, {}), _pose(r), pr)
        res.append(c)
        if name in truth:
            t = truth[name]
            if c["corner_px"] == c["corner_px"] and c["corner_px"] > PARITY_PX:
                div.append(_divergence(name, dev_corners.get(name, {}), pr, s, t))
            for pose, acc in ((_pose(r), err_dev), (pr["pose"], err_py)):
                if pose is not None:
                    acc.append((float(np.linalg.norm(pose[1] - np.asarray(t["tvec"])) * 1000), rotation_deg(pose[0], t["rvec"])))
    out.append(_parity_line("parity (bench)", res))
    for d in div:
        out.append(f"     {d}")
    for label, acc in (("device", err_dev), ("python", err_py)):
        if acc:
            out.append(f"   truth error {label}: {len(acc)} poses, t median {quantile([a[0] for a in acc], 0.5):.2f} mm, "
                       f"max {max(a[0] for a in acc):.2f} mm; r median {quantile([a[1] for a in acc], 0.5):.3f} deg")
    return out


#: corner difference (px) counted as equal in the parity test
PARITY_PX = 0.01


def _divergence(name: str, dev: Dict[int, np.ndarray], py: dict, s: dict, truth: dict) -> str:
    """The most divergent corner of a frame, with each side's distance to the true projection."""
    K = np.asarray(s["K"], float).reshape(3, 3)
    dist = np.asarray(s["dist"], float)
    obj = np.asarray(s["corners"], float).reshape(len(s["ids"]), 4, 3)
    worst = None
    for i in set(dev) & set(py["ids"]):
        d = np.abs(dev[i] - py["corners"][i]).max(axis=1)
        j = int(d.argmax())
        if worst is None or d[j] > worst[0]:
            proj, _ = cv2.projectPoints(obj[s["ids"].index(i)], np.asarray(truth["rvec"], float),
                                        np.asarray(truth["tvec"], float), K, dist)
            proj = proj.reshape(4, 2)[j]
            worst = (float(d[j]), i, j, float(np.linalg.norm(py["corners"][i][j] - proj)),
                     float(np.linalg.norm(dev[i][j] - proj)))
    if worst is None:
        return f"{Path(name).name}: no common marker"
    return (f"{Path(name).name}: id {worst[1]} corner {worst[2]} differs {worst[0]:.3f} px; "
            f"to truth python {worst[3]:.3f} px, device {worst[4]:.3f} px")


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
            + (f"; corners max {max(corner):.4f} px, within {PARITY_PX} px on "
               f"{sum(1 for x in corner if x <= PARITY_PX)}/{len(corner)}" if corner else "")
            + (f"; pose max {max(t):.3f} mm, {max(r):.4f} deg over {len(t)}" if t else "")
            + (f"; pose on one side only {one}" if one else ""))


def _plugin_abi(run: Path) -> int:
    """The plugin ABI a run used, from run.json ``plugin`` ("eyelab_aruco abi N; ..."); 0 when unknown."""
    try:
        meta = json.loads((Path(run) / "run.json").read_text(encoding="utf-8"))
        return int(str(meta.get("plugin", "")).split("abi", 1)[1].split(";", 1)[0])
    except (OSError, ValueError, IndexError):
        return 0


def high_pass(t: np.ndarray, window: int = 9) -> np.ndarray:
    """``t`` (n x 3) minus its running median over ``window`` samples: frame-to-frame noise without slow head drift."""
    t = np.asarray(t, float)
    h = window // 2
    med = np.array([np.median(t[max(0, i - h):i + h + 1], axis=0) for i in range(len(t))])
    return t - med


def _render_margin(fr: List[dict], rows: List[dict], refresh_hz: float) -> Optional[str]:
    """Late Unity frames split by whether the detection worker was busy at that frame (run 2: busy is the cause)."""
    busy = sorted((int(r["start_mono_ns"]), int(r["end_mono_ns"])) for r in rows)
    if not busy or not fr or "mono_ns" not in fr[0]:
        return None
    starts = np.array([b[0] for b in busy])
    ends = np.array([b[1] for b in busy])
    limit = 1000 / refresh_hz * 1.25
    n_busy = n_idle = late_busy = late_idle = 0
    for f in fr[1:]:
        dt, m = _f(f["dt_ms"]), int(f["mono_ns"])
        if dt is None:
            continue
        k = int(np.searchsorted(starts, m, side="right")) - 1
        is_busy = k >= 0 and ends[k] >= m
        late = dt > limit
        if is_busy:
            n_busy, late_busy = n_busy + 1, late_busy + late
        else:
            n_idle, late_idle = n_idle + 1, late_idle + late
    n = n_busy + n_idle
    if n == 0:
        return None
    pct = lambda a, b: f"{a / b * 100:.1f} %" if b else "-"
    return (f"     render: worker busy on {n_busy / n * 100:.0f} % of Unity frames; late when busy {pct(late_busy, n_busy)} "
            f"({n_busy}), idle {pct(late_idle, n_idle)} ({n_idle})")


def ambiguity(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> Optional[Tuple[float, float]]:
    """IPPE (ratio, separation deg) for a planar point set, as ``LStructureDetector._solve_board_pose`` computes it."""
    from pose_estimator import rotation_angle_between
    n, rvecs, _t, errs = cv2.solvePnPGeneric(np.asarray(obj, np.float64), np.asarray(img, np.float64), K, dist,
                                              flags=cv2.SOLVEPNP_IPPE)
    if n < 2:
        return None
    e = np.asarray(errs, np.float64).reshape(-1)
    return float(e[1] / max(e[0], 1e-9)), float(rotation_angle_between(rvecs[0], rvecs[1]))


def _ambiguity_line(sel: List[dict], corners: Dict[str, Dict[int, np.ndarray]], s: dict, device_valid: bool) -> Optional[str]:
    """Share of poses PoseLock would call ambiguous, recomputed from the device's corners; device ratio checked if valid."""
    from pose_lock import PoseLockConfig
    cfg = PoseLockConfig()
    K = np.asarray(s["K"], float).reshape(3, 3)
    dist = np.asarray(s["dist"], float)
    obj = dict(zip(s["ids"], np.asarray(s["corners"], float).reshape(len(s["ids"]), 4, 3)))
    n = amb = 0
    ratio_diff: List[float] = []
    for r in sel:
        if r["pose_ok"] != "1" or r.get("planar") != "1":
            continue
        cs = corners.get(r["seq"], {})
        ids = [i for i in cs if i in obj]
        if not ids:
            continue
        a = ambiguity(np.concatenate([obj[i] for i in ids]), np.concatenate([cs[i] for i in ids]), K, dist)
        if a is None:
            continue
        n += 1
        amb += a[0] < cfg.min_ambiguity_ratio and a[1] > cfg.ambiguity_min_separation_deg
        dev = _f(r.get("amb_ratio"))
        if device_valid and dev is not None:
            ratio_diff.append(abs(dev - a[0]) / max(a[0], 1e-9))
    if n == 0:
        return None
    line = (f"     ambiguous (IPPE ratio < {cfg.min_ambiguity_ratio:g}, > {cfg.ambiguity_min_separation_deg:g} deg apart): "
            f"{amb}/{n} planar poses ({amb / n * 100:.0f} %)")
    if ratio_diff:
        line += f"; device ratio vs Python max {max(ratio_diff) * 100:.2f} %"
    elif not device_valid:
        line += "; device ratio not valid (plugin abi < 2)"
    return line


def live_report(run: Path, set_dir: Path = DEFAULT_SET, refresh_hz: float = 60.0) -> List[str]:
    run = Path(run)
    rows = load_csv(run / "live.csv")
    if not rows:
        return ["   live: no live.csv"]
    frames = load_csv(run / "frames.csv")
    s_amb = load_set(set_dir)
    live_corners = _corners_by_frame(load_csv(run / "live_corners.csv"), "seq") if s_amb is not None else {}
    device_ratio_valid = _plugin_abi(run) >= 2
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
        arrive = [(int(r["arrival_mono_ns"]) - int(r["cam_ts"])) / 1e6 for r in sel]
        eye_ms = [x for x in (_f(f.get("eye_ms")) for f in fr) if x is not None and x > 0]
        out.append(f"     age = arrival {_q(arrive)} + wait {_q(wait)} + processing {_q([_f(r['total_ms']) for r in sel])} ms"
                   + (f"; main-thread Eye work {_q(eye_ms)} ms on {len(eye_ms)} frames" if eye_ms else ""))
        margin = _render_margin(fr, rows, refresh_hz)
        if margin:
            out.append(margin)
        if s_amb is not None:
            amb = _ambiguity_line(sel, live_corners, s_amb, device_ratio_valid)
            if amb:
                out.append(amb)
        still = [_pose(r) for r in sel if r["pose_ok"] == "1"]
        if step.endswith("_still") and len(still) > 5:
            tv = np.array([p[1] for p in still]) * 1000
            out.append(f"     still: {len(still)} poses, t std {np.linalg.norm(tv.std(axis=0)):.2f} mm, "
                       f"distance median {np.median(np.linalg.norm(tv, axis=1)):.0f} mm, rms {_q([_f(r['rms_px']) for r in sel])} px"
                       f"; high-pass std {np.linalg.norm(high_pass(tv).std(axis=0)):.2f} mm (head drift removed, "
                       f"wrong IPPE branches included)")
    s = s_amb
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
        blind = [r for r in saved if int(r["markers"] or 0) == 0 and (run / r["saved"]).exists()]
        if blind:
            flipped = sum(1 for r in blind if len(_detect_ids(read_pgm(run / r["saved"])[::-1].copy(), s["ids"])) >= 2)
            if flipped > len(blind) // 2:
                out.append(f"   WARNING: on {flipped}/{len(blind)} saved frames without markers, plate A is found once "
                           f"the rows are flipped: the frames reached the detector bottom-up")
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
