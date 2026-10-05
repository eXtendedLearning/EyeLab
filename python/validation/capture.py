"""Headless capture: runs EyeLab's own pipeline on a validation plate and records it.

The pipeline is built the way ``eyelab_gui`` builds it for AR (board from the
marker correspondences, pose lock, ROI re-detection, same detector presets), so
the recorded pose is the pose the GUI would draw. Nothing here touches the GUI;
PNG encoding runs on the recorder's writer thread.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .recorder import SessionRecorder
from .target import Target, file_digest, target_digest


def build_pipeline(target: Target, calibration_path: str, camera_index: int = 0,
                   tuning: str = "balanced"):
    from pose_estimator import DETECTOR_TUNING_PRESETS, ArucoPipeline
    corrs = target.marker_correspondences()
    if not Path(calibration_path).exists():
        raise FileNotFoundError(calibration_path)
    return ArucoPipeline(
        camera_index=camera_index, calibration_path=str(calibration_path),
        board_correspondences=corrs, marker_size_mm=corrs[0].marker_size_mm,
        marker_size_by_id_mm={c.marker_id: c.marker_size_mm for c in corrs},
        allowed_ids=set(target.marker_ids()), detector_tuning=DETECTOR_TUNING_PRESETS[tuning])


def _pose_dict(p) -> Optional[dict]:
    if p is None:
        return None
    return {"rvec": p.rvec, "tvec": p.tvec, "marker_ids": list(p.marker_ids),
            "coasted": bool(p.coasted), "rms_reproj_px": p.rms_reproj_px}


def _project(obj: np.ndarray, ids, p, K, dist) -> dict:
    # Same call and dtypes as overlay.project_nodes, without its integer rounding.
    if p is None:
        return {}
    uv, _ = cv2.projectPoints(obj.astype(np.float32).reshape(-1, 1, 3),
                              np.asarray(p.rvec, np.float32), np.asarray(p.tvec, np.float32), K, dist)
    return dict(zip(ids, uv.reshape(-1, 2)))


def record_result(rec: SessionRecorder, result, target: Target, K, dist,
                  save_frame: bool = True) -> int:
    import overlay
    ids = target.ids()
    obj = target.array(ids)
    drawn = {}
    if result.pose is not None:
        drawn = {str(k): v for k, v in overlay.project_nodes(
            target.nodes(), K, dist, rvec=result.pose.rvec, tvec=result.pose.tvec).items()}
    frame = None
    if save_frame:
        # The analysis reads greyscale; storing it cuts session size about 3x.
        frame = (cv2.cvtColor(result.frame, cv2.COLOR_BGR2GRAY) if result.frame.ndim == 3
                 else result.frame)
    return rec.record(
        frame, timestamp=time.perf_counter(),
        pose=_pose_dict(result.pose), measured=_pose_dict(result.measured_pose),
        lock_state=result.lock_state, lock_reject=result.lock_reject_reason,
        markers_seen=[m.marker_id for m in result.markers],
        projected=_project(obj, ids, result.pose, K, dist),
        projected_raw=_project(obj, ids, result.measured_pose, K, dist),
        drawn=drawn, timings_ms={k: round(v, 3) for k, v in result.timings_ms.items()})


def open_camera_locked(camera_index: int, focus: Optional[float]):
    """Open the camera the way the GUI does (1280x720) and record its focus state.
    With ``focus`` given, autofocus is turned off and the focus fixed first."""
    from camera_utils import lock_focus, open_camera_result
    res = open_camera_result(camera_index)
    if res.cap is None:
        raise RuntimeError(res.detail)
    if focus is not None:
        state = lock_focus(res.cap, focus)
    else:
        state = {"requested_focus": None,
                 "autofocus": float(res.cap.get(cv2.CAP_PROP_AUTOFOCUS)),
                 "focus": float(res.cap.get(cv2.CAP_PROP_FOCUS))}
    return res.cap, {"index": camera_index, "backend": res.backend, **state}


def git_state() -> dict:
    """Commit and dirty flag of the EyeLab checkout (read-only git calls)."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", "--no-optional-locks", *a], capture_output=True, text=True,
                              timeout=10, cwd=Path(__file__).resolve().parent).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD") or None,
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def session_meta(pipeline, target: Target, target_path, calibration_path, tuning: str,
                 camera: Optional[dict] = None) -> dict:
    from eyelab_version import VERSION_STRING
    return {"eyelab_version": VERSION_STRING, "git": git_state(), "target": target.name,
            "target_path": str(target_path), "target_digest": target_digest(target_path),
            "calibration": str(calibration_path),
            "calibration_digest": file_digest(calibration_path), "camera": camera or {},
            "pipeline": {"tuning": tuning, "lock": asdict(pipeline.lock_config),
                         "roi_redetect": pipeline.roi_redetector is not None,
                         "board_mode": bool(pipeline.uses_structure_board)}}


def run(pipeline, rec_factory, target: Target, n_frames: int, save_every: int = 1,
        preview: bool = False, wait: bool = False, idle_timeout_s: float = 10.0) -> int:
    """Record ``n_frames`` processed frames. ``rec_factory(image_size)`` opens the
    recorder once the first frame's size is known. Returns frames recorded."""
    K, dist = pipeline.camera_matrix, pipeline.dist_coeffs
    if K is None:
        raise RuntimeError("pipeline has no calibration")
    rec, recording, n, last = None, not wait, 0, time.monotonic()
    try:
        while n < n_frames:
            err = pipeline.capture_error()
            if err:
                raise RuntimeError(err)
            result = pipeline.process_frame()
            if result is None:
                if time.monotonic() - last > idle_timeout_s:
                    raise RuntimeError("no frames from the camera")
                time.sleep(0.005)
                continue
            last = time.monotonic()
            if recording:
                if rec is None:
                    h, w = result.frame.shape[:2]
                    rec = rec_factory((w, h))
                record_result(rec, result, target, K, dist, save_frame=(n % save_every == 0))
                n += 1
            if preview:
                key = _show(pipeline, result, target, K, dist, recording, n, n_frames)
                if key in (ord("q"), 27):
                    break
                if key == ord(" ") and not recording:
                    recording = True
    finally:
        if rec is not None:
            rec.close()
        if preview:
            cv2.destroyAllWindows()
    return n


def _show(pipeline, result, target, K, dist, recording, n, n_frames) -> int:
    vis = pipeline.draw_overlay(result, draw_markers=True, draw_axes=True)
    for uv in _project(target.array(target.ids()), target.ids(), result.pose, K, dist).values():
        cv2.circle(vis, (int(round(uv[0])), int(round(uv[1]))), 4, (0, 220, 255), 1, cv2.LINE_AA)
    state = f"REC {n}/{n_frames}" if recording else "SPACE: start recording  Q: quit"
    lock = result.lock_state + (" (coasting)" if result.pose is not None and result.pose.coasted else "")
    cv2.putText(vis, f"{state} | lock {lock} | markers {len(result.markers)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (255, 255, 100), 2,
                cv2.LINE_AA)
    cv2.imshow("EyeLab validation capture", vis)
    return cv2.waitKey(1) & 0xFF
