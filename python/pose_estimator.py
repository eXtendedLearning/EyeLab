#!/usr/bin/env python3
"""
T2.2 — ArUco Detection & Pose Estimation Pipeline (Python / OpenCV).

Core classes:
    ThreadedCapture      — Threaded webcam capture for non-blocking reads.
    PoseEstimator        — 6-DoF pose via solvePnP + Levenberg-Marquardt refinement.
    PoseKalmanFilter     — 12-state constant-velocity Kalman smoother.
    OpticalFlowTracker   — Lucas-Kanade inter-frame marker corner tracking.
    UDPPoseSender        — Streams quaternion + translation over UDP (for Unity bridge).
    LStructureDetector   — Multi-marker board detection on L-shaped flangia.

Designed to be imported by the GUI (eyelab_gui.py) or used standalone for benchmarking.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
import yaml

from calibrate import load_calibration
from camera_utils import open_camera
from pose_lock import (
    LOCK_SEARCHING,
    MarkerCarryover,
    PoseLock,
    PoseLockConfig,
    RoiRedetector,
)
from registration import MarkerCorrespondence, marker_object_corners


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectedMarker:
    marker_id: int
    corners: np.ndarray          # (4, 2) image-space corners
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None


@dataclass
class PoseResult:
    rvec: np.ndarray
    tvec: np.ndarray
    marker_ids: list[int]
    marker_count: int
    timestamp: float = 0.0
    rms_reproj_px: Optional[float] = None
    coasted: bool = False


@dataclass
class FrameResult:
    """Output of one pipeline iteration."""
    frame: np.ndarray                     # BGR image
    gray: np.ndarray                      # grayscale
    markers: list[DetectedMarker] = field(default_factory=list)
    pose: Optional[PoseResult] = None
    fps: float = 0.0
    timestamp: float = 0.0
    raw_marker_count: int = 0
    allowed_marker_count: int = 0
    rejected_count: int = 0
    mean_marker_area_px: float = 0.0
    used_optical_flow: bool = False
    lock_state: str = LOCK_SEARCHING
    lock_reject_reason: Optional[str] = None
    carryover_count: int = 0
    refine_recovered_count: int = 0
    roi_recovered_count: int = 0


# ── Constants ─────────────────────────────────────────────────────────────────

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE_M = 0.012   # 12 mm
MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE = 3


@dataclass(frozen=True)
class ArucoDetectorTuning:
    """Small set of ArUco knobs exposed to the GUI for live tuning."""
    clip_limit: float = 2.5
    adaptive_thresh_win_size_max: int = 37
    adaptive_thresh_win_size_step: int = 6
    adaptive_thresh_constant: float = 8.0
    min_marker_perimeter_rate: float = 0.022
    polygonal_approx_accuracy_rate: float = 0.055
    error_correction_rate: float = 0.70
    corner_refinement_win_size: int = 5
    min_corner_distance_rate: float = 0.04
    min_marker_distance_rate: float = 0.04
    min_distance_to_border: int = 3
    perspective_remove_pixel_per_cell: int = 6
    perspective_remove_ignored_margin_per_cell: float = 0.16
    min_otsu_std_dev: float = 4.0


DETECTOR_TUNING_PRESETS: dict[str, ArucoDetectorTuning] = {
    "strict": ArucoDetectorTuning(
        clip_limit=2.0,
        adaptive_thresh_win_size_max=23,
        adaptive_thresh_win_size_step=10,
        adaptive_thresh_constant=10.0,
        min_marker_perimeter_rate=0.03,
        polygonal_approx_accuracy_rate=0.03,
        error_correction_rate=0.60,
        corner_refinement_win_size=5,
        min_corner_distance_rate=0.05,
        min_marker_distance_rate=0.05,
        min_distance_to_border=3,
        perspective_remove_pixel_per_cell=4,
        perspective_remove_ignored_margin_per_cell=0.13,
        min_otsu_std_dev=5.0,
    ),
    "balanced": ArucoDetectorTuning(),
    "forgiving": ArucoDetectorTuning(
        clip_limit=3.0,
        adaptive_thresh_win_size_max=53,
        adaptive_thresh_win_size_step=4,
        adaptive_thresh_constant=7.0,
        min_marker_perimeter_rate=0.015,
        polygonal_approx_accuracy_rate=0.08,
        error_correction_rate=0.80,
        corner_refinement_win_size=7,
        min_corner_distance_rate=0.03,
        min_marker_distance_rate=0.03,
        min_distance_to_border=2,
        perspective_remove_pixel_per_cell=8,
        perspective_remove_ignored_margin_per_cell=0.20,
        min_otsu_std_dev=3.0,
    ),
}
DEFAULT_ARUCO_DETECTOR_TUNING = DETECTOR_TUNING_PRESETS["balanced"]
ARUCO_PREPROCESS_CLIP_LIMIT = DEFAULT_ARUCO_DETECTOR_TUNING.clip_limit


# ── Threaded capture ──────────────────────────────────────────────────────────

class ThreadedCapture:
    """Non-blocking webcam capture running in a daemon thread."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = open_camera(camera_index, width=width, height=height, fps=30)

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self) -> "ThreadedCapture":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()


# ── CLAHE pre-processing ─────────────────────────────────────────────────────

def preprocess_frame(gray: np.ndarray, clip_limit: float = ARUCO_PREPROCESS_CLIP_LIMIT) -> np.ndarray:
    """Apply CLAHE contrast enhancement for robust detection under varying lighting."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _odd_int_at_least(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def make_detector_parameters(tuning: ArucoDetectorTuning | None = None) -> cv2.aruco.DetectorParameters:
    """Build OpenCV detector parameters from the current ArUco tuning."""
    tuning = tuning or DEFAULT_ARUCO_DETECTOR_TUNING
    params = cv2.aruco.DetectorParameters()

    # Adaptive thresholding is the main recall/precision dial for uneven light.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = _odd_int_at_least(tuning.adaptive_thresh_win_size_max, 3)
    params.adaptiveThreshWinSizeStep = max(1, int(tuning.adaptive_thresh_win_size_step))
    params.adaptiveThreshConstant = float(tuning.adaptive_thresh_constant)

    params.minMarkerPerimeterRate = max(0.001, float(tuning.min_marker_perimeter_rate))
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = max(0.001, float(tuning.polygonal_approx_accuracy_rate))
    params.minCornerDistanceRate = max(0.001, float(tuning.min_corner_distance_rate))
    params.minMarkerDistanceRate = max(0.001, float(tuning.min_marker_distance_rate))
    params.minDistanceToBorder = max(0, int(tuning.min_distance_to_border))

    params.errorCorrectionRate = min(1.0, max(0.0, float(tuning.error_correction_rate)))
    params.markerBorderBits = 1
    params.perspectiveRemovePixelPerCell = max(1, int(tuning.perspective_remove_pixel_per_cell))
    params.perspectiveRemoveIgnoredMarginPerCell = min(
        0.49,
        max(0.0, float(tuning.perspective_remove_ignored_margin_per_cell)),
    )
    params.minOtsuStdDev = max(0.0, float(tuning.min_otsu_std_dev))

    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = max(1, int(tuning.corner_refinement_win_size))
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.02

    return params


def make_forgiving_detector_parameters(
    tuning: ArucoDetectorTuning | None = None,
) -> cv2.aruco.DetectorParameters:
    """Compatibility wrapper for older tests/imports."""
    return make_detector_parameters(tuning)


# ── Kalman filter ─────────────────────────────────────────────────────────────

class PoseKalmanFilter:
    """
    12-state constant-velocity Kalman filter for 6-DoF pose smoothing.

    State:       [tx, ty, tz, rx, ry, rz, vtx, vty, vtz, vrx, vry, vrz]
    Measurement: [tx, ty, tz, rx, ry, rz]
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.kf = cv2.KalmanFilter(12, 6)

        F = np.eye(12, dtype=np.float32)
        for i in range(6):
            F[i, i + 6] = 1.0
        self.kf.transitionMatrix = F

        H = np.zeros((6, 12), dtype=np.float32)
        for i in range(6):
            H[i, i] = 1.0
        self.kf.measurementMatrix = H

        self.kf.processNoiseCov = np.eye(12, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(12, dtype=np.float32)
        self.kf.statePost = np.zeros((12, 1), dtype=np.float32)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def predict_measurement(self) -> tuple[np.ndarray, np.ndarray]:
        """One-step (rvec, tvec) prediction WITHOUT mutating filter state."""
        s = self.kf.statePost
        pred = s[:6] + s[6:12]
        return (
            pred[3:6].reshape(3, 1).astype(np.float64),
            pred[0:3].reshape(3, 1).astype(np.float64),
        )

    def coast(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance the filter one predict-only step (measurement dropout)."""
        pred = self.kf.predict()
        # Roll the prediction into the posterior so repeated coasting advances
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        return (
            pred[3:6].reshape(3, 1).astype(np.float64),
            pred[0:3].reshape(3, 1).astype(np.float64),
        )

    def update(self, rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        meas = np.array([
            tvec[0, 0], tvec[1, 0], tvec[2, 0],
            rvec[0, 0], rvec[1, 0], rvec[2, 0],
        ], dtype=np.float32).reshape(6, 1)

        if not self._initialized:
            self.kf.statePost[:6] = meas
            self._initialized = True

        self.kf.predict()
        s = self.kf.correct(meas)
        return s[3:6].reshape(3, 1), s[0:3].reshape(3, 1)

    def reset(self) -> None:
        self._initialized = False
        self.kf.statePost = np.zeros((12, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(12, dtype=np.float32)


# ── Optical-flow tracker ──────────────────────────────────────────────────────

class OpticalFlowTracker:
    """
    Lucas-Kanade optical flow to track ArUco corners between full detections.

    Full ArUco detection runs every `detect_interval` frames; between those
    frames, corners are tracked with sparse optical flow for lower latency.
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, detect_interval: int = 3):
        self.detect_interval = max(1, detect_interval)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_corners: Optional[np.ndarray] = None
        self._prev_ids: Optional[np.ndarray] = None
        self._frame_counter = 0

    def should_detect(self) -> bool:
        """Return True if a full ArUco detection should run this frame."""
        return self._prev_corners is None or self._frame_counter % self.detect_interval == 0

    def store_detection(
        self,
        gray: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray,
    ) -> None:
        """Cache the latest full detection for optical-flow tracking."""
        self._prev_gray = gray.copy()
        if len(corners) > 0 and ids is not None:
            self._prev_corners = np.vstack([c.reshape(-1, 2) for c in corners]).astype(np.float32)
            self._prev_ids = ids.flatten()
        else:
            self._prev_corners = None
            self._prev_ids = None

    def track(self, gray: np.ndarray) -> tuple[Optional[list[np.ndarray]], Optional[np.ndarray]]:
        """
        Track previously detected corners into the new frame.

        Returns (tracked_corners_per_marker, ids) or (None, None) if tracking fails.
        """
        if self._prev_gray is None or self._prev_corners is None:
            return None, None

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_corners, None, **self.LK_PARAMS,
        )

        if new_pts is None:
            return None, None

        good = status.flatten().astype(bool)
        if not good.any():
            return None, None

        # Rebuild per-marker corner lists (4 corners each)
        n_markers = len(self._prev_ids)
        corners_per_marker: list[np.ndarray] = []
        valid_ids: list[int] = []

        for m in range(n_markers):
            start = m * 4
            end = start + 4
            if end > len(good):
                break
            if good[start:end].all():
                corners_per_marker.append(new_pts[start:end].reshape(1, 4, 2))
                valid_ids.append(int(self._prev_ids[m]))

        if not valid_ids:
            return None, None

        # Update state for next frame
        self._prev_gray = gray.copy()
        self._prev_corners = np.vstack([c.reshape(-1, 2) for c in corners_per_marker]).astype(np.float32)
        self._prev_ids = np.array(valid_ids)

        return corners_per_marker, np.array(valid_ids).reshape(-1, 1)

    def tick(self) -> None:
        self._frame_counter += 1


# ── UDP pose sender ───────────────────────────────────────────────────────────

class UDPPoseSender:
    """
    Sends pose as a 28-byte packet: [qx, qy, qz, qw, tx, ty, tz] (7 × float32).

    Quaternion is derived from the Rodrigues rotation vector.
    Default target: 127.0.0.1:9000 (for Unity bridge testing).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        R, _ = cv2.Rodrigues(rvec)
        q = _rotation_matrix_to_quaternion(R)
        t = tvec.flatten()
        packet = struct.pack("7f", q[0], q[1], q[2], q[3], t[0], t[1], t[2])
        self._sock.sendto(packet, (self.host, self.port))

    def close(self) -> None:
        self._sock.close()


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


# ── Board loader ──────────────────────────────────────────────────────────────

def load_board_from_yaml(yaml_path: str) -> cv2.aruco.Board:
    """Load a custom non-planar ArUco Board from board_config.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    obj_points = []
    ids = []
    for entry in data["markers"]:
        corners = np.array(entry["corners"], dtype=np.float32)
        obj_points.append(corners)
        ids.append(entry["id"])
    return cv2.aruco.Board(obj_points, dictionary, np.array(ids))


def board_from_correspondences(
    correspondences: list[MarkerCorrespondence],
    default_marker_size_m: float,
    marker_size_by_id_m: Optional[dict[int, float]] = None,
) -> Optional[cv2.aruco.Board]:
    """Build a cv2 ArUco board from marker centres, face normals, and roll angles."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    obj_points: list[np.ndarray] = []
    ids: list[int] = []
    marker_size_by_id_m = marker_size_by_id_m or {}

    for corr in correspondences:
        marker_id = int(corr.marker_id)
        size_m = marker_size_by_id_m.get(marker_id, default_marker_size_m)
        if corr.marker_size_mm is not None:
            size_m = float(corr.marker_size_mm) / 1000.0
        if size_m <= 0:
            continue
        obj_points.append(
            marker_object_corners(corr.unv_position, corr.normal, corr.roll_deg, size_m)
        )
        ids.append(marker_id)

    if not ids:
        return None
    return cv2.aruco.Board(obj_points, dictionary, np.array(ids, dtype=np.int32))


def _ids_from_board(board: Optional[cv2.aruco.Board]) -> Optional[set[int]]:
    """Best-effort extraction of marker IDs from an OpenCV Board."""
    if board is None or not hasattr(board, "getIds"):
        return None
    try:
        ids = board.getIds()
    except Exception:
        return None
    if ids is None:
        return None
    return {int(marker_id) for marker_id in np.asarray(ids).flatten().tolist()}


def _reprojection_rms_px(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
) -> Optional[float]:
    """RMS reprojection error (px) of a solved pose over its point set."""
    try:
        proj, _ = cv2.projectPoints(
            np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3),
            rvec, tvec, camera_matrix, dist_coeffs,
        )
    except cv2.error:
        return None
    proj = proj.reshape(-1, 2)
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    if proj.shape != img.shape or len(proj) == 0:
        return None
    err = np.linalg.norm(proj - img, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def _mean_marker_area_px(corners: list[np.ndarray]) -> float:
    """Average visible marker area in image pixels, useful for detection health."""
    areas: list[float] = []
    for corner in corners:
        pts = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        area = abs(float(cv2.contourArea(pts)))
        if np.isfinite(area):
            areas.append(area)
    return float(np.mean(areas)) if areas else 0.0


# ── L-structure detector ──────────────────────────────────────────────────────

class LStructureDetector:
    """
    Multi-marker detector for the L-shaped flangia.

    Uses cv2.aruco.Board.matchImagePoints → cv2.solvePnP → solvePnPRefineLM.
    Falls back to single-marker pose only when no structure board is configured.
    """

    def __init__(
        self,
        board: Optional[cv2.aruco.Board] = None,
        marker_size_m: float = DEFAULT_MARKER_SIZE_M,
        allowed_ids: Optional[set[int]] = None,
        marker_size_by_id_m: Optional[dict[int, float]] = None,
        board_marker_ids: Optional[set[int]] = None,
        min_board_markers: int = MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE,
        detector_tuning: ArucoDetectorTuning | None = None,
    ):
        self.board = board
        self.marker_size_m = marker_size_m
        self.allowed_ids = allowed_ids
        self.board_marker_ids = (
            {int(marker_id) for marker_id in board_marker_ids}
            if board_marker_ids is not None
            else _ids_from_board(board)
        )
        self.min_board_markers = max(1, int(min_board_markers))
        self.marker_size_by_id_m = {
            int(marker_id): float(size_m)
            for marker_id, size_m in (marker_size_by_id_m or {}).items()
        }
        self.detector_tuning = detector_tuning or DEFAULT_ARUCO_DETECTOR_TUNING
        self.last_rejected_count = 0
        self.last_raw_marker_count = 0
        self.last_allowed_marker_count = 0
        self.last_refine_recovered_count = 0

        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        self._rebuild_detector()

    def _rebuild_detector(self) -> None:
        self.det_params = make_detector_parameters(self.detector_tuning)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.det_params)

    def set_detector_tuning(self, tuning: ArucoDetectorTuning) -> None:
        self.detector_tuning = tuning
        self._rebuild_detector()

    def set_allowed_ids(self, allowed_ids: Optional[set[int]]) -> None:
        self.allowed_ids = allowed_ids

    def detect(
        self,
        gray: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
        """
        Run ArUco detection, optionally filtering by allowed IDs.

        When a board is configured, rejected candidates are re-examined with
        cv2.aruco's board-guided refinement: markers whose decode narrowly
        failed are recovered using the geometry of the markers already found.
        """
        corners, ids, rejected = self.detector.detectMarkers(gray)
        self.last_refine_recovered_count = 0

        if (
            self.board is not None
            and ids is not None and len(ids) > 0
            and rejected is not None and len(rejected) > 0
        ):
            before = int(len(ids))
            try:
                corners, ids, rejected, _recovered = (
                    self.detector.refineDetectedMarkers(
                        gray, self.board, corners, ids, rejected,
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coeffs,
                    )
                )
            except cv2.error:
                pass
            if ids is not None:
                self.last_refine_recovered_count = int(len(ids)) - before

        self.last_rejected_count = len(rejected) if rejected is not None else 0
        self.last_raw_marker_count = 0 if ids is None else int(len(ids))
        self.last_allowed_marker_count = 0
        if ids is None or len(ids) == 0:
            return [], None

        if self.allowed_ids is not None:
            keep = [i for i, mid in enumerate(ids.flatten()) if mid in self.allowed_ids]
            if not keep:
                return [], None
            corners = [corners[i] for i in keep]
            ids = ids[keep]

        self.last_allowed_marker_count = int(len(ids))
        return corners, ids

    def marker_size_for_id(self, marker_id: int) -> float:
        """Return the physical marker edge size in metres for one ArUco ID."""
        return self.marker_size_by_id_m.get(int(marker_id), self.marker_size_m)

    def single_marker_object_points(self, marker_id: int) -> np.ndarray:
        """Return square object points for one marker, centered at marker origin."""
        half = self.marker_size_for_id(marker_id) / 2.0
        return np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

    def estimate_pose(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        min_markers: Optional[int] = None,
    ) -> Optional[PoseResult]:
        """
        Estimate 6-DoF pose.  Uses board-based solvePnP when configured board
        corners are available; falls back to single-marker IPPE_SQUARE otherwise.

        `min_markers` overrides the configured board minimum for this call
        (the pose lock relaxes it to 2 once a lock is established).
        """
        if ids is None or len(ids) == 0:
            return None

        min_board = self.min_board_markers if min_markers is None else max(1, int(min_markers))
        rvec, tvec = None, None
        rms_reproj_px: Optional[float] = None
        ids_flat = ids.flatten().tolist()

        # Board mode
        if self.board is not None:
            board_corners = corners
            board_ids = ids
            board_ids_flat = ids_flat
            if self.board_marker_ids is not None:
                keep = [
                    i for i, marker_id in enumerate(ids_flat)
                    if int(marker_id) in self.board_marker_ids
                ]
                if len(keep) < min_board:
                    return None
                board_corners = [corners[i] for i in keep]
                board_ids = ids[keep]
                board_ids_flat = [ids_flat[i] for i in keep]
            elif len(ids_flat) < min_board:
                return None

            try:
                obj_pts, img_pts = self.board.matchImagePoints(board_corners, board_ids)
            except cv2.error:
                obj_pts, img_pts = None, None
            if obj_pts is not None and len(obj_pts) >= min_board * 4:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    # Levenberg-Marquardt refinement
                    rvec, tvec = cv2.solvePnPRefineLM(
                        obj_pts, img_pts, camera_matrix, dist_coeffs, rvec, tvec,
                    )
                    rms_reproj_px = _reprojection_rms_px(
                        obj_pts, img_pts, rvec, tvec, camera_matrix, dist_coeffs,
                    )
            if rvec is None:
                return None
            ids_flat = board_ids_flat

        # Fallback: single marker
        if rvec is None and self.board is None and len(corners) > 0:
            obj_pts = self.single_marker_object_points(ids_flat[0])
            img_pts = corners[0].reshape(4, 1, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                return None

        if rvec is None:
            return None

        return PoseResult(
            rvec=rvec,
            tvec=tvec,
            marker_ids=ids_flat,
            marker_count=len(ids_flat),
            timestamp=time.time(),
            rms_reproj_px=rms_reproj_px,
        )


# ── Full pipeline ─────────────────────────────────────────────────────────────

class ArucoPipeline:
    """
    End-to-end detection + pose pipeline (used by the GUI).

    Integrates: ThreadedCapture → CLAHE → ArUco detection (or optical flow)
    → pose estimation → Kalman → optional UDP.
    """

    def __init__(
        self,
        camera_index: int = 0,
        calibration_path: Optional[str] = None,
        board_path: Optional[str] = None,
        board_correspondences: Optional[list[MarkerCorrespondence]] = None,
        marker_size_mm: float = 12.0,
        allowed_ids: Optional[set[int]] = None,
        marker_size_by_id_mm: Optional[dict[int, float]] = None,
        detector_tuning: ArucoDetectorTuning | None = None,
        use_optical_flow: bool = True,
        optical_flow_interval: int = 3,
        udp_host: Optional[str] = None,
        udp_port: int = 9000,
        lock_config: Optional[PoseLockConfig] = None,
        enable_roi_redetect: bool = True,
    ):
        self.camera_index = camera_index
        self.marker_size_m = marker_size_mm / 1000.0
        marker_size_by_id_m = {
            int(marker_id): float(size_mm) / 1000.0
            for marker_id, size_mm in (marker_size_by_id_mm or {}).items()
        }
        self.use_optical_flow = use_optical_flow
        self.detector_tuning = detector_tuning or DEFAULT_ARUCO_DETECTOR_TUNING

        # Camera calibration
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        if calibration_path and Path(calibration_path).exists():
            self.camera_matrix, self.dist_coeffs = load_calibration(calibration_path)

        # Board
        board = None
        if board_path and Path(board_path).exists():
            board = load_board_from_yaml(board_path)
        if board is None and board_correspondences:
            board = board_from_correspondences(
                board_correspondences,
                self.marker_size_m,
                marker_size_by_id_m=marker_size_by_id_m,
            )
        self.uses_structure_board = board is not None
        board_marker_ids = (
            {int(corr.marker_id) for corr in board_correspondences}
            if board_correspondences
            else None
        )
        if self.uses_structure_board:
            # Board mode detects every frame; interval-skipping optical flow
            # is replaced by MarkerCarryover gap-filling below.
            self.use_optical_flow = False

        self.l_detector = LStructureDetector(
            board=board,
            marker_size_m=self.marker_size_m,
            allowed_ids=allowed_ids,
            marker_size_by_id_m=marker_size_by_id_m,
            board_marker_ids=board_marker_ids,
            detector_tuning=self.detector_tuning,
        )

        self.kalman = PoseKalmanFilter()
        self.of_tracker = OpticalFlowTracker(detect_interval=optical_flow_interval)

        # Pose-lock resilience layer (board mode)
        self.lock_config = lock_config or PoseLockConfig()
        self.pose_lock = PoseLock(self.kalman, self.lock_config)
        self.carryover = MarkerCarryover() if self.uses_structure_board else None
        self.roi_redetector: Optional[RoiRedetector] = None
        if self.uses_structure_board and enable_roi_redetect:
            self.roi_redetector = RoiRedetector(
                board=board,
                detector=self.l_detector.detector,
            )

        # UDP
        self.udp: Optional[UDPPoseSender] = None
        if udp_host:
            self.udp = UDPPoseSender(udp_host, udp_port)

        # Capture
        self._capture: Optional[ThreadedCapture] = None

        # FPS tracking
        self._fps_t0 = time.perf_counter()
        self._fps_count = 0
        self._fps = 0.0

    def start(self) -> None:
        self._capture = ThreadedCapture(self.camera_index)
        if not self._capture.is_opened:
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        self._capture.start()

    def stop(self) -> None:
        if self._capture:
            self._capture.stop()
            self._capture = None
        if self.udp:
            self.udp.close()

    def apply_detector_tuning(self, tuning: ArucoDetectorTuning) -> None:
        self.detector_tuning = tuning
        self.l_detector.set_detector_tuning(tuning)
        if self.roi_redetector is not None:
            # Rebuilding tuning replaces the cv2 detector; keep ROI in sync
            self.roi_redetector.detector = self.l_detector.detector

    def apply_allowed_ids(self, allowed_ids: Optional[set[int]]) -> None:
        self.l_detector.set_allowed_ids(allowed_ids)

    @property
    def is_running(self) -> bool:
        return self._capture is not None and self._capture.is_opened

    def process_frame(self) -> Optional[FrameResult]:
        """Process one frame. Returns None if no frame available."""
        if self._capture is None:
            return None

        frame = self._capture.read()
        if frame is None:
            return None

        # FPS
        self._fps_count += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._fps = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = preprocess_frame(gray, clip_limit=self.detector_tuning.clip_limit)

        corners: list[np.ndarray] = []
        ids: Optional[np.ndarray] = None
        detector_ran = False
        used_optical_flow = False
        carryover_count = 0
        roi_recovered_count = 0

        # Detection or optical-flow tracking
        self.of_tracker.tick()
        if not self.use_optical_flow or self.of_tracker.should_detect():
            detector_ran = True
            corners, ids = self.l_detector.detect(
                enhanced, self.camera_matrix, self.dist_coeffs,
            )
            self.of_tracker.store_detection(enhanced, corners, ids)
        else:
            used_optical_flow = True
            tracked_corners, tracked_ids = self.of_tracker.track(enhanced)
            if tracked_corners is not None:
                corners = tracked_corners
                ids = tracked_ids

        if detector_ran:
            raw_marker_count = self.l_detector.last_raw_marker_count
            allowed_marker_count = self.l_detector.last_allowed_marker_count
            rejected_count = self.l_detector.last_rejected_count
        else:
            raw_marker_count = 0 if ids is None else int(len(ids))
            allowed_marker_count = raw_marker_count
            rejected_count = 0

        # Board-mode resilience: pose-guided ROI recovery + LK carryover
        if self.uses_structure_board:
            corners, ids, roi_recovered_count = self._recover_missing_markers(
                enhanced, corners, ids,
            )
            if self.carryover is not None:
                corners, ids = self.carryover.process(enhanced, corners, ids)
                carryover_count = self.carryover.last_carryover_count
                if carryover_count:
                    used_optical_flow = True

        mean_marker_area_px = _mean_marker_area_px(corners)

        # Build marker list with per-marker individual poses
        # (tvec = marker centre in camera frame — used by registration)
        markers: list[DetectedMarker] = []
        if ids is not None:
            ids_flat = ids.flatten().tolist()
            for i, mid in enumerate(ids_flat):
                dm = DetectedMarker(marker_id=int(mid), corners=corners[i].reshape(4, 2))
                if self.camera_matrix is not None:
                    obj_pts_single = self.l_detector.single_marker_object_points(int(mid))
                    img_pts = corners[i].reshape(4, 1, 2).astype(np.float32)
                    ok, rv, tv = cv2.solvePnP(
                        obj_pts_single, img_pts,
                        self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                    if ok:
                        dm.rvec = rv
                        dm.tvec = tv
                markers.append(dm)

        # Board-level pose estimation (lock-aware)
        pose: Optional[PoseResult] = None
        if self.camera_matrix is not None:
            measured: Optional[PoseResult] = None
            if ids is not None and len(ids) > 0:
                measured = self.l_detector.estimate_pose(
                    corners, ids, self.camera_matrix, self.dist_coeffs,
                    min_markers=(
                        self.pose_lock.min_markers
                        if self.uses_structure_board
                        else None
                    ),
                )
            if self.uses_structure_board:
                # Lock owns smoothing, validation gates, and coasting
                pose = self.pose_lock.process(measured)
            elif measured is not None:
                measured.rvec, measured.tvec = self.kalman.update(
                    measured.rvec, measured.tvec,
                )
                pose = measured
            else:
                self.kalman.reset()
            if pose is not None and self.udp:
                self.udp.send(pose.rvec, pose.tvec)
        else:
            self.pose_lock.reset()

        return FrameResult(
            frame=frame,
            gray=enhanced,
            markers=markers,
            pose=pose,
            fps=self._fps,
            timestamp=time.time(),
            raw_marker_count=raw_marker_count,
            allowed_marker_count=allowed_marker_count,
            rejected_count=rejected_count,
            mean_marker_area_px=mean_marker_area_px,
            used_optical_flow=used_optical_flow,
            lock_state=self.pose_lock.state,
            lock_reject_reason=self.pose_lock.last_reject_reason,
            carryover_count=carryover_count,
            refine_recovered_count=self.l_detector.last_refine_recovered_count,
            roi_recovered_count=roi_recovered_count,
        )

    def _recover_missing_markers(
        self,
        enhanced: np.ndarray,
        corners: list[np.ndarray],
        ids: Optional[np.ndarray],
    ) -> tuple[list[np.ndarray], Optional[np.ndarray], int]:
        """
        Pose-guided ROI re-detection for board markers missing this frame.

        Only runs when the pose lock has a usable prediction and fewer board
        markers than the acquisition minimum are currently visible.
        """
        if (
            self.roi_redetector is None
            or self.camera_matrix is None
            or not self.pose_lock.is_locked
        ):
            return corners, ids, 0

        board_ids = self.l_detector.board_marker_ids or set(
            self.roi_redetector.object_corners_by_id
        )
        present = (
            {int(i) for i in ids.flatten()} if ids is not None and len(ids) else set()
        )
        present_board = present & board_ids
        if len(present_board) >= self.lock_config.acquire_min_markers:
            return corners, ids, 0

        prediction = self.pose_lock.predicted_pose()
        if prediction is None:
            return corners, ids, 0
        pred_rvec, pred_tvec = prediction

        missing = sorted(board_ids - present)
        rec_corners, rec_ids = self.roi_redetector.recover(
            enhanced, missing, pred_rvec, pred_tvec,
            self.camera_matrix, self.dist_coeffs,
        )
        if not rec_ids:
            return corners, ids, 0

        merged_corners = list(corners) + rec_corners
        merged_ids = (
            [int(i) for i in ids.flatten()] if ids is not None and len(ids) else []
        ) + rec_ids
        return (
            merged_corners,
            np.array(merged_ids, dtype=np.int32).reshape(-1, 1),
            len(rec_ids),
        )

    def draw_overlay(
        self,
        result: FrameResult,
        draw_markers: bool = True,
        draw_axes: bool = True,
        axis_length_m: float = 0.02,
    ) -> np.ndarray:
        """Draw detection + pose overlay on a copy of the frame."""
        vis = result.frame.copy()

        if draw_markers and result.markers:
            corners_list = [m.corners.reshape(1, 4, 2) for m in result.markers]
            ids_arr = np.array([m.marker_id for m in result.markers]).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(vis, corners_list, ids_arr)

        if draw_axes and result.pose and self.camera_matrix is not None:
            cv2.drawFrameAxes(
                vis, self.camera_matrix, self.dist_coeffs,
                result.pose.rvec, result.pose.tvec, axis_length_m,
            )

        return vis
