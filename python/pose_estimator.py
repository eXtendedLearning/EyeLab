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

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from calibrate import load_calibration


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


@dataclass
class FrameResult:
    """Output of one pipeline iteration."""
    frame: np.ndarray                     # BGR image
    gray: np.ndarray                      # grayscale
    markers: list[DetectedMarker] = field(default_factory=list)
    pose: Optional[PoseResult] = None
    fps: float = 0.0
    timestamp: float = 0.0


# ── Constants ─────────────────────────────────────────────────────────────────

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE_M = 0.012   # 12 mm


# ── Threaded capture ──────────────────────────────────────────────────────────

class ThreadedCapture:
    """Non-blocking webcam capture running in a daemon thread."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

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

def preprocess_frame(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement for robust detection under varying lighting."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


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
        return self._frame_counter % self.detect_interval == 0

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


# ── L-structure detector ──────────────────────────────────────────────────────

class LStructureDetector:
    """
    Multi-marker detector for the L-shaped flangia.

    Uses cv2.aruco.Board.matchImagePoints → cv2.solvePnP → solvePnPRefineLM.
    Falls back to single-marker pose when < 3 board markers are visible.
    """

    def __init__(
        self,
        board: Optional[cv2.aruco.Board] = None,
        marker_size_m: float = DEFAULT_MARKER_SIZE_M,
        allowed_ids: Optional[set[int]] = None,
    ):
        self.board = board
        self.marker_size_m = marker_size_m
        self.allowed_ids = allowed_ids

        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        self.det_params = cv2.aruco.DetectorParameters()
        # Tuned for lab conditions
        self.det_params.adaptiveThreshConstant = 10
        self.det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.det_params)

    def detect(
        self, gray: np.ndarray,
    ) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
        """Run ArUco detection, optionally filtering by allowed IDs."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return [], None

        if self.allowed_ids is not None:
            keep = [i for i, mid in enumerate(ids.flatten()) if mid in self.allowed_ids]
            if not keep:
                return [], None
            corners = [corners[i] for i in keep]
            ids = ids[keep]

        return corners, ids

    def estimate_pose(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Optional[PoseResult]:
        """
        Estimate 6-DoF pose.  Uses board-based solvePnP when ≥3 board markers
        are visible; falls back to single-marker IPPE_SQUARE otherwise.
        """
        if ids is None or len(ids) == 0:
            return None

        rvec, tvec = None, None
        ids_flat = ids.flatten().tolist()

        # Board mode
        if self.board is not None:
            obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
            if obj_pts is not None and len(obj_pts) >= 4:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    # Levenberg-Marquardt refinement
                    rvec, tvec = cv2.solvePnPRefineLM(
                        obj_pts, img_pts, camera_matrix, dist_coeffs, rvec, tvec,
                    )

        # Fallback: single marker
        if rvec is None and len(corners) > 0:
            half = self.marker_size_m / 2.0
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
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
        marker_size_mm: float = 12.0,
        allowed_ids: Optional[set[int]] = None,
        use_optical_flow: bool = True,
        optical_flow_interval: int = 3,
        udp_host: Optional[str] = None,
        udp_port: int = 9000,
    ):
        self.camera_index = camera_index
        self.marker_size_m = marker_size_mm / 1000.0
        self.use_optical_flow = use_optical_flow

        # Camera calibration
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        if calibration_path and Path(calibration_path).exists():
            self.camera_matrix, self.dist_coeffs = load_calibration(calibration_path)

        # Board
        board = None
        if board_path and Path(board_path).exists():
            board = load_board_from_yaml(board_path)

        self.l_detector = LStructureDetector(
            board=board,
            marker_size_m=self.marker_size_m,
            allowed_ids=allowed_ids,
        )

        self.kalman = PoseKalmanFilter()
        self.of_tracker = OpticalFlowTracker(detect_interval=optical_flow_interval)

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
        enhanced = preprocess_frame(gray)

        corners: list[np.ndarray] = []
        ids: Optional[np.ndarray] = None

        # Detection or optical-flow tracking
        self.of_tracker.tick()
        if not self.use_optical_flow or self.of_tracker.should_detect():
            corners, ids = self.l_detector.detect(enhanced)
            self.of_tracker.store_detection(enhanced, corners, ids)
        else:
            tracked_corners, tracked_ids = self.of_tracker.track(enhanced)
            if tracked_corners is not None:
                corners = tracked_corners
                ids = tracked_ids

        # Build marker list with per-marker individual poses
        # (tvec = marker centre in camera frame — used by registration)
        markers: list[DetectedMarker] = []
        if ids is not None:
            half = self.l_detector.marker_size_m / 2.0
            obj_pts_single = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
            ids_flat = ids.flatten().tolist()
            for i, mid in enumerate(ids_flat):
                dm = DetectedMarker(marker_id=int(mid), corners=corners[i].reshape(4, 2))
                if self.camera_matrix is not None:
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

        # Board-level pose estimation
        pose: Optional[PoseResult] = None
        if self.camera_matrix is not None and ids is not None and len(ids) > 0:
            pose = self.l_detector.estimate_pose(
                corners, ids, self.camera_matrix, self.dist_coeffs,
            )
            if pose is not None:
                # Kalman smoothing
                pose.rvec, pose.tvec = self.kalman.update(pose.rvec, pose.tvec)

                # UDP
                if self.udp:
                    self.udp.send(pose.rvec, pose.tvec)
        else:
            self.kalman.reset()

        return FrameResult(
            frame=frame,
            gray=enhanced,
            markers=markers,
            pose=pose,
            fps=self._fps,
            timestamp=time.time(),
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
