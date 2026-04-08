#!/usr/bin/env python3
"""
EyeLab Phase 1 Webcam MVP Pipeline.

Runs in real-time on a standard webcam:
  1. ArUco marker detection (DICT_4X4_50)
  2. Pose estimation via custom non-planar cv2.aruco.Board + cv2.solvePnP
  3. Kalman filter for pose smoothing
  4. Wireframe overlay via cv2.projectPoints + cv2.line

Usage:
    # Minimal — single-marker pose estimation (no board config required)
    python webcam_pipeline.py --calibration camera_params.yaml

    # With custom board + wireframe
    python webcam_pipeline.py --calibration camera_params.yaml \\
                              --board board_config.yaml \\
                              --wireframe wireframe.json

    # Save output video
    python webcam_pipeline.py --calibration camera_params.yaml --save output.mp4

Controls (window must be focused):
    Q / ESC  — quit
    K        — toggle Kalman filter on/off
    W        — toggle wireframe overlay on/off
    A        — toggle coordinate-axes overlay on/off
    S        — save current frame as PNG
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibrate import load_calibration


# ── Constants ──────────────────────────────────────────────────────────────────
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
AXIS_LENGTH_MM = 20.0          # drawn coordinate axes length
WIREFRAME_COLOR = (0, 220, 255)
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2


# ── Kalman filter ─────────────────────────────────────────────────────────────

class PoseKalmanFilter:
    """
    6-DoF pose Kalman filter with constant-velocity motion model.

    State  (12): [x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz]
    Measurement (6): [x, y, z, rx, ry, rz]
    """

    def __init__(self, q: float = KALMAN_PROCESS_NOISE, r: float = KALMAN_MEASUREMENT_NOISE):
        self.kf = cv2.KalmanFilter(12, 6)

        # Transition matrix (constant-velocity)
        F = np.eye(12, dtype=np.float32)
        F[0, 6] = F[1, 7] = F[2, 8] = 1.0    # pos += vel * dt  (dt=1 frame)
        F[3, 9] = F[4, 10] = F[5, 11] = 1.0
        self.kf.transitionMatrix = F

        # Measurement matrix: observe positions only
        self.kf.measurementMatrix = np.zeros((6, 12), dtype=np.float32)
        for i in range(6):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov      = np.eye(12, dtype=np.float32) * q
        self.kf.measurementNoiseCov  = np.eye(6,  dtype=np.float32) * r
        self.kf.errorCovPost         = np.eye(12, dtype=np.float32)
        self.kf.statePost            = np.zeros((12, 1), dtype=np.float32)
        self._initialized = False

    def update(self, rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feed a new raw pose measurement; return the smoothed (rvec, tvec)."""
        measurement = np.array(
            [tvec[0, 0], tvec[1, 0], tvec[2, 0],
             rvec[0, 0], rvec[1, 0], rvec[2, 0]],
            dtype=np.float32,
        ).reshape(6, 1)

        if not self._initialized:
            self.kf.statePost[:6] = measurement
            self._initialized = True

        self.kf.predict()
        smoothed = self.kf.correct(measurement)

        t_smooth = smoothed[0:3].reshape(3, 1)
        r_smooth = smoothed[3:6].reshape(3, 1)
        return r_smooth, t_smooth


# ── Board loader ──────────────────────────────────────────────────────────────

def load_board(yaml_path: str) -> cv2.aruco.Board:
    """
    Load a custom non-planar ArUco Board from a YAML config file.

    YAML structure (all lengths in metres):
        markers:
          - id: 0
            corners:   # 3D coordinates of the 4 marker corners (TL, TR, BR, BL)
              - [x, y, z]
              - [x, y, z]
              - [x, y, z]
              - [x, y, z]
          - id: 1
            corners: ...
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    obj_points = []
    ids = []

    for entry in data["markers"]:
        corners = np.array(entry["corners"], dtype=np.float32)
        if corners.shape != (4, 3):
            raise ValueError(
                f"Marker ID {entry['id']}: corners must be shape (4, 3), got {corners.shape}"
            )
        obj_points.append(corners)
        ids.append(entry["id"])

    board = cv2.aruco.Board(obj_points, dictionary, np.array(ids))
    return board


# ── Wireframe loader ──────────────────────────────────────────────────────────

def load_wireframe(json_path: str) -> tuple[dict[int, np.ndarray], list[tuple[int, int]]]:
    """
    Load wireframe from a JSON file (EyeLab format from T1.3 UNV parser).

    JSON structure:
        {
          "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}],
          "traceLines": [[1, 2], [2, 3]]
        }

    Returns:
        nodes  — dict mapping node_id → (3,) float32 array (metres)
        edges  — list of (id_a, id_b) pairs
    """
    with open(json_path) as f:
        data = json.load(f)

    nodes: dict[int, np.ndarray] = {
        n["id"]: np.array([n["x"], n["y"], n["z"]], dtype=np.float32)
        for n in data["nodes"]
    }
    edges: list[tuple[int, int]] = [tuple(e) for e in data["traceLines"]]
    return nodes, edges


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_axes(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length_m: float,
) -> None:
    """Draw XYZ coordinate axes at the board origin."""
    origin = np.zeros((1, 3), dtype=np.float32)
    axes_3d = np.float32([
        [length_m, 0, 0],
        [0, length_m, 0],
        [0, 0, -length_m],  # Z into the board (OpenCV convention)
    ]).reshape(-1, 1, 3)
    all_pts = np.vstack([origin.reshape(1, 1, 3), axes_3d])

    img_pts, _ = cv2.projectPoints(all_pts, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    o = tuple(img_pts[0])
    for i, color in enumerate(AXIS_COLORS):
        cv2.line(frame, o, tuple(img_pts[i + 1]), color, 2, cv2.LINE_AA)


def draw_wireframe(
    frame: np.ndarray,
    nodes: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> None:
    """Project 3D wireframe onto frame and draw edges."""
    # Collect unique node positions
    unique_ids = list({nid for edge in edges for nid in edge} & nodes.keys())
    if not unique_ids:
        return

    pts_3d = np.array([nodes[nid] for nid in unique_ids], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(
        pts_3d.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    id_to_px = {nid: tuple(pts_2d[i]) for i, nid in enumerate(unique_ids)}

    for a_id, b_id in edges:
        if a_id in id_to_px and b_id in id_to_px:
            cv2.line(frame, id_to_px[a_id], id_to_px[b_id],
                     WIREFRAME_COLOR, 1, cv2.LINE_AA)


# ── Pose estimation ───────────────────────────────────────────────────────────

def estimate_pose_board(
    corners: list,
    ids: np.ndarray,
    board: cv2.aruco.Board,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate board pose from detected markers using solvePnP internally."""
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None, None

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None
    return rvec, tvec


def estimate_pose_single(
    corner: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pose of a single square marker using solvePnP."""
    half = marker_size_m / 2.0
    obj_pts = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)
    img_pts = corner.reshape(4, 1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for single marker")
    return rvec, tvec


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> int:
    # Load calibration
    print(f"Loading calibration: {args.calibration}")
    camera_matrix, dist_coeffs = load_calibration(args.calibration)

    # Load optional board config
    board: cv2.aruco.Board | None = None
    if args.board:
        print(f"Loading board config: {args.board}")
        board = load_board(args.board)

    # Load optional wireframe
    wf_nodes: dict | None = None
    wf_edges: list | None = None
    if args.wireframe:
        print(f"Loading wireframe: {args.wireframe}")
        wf_nodes, wf_edges = load_wireframe(args.wireframe)
        print(f"  {len(wf_nodes)} nodes, {len(wf_edges)} edges")

    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    det_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, det_params)

    # Kalman filter (one per pipeline; resets if marker lost)
    kf = PoseKalmanFilter()

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}.", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {actual_w}×{actual_h}")

    # Optional video writer
    writer: cv2.VideoWriter | None = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 30, (actual_w, actual_h))

    # Feature toggles
    use_kalman    = True
    show_wireframe = bool(args.wireframe)
    show_axes      = True

    marker_size_m = args.marker_size / 1000.0  # mm → m

    fps_t0 = time.perf_counter()
    fps_count = 0
    fps_display = 0.0
    frame_idx = 0

    print("\nRunning — press Q or ESC to quit.")
    print("  K — toggle Kalman  |  W — toggle wireframe  |  A — toggle axes  |  S — save frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        fps_count += 1
        now = time.perf_counter()
        if now - fps_t0 >= 1.0:
            fps_display = fps_count / (now - fps_t0)
            fps_count = 0
            fps_t0 = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        rvec_raw = rvec_smooth = None
        tvec_raw = tvec_smooth = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Pose estimation
            if board is not None:
                rvec_raw, tvec_raw = estimate_pose_board(
                    corners, ids, board, camera_matrix, dist_coeffs
                )
            else:
                # Fall back to single-marker pose (first detected)
                try:
                    rvec_raw, tvec_raw = estimate_pose_single(
                        corners[0], marker_size_m, camera_matrix, dist_coeffs
                    )
                except RuntimeError:
                    pass

            if rvec_raw is not None:
                if use_kalman:
                    rvec_smooth, tvec_smooth = kf.update(rvec_raw, tvec_raw)
                else:
                    rvec_smooth, tvec_smooth = rvec_raw, tvec_raw

                # Coordinate axes
                if show_axes:
                    draw_axes(
                        frame, camera_matrix, dist_coeffs,
                        rvec_smooth, tvec_smooth,
                        AXIS_LENGTH_MM / 1000.0,
                    )

                # Wireframe overlay
                if show_wireframe and wf_nodes is not None:
                    draw_wireframe(
                        frame, wf_nodes, wf_edges,
                        camera_matrix, dist_coeffs,
                        rvec_smooth, tvec_smooth,
                    )

                # HUD: pose values
                t = tvec_smooth.ravel()
                cv2.putText(
                    frame,
                    f"T: [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm",
                    (10, actual_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
                )
        else:
            kf._initialized = False  # reset on marker loss

        # Status overlay
        status_parts = [
            f"FPS: {fps_display:.1f}",
            f"Markers: {len(ids) if ids is not None else 0}",
            f"Kalman: {'ON' if use_kalman else 'OFF'}",
        ]
        if wf_nodes:
            status_parts.append(f"Wire: {'ON' if show_wireframe else 'OFF'}")
        cv2.putText(frame, "  |  ".join(status_parts),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 100), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow("EyeLab Webcam MVP", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("k"):
            use_kalman = not use_kalman
            print(f"Kalman filter: {'ON' if use_kalman else 'OFF'}")
        elif key == ord("w"):
            show_wireframe = not show_wireframe
            print(f"Wireframe: {'ON' if show_wireframe else 'OFF'}")
        elif key == ord("a"):
            show_axes = not show_axes
            print(f"Axes: {'ON' if show_axes else 'OFF'}")
        elif key == ord("s"):
            save_path = f"frame_{frame_idx:05d}.png"
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="EyeLab Phase 1 webcam MVP pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--calibration", "-c",
        required=True,
        help="Path to camera_params.yaml produced by calibrate.py",
    )
    parser.add_argument(
        "--board", "-b",
        default=None,
        help="Path to board_config.yaml (custom non-planar board). "
             "If omitted, single-marker pose estimation is used.",
    )
    parser.add_argument(
        "--wireframe", "-w",
        default=None,
        help="Path to wireframe.json (UNV geometry from T1.3 parser). "
             "If omitted, wireframe overlay is disabled.",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--marker-size", type=float, default=12.0,
        help="Physical marker size in mm for single-marker mode (default: 12.0)",
    )
    parser.add_argument(
        "--save", default=None, metavar="FILE",
        help="Save output video to FILE (mp4)",
    )
    return run_pipeline(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
