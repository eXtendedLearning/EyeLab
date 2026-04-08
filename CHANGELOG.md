# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 1 Webcam MVP

Initial end-to-end webcam pipeline covering UNV ingest, ArUco detection,
camera calibration, marker-to-mesh registration, and live AR overlay.

#### Environment & tooling (T1.2)
- `python/requirements.txt` — pinned dependency floors (`numpy`, `scipy`,
  `opencv-python`, `opencv-contrib-python`, `pyuff`, `pandas`, `matplotlib`,
  `pyyaml`, `Pillow`).
- `python/test_env.py` — smoke test that verifies every required library
  imports cleanly and that a webcam can be opened.
- `python/calibrate.py` — ChArUco board factory plus `load_calibration` /
  `save_calibration` helpers backed by OpenCV `FileStorage` YAML.
- `python/webcam_pipeline.py` — standalone CLI pipeline with Kalman smoothing,
  axes / wireframe toggles, and frame capture.
- `run_eyelab.bat` — Windows launcher that checks the Python version, creates
  and activates a local `venv/`, installs dependencies, and starts the GUI.

#### UNV parsing (T1.3)
- `python/unv_to_json.py` — `UNVParser` for datasets 2411 / 82 / 2420 / 164,
  with strict displacement coordinate-system cross-reference validation and a
  CLI front end.
- `python/generate_test_unv.py` — synthetic UNV generator (`--minimal`,
  `--multi-cs`, `--missing-cs`, `--large N`) for unit testing.

#### ArUco detection & pose estimation (T2.2)
- `python/pose_estimator.py`:
  - `ThreadedCapture` — daemon-thread frame grabber.
  - `LStructureDetector` — `cv2.aruco.ArucoDetector` wrapper with optional ID
    filtering, board-level `solvePnP` (`SOLVEPNP_ITERATIVE` + `solvePnPRefineLM`)
    and per-marker `IPPE_SQUARE` fallback.
  - `PoseKalmanFilter` — 12-state constant-velocity smoother.
  - `OpticalFlowTracker` — Lucas-Kanade inter-frame corner tracking between
    full detections.
  - `UDPPoseSender` — 28-byte quaternion + translation packet broadcast.
  - `ArucoPipeline` orchestrator with `process_frame()` and `draw_overlay()`.
- `python/board_config.yaml` — multi-face L-structure board layout placeholder.
- `python/generate_markers.py` — DICT_4X4_50, 12 mm marker sheet generator.
  Output files use the generic `aruco01.png`, `aruco02.png`, … naming scheme so
  printed markers can be reused across sessions.

#### Marker-to-mesh registration (T2.3)
- `python/registration.py`:
  - `MarkerCorrespondence` / `RegistrationResult` dataclasses.
  - `SpatialRegistration` — Kabsch / Procrustes solver with reflection
    correction, RMS error, per-marker residuals, condition-number quality
    metric, drift monitoring, and `transform_point` / `transform_points`
    helpers.
  - `load_marker_config` / `save_marker_config` JSON I/O.

#### GUI & AR overlay (T2.4)
- `python/eyelab_gui.py` — Tkinter application that integrates every previous
  module:
  - File menu loads a `.unv`, parses it via `UNVParser`, and renders an
    interactive 3D preview using an embedded matplotlib canvas.
  - Marker management window generates / lists / re-prints `arucoNN` sheets.
  - Camera selection drop-down with live re-probe.
  - ChArUco calibration window with live capture, persistent
    `python/config/camera_params.yaml`, and a status indicator.
  - Correspondence editor + node-picker dialog for assigning marker IDs to UNV
    nodes (visually from the 3D preview or from a node treeview).
  - AR view tab with start/stop toggle, screenshot capture, and a wireframe
    overlay rendered by transforming UNV nodes through the registered Kabsch
    transform and projecting them with `cv2.projectPoints`.
  - Session log panel with timestamped messages.
- `python/wireframe.json` — placeholder geometry for the standalone pipeline.

#### Repository hygiene
- `.gitignore` — excludes `venv/`, `__pycache__/`, `*.pyc`, generated marker
  PNGs, the calibration YAML, screenshots, and recorded video.

---

## [0.0.1] — 2026-03-17

### Added
- Initial commit (project scaffold and README stub).
