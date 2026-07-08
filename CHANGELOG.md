# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.1.0] — 2026-07-08

First versioned release of the Phase 1 Webcam MVP: the full pipeline (UNV
ingest, ArUco detection, camera calibration, marker-to-mesh registration,
live AR overlay) runs on a standard webcam in Python + OpenCV, plus the
tracked Unity project baseline for the XREAL bring-up.

### Added
- `python/eyelab_version.py` - single source of truth for the version string,
  shown in the GUI title, CLI banners, and launcher.
- `python/gui_common.py`, `python/gui_calibration.py`, `python/gui_markers.py`
  - GUI split into modules: shared paths/constants, calibration & help
  sub-windows, and marker/correspondence dialogs. `eyelab_gui.py` now hosts
  only the application shell.
- `python/overlay.py` - overlay drawing extracted from the pipeline into its
  own module (`project_nodes`, `wireframe_segments`), with unit tests.
- `python/pose_estimator.py` - marker-orientation support (per-marker normal +
  roll in `marker_config.json`) enabling oriented board pose from multiple
  markers; `ArucoDetectorTuning` dataclass with `strict` / `balanced` /
  `forgiving` presets and live GUI tuning controls; per-frame detection
  diagnostics (raw/accepted/rejected counters, mean marker area).
- `python/eyelab_gui.py` - View menu (workspace tab shortcuts, control-panel
  visibility and per-section toggles), filtered-detector-image tab (FLT),
  scrollable left control panel, configurable AR loop period, detection
  diagnostics panel with health hints and board-spec ruler check; AR/FLT
  rendering is skipped for non-visible tabs.
- `python/eyelab_gui.py` - in-app Help menu with an information center, and a
  step-by-step ArUco/ChArUco calibration wizard with direct actions for board
  generation and live calibration.
- `python/eyelab_gui.py` - bottom-right 3-axis orientation globe in the UNV
  geometry preview; clicking X/Y/Z snaps the model to that axis view.
- `python/calibrate.py` - `--wizard` CLI mode that prints a lightweight
  calibration tutorial even when OpenCV is not available in the active Python
  environment.
- `eyelab_xreal/` - tracked Unity project baseline for the XREAL One Pro + Eye
  port, with Unity-safe Git ignore policy and local XREAL SDK install convention.
- `eyelab_xreal/Assets/EyeLab/` - Phase 0 folder skeleton for Bridge, Frames,
  Pose, Geometry, Calibration, and Scenes.
- `.github/workflows/ci.yml` - CI running ruff (error-level checks) and the
  unit test suite on Python 3.11.
- `.gitattributes` - line-ending normalization (LF in repo) to stop CRLF/LF
  churn between Windows checkouts and Linux agents.

### Fixed
- `python/config/marker_config.json` - marker 0 axis order; markers 4 and 6
  measured positions.
- `python/unv_to_json.py` - use `timezone.utc` instead of `datetime.UTC` so
  the parser runs on Python 3.9/3.10 as documented; previously also replaced
  deprecated `datetime.utcnow()` with a timezone-aware UTC timestamp.
- `python/eyelab_gui.py` - restored the `main()` entry point lost during
  concurrent agent edits; screenshot now logs a note when the AR view is not
  active (stored frame may be stale).
- `python/eyelab_gui.py` - stopped the UNV preview's visual axis reference from
  flipping between sides of Matplotlib's 3D box during model rotation by using a
  stable custom box/grid/axis overlay.
- `README.md`, `CONTEXT.md`, and `docs/tasks/T2.6-implementation-checklist.md`
  - updated the XREAL project layout from the old external `xreal_test` path to
  the in-repo `eyelab_xreal/` project and documented what Unity files belong in
  Git.
- Removed unused imports and placeholder-less f-strings flagged by ruff.

### Phase 1 Webcam MVP (initial implementation)

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
