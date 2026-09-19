# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **AR freeze diagnostics** (`python/ar_watchdog.py`,
  [`docs/ar-freeze-diagnostics.md`](docs/ar-freeze-diagnostics.md)). `Start AR`
  can render a couple of frames and then leave the GUI black and unresponsive
  with nothing in the console or the session log, because no exception is ever
  raised - the tk main thread simply stops. A watchdog thread now samples a
  liveness stamp that the main thread refreshes and writes
  `.logs/ar_debug_<ts>.jsonl`: `stall` records naming the AR stage the main
  thread was in, with every thread's Python stack, plus per-stage timings and
  capture-thread health. A `faulthandler` timer, re-armed only while the main
  thread is healthy, dumps all stacks to `.logs/ar_stacks_<ts>.log` during a
  hard freeze, when no Python code can run at all. `ThreadedCapture` gained
  `stats()` (plain counters written by the reader thread, no lock) and
  `ArucoPipeline` gained `capture_stats()`. Purely passive: one daemon thread
  and two files, no runtime behaviour change.
- `python/test_ar_watchdog.py` (8 tests) and `ThreadedCaptureStatsTests` in
  `python/test_pose_estimator.py`.

### Fixed

- **Opening a camera froze the GUI, and a silent camera was read in a tight
  loop.** Diagnosed from `.logs/ar_debug_20260919_113447.jsonl`; details and
  the raw numbers in
  [`docs/ar-freeze-diagnostics.md`](docs/ar-freeze-diagnostics.md).
  - `cv2.VideoCapture(0, CAP_DSHOW)` blocked its caller for up to 32 s, and
    every call site was on the tkinter main thread: startup camera scan
    (14.7 s), `CalibrationWindow` (41.2 s), `Start AR` (32.2 s). The window
    could not repaint, which is the "black screen / not responding" report.
    Camera opens and index scans now run on a worker thread
    (`camera_utils.CameraOpener`, `camera_utils.CameraScanner`); the GUI shows
    the elapsed seconds, stays usable, and can cancel. A capture that arrives
    after a cancel is released instead of leaked.
  - A backend that reports `isOpened()` and then delivers nothing is no longer
    accepted: `camera_utils.open_camera_result()` requires a real frame within
    `WARMUP_S` and otherwise falls through to the next backend (DSHOW -> MSMF),
    reporting each attempt so the GUI can say *why* a camera failed.
    `EYELAB_CAPTURE_BACKEND=dshow|msmf|any` forces one backend.
  - `ThreadedCapture._reader` had no pause and no give-up on failed reads:
    70 905 456 failed reads in ~30 s were recorded, starving the tk main
    thread and leaving the webcam needing a replug. It now paces retries
    (`CAPTURE_RETRY_SLEEP_S`) and stops after `CAPTURE_GIVE_UP_S` with an
    error the AR loop surfaces before stopping cleanly.
  - `CalibrationWindow` opens asynchronously and gives up on a camera that
    delivers no frame for 5 s, with the reason on screen, instead of sitting
    on "Waiting for camera frame..." indefinitely. It also releases the device
    on every exit path.
  - `python/test_camera_utils.py` (9 tests) plus reader give-up and
    pre-opened-capture tests in `python/test_pose_estimator.py`.

- **Three silent scale errors** (2026-09-18 audit, findings S1a-S1c). Each made
  every pose wrong by a constant factor while leaving the reprojection residual
  at zero, so no quality gate in the pipeline could detect them.
  - **Marker size.** Default raised from 12 mm to 20 mm to match the printed
    sheet in `output/pdf/` (`generate_markers.MARKER_SIZE_MM`, grid spacing
    16 -> 26 mm). The 12 mm default against 20 mm markers reported depth 40 %
    short. Marker size is now also stored **per structure** in that structure's
    marker config (`defaultMarkerSizeMm`), loaded into the GUI on config load,
    and still overridable per marker via `markerSizeMm`.
  - **Calibration resolution.** `save_calibration()` wrote `image_width` /
    `image_height`; nothing read them back, so a calibration taken at one
    resolution and used at another silently scaled every distance (measured:
    +50 % depth error, 5.5 deg rotation error, 0.71 px reprojection RMS). Added
    `calibrate.read_calibration()` / `CalibrationData` and
    `describe_resolution_mismatch()`; `ArucoPipeline.start()` and
    `webcam_pipeline` now refuse to run on a mismatch. `load_calibration()`
    keeps its old two-tuple signature.
  - **UNV units.** `unv_to_json._parse_units` read `unit_code` and `factors`;
    pyuff emits `units_code`, `length`, `force`, `temp`. Every file therefore
    reported SI with a length factor of 1.0 whatever it declared, and the factor
    was never applied anyway - a Testlab geometry in millimetres was consumed as
    metres, 1000x oversized. `UNVParser.parse()` now reads the real keys and
    scales node coordinates to **metres unconditionally**. Unit code 9
    (USER_DEFINED) registered.
- `unv_to_json._parse_nodes` read `coord_sys` / `disp_coord_sys` for Dataset
  2411; pyuff emits `def_cs` / `disp_cs`, so every node's coordinate system
  silently defaulted to 0 and the CS checks never ran. Both spellings accepted.
- Geometry whose nodes span multiple coordinate systems, or use a cylindrical /
  spherical system, is now refused with an explanatory error instead of being
  read as if it were Cartesian in one frame.
- `python/test_scale_contracts.py` - 16 regression tests pinning all of the above,
  including the "wrong but reprojects perfectly" property that hid them.
- `calibrate.generate_board_image()` stretched the ChArUco board to fill the whole
  printable area, so `--generate` produced squares of ~38.0 x 39.6 mm - neither
  square nor the 25 mm the software reports. It now renders at **true physical
  scale**, centred, and refuses a board that does not fit the page. This does not
  affect calibration accuracy (camera intrinsics are invariant to the target's
  scale - verified: identical fx/fy/cx/cy for declared sizes from 0.4x to 2.0x),
  but it is the only way an operator can confirm their printed board against the
  GUI's ruler check.

### Added

- `calibrate.py --generate` can now write a **PDF** as well as a PNG (chosen by
  file extension; PDF via Pillow, no new dependency) and honours `--square`,
  `--cols` and `--rows`, which it previously ignored. Every sheet carries a
  100 mm ruler line so a rescaled print is caught before it is used.
- `output/pdf/charuco_5x7_25mm_A4.pdf` - 5x7, 25 mm square / 19 mm marker,
  matching the `CHARUCO_SQUARE_M` / `CHARUCO_MARKER_M` defaults.
- `output/pdf/charuco_5x7_35mm_A4.pdf` - same grid at 35 mm square / 26 mm
  marker, nearly filling A4. Preferred for new prints: a larger board fills more
  of the frame, which improves the corner coverage that `.docs/THEORY.md` 3.3
  identifies as a calibration-quality requirement. Calibrate with
  `--square 0.035 --marker 0.026`.
- CI break from OpenCV 5.0 (`opencv-contrib-python` 5.0.0.93, PyPI
  2026-07-02): OpenCV 5 removed the legacy `cv2.aruco.calibrateCameraCharuco`
  used by `gui_calibration.py` and `calibrate.py` (and mocked in
  `test_gui_calibration.py`). Pinned `opencv-contrib-python>=4.8.0,<5.0`;
  lift after migrating to `board.matchImagePoints` + `cv2.calibrateCamera`.
- CI workflow: bumped `actions/checkout@v5` and `actions/setup-python@v6`
  (Node 20 runner deprecation warning).

### Added
- Single-marker pose sustain: `PoseLockConfig.sustain_min_markers` default
  lowered to 1. Planar point sets (one marker, or several coplanar markers)
  are now solved with `SOLVEPNP_IPPE` via `solvePnPGeneric`; both mirror
  solutions of the planar pose ambiguity are computed and the one closest to
  the Kalman prediction is kept (`LStructureDetector._solve_board_pose`,
  `estimate_pose(prior=...)`). Acquisition still requires 3 markers — with no
  prior there is no reliable way to disambiguate a single planar marker.
- Per-marker orientation axes in the AR overlay
  (`draw_overlay(draw_marker_axes=True)`, "Marker axes" checkbox): X red /
  Y green / Z blue drawn on each detected marker to catch upside-down or
  rolled prints.
- `python/gui_geometry_editor.py` - interactive synthetic-geometry editor
  ("Geometry Editor" tab, also runs standalone): click on a working plane
  (XY/XZ/YZ + offset) to insert nodes, drag to rotate, two-click node
  connection into trace lines, per-axis sliders + numeric entries for precise
  positioning, undo, JSON import/export, .unv export, hand-off to the main
  viewer.
- `python/unv_writer.py` - UNV writer (datasets 164 + 2411 + 82 via pyuff),
  inverse of `unv_to_json.py`; greedy segment-to-polyline chaining with
  pen-up separators.
- `python/test_unv_writer.py` - writer round-trip tests through `UNVParser`
  plus editor math tests (click ray, plane intersection, node picking).
- `docs/aruco_marker_best_practices.pdf` - sourced field guide on printing,
  flatness/mounting, marker orientation, size/distance/angle budgets, the
  planar pose ambiguity, multi-marker layout, calibration and lighting.

### Previously added
- `python/pose_lock.py` - pose-lock resilience layer for the board pipeline:
  `PoseLock` state machine (acquire with 3 markers, sustain with 2, coast on
  Kalman prediction up to 0.3 s through full dropouts; full-count poses are
  accepted unconditionally while 2-marker sustain poses pass an adaptive
  reprojection-RMS gate — max(3 px, 2.5x the learned baseline) — plus
  translation/rotation jump gates; `last_reject_reason` surfaces why a pose
  was refused), `MarkerCarryover` (LK optical
  flow carries recently seen marker corners through detection flicker), and
  `RoiRedetector` (projects missing board markers through the predicted pose
  and re-detects in a 2x-upscaled crop for small/distant markers).
- `python/pose_estimator.py` - board-guided recovery of rejected candidates
  via `cv2.aruco.refineDetectedMarkers`; board reprojection RMS reported in
  `PoseResult.rms_reproj_px`; `estimate_pose(min_markers=...)` override;
  `PoseKalmanFilter.coast()` / `predict_measurement()`; `FrameResult` gains
  `lock_state`, `carryover_count`, `refine_recovered_count`,
  `roi_recovered_count`.
- `python/test_pose_lock.py` - unit tests for the lock state machine, rvec
  continuity, Kalman coasting, marker carryover, and ROI re-detection.

### Changed
- `python/eyelab_gui.py` - detection diagnostics show lock state and
  recovered-marker counters; AR status marks coasting poses; hints explain
  acquire-3 / sustain-1 behaviour; new Geometry Editor tab and "Marker axes"
  toggle.

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
