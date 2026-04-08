# EyeLab

**EyeLab** is an XR-based spatial localization system designed to assist engineers performing
Experimental Modal Analysis (EMA) impact hammer testing. It overlays the measurement geometry
(parsed from Simcenter Testlab `.unv` files) on a live camera feed using ArUco marker
registration, so the operator can see exactly where each excitation point sits on the physical
structure.

This repository contains the **Phase 1 Webcam MVP** — the full pipeline runs on a standard
webcam in Python + OpenCV. A Unity / XREAL port is planned for Phase 2.

---

## Features

- **UNV parsing** — load Siemens Universal File Format geometry (datasets 2411 / 82 / 2420 / 164)
  via `pyuff`, with displacement coordinate-system validation.
- **3D geometry preview** — interactive matplotlib viewer embedded in the GUI.
- **ArUco marker generation** — DICT_4X4_50, 12 mm markers, named generically
  (`aruco01`, `aruco02`, …) so printed sheets can be reused across sessions.
- **ChArUco camera calibration** — 5×7 board, persistent OpenCV YAML output.
- **Marker-to-mesh registration** — Kabsch / Procrustes rigid alignment between detected
  markers (camera frame) and assigned UNV nodes, with RMS / per-marker / condition-number
  quality metrics.
- **Live AR overlay** — webcam feed with the wireframe projected through the registered
  transform, screenshot capture, session log.
- **Pose estimation** — board-level `solvePnP` (`SOLVEPNP_ITERATIVE` + LM refinement) with
  per-marker `IPPE_SQUARE` fallback, optional Kalman smoothing, optical-flow tracking
  between full detections, optional UDP pose broadcast for downstream consumers.

---

## Repository layout

```
EyeLab/
├── run_eyelab.bat           Windows launcher: checks Python, creates venv, installs deps, starts GUI
├── python/
│   ├── requirements.txt     Pinned dependency floors
│   ├── eyelab_gui.py        Tkinter GUI — main entry point
│   ├── pose_estimator.py    ArUco detection + pose estimation pipeline
│   ├── registration.py      Kabsch-based UNV ↔ camera registration
│   ├── calibrate.py         ChArUco calibration helpers (load / save / board factory)
│   ├── webcam_pipeline.py   Standalone CLI pipeline (no GUI)
│   ├── generate_markers.py  Printable ArUco sheet generator
│   ├── unv_to_json.py       UNV → JSON converter (CLI)
│   ├── generate_test_unv.py Synthetic UNV file generator (for testing)
│   ├── test_env.py          Dev-environment smoke test
│   ├── board_config.yaml    Multi-face ArUco board layout
│   └── wireframe.json       Placeholder geometry for the standalone pipeline
└── .docs/                   Project specification & per-task design notes (not tracked)
```

---

## Quick start (Windows)

1. Install **Python 3.9+** and ensure it is on your `PATH`.
2. Double-click `run_eyelab.bat` (or run it from a terminal in the repo root).
   The script will:
   - verify the Python version,
   - create a `venv/` if one does not already exist,
   - `pip install -r python/requirements.txt`,
   - launch the GUI.
3. In the GUI:
   1. **File → Load UNV…** to import a `.unv` geometry file.
   2. **Markers → Generate…** to print an ArUco sheet (or use existing markers).
   3. **Camera → Calibrate…** to run a ChArUco calibration (saved to
      `python/config/camera_params.yaml`).
   4. **Markers → Edit correspondences…** to assign each marker ID to a UNV node.
   5. Switch to the **AR View** tab and toggle the live overlay.

---

## Quick start (manual / cross-platform)

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r python/requirements.txt
python python/eyelab_gui.py
```

Standalone CLI pipeline (no GUI):

```bash
python python/webcam_pipeline.py --camera 0 --calibration python/config/camera_params.yaml
```

---

## Roadmap

**Phase A — Webcam MVP (this repo).** Validate the full UNV → registration → AR overlay
chain on a desktop webcam before any glasses-specific code.

**Phase B — XREAL port.** Move the same pipeline to Unity + XREAL Beam Pro AR glasses.
Core logic is camera-agnostic by design; only the capture / display layer needs to change.

**Phase C — Impact verification & Testlab integration.** Hammer-tip tracking, hit-point
matching, and a Simcenter Testlab COM bridge for live state synchronisation.

---

## Requirements

- Python 3.9+
- A USB webcam (any resolution; 720p+ recommended for marker detection at distance)
- Printed ArUco markers (DICT_4X4_50, 12 mm) and a printed ChArUco board for calibration

Core Python dependencies: `numpy`, `scipy`, `opencv-python`, `opencv-contrib-python`,
`pyuff`, `pandas`, `matplotlib`, `pyyaml`, `Pillow`. See `python/requirements.txt`.

---

## Status

Phase 1 (Webcam MVP) — feature complete. See `CHANGELOG.md` for release history.

## License

To be determined.
