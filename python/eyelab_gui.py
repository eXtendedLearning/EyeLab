#!/usr/bin/env python3
"""
EyeLab GUI — Phase 1 Webcam MVP.

Integrates all pipeline modules into a single tkinter application:
  - Load & preview UNV geometry (3D interactive plot)
  - Generate / manage ArUco markers (aruco01, aruco02 ...)
  - Camera selection & calibration (with persistent status)
  - Marker-to-mesh positioning (assign markers to UNV nodes visually)
  - Live AR overlay (webcam + wireframe, toggle on/off)
  - Session log, screenshot capture

Usage:
    python eyelab_gui.py
    (or via run_eyelab.bat)
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
from PIL import Image, ImageTk

# Matplotlib for 3D preview (embedded in tkinter)
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401 - registers 3D projection

# EyeLab modules
from calibrate import load_calibration, save_calibration, make_charuco_board, generate_board_image
from camera_utils import list_cameras, open_camera
from eyelab_logger import SessionLogger
from generate_markers import generate_markers, MARKER_SIZE_MM, GRID_SPACING_MM
import overlay
from pose_estimator import (
    ArucoDetectorTuning,
    ArucoPipeline,
    DETECTOR_TUNING_PRESETS,
    ThreadedCapture,
    FrameResult,
    MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE,
)
from registration import (
    AXIS_NORMALS,
    SpatialRegistration,
    MarkerCorrespondence,
    load_marker_config,
    marker_axes_from_normal,
    marker_object_corners,
    normal_label,
    save_marker_config,
)
from unv_to_json import UNVParser

# Alias (module uses lowercase)
# ── Constants ─────────────────────────────────────────────────────────────────

APP_TITLE = "EyeLab — Phase 1 Webcam MVP"
WINDOW_SIZE = "1400x860"
PREVIEW_W, PREVIEW_H = 640, 480
CONFIG_DIR = Path(__file__).parent / "config"
MARKERS_DIR = Path(__file__).parent / "markers"
LOG_DIR = Path(__file__).parent / ".logs"
TEST_ASSETS_DIR = Path(__file__).resolve().parent.parent / "test_assets"
CALIBRATION_FILE = CONFIG_DIR / "camera_params.yaml"
MARKER_CONFIG_FILE = CONFIG_DIR / "marker_config.json"
HAMMER_MARKER_CONFIG_FILE = CONFIG_DIR / "hammer_marker_config.json"
POSITION_UI_UNIT = "cm"
POSITION_UI_SCALE = 100.0


def position_m_to_ui(value_m: float) -> float:
    """Convert stored metre coordinates to the correspondence editor unit."""
    return float(value_m) * POSITION_UI_SCALE


def position_ui_to_m(value_ui: float) -> float:
    """Convert correspondence editor coordinates back to stored metres."""
    return float(value_ui) / POSITION_UI_SCALE


def format_position_ui(value_m: float) -> str:
    return f"{position_m_to_ui(value_m):.2f}"


def marker_up_label(normal: np.ndarray | list[float] | tuple[float, float, float] | str, roll_deg: float) -> str:
    if isinstance(normal, str):
        normal = AXIS_NORMALS.get(normal, AXIS_NORMALS["+Z"])
    _, up_axis, _ = marker_axes_from_normal(normal, roll_deg)
    return normal_label(up_axis)

HELP_TOPICS = {
    "Quick start": (
        "1. Select the camera in the Camera panel.\n"
        "2. Calibrate the camera with a printed ChArUco board.\n"
        "3. Load a UNV file and inspect it in the 3D Preview tab.\n"
        "4. Generate or load ArUco marker images and print them at 100% scale.\n"
        "5. Link marker IDs to UNV nodes in Edit correspondences.\n"
        "6. Start AR and check that the wireframe follows the physical structure.\n\n"
        "Good results depend on three things: a camera calibration with low RMS error, "
        "markers printed at the physical size entered in EyeLab, and correct marker-to-node "
        "correspondences."
    ),
    "UNV geometry viewer": (
        "Load UNV accepts Siemens/Testlab .unv or .uff geometry files. EyeLab parses nodes, "
        "trace lines, coordinate systems, and units, then shows the nodes and wireframe in "
        "the 3D preview.\n\n"
        "Drag in the plot to rotate the model. The bottom-right orientation globe stays pinned "
        "to the screen and shows the current X/Y/Z direction. Click an arrow on that globe to "
        "snap the model to the X, Y, or Z view."
    ),
    "Markers and registration": (
        "EyeLab uses printed ArUco markers as known physical anchors. Every marker used for "
        "registration needs a matching UNV pose: centre position, outward face normal, in-plane "
        "roll, and physical size. Use Edit correspondences to assign marker numbers such as "
        "aruco01 to a UNV node or a measured XYZ position.\n\n"
        "Use at least 3 non-collinear correspondences. Four or more are better because EyeLab "
        "can report residual error and is less sensitive to one bad point. One oriented marker "
        "can still define a pose, but it gives you less redundancy."
    ),
    "Camera calibration": (
        "Camera calibration estimates the camera matrix and lens distortion. EyeLab stores the "
        "result in python/config/camera_params.yaml and uses it for marker pose and wireframe "
        "projection.\n\n"
        "This step uses the printed ChArUco calibration board held in front of the camera. It is "
        "not the same thing as placing the ArUco markers on the physical structure.\n\n"
        "Aim for RMS reprojection error below 1.0 px. If the error is higher, capture more varied "
        "views of the board: near, far, tilted, left/right, and close to the image corners."
    ),
    "AR overlay": (
        "Start AR opens the selected camera and runs ArUco detection. With calibration loaded and "
        "registration solved, EyeLab projects the UNV wireframe onto the camera frame.\n\n"
        "If the overlay jumps, check that the marker size in millimetres matches the printed marker "
        "size, and that the correspondences point to the real marker locations on the structure."
    ),
    "Troubleshooting": (
        "No camera found: press Refresh, close other apps using the webcam, or try another camera index.\n\n"
        "Markers not detected: improve lighting, keep the full black border visible, and avoid glossy paper.\n\n"
        "Wireframe mirrored or shifted: re-check marker-to-node assignments and units in the UNV file.\n\n"
        "Calibration RMS too high: recapture with more board angles and avoid blurred frames."
    ),
}

ARUCO_CALIBRATION_STEPS = [
    (
        "1. Print the ChArUco board",
        "Generate the ChArUco board image and print it at 100% scale. Do not use 'fit to page'. "
        "The default EyeLab board is 5 x 7 squares, with 25 mm squares and 19 mm ArUco markers."
    ),
    (
        "2. Prepare the camera view",
        "Select the camera in the main Camera panel. Hold the printed ChArUco board in front of "
        "the camera, visible to the lens, not on the test structure. Use even lighting, keep the "
        "paper flat, and make sure the black/white pattern is sharp."
    ),
    (
        "3. Capture varied frames",
        "Open the live calibration window. Press SPACE only when the board corners are detected. "
        "The Capture button does the same thing if keyboard focus is awkward. Capture at least "
        "15 frames with different board positions: center, corners, near, far, and several tilted "
        "angles."
    ),
    (
        "4. Finish and read RMS",
        "Press ESC after the target frame count is reached. EyeLab computes calibration and reports "
        "RMS reprojection error. Below 1.0 px is the normal target; lower is better."
    ),
    (
        "5. Use the saved calibration",
        "EyeLab saves python/config/camera_params.yaml. After calibration, load UNV geometry, assign "
        "structure marker poses in Edit correspondences, and start AR. For each ArUco on the "
        "structure, set centre position, face normal, roll, and physical marker size. If the overlay "
        "is unstable, redo calibration with sharper and more varied frames."
    ),
]


# ── Main application ──────────────────────────────────────────────────────────

class EyeLabApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        MARKERS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Start session logger (captures stdout/stderr/exceptions to .logs/*.jsonl)
        self.session_log = SessionLogger.start(LOG_DIR)

        # Route uncaught Tk callback errors into the session log
        def _tk_callback_exception(exc, val, tb):
            import traceback as _tb
            text = "".join(_tb.format_exception(exc, val, tb))
            self.session_log.error(f"Tk callback exception:\n{text}")
            # Show in GUI log too
            try:
                self.log(f"Tk callback exception: {val}", level="ERROR")
            except Exception:
                pass
        self.root.report_callback_exception = _tk_callback_exception

        # Replace the default tkinter "feather" icon with a custom EyeLab one
        self._app_icon: Optional[ImageTk.PhotoImage] = None
        self._set_app_icon()

        # ── State ─────────────────────────────────────────────────────────
        self.geometry_data: Optional[dict] = None       # parsed UNV JSON
        self.geometry_path: Optional[Path] = None
        self.calibration_loaded = False
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        self.pipeline: Optional[ArucoPipeline] = None
        self.ar_running = False
        self._ar_after_id: Optional[str] = None
        self.fullscreen_win: Optional[tk.Toplevel] = None
        self.fullscreen_label: Optional[tk.Label] = None
        self._fullscreen_photo: Optional[ImageTk.PhotoImage] = None

        self.registration = SpatialRegistration([])
        self.correspondences: list[MarkerCorrespondence] = []
        self.hammer_marker_ids: set[int] = set(range(40, 45))
        self.hammer_marker_size_mm = MARKER_SIZE_MM
        self._camera_indices: list[int] = []
        self._preview_elev = 24.0
        self._preview_azim = -60.0
        self._preview_roll = 0.0
        self._marker_drag_id: Optional[int] = None
        self._marker_dragging = False
        self._marker_drag_editor = None

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_menu()
        self._build_layout()
        self._load_persistent_state()
        self.root.bind("<F11>", lambda _event: self._toggle_fullscreen_ar())

        self.log("EyeLab GUI started.")

    # ══════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load UNV file...", command=self._load_unv)
        file_menu.add_command(label="Load wireframe JSON...", command=self._load_json)
        file_menu.add_separator()
        file_menu.add_command(label="Open log folder", command=self._open_log_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Generate markers...", command=self._show_marker_gen)
        tools_menu.add_command(label="Load markers from directory...", command=self._show_marker_loader)
        tools_menu.add_command(label="Generate hammer markers", command=self._generate_hammer_markers)
        tools_menu.add_command(label="Calibrate camera...", command=self._start_calibration)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Information center...", command=self._show_info_center)
        help_menu.add_command(label="ArUco calibration wizard...", command=self._show_aruco_wizard)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _set_app_icon(self) -> None:
        """
        Replace the default tkinter feather icon. The feather icon is the same
        one used by another tool the user runs, so we draw a small EyeLab icon
        (an eye) in-memory and apply it via iconphoto so we don't need any
        external asset file.
        """
        try:
            size = 64
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            # Outer eye almond shape (two arcs would be ideal; ellipse is good enough)
            d.ellipse((4, 16, 60, 48), outline=(20, 80, 160, 255), width=4,
                      fill=(230, 240, 255, 255))
            # Iris
            d.ellipse((22, 18, 42, 46), fill=(20, 110, 200, 255))
            # Pupil
            d.ellipse((28, 24, 36, 40), fill=(0, 0, 0, 255))
            # Highlight
            d.ellipse((30, 26, 33, 29), fill=(255, 255, 255, 255))
            self._app_icon = ImageTk.PhotoImage(img)
            self.root.iconphoto(True, self._app_icon)
        except Exception as e:
            # Non-fatal — keep going with whatever Tk gives us
            if SessionLogger.get():
                SessionLogger.get().warning(f"Failed to set app icon: {e}")

    def _open_log_folder(self) -> None:
        try:
            import os
            os.startfile(str(LOG_DIR))  # Windows
        except Exception as e:
            messagebox.showinfo("Logs", f"Log directory:\n{LOG_DIR}\n\n({e})")

    def _show_info_center(self) -> None:
        InfoCenterWindow(self.root, self)

    def _show_aruco_wizard(self) -> None:
        ArucoCalibrationWizard(self.root, self)

    def _build_layout(self) -> None:
        # Main paned window: left panel | right panel
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Left panel: controls ──────────────────────────────────────────
        left = ttk.Frame(pw, width=340)
        pw.add(left, weight=0)

        # Camera
        cam_frame = ttk.LabelFrame(left, text="Camera")
        cam_frame.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(cam_frame, text="Device:").grid(row=0, column=0, sticky="w", padx=4)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(cam_frame, textvariable=self.camera_var, width=18, state="readonly")
        self.camera_combo.grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(cam_frame, text="Refresh", command=self._refresh_cameras, width=7).grid(row=0, column=2, padx=2)
        self._refresh_cameras()

        # Calibration status
        cal_frame = ttk.LabelFrame(left, text="Calibration")
        cal_frame.pack(fill=tk.X, padx=4, pady=2)
        self.cal_status_var = tk.StringVar(value="Not loaded")
        ttk.Label(cal_frame, textvariable=self.cal_status_var, wraplength=300).pack(anchor="w", padx=4, pady=2)
        ttk.Button(cal_frame, text="Calibrate (ChArUco)...", command=self._start_calibration).pack(anchor="w", padx=4, pady=2)
        ttk.Button(cal_frame, text="ArUco tutorial...", command=self._show_aruco_wizard).pack(anchor="w", padx=4, pady=2)
        ttk.Button(cal_frame, text="Load calibration file...", command=self._load_calibration_file).pack(anchor="w", padx=4, pady=2)

        # Geometry
        geo_frame = ttk.LabelFrame(left, text="Geometry (UNV)")
        geo_frame.pack(fill=tk.X, padx=4, pady=2)
        self.geo_status_var = tk.StringVar(value="No file loaded")
        ttk.Label(geo_frame, textvariable=self.geo_status_var, wraplength=300).pack(anchor="w", padx=4, pady=2)
        btn_row = ttk.Frame(geo_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_row, text="Load UNV...", command=self._load_unv).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Load JSON...", command=self._load_json).pack(side=tk.LEFT, padx=2)

        # Marker config
        mk_frame = ttk.LabelFrame(left, text="Marker ↔ Mesh Positioning")
        mk_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(mk_frame, text="Marker size (mm):").grid(row=0, column=0, sticky="w", padx=4)
        self.marker_size_var = tk.DoubleVar(value=MARKER_SIZE_MM)
        ttk.Entry(mk_frame, textvariable=self.marker_size_var, width=8).grid(row=0, column=1, padx=4, pady=2)

        ttk.Button(mk_frame, text="Generate markers", command=self._show_marker_gen).grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Button(mk_frame, text="Load markers from folder...", command=self._show_marker_loader).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Button(mk_frame, text="Edit correspondences...", command=self._show_correspondence_editor).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        self.corr_status_var = tk.StringVar(value="0 correspondences")
        ttk.Label(mk_frame, textvariable=self.corr_status_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4)

        # Hammer marker category
        hammer_frame = ttk.LabelFrame(left, text="Hammer ArUco Markers")
        hammer_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(hammer_frame, text="Markers:").grid(row=0, column=0, sticky="w", padx=4)
        self.hammer_ids_var = tk.StringVar(value="41-45")
        ttk.Entry(hammer_frame, textvariable=self.hammer_ids_var, width=14).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(hammer_frame, text="Size (mm):").grid(row=1, column=0, sticky="w", padx=4)
        self.hammer_marker_size_var = tk.DoubleVar(value=MARKER_SIZE_MM)
        ttk.Entry(hammer_frame, textvariable=self.hammer_marker_size_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(hammer_frame, text="Save hammer set", command=self._save_hammer_marker_config).grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Button(hammer_frame, text="Generate hammer markers", command=self._generate_hammer_markers).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        self.hammer_status_var = tk.StringVar(value="Suggested: aruco41-45")
        ttk.Label(hammer_frame, textvariable=self.hammer_status_var, wraplength=300).grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

        # AR controls
        ar_frame = ttk.LabelFrame(left, text="AR Overlay")
        ar_frame.pack(fill=tk.X, padx=4, pady=2)
        self.ar_btn = ttk.Button(ar_frame, text="Start AR", command=self._toggle_ar)
        self.ar_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.screenshot_btn = ttk.Button(ar_frame, text="Screenshot", command=self._take_screenshot, state="disabled")
        self.screenshot_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.fullscreen_btn = ttk.Button(ar_frame, text="Fullscreen", command=self._toggle_fullscreen_ar, state="disabled")
        self.fullscreen_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.ar_fps_var = tk.StringVar(value="")
        ttk.Label(ar_frame, textvariable=self.ar_fps_var).pack(side=tk.LEFT, padx=8)

        self._build_detection_tuning_controls(left)

        # ── Right panel: display area ─────────────────────────────────────
        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: 3D preview
        self.preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_tab, text="3D Preview")
        self._build_3d_preview(self.preview_tab)

        # Tab 2: AR view
        self.ar_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ar_tab, text="AR View")
        self.ar_canvas_label = ttk.Label(self.ar_tab, text="Press 'Start AR' to begin.")
        self.ar_canvas_label.pack(fill=tk.BOTH, expand=True)
        self._ar_photo: Optional[ImageTk.PhotoImage] = None

        # ── Bottom: log ───────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.log_text = tk.Text(log_frame, height=6, state="disabled", wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill=tk.X, padx=2, pady=2)

    def _build_detection_tuning_controls(self, parent: ttk.Frame) -> None:
        tuning = DETECTOR_TUNING_PRESETS["balanced"]
        tune_frame = ttk.LabelFrame(parent, text="Detection Tuning")
        tune_frame.pack(fill=tk.X, padx=4, pady=2)

        self.det_clip_var = tk.DoubleVar(value=tuning.clip_limit)
        self.det_thresh_max_var = tk.IntVar(value=tuning.adaptive_thresh_win_size_max)
        self.det_thresh_step_var = tk.IntVar(value=tuning.adaptive_thresh_win_size_step)
        self.det_thresh_const_var = tk.DoubleVar(value=tuning.adaptive_thresh_constant)
        self.det_min_perim_var = tk.DoubleVar(value=tuning.min_marker_perimeter_rate)
        self.det_poly_var = tk.DoubleVar(value=tuning.polygonal_approx_accuracy_rate)
        self.det_error_var = tk.DoubleVar(value=tuning.error_correction_rate)
        self.det_expected_only_var = tk.BooleanVar(value=True)
        self.det_tuning_status_var = tk.StringVar(value="Balanced")

        fields = (
            ("Contrast", self.det_clip_var, 1.0, 5.0, 0.1),
            ("Thresh max", self.det_thresh_max_var, 3, 73, 2),
            ("Thresh step", self.det_thresh_step_var, 1, 20, 1),
            ("Thresh C", self.det_thresh_const_var, 3.0, 15.0, 0.5),
            ("Min size", self.det_min_perim_var, 0.005, 0.05, 0.001),
            ("Shape tol", self.det_poly_var, 0.02, 0.10, 0.005),
            ("Err corr", self.det_error_var, 0.40, 0.90, 0.05),
        )
        for row, (label, var, from_, to, inc) in enumerate(fields):
            ttk.Label(tune_frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
            ttk.Spinbox(
                tune_frame,
                textvariable=var,
                from_=from_,
                to=to,
                increment=inc,
                width=8,
            ).grid(row=row, column=1, sticky="w", padx=4, pady=1)

        preset_row = ttk.Frame(tune_frame)
        preset_row.grid(row=len(fields), column=0, columnspan=2, sticky="w", padx=2, pady=(4, 1))
        ttk.Button(preset_row, text="Strict", command=lambda: self._set_detection_tuning_preset("strict")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_row, text="Balanced", command=lambda: self._set_detection_tuning_preset("balanced")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_row, text="Forgiving", command=lambda: self._set_detection_tuning_preset("forgiving")).pack(side=tk.LEFT, padx=2)

        apply_row = ttk.Frame(tune_frame)
        apply_row.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="we", padx=2, pady=(2, 4))
        ttk.Checkbutton(apply_row, text="Expected IDs", variable=self.det_expected_only_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(apply_row, text="Apply", command=self._apply_detection_tuning).pack(side=tk.LEFT, padx=2)
        ttk.Label(apply_row, textvariable=self.det_tuning_status_var).pack(side=tk.LEFT, padx=6)

    def _set_detection_tuning_preset(self, preset_name: str) -> None:
        tuning = DETECTOR_TUNING_PRESETS[preset_name]
        self.det_clip_var.set(tuning.clip_limit)
        self.det_thresh_max_var.set(tuning.adaptive_thresh_win_size_max)
        self.det_thresh_step_var.set(tuning.adaptive_thresh_win_size_step)
        self.det_thresh_const_var.set(tuning.adaptive_thresh_constant)
        self.det_min_perim_var.set(tuning.min_marker_perimeter_rate)
        self.det_poly_var.set(tuning.polygonal_approx_accuracy_rate)
        self.det_error_var.set(tuning.error_correction_rate)
        self.det_tuning_status_var.set(preset_name.title())
        if self.pipeline is not None:
            self._apply_detection_tuning()

    def _detection_tuning_from_ui(self) -> ArucoDetectorTuning:
        try:
            return ArucoDetectorTuning(
                clip_limit=float(self.det_clip_var.get()),
                adaptive_thresh_win_size_max=int(self.det_thresh_max_var.get()),
                adaptive_thresh_win_size_step=int(self.det_thresh_step_var.get()),
                adaptive_thresh_constant=float(self.det_thresh_const_var.get()),
                min_marker_perimeter_rate=float(self.det_min_perim_var.get()),
                polygonal_approx_accuracy_rate=float(self.det_poly_var.get()),
                error_correction_rate=float(self.det_error_var.get()),
            )
        except (ValueError, tk.TclError) as e:
            raise ValueError(f"Invalid detection tuning value: {e}") from e

    def _apply_detection_tuning(self) -> None:
        try:
            tuning = self._detection_tuning_from_ui()
        except ValueError as e:
            messagebox.showerror("Detection Tuning", str(e))
            return
        if self.pipeline is not None:
            self.pipeline.apply_detector_tuning(tuning)
            self.pipeline.apply_allowed_ids(self._expected_detection_ids())
        self.det_tuning_status_var.set(
            f"C {tuning.clip_limit:.1f} / T {tuning.adaptive_thresh_win_size_max}"
        )

    def _build_3d_preview(self, parent: ttk.Frame) -> None:
        self.fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self._set_3d_view(self.ax3d)
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("No geometry loaded")
        self._style_3d_axes()
        self.orient_ax = self.fig.add_axes([0.80, 0.05, 0.16, 0.16], projection="3d")
        self.orient_ax.set_navigate(False)
        self._draw_orientation_globe()
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_3d.mpl_connect("button_press_event", self._on_3d_canvas_press)
        self.canvas_3d.mpl_connect("motion_notify_event", self._on_3d_canvas_motion)
        self.canvas_3d.mpl_connect("button_release_event", self._on_3d_canvas_release)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ══════════════════════════════════════════════════════════════════════
    #  Logging
    # ══════════════════════════════════════════════════════════════════════

    def log(self, msg: str, level: str = "INFO") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        # Mirror into the persistent session log
        sl = SessionLogger.get()
        if sl is not None:
            if level == "ERROR":
                sl.error(msg)
            elif level == "WARNING":
                sl.warning(msg)
            elif level == "DEBUG":
                sl.debug(msg)
            else:
                sl.info(msg)

    # ══════════════════════════════════════════════════════════════════════
    #  Camera
    # ══════════════════════════════════════════════════════════════════════

    def _refresh_cameras(self) -> None:
        cams = list_cameras()
        labels = [f"Camera {i}" for i in cams]
        self.camera_combo["values"] = labels
        if labels:
            self.camera_combo.current(0)
        else:
            self.camera_var.set("No camera found")
        self._camera_indices = cams

    def _get_camera_index(self) -> int:
        idx = self.camera_combo.current()
        if idx < 0 or idx >= len(self._camera_indices):
            return 0
        return self._camera_indices[idx]

    # ══════════════════════════════════════════════════════════════════════
    #  Calibration
    # ══════════════════════════════════════════════════════════════════════

    def _load_persistent_state(self) -> None:
        """Load calibration and marker config from disk if they exist."""
        if CALIBRATION_FILE.exists():
            try:
                self.camera_matrix, self.dist_coeffs = load_calibration(str(CALIBRATION_FILE))
                self.calibration_loaded = True
                self.cal_status_var.set(f"Loaded: {CALIBRATION_FILE.name}")
                self.log(f"Calibration loaded from {CALIBRATION_FILE}")
            except Exception as e:
                self.log(f"Failed to load calibration: {e}")

        if MARKER_CONFIG_FILE.exists():
            try:
                self.correspondences = load_marker_config(str(MARKER_CONFIG_FILE))
                self.registration.set_correspondences(self.correspondences)
                self.corr_status_var.set(f"{len(self.correspondences)} correspondences")
                self.log(f"Marker config loaded: {len(self.correspondences)} correspondences")
            except Exception as e:
                self.log(f"Failed to load marker config: {e}")

        self._load_hammer_marker_config()

    def _load_calibration_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select calibration YAML",
            initialdir=str(CONFIG_DIR),
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.camera_matrix, self.dist_coeffs = load_calibration(path)
            self.calibration_loaded = True
            # Copy to config dir for persistence
            import shutil
            shutil.copy2(path, str(CALIBRATION_FILE))
            self.cal_status_var.set(f"Loaded: {Path(path).name}")
            self.log(f"Calibration loaded from {path}")
        except Exception as e:
            messagebox.showerror("Calibration Error", str(e))

    def _start_calibration(self) -> None:
        """Open a calibration window using the live webcam."""
        cam_idx = self._get_camera_index()
        CalibrationWindow(self.root, cam_idx, self._on_calibration_done)

    def _on_calibration_done(
        self,
        cam_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rms: float,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.camera_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.calibration_loaded = True
        image_size = image_size or (1280, 720)
        save_calibration(
            str(CALIBRATION_FILE), cam_matrix, dist_coeffs,
            image_size, rms, 5, 7, 0.025, 0.019,
        )
        self.cal_status_var.set(f"Calibrated — RMS: {rms:.3f} px")
        self.log(f"Calibration complete. RMS: {rms:.4f} px. Saved to {CALIBRATION_FILE}")

    # ══════════════════════════════════════════════════════════════════════
    #  Geometry loading
    # ══════════════════════════════════════════════════════════════════════

    def _load_unv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select UNV file",
            initialdir=str(TEST_ASSETS_DIR),
            filetypes=[("UNV files", "*.unv *.uff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            parser = UNVParser(Path(path), validate_cs=False, verbose=False)
            self.geometry_data = parser.parse()
            self.geometry_path = Path(path)
            n = self.geometry_data["metadata"]["nodeCount"]
            e = self.geometry_data["metadata"]["lineCount"]
            self.geo_status_var.set(f"{Path(path).name}: {n} nodes, {e} edges")
            self.log(f"Loaded UNV: {Path(path).name} ({n} nodes, {e} edges)")
            self._update_3d_preview()
        except Exception as e:
            messagebox.showerror("UNV Parse Error", str(e))
            self.log(f"UNV parse failed: {e}")

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select wireframe JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                self.geometry_data = json.load(f)
            self.geometry_path = Path(path)
            n = len(self.geometry_data.get("nodes", []))
            e = len(self.geometry_data.get("traceLines", []))
            self.geo_status_var.set(f"{Path(path).name}: {n} nodes, {e} edges")
            self.log(f"Loaded JSON: {Path(path).name} ({n} nodes, {e} edges)")
            self._update_3d_preview()
        except Exception as e:
            messagebox.showerror("JSON Error", str(e))

    # ══════════════════════════════════════════════════════════════════════
    #  3D Preview
    # ══════════════════════════════════════════════════════════════════════

    def _update_3d_preview(self) -> None:
        if self.geometry_data is None:
            return

        self.ax3d.clear()
        nodes = self.geometry_data.get("nodes", [])
        edges = self.geometry_data.get("traceLines", [])

        if not nodes:
            self.ax3d.set_title("No nodes in geometry")
            self.canvas_3d.draw()
            return

        node_map = {n["id"]: (n["x"], n["y"], n["z"]) for n in nodes}
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]
        zs = [n["z"] for n in nodes]

        # Scatter nodes
        self.ax3d.scatter(xs, ys, zs, c="steelblue", s=18, depthshade=True)

        # Node labels
        for n in nodes:
            self.ax3d.text(n["x"], n["y"], n["z"], f' {n["id"]}', fontsize=6, color="gray")

        # Draw edges
        for a_id, b_id in edges:
            if a_id in node_map and b_id in node_map:
                a, b = node_map[a_id], node_map[b_id]
                self.ax3d.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                              color="darkorange", linewidth=1.0)

        # Draw marker positions if correspondences exist
        for corr in self.correspondences:
            p = corr.unv_position
            size_m = (corr.marker_size_mm or float(self.marker_size_var.get())) / 1000.0
            corners = marker_object_corners(p, corr.normal, corr.roll_deg, size_m)
            closed = np.vstack([corners, corners[0]])
            self.ax3d.plot(
                closed[:, 0], closed[:, 1], closed[:, 2],
                color="crimson", linewidth=1.8, zorder=10,
            )
            self.ax3d.scatter([p[0]], [p[1]], [p[2]], c="crimson", s=35, marker="o", zorder=11)
            _, up_axis, normal_axis = marker_axes_from_normal(corr.normal, corr.roll_deg)
            arrow_len = max(size_m * 1.8, 0.008)
            n = normal_axis * arrow_len
            up = up_axis * arrow_len
            self.ax3d.quiver(
                p[0], p[1], p[2], n[0], n[1], n[2],
                color="purple", arrow_length_ratio=0.35, linewidth=1.3, normalize=False,
            )
            self.ax3d.quiver(
                p[0], p[1], p[2], up[0], up[1], up[2],
                color="green", arrow_length_ratio=0.35, linewidth=1.3, normalize=False,
            )
            self.ax3d.text(
                p[0], p[1], p[2],
                f" aruco{corr.marker_id + 1:02d} face {normal_label(corr.normal)} up {normal_label(up_axis)}",
                fontsize=7, color="crimson",
            )

        self._apply_equal_3d_limits(xs, ys, zs)
        self._style_3d_axes()
        self._draw_fixed_reference_grid()
        self._set_3d_view(self.ax3d)
        self._draw_orientation_globe()

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        title = self.geometry_path.name if self.geometry_path else "Geometry"
        self.ax3d.set_title(f"{title} — {len(nodes)} nodes, {len(edges)} edges")
        self.canvas_3d.draw()

    # ══════════════════════════════════════════════════════════════════════
    #  3D preview helpers
    # ══════════════════════════════════════════════════════════════════════

    def _set_3d_view(self, axes) -> None:
        try:
            axes.view_init(
                elev=self._preview_elev,
                azim=self._preview_azim,
                roll=self._preview_roll,
            )
        except TypeError:
            axes.view_init(elev=self._preview_elev, azim=self._preview_azim)

    def _remember_3d_view(self) -> None:
        self._preview_elev = float(getattr(self.ax3d, "elev", self._preview_elev))
        self._preview_azim = float(getattr(self.ax3d, "azim", self._preview_azim))
        self._preview_roll = float(getattr(self.ax3d, "roll", self._preview_roll))

    def _begin_marker_drag(self, marker_id: int, editor=None) -> None:
        if self.geometry_data is None:
            messagebox.showwarning("No Geometry", "Load a UNV or JSON file first.")
            return
        self._marker_drag_id = int(marker_id)
        self._marker_dragging = False
        self._marker_drag_editor = editor
        self.notebook.select(self.preview_tab)
        self.log(f"Drag placement armed for aruco{marker_id + 1:02d}.")

    def _nearest_geometry_node_from_event(self, event):
        if self.geometry_data is None or event.inaxes is not self.ax3d:
            return None
        nodes = self.geometry_data.get("nodes", [])
        if not nodes:
            return None

        best_node = None
        best_distance = float("inf")
        proj = self.ax3d.get_proj()
        for node in nodes:
            x2d, y2d, _ = proj3d.proj_transform(node["x"], node["y"], node["z"], proj)
            px, py = self.ax3d.transData.transform((x2d, y2d))
            distance = float(np.hypot(px - event.x, py - event.y))
            if distance < best_distance:
                best_distance = distance
                best_node = node
        if best_distance > 80.0:
            return None
        return best_node

    def _upsert_marker_at_node(self, marker_id: int, node: dict) -> None:
        position = np.array([node["x"], node["y"], node["z"]], dtype=np.float64)
        for corr in self.correspondences:
            if corr.marker_id == marker_id:
                corr.unv_position = position
                corr.node_id = int(node["id"])
                if not corr.description or corr.description.startswith("Node "):
                    corr.description = f"Node {node['id']}"
                break
        else:
            self.correspondences.append(MarkerCorrespondence(
                marker_id=marker_id,
                unv_position=position,
                node_id=int(node["id"]),
                description=f"Node {node['id']}",
                marker_size_mm=float(self.marker_size_var.get()),
            ))
        self._save_correspondences()
        if self._marker_drag_editor is not None:
            try:
                self._marker_drag_editor.refresh()
            except tk.TclError:
                self._marker_drag_editor = None

    def _drag_marker_to_event(self, event) -> bool:
        if self._marker_drag_id is None:
            return False
        node = self._nearest_geometry_node_from_event(event)
        if node is None:
            return False
        self._upsert_marker_at_node(self._marker_drag_id, node)
        return True

    def _on_3d_canvas_release(self, event) -> None:
        if self._marker_dragging:
            marker_id = self._marker_drag_id
            self._drag_marker_to_event(event)
            self._marker_dragging = False
            self._marker_drag_id = None
            if marker_id is not None:
                self.log(f"Drag placement finished for aruco{marker_id + 1:02d}.")
            return
        if event.inaxes is self.ax3d:
            self._remember_3d_view()
            self._draw_orientation_globe()
            self.canvas_3d.draw_idle()

    def _on_3d_canvas_press(self, event) -> None:
        if event.inaxes is self.ax3d and event.button == 1 and self._marker_drag_id is not None:
            self._marker_dragging = self._drag_marker_to_event(event)
            return

        if event.inaxes is not getattr(self, "orient_ax", None) or event.button != 1:
            return

        nearest_axis = None
        nearest_distance = float("inf")
        arrow_tips = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }
        for axis_name, point in arrow_tips.items():
            x2d, y2d, _ = proj3d.proj_transform(*point, self.orient_ax.get_proj())
            px, py = self.orient_ax.transData.transform((x2d, y2d))
            distance = float(np.hypot(px - event.x, py - event.y))
            if distance < nearest_distance:
                nearest_axis = axis_name
                nearest_distance = distance

        if nearest_axis and nearest_distance < 55:
            self._snap_preview_to_axis(nearest_axis)

    def _on_3d_canvas_motion(self, event) -> None:
        if self._marker_dragging:
            self._drag_marker_to_event(event)

    def _snap_preview_to_axis(self, axis_name: str) -> None:
        views = {
            "X": (0.0, -90.0),
            "Y": (0.0, 0.0),
            "Z": (90.0, -90.0),
        }
        self._preview_elev, self._preview_azim = views[axis_name]
        self._preview_roll = 0.0
        self._set_3d_view(self.ax3d)
        self._draw_orientation_globe()
        self.canvas_3d.draw_idle()
        self.log(f"3D preview snapped to {axis_name}-axis view.")

    def _draw_orientation_globe(self) -> None:
        if not hasattr(self, "orient_ax"):
            return

        ax = self.orient_ax
        ax.clear()
        self._set_3d_view(ax)
        ax.set_axis_off()
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.15, 1.15)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        u = np.linspace(0, 2 * np.pi, 18)
        v = np.linspace(0, np.pi, 10)
        xs = 0.68 * np.outer(np.cos(u), np.sin(v))
        ys = 0.68 * np.outer(np.sin(u), np.sin(v))
        zs = 0.68 * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color=(0.55, 0.58, 0.62, 0.35), linewidth=0.4)

        axes = (
            ("X", (1.0, 0.0, 0.0), "crimson"),
            ("Y", (0.0, 1.0, 0.0), "seagreen"),
            ("Z", (0.0, 0.0, 1.0), "royalblue"),
        )
        for label, direction, color in axes:
            dx, dy, dz = direction
            ax.quiver(
                0.0, 0.0, 0.0, dx, dy, dz,
                color=color, arrow_length_ratio=0.22, linewidth=2.0, normalize=False,
            )
            ax.text(
                dx * 1.15, dy * 1.15, dz * 1.15,
                label, color=color, fontsize=8, ha="center", va="center",
            )

    def _apply_equal_3d_limits(self, xs: list[float], ys: list[float], zs: list[float]) -> None:
        """Set stable equal-ish limits so custom grid/axes do not jump on redraw."""
        mins = np.array([min(xs), min(ys), min(zs)], dtype=float)
        maxs = np.array([max(xs), max(ys), max(zs)], dtype=float)
        center = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins))
        if span <= 0:
            span = 1.0
        radius = span * 0.58
        self.ax3d.set_xlim(center[0] - radius, center[0] + radius)
        self.ax3d.set_ylim(center[1] - radius, center[1] + radius)
        self.ax3d.set_zlim(center[2] - radius, center[2] + radius)
        try:
            self.ax3d.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    def _style_3d_axes(self) -> None:
        """Hide mplot3d's dynamic box so axes do not jump between cube sides."""
        self.ax3d.set_axis_off()
        self.ax3d.grid(False)
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["linewidth"] = 0.0

    def _draw_fixed_reference_grid(self) -> None:
        """Draw a stable world-space box, XY grid, and colored axes."""
        x0, x1 = self.ax3d.get_xlim3d()
        y0, y1 = self.ax3d.get_ylim3d()
        z0, z1 = self.ax3d.get_zlim3d()
        z = z0
        grid_color = (0.72, 0.72, 0.72, 0.55)
        box_color = (0.48, 0.50, 0.54, 0.55)

        for x in np.linspace(x0, x1, 9):
            self.ax3d.plot([x, x], [y0, y1], [z, z], color=grid_color, linewidth=0.6, zorder=0)
        for y in np.linspace(y0, y1, 9):
            self.ax3d.plot([x0, x1], [y, y], [z, z], color=grid_color, linewidth=0.6, zorder=0)

        corners = {
            "000": (x0, y0, z0), "100": (x1, y0, z0),
            "010": (x0, y1, z0), "110": (x1, y1, z0),
            "001": (x0, y0, z1), "101": (x1, y0, z1),
            "011": (x0, y1, z1), "111": (x1, y1, z1),
        }
        for a, b in (
            ("000", "100"), ("010", "110"), ("001", "101"), ("011", "111"),
            ("000", "010"), ("100", "110"), ("001", "011"), ("101", "111"),
            ("000", "001"), ("100", "101"), ("010", "011"), ("110", "111"),
        ):
            pa, pb = corners[a], corners[b]
            self.ax3d.plot(
                [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                color=box_color, linewidth=0.7, zorder=0,
            )

        span = max(x1 - x0, y1 - y0, z1 - z0)
        origin = np.array([x0, y0, z], dtype=float)
        axis_len = span * 0.18
        axes = (
            ("X", (axis_len, 0.0, 0.0), "crimson"),
            ("Y", (0.0, axis_len, 0.0), "seagreen"),
            ("Z", (0.0, 0.0, axis_len), "royalblue"),
        )
        for label, direction, color in axes:
            dx, dy, dz = direction
            self.ax3d.quiver(
                origin[0], origin[1], origin[2], dx, dy, dz,
                color=color, arrow_length_ratio=0.12, linewidth=1.4, normalize=False,
            )
            self.ax3d.text(
                origin[0] + dx, origin[1] + dy, origin[2] + dz,
                f" {label}", color=color, fontsize=8,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Marker generation
    # ══════════════════════════════════════════════════════════════════════

    def _show_marker_gen(self) -> None:
        MarkerGenWindow(self.root, self)

    def _format_marker_numbers(self, marker_ids: set[int]) -> str:
        return ", ".join(f"aruco{marker_id + 1:02d}" for marker_id in sorted(marker_ids))

    def _parse_marker_numbers(self, text: str) -> set[int]:
        marker_ids: set[int] = set()
        normalized = text.lower().replace("aruco", "").replace(",", " ").replace(";", " ")
        for token in normalized.split():
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start_num = int(start_text)
                end_num = int(end_text)
                if end_num < start_num:
                    start_num, end_num = end_num, start_num
                numbers = range(start_num, end_num + 1)
            else:
                numbers = (int(token),)

            for number in numbers:
                if number < 1 or number > 50:
                    raise ValueError("Use printed marker numbers 1 through 50.")
                marker_ids.add(number - 1)

        if not marker_ids:
            raise ValueError("Enter at least one hammer marker number.")
        return marker_ids

    def _read_hammer_marker_inputs(self) -> tuple[set[int], float]:
        marker_ids = self._parse_marker_numbers(self.hammer_ids_var.get())
        marker_size_mm = float(self.hammer_marker_size_var.get())
        if marker_size_mm <= 0:
            raise ValueError("Hammer marker size must be greater than zero.")
        return marker_ids, marker_size_mm

    def _compact_marker_numbers(self, marker_ids: set[int]) -> str:
        numbers = sorted(marker_id + 1 for marker_id in marker_ids)
        if not numbers:
            return ""
        ranges: list[str] = []
        start = prev = numbers[0]
        for number in numbers[1:]:
            if number == prev + 1:
                prev = number
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = number
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ", ".join(ranges)

    def _apply_hammer_marker_state(self, marker_ids: set[int], marker_size_mm: float) -> None:
        self.hammer_marker_ids = set(marker_ids)
        self.hammer_marker_size_mm = float(marker_size_mm)
        self.hammer_ids_var.set(self._compact_marker_numbers(self.hammer_marker_ids))
        self.hammer_marker_size_var.set(self.hammer_marker_size_mm)
        self.hammer_status_var.set(
            f"{len(self.hammer_marker_ids)} hammer markers: {self._format_marker_numbers(self.hammer_marker_ids)}"
        )

    def _load_hammer_marker_config(self) -> None:
        if not HAMMER_MARKER_CONFIG_FILE.exists():
            self._apply_hammer_marker_state(self.hammer_marker_ids, self.hammer_marker_size_mm)
            return
        try:
            data = json.loads(HAMMER_MARKER_CONFIG_FILE.read_text(encoding="utf-8"))
            raw_ids = data.get("markerIds")
            if raw_ids is None:
                raw_ids = [int(number) - 1 for number in data.get("markerNumbers", [])]
            marker_ids = {int(marker_id) for marker_id in raw_ids}
            marker_size_mm = float(data.get("markerSizeMm", MARKER_SIZE_MM))
            self._apply_hammer_marker_state(marker_ids, marker_size_mm)
            self.log(f"Hammer marker config loaded: {len(marker_ids)} marker(s)")
        except Exception as e:
            self.log(f"Failed to load hammer marker config: {e}", level="WARNING")
            self._apply_hammer_marker_state(self.hammer_marker_ids, self.hammer_marker_size_mm)

    def _save_hammer_marker_config(self) -> None:
        try:
            marker_ids, marker_size_mm = self._read_hammer_marker_inputs()
        except Exception as e:
            messagebox.showerror("Hammer Marker Error", str(e))
            return

        overlap = marker_ids & self._structure_marker_ids()
        if overlap:
            self.log(
                f"Hammer markers overlap structure correspondences: {self._format_marker_numbers(overlap)}",
                level="WARNING",
            )

        data = {
            "markerIds": sorted(marker_ids),
            "markerNumbers": [marker_id + 1 for marker_id in sorted(marker_ids)],
            "markerSizeMm": marker_size_mm,
        }
        HAMMER_MARKER_CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._apply_hammer_marker_state(marker_ids, marker_size_mm)
        self.log(f"Hammer marker config saved: {len(marker_ids)} marker(s)")

    def _generate_hammer_markers(self) -> None:
        try:
            marker_ids, marker_size_mm = self._read_hammer_marker_inputs()
        except Exception as e:
            messagebox.showerror("Hammer Marker Error", str(e))
            return

        self._save_hammer_marker_config()
        output_dir = MARKERS_DIR / "hammer"
        grid_spacing_mm = max(marker_size_mm + 4.0, marker_size_mm)
        try:
            generate_markers(
                str(output_dir),
                dpi=300,
                marker_ids=sorted(marker_ids),
                add_id_label=True,
                marker_size_mm=marker_size_mm,
                grid_spacing_mm=grid_spacing_mm,
                label_prefix="hammer",
            )
            self.log(f"Hammer markers generated in {output_dir}")
            messagebox.showinfo("Hammer Markers", f"Generated {len(marker_ids)} marker(s) in:\n{output_dir}")
        except Exception as e:
            messagebox.showerror("Hammer Marker Error", str(e))

    def _structure_marker_ids(self) -> set[int]:
        return {corr.marker_id for corr in self.correspondences}

    def _marker_size_by_id_mm(self) -> dict[int, float]:
        return {marker_id: self.hammer_marker_size_mm for marker_id in self.hammer_marker_ids}

    def _expected_detection_ids(self) -> set[int] | None:
        if hasattr(self, "det_expected_only_var") and not self.det_expected_only_var.get():
            return None
        structure_ids = self._structure_marker_ids()
        if not structure_ids:
            return None
        return structure_ids | set(self.hammer_marker_ids)

    # ══════════════════════════════════════════════════════════════════════
    #  Correspondence editor (marker ↔ mesh node)
    # ══════════════════════════════════════════════════════════════════════

    def _show_correspondence_editor(self) -> None:
        CorrespondenceEditor(self.root, self)

    def _save_correspondences(self) -> None:
        save_marker_config(str(MARKER_CONFIG_FILE), self.correspondences)
        self.registration.set_correspondences(self.correspondences)
        self.corr_status_var.set(f"{len(self.correspondences)} correspondences")
        self._update_3d_preview()

    # ══════════════════════════════════════════════════════════════════════
    #  AR Overlay
    # ══════════════════════════════════════════════════════════════════════

    def _toggle_ar(self) -> None:
        if self.ar_running:
            self._stop_ar()
        else:
            self._start_ar()

    def _start_ar(self) -> None:
        if not self.calibration_loaded:
            messagebox.showwarning("No Calibration", "Load or run a camera calibration first.")
            return
        if self.geometry_data is None:
            proceed = messagebox.askyesno(
                "No Geometry Loaded",
                "No geometry is loaded. Start webcam marker preview without a wireframe overlay?",
            )
            if not proceed:
                return
        if self.geometry_data is not None and len(self.correspondences) < 1:
            proceed = messagebox.askyesno(
                "Registration Not Ready",
                "At least one oriented structure marker is needed for the geometry overlay. Start webcam marker preview anyway?",
            )
            if not proceed:
                return
        elif self.geometry_data is not None and len(self.correspondences) < 3:
            self.log(
                "Using fewer than 3 structure markers. Orientation-aware board pose can run, but accuracy improves with multiple markers.",
                level="WARNING",
            )

        cam_idx = self._get_camera_index()
        try:
            detector_tuning = self._detection_tuning_from_ui()
        except ValueError as e:
            messagebox.showerror("Detection Tuning", str(e))
            return
        try:
            self.pipeline = ArucoPipeline(
                camera_index=cam_idx,
                calibration_path=str(CALIBRATION_FILE),
                board_correspondences=self.correspondences,
                marker_size_mm=self.marker_size_var.get(),
                marker_size_by_id_mm=self._marker_size_by_id_mm(),
                allowed_ids=self._expected_detection_ids(),
                detector_tuning=detector_tuning,
            )
            self.pipeline.start()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            return

        self.ar_running = True
        self.ar_btn.configure(text="Stop AR")
        self.screenshot_btn.configure(state="normal")
        self.fullscreen_btn.configure(state="normal")
        self.notebook.select(self.ar_tab)
        self.log("AR overlay started.")
        self._ar_loop()

    def _stop_ar(self) -> None:
        self.ar_running = False
        self._close_fullscreen_ar()
        if self._ar_after_id is not None:
            self.root.after_cancel(self._ar_after_id)
            self._ar_after_id = None
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None
        self.ar_btn.configure(text="Start AR")
        self.screenshot_btn.configure(state="disabled")
        self.fullscreen_btn.configure(state="disabled")
        self.ar_fps_var.set("")
        self.ar_canvas_label.configure(image="", text="AR stopped.")
        self.log("AR overlay stopped.")

    def _ar_loop(self) -> None:
        if not self.ar_running or self.pipeline is None:
            return

        result = self.pipeline.process_frame()
        if result is not None:
            registration_result = None
            vis = self.pipeline.draw_overlay(result, draw_markers=True, draw_axes=True)
            self._draw_marker_roles(vis, result)

            structure_ids = self._structure_marker_ids()
            structure_seen = sum(1 for marker in result.markers if marker.marker_id in structure_ids)
            hammer_seen = sum(1 for marker in result.markers if marker.marker_id in self.hammer_marker_ids)
            pose_seen = result.pose.marker_count if result.pose is not None else 0
            self.registration.clear_detected_positions()
            for m in result.markers:
                if m.marker_id in structure_ids and m.rvec is not None and m.tvec is not None:
                    self.registration.update_detected_position(
                        m.marker_id, m.tvec.flatten()
                    )
            registration_result = self.registration.compute()

            if result.pose:
                t = result.pose.tvec.flatten()
                info = (
                    f"T: [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm"
                    f"  |  Pose markers: {pose_seen}  |  Structure: {structure_seen}  |  Hammer: {hammer_seen}"
                )
                cv2.putText(vis, info, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                if structure_seen:
                    info = (
                        f"Need {MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE}+ structure markers for stable pose"
                        f"  |  Structure: {structure_seen}  |  Hammer: {hammer_seen}"
                    )
                else:
                    info = f"Structure: {structure_seen}  |  Hammer: {hammer_seen}"
                cv2.putText(vis, info, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw wireframe from an oriented board pose, or from legacy centre registration.
            board_pose_ready = (
                self.pipeline is not None
                and self.pipeline.uses_structure_board
                and result.pose is not None
            )
            if self.geometry_data and (board_pose_ready or registration_result is not None):
                self._draw_registered_wireframe(vis, result)

            self._update_ar_display(vis)

            # FPS
            self.ar_fps_var.set(
                f"FPS: {result.fps:.1f} | S {structure_seen} | Pose {pose_seen} | H {hammer_seen}"
            )

            # Store last frame for screenshot
            self._last_ar_frame = vis

        self._ar_after_id = self.root.after(16, self._ar_loop)  # ~60 Hz GUI refresh

    def _marker_role(self, marker_id: int) -> str:
        if marker_id in self.hammer_marker_ids:
            return "hammer"
        if marker_id in self._structure_marker_ids():
            return "structure"
        return "unassigned"

    def _draw_marker_roles(self, vis: np.ndarray, result: FrameResult) -> None:
        styles = {
            "structure": ((70, 220, 70), "STRUCT"),
            "hammer": ((0, 165, 255), "HAMMER"),
            "unassigned": ((190, 190, 190), "ARUCO"),
        }
        for marker in result.markers:
            role = self._marker_role(marker.marker_id)
            color, label_prefix = styles[role]
            corners = marker.corners.astype(int)
            cv2.polylines(vis, [corners], True, color, 2, cv2.LINE_AA)
            x = int(np.min(corners[:, 0]))
            y = int(np.min(corners[:, 1])) - 8
            y = max(18, y)
            label = f"{label_prefix} aruco{marker.marker_id + 1:02d}"
            if role == "hammer" and marker.tvec is not None:
                label += f" {float(np.linalg.norm(marker.tvec)):.2f}m"
            cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 2, cv2.LINE_AA)

    def _resize_rgb_for_box(self, rgb: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        max_w = max(1, int(max_w))
        max_h = max(1, int(max_h))
        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)

    def _update_ar_display(self, vis: np.ndarray) -> None:
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        max_w = max(PREVIEW_W, self.ar_canvas_label.winfo_width())
        max_h = max(PREVIEW_H, self.ar_canvas_label.winfo_height())
        display_rgb = self._resize_rgb_for_box(rgb, max_w, max_h)
        self._ar_photo = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        self.ar_canvas_label.configure(image=self._ar_photo, text="")
        self._update_fullscreen_display(rgb)

    def _fullscreen_is_open(self) -> bool:
        try:
            return self.fullscreen_win is not None and bool(self.fullscreen_win.winfo_exists())
        except tk.TclError:
            return False

    def _toggle_fullscreen_ar(self) -> None:
        if self._fullscreen_is_open():
            self._close_fullscreen_ar()
        else:
            self._open_fullscreen_ar()

    def _open_fullscreen_ar(self) -> None:
        if not self.ar_running:
            self._start_ar()
            if not self.ar_running:
                return

        win = tk.Toplevel(self.root)
        self.fullscreen_win = win
        win.title("EyeLab AR Fullscreen")
        win.configure(bg="black")
        win.attributes("-fullscreen", True)
        win.bind("<Escape>", lambda _event: self._close_fullscreen_ar())
        win.bind("<F11>", lambda _event: self._close_fullscreen_ar())
        win.protocol("WM_DELETE_WINDOW", self._close_fullscreen_ar)

        self.fullscreen_label = tk.Label(win, bg="black", bd=0)
        self.fullscreen_label.pack(fill=tk.BOTH, expand=True)
        self.fullscreen_btn.configure(text="Exit Fullscreen")
        win.focus_force()
        self.log("Fullscreen AR overlay opened.")

    def _close_fullscreen_ar(self) -> None:
        if self._fullscreen_is_open():
            try:
                self.fullscreen_win.destroy()
            except tk.TclError:
                pass
        self.fullscreen_win = None
        self.fullscreen_label = None
        self._fullscreen_photo = None
        if hasattr(self, "fullscreen_btn"):
            self.fullscreen_btn.configure(text="Fullscreen")

    def _update_fullscreen_display(self, rgb: np.ndarray) -> None:
        if not self._fullscreen_is_open() or self.fullscreen_label is None:
            return
        max_w = self.fullscreen_label.winfo_width() or self.fullscreen_win.winfo_screenwidth()
        max_h = self.fullscreen_label.winfo_height() or self.fullscreen_win.winfo_screenheight()
        display_rgb = self._resize_rgb_for_box(rgb, max_w, max_h)
        self._fullscreen_photo = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        self.fullscreen_label.configure(image=self._fullscreen_photo)

    def _draw_registered_wireframe(self, vis: np.ndarray, result: FrameResult) -> None:
        """Project the registered wireframe onto the AR frame."""
        if self.camera_matrix is None:
            return

        nodes = self.geometry_data.get("nodes", [])
        edges = self.geometry_data.get("traceLines", [])
        if not nodes or not edges:
            return

        if self.pipeline is not None and self.pipeline.uses_structure_board and result.pose is not None:
            self._draw_board_pose_wireframe(vis, result, nodes, edges)
            return

        if not self.registration.is_registered:
            return

        # Transform UNV nodes to world via registration (points end up in the
        # camera frame, so projection uses an identity pose) and draw.
        node_px = overlay.project_nodes(
            nodes, self.camera_matrix, self.dist_coeffs,
            node_transform=self.registration.transform_point,
        )
        for p1, p2 in overlay.wireframe_segments(node_px, edges):
            cv2.line(vis, p1, p2, (0, 220, 255), 1, cv2.LINE_AA)

    def _draw_board_pose_wireframe(
        self,
        vis: np.ndarray,
        result: FrameResult,
        nodes: list[dict],
        edges: list[list[int]],
    ) -> None:
        node_px = overlay.project_nodes(
            nodes, self.camera_matrix, self.dist_coeffs,
            rvec=result.pose.rvec, tvec=result.pose.tvec,
        )
        for p1, p2 in overlay.wireframe_segments(node_px, edges):
            cv2.line(vis, p1, p2, (0, 220, 255), 1, cv2.LINE_AA)

    def _take_screenshot(self) -> None:
        if not hasattr(self, "_last_ar_frame") or self._last_ar_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(__file__).parent / f"screenshot_{ts}.png"
        cv2.imwrite(str(path), self._last_ar_frame)
        self.log(f"Screenshot saved: {path.name}")

    # ══════════════════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════════════════

    def _on_close(self) -> None:
        if self.ar_running:
            self._stop_ar()
        else:
            self._close_fullscreen_ar()
        plt.close("all")
        SessionLogger.shutdown()
        self.root.destroy()

    # ══════════════════════════════════════════════════════════════════════
    #  Marker loading from directory
    # ══════════════════════════════════════════════════════════════════════

    def _show_marker_loader(self) -> None:
        MarkerLoaderWindow(self.root, self)


# ══════════════════════════════════════════════════════════════════════════
#  Sub-windows
# ══════════════════════════════════════════════════════════════════════════

class InfoCenterWindow:
    """Browsable, in-app help for the main EyeLab workflows."""

    def __init__(self, parent: tk.Tk, app: EyeLabApp):
        self.app = app
        self.win = tk.Toplevel(parent)
        self.win.title("EyeLab Information Center")
        self.win.geometry("760x460")
        self.win.transient(parent)

        outer = ttk.Frame(self.win)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.topic_list = tk.Listbox(outer, exportselection=False, width=28)
        self.topic_list.pack(side=tk.LEFT, fill=tk.Y)
        for topic in HELP_TOPICS:
            self.topic_list.insert(tk.END, topic)

        right = ttk.Frame(outer)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.title_var = tk.StringVar()
        ttk.Label(right, textvariable=self.title_var, font=("", 12, "bold")).pack(anchor="w")

        self.text = tk.Text(right, wrap=tk.WORD, height=16)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.text.configure(state="disabled")

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="ArUco calibration wizard", command=app._show_aruco_wizard).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=self.win.destroy).pack(side=tk.RIGHT)

        self.topic_list.bind("<<ListboxSelect>>", self._show_selected)
        self.topic_list.selection_set(0)
        self._show_topic(next(iter(HELP_TOPICS)))

    def _show_selected(self, event=None) -> None:
        sel = self.topic_list.curselection()
        if not sel:
            return
        self._show_topic(self.topic_list.get(sel[0]))

    def _show_topic(self, topic: str) -> None:
        self.title_var.set(topic)
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", HELP_TOPICS[topic])
        self.text.configure(state="disabled")


class ArucoCalibrationWizard:
    """Step-by-step ChArUco calibration tutorial with direct actions."""

    def __init__(self, parent: tk.Tk, app: EyeLabApp):
        self.app = app
        self.index = 0
        self.win = tk.Toplevel(parent)
        self.win.title("ArUco Calibration Wizard")
        self.win.geometry("780x430")
        self.win.transient(parent)

        main = ttk.Frame(self.win)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.step_var = tk.StringVar()
        ttk.Label(main, textvariable=self.step_var, font=("", 13, "bold")).pack(anchor="w")

        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        self.body = tk.Text(content, wrap=tk.WORD, height=9)
        self.body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.body.configure(state="disabled")
        self.preview_label = ttk.Label(content, anchor="center")
        self.preview_label.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self._preview_photo = None

        actions = ttk.Frame(main)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Generate board image...", command=self._generate_board).pack(side=tk.LEFT)
        ttk.Button(actions, text="Start live calibration", command=self._start_calibration).pack(side=tk.LEFT, padx=6)

        nav = ttk.Frame(main)
        nav.pack(fill=tk.X, pady=(12, 0))
        self.back_btn = ttk.Button(nav, text="Back", command=self._back)
        self.back_btn.pack(side=tk.LEFT)
        self.next_btn = ttk.Button(nav, text="Next", command=self._next)
        self.next_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Close", command=self.win.destroy).pack(side=tk.RIGHT)

        self._render()
        self._render_board_preview()

    def _render(self) -> None:
        title, text = ARUCO_CALIBRATION_STEPS[self.index]
        self.step_var.set(f"{title}  ({self.index + 1}/{len(ARUCO_CALIBRATION_STEPS)})")
        self.body.configure(state="normal")
        self.body.delete("1.0", tk.END)
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")
        self.back_btn.configure(state="normal" if self.index > 0 else "disabled")
        self.next_btn.configure(
            text="Next" if self.index < len(ARUCO_CALIBRATION_STEPS) - 1 else "Done"
        )

    def _render_board_preview(self) -> None:
        try:
            board = make_charuco_board(5, 7, 0.025, 0.019)
            img = board.generateImage((220, 320), marginSize=8, borderBits=1)
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self._preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception:
            self.preview_label.configure(text="ChArUco\nboard\npreview")

    def _back(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render()

    def _next(self) -> None:
        if self.index < len(ARUCO_CALIBRATION_STEPS) - 1:
            self.index += 1
            self._render()
        else:
            self.win.destroy()

    def _generate_board(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save ChArUco board image",
            initialfile="charuco_board.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            board = make_charuco_board(5, 7, 0.025, 0.019)
            generate_board_image(board, path)
            self.app.log(f"Generated ChArUco board image: {path}")
            messagebox.showinfo(
                "Board Generated",
                f"Saved ChArUco board image:\n{path}\n\nPrint at 100% scale.",
            )
        except Exception as e:
            messagebox.showerror("Board Generation Error", str(e))

    def _start_calibration(self) -> None:
        self.app._start_calibration()


class MarkerGenWindow:
    """Dialog for generating ArUco markers."""

    def __init__(self, parent: tk.Tk, app: EyeLabApp):
        self.app = app
        self.win = tk.Toplevel(parent)
        self.win.title("Generate ArUco Markers")
        self.win.geometry("380x200")
        self.win.transient(parent)

        ttk.Label(self.win, text="Number of markers:").grid(row=0, column=0, padx=8, pady=4, sticky="w")
        self.count_var = tk.IntVar(value=10)
        ttk.Spinbox(self.win, from_=1, to=50, textvariable=self.count_var, width=6).grid(row=0, column=1, padx=8)

        ttk.Label(self.win, text="DPI:").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self.dpi_var = tk.IntVar(value=300)
        ttk.Entry(self.win, textvariable=self.dpi_var, width=8).grid(row=1, column=1, padx=8)

        ttk.Label(self.win, text="Marker size comes from the main Marker panel.").grid(
            row=2, column=0, columnspan=2, padx=8, pady=4, sticky="w")
        ttk.Label(self.win, text="Dictionary: DICT_4X4_50").grid(
            row=3, column=0, columnspan=2, padx=8, sticky="w")

        ttk.Label(self.win, text=f"Output: {MARKERS_DIR}").grid(
            row=4, column=0, columnspan=2, padx=8, pady=4, sticky="w")

        ttk.Button(self.win, text="Generate", command=self._generate).grid(
            row=5, column=0, columnspan=2, pady=10)

    def _generate(self) -> None:
        count = self.count_var.get()
        dpi = self.dpi_var.get()
        try:
            ids = list(range(count))
            marker_size_mm = float(self.app.marker_size_var.get())
            grid_spacing_mm = max(marker_size_mm + 4.0, marker_size_mm)
            generate_markers(
                str(MARKERS_DIR),
                dpi=dpi,
                marker_ids=ids,
                add_id_label=True,
                marker_size_mm=marker_size_mm,
                grid_spacing_mm=grid_spacing_mm,
            )
            self.app.log(f"Generated {count} markers (aruco01–aruco{count:02d}) in {MARKERS_DIR}")
            messagebox.showinfo("Done", f"Generated {count} markers in:\n{MARKERS_DIR}")
            self.win.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


class MarkerLoaderWindow:
    """
    Dialog to load existing ArUco marker images from any directory and
    re-render them at a user-specified physical size (mm) and DPI, so the
    printed result matches the dimensions used by the AR pipeline.

    The loaded marker PNGs are scaled (nearest-neighbour, no smoothing) to the
    exact pixel count corresponding to `marker_size_mm` at the chosen DPI,
    then padded with the same proportional white border that
    generate_markers.py uses, and written to MARKERS_DIR.

    The chosen marker size is also pushed into the main app's
    `marker_size_var` so the AR pipeline uses the matching physical size.
    """

    def __init__(self, parent: tk.Tk, app: "EyeLabApp"):
        self.app = app
        self.win = tk.Toplevel(parent)
        self.win.title("Load Markers From Directory")
        self.win.geometry("480x300")
        self.win.transient(parent)

        # Source directory
        ttk.Label(self.win, text="Source folder containing marker images (PNG):").grid(
            row=0, column=0, columnspan=3, padx=8, pady=(10, 2), sticky="w")
        self.src_var = tk.StringVar(value=str(MARKERS_DIR))
        ttk.Entry(self.win, textvariable=self.src_var, width=48).grid(
            row=1, column=0, columnspan=2, padx=8, sticky="we")
        ttk.Button(self.win, text="Browse...", command=self._browse).grid(
            row=1, column=2, padx=4)

        # Marker size (mm)
        ttk.Label(self.win, text="Marker physical size (mm):").grid(
            row=2, column=0, padx=8, pady=(12, 2), sticky="w")
        self.size_var = tk.DoubleVar(value=app.marker_size_var.get())
        ttk.Entry(self.win, textvariable=self.size_var, width=10).grid(
            row=2, column=1, padx=8, pady=(12, 2), sticky="w")

        # Grid spacing (mm) — used only for the white border padding
        ttk.Label(self.win, text="Grid cell size (mm):").grid(
            row=3, column=0, padx=8, pady=2, sticky="w")
        self.grid_var = tk.DoubleVar(value=GRID_SPACING_MM)
        ttk.Entry(self.win, textvariable=self.grid_var, width=10).grid(
            row=3, column=1, padx=8, pady=2, sticky="w")

        # DPI
        ttk.Label(self.win, text="Output DPI:").grid(
            row=4, column=0, padx=8, pady=2, sticky="w")
        self.dpi_var = tk.IntVar(value=300)
        ttk.Entry(self.win, textvariable=self.dpi_var, width=10).grid(
            row=4, column=1, padx=8, pady=2, sticky="w")

        # Apply pipeline size
        self.update_pipeline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.win,
            text="Also set this size as the AR pipeline marker size",
            variable=self.update_pipeline_var,
        ).grid(row=5, column=0, columnspan=3, padx=8, pady=(8, 2), sticky="w")

        ttk.Label(
            self.win,
            text=(
                f"Output: {MARKERS_DIR}\n"
                "Markers will be re-rendered (nearest-neighbour) at the\n"
                "exact pixel count for the requested physical size."
            ),
            justify="left",
        ).grid(row=6, column=0, columnspan=3, padx=8, pady=(8, 4), sticky="w")

        ttk.Button(self.win, text="Load & Re-render", command=self._load).grid(
            row=7, column=0, columnspan=3, pady=10)

    def _browse(self) -> None:
        d = filedialog.askdirectory(
            title="Select marker source directory",
            initialdir=str(MARKERS_DIR),
        )
        if d:
            self.src_var.set(d)

    def _load(self) -> None:
        src = Path(self.src_var.get().strip())
        if not src.is_dir():
            messagebox.showerror("Invalid Source", f"Not a directory:\n{src}")
            return
        try:
            size_mm = float(self.size_var.get())
            grid_mm = float(self.grid_var.get())
            dpi = int(self.dpi_var.get())
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Invalid Input", str(e))
            return
        if size_mm <= 0 or dpi <= 0:
            messagebox.showerror("Invalid Input", "Size and DPI must be positive.")
            return
        if grid_mm < size_mm:
            messagebox.showerror("Invalid Input",
                                 "Grid cell size must be ≥ marker size.")
            return

        files = sorted(
            [p for p in src.iterdir()
             if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        )
        if not files:
            messagebox.showwarning("No Images", "No image files found in that folder.")
            return

        marker_px = int(round(size_mm * dpi / 25.4))
        border_px = int(round((grid_mm - size_mm) / 2.0 * dpi / 25.4))
        MARKERS_DIR.mkdir(parents=True, exist_ok=True)

        count = 0
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.app.log(f"Skipped (not an image): {f.name}", level="WARNING")
                continue
            # Crop any existing white padding by tight-binarising and bounding box.
            # Falls back to the original image if cropping fails.
            try:
                _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
                ys, xs = np.where(bw > 0)
                if ys.size and xs.size:
                    y0, y1 = ys.min(), ys.max() + 1
                    x0, x1 = xs.min(), xs.max() + 1
                    img = img[y0:y1, x0:x1]
            except Exception:
                pass
            # Resize to exact pixel count for the requested physical size
            resized = cv2.resize(img, (marker_px, marker_px),
                                 interpolation=cv2.INTER_NEAREST)
            # Pad with white border to reach the grid cell size
            if border_px > 0:
                resized = cv2.copyMakeBorder(
                    resized, border_px, border_px, border_px, border_px,
                    cv2.BORDER_CONSTANT, value=255,
                )
            out = MARKERS_DIR / f.name
            cv2.imwrite(str(out), resized)
            count += 1

        if self.update_pipeline_var.get():
            self.app.marker_size_var.set(size_mm)

        self.app.log(
            f"Loaded {count} marker(s) from {src} at {size_mm} mm "
            f"({marker_px}px @ {dpi} DPI). Output: {MARKERS_DIR}"
        )
        messagebox.showinfo(
            "Done",
            f"Re-rendered {count} marker(s) at {size_mm} mm.\n"
            f"Saved to:\n{MARKERS_DIR}\n\n"
            "Print at 100% scale (no 'fit to page')."
        )
        self.win.destroy()


class CorrespondenceEditor:
    """Dialog for editing marker ↔ UNV node correspondences."""

    def __init__(self, parent: tk.Tk, app: EyeLabApp):
        self.app = app
        self.win = tk.Toplevel(parent)
        self.win.title("Marker ↔ Mesh Node Correspondences")
        self.win.geometry("920x480")
        self.win.transient(parent)

        ttk.Label(self.win, text=(
            "Assign ArUco markers to UNV node positions.\n"
            "Each marker placed on the physical structure must be linked\n"
            "to its UNV node ID so registration can compute the alignment."
        ), wraplength=500, justify="left").pack(padx=8, pady=6)

        # Treeview for correspondences
        cols = ("marker", "node_id", "x", "y", "z", "face", "up", "roll", "size", "desc")
        self.tree = ttk.Treeview(self.win, columns=cols, show="headings", height=10)
        self.tree.heading("marker", text="Marker")
        self.tree.heading("node_id", text="Node ID")
        self.tree.heading("x", text=f"X ({POSITION_UI_UNIT})")
        self.tree.heading("y", text=f"Y ({POSITION_UI_UNIT})")
        self.tree.heading("z", text=f"Z ({POSITION_UI_UNIT})")
        self.tree.heading("face", text="Face")
        self.tree.heading("up", text="Up")
        self.tree.heading("roll", text="Roll deg")
        self.tree.heading("size", text="Size mm")
        self.tree.heading("desc", text="Description")
        for c in cols:
            self.tree.column(c, width=65)
        self.tree.column("desc", width=120)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.refresh()

        btn_row = ttk.Frame(self.win)
        btn_row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_row, text="Add", command=self._add).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Edit values...", command=self._edit).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Remove", command=self._remove).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Pick from mesh...", command=self._pick_from_mesh).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Drag selected", command=self._drag_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Load file...", command=self._load_from_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Save copy...", command=self._save_to_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Save & Close", command=self._save).pack(side=tk.RIGHT, padx=4)

    def refresh(self) -> None:
        selected_marker = self._selected_marker_id()
        self.tree.delete(*self.tree.get_children())
        for corr in self.app.correspondences:
            self._insert_corr(corr)
        if selected_marker is not None:
            for item in self.tree.get_children():
                vals = self.tree.item(item, "values")
                if self._parse_marker_name(vals[0]) == selected_marker:
                    self.tree.selection_set(item)
                    self.tree.see(item)
                    break

    def _insert_corr(self, corr: MarkerCorrespondence) -> None:
        p = corr.unv_position
        size_mm = corr.marker_size_mm or float(self.app.marker_size_var.get())
        self.tree.insert("", "end", values=(
            f"aruco{corr.marker_id + 1:02d}", corr.node_id or "",
            format_position_ui(p[0]), format_position_ui(p[1]), format_position_ui(p[2]),
            normal_label(corr.normal), marker_up_label(corr.normal, corr.roll_deg),
            f"{corr.roll_deg:.1f}", f"{size_mm:.2f}",
            corr.description,
        ))

    def _add(self) -> None:
        AddCorrespondenceDialog(self.win, self.app, self.tree)

    def _edit(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a marker row first.")
            return
        EditCorrespondenceDialog(self.win, self.app, self, sel[0])

    def _remove(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        for item in sel:
            idx = self.tree.index(item)
            self.tree.delete(item)
            if idx < len(self.app.correspondences):
                self.app.correspondences.pop(idx)
        self.apply_tree_to_app()

    def _pick_from_mesh(self) -> None:
        """Let user select a node from the loaded geometry as the UNV position."""
        if self.app.geometry_data is None:
            messagebox.showwarning("No Geometry", "Load a UNV or JSON file first.")
            return
        NodePickerDialog(self.win, self.app, self.tree)

    def _drag_selected(self) -> None:
        marker_id = self._selected_marker_id()
        if marker_id is None:
            messagebox.showwarning("No Selection", "Select a marker row first.")
            return
        self.apply_tree_to_app()
        self.app._begin_marker_drag(marker_id, editor=self)

    def _selected_marker_id(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        if not vals:
            return None
        return self._parse_marker_name(vals[0])

    @staticmethod
    def _parse_marker_name(marker_name: str) -> Optional[int]:
        try:
            return int(str(marker_name).replace("aruco", "")) - 1
        except ValueError:
            return None

    def _collect_correspondences_from_tree(self) -> list[MarkerCorrespondence]:
        correspondences: list[MarkerCorrespondence] = []
        for item in self.tree.get_children():
            vals = self.tree.item(item, "values")
            # Parse marker name back to ID
            marker_name = vals[0]  # "aruco01"
            marker_id = self._parse_marker_name(marker_name)
            if marker_id is None:
                continue
            node_id = int(vals[1]) if vals[1] else None
            x, y, z = (
                position_ui_to_m(float(vals[2])),
                position_ui_to_m(float(vals[3])),
                position_ui_to_m(float(vals[4])),
            )
            face_label = vals[5] if vals[5] in AXIS_NORMALS else "+Z"
            roll_deg = float(vals[7]) if vals[7] else 0.0
            size_mm = float(vals[8]) if vals[8] else None
            desc = vals[9]
            correspondences.append(MarkerCorrespondence(
                marker_id=marker_id,
                unv_position=np.array([x, y, z], dtype=np.float64),
                node_id=node_id,
                description=desc,
                normal=AXIS_NORMALS[face_label],
                roll_deg=roll_deg,
                marker_size_mm=size_mm,
            ))
        return correspondences

    def apply_tree_to_app(self) -> None:
        self.app.correspondences = self._collect_correspondences_from_tree()
        self.app._save_correspondences()

    def _load_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load marker correspondences",
            initialdir=str(CONFIG_DIR),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.app.correspondences = load_marker_config(path)
            self.app._save_correspondences()
            self.refresh()
            self.app.log(f"Loaded {len(self.app.correspondences)} correspondences from {path}")
        except Exception as e:
            messagebox.showerror("Load Correspondences", str(e))

    def _save_to_file(self) -> None:
        self.apply_tree_to_app()
        path = filedialog.asksaveasfilename(
            title="Save marker correspondences",
            initialdir=str(CONFIG_DIR),
            initialfile="marker_config.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_marker_config(path, self.app.correspondences)
            self.app.log(f"Saved {len(self.app.correspondences)} correspondences to {path}")
        except Exception as e:
            messagebox.showerror("Save Correspondences", str(e))

    def _save(self) -> None:
        # Rebuild correspondences from treeview
        self.apply_tree_to_app()
        self.app.log(f"Saved {len(self.app.correspondences)} correspondences.")
        self.win.destroy()


class AddCorrespondenceDialog:
    """Small dialog to add a single marker ↔ position correspondence."""

    def __init__(self, parent: tk.Toplevel, app: EyeLabApp, tree: ttk.Treeview):
        self.app = app
        self.tree = tree
        self.win = tk.Toplevel(parent)
        self.win.title("Add Correspondence")
        self.win.geometry("330x340")
        self.win.transient(parent)

        ttk.Label(self.win, text="ArUco marker number (1–50):").grid(row=0, column=0, padx=8, pady=4, sticky="w")
        self.mid_var = tk.IntVar(value=1)
        ttk.Spinbox(self.win, from_=1, to=50, textvariable=self.mid_var, width=6).grid(row=0, column=1, padx=8)

        ttk.Label(self.win, text="UNV Node ID (optional):").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self.nid_var = tk.StringVar(value="")
        ttk.Entry(self.win, textvariable=self.nid_var, width=8).grid(row=1, column=1, padx=8)

        for i, axis in enumerate((f"X ({POSITION_UI_UNIT}):", f"Y ({POSITION_UI_UNIT}):", f"Z ({POSITION_UI_UNIT}):")):
            ttk.Label(self.win, text=axis).grid(row=2 + i, column=0, padx=8, pady=2, sticky="w")
        self.x_var = tk.DoubleVar(value=0.0)
        self.y_var = tk.DoubleVar(value=0.0)
        self.z_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.win, textvariable=self.x_var, width=10).grid(row=2, column=1, padx=8)
        ttk.Entry(self.win, textvariable=self.y_var, width=10).grid(row=3, column=1, padx=8)
        ttk.Entry(self.win, textvariable=self.z_var, width=10).grid(row=4, column=1, padx=8)

        ttk.Label(self.win, text="Face normal:").grid(row=5, column=0, padx=8, pady=2, sticky="w")
        self.face_var = tk.StringVar(value="+Z")
        ttk.Combobox(
            self.win, textvariable=self.face_var, values=list(AXIS_NORMALS.keys()),
            state="readonly", width=8,
        ).grid(row=5, column=1, padx=8, sticky="w")

        ttk.Label(self.win, text="Roll (deg):").grid(row=6, column=0, padx=8, pady=2, sticky="w")
        self.roll_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.win, textvariable=self.roll_var, width=10).grid(row=6, column=1, padx=8)

        ttk.Label(self.win, text="Size (mm):").grid(row=7, column=0, padx=8, pady=2, sticky="w")
        self.size_var = tk.DoubleVar(value=float(app.marker_size_var.get()))
        ttk.Entry(self.win, textvariable=self.size_var, width=10).grid(row=7, column=1, padx=8)

        ttk.Label(self.win, text="Description:").grid(row=8, column=0, padx=8, pady=2, sticky="w")
        self.desc_var = tk.StringVar(value="")
        ttk.Entry(self.win, textvariable=self.desc_var, width=18).grid(row=8, column=1, padx=8)

        ttk.Button(self.win, text="Add", command=self._add).grid(row=9, column=0, columnspan=2, pady=10)

    def _add(self) -> None:
        mid = self.mid_var.get() - 1   # internal 0-indexed
        nid_str = self.nid_var.get().strip()
        node_id = int(nid_str) if nid_str else None
        x, y, z = self.x_var.get(), self.y_var.get(), self.z_var.get()
        self.tree.insert("", "end", values=(
            f"aruco{mid + 1:02d}", node_id or "",
            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
            self.face_var.get(), marker_up_label(self.face_var.get(), float(self.roll_var.get())),
            f"{float(self.roll_var.get()):.1f}",
            f"{float(self.size_var.get()):.2f}", self.desc_var.get().strip(),
        ))
        self.win.destroy()


class EditCorrespondenceDialog:
    """Edit one marker pose row with numeric fields."""

    def __init__(self, parent: tk.Toplevel, app: EyeLabApp, editor: CorrespondenceEditor, item_id):
        self.app = app
        self.editor = editor
        self.item_id = item_id
        vals = editor.tree.item(item_id, "values")

        self.win = tk.Toplevel(parent)
        self.win.title("Edit Marker Pose")
        self.win.geometry("340x360")
        self.win.transient(parent)

        marker_id = editor._parse_marker_name(vals[0])
        self.mid_var = tk.IntVar(value=(marker_id + 1) if marker_id is not None else 1)
        self.node_var = tk.StringVar(value=str(vals[1]) if vals[1] else "")
        self.x_var = tk.DoubleVar(value=float(vals[2]))
        self.y_var = tk.DoubleVar(value=float(vals[3]))
        self.z_var = tk.DoubleVar(value=float(vals[4]))
        self.face_var = tk.StringVar(value=vals[5] if vals[5] in AXIS_NORMALS else "+Z")
        self.roll_var = tk.DoubleVar(value=float(vals[7]) if vals[7] else 0.0)
        self.size_var = tk.DoubleVar(value=float(vals[8]) if vals[8] else float(app.marker_size_var.get()))
        self.desc_var = tk.StringVar(value=vals[9] if len(vals) > 9 else "")

        fields = ttk.Frame(self.win)
        fields.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(fields, text="ArUco #:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Spinbox(fields, from_=1, to=50, textvariable=self.mid_var, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(fields, text="Node ID:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(fields, textvariable=self.node_var, width=10).grid(row=1, column=1, sticky="w")

        for i, (label, var) in enumerate((
            (f"X ({POSITION_UI_UNIT}):", self.x_var),
            (f"Y ({POSITION_UI_UNIT}):", self.y_var),
            (f"Z ({POSITION_UI_UNIT}):", self.z_var),
        ), start=2):
            ttk.Label(fields, text=label).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(fields, textvariable=var, width=12).grid(row=i, column=1, sticky="w")

        ttk.Label(fields, text="Face normal:").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(
            fields, textvariable=self.face_var, values=list(AXIS_NORMALS.keys()),
            state="readonly", width=8,
        ).grid(row=5, column=1, sticky="w")

        ttk.Label(fields, text="Roll (deg):").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Entry(fields, textvariable=self.roll_var, width=12).grid(row=6, column=1, sticky="w")

        ttk.Label(fields, text="Size (mm):").grid(row=7, column=0, sticky="w", pady=3)
        ttk.Entry(fields, textvariable=self.size_var, width=12).grid(row=7, column=1, sticky="w")

        ttk.Label(fields, text="Description:").grid(row=8, column=0, sticky="w", pady=3)
        ttk.Entry(fields, textvariable=self.desc_var, width=22).grid(row=8, column=1, sticky="w")

        buttons = ttk.Frame(self.win)
        buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(buttons, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(buttons, text="Cancel", command=self.win.destroy).pack(side=tk.RIGHT)

    def _apply(self) -> None:
        marker_id = int(self.mid_var.get()) - 1
        node_text = self.node_var.get().strip()
        node_value = node_text if node_text else ""
        self.editor.tree.item(self.item_id, values=(
            f"aruco{marker_id + 1:02d}",
            node_value,
            f"{float(self.x_var.get()):.4f}",
            f"{float(self.y_var.get()):.4f}",
            f"{float(self.z_var.get()):.4f}",
            self.face_var.get(),
            marker_up_label(self.face_var.get(), float(self.roll_var.get())),
            f"{float(self.roll_var.get()):.1f}",
            f"{float(self.size_var.get()):.2f}",
            self.desc_var.get().strip(),
        ))
        self.editor.apply_tree_to_app()
        self.win.destroy()


class NodePickerDialog:
    """Let user pick a node from the loaded geometry to use as a correspondence position."""

    def __init__(self, parent: tk.Toplevel, app: EyeLabApp, tree: ttk.Treeview):
        self.app = app
        self.tree = tree
        self.win = tk.Toplevel(parent)
        self.win.title("Pick Node from Mesh")
        self.win.geometry("430x460")
        self.win.transient(parent)

        ttk.Label(self.win, text="Select a node, then assign a marker:").pack(padx=8, pady=4)

        # Node list
        cols = ("id", "x", "y", "z")
        self.node_tree = ttk.Treeview(self.win, columns=cols, show="headings", height=12)
        self.node_tree.heading("id", text="ID")
        self.node_tree.heading("x", text=f"X ({POSITION_UI_UNIT})")
        self.node_tree.heading("y", text=f"Y ({POSITION_UI_UNIT})")
        self.node_tree.heading("z", text=f"Z ({POSITION_UI_UNIT})")
        for c in cols:
            self.node_tree.column(c, width=80)
        self.node_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        for n in app.geometry_data.get("nodes", []):
            self.node_tree.insert("", "end", values=(
                n["id"],
                format_position_ui(n["x"]),
                format_position_ui(n["y"]),
                format_position_ui(n["z"]),
            ))

        row = ttk.Frame(self.win)
        row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="Assign to aruco #:").pack(side=tk.LEFT)
        self.mid_var = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.mid_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Assign", command=self._assign).pack(side=tk.LEFT, padx=8)

        pose_row = ttk.Frame(self.win)
        pose_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(pose_row, text="Face:").pack(side=tk.LEFT)
        self.face_var = tk.StringVar(value="+Z")
        ttk.Combobox(
            pose_row, textvariable=self.face_var, values=list(AXIS_NORMALS.keys()),
            state="readonly", width=5,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(pose_row, text="Roll:").pack(side=tk.LEFT)
        self.roll_var = tk.DoubleVar(value=0.0)
        ttk.Entry(pose_row, textvariable=self.roll_var, width=7).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(pose_row, text="Size mm:").pack(side=tk.LEFT)
        self.size_var = tk.DoubleVar(value=float(app.marker_size_var.get()))
        ttk.Entry(pose_row, textvariable=self.size_var, width=7).pack(side=tk.LEFT, padx=4)

    def _assign(self) -> None:
        sel = self.node_tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a node first.")
            return
        vals = self.node_tree.item(sel[0], "values")
        node_id = int(vals[0])
        x, y, z = float(vals[1]), float(vals[2]), float(vals[3])
        mid = self.mid_var.get() - 1
        self.tree.insert("", "end", values=(
            f"aruco{mid + 1:02d}", node_id,
            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
            self.face_var.get(), marker_up_label(self.face_var.get(), float(self.roll_var.get())),
            f"{float(self.roll_var.get()):.1f}",
            f"{float(self.size_var.get()):.2f}",
            f"Node {node_id}",
        ))
        self.app.log(
            f"Assigned aruco{mid + 1:02d} to Node {node_id} "
            f"({x:.2f}, {y:.2f}, {z:.2f} {POSITION_UI_UNIT})"
        )
        self.win.destroy()


class CalibrationWindow:
    """Live ChArUco calibration window."""

    def __init__(self, parent: tk.Tk, camera_index: int, callback):
        self.callback = callback
        self.win = tk.Toplevel(parent)
        self.win.title("Camera Calibration (ChArUco)")
        self.win.geometry("720x560")
        self.win.transient(parent)
        self.win.protocol("WM_DELETE_WINDOW", self._abort)

        self.cap = open_camera(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera {camera_index}.")
            self.win.destroy()
            return

        self.board = make_charuco_board(5, 7, 0.025, 0.019)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.all_corners = []
        self.all_ids = []
        self.image_size = None
        self.min_frames = 15
        self._current_corners = None
        self._current_ids = None
        self._frame_failures = 0

        self.label = ttk.Label(self.win, text="Waiting for camera frame...", anchor="center")
        self.label.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(self.win)
        status.pack(fill=tk.X, padx=8, pady=4)
        self.status_var = tk.StringVar(value=f"Captured: 0/{self.min_frames}  -  SPACE/Capture to save frame, ESC/Finish to compute")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(status, text="Capture", command=self._capture).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(status, text="Finish", command=self._finish).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(status, text="Abort", command=self._abort).pack(side=tk.RIGHT, padx=(4, 0))

        self.win.bind_all("<space>", self._capture, add="+")
        self.win.bind_all("<Escape>", self._finish, add="+")
        self._photo = None
        self._running = True
        self.win.after(100, self.win.focus_force)
        self._loop()

    def _loop(self) -> None:
        if not self._running:
            return
        ok, frame = self.cap.read()
        if ok:
            self._frame_failures = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.image_size is None:
                self.image_size = (gray.shape[1], gray.shape[0])
            corners, ids, _, _ = self.detector.detectBoard(gray)
            self._current_corners = corners
            self._current_ids = ids
            if corners is not None and ids is not None and len(ids) >= 4:
                cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
                detect_text = f"Detected {len(ids)} ChArUco corners - ready to capture"
                detect_color = (40, 210, 80)
            else:
                detect_text = "Show the printed ChArUco board to the camera"
                detect_color = (0, 170, 255)
            cv2.putText(
                frame,
                detect_text,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                detect_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Captured {len(self.all_corners)}/{self.min_frames}",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (700, 520))
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.label.configure(image=self._photo)
            self._gray = gray
        else:
            self._frame_failures += 1
            if self._frame_failures == 1 or self._frame_failures % 30 == 0:
                self.status_var.set("No camera frame received. Check camera selection and close other camera apps.")
        self.win.after(30, self._loop)

    def _capture(self, event=None) -> None:
        if not hasattr(self, "_gray"):
            self.status_var.set("No camera frame yet. Wait for the live image.")
            return
        corners, ids = self._current_corners, self._current_ids
        if corners is not None and ids is not None and len(ids) >= 4:
            self.all_corners.append(corners)
            self.all_ids.append(ids)
            n = len(self.all_corners)
            self.status_var.set(f"Captured: {n}/{self.min_frames}  -  move/tilt board, then capture another")
        else:
            self.status_var.set("Board corners not detected. Move closer, improve light, or reduce blur.")

    def _finish(self, event=None) -> None:
        n = len(self.all_corners)
        if n < self.min_frames:
            self.status_var.set(f"Need {self.min_frames} frames (have {n}). Keep capturing.")
            return
        self._running = False
        self.cap.release()
        self._unbind_keys()

        rms, cam_mat, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            self.all_corners, self.all_ids, self.board, self.image_size, None, None,
        )
        self.win.destroy()
        self.callback(cam_mat, dist, rms, self.image_size)

    def _abort(self) -> None:
        self._running = False
        self.cap.release()
        self._unbind_keys()
        self.win.destroy()

    def _unbind_keys(self) -> None:
        try:
            self.win.unbind_all("<space>")
            self.win.unbind_all("<Escape>")
        except tk.TclError:
            pass


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    EyeLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
