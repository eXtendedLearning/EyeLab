#!/usr/bin/env python3
"""
EyeLab GUI — Phase 1 Webcam MVP.

Main application shell. Integrates all pipeline modules into a single
tkinter application:
  - Load & preview UNV geometry (3D interactive plot)
  - Generate / manage ArUco markers (aruco01, aruco02 ...)
  - Camera selection & calibration (with persistent status)
  - Marker-to-mesh positioning (assign markers to UNV nodes visually)
  - Live AR overlay (webcam + wireframe, toggle on/off)
  - Session log, screenshot capture

Sub-windows live in gui_calibration.py (help, wizard, live calibration) and
gui_markers.py (marker generation/loading, correspondence editing). Shared
paths/constants live in gui_common.py.

Usage:
    python eyelab_gui.py
    (or via run_eyelab.bat)
"""

from __future__ import annotations

import json
import os
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
from calibrate import (
    BOARD_COLS,
    BOARD_ROWS,
    load_calibration,
    save_calibration,
)
from camera_utils import list_cameras
from eyelab_logger import SessionLogger
from eyelab_version import VERSION_STRING
from generate_markers import generate_markers, MARKER_SIZE_MM
from gui_calibration import ArucoCalibrationWizard, CalibrationWindow, InfoCenterWindow
from gui_common import (
    CALIBRATION_FILE,
    CHARUCO_MARKER_M,
    CHARUCO_SQUARE_M,
    CONFIG_DIR,
    HAMMER_MARKER_CONFIG_FILE,
    LOG_DIR,
    MARKER_CONFIG_FILE,
    MARKERS_DIR,
    TEST_ASSETS_DIR,
)
from gui_geometry_editor import GeometryEditorTab
from gui_markers import CorrespondenceEditor, MarkerGenWindow, MarkerLoaderWindow
import overlay
from pose_estimator import (
    ArucoDetectorTuning,
    ArucoPipeline,
    DETECTOR_TUNING_PRESETS,
    FrameResult,
    MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE,
)
from registration import (
    SpatialRegistration,
    MarkerCorrespondence,
    load_marker_config,
    marker_axes_from_normal,
    marker_object_corners,
    normal_label,
    save_marker_config,
)
from unv_to_json import UNVParser

# ── Constants ─────────────────────────────────────────────────────────────────

APP_TITLE = f"{VERSION_STRING} — Phase 1 Webcam MVP"
WINDOW_SIZE = "1400x860"
PREVIEW_W, PREVIEW_H = 640, 480

CONTROL_PANEL_SECTIONS = (
    ("camera", "Camera"),
    ("calibration", "Calibration"),
    ("geometry", "Geometry (UNV)"),
    ("markers", "Marker Mesh Positioning"),
    ("hammer", "Hammer ArUco Markers"),
    ("ar", "AR Overlay"),
    ("tuning", "Detection Tuning"),
    ("diagnostics", "Diagnostics"),
)

DETECTION_TUNING_HELP = {
    "Contrast": "CLAHE contrast applied before ArUco detection. Higher can reveal weak markers, but it can also amplify sensor noise.",
    "Thresh max": "Largest adaptive-threshold window. Higher handles uneven lighting better, but can be slower and more permissive.",
    "Thresh step": "Step between threshold window sizes. Lower scans more options, which is more forgiving but heavier.",
    "Thresh C": "Adaptive-threshold constant. Lower accepts darker/low-contrast cells; too low can create false candidates.",
    "Min size": "Smallest marker perimeter relative to the image. Lower allows smaller/farther markers; too low starts accepting noise.",
    "Shape tol": "Quadrilateral shape tolerance. Higher accepts more perspective distortion; too high allows bad shapes.",
    "Err corr": "ArUco dictionary bit-error correction. Higher accepts damaged markers; too high risks false marker IDs.",
    "Loop ms": "Delay between AR UI refreshes. Higher lowers CPU load and stutter; lower feels more live.",
    "Expected IDs": "When enabled, markers outside the structure and hammer lists are ignored after decoding.",
}


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
        self.control_panel_visible_var = tk.BooleanVar(value=True)
        self.control_section_vars = {
            key: tk.BooleanVar(value=True) for key, _label in CONTROL_PANEL_SECTIONS
        }
        self.control_section_frames: dict[str, ttk.LabelFrame] = {}
        self._main_paned: Optional[ttk.PanedWindow] = None
        self._controls_outer: Optional[ttk.Frame] = None
        self._controls_canvas: Optional[tk.Canvas] = None
        self._controls_window_id: Optional[int] = None
        self._flt_photo: Optional[ImageTk.PhotoImage] = None
        self._calibration_meta: dict[str, object] = {}

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

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="3D Preview", command=lambda: self._select_workspace_tab("3d"))
        view_menu.add_command(label="AR View", command=lambda: self._select_workspace_tab("ar"))
        view_menu.add_command(label="Filtered View", command=lambda: self._select_workspace_tab("flt"))
        view_menu.add_separator()
        control_menu = tk.Menu(view_menu, tearoff=0)
        control_menu.add_checkbutton(
            label="Show panel",
            variable=self.control_panel_visible_var,
            command=self._toggle_control_panel,
        )
        control_menu.add_separator()
        for key, label in CONTROL_PANEL_SECTIONS:
            control_menu.add_checkbutton(
                label=label,
                variable=self.control_section_vars[key],
                command=self._apply_control_section_visibility,
            )
        view_menu.add_cascade(label="Control Panel", menu=control_menu)
        menubar.add_cascade(label="View", menu=view_menu)

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

    def _select_workspace_tab(self, tab_name: str) -> None:
        if not hasattr(self, "notebook"):
            return
        tabs = {
            "3d": getattr(self, "preview_tab", None),
            "ar": getattr(self, "ar_tab", None),
            "flt": getattr(self, "flt_tab", None),
        }
        tab = tabs.get(tab_name)
        if tab is not None:
            self.notebook.select(tab)

    def _toggle_control_panel(self) -> None:
        if self._main_paned is None or self._controls_outer is None:
            return
        visible = self.control_panel_visible_var.get()
        panes = set(str(pane) for pane in self._main_paned.panes())
        controls_path = str(self._controls_outer)
        if visible and controls_path not in panes:
            self._main_paned.insert(0, self._controls_outer, weight=0)
        elif not visible and controls_path in panes:
            self._main_paned.forget(self._controls_outer)

    def _register_control_section(self, key: str, frame: ttk.LabelFrame) -> None:
        self.control_section_frames[key] = frame
        self._apply_control_section_visibility()

    def _apply_control_section_visibility(self) -> None:
        if not hasattr(self, "control_section_frames"):
            return
        for key, _label in CONTROL_PANEL_SECTIONS:
            frame = self.control_section_frames.get(key)
            if frame is not None and frame.winfo_manager():
                frame.pack_forget()
        for key, _label in CONTROL_PANEL_SECTIONS:
            frame = self.control_section_frames.get(key)
            var = self.control_section_vars.get(key)
            if frame is not None and (var is None or var.get()):
                frame.pack(fill=tk.X, padx=4, pady=2)
        self._on_controls_frame_configure()

    def _build_layout(self) -> None:
        # Main paned window: left panel | right panel
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._main_paned = pw

        # ── Left panel: controls ──────────────────────────────────────────
        self._controls_outer = ttk.Frame(pw, width=360)
        pw.add(self._controls_outer, weight=0)
        self._controls_canvas = tk.Canvas(self._controls_outer, highlightthickness=0, width=360)
        controls_scroll = ttk.Scrollbar(
            self._controls_outer,
            orient=tk.VERTICAL,
            command=self._controls_canvas.yview,
        )
        self._controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left = ttk.Frame(self._controls_canvas)
        self._controls_window_id = self._controls_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", self._on_controls_frame_configure)
        self._controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        left.bind("<Enter>", self._bind_control_panel_wheel)
        left.bind("<Leave>", self._unbind_control_panel_wheel)

        # Camera
        cam_frame = ttk.LabelFrame(left, text="Camera")
        cam_frame.pack(fill=tk.X, padx=4, pady=2)
        self._register_control_section("camera", cam_frame)

        ttk.Label(cam_frame, text="Device:").grid(row=0, column=0, sticky="w", padx=4)
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(cam_frame, textvariable=self.camera_var, width=18, state="readonly")
        self.camera_combo.grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(cam_frame, text="Refresh", command=self._refresh_cameras, width=7).grid(row=0, column=2, padx=2)
        self._refresh_cameras()

        # Calibration status
        cal_frame = ttk.LabelFrame(left, text="Calibration")
        cal_frame.pack(fill=tk.X, padx=4, pady=2)
        self._register_control_section("calibration", cal_frame)
        self.cal_status_var = tk.StringVar(value="Not loaded")
        ttk.Label(cal_frame, textvariable=self.cal_status_var, wraplength=300).pack(anchor="w", padx=4, pady=2)
        ttk.Button(cal_frame, text="Calibrate (ChArUco)...", command=self._start_calibration).pack(anchor="w", padx=4, pady=2)
        ttk.Button(cal_frame, text="ArUco tutorial...", command=self._show_aruco_wizard).pack(anchor="w", padx=4, pady=2)
        ttk.Button(cal_frame, text="Load calibration file...", command=self._load_calibration_file).pack(anchor="w", padx=4, pady=2)

        # Geometry
        geo_frame = ttk.LabelFrame(left, text="Geometry (UNV)")
        geo_frame.pack(fill=tk.X, padx=4, pady=2)
        self._register_control_section("geometry", geo_frame)
        self.geo_status_var = tk.StringVar(value="No file loaded")
        ttk.Label(geo_frame, textvariable=self.geo_status_var, wraplength=300).pack(anchor="w", padx=4, pady=2)
        btn_row = ttk.Frame(geo_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(btn_row, text="Load UNV...", command=self._load_unv).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Load JSON...", command=self._load_json).pack(side=tk.LEFT, padx=2)

        # Marker config
        mk_frame = ttk.LabelFrame(left, text="Marker ↔ Mesh Positioning")
        mk_frame.pack(fill=tk.X, padx=4, pady=2)
        self._register_control_section("markers", mk_frame)
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
        self._register_control_section("hammer", hammer_frame)
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
        self._register_control_section("ar", ar_frame)
        self.ar_btn = ttk.Button(ar_frame, text="Start AR", command=self._toggle_ar)
        self.ar_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.screenshot_btn = ttk.Button(ar_frame, text="Screenshot", command=self._take_screenshot, state="disabled")
        self.screenshot_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.fullscreen_btn = ttk.Button(ar_frame, text="Fullscreen", command=self._toggle_fullscreen_ar, state="disabled")
        self.fullscreen_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.show_marker_axes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ar_frame, text="Marker axes", variable=self.show_marker_axes_var).pack(side=tk.LEFT, padx=4)
        self.ar_fps_var = tk.StringVar(value="")
        ttk.Label(ar_frame, textvariable=self.ar_fps_var).pack(side=tk.LEFT, padx=8)

        self._build_detection_tuning_controls(left)
        self._build_detection_diagnostics_controls(left)

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

        # Tab 3: filtered detector view
        self.flt_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.flt_tab, text="FLT")
        self.flt_canvas_label = ttk.Label(self.flt_tab, text="Press 'Start AR' to inspect the filtered detector image.")
        self.flt_canvas_label.pack(fill=tk.BOTH, expand=True)

        # Tab 4: synthetic geometry editor
        self.editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.editor_tab, text="Geometry Editor")
        self.geometry_editor = GeometryEditorTab(
            self.editor_tab,
            log=self.log,
            on_send_geometry=self._use_editor_geometry,
            get_current_geometry=lambda: self.geometry_data,
        )

        # ── Bottom: log ───────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.log_text = tk.Text(log_frame, height=6, state="disabled", wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill=tk.X, padx=2, pady=2)

    def _on_controls_frame_configure(self, _event=None) -> None:
        if self._controls_canvas is None:
            return
        self._controls_canvas.configure(scrollregion=self._controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event) -> None:
        if self._controls_canvas is None or self._controls_window_id is None:
            return
        self._controls_canvas.itemconfigure(self._controls_window_id, width=event.width)

    def _bind_control_panel_wheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_control_panel_mousewheel)
        self.root.bind_all("<Button-4>", self._on_control_panel_mousewheel)
        self.root.bind_all("<Button-5>", self._on_control_panel_mousewheel)

    def _unbind_control_panel_wheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_control_panel_mousewheel(self, event) -> None:
        if self._controls_canvas is None:
            return
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self._controls_canvas.yview_scroll(delta, "units")

    def _show_detection_tuning_help(self, label: str) -> None:
        message = DETECTION_TUNING_HELP.get(label, "No description available.")
        messagebox.showinfo(f"Detection Tuning: {label}", message)

    def _build_detection_tuning_controls(self, parent: ttk.Frame) -> None:
        tuning = DETECTOR_TUNING_PRESETS["balanced"]
        tune_frame = ttk.LabelFrame(parent, text="Detection Tuning")
        tune_frame.pack(fill=tk.X, padx=4, pady=2)
        self._register_control_section("tuning", tune_frame)

        self.det_clip_var = tk.DoubleVar(value=tuning.clip_limit)
        self.det_thresh_max_var = tk.IntVar(value=tuning.adaptive_thresh_win_size_max)
        self.det_thresh_step_var = tk.IntVar(value=tuning.adaptive_thresh_win_size_step)
        self.det_thresh_const_var = tk.DoubleVar(value=tuning.adaptive_thresh_constant)
        self.det_min_perim_var = tk.DoubleVar(value=tuning.min_marker_perimeter_rate)
        self.det_poly_var = tk.DoubleVar(value=tuning.polygonal_approx_accuracy_rate)
        self.det_error_var = tk.DoubleVar(value=tuning.error_correction_rate)
        self.det_expected_only_var = tk.BooleanVar(value=True)
        self.ar_loop_ms_var = tk.IntVar(value=33)
        self.det_tuning_status_var = tk.StringVar(value="Balanced")

        fields = (
            ("Contrast", self.det_clip_var, 1.0, 5.0, 0.1),
            ("Thresh max", self.det_thresh_max_var, 3, 73, 2),
            ("Thresh step", self.det_thresh_step_var, 1, 20, 1),
            ("Thresh C", self.det_thresh_const_var, 3.0, 15.0, 0.5),
            ("Min size", self.det_min_perim_var, 0.005, 0.05, 0.001),
            ("Shape tol", self.det_poly_var, 0.02, 0.10, 0.005),
            ("Err corr", self.det_error_var, 0.40, 0.90, 0.05),
            ("Loop ms", self.ar_loop_ms_var, 16, 120, 1),
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
            ttk.Button(
                tune_frame,
                text="(i)",
                width=3,
                command=lambda name=label: self._show_detection_tuning_help(name),
            ).grid(row=row, column=2, sticky="w", padx=(0, 4), pady=1)

        preset_row = ttk.Frame(tune_frame)
        preset_row.grid(row=len(fields), column=0, columnspan=3, sticky="w", padx=2, pady=(4, 1))
        ttk.Button(preset_row, text="Strict", command=lambda: self._set_detection_tuning_preset("strict")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_row, text="Balanced", command=lambda: self._set_detection_tuning_preset("balanced")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_row, text="Forgiving", command=lambda: self._set_detection_tuning_preset("forgiving")).pack(side=tk.LEFT, padx=2)

        apply_row = ttk.Frame(tune_frame)
        apply_row.grid(row=len(fields) + 1, column=0, columnspan=3, sticky="we", padx=2, pady=(2, 4))
        ttk.Checkbutton(apply_row, text="Expected IDs", variable=self.det_expected_only_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            apply_row,
            text="(i)",
            width=3,
            command=lambda: self._show_detection_tuning_help("Expected IDs"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(apply_row, text="Apply", command=self._apply_detection_tuning).pack(side=tk.LEFT, padx=2)
        ttk.Label(apply_row, textvariable=self.det_tuning_status_var).pack(side=tk.LEFT, padx=6)

    def _build_detection_diagnostics_controls(self, parent: ttk.Frame) -> None:
        diag_frame = ttk.LabelFrame(parent, text="Diagnostics")
        diag_frame.pack(fill=tk.X, padx=4, pady=2)
        self._register_control_section("diagnostics", diag_frame)

        self.det_diag_var = tk.StringVar(
            value="Start AR, then use FLT plus these counters to diagnose detection."
        )
        ttk.Label(
            diag_frame,
            textvariable=self.det_diag_var,
            wraplength=320,
            justify=tk.LEFT,
        ).pack(fill=tk.X, anchor="w", padx=4, pady=3)

        btn_row = ttk.Frame(diag_frame)
        btn_row.pack(fill=tk.X, padx=2, pady=(0, 4))
        ttk.Button(btn_row, text="Board spec", command=self._show_board_spec).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Log diagnosis", command=self._log_detection_diagnostics).pack(side=tk.LEFT, padx=2)

    def _read_calibration_metadata(self) -> dict[str, object]:
        data: dict[str, object] = {
            "exists": CALIBRATION_FILE.exists(),
            "path": str(CALIBRATION_FILE),
            "board_cols": BOARD_COLS,
            "board_rows": BOARD_ROWS,
            "square_length_m": CHARUCO_SQUARE_M,
            "marker_length_m": CHARUCO_MARKER_M,
            "image_width": None,
            "image_height": None,
            "rms_error": None,
            "error": None,
        }
        if not CALIBRATION_FILE.exists():
            return data

        fs = cv2.FileStorage(str(CALIBRATION_FILE), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            data["error"] = "Cannot open calibration file"
            return data

        def real_node(name: str, default: float | None) -> float | None:
            node = fs.getNode(name)
            try:
                if node.empty():
                    return default
                return float(node.real())
            except Exception:
                return default

        try:
            data["board_cols"] = int(real_node("board_cols", float(BOARD_COLS)) or BOARD_COLS)
            data["board_rows"] = int(real_node("board_rows", float(BOARD_ROWS)) or BOARD_ROWS)
            data["square_length_m"] = real_node("square_length_m", CHARUCO_SQUARE_M)
            data["marker_length_m"] = real_node("marker_length_m", CHARUCO_MARKER_M)
            width = real_node("image_width", None)
            height = real_node("image_height", None)
            data["image_width"] = int(width) if width else None
            data["image_height"] = int(height) if height else None
            data["rms_error"] = real_node("rms_error", None)
        finally:
            fs.release()
        return data

    def _refresh_calibration_diagnostics(self) -> None:
        self._calibration_meta = self._read_calibration_metadata()

    def _current_calibration_rms(self) -> float | None:
        rms = self._calibration_meta.get("rms_error")
        return float(rms) if isinstance(rms, (float, int)) else None

    def _board_spec_text(self) -> str:
        meta = self._calibration_meta or self._read_calibration_metadata()
        cols = int(meta.get("board_cols") or BOARD_COLS)
        rows = int(meta.get("board_rows") or BOARD_ROWS)
        square_mm = float(meta.get("square_length_m") or CHARUCO_SQUARE_M) * 1000.0
        marker_mm = float(meta.get("marker_length_m") or CHARUCO_MARKER_M) * 1000.0
        active_w_mm = cols * square_mm
        active_h_mm = rows * square_mm
        rms = meta.get("rms_error")
        image_w = meta.get("image_width")
        image_h = meta.get("image_height")
        calibration_line = "Calibration file: not saved yet."
        if meta.get("exists"):
            rms_text = f"{float(rms):.3f} px" if isinstance(rms, (float, int)) else "not stored"
            size_text = f"{image_w}x{image_h}" if image_w and image_h else "unknown image size"
            calibration_line = f"Calibration file: {CALIBRATION_FILE.name}, RMS {rms_text}, {size_text}."
        if meta.get("error"):
            calibration_line += f"\nWarning: {meta['error']}"

        try:
            structure_marker_mm = float(self.marker_size_var.get())
        except (tk.TclError, ValueError):
            structure_marker_mm = MARKER_SIZE_MM
        try:
            hammer_marker_mm = float(self.hammer_marker_size_var.get())
        except (tk.TclError, ValueError):
            hammer_marker_mm = self.hammer_marker_size_mm

        return (
            "ChArUco board ruler check\n\n"
            f"Squares: {cols} x {rows}\n"
            f"Each black/white chessboard square: {square_mm:.1f} mm per side\n"
            f"ArUco code inside each marker square: {marker_mm:.1f} mm per side\n"
            f"Active grid: {active_w_mm:.1f} x {active_h_mm:.1f} mm\n\n"
            "Print/check notes:\n"
            "- Print at 100% scale, no fit-to-page scaling.\n"
            "- Measure several squares with a ruler; they should match the square size above.\n"
            "- Measure the black ArUco code edge-to-edge; it should match the marker size above.\n"
            "- If your print measures differently, recalibrate with the measured square/marker values.\n\n"
            "Detection note:\n"
            "- Camera calibration does not make ArUco IDs decode better; it affects pose/overlay scale and stability after decoding.\n\n"
            f"{calibration_line}\n\n"
            "Structure marker sizes used by AR:\n"
            f"- Structure markers: {structure_marker_mm:.1f} mm\n"
            f"- Hammer markers: {hammer_marker_mm:.1f} mm"
        )

    def _show_board_spec(self) -> None:
        self._refresh_calibration_diagnostics()
        messagebox.showinfo("Board / Calibration Check", self._board_spec_text())

    def _detection_health_hint(
        self,
        result: FrameResult,
        structure_seen: int,
        pose_seen: int,
    ) -> str:
        hints: list[str] = []
        if result.raw_marker_count == 0 and result.rejected_count > 0:
            hints.append("marker-like shapes are being rejected: check blur, glare, borders, print quality, and dictionary")
        elif result.raw_marker_count == 0:
            hints.append("no marker candidates: move closer, improve light, or check the FLT image contrast")
        if result.raw_marker_count > result.allowed_marker_count:
            hints.append("some decoded markers are filtered out by Expected IDs")
        if 0 < result.mean_marker_area_px < 900:
            hints.append("markers are very small in the image; move closer or use larger print")
        if (
            structure_seen > 0
            and pose_seen < MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE
            and result.lock_state == "searching"
        ):
            hints.append(
                f"need {MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE}+ structure markers to acquire pose"
                " (1 sustains it once locked)"
            )
        rms = self._current_calibration_rms()
        if rms is None and not self.calibration_loaded:
            hints.append("no camera calibration loaded")
        elif rms is not None and rms > 1.0:
            hints.append("calibration RMS is high; repeat calibration with sharper, varied board views")
        return "; ".join(hints) if hints else "detection looks healthy; instability is likely pose geometry, motion blur, or too few structure markers"

    def _update_detection_diagnostics(
        self,
        result: FrameResult,
        structure_seen: int,
        hammer_seen: int,
        pose_seen: int,
    ) -> None:
        if not hasattr(self, "det_diag_var"):
            return
        expected = self._expected_detection_ids()
        expected_text = "off" if expected is None else str(len(expected))
        rms = self._current_calibration_rms()
        rms_text = "n/a" if rms is None else f"{rms:.3f} px"
        mode = "flow" if result.used_optical_flow else "detect"
        hint = self._detection_health_hint(result, structure_seen, pose_seen)
        recovered_bits = []
        if result.refine_recovered_count:
            recovered_bits.append(f"refine +{result.refine_recovered_count}")
        if result.roi_recovered_count:
            recovered_bits.append(f"roi +{result.roi_recovered_count}")
        if result.carryover_count:
            recovered_bits.append(f"carry +{result.carryover_count}")
        recovered_text = f" | Recovered: {', '.join(recovered_bits)}" if recovered_bits else ""
        lock_text = result.lock_state
        if result.lock_reject_reason:
            lock_text += f" (reject: {result.lock_reject_reason})"
        self.det_diag_var.set(
            f"Mode: {mode} | Lock: {lock_text}{recovered_text} | Expected IDs: {expected_text}\n"
            f"Decoded: {result.raw_marker_count} raw / {result.allowed_marker_count} accepted\n"
            f"Rejected candidates: {result.rejected_count} | Avg area: {result.mean_marker_area_px:.0f} px^2\n"
            f"Structure: {structure_seen} | Pose markers: {pose_seen} | Hammer: {hammer_seen}\n"
            f"Calibration RMS: {rms_text}\n"
            f"Hint: {hint}"
        )

    def _log_detection_diagnostics(self) -> None:
        if not hasattr(self, "det_diag_var"):
            return
        self.log(f"Detection diagnostics: {self.det_diag_var.get().replace(chr(10), ' | ')}")

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
        self._refresh_calibration_diagnostics()

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
            self._refresh_calibration_diagnostics()
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
            image_size, rms, BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M,
        )
        self.cal_status_var.set(f"Calibrated — RMS: {rms:.3f} px")
        self._refresh_calibration_diagnostics()
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

    def _use_editor_geometry(self, data: dict) -> None:
        """Adopt geometry handed over by the Geometry Editor tab."""
        self.geometry_data = data
        self.geometry_path = None
        n = len(data.get("nodes", []))
        e = len(data.get("traceLines", []))
        self.geo_status_var.set(f"Geometry editor: {n} nodes, {e} edges")
        self.log(f"Loaded geometry from editor ({n} nodes, {e} edges)")
        self._update_3d_preview()
        self.notebook.select(self.preview_tab)

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
        self.flt_canvas_label.configure(image="", text="AR stopped.")
        if hasattr(self, "det_diag_var"):
            self.det_diag_var.set("AR stopped.")
        self.log("AR overlay stopped.")

    def _ar_loop(self) -> None:
        if not self.ar_running or self.pipeline is None:
            return

        result = self.pipeline.process_frame()
        if result is not None:
            registration_result = None
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

            status_text = self._ar_status_text(result, structure_seen, hammer_seen, pose_seen)
            self._update_detection_diagnostics(result, structure_seen, hammer_seen, pose_seen)
            show_ar = self._is_workspace_tab_selected(self.ar_tab) or self._fullscreen_is_open()
            show_flt = self._is_workspace_tab_selected(self.flt_tab)

            if show_ar:
                vis = self.pipeline.draw_overlay(
                    result, draw_markers=True, draw_axes=True,
                    draw_marker_axes=bool(self.show_marker_axes_var.get()),
                )
                self._draw_marker_roles(vis, result)
                cv2.putText(vis, status_text, (10, vis.shape[0] - 15),
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
                self._last_ar_frame = vis

            if show_flt:
                self._update_filtered_display(result, status_text)

            # FPS
            self.ar_fps_var.set(
                f"FPS: {result.fps:.1f} | S {structure_seen} | Pose {pose_seen} | H {hammer_seen}"
            )

        self._ar_after_id = self.root.after(self._ar_loop_delay_ms(), self._ar_loop)

    def _is_workspace_tab_selected(self, tab: ttk.Frame) -> bool:
        try:
            return hasattr(self, "notebook") and self.notebook.select() == str(tab)
        except tk.TclError:
            return False

    def _ar_loop_delay_ms(self) -> int:
        try:
            return max(16, min(250, int(self.ar_loop_ms_var.get())))
        except (tk.TclError, ValueError):
            return 33

    def _ar_status_text(
        self,
        result: FrameResult,
        structure_seen: int,
        hammer_seen: int,
        pose_seen: int,
    ) -> str:
        if result.pose:
            t = result.pose.tvec.flatten()
            coasting = "  |  COASTING" if result.pose.coasted else ""
            return (
                f"T: [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm{coasting}"
                f"  |  Pose markers: {pose_seen}  |  Structure: {structure_seen}  |  Hammer: {hammer_seen}"
            )
        if structure_seen:
            return (
                f"Need {MIN_STRUCTURE_MARKERS_FOR_BOARD_POSE}+ structure markers to acquire pose"
                f"  |  Structure: {structure_seen}  |  Hammer: {hammer_seen}"
            )
        return f"Structure: {structure_seen}  |  Hammer: {hammer_seen}"

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

    def _update_filtered_display(self, result: FrameResult, status_text: str) -> None:
        vis = cv2.cvtColor(result.gray, cv2.COLOR_GRAY2BGR)
        self._draw_marker_roles(vis, result)
        cv2.putText(vis, "FLT: CLAHE grayscale passed to ArUco", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        diag_text = (
            f"Decoded {result.raw_marker_count}/{result.allowed_marker_count}"
            f" | Rejected {result.rejected_count}"
            f" | Area {result.mean_marker_area_px:.0f}px^2"
        )
        cv2.putText(vis, diag_text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, status_text, (10, vis.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        max_w = max(PREVIEW_W, self.flt_canvas_label.winfo_width())
        max_h = max(PREVIEW_H, self.flt_canvas_label.winfo_height())
        display_rgb = self._resize_rgb_for_box(rgb, max_w, max_h)
        self._flt_photo = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        self.flt_canvas_label.configure(image=self._flt_photo, text="")

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
        if not (self._is_workspace_tab_selected(self.ar_tab) or self._fullscreen_is_open()):
            # AR rendering is skipped while another tab is selected, so the
            # stored frame may be stale.
            self.log("Note: AR view not active — screenshot uses the last rendered AR frame.")
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
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    EyeLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
