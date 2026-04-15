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
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# Matplotlib for 3D preview (embedded in tkinter)
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# EyeLab modules
from calibrate import load_calibration, save_calibration, make_charuco_board
from eyelab_logger import SessionLogger
from generate_markers import generate_markers, MARKER_SIZE_MM, GRID_SPACING_MM
from pose_estimator import ArucoPipeline, ThreadedCapture, FrameResult
from registration import (
    SpatialRegistration,
    MarkerCorrespondence,
    load_marker_config,
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
CALIBRATION_FILE = CONFIG_DIR / "camera_params.yaml"
MARKER_CONFIG_FILE = CONFIG_DIR / "marker_config.json"


# ── Helper: list available cameras ────────────────────────────────────────────

def list_cameras(max_test: int = 8) -> list[int]:
    """Probe camera indices 0..max_test and return those that open successfully."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


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

        self.registration = SpatialRegistration([])
        self.correspondences: list[MarkerCorrespondence] = []
        self._camera_indices: list[int] = []

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_menu()
        self._build_layout()
        self._load_persistent_state()

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
        tools_menu.add_command(label="Calibrate camera...", command=self._start_calibration)
        menubar.add_cascade(label="Tools", menu=tools_menu)

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

        # AR controls
        ar_frame = ttk.LabelFrame(left, text="AR Overlay")
        ar_frame.pack(fill=tk.X, padx=4, pady=2)
        self.ar_btn = ttk.Button(ar_frame, text="Start AR", command=self._toggle_ar)
        self.ar_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.screenshot_btn = ttk.Button(ar_frame, text="Screenshot", command=self._take_screenshot, state="disabled")
        self.screenshot_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.ar_fps_var = tk.StringVar(value="")
        ttk.Label(ar_frame, textvariable=self.ar_fps_var).pack(side=tk.LEFT, padx=8)

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

    def _build_3d_preview(self, parent: ttk.Frame) -> None:
        self.fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("No geometry loaded")
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=parent)
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

    def _load_calibration_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select calibration YAML",
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

    def _on_calibration_done(self, cam_matrix: np.ndarray, dist_coeffs: np.ndarray, rms: float) -> None:
        self.camera_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.calibration_loaded = True
        save_calibration(
            str(CALIBRATION_FILE), cam_matrix, dist_coeffs,
            (1280, 720), rms, 5, 7, 0.025, 0.019,
        )
        self.cal_status_var.set(f"Calibrated — RMS: {rms:.3f} px")
        self.log(f"Calibration complete. RMS: {rms:.4f} px. Saved to {CALIBRATION_FILE}")

    # ══════════════════════════════════════════════════════════════════════
    #  Geometry loading
    # ══════════════════════════════════════════════════════════════════════

    def _load_unv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select UNV file",
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
            self.ax3d.scatter([p[0]], [p[1]], [p[2]], c="red", s=60, marker="^",
                             zorder=10, label=f"aruco{corr.marker_id + 1:02d}")

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        title = self.geometry_path.name if self.geometry_path else "Geometry"
        self.ax3d.set_title(f"{title} — {len(nodes)} nodes, {len(edges)} edges")
        self.canvas_3d.draw()

    # ══════════════════════════════════════════════════════════════════════
    #  Marker generation
    # ══════════════════════════════════════════════════════════════════════

    def _show_marker_gen(self) -> None:
        MarkerGenWindow(self.root, self)

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

        cam_idx = self._get_camera_index()
        try:
            self.pipeline = ArucoPipeline(
                camera_index=cam_idx,
                calibration_path=str(CALIBRATION_FILE),
                marker_size_mm=self.marker_size_var.get(),
            )
            self.pipeline.start()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            return

        self.ar_running = True
        self.ar_btn.configure(text="Stop AR")
        self.screenshot_btn.configure(state="normal")
        self.notebook.select(self.ar_tab)
        self.log("AR overlay started.")
        self._ar_loop()

    def _stop_ar(self) -> None:
        self.ar_running = False
        if self._ar_after_id is not None:
            self.root.after_cancel(self._ar_after_id)
            self._ar_after_id = None
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.ar_btn.configure(text="Start AR")
        self.screenshot_btn.configure(state="disabled")
        self.ar_fps_var.set("")
        self.ar_canvas_label.configure(image="", text="AR stopped.")
        self.log("AR overlay stopped.")

    def _ar_loop(self) -> None:
        if not self.ar_running or self.pipeline is None:
            return

        result = self.pipeline.process_frame()
        if result is not None:
            # Draw overlay
            vis = self.pipeline.draw_overlay(result, draw_markers=True, draw_axes=True)

            # Draw wireframe if geometry loaded and registration done
            if self.geometry_data and result.pose and self.registration.is_registered:
                self._draw_registered_wireframe(vis, result)

            # Display registration info on frame
            if result.pose:
                t = result.pose.tvec.flatten()
                info = f"T: [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm  |  Markers: {result.pose.marker_count}"
                cv2.putText(vis, info, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # Update registration detected positions
                for m in result.markers:
                    if m.rvec is not None and m.tvec is not None:
                        self.registration.update_detected_position(
                            m.marker_id, m.tvec.flatten()
                        )

            # Convert to tkinter image
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            # Resize to fit
            h, w = rgb.shape[:2]
            scale = min(PREVIEW_W / w, PREVIEW_H / h, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            pil_img = Image.fromarray(rgb)
            self._ar_photo = ImageTk.PhotoImage(pil_img)
            self.ar_canvas_label.configure(image=self._ar_photo, text="")

            # FPS
            self.ar_fps_var.set(f"FPS: {result.fps:.1f}")

            # Store last frame for screenshot
            self._last_ar_frame = vis

        self._ar_after_id = self.root.after(16, self._ar_loop)  # ~60 Hz GUI refresh

    def _draw_registered_wireframe(self, vis: np.ndarray, result: FrameResult) -> None:
        """Project the registered wireframe onto the AR frame."""
        if not self.registration.is_registered or self.camera_matrix is None:
            return

        nodes = self.geometry_data.get("nodes", [])
        edges = self.geometry_data.get("traceLines", [])
        if not nodes or not edges:
            return

        # Transform UNV nodes → world via registration, then project to image
        node_map = {}
        for n in nodes:
            unv_pt = np.array([n["x"], n["y"], n["z"]], dtype=np.float64)
            world_pt = self.registration.transform_point(unv_pt)
            if world_pt is not None:
                node_map[n["id"]] = world_pt

        if not node_map:
            return

        # Project world points using the camera pose from ArUco detection
        unique_ids = list({nid for e in edges for nid in e} & node_map.keys())
        if not unique_ids:
            return

        pts_3d = np.array([node_map[nid] for nid in unique_ids], dtype=np.float32)
        # Use identity pose since points are already in camera frame via registration
        rvec_zero = np.zeros((3, 1), dtype=np.float32)
        tvec_zero = np.zeros((3, 1), dtype=np.float32)
        pts_2d, _ = cv2.projectPoints(
            pts_3d.reshape(-1, 1, 3), rvec_zero, tvec_zero,
            self.camera_matrix, self.dist_coeffs,
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(int)
        id_to_px = {nid: tuple(pts_2d[i]) for i, nid in enumerate(unique_ids)}

        for a_id, b_id in edges:
            if a_id in id_to_px and b_id in id_to_px:
                cv2.line(vis, id_to_px[a_id], id_to_px[b_id],
                         (0, 220, 255), 1, cv2.LINE_AA)

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

        ttk.Label(self.win, text=f"Marker size: {MARKER_SIZE_MM} mm | Grid: {GRID_SPACING_MM} mm").grid(
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
            generate_markers(str(MARKERS_DIR), dpi=dpi, marker_ids=ids, add_id_label=True)
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
        self.src_var = tk.StringVar(value="")
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
        d = filedialog.askdirectory(title="Select marker source directory")
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
        self.win.geometry("520x420")
        self.win.transient(parent)

        ttk.Label(self.win, text=(
            "Assign ArUco markers to UNV node positions.\n"
            "Each marker placed on the physical structure must be linked\n"
            "to its UNV node ID so registration can compute the alignment."
        ), wraplength=500, justify="left").pack(padx=8, pady=6)

        # Treeview for correspondences
        cols = ("marker", "node_id", "x", "y", "z", "desc")
        self.tree = ttk.Treeview(self.win, columns=cols, show="headings", height=10)
        self.tree.heading("marker", text="Marker")
        self.tree.heading("node_id", text="Node ID")
        self.tree.heading("x", text="X (m)")
        self.tree.heading("y", text="Y (m)")
        self.tree.heading("z", text="Z (m)")
        self.tree.heading("desc", text="Description")
        for c in cols:
            self.tree.column(c, width=75)
        self.tree.column("desc", width=120)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Populate
        for corr in self.app.correspondences:
            p = corr.unv_position
            self.tree.insert("", "end", values=(
                f"aruco{corr.marker_id + 1:02d}", corr.node_id or "",
                f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", corr.description,
            ))

        btn_row = ttk.Frame(self.win)
        btn_row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_row, text="Add", command=self._add).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Remove", command=self._remove).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Pick from mesh...", command=self._pick_from_mesh).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Save & Close", command=self._save).pack(side=tk.RIGHT, padx=4)

    def _add(self) -> None:
        AddCorrespondenceDialog(self.win, self.app, self.tree)

    def _remove(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        for item in sel:
            idx = self.tree.index(item)
            self.tree.delete(item)
            if idx < len(self.app.correspondences):
                self.app.correspondences.pop(idx)

    def _pick_from_mesh(self) -> None:
        """Let user select a node from the loaded geometry as the UNV position."""
        if self.app.geometry_data is None:
            messagebox.showwarning("No Geometry", "Load a UNV or JSON file first.")
            return
        NodePickerDialog(self.win, self.app, self.tree)

    def _save(self) -> None:
        # Rebuild correspondences from treeview
        self.app.correspondences = []
        for item in self.tree.get_children():
            vals = self.tree.item(item, "values")
            # Parse marker name back to ID
            marker_name = vals[0]  # "aruco01"
            try:
                marker_id = int(marker_name.replace("aruco", "")) - 1
            except ValueError:
                continue
            node_id = int(vals[1]) if vals[1] else None
            x, y, z = float(vals[2]), float(vals[3]), float(vals[4])
            desc = vals[5]
            self.app.correspondences.append(MarkerCorrespondence(
                marker_id=marker_id,
                unv_position=np.array([x, y, z], dtype=np.float64),
                node_id=node_id,
                description=desc,
            ))
        self.app._save_correspondences()
        self.app.log(f"Saved {len(self.app.correspondences)} correspondences.")
        self.win.destroy()


class AddCorrespondenceDialog:
    """Small dialog to add a single marker ↔ position correspondence."""

    def __init__(self, parent: tk.Toplevel, app: EyeLabApp, tree: ttk.Treeview):
        self.app = app
        self.tree = tree
        self.win = tk.Toplevel(parent)
        self.win.title("Add Correspondence")
        self.win.geometry("300x220")
        self.win.transient(parent)

        ttk.Label(self.win, text="ArUco marker number (1–50):").grid(row=0, column=0, padx=8, pady=4, sticky="w")
        self.mid_var = tk.IntVar(value=1)
        ttk.Spinbox(self.win, from_=1, to=50, textvariable=self.mid_var, width=6).grid(row=0, column=1, padx=8)

        ttk.Label(self.win, text="UNV Node ID (optional):").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self.nid_var = tk.StringVar(value="")
        ttk.Entry(self.win, textvariable=self.nid_var, width=8).grid(row=1, column=1, padx=8)

        for i, axis in enumerate(("X (m):", "Y (m):", "Z (m):")):
            ttk.Label(self.win, text=axis).grid(row=2 + i, column=0, padx=8, pady=2, sticky="w")
        self.x_var = tk.DoubleVar(value=0.0)
        self.y_var = tk.DoubleVar(value=0.0)
        self.z_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.win, textvariable=self.x_var, width=10).grid(row=2, column=1, padx=8)
        ttk.Entry(self.win, textvariable=self.y_var, width=10).grid(row=3, column=1, padx=8)
        ttk.Entry(self.win, textvariable=self.z_var, width=10).grid(row=4, column=1, padx=8)

        ttk.Button(self.win, text="Add", command=self._add).grid(row=5, column=0, columnspan=2, pady=10)

    def _add(self) -> None:
        mid = self.mid_var.get() - 1   # internal 0-indexed
        nid_str = self.nid_var.get().strip()
        node_id = int(nid_str) if nid_str else None
        x, y, z = self.x_var.get(), self.y_var.get(), self.z_var.get()
        self.tree.insert("", "end", values=(
            f"aruco{mid + 1:02d}", node_id or "", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", "",
        ))
        self.win.destroy()


class NodePickerDialog:
    """Let user pick a node from the loaded geometry to use as a correspondence position."""

    def __init__(self, parent: tk.Toplevel, app: EyeLabApp, tree: ttk.Treeview):
        self.app = app
        self.tree = tree
        self.win = tk.Toplevel(parent)
        self.win.title("Pick Node from Mesh")
        self.win.geometry("400x380")
        self.win.transient(parent)

        ttk.Label(self.win, text="Select a node, then assign a marker:").pack(padx=8, pady=4)

        # Node list
        cols = ("id", "x", "y", "z")
        self.node_tree = ttk.Treeview(self.win, columns=cols, show="headings", height=12)
        for c in cols:
            self.node_tree.heading(c, text=c.upper())
            self.node_tree.column(c, width=80)
        self.node_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        for n in app.geometry_data.get("nodes", []):
            self.node_tree.insert("", "end", values=(n["id"], f"{n['x']:.4f}", f"{n['y']:.4f}", f"{n['z']:.4f}"))

        row = ttk.Frame(self.win)
        row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="Assign to aruco #:").pack(side=tk.LEFT)
        self.mid_var = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.mid_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Assign", command=self._assign).pack(side=tk.LEFT, padx=8)

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
            f"aruco{mid + 1:02d}", node_id, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
            f"Node {node_id}",
        ))
        self.app.log(f"Assigned aruco{mid + 1:02d} → Node {node_id} ({x:.4f}, {y:.4f}, {z:.4f})")
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

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.board = make_charuco_board(5, 7, 0.025, 0.019)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.all_corners = []
        self.all_ids = []
        self.image_size = None
        self.min_frames = 15

        self.label = ttk.Label(self.win)
        self.label.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(self.win)
        status.pack(fill=tk.X, padx=8, pady=4)
        self.status_var = tk.StringVar(value=f"Captured: 0/{self.min_frames}  —  SPACE to capture, ESC to finish")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

        self.win.bind("<space>", self._capture)
        self.win.bind("<Escape>", self._finish)
        self._photo = None
        self._running = True
        self._loop()

    def _loop(self) -> None:
        if not self._running:
            return
        ok, frame = self.cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.image_size is None:
                self.image_size = (gray.shape[1], gray.shape[0])
            corners, ids, _, _ = self.detector.detectBoard(gray)
            if corners is not None and ids is not None and len(ids) >= 4:
                cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (700, 520))
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.label.configure(image=self._photo)
            self._gray = gray
        self.win.after(30, self._loop)

    def _capture(self, event=None) -> None:
        if not hasattr(self, "_gray"):
            return
        corners, ids, _, _ = self.detector.detectBoard(self._gray)
        if corners is not None and ids is not None and len(ids) >= 4:
            self.all_corners.append(corners)
            self.all_ids.append(ids)
            n = len(self.all_corners)
            self.status_var.set(f"Captured: {n}/{self.min_frames}  —  SPACE to capture, ESC to finish")

    def _finish(self, event=None) -> None:
        n = len(self.all_corners)
        if n < self.min_frames:
            self.status_var.set(f"Need {self.min_frames} frames (have {n}). Keep capturing.")
            return
        self._running = False
        self.cap.release()

        rms, cam_mat, dist = cv2.aruco.calibrateCameraCharuco(
            self.all_corners, self.all_ids, self.board, self.image_size, None, None,
        )
        self.win.destroy()
        self.callback(cam_mat, dist, rms)

    def _abort(self) -> None:
        self._running = False
        self.cap.release()
        self.win.destroy()


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    EyeLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
