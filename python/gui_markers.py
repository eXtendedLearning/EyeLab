"""Marker generation, loading, and correspondence-editing sub-windows.

Extracted from eyelab_gui.py so the main module only hosts the application
shell. Every dialog receives the running EyeLabApp instance and talks back
through its public state (correspondences, marker_size_var, log, ...).
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Optional

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np

from generate_markers import generate_markers, GRID_SPACING_MM
from gui_common import (
    CONFIG_DIR,
    MARKERS_DIR,
    POSITION_UI_UNIT,
    format_position_ui,
    marker_up_label,
    position_ui_to_m,
)
from registration import (
    AXIS_NORMALS,
    MarkerCorrespondence,
    load_marker_config,
    normal_label,
    save_marker_config,
)

if TYPE_CHECKING:
    from eyelab_gui import EyeLabApp


class MarkerGenWindow:
    """Dialog for generating ArUco markers."""

    def __init__(self, parent: tk.Tk, app: "EyeLabApp"):
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

    def __init__(self, parent: tk.Tk, app: "EyeLabApp"):
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

    def __init__(self, parent: tk.Toplevel, app: "EyeLabApp", tree: ttk.Treeview):
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

    def __init__(self, parent: tk.Toplevel, app: "EyeLabApp", editor: CorrespondenceEditor, item_id):
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

    def __init__(self, parent: tk.Toplevel, app: "EyeLabApp", tree: ttk.Treeview):
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
