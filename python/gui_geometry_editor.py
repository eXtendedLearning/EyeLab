#!/usr/bin/env python3
"""
Interactive synthetic-geometry editor for EyeLab.

A 3D point-and-wireframe editor styled after the 3D preview tab:

    - click on the canvas to insert nodes on a selectable working plane;
    - drag to rotate the view (clicks and drags are disambiguated);
    - select a node to fine-tune X/Y/Z with sliders and numeric entries;
    - connect two nodes into trace lines with two clicks;
    - export to .unv (datasets 164 + 2411 + 82 via unv_writer) or JSON,
      or hand the geometry straight to the main EyeLab GUI.

Module-level math helpers (click ray, plane intersection, nearest node) are
tkinter-free so they stay unit-testable in headless environments; the tkinter
import is guarded for the same reason.

Usage:
    Embedded:   GeometryEditorTab(parent_frame, log=..., on_send_geometry=...)
    Standalone: python gui_geometry_editor.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # headless test environments have no tkinter
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover
    tk = None

if tk is not None:  # pragma: no cover - Tk availability is environment-bound
    # The canvas is selected explicitly below; keep MPLBACKEND untouched so
    # importing the geometry helpers remains safe in headless environments.
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

PLANES = ("XY", "XZ", "YZ")
CLICK_DRAG_THRESHOLD_PX = 4.0
PICK_RADIUS_PX = 14.0


# ── Pure geometry helpers (unit-testable, no UI) ─────────────────────────────

def _inv_proj_point(xd: float, yd: float, zd: float, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Homogeneous inverse of matplotlib's 3D projection.

    Implemented directly (instead of ``proj3d.inv_transform``) because the
    matplotlib API changed between versions: older releases took the forward
    matrix, newer ones require the pre-inverted matrix.
    """
    inv = np.linalg.inv(np.asarray(proj_matrix, dtype=np.float64))
    vec = inv @ np.array([xd, yd, zd, 1.0], dtype=np.float64)
    return vec[:3] / vec[3]


def click_ray(xd: float, yd: float, proj_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D click in axes data-projection coords into a 3D ray.

    Args:
        xd, yd: click position after ``ax.transData.inverted()``.
        proj_matrix: ``ax.get_proj()``.

    Returns:
        (origin, direction) of the ray in data coordinates.

    Depth samples are negative: with matplotlib's perspective matrix the
    homogeneous w vanishes at zd=0 (the eye plane) and the camera-front
    frustum maps to zd < 0. Orthographic projections accept any depth.
    """
    p_near = _inv_proj_point(xd, yd, -0.5, proj_matrix)
    p_far = _inv_proj_point(xd, yd, -2.0, proj_matrix)
    direction = p_far - p_near
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    return p_near, direction / norm


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: str,
    offset: float,
) -> Optional[np.ndarray]:
    """
    Intersect a ray with an axis-aligned working plane.

    ``plane`` is "XY" (z=offset), "XZ" (y=offset) or "YZ" (x=offset).
    Returns the (3,) intersection point, or None if the ray is parallel.
    """
    axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane]
    d = float(direction[axis])
    if abs(d) < 1e-12:
        return None
    t = (float(offset) - float(origin[axis])) / d
    return origin + t * direction


def nearest_node_index(
    nodes: list[dict],
    display_xy: np.ndarray,
    node_display_xy: np.ndarray,
    radius_px: float = PICK_RADIUS_PX,
) -> Optional[int]:
    """
    Index of the node whose projected display position is nearest the click.

    Args:
        nodes: node dict list (only used for its length).
        display_xy: (2,) click position in display pixels.
        node_display_xy: (N, 2) projected node positions in display pixels.
        radius_px: maximum pick distance.

    Returns None if no node is within ``radius_px``.
    """
    if not nodes or len(node_display_xy) == 0:
        return None
    dists = np.linalg.norm(node_display_xy - display_xy.reshape(1, 2), axis=1)
    idx = int(np.argmin(dists))
    return idx if dists[idx] <= radius_px else None


def next_node_id(nodes: list[dict]) -> int:
    return max((int(n["id"]) for n in nodes), default=0) + 1


def remove_node(nodes: list[dict], lines: list[list[int]], node_id: int) -> tuple[list[dict], list[list[int]]]:
    """Return copies of (nodes, lines) with the node and its lines removed."""
    nodes = [n for n in nodes if int(n["id"]) != int(node_id)]
    lines = [seg for seg in lines if int(node_id) not in [int(v) for v in seg]]
    return nodes, lines


# ── Editor widget ─────────────────────────────────────────────────────────────

class GeometryEditorTab:
    """Geometry editor embedded in a parent frame (or standalone window)."""

    def __init__(
        self,
        parent,
        log: Callable[[str], None] = print,
        on_send_geometry: Optional[Callable[[dict], None]] = None,
        get_current_geometry: Optional[Callable[[], Optional[dict]]] = None,
    ):
        if tk is None:  # pragma: no cover
            raise RuntimeError("tkinter is not available in this environment")
        self.parent = parent
        self.log = log
        self.on_send_geometry = on_send_geometry
        self.get_current_geometry = get_current_geometry

        self.nodes: list[dict] = []          # {"id", "x", "y", "z"}
        self.lines: list[list[int]] = []     # [node_id_a, node_id_b]
        self._undo_stack: list[tuple[list[dict], list[list[int]]]] = []
        self._selected_id: Optional[int] = None
        self._connect_first_id: Optional[int] = None
        self._press_xy: Optional[tuple[float, float]] = None
        self._updating_widgets = False

        self._build_ui()
        self._redraw()

    # ── UI construction ──

    def _build_ui(self) -> None:
        pw = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pw, width=290)
        pw.add(left, weight=0)
        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        # Mode
        mode_frame = ttk.LabelFrame(left, text="Mode (drag always rotates)")
        mode_frame.pack(fill=tk.X, padx=4, pady=2)
        self.mode_var = tk.StringVar(value="add")
        for text, value in (("Select node", "select"), ("Add node (click)", "add"), ("Connect nodes", "connect")):
            ttk.Radiobutton(mode_frame, text=text, value=value, variable=self.mode_var,
                            command=self._on_mode_change).pack(anchor="w", padx=6)

        # Working plane
        plane_frame = ttk.LabelFrame(left, text="Working plane (for Add)")
        plane_frame.pack(fill=tk.X, padx=4, pady=2)
        self.plane_var = tk.StringVar(value="XY")
        row = ttk.Frame(plane_frame)
        row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Combobox(row, textvariable=self.plane_var, values=PLANES, width=5,
                     state="readonly").pack(side=tk.LEFT)
        ttk.Label(row, text="offset (m):").pack(side=tk.LEFT, padx=(8, 2))
        self.plane_offset_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(row, textvariable=self.plane_offset_var, from_=-10.0, to=10.0,
                    increment=0.01, width=8, format="%.4f").pack(side=tk.LEFT)

        # Node table
        nodes_frame = ttk.LabelFrame(left, text="Nodes")
        nodes_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        columns = ("id", "x", "y", "z")
        self.node_tree = ttk.Treeview(nodes_frame, columns=columns, show="headings", height=8)
        for col, w in zip(columns, (40, 70, 70, 70)):
            self.node_tree.heading(col, text=col.upper())
            self.node_tree.column(col, width=w, anchor="center")
        self.node_tree.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.node_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Precise position editing
        pos_frame = ttk.LabelFrame(left, text="Selected node position (m)")
        pos_frame.pack(fill=tk.X, padx=4, pady=2)
        span_row = ttk.Frame(pos_frame)
        span_row.pack(fill=tk.X, padx=4)
        ttk.Label(span_row, text="Slider span ±").pack(side=tk.LEFT)
        self.span_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(span_row, textvariable=self.span_var, from_=0.01, to=5.0,
                    increment=0.05, width=6, format="%.2f",
                    command=self._recenter_sliders).pack(side=tk.LEFT, padx=4)

        self.axis_vars: dict[str, tk.DoubleVar] = {}
        self.axis_scales: dict[str, ttk.Scale] = {}
        for axis in ("x", "y", "z"):
            arow = ttk.Frame(pos_frame)
            arow.pack(fill=tk.X, padx=4, pady=1)
            ttk.Label(arow, text=axis.upper(), width=2).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            self.axis_vars[axis] = var
            scale = ttk.Scale(arow, from_=-0.25, to=0.25, orient=tk.HORIZONTAL,
                              command=lambda val, a=axis: self._on_slider(a, val))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            self.axis_scales[axis] = scale
            spin = ttk.Spinbox(arow, textvariable=var, from_=-100.0, to=100.0,
                               increment=0.001, width=9, format="%.4f",
                               command=lambda a=axis: self._on_spinbox(a))
            spin.pack(side=tk.LEFT)
            spin.bind("<Return>", lambda _e, a=axis: self._on_spinbox(a))
            spin.bind("<FocusOut>", lambda _e, a=axis: self._on_spinbox(a))

        # Edit actions
        act_frame = ttk.Frame(left)
        act_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(act_frame, text="Delete node", command=self._delete_selected).grid(row=0, column=0, padx=2, pady=1, sticky="we")
        ttk.Button(act_frame, text="Delete last line", command=self._delete_last_line).grid(row=0, column=1, padx=2, pady=1, sticky="we")
        ttk.Button(act_frame, text="Undo", command=self._undo).grid(row=1, column=0, padx=2, pady=1, sticky="we")
        ttk.Button(act_frame, text="Clear all", command=self._clear_all).grid(row=1, column=1, padx=2, pady=1, sticky="we")
        act_frame.columnconfigure((0, 1), weight=1)

        # I/O
        io_frame = ttk.LabelFrame(left, text="Import / Export")
        io_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(io_frame, text="Load EyeLab geometry", command=self._load_from_host).pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(io_frame, text="Load JSON...", command=self._load_json).pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(io_frame, text="Export .unv...", command=self._export_unv).pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(io_frame, text="Export JSON...", command=self._export_json).pack(fill=tk.X, padx=4, pady=1)
        if self.on_send_geometry is not None:
            ttk.Button(io_frame, text="Send to EyeLab viewer", command=self._send_to_host).pack(fill=tk.X, padx=4, pady=1)

        self.status_var = tk.StringVar(value="0 nodes, 0 lines")
        ttk.Label(left, textvariable=self.status_var, wraplength=270).pack(anchor="w", padx=6, pady=2)

        # 3D canvas
        self.fig = plt.Figure(figsize=(6, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # ── Undo / state ──

    def _push_undo(self) -> None:
        self._undo_stack.append((copy.deepcopy(self.nodes), copy.deepcopy(self.lines)))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self.nodes, self.lines = self._undo_stack.pop()
        if self._selected_id is not None and self._node_by_id(self._selected_id) is None:
            self._selected_id = None
        self._connect_first_id = None
        self._refresh_tree()
        self._redraw()

    def _clear_all(self) -> None:
        if self.nodes and not messagebox.askyesno("Clear", "Delete all nodes and lines?"):
            return
        self._push_undo()
        self.nodes = []
        self.lines = []
        self._selected_id = None
        self._connect_first_id = None
        self._refresh_tree()
        self._redraw()

    def _node_by_id(self, node_id: int) -> Optional[dict]:
        for n in self.nodes:
            if int(n["id"]) == int(node_id):
                return n
        return None

    # ── Mouse interaction ──

    def _on_mode_change(self) -> None:
        self._connect_first_id = None
        self._redraw()

    def _on_press(self, event) -> None:
        if event.inaxes is self.ax and event.button == 1:
            self._press_xy = (event.x, event.y)
        else:
            self._press_xy = None

    def _on_release(self, event) -> None:
        if self._press_xy is None or event.inaxes is not self.ax or event.button != 1:
            return
        dx = event.x - self._press_xy[0]
        dy = event.y - self._press_xy[1]
        self._press_xy = None
        if np.hypot(dx, dy) > CLICK_DRAG_THRESHOLD_PX:
            return  # it was a rotation drag
        mode = self.mode_var.get()
        if mode == "add":
            self._add_node_at_click(event)
        elif mode == "connect":
            self._connect_at_click(event)
        else:
            self._select_at_click(event)

    def _add_node_at_click(self, event) -> None:
        try:
            xd, yd = self.ax.transData.inverted().transform((event.x, event.y))
            origin, direction = click_ray(xd, yd, self.ax.get_proj())
        except Exception as e:
            self.log(f"Editor: cannot map click to 3D ({e})")
            return
        point = ray_plane_intersection(
            origin, direction, self.plane_var.get(), self._plane_offset(),
        )
        if point is None:
            self.log("Editor: click ray is parallel to the working plane")
            return
        self._push_undo()
        node = {
            "id": next_node_id(self.nodes),
            "x": round(float(point[0]), 6),
            "y": round(float(point[1]), 6),
            "z": round(float(point[2]), 6),
        }
        self.nodes.append(node)
        self._selected_id = node["id"]
        self._refresh_tree(select=node["id"])
        self._redraw()

    def _connect_at_click(self, event) -> None:
        idx = self._pick_node(event)
        if idx is None:
            return
        node_id = int(self.nodes[idx]["id"])
        if self._connect_first_id is None:
            self._connect_first_id = node_id
        elif self._connect_first_id != node_id:
            pair = [self._connect_first_id, node_id]
            if pair not in self.lines and pair[::-1] not in self.lines:
                self._push_undo()
                self.lines.append(pair)
            self._connect_first_id = None
        else:
            self._connect_first_id = None
        self._redraw()

    def _select_at_click(self, event) -> None:
        idx = self._pick_node(event)
        if idx is None:
            return
        self._selected_id = int(self.nodes[idx]["id"])
        self._refresh_tree(select=self._selected_id)
        self._redraw()

    def _pick_node(self, event) -> Optional[int]:
        if not self.nodes:
            return None
        proj = self.ax.get_proj()
        display = []
        for n in self.nodes:
            x2, y2, _ = proj3d.proj_transform(n["x"], n["y"], n["z"], proj)
            display.append(self.ax.transData.transform((x2, y2)))
        return nearest_node_index(
            self.nodes, np.array([event.x, event.y], dtype=np.float64),
            np.asarray(display, dtype=np.float64),
        )

    def _plane_offset(self) -> float:
        try:
            return float(self.plane_offset_var.get())
        except (tk.TclError, ValueError):
            return 0.0

    # ── Node table / sliders ──

    def _refresh_tree(self, select: Optional[int] = None) -> None:
        self._updating_widgets = True
        try:
            self.node_tree.delete(*self.node_tree.get_children())
            for n in self.nodes:
                self.node_tree.insert(
                    "", "end", iid=str(n["id"]),
                    values=(n["id"], f"{n['x']:.4f}", f"{n['y']:.4f}", f"{n['z']:.4f}"),
                )
            if select is not None and self.node_tree.exists(str(select)):
                self.node_tree.selection_set(str(select))
                self.node_tree.see(str(select))
        finally:
            self._updating_widgets = False
        self._load_selected_into_widgets()
        self._update_status()

    def _on_tree_select(self, _event=None) -> None:
        if self._updating_widgets:
            return
        sel = self.node_tree.selection()
        if sel:
            self._selected_id = int(sel[0])
            self._load_selected_into_widgets()
            self._redraw()

    def _load_selected_into_widgets(self) -> None:
        node = self._node_by_id(self._selected_id) if self._selected_id is not None else None
        self._updating_widgets = True
        try:
            for axis in ("x", "y", "z"):
                value = float(node[axis]) if node else 0.0
                self.axis_vars[axis].set(round(value, 6))
                self._set_scale(axis, value)
        finally:
            self._updating_widgets = False

    def _set_scale(self, axis: str, center: float) -> None:
        span = max(0.01, float(self.span_var.get() or 0.25))
        scale = self.axis_scales[axis]
        scale.configure(from_=center - span, to=center + span)
        scale.set(center)

    def _recenter_sliders(self) -> None:
        self._load_selected_into_widgets()

    def _on_slider(self, axis: str, value: str) -> None:
        if self._updating_widgets:
            return
        node = self._node_by_id(self._selected_id) if self._selected_id is not None else None
        if node is None:
            return
        val = round(float(value), 6)
        node[axis] = val
        self._updating_widgets = True
        try:
            self.axis_vars[axis].set(val)
        finally:
            self._updating_widgets = False
        self._update_tree_row(node)
        self._redraw()

    def _on_spinbox(self, axis: str) -> None:
        if self._updating_widgets:
            return
        node = self._node_by_id(self._selected_id) if self._selected_id is not None else None
        if node is None:
            return
        try:
            val = round(float(self.axis_vars[axis].get()), 6)
        except (tk.TclError, ValueError):
            return
        node[axis] = val
        self._updating_widgets = True
        try:
            self._set_scale(axis, val)  # recenter slider on typed value
        finally:
            self._updating_widgets = False
        self._update_tree_row(node)
        self._redraw()

    def _update_tree_row(self, node: dict) -> None:
        iid = str(node["id"])
        if self.node_tree.exists(iid):
            self.node_tree.item(
                iid,
                values=(node["id"], f"{node['x']:.4f}", f"{node['y']:.4f}", f"{node['z']:.4f}"),
            )
        self._update_status()

    def _delete_selected(self) -> None:
        if self._selected_id is None:
            return
        self._push_undo()
        self.nodes, self.lines = remove_node(self.nodes, self.lines, self._selected_id)
        self._selected_id = None
        self._connect_first_id = None
        self._refresh_tree()
        self._redraw()

    def _delete_last_line(self) -> None:
        if not self.lines:
            return
        self._push_undo()
        self.lines.pop()
        self._redraw()
        self._update_status()

    def _update_status(self) -> None:
        pending = ""
        if self._connect_first_id is not None:
            pending = f" | connecting from node {self._connect_first_id}..."
        self.status_var.set(f"{len(self.nodes)} nodes, {len(self.lines)} lines{pending}")

    # ── Drawing ──

    def _redraw(self) -> None:
        # Preserve the user's viewpoint across redraws
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        title = "Geometry editor — click to add nodes" if not self.nodes else "Geometry editor"
        self.ax.set_title(title, fontsize=10)

        if self.nodes:
            xs = [n["x"] for n in self.nodes]
            ys = [n["y"] for n in self.nodes]
            zs = [n["z"] for n in self.nodes]
            colors = []
            for n in self.nodes:
                if int(n["id"]) == (self._selected_id or -1):
                    colors.append("red")
                elif int(n["id"]) == (self._connect_first_id or -1):
                    colors.append("orange")
                else:
                    colors.append("tab:blue")
            self.ax.scatter(xs, ys, zs, c=colors, s=36, depthshade=False)
            for n in self.nodes:
                self.ax.text(n["x"], n["y"], n["z"], f" {n['id']}", fontsize=8)

            by_id = {int(n["id"]): n for n in self.nodes}
            for a, b in self.lines:
                na, nb = by_id.get(int(a)), by_id.get(int(b))
                if na and nb:
                    self.ax.plot(
                        [na["x"], nb["x"]], [na["y"], nb["y"]], [na["z"], nb["z"]],
                        color="gray", linewidth=1.2,
                    )
            # Cube-ish aspect
            span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 0.1)
            cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)
            half = span / 2 * 1.2
            self.ax.set_xlim(cx - half, cx + half)
            self.ax.set_ylim(cy - half, cy + half)
            self.ax.set_zlim(cz - half, cz + half)
        else:
            self.ax.set_xlim(-0.5, 0.5)
            self.ax.set_ylim(-0.5, 0.5)
            self.ax.set_zlim(-0.5, 0.5)

        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
        self._update_status()

    # ── Import / export ──

    def geometry_dict(self) -> dict:
        """Geometry in the same shape unv_to_json produces (nodes + traceLines)."""
        return {
            "metadata": {
                "sourceFile": "geometry_editor",
                "nodeCount": len(self.nodes),
                "lineCount": len(self.lines),
                "coordinateSystemCount": 0,
            },
            "nodes": [
                {"id": int(n["id"]), "x": float(n["x"]), "y": float(n["y"]),
                 "z": float(n["z"]), "exportCS": 0, "displacementCS": 0}
                for n in self.nodes
            ],
            "traceLines": [[int(a), int(b)] for a, b in self.lines],
            "coordinateSystems": [],
            "units": {"code": 1, "name": "SI"},
        }

    def load_geometry(self, data: dict) -> None:
        nodes = data.get("nodes", [])
        self._push_undo()
        self.nodes = [
            {"id": int(n["id"]), "x": float(n["x"]), "y": float(n["y"]), "z": float(n["z"])}
            for n in nodes
        ]
        self.lines = []
        for seg in data.get("traceLines", []):
            ids = [int(v) for v in seg]
            for a, b in zip(ids, ids[1:]):  # split polylines into segments
                if [a, b] not in self.lines and [b, a] not in self.lines:
                    self.lines.append([a, b])
        self._selected_id = None
        self._connect_first_id = None
        self._refresh_tree()
        self._redraw()
        self.log(f"Editor: loaded {len(self.nodes)} nodes, {len(self.lines)} lines")

    def _load_from_host(self) -> None:
        data = self.get_current_geometry() if self.get_current_geometry else None
        if not data:
            messagebox.showinfo("Geometry editor", "No geometry loaded in EyeLab yet.")
            return
        self.load_geometry(data)

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load geometry JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                self.load_geometry(json.load(f))
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("Geometry editor", f"Cannot load JSON: {e}")

    def _export_unv(self) -> None:
        if not self.nodes:
            messagebox.showinfo("Geometry editor", "Nothing to export — add nodes first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export UNV",
            defaultextension=".unv",
            filetypes=[("Universal File", "*.unv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            from unv_writer import write_unv
            write_unv(path, self.nodes, self.lines)
        except Exception as e:
            messagebox.showerror("Geometry editor", f"UNV export failed: {e}")
            return
        self.log(f"Editor: exported {len(self.nodes)} nodes, {len(self.lines)} lines -> {Path(path).name}")

    def _export_json(self) -> None:
        if not self.nodes:
            messagebox.showinfo("Geometry editor", "Nothing to export — add nodes first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export geometry JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self.geometry_dict(), f, indent=2)
        except OSError as e:
            messagebox.showerror("Geometry editor", f"JSON export failed: {e}")
            return
        self.log(f"Editor: exported JSON -> {Path(path).name}")

    def _send_to_host(self) -> None:
        if not self.nodes:
            messagebox.showinfo("Geometry editor", "Nothing to send — add nodes first.")
            return
        if self.on_send_geometry is not None:
            self.on_send_geometry(self.geometry_dict())


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover
    root = tk.Tk()
    root.title("EyeLab — Geometry Editor")
    root.geometry("1100x700")
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    GeometryEditorTab(frame)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
