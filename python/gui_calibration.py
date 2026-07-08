"""Calibration and help sub-windows for the EyeLab GUI.

Contains the in-app information center, the step-by-step ArUco/ChArUco
calibration wizard, and the live ChArUco calibration window. Extracted from
eyelab_gui.py so the main module only hosts the application shell.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
from PIL import Image, ImageTk

from calibrate import (
    BOARD_COLS,
    BOARD_ROWS,
    make_charuco_board,
    generate_board_image,
)
from camera_utils import open_camera
from gui_common import CHARUCO_SQUARE_M, CHARUCO_MARKER_M

if TYPE_CHECKING:
    from eyelab_gui import EyeLabApp


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


class InfoCenterWindow:
    """Browsable, in-app help for the main EyeLab workflows."""

    def __init__(self, parent: tk.Tk, app: "EyeLabApp"):
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

    def __init__(self, parent: tk.Tk, app: "EyeLabApp"):
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
            board = make_charuco_board(BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M)
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
            board = make_charuco_board(BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M)
            generate_board_image(board, path)
            self.app.log(f"Generated ChArUco board image: {path}")
            messagebox.showinfo(
                "Board Generated",
                f"Saved ChArUco board image:\n{path}\n\n"
                "Print at 100% scale, then verify with a ruler:\n"
                f"- chessboard square: {CHARUCO_SQUARE_M * 1000:.1f} mm\n"
                f"- ArUco code edge: {CHARUCO_MARKER_M * 1000:.1f} mm",
            )
        except Exception as e:
            messagebox.showerror("Board Generation Error", str(e))

    def _start_calibration(self) -> None:
        self.app._start_calibration()


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

        self.board = make_charuco_board(BOARD_COLS, BOARD_ROWS, CHARUCO_SQUARE_M, CHARUCO_MARKER_M)
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
