#!/usr/bin/env python3
"""
Camera calibration for EyeLab using a ChArUco board.

A ChArUco board combines a checkerboard with ArUco markers, giving more
calibration corners per frame and robust sub-pixel accuracy.

Workflow
--------
1. Generate the calibration board image (--generate):
       python calibrate.py --generate --output charuco_board.png
   Print it on A4 / letter paper and measure the actual square size.

2. Collect calibration images or use live webcam:
   a) From image files:
       python calibrate.py --images calib_images/*.jpg --square 0.025 --marker 0.019
   b) Live webcam (press SPACE to capture, ESC when done):
       python calibrate.py --live --camera 0 --square 0.025 --marker 0.019

   --square  physical size of one checkerboard square (metres)
   --marker  physical size of the ArUco marker inside the square (metres)

3. Results are written to camera_params.yaml in OpenCV YAML format,
   portable to Phase 2 (Unity / XREAL).

Board spec (default):
    5 × 7 squares, DICT_4X4_50, matching the flangia grid spacing.

Acceptance criteria:
    RMS reprojection error < 1.0 pixel.
"""

import argparse

from eyelab_version import VERSION_STRING
import os
import sys
from pathlib import Path

CALIBRATION_WIZARD_TEXT = """
EyeLab ChArUco calibration wizard
=================================

1. Generate and print the board
   python calibrate.py --generate --board-image charuco_board.pdf
   Print at 100% scale. Do not use 'fit to page'.

2. Measure or confirm physical sizes
   Defaults are square=0.025 m and marker=0.019 m. If your print is
   different, pass the measured values with --square and --marker.

3. Capture live frames
   python calibrate.py --live --camera 0 --output config/camera_params.yaml
   Hold the ChArUco board in front of the camera, not on the structure.
   Press SPACE only when corners are detected. Capture at least 15 frames:
   center, corners, near, far, and tilted views.

4. Finish and check RMS
   Press ESC after enough frames are captured. Aim for RMS < 1.0 px.
   If RMS is high, repeat with sharper, more varied board views.

5. Use the result in EyeLab
   Load the generated camera_params.yaml in the GUI, then run marker
   correspondences and AR overlay. Structure ArUco markers are configured
   separately: centre position, face normal, roll, and physical size.
""".strip()

if Path(sys.argv[0]).name.lower() == "calibrate.py" and "--wizard" in sys.argv[1:]:
    print(CALIBRATION_WIZARD_TEXT)
    sys.exit(0)

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
from dataclasses import dataclass
from typing import Optional

import numpy as np

from camera_utils import open_camera


# ── Board defaults ────────────────────────────────────────────────────────────
CHARUCO_SQUARE_M = 0.025   # default printed square side (m); see gui_common
BOARD_COLS = 5          # number of squares in X
BOARD_ROWS = 7          # number of squares in Y
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MIN_FRAMES = 15         # minimum captured frames for calibration


# ── Board construction ────────────────────────────────────────────────────────

def make_charuco_board(
    cols: int,
    rows: int,
    square_m: float,
    marker_m: float,
) -> cv2.aruco.CharucoBoard:
    """Return a CharucoBoard with DICT_4X4_50."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (cols, rows),
        squareLength=square_m,
        markerLength=marker_m,
        dictionary=dictionary,
    )
    return board


# ── Image generation ──────────────────────────────────────────────────────────

def generate_board_image(
    board: cv2.aruco.CharucoBoard,
    output_path: str,
    dpi: int = 300,
    page_width_mm: float = 210.0,
    page_height_mm: float = 297.0,
    margin_mm: float = 10.0,
    square_m: float | None = None,
    rows_cols: tuple[int, int] | None = None,
) -> None:
    """Render the ChArUco board at TRUE PHYSICAL SCALE, centred on the page.

    Writes a .png or a .pdf depending on the extension of `output_path`; the PDF
    carries the physical size in its own metadata, so it prints to scale on any
    printer that is not asked to fit-to-page.

    The previous version stretched the board to fill the whole printable area,
    which produced squares of about 38.0 x 39.6 mm — neither square nor the size
    the software reported. That does not corrupt the intrinsics (they are
    invariant to the calibration target's scale), but it makes the GUI's ruler
    check disagree with the printed sheet, which is the only way an operator has
    to confirm the board is right.

    A ruler reference line is printed below the board: measure it, and if it is
    not exactly 100 mm the print was scaled and the sheet should be reprinted.
    """
    cols, rows = rows_cols if rows_cols else (BOARD_COLS, BOARD_ROWS)
    square_mm = (square_m if square_m else CHARUCO_SQUARE_M) * 1000.0

    px_per_mm = dpi / 25.4
    mm2px = lambda mm: int(round(mm * px_per_mm))

    board_w_mm = cols * square_mm
    board_h_mm = rows * square_mm
    avail_w_mm = page_width_mm - 2 * margin_mm
    avail_h_mm = page_height_mm - 2 * margin_mm - 22.0   # room for the ruler strip
    if board_w_mm > avail_w_mm or board_h_mm > avail_h_mm:
        raise ValueError(
            f"A {cols}x{rows} board of {square_mm:.1f} mm squares is "
            f"{board_w_mm:.0f}x{board_h_mm:.0f} mm and does not fit the "
            f"{avail_w_mm:.0f}x{avail_h_mm:.0f} mm printable area. Use smaller "
            f"squares, a smaller grid, or a larger page."
        )

    # Render the board at exactly its physical size — square squares, true scale.
    img = board.generateImage((mm2px(board_w_mm), mm2px(board_h_mm)),
                              marginSize=0, borderBits=1)

    page = np.full((mm2px(page_height_mm), mm2px(page_width_mm)), 255, dtype=np.uint8)
    x0 = (page.shape[1] - img.shape[1]) // 2
    y0 = mm2px(margin_mm + 8.0)
    page[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img

    def label(text, x_mm, y_mm, scale=0.5, thickness=1):
        cv2.putText(page, text, (mm2px(x_mm), mm2px(y_mm)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * dpi / 150.0,
                    0, max(1, int(thickness * dpi / 150.0)), cv2.LINE_AA)

    label(f"EyeLab ChArUco calibration board - {cols}x{rows} squares, "
          f"{square_mm:.1f} mm square, {board.getMarkerLength()*1000:.1f} mm marker, DICT_4X4_50",
          margin_mm, margin_mm + 4.0, scale=0.42)

    # Ruler reference: the single check that the print was not rescaled.
    ruler_y_mm = margin_mm + 8.0 + board_h_mm + 10.0
    rx0, rx1 = mm2px(margin_mm), mm2px(margin_mm + 100.0)
    ry = mm2px(ruler_y_mm)
    cv2.line(page, (rx0, ry), (rx1, ry), 0, max(1, int(dpi / 150)), cv2.LINE_AA)
    for i in range(11):
        tick = mm2px(margin_mm + i * 10.0)
        h = mm2px(3.0 if i % 5 == 0 else 1.8)
        cv2.line(page, (tick, ry - h), (tick, ry), 0, max(1, int(dpi / 200)), cv2.LINE_AA)
    label("|<------------------ this line must measure exactly 100 mm ------------------>|",
          margin_mm, ruler_y_mm + 6.0, scale=0.38)
    label("Print at 100% scale. Disable 'fit to page' / 'shrink oversized pages'.",
          margin_mm, ruler_y_mm + 11.0, scale=0.38)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".pdf":
        from PIL import Image
        Image.fromarray(page).convert("L").save(
            str(out), "PDF", resolution=float(dpi), title="EyeLab ChArUco board"
        )
    else:
        cv2.imwrite(str(out), page)

    print(f"Board saved to: {out}")
    print(f"  Grid        : {cols} x {rows} squares")
    print(f"  Square      : {square_mm:.1f} mm   Marker: {board.getMarkerLength()*1000:.1f} mm")
    print(f"  Board area  : {board_w_mm:.1f} x {board_h_mm:.1f} mm on A4 "
          f"({page_width_mm:.0f} x {page_height_mm:.0f} mm), {dpi} DPI")
    print("  Print at 100% - do NOT scale to fit; verify with the 100 mm ruler line.")


# ── Frame collection helpers ──────────────────────────────────────────────────

def _detect_charuco(
    frame_gray: np.ndarray,
    detector: cv2.aruco.CharucoDetector,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (charuco_corners, charuco_ids) or (None, None) if too few detected."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame_gray)
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None
    return charuco_corners, charuco_ids


def collect_from_images(
    image_paths: list[Path],
    detector: cv2.aruco.CharucoDetector,
) -> tuple[list, list, tuple[int, int]]:
    all_corners, all_ids = [], []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARN: could not read {path.name}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        corners, ids = _detect_charuco(gray, detector)
        if corners is not None:
            all_corners.append(corners)
            all_ids.append(ids)
            print(f"  {path.name}: {len(ids)} corners detected  ✓")
        else:
            print(f"  {path.name}: too few corners, skipped.")

    return all_corners, all_ids, image_size


def collect_from_webcam(
    camera_index: int,
    detector: cv2.aruco.CharucoDetector,
    board: cv2.aruco.CharucoBoard,
    target_frames: int = MIN_FRAMES,
) -> tuple[list, list, tuple[int, int]]:
    cap = open_camera(camera_index, fps=None)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    all_corners, all_ids = [], []
    image_size = None

    print(f"\nLive calibration — camera {camera_index}")
    print("  SPACE  — capture current frame")
    print(f"  ESC    — finish (need at least {target_frames} frames)")
    print("  Q      — abort\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        corners, ids = _detect_charuco(gray, detector)

        # Draw detected corners on preview
        preview = frame.copy()
        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)

        n = len(all_corners)
        status = f"Captured: {n}/{target_frames}"
        color = (0, 200, 0) if n >= target_frames else (0, 140, 255)
        cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)

        if corners is not None:
            cv2.putText(preview, f"Detected {len(ids)} corners — SPACE to capture",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1, cv2.LINE_AA)

        cv2.imshow("EyeLab Calibration", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Aborted.")
            sys.exit(0)

        if key == 27:  # ESC
            if n >= target_frames:
                break
            print(f"  Need at least {target_frames} frames (have {n}). Keep going.")

        if key == ord(" ") and corners is not None:
            all_corners.append(corners)
            all_ids.append(ids)
            print(f"  Frame {n + 1}: {len(ids)} corners captured  ✓")

    cap.release()
    cv2.destroyAllWindows()
    return all_corners, all_ids, image_size


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    all_corners: list,
    all_ids: list,
    board: cv2.aruco.CharucoBoard,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run cv2 camera calibration and return (rms, camera_matrix, dist_coeffs)."""
    if len(all_corners) < MIN_FRAMES:
        raise ValueError(
            f"Need at least {MIN_FRAMES} valid frames, got {len(all_corners)}."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None,
    )
    return rms, camera_matrix, dist_coeffs


# ── YAML I/O ──────────────────────────────────────────────────────────────────

def save_calibration(
    output_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    board_cols: int,
    board_rows: int,
    square_m: float,
    marker_m: float,
) -> None:
    """Save calibration to OpenCV-compatible YAML (readable by cv2.FileStorage)."""
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width",  image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs",   dist_coeffs)
    fs.write("rms_error", rms)
    fs.write("board_cols",   board_cols)
    fs.write("board_rows",   board_rows)
    fs.write("square_length_m",  square_m)
    fs.write("marker_length_m",  marker_m)
    fs.write("aruco_dict", "DICT_4X4_50")
    fs.release()
    print(f"Calibration saved to: {output_path}")


@dataclass
class CalibrationData:
    """A calibration plus the image size it was computed at.

    The image size is load-bearing, not metadata: the intrinsics are expressed in
    pixels of a specific resolution. Using them at a different capture resolution
    scales every pose while leaving the reprojection residual near zero, so no
    downstream quality gate can detect the mistake. Always pair the intrinsics
    with the size they came from.
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Optional[tuple[int, int]] = None   # (width, height)
    rms_error: Optional[float] = None


def read_calibration(yaml_path: str) -> CalibrationData:
    """Load the full calibration record, including the image size it was taken at."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {yaml_path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs   = fs.getNode("dist_coeffs").mat()

    def _num(key):
        node = fs.getNode(key)
        return None if node.isNone() else float(node.real())

    width, height = _num("image_width"), _num("image_height")
    rms = _num("rms_error")
    fs.release()

    image_size = (int(width), int(height)) if width and height else None
    return CalibrationData(camera_matrix, dist_coeffs, image_size, rms)


def load_calibration(yaml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera_matrix and dist_coeffs from a calibration YAML.

    Prefer `read_calibration`, which also returns the image size the calibration
    was computed at — see `CalibrationData` for why that matters.
    """
    data = read_calibration(yaml_path)
    return data.camera_matrix, data.dist_coeffs


def describe_resolution_mismatch(
    calibration_size: Optional[tuple[int, int]],
    capture_size: tuple[int, int],
) -> Optional[str]:
    """Return an explanatory message if a calibration is being used at the wrong size.

    Returns None when the sizes match, or when the calibration predates the
    image_width/image_height fields and the check cannot be made.
    """
    if calibration_size is None:
        return None
    if tuple(calibration_size) == tuple(capture_size):
        return None
    cal_w, cal_h = calibration_size
    cap_w, cap_h = capture_size
    scale = cap_w / cal_w if cal_w else float("nan")
    return (
        f"Calibration resolution mismatch: the camera parameters were computed at "
        f"{cal_w}x{cal_h} but the camera is delivering {cap_w}x{cap_h}. "
        f"Using them as-is would scale every distance by roughly {scale:.2f}x "
        f"while still reporting a low reprojection error, so nothing downstream "
        f"would flag it. Recalibrate at {cap_w}x{cap_h}, or force the camera to "
        f"{cal_w}x{cal_h}."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{VERSION_STRING} — ChArUco camera calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true",
                      help="Generate the ChArUco board image for printing")
    mode.add_argument("--images", nargs="+", metavar="IMG",
                      help="Calibrate from existing image files")
    mode.add_argument("--live", action="store_true",
                      help="Calibrate from live webcam feed")
    mode.add_argument("--wizard", action="store_true",
                      help="Print a step-by-step calibration tutorial")

    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for --live mode (default: 0)")
    parser.add_argument("--square", type=float, default=0.025,
                        help="Physical square side length in metres (default: 0.025)")
    parser.add_argument("--marker", type=float, default=0.019,
                        help="Physical ArUco marker side length in metres (default: 0.019)")
    parser.add_argument("--cols", type=int, default=BOARD_COLS,
                        help=f"Board columns (default: {BOARD_COLS})")
    parser.add_argument("--rows", type=int, default=BOARD_ROWS,
                        help=f"Board rows (default: {BOARD_ROWS})")
    parser.add_argument("--output", "-o", default="camera_params.yaml",
                        help="Output calibration YAML file (default: camera_params.yaml)")
    parser.add_argument("--board-image", default="charuco_board.pdf",
                        help="Output board path for --generate; .pdf or .png "
                             "(default: charuco_board.pdf)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for generated board image (default: 300)")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help=f"Minimum captured frames (default: {MIN_FRAMES})")

    args = parser.parse_args()

    if args.wizard:
        print(CALIBRATION_WIZARD_TEXT)
        return 0

    board = make_charuco_board(args.cols, args.rows, args.square, args.marker)

    # ── Generate mode ────────────────────────────────────────────────────────
    if args.generate:
        generate_board_image(
            board, args.board_image, dpi=args.dpi,
            square_m=args.square, rows_cols=(args.cols, args.rows),
        )
        return 0

    # ── Calibration modes ────────────────────────────────────────────────────
    detector = cv2.aruco.CharucoDetector(board)

    if args.images:
        paths = [Path(p) for p in args.images]
        print(f"Processing {len(paths)} image(s)...")
        all_corners, all_ids, image_size = collect_from_images(paths, detector)
    else:  # --live
        all_corners, all_ids, image_size = collect_from_webcam(
            args.camera, detector, board, target_frames=args.min_frames
        )

    if len(all_corners) < args.min_frames:
        print(
            f"ERROR: Only {len(all_corners)} valid frames collected "
            f"(minimum {args.min_frames}).",
            file=sys.stderr,
        )
        return 1

    print(f"\nRunning calibration on {len(all_corners)} frames...")
    rms, camera_matrix, dist_coeffs = calibrate(
        all_corners, all_ids, board, image_size
    )

    print("\n=== Calibration Results ===")
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  Camera matrix :\n{camera_matrix}")
    print(f"  Distortion coeffs : {dist_coeffs.ravel()}")

    if rms > 1.0:
        print(f"\n  WARNING: RMS error {rms:.3f} px is above the 1.0 px threshold.")
        print("  Consider recollecting calibration images.")

    save_calibration(
        args.output,
        camera_matrix,
        dist_coeffs,
        image_size,
        rms,
        args.cols,
        args.rows,
        args.square,
        args.marker,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
