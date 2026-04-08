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
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# ── Board defaults ────────────────────────────────────────────────────────────
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
) -> None:
    """Render the ChArUco board to a PNG sized for A4 printing at `dpi`."""
    def mm2px(mm: float) -> int:
        return int(round(mm * dpi / 25.4))

    w_px = mm2px(page_width_mm - 2 * margin_mm)
    h_px = mm2px(page_height_mm - 2 * margin_mm)

    img = board.generateImage((w_px, h_px), marginSize=0, borderBits=1)

    # Embed to full page with margin
    page_w = mm2px(page_width_mm)
    page_h = mm2px(page_height_mm)
    m = mm2px(margin_mm)
    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    page[m : m + h_px, m : m + w_px] = img

    cv2.imwrite(output_path, page)
    print(f"Board image saved to: {output_path}")
    print(f"  Resolution : {page_w}×{page_h} px  ({dpi} DPI)")
    print(f"  Page size  : A4 ({page_width_mm}×{page_height_mm} mm)")
    print(f"  Print at 100% — do NOT scale to fit.")


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
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    all_corners, all_ids = [], []
    image_size = None

    print(f"\nLive calibration — camera {camera_index}")
    print(f"  SPACE  — capture current frame")
    print(f"  ESC    — finish (need at least {target_frames} frames)")
    print(f"  Q      — abort\n")

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


def load_calibration(yaml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera_matrix and dist_coeffs from a calibration YAML."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {yaml_path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs   = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ChArUco camera calibration for EyeLab.",
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
    parser.add_argument("--board-image", default="charuco_board.png",
                        help="Output board image path for --generate (default: charuco_board.png)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for generated board image (default: 300)")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help=f"Minimum captured frames (default: {MIN_FRAMES})")

    args = parser.parse_args()

    board = make_charuco_board(args.cols, args.rows, args.square, args.marker)

    # ── Generate mode ────────────────────────────────────────────────────────
    if args.generate:
        generate_board_image(board, args.board_image, dpi=args.dpi)
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

    print(f"\n=== Calibration Results ===")
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  Camera matrix :\n{camera_matrix}")
    print(f"  Distortion coeffs : {dist_coeffs.ravel()}")

    if rms > 1.0:
        print(f"\n  WARNING: RMS error {rms:.3f} px is above the 1.0 px threshold.")
        print(f"  Consider recollecting calibration images.")

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
