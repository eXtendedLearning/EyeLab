#!/usr/bin/env python3
"""
Generate ArUco markers for EyeLab EMA testing.

Dictionary: DICT_4X4_50
Physical size: 12mm markers for the flangia's 16mm grid spacing.

Usage:
    python generate_markers.py --output markers/ --count 20
    python generate_markers.py --output markers/ --ids 0,1,2,5,10
    python generate_markers.py --output markers/ --count 20 --dpi 300
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# EyeLab marker specification (flangia / L-shaped structure)
ARUCO_DICT = cv2.aruco.DICT_4X4_50   # 4x4 bit, 50 unique IDs
MARKER_SIZE_MM = 12.0                  # Physical marker edge length (mm)
GRID_SPACING_MM = 16.0                 # Grid spacing on flangia (mm)
# Physical border between markers: 16 - 12 = 4 mm gap on each side => 2 mm each side


def mm_to_px(mm: float, dpi: int) -> int:
    """Convert millimetres to pixels at given DPI."""
    return int(round(mm * dpi / 25.4))


def generate_markers(
    output_dir: str,
    dpi: int = 300,
    marker_ids: list[int] | None = None,
    add_id_label: bool = True,
) -> None:
    """
    Generate ArUco markers as PNG images ready for printing.

    The images include a white border sized so that, when printed at the given DPI,
    each marker occupies exactly one grid cell (16 mm) and the black pattern is 12 mm.

    Args:
        output_dir:    Directory to write marker PNGs into.
        dpi:           Dots per inch for the output images.
        marker_ids:    List of ArUco IDs to generate. If None, generates 0–9.
        add_id_label:  Draw the numeric ID below the marker image.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    if marker_ids is None:
        marker_ids = list(range(10))

    # Pixel sizes at the requested DPI
    marker_px = mm_to_px(MARKER_SIZE_MM, dpi)    # black pattern region
    border_px = mm_to_px((GRID_SPACING_MM - MARKER_SIZE_MM) / 2.0, dpi)  # 2 mm each side
    total_px = marker_px + 2 * border_px          # one grid cell = 16 mm

    print(f"Generating {len(marker_ids)} marker(s):")
    print(f"  Dictionary   : DICT_4X4_50 (4×4 bit, 50 unique IDs)")
    print(f"  Marker size  : {MARKER_SIZE_MM} mm ({marker_px} px at {dpi} DPI)")
    print(f"  Grid spacing : {GRID_SPACING_MM} mm ({total_px} px per cell)")
    print(f"  Border       : {(GRID_SPACING_MM - MARKER_SIZE_MM) / 2:.1f} mm ({border_px} px)")
    print(f"  Output       : {output_path.resolve()}\n")

    for mid in marker_ids:
        # generateImageMarker writes the marker into a (marker_px × marker_px) image
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_px)

        # Pad with white border to reach grid-cell size
        padded = cv2.copyMakeBorder(
            marker_img,
            border_px, border_px, border_px, border_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        if add_id_label:
            # Add a narrow label strip below the marker showing the ID
            label_height = max(20, border_px)
            label = np.full((label_height, total_px), 255, dtype=np.uint8)
            cv2.putText(
                label,
                f"aruco{mid + 1:02d}  |  ID {mid}  |  DICT_4X4_50  |  {MARKER_SIZE_MM:.0f}mm",
                (4, label_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                0,
                1,
                cv2.LINE_AA,
            )
            padded = np.vstack([padded, label])

        out_file = output_path / f"aruco{mid + 1:02d}.png"
        cv2.imwrite(str(out_file), padded)
        print(f"  Saved: {out_file.name}")

    print(f"\nAll markers saved to {output_path.resolve()}")
    _print_instructions(dpi, marker_px, total_px)


def _print_instructions(dpi: int, marker_px: int, total_px: int) -> None:
    print("\n=== PRINTING INSTRUCTIONS ===")
    print(f"1. Print at EXACTLY 100% scale (disable 'fit to page').")
    print(f"2. Set printer DPI to {dpi}.")
    print(f"3. Verify physical size after printing:")
    print(f"     - Black pattern area : {MARKER_SIZE_MM} mm × {MARKER_SIZE_MM} mm")
    print(f"     - Full cell (incl. border) : {GRID_SPACING_MM} mm × {GRID_SPACING_MM} mm")
    print(f"4. Use matte paper to reduce glare in lab lighting.")
    print(f"5. Laminate markers for durability during impact hammer testing.\n")
    print("=== PLACEMENT ON FLANGIA ===")
    print(f"Place markers on the L-shaped flangia at 16 mm grid intervals.")
    print(f"The outer white border aligns markers to the grid automatically.")
    print(f"Optimal detection distance: 0.3 m – 1.5 m.\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate printable ArUco markers for EyeLab EMA testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default="markers",
        help="Output directory for marker PNG files (default: markers/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Printer DPI (default: 300)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of markers to generate starting from ID 0 (default: 10)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of specific marker IDs (e.g. '0,1,5,10'). "
             "Overrides --count.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Omit the ID label strip below each marker",
    )

    args = parser.parse_args()

    if args.ids is not None:
        try:
            ids = [int(x.strip()) for x in args.ids.split(",")]
        except ValueError:
            print("ERROR: --ids must be a comma-separated list of integers.", file=sys.stderr)
            return 1
        max_id = max(ids)
        if max_id >= 50:
            print(
                f"ERROR: DICT_4X4_50 only has IDs 0–49 (requested ID {max_id}).",
                file=sys.stderr,
            )
            return 1
    else:
        if args.count > 50:
            print(
                f"ERROR: DICT_4X4_50 only has 50 unique IDs (requested {args.count}).",
                file=sys.stderr,
            )
            return 1
        ids = list(range(args.count))

    generate_markers(
        output_dir=args.output,
        dpi=args.dpi,
        marker_ids=ids,
        add_id_label=not args.no_label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
