"""Shared constants and small helpers for the EyeLab GUI modules.

Kept free of tkinter imports so it can be used from tests without a display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import eyelab_paths as _paths
from registration import AXIS_NORMALS, marker_axes_from_normal, normal_label

# ── Paths ─────────────────────────────────────────────────────────────────────
# Configuration files (camera profiles, marker configs, geometries, validation
# targets) live in the shared tree defined by eyelab_paths; the active camera
# calibration and marker config are resolved at call time:
#   eyelab_paths.active_calibration_file(), eyelab_paths.active_marker_config()

CONFIG_DIR = _paths.CONFIG_DIR
CAMERAS_DIR = _paths.CAMERAS_DIR
MARKER_CONFIGS_DIR = _paths.MARKER_CONFIGS_DIR
GEOMETRY_DIR = _paths.GEOMETRY_DIR
HAMMER_MARKER_CONFIG_FILE = _paths.HAMMER_MARKER_CONFIG
MARKERS_DIR = Path(__file__).parent / "markers"     # generated marker images (git-ignored)
LOG_DIR = Path(__file__).parent / ".logs"

# ── UI units & board geometry ────────────────────────────────────────────────

POSITION_UI_UNIT = "cm"
POSITION_UI_SCALE = 100.0
CHARUCO_SQUARE_M = 0.025
CHARUCO_MARKER_M = 0.019
# The lab's printed board, as measured (2026-11-24): a black ArUco marker is 25.97 x 25.98 mm (caliper), so a square
# is 25.97 x 25/19 = 34.17 mm; a ruler agrees (5 squares 170 mm, 7 squares about 240 mm). The print is 1.367x the
# nominal size above. The nominal size draws the board; poses of it must use the printed size, since a scale error
# reprojects perfectly (AGENTS.md). (0.4.016 took the caliper value for a square, 1.316x too small.)
CHARUCO_MARKER_PRINTED_M = 0.02597
CHARUCO_SQUARE_PRINTED_M = CHARUCO_MARKER_PRINTED_M * CHARUCO_SQUARE_M / CHARUCO_MARKER_M


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
