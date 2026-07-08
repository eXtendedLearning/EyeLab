"""Shared constants and small helpers for the EyeLab GUI modules.

Kept free of tkinter imports so it can be used from tests without a display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from registration import AXIS_NORMALS, marker_axes_from_normal, normal_label

# ── Paths ─────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent / "config"
MARKERS_DIR = Path(__file__).parent / "markers"
LOG_DIR = Path(__file__).parent / ".logs"
TEST_ASSETS_DIR = Path(__file__).resolve().parent.parent / "test_assets"
CALIBRATION_FILE = CONFIG_DIR / "camera_params.yaml"
MARKER_CONFIG_FILE = CONFIG_DIR / "marker_config.json"
HAMMER_MARKER_CONFIG_FILE = CONFIG_DIR / "hammer_marker_config.json"

# ── UI units & board geometry ────────────────────────────────────────────────

POSITION_UI_UNIT = "cm"
POSITION_UI_SCALE = 100.0
CHARUCO_SQUARE_M = 0.025
CHARUCO_MARKER_M = 0.019


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
