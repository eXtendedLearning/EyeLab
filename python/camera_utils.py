#!/usr/bin/env python3
"""Shared camera helpers for OpenCV capture."""

from __future__ import annotations

import os
import sys

# This must be set before importing cv2. It keeps failed probe attempts from
# printing backend warnings while Python-side errors still report what failed.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2


def _capture_backends() -> list[int]:
    """Return preferred OpenCV capture backends for the current platform."""
    if sys.platform.startswith("win"):
        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        return backends or [cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(
    camera_index: int = 0,
    width: int | None = 1280,
    height: int | None = 720,
    fps: int | None = 30,
) -> cv2.VideoCapture:
    """Open a camera with a deterministic backend and common capture settings."""
    last_cap: cv2.VideoCapture | None = None
    for backend in _capture_backends():
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, fps)
            return cap
        cap.release()
        last_cap = cap

    return last_cap if last_cap is not None else cv2.VideoCapture()


def list_cameras(max_test: int = 8) -> list[int]:
    """Probe likely camera indices and return those that open successfully."""
    available: list[int] = []
    for index in range(max_test):
        cap = open_camera(index, width=None, height=None, fps=None)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available
