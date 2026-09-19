#!/usr/bin/env python3
"""Shared camera helpers for OpenCV capture.

Opening a camera on Windows is neither quick nor reliable, and both failure
modes were measured on this project's own hardware on 2026-09-19
(`.logs/ar_debug_20260919_113447.jsonl`):

* ``cv2.VideoCapture(0, CAP_DSHOW)`` blocked its caller for ~32 s. Every
  call site was on the tkinter main thread, so the GUI stopped repainting
  and Windows reported it as not responding.
* The capture it eventually returned reported ``isOpened() == True`` and
  then never delivered a single frame, which the GUI could only show as
  "Waiting for camera frame..." forever.

Both are handled here rather than in each caller:

* :func:`open_camera_result` accepts a backend only after it has actually
  delivered a frame, and falls through to the next backend otherwise. It
  reports what it tried, so the GUI can say why an open failed.
* :class:`CameraOpener` and :class:`CameraScanner` run that work on a worker
  thread, so a slow backend cannot block an event loop. A capture that
  arrives after the caller gave up is released rather than leaked, since a
  leaked handle is what leaves a webcam unusable until it is replugged.

The backend order can be overridden with ``EYELAB_CAPTURE_BACKEND``
(``dshow``, ``msmf`` or ``any``) when one Windows backend works on a machine
and the other does not.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# This must be set before importing cv2. It keeps failed probe attempts from
# printing backend warnings while Python-side errors still report what failed.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

# A camera that has just been opened may need a moment before its first frame.
WARMUP_S = 3.0
# Total time list_cameras() may spend probing, however many indices remain.
PROBE_BUDGET_S = 8.0
# Polling interval while waiting for the first frame.
_WARMUP_POLL_S = 0.02


@dataclass
class CameraOpenResult:
    """Outcome of one open attempt, including the backends that were tried."""

    cap: Optional[cv2.VideoCapture]
    backend: str
    delivered: bool
    detail: str
    elapsed_s: float
    attempts: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.cap is not None and self.delivered


def _capture_backends() -> list[tuple[str, int]]:
    """Preferred capture backends for this platform, most preferred first."""
    override = os.environ.get("EYELAB_CAPTURE_BACKEND", "").strip().lower()
    known = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "any": cv2.CAP_ANY,
    }
    if override in known and known[override] is not None:
        return [(override.upper(), known[override])]

    if sys.platform.startswith("win"):
        backends = [
            (name, known[name]) for name in ("dshow", "msmf") if known[name] is not None
        ]
        return [(n.upper(), b) for n, b in backends] or [("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def _delivers_frame(cap: cv2.VideoCapture, warmup_s: float) -> bool:
    """True once the capture returns a real frame, False if it never does."""
    deadline = time.perf_counter() + warmup_s
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(_WARMUP_POLL_S)


def _failure_detail(camera_index: int, attempts: list[dict]) -> str:
    opened_but_silent = [a for a in attempts if a["opened"] and not a["delivered"]]
    tried = ", ".join(a["backend"] for a in attempts) or "no backend"
    if opened_but_silent:
        backend = opened_but_silent[0]["backend"]
        return (
            f"Camera {camera_index} opened ({backend}) but delivered no frames. "
            "It is most likely held by another application, or was left in a bad "
            "state by a previous run - close other camera apps, or unplug and "
            "replug the webcam, then try again."
        )
    return (
        f"Camera {camera_index} could not be opened (tried {tried}). "
        "Check that it is connected and not disabled in Windows camera privacy "
        "settings."
    )


def open_camera_result(
    camera_index: int = 0,
    width: int | None = 1280,
    height: int | None = 720,
    fps: int | None = 30,
    warmup_s: float = WARMUP_S,
) -> CameraOpenResult:
    """Open a camera and accept it only once it has delivered a frame.

    ``warmup_s <= 0`` skips the delivery check, which is what the index probe
    in :func:`list_cameras` wants: cheap, and it never consumes a frame.
    """
    started = time.perf_counter()
    attempts: list[dict] = []
    for name, backend in _capture_backends():
        attempt_t0 = time.perf_counter()
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            opened = bool(cap.isOpened())
            delivered = False
            if opened:
                if width is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if fps is not None:
                    cap.set(cv2.CAP_PROP_FPS, fps)
                delivered = warmup_s <= 0 or _delivers_frame(cap, warmup_s)
        except Exception as e:                       # cv2 can raise, not only fail
            attempts.append({
                "backend": name, "opened": False, "delivered": False,
                "seconds": round(time.perf_counter() - attempt_t0, 2), "error": str(e),
            })
            continue

        attempts.append({
            "backend": name,
            "opened": opened,
            "delivered": delivered,
            "seconds": round(time.perf_counter() - attempt_t0, 2),
        })
        if opened and delivered:
            return CameraOpenResult(
                cap=cap,
                backend=name,
                delivered=True,
                detail=f"Camera {camera_index} opened via {name}.",
                elapsed_s=round(time.perf_counter() - started, 2),
                attempts=attempts,
            )
        cap.release()

    return CameraOpenResult(
        cap=None,
        backend="",
        delivered=False,
        detail=_failure_detail(camera_index, attempts),
        elapsed_s=round(time.perf_counter() - started, 2),
        attempts=attempts,
    )


def open_camera(
    camera_index: int = 0,
    width: int | None = 1280,
    height: int | None = 720,
    fps: int | None = 30,
    warmup_s: float = WARMUP_S,
) -> cv2.VideoCapture:
    """Open a camera, returning a closed capture when no backend delivers.

    Kept for callers that only need a ``VideoCapture``; prefer
    :func:`open_camera_result` (or :class:`CameraOpener`) where the reason
    for a failure should reach the user.
    """
    result = open_camera_result(camera_index, width, height, fps, warmup_s)
    return result.cap if result.cap is not None else cv2.VideoCapture()


def list_cameras(max_test: int = 8, budget_s: float = PROBE_BUDGET_S) -> list[int]:
    """Probe camera indices and return those that open.

    Opening alone is the test: requesting a frame from every index would cost
    seconds per index and wakes devices the user did not ask for. The probe
    also stops once ``budget_s`` is spent, because a single ``VideoCapture``
    call can take tens of seconds on Windows.
    """
    available: list[int] = []
    deadline = time.perf_counter() + budget_s
    for index in range(max_test):
        if time.perf_counter() >= deadline:
            break
        result = open_camera_result(index, width=None, height=None, fps=None, warmup_s=0.0)
        if result.cap is not None:
            available.append(index)
            result.cap.release()
    return available


class _ThreadedCall:
    """Run one call on a daemon thread and collect its result without blocking.

    ``poll()`` returns None until the call finishes. ``cancel()`` gives up on
    it; anything the call produced afterwards goes to ``_discard``.
    """

    def __init__(self, func: Callable[[], Any], name: str):
        self._func = func
        self._name = name
        self._result: Any = None
        self._done = threading.Event()
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "_ThreadedCall":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        try:
            result = self._func()
        except Exception:
            result = None
        if self._cancelled:
            self._discard(result)
            return
        self._result = result
        self._done.set()

    def poll(self) -> Any:
        """The result, or None while the call is still running."""
        return self._result if self._done.is_set() else None

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        if self._done.is_set():
            self._discard(self._result)
            self._result = None

    def _discard(self, result: Any) -> None:
        """Release whatever the abandoned call produced. Overridden below."""


class CameraOpener(_ThreadedCall):
    """Open a camera off the caller's thread; release it if it lands too late."""

    def __init__(self, camera_index: int = 0, **open_kwargs: Any):
        self.camera_index = camera_index

        def _open() -> CameraOpenResult:
            try:
                return open_camera_result(camera_index, **open_kwargs)
            except Exception as e:
                return CameraOpenResult(
                    cap=None, backend="", delivered=False,
                    detail=f"Camera {camera_index} could not be opened: {e}",
                    elapsed_s=0.0,
                )

        super().__init__(_open, name=f"camera-open-{camera_index}")

    def _discard(self, result: Any) -> None:
        cap = getattr(result, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


class CameraScanner(_ThreadedCall):
    """Run :func:`list_cameras` off the caller's thread."""

    def __init__(self, max_test: int = 8):
        def _scan() -> list[int]:
            try:
                return list_cameras(max_test)
            except Exception:
                return []

        super().__init__(_scan, name="camera-scan")
