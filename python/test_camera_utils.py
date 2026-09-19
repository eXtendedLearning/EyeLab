#!/usr/bin/env python3
"""Tests for camera opening: frame verification, backend fallback, threading."""

import time
import unittest
from unittest.mock import patch

import numpy as np

import camera_utils
from camera_utils import (
    CameraOpener,
    CameraScanner,
    list_cameras,
    open_camera,
    open_camera_result,
)


class _FakeCapture:
    """VideoCapture stand-in: opens or not, delivers frames or not."""

    instances: list["_FakeCapture"] = []

    def __init__(self, index=0, backend=None, opens=True, delivers=True):
        self.index = index
        self.backend = backend
        self._opens = opens
        self._delivers = delivers
        self.released = False
        self.reads = 0
        self.props = {}
        _FakeCapture.instances.append(self)

    def isOpened(self):
        return self._opens and not self.released

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        self.reads += 1
        if self._delivers and not self.released:
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


def _factory(**kwargs):
    """A VideoCapture replacement with fixed behaviour."""
    def make(*args):
        index = args[0] if args else 0
        backend = args[1] if len(args) > 1 else None
        return _FakeCapture(index, backend, **kwargs)
    return make


def _per_backend(behaviour: dict):
    """A VideoCapture replacement whose behaviour depends on the backend id."""
    def make(*args):
        index = args[0] if args else 0
        backend = args[1] if len(args) > 1 else None
        opens, delivers = behaviour.get(backend, (False, False))
        return _FakeCapture(index, backend, opens=opens, delivers=delivers)
    return make


class OpenCameraTests(unittest.TestCase):
    def setUp(self):
        _FakeCapture.instances = []

    def test_accepts_a_backend_that_delivers_a_frame(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=True, delivers=True)):
            result = open_camera_result(0, warmup_s=0.2)
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.cap)
        self.assertFalse(result.cap.released)
        self.assertEqual([a["delivered"] for a in result.attempts], [True])

    def test_rejects_a_camera_that_opens_but_never_delivers(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=True, delivers=False)):
            result = open_camera_result(0, warmup_s=0.1)
        self.assertFalse(result.ok)
        self.assertIsNone(result.cap)
        self.assertTrue(all(c.released for c in _FakeCapture.instances))
        self.assertIn("delivered no frames", result.detail)

    def test_falls_through_to_the_next_backend(self):
        backends = [("DSHOW", 700), ("MSMF", 1400)]
        behaviour = {700: (True, False), 1400: (True, True)}
        with patch.object(camera_utils, "_capture_backends", lambda: backends), \
             patch.object(camera_utils.cv2, "VideoCapture", _per_backend(behaviour)):
            result = open_camera_result(0, warmup_s=0.1)
        self.assertTrue(result.ok)
        self.assertEqual(result.backend, "MSMF")
        self.assertEqual([a["backend"] for a in result.attempts], ["DSHOW", "MSMF"])
        dshow_cap = _FakeCapture.instances[0]
        self.assertTrue(dshow_cap.released, "a silent backend must be released")

    def test_open_camera_wrapper_returns_a_closed_capture_on_failure(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=False, delivers=False)):
            cap = open_camera(0, warmup_s=0.05)
        self.assertFalse(cap.isOpened())

    def test_probe_skips_the_frame_check_and_releases(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=True, delivers=False)):
            found = list_cameras(max_test=3)
        self.assertEqual(found, [0, 1, 2])
        self.assertTrue(all(c.released for c in _FakeCapture.instances))
        self.assertTrue(all(c.reads == 0 for c in _FakeCapture.instances),
                        "probing must not consume frames")

    def test_probe_stops_when_its_time_budget_is_spent(self):
        def slow_open(*args):
            time.sleep(0.05)
            return _FakeCapture(args[0] if args else 0, opens=True)

        with patch.object(camera_utils.cv2, "VideoCapture", slow_open):
            found = list_cameras(max_test=8, budget_s=0.12)
        self.assertLess(len(found), 8)


class CameraOpenerTests(unittest.TestCase):
    def setUp(self):
        _FakeCapture.instances = []

    def test_result_arrives_without_blocking_the_caller(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=True, delivers=True)):
            opener = CameraOpener(0, warmup_s=0.1).start()
            deadline = time.monotonic() + 3.0
            while opener.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            result = opener.poll()
        self.assertIsNotNone(result)
        self.assertTrue(result.ok)

    def test_cancel_releases_a_capture_that_lands_late(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=True, delivers=True)):
            opener = CameraOpener(0, warmup_s=0.1).start()
            opener.cancel()
            time.sleep(0.4)
        self.assertIsNone(opener.poll())
        self.assertTrue(_FakeCapture.instances, "the open still happened")
        self.assertTrue(
            all(c.released for c in _FakeCapture.instances),
            "an abandoned capture must be released, or the webcam stays locked",
        )

    def test_scanner_returns_indices(self):
        with patch.object(camera_utils.cv2, "VideoCapture", _factory(opens=True, delivers=False)):
            scanner = CameraScanner(max_test=2).start()
            deadline = time.monotonic() + 3.0
            while scanner.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual(scanner.poll(), [0, 1])


if __name__ == "__main__":
    unittest.main()
