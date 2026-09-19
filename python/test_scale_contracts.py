#!/usr/bin/env python3
"""
Regression tests for the three silent scale errors found in the 2026-09-18 audit.

All three shared one property: a pure scale error reprojects almost perfectly, so
no reprojection-based quality gate could detect it. Each test below asserts the
contract that now prevents the error, plus — for the ones that are measurable —
the magnitude of the damage the contract is guarding against.

See docs/AUDIT-2026-09-18.md findings S1a, S1b, S1c.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibrate import describe_resolution_mismatch, read_calibration, save_calibration
from generate_markers import GRID_SPACING_MM, MARKER_SIZE_MM
from registration import (
    MarkerCorrespondence,
    read_marker_config,
    save_marker_config,
)
from unv_to_json import UNVParseError, UNVParser

try:
    import pyuff
except ImportError:  # pragma: no cover
    pyuff = None


K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)


def _square(size_m: float) -> np.ndarray:
    h = size_m / 2.0
    return np.array(
        [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64
    )


# ── S1a — marker size ─────────────────────────────────────────────────────────

class MarkerSizeContractTests(unittest.TestCase):
    """The configured marker size must match the printed marker."""

    def test_default_is_the_project_standard_20mm(self):
        self.assertEqual(MARKER_SIZE_MM, 20.0)

    def test_grid_spacing_leaves_a_quiet_zone(self):
        # generate_markers() rejects spacing < marker size; a quiet zone is also
        # required for reliable detection.
        self.assertGreater(GRID_SPACING_MM, MARKER_SIZE_MM)

    def test_size_mismatch_scales_depth_but_not_reprojection_error(self):
        """Why this matters: the error is invisible to every in-pipeline gate."""
        true_size, assumed_size = 0.020, 0.012
        rvec = np.array([[0.1], [0.2], [0.05]])
        tvec = np.array([[0.02], [0.01], [0.50]])

        img = cv2.projectPoints(_square(true_size), rvec, tvec, K, DIST)[0].reshape(-1, 2)
        ok, rvec_est, tvec_est = cv2.solvePnP(
            _square(assumed_size), img, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        self.assertTrue(ok)

        # Depth is wrong by exactly the size ratio ...
        self.assertAlmostEqual(
            float(tvec_est[2, 0]) / float(tvec[2, 0]), assumed_size / true_size, places=6
        )
        # ... while the reprojection residual stays at zero.
        reproj = cv2.projectPoints(
            _square(assumed_size), rvec_est, tvec_est, K, DIST
        )[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((reproj - img) ** 2, axis=1))))
        self.assertLess(rms, 1e-6)

    def test_marker_config_round_trips_the_per_structure_default(self):
        path = str(Path(tempfile.mkdtemp()) / "marker_config.json")
        markers = [
            MarkerCorrespondence(marker_id=0, unv_position=np.zeros(3)),
            MarkerCorrespondence(
                marker_id=1, unv_position=np.array([0.1, 0.0, 0.0]), marker_size_mm=12.0
            ),
        ]
        save_marker_config(path, markers, default_marker_size_mm=25.0)

        config = read_marker_config(path)
        self.assertEqual(config.default_marker_size_mm, 25.0)
        self.assertIsNone(config.markers[0].marker_size_mm)   # inherits the default
        self.assertEqual(config.markers[1].marker_size_mm, 12.0)  # per-marker override

    def test_marker_config_without_a_default_still_loads(self):
        path = Path(tempfile.mkdtemp()) / "legacy.json"
        path.write_text(json.dumps({"markers": [{"markerId": 3, "unvPosition": [0, 0, 0]}]}))
        config = read_marker_config(str(path))
        self.assertIsNone(config.default_marker_size_mm)
        self.assertEqual(len(config.markers), 1)


# ── S1b — calibration resolution ──────────────────────────────────────────────

class CalibrationResolutionContractTests(unittest.TestCase):
    """Intrinsics are pixel quantities and are only valid at their own resolution."""

    def _write(self, size):
        path = str(Path(tempfile.mkdtemp()) / "camera_params.yaml")
        save_calibration(path, K, np.zeros((1, 5)), size, 0.42, 5, 7, 0.030, 0.022)
        return path

    def test_image_size_survives_the_round_trip(self):
        data = read_calibration(self._write((1280, 720)))
        self.assertEqual(data.image_size, (1280, 720))
        self.assertAlmostEqual(data.rms_error, 0.42, places=6)

    def test_matching_resolution_is_accepted(self):
        self.assertIsNone(describe_resolution_mismatch((1280, 720), (1280, 720)))

    def test_mismatch_is_reported_with_the_scale_factor(self):
        message = describe_resolution_mismatch((1920, 1080), (1280, 720))
        self.assertIsNotNone(message)
        self.assertIn("1920x1080", message)
        self.assertIn("1280x720", message)

    def test_calibration_without_a_recorded_size_is_not_falsely_rejected(self):
        # Calibrations written before image_width/image_height existed.
        self.assertIsNone(describe_resolution_mismatch(None, (1280, 720)))

    def test_mismatch_corrupts_depth_while_reprojection_stays_clean(self):
        scale = 1280 / 1920
        k_true = np.array([[1400 * scale, 0, 960 * scale],
                           [0, 1400 * scale, 540 * scale],
                           [0, 0, 1.0]])
        k_calibrated = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1.0]])

        obj = np.vstack([
            _square(0.020) + np.array([dx, dy, 0.0])
            for dx, dy in ((0, 0), (0.08, 0), (0.08, 0.08), (0, 0.08))
        ])
        rvec = np.array([[0.05], [0.12], [0.02]])
        tvec = np.array([[-0.04], [-0.04], [0.50]])
        img = cv2.projectPoints(obj, rvec, tvec, k_true, DIST)[0].reshape(-1, 2)

        ok, rvec_est, tvec_est = cv2.solvePnP(obj, img, k_calibrated, DIST)
        self.assertTrue(ok)
        rvec_est, tvec_est = cv2.solvePnPRefineLM(obj, img, k_calibrated, DIST, rvec_est, tvec_est)

        self.assertGreater(abs(float(tvec_est[2, 0]) / 0.50 - 1.0), 0.4)  # >40% depth error
        reproj = cv2.projectPoints(obj, rvec_est, tvec_est, k_calibrated, DIST)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((reproj - img) ** 2, axis=1))))
        self.assertLess(rms, 1.0)  # under every RMS gate in the pipeline


# ── S1c — UNV units and coordinate systems ────────────────────────────────────

@unittest.skipIf(pyuff is None, "pyuff not installed")
class UnvUnitContractTests(unittest.TestCase):
    """UNVParser.parse() emits metres regardless of the file's declared units."""

    def _build(self, units_code, length, xs, ys, def_cs=None):
        path = Path(tempfile.mkdtemp()) / "geometry.unv"
        n = len(xs)
        pyuff.UFF(str(path)).write_sets([
            pyuff.prepare_164(
                units_code=units_code, units_description="test", temp_mode=1,
                length=length, force=1.0, temp=1.0, temp_offset=273.15,
            ),
            pyuff.prepare_2411(
                node_nums=np.arange(1, n + 1),
                def_cs=np.array(def_cs if def_cs else [0] * n),
                disp_cs=np.zeros(n, int), color=np.zeros(n, int),
                x=np.array(xs, float), y=np.array(ys, float), z=np.zeros(n),
            ),
        ], mode="overwrite")
        return path

    def _span_x(self, path):
        data = UNVParser(path, validate_cs=False).parse()
        xs = [node["x"] for node in data["nodes"]]
        return max(xs) - min(xs), data

    def test_millimetre_geometry_is_converted_to_metres(self):
        # 200 x 100 mm plate, unit code 2 — what Testlab commonly exports.
        span, data = self._span_x(self._build(2, 0.001, [0.0, 200.0, 200.0, 0.0],
                                              [0.0, 0.0, 100.0, 100.0]))
        self.assertAlmostEqual(span, 0.200, places=9)
        self.assertEqual(data["units"]["code"], 2)
        self.assertAlmostEqual(data["units"]["lengthFactor"], 0.001, places=12)

    def test_si_geometry_is_unchanged(self):
        span, _ = self._span_x(self._build(1, 1.0, [0.0, 0.2, 0.2, 0.0],
                                           [0.0, 0.0, 0.1, 0.1]))
        self.assertAlmostEqual(span, 0.200, places=9)

    def test_inch_geometry_is_converted_to_metres(self):
        span, _ = self._span_x(self._build(6, 0.0254, [0.0, 10.0, 10.0, 0.0],
                                           [0.0, 0.0, 5.0, 5.0]))
        self.assertAlmostEqual(span, 0.254, places=9)

    def test_unit_code_is_read_not_defaulted(self):
        """The original parser looked for 'unit_code'; pyuff writes 'units_code'."""
        _, data = self._span_x(self._build(2, 0.001, [0.0, 1.0], [0.0, 0.0]))
        self.assertNotEqual(data["units"]["code"], 1, "unit code silently defaulted to SI")

    def test_node_definition_coordinate_system_is_read(self):
        """The original parser looked for 'coord_sys'; pyuff writes 'def_cs'."""
        path = self._build(1, 1.0, [0.0, 0.1], [0.0, 0.0], def_cs=[4, 4])
        data = UNVParser(path, validate_cs=False).parse()
        self.assertEqual({n["exportCS"] for n in data["nodes"]}, {4})

    def test_multiple_coordinate_systems_are_refused(self):
        path = self._build(1, 1.0, [0.0, 0.2, 0.2, 0.0], [0.0, 0.0, 0.1, 0.1],
                           def_cs=[0, 0, 7, 7])
        with self.assertRaises(UNVParseError):
            UNVParser(path, validate_cs=False).parse()


if __name__ == "__main__":
    unittest.main()
