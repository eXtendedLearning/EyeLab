"""The analysis must recover a known injected EyeLab error, including errors
larger than the detector's search radius (regression: the first draft seeded
detection with EyeLab's own projection and reported a 15 mm error as 0.1 mm)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validation.metrics import overlay_jitter_mm, per_node, registration_errors
from validation.plates import build_target, plate_a, plate_l
from validation.session import load_session
from validation.synth import generate_session, view_pose

K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])


def _run(layout=None, distance=0.6, tilt=0.0, n=3, **kw):
    layout = layout or plate_a()
    t = build_target(layout)
    rv, tv = view_pose(t, distance, tilt)
    with tempfile.TemporaryDirectory() as d:
        generate_session(Path(d) / "s", layout, t, K, rv, tv, n_frames=n, **kw)
        rows, qa = registration_errors(load_session(Path(d) / "s"), t)
    return rows, qa


class TestRegistration(unittest.TestCase):
    def test_zero_error(self):
        rows, qa = _run()
        self.assertTrue(all(r.detected for r in rows))
        self.assertLess(np.median([r.err_px for r in rows]), 0.1)
        self.assertLess(np.median([r.err_mm for r in rows]), 0.05)
        self.assertLess(np.median(qa["homography_residual_px"]), 0.1)

    def test_zero_error_oblique(self):
        rows, _ = _run(tilt=45.0)
        self.assertTrue(all(r.detected for r in rows))
        self.assertLess(np.percentile([r.err_mm for r in rows], 95), 0.1)

    def test_known_bias_recovered_in_mm_and_px(self):
        rows, _ = _run(bias_mm=(2.0, 0, 0))
        self.assertAlmostEqual(np.median([r.err_mm for r in rows]), 2.0, delta=0.05)
        self.assertAlmostEqual(np.median([r.err_px for r in rows]), 2.0 * 1000 / 600, delta=0.1)
        # camera x is plate +X in the frontal view: the signed error points along +u
        self.assertAlmostEqual(np.median([r.err_du_mm for r in rows]), 2.0, delta=0.05)

    def test_large_error_beyond_search_radius_is_reported_not_hidden(self):
        for b in (15.0, 30.0):
            rows, _ = _run(bias_mm=(b, 0, 0), n=2)
            self.assertTrue(all(r.detected for r in rows))
            self.assertAlmostEqual(np.median([r.err_mm for r in rows]), b, delta=0.1)

    def test_folded_plate_bias(self):
        rows, _ = _run(plate_l(), bias_mm=(0.0, 1.5, 0.0), n=2)
        det = [r for r in rows if r.detected]
        self.assertEqual(len(det), len(rows))
        for face in ("H", "V"):
            self.assertGreater(np.median([r.err_mm for r in det if r.face == face]), 0.5)

    def test_integer_drawn_pixels_add_rounding_error(self):
        rows, _ = _run()
        drawn = np.array([r.err_px_drawn for r in rows])
        self.assertGreater(np.median(drawn), 0.25)
        self.assertLess(np.max(drawn), 0.8)

    def test_overlay_jitter_and_node_bias(self):
        rows, _ = _run(trans_noise_mm=0.5, n=8, bias_mm=(1.0, 0, 0))
        jit = overlay_jitter_mm(rows)
        self.assertGreater(np.median(jit), 0.1)
        nodes = per_node(rows)
        self.assertEqual(len(nodes), 20)
        self.assertAlmostEqual(np.median([n["bias_du_mm"] for n in nodes]), 1.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
