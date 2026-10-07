"""Instrument-aware uncertainty, fold angle from images, setup assistant."""
import math
import unittest

import cv2
import numpy as np

from validation import uncertainty as U
from validation.detect import locate_points
from validation.fold import fit_frame
from validation.plates import build_target, plate_a, plate_l
from validation.setup_assist import all_ok, measure, rows_for
from validation.synth import render_scene, view_pose

K = np.array([[1100.0, 0, 640], [0, 1100.0, 360], [0, 0, 1]])


class TestUncertainty(unittest.TestCase):
    def test_rectangular_and_tur(self):
        self.assertAlmostEqual(U.rect_u(1.0), 1 / math.sqrt(3))
        self.assertAlmostEqual(U.tur(0.125), 4.0)
        self.assertIn("INSUFFICIENT", U.verdict(0.3))
        self.assertIn("OK", U.verdict(0.05))

    def test_ruler_vs_caliper_on_plate_a(self):
        t = build_target(plate_a())
        self.assertAlmostEqual(U.lever_mm(t)["r"], math.hypot(118, 70), places=3)
        ruler = U.point_uncertainty(t, U.reading_u([200.0] * 3, 1.0), U.reading_u([150.0] * 3, 1.0),
                                    200.0, 150.0)
        caliper = U.point_uncertainty(t, U.reading_u([200.0] * 3, 0.03),
                                      U.reading_u([150.0] * 3, 0.03), 200.0, 150.0)
        self.assertGreater(ruler["u_mm"], 0.25)       # ~0.33 mm: cannot resolve a 1 mm goal
        self.assertLess(ruler["tur"], U.MIN_TUR)
        self.assertLess(caliper["u_mm"], 0.07)
        self.assertGreaterEqual(caliper["tur"], U.MIN_TUR)
        # uniform scale is not in the budget: only anisotropy, placement, flatness
        self.assertEqual(set(ruler["budget"]), {"anisotropy_mm", "print_placement_mm", "paper_flatness_mm"})

    def test_fold_term_only_for_plate_l(self):
        t = build_target(plate_l())
        self.assertAlmostEqual(U.lever_mm(t)["fold_arm"], 64.0, places=6)
        b = U.point_uncertainty(t, 0.02, 0.02, 200.0, 75.0, u_fold_deg=1 / math.sqrt(3))
        self.assertAlmostEqual(b["budget"]["fold_angle_mm"], 64 * math.radians(1 / math.sqrt(3)))

    def test_step(self):
        self.assertAlmostEqual(U.step_uncertainty(1.0), math.sqrt(2 / 3))
        self.assertLess(U.tur(U.step_uncertainty(1.0)), 1.0)
        self.assertGreater(U.tur(U.step_uncertainty(0.02)), U.MIN_TUR)


class TestFoldFromImages(unittest.TestCase):
    def test_recovers_a_folded_plate(self):
        layout = plate_l()
        nominal = build_target(layout)
        rng = np.random.default_rng(7)
        for true in (88.0, 91.5):
            truth = build_target(layout, fold_interior_deg=true)
            rv, tv = view_pose(truth, 0.55, 10.0)
            img = render_scene(layout, truth, K, rv, tv)
            img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
            loc = locate_points(img, nominal, K, np.zeros(5))
            f = fit_frame(nominal, loc.points, K, np.zeros(5))
            self.assertIsNotNone(f)
            self.assertAlmostEqual(f.interior_deg, true, delta=0.15)
            self.assertLess(f.rms_px, 0.3)

    def test_needs_both_faces(self):
        t = build_target(plate_l())
        only_h = {i: np.array([600.0, 400.0]) + 10 * k for k, i in enumerate(t.face_ids("H"))}
        self.assertIsNone(fit_frame(t, only_h, K, np.zeros(5)))


class TestSetupAssist(unittest.TestCase):
    def test_plate_a_angle_distance(self):
        t = build_target(plate_a())
        rv, tv = view_pose(t, 0.6, 30.0)
        r = measure(t, rv, tv, K, (1280, 720))
        self.assertAlmostEqual(r.distance_mm, 600.0, delta=1.0)
        self.assertAlmostEqual(r.angle_deg, 30.0, delta=0.2)
        self.assertLess(r.centre_offset, 0.02)
        self.assertTrue(all_ok(rows_for({"test": "B1", "distance_mm": 600, "angle_deg": 30}, r)))
        self.assertFalse(all_ok(rows_for({"test": "B1", "distance_mm": 600, "angle_deg": 0}, r)))

    def test_plate_l_elevation(self):
        t = build_target(plate_l())
        rv, tv = view_pose(t, 0.5)           # along the bisector of the two faces
        R = cv2.Rodrigues(rv)[0]
        # view_pose aims at the points' centroid; measure() uses the fold centre
        r = measure(t, rv, tv, K, (1280, 720))
        cam = -R.T @ tv
        expected = math.degrees(math.atan2(cam[2], math.hypot(cam[0], cam[1])))
        self.assertAlmostEqual(r.elevation_deg, expected, delta=0.05)
        self.assertGreater(r.elevation_deg, 35.0)


if __name__ == "__main__":
    unittest.main()
