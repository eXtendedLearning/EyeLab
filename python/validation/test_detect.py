import unittest

import cv2
import numpy as np

from validation.detect import detect_saddle, is_x_junction, locate_points
from validation.plates import build_target, plate_a, plate_l
from validation.synth import render_scene, view_pose

K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
DIST = np.zeros(5)


def _scene(layout, distance, tilt, noise=1.0, seed=0):
    t = build_target(layout)
    rv, tv = view_pose(t, distance, tilt)
    img = render_scene(layout, t, K, rv, tv)
    img = np.clip(img + np.random.default_rng(seed).normal(0, noise, img.shape), 0, 255)
    truth = cv2.projectPoints(t.array(t.ids()), rv, tv, K, DIST)[0].reshape(-1, 2)
    return t, img.astype(np.uint8), dict(zip(t.ids(), truth))


class TestLocate(unittest.TestCase):
    def _check(self, layout, distance, tilt, tol_px):
        t, img, truth = _scene(layout, distance, tilt)
        loc = locate_points(img, t, K, DIST)
        self.assertEqual(sorted(loc.points), sorted(t.ids()), (layout.name, distance, tilt))
        err = [np.linalg.norm(loc.points[i] - truth[i]) for i in t.ids()]
        self.assertLess(max(err), tol_px, (layout.name, distance, tilt))
        for res in loc.residual_px.values():
            self.assertLess(np.median(res), tol_px)

    def test_planar_plate_frontal_and_oblique(self):
        self._check(plate_a(), 0.6, 0.0, 0.15)
        self._check(plate_a(), 0.6, 40.0, 0.2)
        self._check(plate_a(), 0.9, 60.0, 0.3)

    def test_folded_plate_including_face_without_visible_marker(self):
        self._check(plate_l(), 0.6, 0.0, 0.2)
        # at this view the analysis sees no marker on face H: it is seeded by PnP
        self._check(plate_l(), 0.6, 20.0, 0.2)

    def test_does_not_depend_on_any_eyelab_output(self):
        t, img, truth = _scene(plate_a(), 0.6, 0.0)
        loc = locate_points(img, t, K, DIST)
        self.assertEqual(len(loc.points), len(t.ids()))


class TestJunction(unittest.TestCase):
    def setUp(self):
        img = np.full((80, 80), 230, np.uint8)
        img[20:40, 20:40] = 20   # saddle at (40, 40); L-corner at (20, 20)
        img[40:60, 40:60] = 20
        self.img = cv2.GaussianBlur(img, (0, 0), 1.0)

    def test_x_junction_accepted_l_corner_rejected(self):
        self.assertTrue(is_x_junction(self.img, (39.5, 39.5), 6.0))
        self.assertFalse(is_x_junction(self.img, (19.5, 19.5), 6.0))
        self.assertFalse(is_x_junction(self.img, (29.5, 19.5), 6.0))   # edge

    def test_l_corner_found_by_response_is_rejected_by_ring_test(self):
        # The Hessian response alone latches onto an L-corner (the first draft's
        # failure mode); the ring test is what rejects it.
        p = detect_saddle(self.img, (21.0, 21.0), 5.0)
        self.assertIsNotNone(p)
        self.assertFalse(is_x_junction(self.img, p, 6.0))
        q = detect_saddle(self.img, (41.0, 38.0), 5.0)
        np.testing.assert_allclose(q, [39.5, 39.5], atol=0.1)
        self.assertTrue(is_x_junction(self.img, q, 6.0))
        self.assertIsNone(detect_saddle(self.img, (70.0, 70.0), 3.0))


if __name__ == "__main__":
    unittest.main()
