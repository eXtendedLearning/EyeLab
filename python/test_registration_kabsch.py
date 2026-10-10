import unittest

import numpy as np

from registration import MarkerCorrespondence, SpatialRegistration


def _rot_z(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# Non-coplanar tetrahedron in the UNV frame (metres).
_UNV_PTS = {
    10: np.array([0.0, 0.0, 0.0]),
    11: np.array([1.0, 0.0, 0.0]),
    12: np.array([0.0, 1.0, 0.0]),
    13: np.array([0.0, 0.0, 1.0]),
}


def _registration_for(R, t):
    corrs = [MarkerCorrespondence(marker_id=mid, unv_position=p)
             for mid, p in _UNV_PTS.items()]
    reg = SpatialRegistration(corrs)
    for mid, p in _UNV_PTS.items():
        reg.update_detected_position(mid, R @ p + t)
    return reg


class KabschRegistrationTests(unittest.TestCase):
    def test_recovers_known_rigid_transform(self):
        R = _rot_z(30.0)
        t = np.array([0.10, 0.20, 0.30])
        result = _registration_for(R, t).compute()

        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result.R, R, atol=1e-9))
        self.assertTrue(np.allclose(result.t.flatten(), t, atol=1e-9))
        self.assertLess(result.rms_error_mm, 1e-6)

    def test_rotation_is_proper_no_reflection(self):
        # Reflection correction must guarantee det(R) = +1, never -1.
        result = _registration_for(_rot_z(127.0), np.array([1.0, -2.0, 0.5])).compute()
        self.assertAlmostEqual(float(np.linalg.det(result.R)), 1.0, places=9)

    def test_transform_point_round_trips(self):
        R = _rot_z(45.0)
        t = np.array([0.5, 0.0, -0.2])
        reg = _registration_for(R, t)
        reg.compute()
        for p in _UNV_PTS.values():
            self.assertTrue(np.allclose(reg.transform_point(p), R @ p + t, atol=1e-9))

    def test_collinear_markers_rejected(self):
        # Four points on the x-axis -> ill-conditioned -> compute() returns None.
        pts = {i: np.array([float(i), 0.0, 0.0]) for i in range(4)}
        corrs = [MarkerCorrespondence(marker_id=i, unv_position=p) for i, p in pts.items()]
        reg = SpatialRegistration(corrs)
        for i, p in pts.items():
            reg.update_detected_position(i, p + np.array([1.0, 1.0, 1.0]))
        self.assertIsNone(reg.compute())

    def _planar(self, pts, R, t):
        corrs = [MarkerCorrespondence(marker_id=i, unv_position=p) for i, p in enumerate(pts)]
        reg = SpatialRegistration(corrs)
        for i, p in enumerate(pts):
            reg.update_detected_position(i, R @ p + t)
        return reg.compute()

    @staticmethod
    def _rot(axis, deg):
        axis = np.asarray(axis, float)
        axis = axis / np.linalg.norm(axis)
        a = np.deg2rad(deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)

    def test_exact_planar_triangle_is_registered(self):
        # Audit F06: an exact 100 mm triangle, rigidly moved without noise.
        tri = [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0])]
        R = self._rot([1, 2, 3], 37.0)
        t = np.array([0.3, -0.1, 0.8])
        result = self._planar(tri, R, t)
        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result.R, R, atol=1e-9))
        self.assertTrue(np.allclose(result.t.flatten(), t, atol=1e-9))
        self.assertAlmostEqual(float(np.linalg.det(result.R)), 1.0, places=9)
        self.assertLess(result.rms_error_mm, 1e-6)

    def test_larger_planar_set_is_registered(self):
        grid = [np.array([x, y, 0.0]) for x in (0.0, 0.05, 0.1, 0.15) for y in (0.0, 0.08, 0.16)]
        R = self._rot([0, 1, 0], 170.0)
        t = np.array([1.0, 2.0, 3.0])
        result = self._planar(grid, R, t)
        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result.R, R, atol=1e-9))

    def test_planar_registration_is_scale_independent(self):
        for scale in (1e-3, 1.0, 1e3):
            tri = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.2, 0.9, 0.0])]
            tri = [p * scale for p in tri]
            R = self._rot([1, 1, 0], 25.0)
            result = self._planar(tri, R, np.array([0.1, 0.2, 0.3]) * scale)
            self.assertIsNotNone(result, f"scale={scale}")
            self.assertTrue(np.allclose(result.R, R, atol=1e-8), f"scale={scale}")

    def test_coincident_points_rejected(self):
        pts = [np.array([0.2, 0.2, 0.2])] * 4
        self.assertIsNone(self._planar(pts, np.eye(3), np.zeros(3)))

    def test_nearly_collinear_points_rejected_at_any_scale(self):
        for scale in (0.01, 1.0, 100.0):
            pts = [np.array([float(i), 1e-5 * (i % 2), 0.0]) * scale for i in range(5)]
            self.assertIsNone(self._planar(pts, np.eye(3), np.zeros(3)), f"scale={scale}")

    def test_collinear_detections_rejected_even_if_model_is_planar(self):
        # Model spans a plane but the detected positions are collinear.
        tri = [np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0])]
        corrs = [MarkerCorrespondence(marker_id=i, unv_position=p) for i, p in enumerate(tri)]
        reg = SpatialRegistration(corrs)
        for i in range(3):
            reg.update_detected_position(i, np.array([0.1 * i, 0.0, 0.0]))
        self.assertIsNone(reg.compute())

    def test_too_few_correspondences_returns_none(self):
        corrs = [MarkerCorrespondence(marker_id=i, unv_position=_UNV_PTS[10 + i])
                 for i in range(2)]
        reg = SpatialRegistration(corrs)
        for i in range(2):
            reg.update_detected_position(i, _UNV_PTS[10 + i])
        self.assertIsNone(reg.compute())

    def test_residuals_reported_for_noisy_detection(self):
        R = _rot_z(10.0)
        t = np.zeros(3)
        reg = _registration_for(R, t)
        # Perturb one detection by 5 mm; registration should still solve and
        # report a non-zero RMS.
        reg.update_detected_position(11, R @ _UNV_PTS[11] + t + np.array([0.005, 0.0, 0.0]))
        result = reg.compute()
        self.assertIsNotNone(result)
        self.assertGreater(result.rms_error_mm, 0.0)
        self.assertIn(11, result.per_marker_errors_mm)


if __name__ == "__main__":
    unittest.main()
