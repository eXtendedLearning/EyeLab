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
