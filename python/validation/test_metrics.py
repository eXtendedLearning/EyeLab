import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from validation.metrics import (chordal_mean_rotation, detection_stats, pose_jitter,
                                rotation_angle_deg, step_error)
from validation.session import FrameRecord, Pose
from validation.stats import across_units, summarize


def _rec(i, measured=True, coasted=False, n=3):
    p = Pose(np.zeros(3), np.array([0, 0, 0.6]), tuple(range(n)), coasted)
    return FrameRecord(i, float(i), None, p if (measured or coasted) else None,
                       None if coasted or not measured else p, "locked", (), {}, {}, {})


class TestMetrics(unittest.TestCase):
    def test_translation_jitter_in_mm_from_metres(self):
        rng = np.random.default_rng(1)
        t = np.array([0, 0, 0.6]) + rng.normal(0, 0.0005, (5000, 3))   # 0.5 mm SD
        j = pose_jitter(np.zeros((5000, 3)), t)
        np.testing.assert_allclose(j["t_sd_xyz_mm"], 0.5, rtol=0.05)
        self.assertLess(j["rot_dev_deg"].max(), 1e-9)

    def test_rotation_jitter_recovers_angle(self):
        rng = np.random.default_rng(2)
        R0 = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))[0]
        rv = []
        for _ in range(3000):
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            rv.append(cv2.Rodrigues(cv2.Rodrigues(a * np.radians(rng.normal(0, 0.2)))[0] @ R0)[0].ravel())
        j = pose_jitter(np.array(rv), np.zeros((3000, 3)))
        self.assertAlmostEqual(np.sqrt(np.mean(j["rot_dev_deg"] ** 2)), 0.2, delta=0.02)

    def test_chordal_mean_identity(self):
        R = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]
        self.assertLess(rotation_angle_deg(chordal_mean_rotation(np.stack([R, R])), R), 1e-4)

    def test_step_error_mm(self):
        a = SimpleNamespace(poses=lambda w: (np.zeros((2, 3)), np.array([[0, 0, 0.6]] * 2)))
        b = SimpleNamespace(poses=lambda w: (np.zeros((2, 3)), np.array([[0.0102, 0, 0.6]] * 2)))
        r = step_error(a, b, 10.0)
        self.assertAlmostEqual(r["estimated_mm"], 10.2)
        self.assertAlmostEqual(r["error_mm"], 0.2)

    def test_coasted_frames_are_not_counted_as_measured(self):
        recs = [_rec(0), _rec(1, n=1), _rec(2, coasted=True), _rec(3, measured=False)]
        d = detection_stats(recs)
        self.assertAlmostEqual(d["measured_rate"], 0.5)
        self.assertAlmostEqual(d["displayed_rate"], 0.75)
        self.assertAlmostEqual(d["coasted_rate"], 0.25)
        self.assertEqual(d["markers_in_measured_pose"], {"1": 1, "3": 1})

    def test_summarize(self):
        s = summarize([3.0, 4.0, float("nan")])
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["rmse"], np.sqrt(12.5))
        self.assertNotIn("mean_ci95", s)
        self.assertEqual(summarize([])["n"], 0)
        self.assertEqual(across_units([1.0, 3.0])["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
