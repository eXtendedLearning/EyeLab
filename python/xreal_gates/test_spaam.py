import unittest

import numpy as np

from xreal_gates import spaam


def _taps(params, depths=(0.35, 0.6, 1.0)):
    """Points seen aligned with crosshairs at the four T6.11 positions, at each depth, for a true model ``params``."""
    C, w, s, p = params[:3], params[3], params[4], params[5:7]
    D = 1 / w
    P, uv = [], []
    for d in depths:
        for dx, dy in ((-8, 4), (8, 4), (8, -4), (-8, -4), (0, 0)):
            u, v = np.tan(np.radians(dx)), np.tan(np.radians(dy))
            X = D * np.array([s * u + p[0], s * v + p[1], 1.0])
            mu = (d - C[2]) / (X[2] - C[2])
            P.append(C + mu * (X - C))
            uv.append([u, v])
    return np.array(P), np.array(uv)


class SpaamModel(unittest.TestCase):
    TRUE = np.array([-0.003, 0.0005, 0.004, 0.25, 0.95, 0.003, -0.002])

    def test_sdk_model(self):
        """C = 0, s = 1, p = 0: the point straight along the crosshair, whatever the display distance."""
        P = np.array([[0.1, -0.05, 0.5], [0.0, 0.02, 1.2]])
        for w in (0.0, 0.25, 1.0):
            self.assertTrue(np.allclose(spaam.predict(np.array([0, 0, 0, w, 1, 0, 0.0]), P), P[:, :2] / P[:, 2:3]))

    def test_model_round_trip(self):
        P, uv = _taps(self.TRUE)
        self.assertTrue(np.allclose(spaam.predict(self.TRUE, P), uv, atol=1e-12))

    def test_lateral_eye_error_grows_near(self):
        """An eye 3 mm off sideways: the error at the target is about C_x (1 - d / D), zero at the display distance."""
        params = np.array([0.003, 0, 0, 0.25, 1, 0, 0.0])
        for d in (0.35, 1.0, 4.0):
            P = np.array([[0.0, 0.0, d]])
            err_mm = (spaam.predict(params, P)[0, 0] - 0.0) * d * 1000
            self.assertAlmostEqual(err_mm, -3.0 * (1 - d / 4.0), places=6)

    def test_fit_recovers(self):
        P, uv = _taps(self.TRUE)
        params, rms = spaam.fit_model(P, uv, fixed_w=0.25)
        self.assertLess(rms, 1e-6)
        self.assertTrue(np.allclose(params, self.TRUE, atol=1e-6))
        rng = np.random.default_rng(1)
        noisy = uv + rng.normal(0, np.radians(0.05), uv.shape)
        params, rms = spaam.fit_model(P, noisy, fixed_w=0.25)
        self.assertLess(np.linalg.norm(params[:2] - self.TRUE[:2]), 0.0015)   # eye x, y within 1.5 mm at 0.05 deg noise

    def test_dlt_centre(self):
        P, uv = _taps(self.TRUE)
        self.assertTrue(np.allclose(spaam.dlt_centre(P, uv), self.TRUE[:3], atol=1e-6))
        self.assertIsNone(spaam.dlt_centre(P[:5], uv[:5]))

    def test_leave_one_out(self):
        """Exact taps: a fresh tap is predicted exactly; noisy taps: about the noise, and None when too few."""
        P, uv = _taps(self.TRUE)
        self.assertLess(np.abs(spaam.leave_one_out(P, uv, "full")).max(), 1e-4)
        eye_only = np.array([-0.003, 0.0005, 0.004, 0.25, 1, 0, 0.0])
        P, uv = _taps(eye_only)
        rng = np.random.default_rng(2)
        loo = spaam.leave_one_out(P, uv + rng.normal(0, np.radians(0.2), uv.shape), "eye")
        self.assertLess(np.median(np.linalg.norm(loo, axis=1)), 4.0)   # 0.2 deg at 35-100 cm
        self.assertIsNone(spaam.leave_one_out(P[:2], uv[:2], "full"))

    def test_trusted_board(self):
        row = {"pose_ok": "1", "target": "board25", "accepted": "0", "board_markers": "16", "rms_px": "1.0", "t2": "0.5"}
        self.assertTrue(spaam.trusted_board(row))
        self.assertFalse(spaam.trusted_board(dict(row, board_markers="8")))
        self.assertFalse(spaam.trusted_board(dict(row, t2="-0.5")))
        self.assertTrue(spaam.trusted_board(dict(row, board_markers="8", accepted="1")))

    def test_device_formula(self):
        """EyeModel.cs's Predict (eye centre only, display at 4 m) is the Python model "eye"."""
        c = np.array([0.0094, -0.0074, -0.0148])
        P = np.array([[0.02, -0.03, 0.4], [-0.1, 0.05, 0.9]])
        w = 0.25
        u = w * c[0] + (1 - w * c[2]) * (P[:, 0] - c[0]) / (P[:, 2] - c[2])
        v = w * c[1] + (1 - w * c[2]) * (P[:, 1] - c[1]) / (P[:, 2] - c[2])
        self.assertTrue(np.allclose(spaam.predict(np.array([*c, w, 1, 0, 0]), P), np.stack([u, v], 1)))

    def test_pose_from_meta(self):
        R, t, ok = spaam.pose_from_meta("ok; p -0.0315 0.001 0.02; q 0 0 0 1")
        self.assertTrue(ok and np.allclose(t, [-0.0315, 0.001, 0.02]) and np.allclose(R, np.eye(3)))
        self.assertFalse(spaam.pose_from_meta("")[2])


if __name__ == "__main__":
    unittest.main()
