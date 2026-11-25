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
        """EyeModel.cs's Predict ([C, s, p], display at 4 m) is the Python model with w = 0.25."""
        c = np.array([0.0094, -0.0074, -0.0148])
        s, px, py = 0.975, -0.0138, 0.0113
        P = np.array([[0.02, -0.03, 0.4], [-0.1, 0.05, 0.9]])
        w = 0.25
        u = (w * c[0] + (1 - w * c[2]) * (P[:, 0] - c[0]) / (P[:, 2] - c[2]) - px) / s
        v = (w * c[1] + (1 - w * c[2]) * (P[:, 1] - c[1]) / (P[:, 2] - c[2]) - py) / s
        self.assertTrue(np.allclose(spaam.predict(np.array([*c, w, s, px, py]), P), np.stack([u, v], 1)))

    def test_device_model_recovers_an_angle_and_scale(self):
        """T6.13: the device's model ("display") recovers a fixed angle and scale with the SDK's eye kept."""
        true = np.array([0, 0, 0, 0.25, 0.975, np.tan(np.radians(-0.5)), np.tan(np.radians(0.8))])
        P, uv = _taps(true)
        params, rms = spaam.fit_model(P, uv, fixed_w=0.25, model=spaam.DEVICE_MODEL)
        self.assertLess(rms, 1e-6)
        self.assertTrue(np.allclose(params, true, atol=1e-7))

    def test_pose_from_meta(self):
        R, t, ok = spaam.pose_from_meta("ok; p -0.0315 0.001 0.02; q 0 0 0 1")
        self.assertTrue(ok and np.allclose(t, [-0.0315, 0.001, 0.02]) and np.allclose(R, np.eye(3)))
        self.assertFalse(spaam.pose_from_meta("")[2])


if __name__ == "__main__":
    unittest.main()


class TapPointSourcesTest(unittest.TestCase):
    """T6.13: the tap point follows what the run logged, and fills in what T6.11 run 1 did not log."""

    def test_board_centre_from_meta_or_nominal(self):
        from xreal_gates.spaam import BOARD_CENTRE, board_centre
        np.testing.assert_allclose(board_centre({}), BOARD_CENTRE)
        np.testing.assert_allclose(board_centre({"board_centre_m": "0.0649 0.0909 0"}), [0.0649, 0.0909, 0.0])

    def test_rgb_from_head_filled_in_only_when_missing(self):
        from xreal_gates import b3a
        from xreal_gates.spaam import rgb_meta
        R, t, ok = b3a.rgb_from_head(rgb_meta({}))
        self.assertTrue(ok)
        self.assertAlmostEqual(float(t[2]), 0.0281137042, places=9)
        logged = {"rgb_from_head": "ok; p 0 0 0; q 0 0 0 1"}
        self.assertIs(rgb_meta(logged), logged)

    def test_camera_key_from_the_set_line(self):
        from xreal_gates.spaam import run_camera
        meta = {"set": "eyelab-b3a/1; board plate_A; calibration cameras/beampro_eye.json 20261124_camcal; frames 32; markers 9"}
        self.assertEqual(run_camera(meta), "cameras/beampro_eye.json 20261124_camcal")
        self.assertEqual(run_camera({}), "?")

    def test_board_scale_re_poses_at_the_printed_size(self):
        """T6.13: a run that posed board25 at 25 mm (no centre logged) or 25.97 mm squares is scaled to the print."""
        from gui_common import CHARUCO_SQUARE_PRINTED_M
        from xreal_gates.spaam import board_scale
        self.assertAlmostEqual(board_scale({}), CHARUCO_SQUARE_PRINTED_M / 0.025, places=9)
        self.assertAlmostEqual(board_scale({"board_centre_m": f"{2.5 * 0.02597} {3.5 * 0.02597} 0"}),
                               CHARUCO_SQUARE_PRINTED_M / 0.02597, places=9)
        printed = CHARUCO_SQUARE_PRINTED_M
        self.assertAlmostEqual(board_scale({"board_centre_m": f"{2.5 * printed} {3.5 * printed} 0"}), 1.0, places=9)

    def test_repose_resolves_with_another_camera(self):
        """T6.12 run 3: a pose re-solved from the logged corners under the camera they came from is the true pose (from a
        nominal-size start), and a principal point 5 px off turns it by about 5 / fx rad, which a tape cannot see."""
        import math
        import tempfile
        from pathlib import Path
        import cv2
        from xreal_gates.spaam import _resolve
        K = np.array([[627.0, 0, 641.3], [0, 625.7, 360.8], [0, 0, 1]])
        sq = 0.03417
        board = {i: np.array([[x, y, 0], [x + 0.026, y, 0], [x + 0.026, y + 0.026, 0], [x, y + 0.026, 0]])
                 for i, (x, y) in enumerate([(0.004, 0.004), (0.072, 0.004), (0.004, 0.072), (0.072, 0.072), (0.14, 0.14)])}
        rv, tv = np.array([0.1, -0.2, 0.05]), np.array([-0.08, -0.1, 0.6])
        rows = ["seq,id,x0,y0,x1,y1,x2,y2,x3,y3"]
        for i, c in board.items():
            px = cv2.projectPoints(c, rv, tv, K, np.zeros(5))[0].reshape(-1)
            rows.append("7," + str(i) + "," + ",".join(f"{v:.4f}" for v in px))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "live_corners.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            k = sq / 0.025   # the run posed the board at 25 mm squares: its logged tvec is k times too near
            row = {"seq": "7", "pose_ok": "1", **{f"r{j}": str(rv[j]) for j in range(3)},
                   **{f"t{j}": str(tv[j] / k) for j in range(3)}}
            got = _resolve([row], run, {"K": K, "dist": np.zeros(5), "board": board}, k)["7"]
            np.testing.assert_allclose(got[1], tv, atol=2e-4)
            K2 = K.copy()
            K2[0, 2] -= 5.0
            moved = _resolve([row], run, {"K": K2, "dist": np.zeros(5), "board": board}, k)["7"]
            a = math.degrees(math.atan2(moved[1][0], moved[1][2]) - math.atan2(got[1][0], got[1][2]))
            self.assertAlmostEqual(abs(a), math.degrees(5.0 / 627.0), delta=0.05)
