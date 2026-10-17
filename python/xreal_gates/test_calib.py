import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xreal_gates.calib import calibrate_run, coverage, latest_calib_run, select_views, View
from xreal_gates.eyecam import vendor_camera


def _render_board(board, K, rvec, tvec, ppm=8.0):
    cols, rows = board.getChessboardSize()
    sq = board.getSquareLength()
    page = board.generateImage((int(cols * sq * 1000 * ppm), int(rows * sq * 1000 * ppm)), marginSize=0)
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    A = np.array([[1 / (1000 * ppm), 0, 0], [0, 1 / (1000 * ppm), 0], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, float)]) @ A
    out = np.full((720, 1280), 128, np.uint8)
    warped = cv2.warpPerspective(page, Hm, (1280, 720), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(page, 255), Hm, (1280, 720)) > 0
    out[mask] = warped[mask]
    return out


class CalibTest(unittest.TestCase):
    def test_recovers_known_intrinsics(self):
        from xreal_gates.calib import _board
        board, _ = _board()
        K = vendor_camera().K
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261016_150000_Calib_calib"
            (run / "snapshots").mkdir(parents=True)
            n = 0
            for z in (0.3, 0.45):
                for x in (-0.25, -0.1, 0.05):
                    for y in (-0.12, 0.0, 0.1):
                        rvec = rng.normal(0, 0.25, 3)
                        img = _render_board(board, K, rvec, (x * z / 0.45, y * z / 0.45, z))
                        cv2.imwrite(str(run / "snapshots" / f"{n:04d}.pgm"), img)
                        n += 1
            self.assertEqual(latest_calib_run(Path(tmp)), run)
            r = calibrate_run(run)
            self.assertLess(r.rms_px, 1.0)
            self.assertLess(abs(r.K[0, 0] - K[0, 0]) / K[0, 0], 0.01)
            self.assertLess(abs(r.K[0, 2] - K[0, 2]), 6.0)
            self.assertTrue(r.yaml_path.exists() and r.report_path.exists())

    def test_refuses_board_seen_in_one_place(self):
        from xreal_gates.calib import _board
        board, _ = _board()
        K = vendor_camera().K
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261016_150000_Calib_calib"
            (run / "snapshots").mkdir(parents=True)
            for n in range(20):
                img = _render_board(board, K, rng.normal(0, 0.05, 3), (-0.06, -0.08, 0.8))
                cv2.imwrite(str(run / "snapshots" / f"{n:04d}.pgm"), img)
            with self.assertRaises(ValueError):
                calibrate_run(run)
            self.assertFalse((run / "eye_calibration.yaml").exists())

    def test_select_views_spreads_and_caps(self):
        views = [View(Path(f"{i}.pgm"), np.array([[[x, 360.0]]]), np.arange(10 + i % 5).reshape(-1, 1),
                      np.array([x, 360.0]), 1000.0 + i)
                 for i, x in enumerate(np.linspace(10, 1270, 100))]
        chosen = select_views(views, 20)
        self.assertEqual(len(chosen), 20)
        self.assertGreater(np.ptp([v.centre[0] for v in chosen]), 900)
        self.assertGreater(coverage(chosen), 0.0)


if __name__ == "__main__":
    unittest.main()
