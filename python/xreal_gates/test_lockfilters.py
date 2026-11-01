import unittest

import numpy as np

from xreal_gates.lockfilters import Euro, Hysteresis, Latest, Meas, compare, q_angle_deg

Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def _noisy(n, sigma_m, rng, dt=0.135, start=0.0, offset=(0.0, 0.0, 0.8)):
    out = []
    for k in range(n):
        p = np.asarray(offset) + rng.normal(0, sigma_m, 3)
        out.append((start + k * dt, p))
    return out


class FilterTest(unittest.TestCase):
    def test_euro_and_hysteresis_calmer_than_latest_on_noise(self):
        rng = np.random.default_rng(3)
        data = _noisy(150, 0.004, rng)
        moves = {}
        for f in (Latest(), Euro(), Hysteresis()):
            prev, path = None, 0.0
            for t, p in data:
                f.add(p, Q0, t)
                if prev is not None:
                    path += np.linalg.norm(f.p - prev)
                prev = f.p.copy()
            moves[f.name] = path
        self.assertLess(moves["euro"], 0.5 * moves["latest"])
        self.assertLess(moves["hysteresis"], 0.5 * moves["latest"])

    def test_big_jump_needs_confirmation_then_snaps(self):
        for f in (Latest(), Euro(), Hysteresis()):
            f.add([0, 0, 0.8], Q0, 0.0)
            self.assertEqual(f.add([0.3, 0, 0.8], Q0, 0.1), "Pending")
            np.testing.assert_allclose(f.p, [0, 0, 0.8])
            self.assertEqual(f.add([0.305, 0, 0.8], Q0, 0.2), "Confirmed")
            np.testing.assert_allclose(f.p, [0.305, 0, 0.8])

    def test_single_outlier_is_ignored(self):
        f = Euro()
        f.add([0, 0, 0.8], Q0, 0.0)
        f.add([0.5, 0, 0.8], Q0, 0.1)          # wrong pose
        f.add([0.001, 0, 0.8], Q0, 0.2)        # back to normal
        self.assertLess(np.linalg.norm(f.p - [0, 0, 0.8]), 0.002)

    def test_hysteresis_holds_inside_threshold_and_follows_a_real_shift(self):
        f = Hysteresis()
        f.add([0, 0, 0.8], Q0, 0.0)
        for k in range(1, 20):
            f.add([0.003, 0, 0.8], Q0, 0.135 * k)    # inside 5 mm: never moves
        np.testing.assert_allclose(f.p, [0, 0, 0.8])
        for k in range(20, 60):
            f.add([0.03, 0, 0.8], Q0, 0.135 * k)     # 3 cm: follows within a few seconds
        self.assertLess(abs(f.p[0] - 0.03), 0.006)

    def test_compare_reports_all_columns(self):
        rng = np.random.default_rng(4)
        meas = [Meas(t, "static", p, Q0) for t, p in _noisy(80, 0.003, rng)]
        res = compare(meas, "board25")
        self.assertIn("static", res)
        self.assertEqual(set(res["static"]), {"rev3", "latest", "euro", "hysteresis"})
        self.assertLess(res["static"]["euro"].jitter_mm_s, res["static"]["rev3"].jitter_mm_s)

    def test_quaternion_angle(self):
        s = np.sqrt(0.5)
        self.assertAlmostEqual(q_angle_deg(Q0, [0, s, 0, s]), 90.0, places=6)


if __name__ == "__main__":
    unittest.main()
