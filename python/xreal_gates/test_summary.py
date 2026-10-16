import tempfile
import unittest
from pathlib import Path

from xreal_gates.summary import _detail, _full_clock, _recv_fps


class FullClockTest(unittest.TestCase):
    def _perf(self, clocks, t0=10.0):
        return [{"t_unity": str(t0 + i), "cpu_performance_pct": str(c)} for i, c in enumerate(clocks)]

    def test_throttle_needs_three_samples_below(self):
        perf = self._perf([220, 220, 140, 220, 220, 140, 130, 120, 110])
        self.assertEqual(_full_clock(perf, 10.0), (5.0, False))

    def test_never_throttled(self):
        self.assertEqual(_full_clock(self._perf([220] * 10), 10.0), (None, True))

    def test_no_xr(self):
        self.assertEqual(_full_clock(self._perf([60] * 10), None), (None, False))


class RecvFpsTest(unittest.TestCase):
    def test_longest_stretch_of_the_counter(self):
        lines = [
            "10.0 Log [x] [INFO] [NRSDK] rgb recv, 121, 62206, 1280 720",
            "26.0 Log [x] [INFO] [NRSDK] rgb recv, 601, 69100, 1280 720",
            "42.0 Log [x] [INFO] [NRSDK] rgb recv, 1081, 76650, 1280 720",
            "43.0 Log [x] [INFO] [NRSDK] rgb recv, 1, 70000, 1280 720",  # new capture session
            "44.0 Log [x] [INFO] [NRSDK] gray recv, 121, 193536, 504 378, 1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertAlmostEqual(_recv_fps(path), 30.0)

    def test_missing_log(self):
        self.assertIsNone(_recv_fps(Path("does-not-exist.log")))


class DetailTest(unittest.TestCase):
    def test_key_value(self):
        self.assertEqual(_detail("n=3 frames=3 first_s=0.912 on_s=1.0", "first_s"), "0.912")
        self.assertIsNone(_detail("n=3", "first_s"))


if __name__ == "__main__":
    unittest.main()
