import csv
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from xreal_gates import beampro, guided

EV_HEADER = ["event", "detail", "mono_ns", "boot_ns", "real_ns", "hmd_ns", "mono_after_ns", "unity_s"]
HEAD_HEADER = ["unity_frame", "mono_ns", "tracking_type", "valid", "tracked", "state",
               "px", "py", "pz", "qx", "qy", "qz", "qw"]


def _w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _ev(name, detail, t_ns):
    return [name, detail, int(t_ns), int(t_ns), 0, int(t_ns), int(t_ns), t_ns / 1e9]


def _yaw_q(deg):
    h = math.radians(deg) / 2
    return (0.0, math.sin(h), 0.0, math.cos(h))   # rotation about y (Unity up)


class LoopTest(unittest.TestCase):
    def test_phases_turns_and_closure(self):
        steps = [("ready", 2), ("still_1", 6), ("turn_left", 4), ("front_1", 4), ("walk_out", 4), ("still_2", 6)]
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            ev, t = [], 0.0
            bounds = {}
            for name, sec in steps:
                ev.append(_ev("cue", name, t * 1e9))
                bounds[name] = (t, t + sec)
                t += sec
            ev.append(_ev("cue", "end", t * 1e9))
            _w(run / "events.csv", EV_HEADER, ev)
            rows = []
            for k in range(int(t * 50)):
                tt = k / 50
                yaw, x = 0.0, 0.0
                if bounds["turn_left"][0] <= tt < bounds["turn_left"][1]:
                    yaw = -88.0 * min(1.0, (tt - bounds["turn_left"][0]) / 2)
                elif bounds["front_1"][0] <= tt < bounds["front_1"][1]:
                    yaw = -88.0 * max(0.0, 1 - (tt - bounds["front_1"][0]) / 2)
                if tt >= bounds["still_2"][0]:
                    x = 0.12   # came back 12 cm off the mark
                if abs(tt - bounds["walk_out"][0] - 1.0) < 1e-6:
                    x = 0.5    # one relocalisation jump
                q = _yaw_q(yaw)
                rows.append([k, int(tt * 1e9), "MODE_6DOF", 1, 1, 3, x, -0.4, 0.2, *q])
            _w(run / "head.csv", HEAD_HEADER, rows)
            text = "\n".join(guided.loop_report(run))
        self.assertIn("turn_left: yaw change -88.0 deg", text)
        self.assertIn("front_1: yaw change +88.0 deg", text)
        self.assertIn("LOOP CLOSURE (still_2 - still_1, same floor mark): horizontal 120 mm, height +0 mm", text)
        self.assertIn("jump 500 mm", text)
        self.assertIn("phase walk_out", text)


def _texture(h, w, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    import cv2
    t = rng.random((h, w)).astype(np.float32)
    return cv2.GaussianBlur(t, (0, 0), 2.0)


class ShiftTest(unittest.TestCase):
    def test_phase_correlation_shift(self):
        import numpy as np
        tex = _texture(200, 300)
        a = (tex[50:140, 50:210] * 255).astype(np.uint8)
        b = (tex[50:140, 53:213] * 255).astype(np.uint8)   # content moves 3 px left
        s = guided.image_shifts([a, b])
        self.assertIsNone(s[0])
        self.assertAlmostEqual(s[1][0], -3.0, delta=0.2)
        self.assertAlmostEqual(s[1][1], 0.0, delta=0.2)


def _motion(t_s):
    """Yaw angle (rad) of a 1.1 Hz shake plus a slower 0.4 Hz sway."""
    return 0.25 * math.sin(2 * math.pi * 1.1 * t_s) + 0.1 * math.sin(2 * math.pi * 0.4 * t_s + 1.0)


class OffsetTest(unittest.TestCase):
    def test_recovers_delta(self):
        ref_t = [i * 5e6 for i in range(2400)]              # 12 s, 5 ms grid
        theta = [(0.0, _motion(t / 1e9), 0.0) for t in ref_t]
        frames = [1e9 + k * 33.3e6 for k in range(300)]
        true_d = -27.0
        iv, ys = [], []
        for a, b in zip(frames, frames[1:]):
            dyaw = _motion((b + true_d * 1e6) / 1e9) - _motion((a + true_d * 1e6) / 1e9)
            iv.append((a, b))
            ys.append((-260.0 * dyaw, 0.3 * dyaw))
        r = guided.best_offset(iv, ys, ref_t, theta)
        self.assertAlmostEqual(r["delta_ms"], true_d, delta=1.0)
        self.assertGreater(r["r2"], 0.99)
        self.assertLess(r["r2_zero"], r["r2"])

    def test_rotation_increments(self):
        def rz(a):
            c, s = math.cos(a), math.sin(a)
            return [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        acc = guided.cumulative_rotation([0, 1, 2], [rz(0.0), rz(0.1), rz(0.3)])
        self.assertAlmostEqual(acc[-1][2], 0.3, places=9)


class PanTest(unittest.TestCase):
    def test_sensor_offset(self):
        import numpy as np
        w, h = 160, 90
        tex = _texture(400, 1200, seed=3)
        px_per_rad = 300.0
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            t0 = 5_000_000_000_000
            imu = []
            for i in range(400 * 12):                      # 12 s of gyro at 400 Hz, rate = d(angle)/dt about y
                t = t0 + i * 2_500_000
                ts = (t - t0) / 1e9
                rate = (_motion(ts + 1e-4) - _motion(ts - 1e-4)) / 2e-4
                imu.append([4, t, t + 1_000_000, t, 0.0, rate, 0.0, 3])
            _w(run / "imu.csv", ["type", "event_ns", "boot_ns", "mono_ns", "v0", "v1", "v2", "accuracy"], imu)
            offsets = {"0": 0.0, "2": -12.0}             # sensor 2 exposes 12 ms before its stamp
            for pid, off in offsets.items():
                with open(run / f"stereo_thumbs_{pid}.bin", "wb") as f:
                    for k in range(300):
                        stamp = t0 + 1_000_000_000 + k * 33_300_000
                        ang = _motion((stamp - t0) / 1e9 + off / 1e3)
                        x = 500.0 + px_per_rad * ang
                        xi = int(math.floor(x))
                        fr = x - xi
                        crop = (1 - fr) * tex[150:150 + h, xi:xi + w] + fr * tex[150:150 + h, xi + 1:xi + 1 + w]
                        f.write(struct.pack(">q", stamp) + (crop * 255).astype(np.uint8).tobytes())
            (run / "run.json").write_text(json.dumps({"probe": "pan", "stereo_start": {"result": "opening"},
                                                      "stereo_stop": {"result": "streaming", "thumb_size": f"{w}x{h}"}}))
            _w(run / "stereo.csv", ["frame", "logical_ts", "phys_a", "ts_a", "phys_b", "ts_b", "boot_ns_at_result"], [])
            text = "\n".join(beampro.report(run))
        self.assertIn("sensor 0 exposure - image timestamp (gyro)", text)
        line = next(x for x in text.splitlines() if "EXPOSURE OFFSET" in x)
        val = float(line.split(":")[1].split("ms")[0])
        self.assertAlmostEqual(val, 12.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
