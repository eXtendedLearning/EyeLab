"""Tests for the frame-age estimators used by ``python -m xreal_gates latency``."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from .latency import CausalQueueAge, causal_arrivals, estimate, queue_age


class QueueAgeTest(unittest.TestCase):
    def test_ramp_inside_window_is_measured_from_freshest_frame(self):
        # Two stream windows; inside each the host receives frames 0, 20, 40 ms later than capture spacing.
        rows = []
        seq = 0
        for start in (0.0, 5.0):
            for k, lag in enumerate((0.00, 0.02, 0.04)):
                cam = start + k * 0.1
                rows.append({"seq": str(seq), "t_host": str(100.0 + cam + 0.07 + lag), "cam_ts": str(int(cam * 1e9))})
                seq += 1
        age = queue_age(rows)
        self.assertAlmostEqual(age[0], 0.0, places=6)
        self.assertAlmostEqual(age[2], 0.04, places=6)
        self.assertAlmostEqual(age[3], 0.0, places=6)
        self.assertAlmostEqual(age[5], 0.04, places=6)

    def test_rows_without_cam_ts_are_skipped(self):
        self.assertEqual(queue_age([{"seq": "1", "t_host": "1.0", "cam_ts": ""}]), {})


class CausalQueueAgeTest(unittest.TestCase):
    """Mirror of CameraProbe.UpdateQueueAge in the Unity gate project: keep the two in step."""

    def test_sawtooth_with_restarts_is_measured_from_last_restart(self):
        est = CausalQueueAge(window_s=20.0)
        ages = []
        # Three stream windows of 5 s; frames 1/30 s apart at capture, delivered at 20 fps (queue grows).
        cam = 0.0
        for w in range(3):
            t = 10.0 + 6.0 * w
            cam = t - 0.08 + 100.0  # device clock offset 100 s; a restart's first frame is fresh
            for k in range(100):
                ages.append(est.add(t + k * 0.05, int((cam + k / 30.0 - 100.0 + 50.0) * 1e9))[1])
        self.assertAlmostEqual(ages[0], 0.0, places=9)
        self.assertAlmostEqual(ages[99], 99 * (0.05 - 1 / 30.0), places=6)
        self.assertAlmostEqual(ages[100], 0.0, places=6)
        self.assertAlmostEqual(ages[250], 50 * (0.05 - 1 / 30.0), places=6)

    def test_minimum_is_causal_and_expires_after_window(self):
        est = CausalQueueAge(window_s=1.0)
        est.add(0.0, int(0.0 * 1e9))          # D = 0.0
        self.assertAlmostEqual(est.add(0.5, int(0.3 * 1e9))[1], 0.2)   # D = 0.2 above 0.0
        # At t = 1.4 the D = 0.0 sample is older than 1 s; the floor is now 0.2.
        self.assertAlmostEqual(est.add(1.4, int(1.1 * 1e9))[1], 0.1)   # D = 0.3

    def test_causal_arrivals_skips_duplicates(self):
        rows = [{"t_unity": "1.0", "cam_ts": "1000000000", "dup_ts": "0"},
                {"t_unity": "1.1", "cam_ts": "1000000000", "dup_ts": "1"},
                {"t_unity": "1.2", "cam_ts": "1100000000", "dup_ts": "0"}]
        arr = causal_arrivals(rows)
        self.assertEqual(arr[1000000000], (1.0, 0.0))
        self.assertAlmostEqual(arr[1100000000][1], 0.1)


class LoggedPerFrameAgeTest(unittest.TestCase):
    """A synthetic 0.3.007 run: a head turning at constant rate, a fixed target, per-lock age_ms logged."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, header, rows):
        with (self.dir / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    def test_true_age_logged_gives_zero_residual(self):
        import numpy as np
        target = np.array([0.0, 0.0, 0.8])

        def q_at(t):
            # The head swings in yaw (up to 43 deg/s): a wrong age gives an error that changes from lock to
            # lock. (At a constant turn rate a constant age error moves every lock alike and is invisible.)
            a = 0.5 * np.sin(1.5 * t) / 2
            return np.array([0.0, np.sin(a), 0.0, np.cos(a)])  # x y z w, yaw about +y

        def rot(q, v):
            x, y, z, w = q
            u = np.array([x, y, z])
            return v + 2 * np.cross(u, np.cross(u, v) + w * v)

        pose = [["phase", "t_unity", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w"]]
        for i in range(0, 1200):
            t = i / 60.0
            pose.append(["update", t, 0, 0, 0, *q_at(t)])
        self._write("pose.csv", pose[0], pose[1:])
        (self.dir / "run.json").write_text(json.dumps({"device_xr": {"rgb_camera_pose_from_head": {
            "position_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}}}), encoding="utf-8")
        events, svc = [], []
        rng = np.random.default_rng(1)
        for seq in range(1, 80):
            t_send = 2.0 + seq * 0.2
            age = 0.09 + float(rng.uniform(0, 0.4))
            t_cap = t_send - age
            # Camera-frame target position (OpenCV: y down) from the head pose at capture.
            q = q_at(t_cap)
            qi = np.array([-q[0], -q[1], -q[2], q[3]])
            cam_unity = rot(qi, target)
            cv = cam_unity * np.array([1, -1, 1])
            svc.append([seq, 1, *cv, ""])
            events.append([t_send + 0.07, 0, "lock", "UPDATE",
                           f"target=plateA seq={seq} rt_ms=70 cam_ts= t_send={t_send:.4f} age_ms={age * 1000:.1f} world=0,0,0"])
        self._write("pose_service.csv", ["seq", "ok", "tx", "ty", "tz", "cam_ts"], svc)
        self._write("events.csv", ["t_unity", "frame", "event", "result", "detail"],
                    [[e[0], e[1], e[2], e[3], e[4]] for e in events])
        rows, fixed = estimate(self.dir, delays_ms=range(0, 101, 10))
        self.assertIsNone(fixed)
        best = min(rows, key=lambda r: r[1])
        self.assertEqual(best[0], 0)
        self.assertLess(best[1], 0.1)  # head interpolation at 60 Hz only
        self.assertGreater(dict((r[0], r[1]) for r in rows)[50], 10 * best[1])


if __name__ == "__main__":
    unittest.main()
