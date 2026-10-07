"""Gravity -> camera angles; graceful absence of the Windows sensor API."""
import sys
import unittest

import numpy as np

from surface_imu import DeviceImu, MotionMonitor, tilt_from_gravity


class TestTilt(unittest.TestCase):
    def test_flat_face_up_rear_camera_looks_down(self):
        t = tilt_from_gravity([0.0, 0.0, -1.0], "rear")
        self.assertAlmostEqual(t.axis_from_down_deg, 0.0, places=6)
        self.assertAlmostEqual(t.elevation_deg, 90.0, places=6)
        self.assertAlmostEqual(tilt_from_gravity([0, 0, -1.0], "front").axis_from_down_deg, 180.0)

    def test_upright_and_tilted(self):
        up = tilt_from_gravity([0.0, -1.0, 0.0], "rear")       # standing in landscape
        self.assertAlmostEqual(up.elevation_deg, 0.0, places=6)
        self.assertAlmostEqual(up.roll_deg, 0.0, places=6)
        a = np.radians(30.0)                                   # top tipped away by 60 deg
        t = tilt_from_gravity([0.0, -np.sin(a), -np.cos(a)], "rear")
        self.assertAlmostEqual(t.axis_from_down_deg, 30.0, places=6)
        self.assertAlmostEqual(t.elevation_deg, 60.0, places=6)
        r = tilt_from_gravity([np.sin(np.radians(5)), -np.cos(np.radians(5)), 0.0], "rear")
        self.assertAlmostEqual(abs(r.roll_deg), 5.0, places=6)
        self.assertAlmostEqual(tilt_from_gravity([0, 0, -2.0]).g, 2.0)

    def test_motion_monitor(self):
        m = MotionMonitor(window=10, max_sd_g=0.004)
        for _ in range(10):
            m.add([0, 0, -1.0])
        self.assertTrue(m.still)
        m.add([0.05, 0, -1.0])
        self.assertFalse(m.still)

    @unittest.skipIf(sys.platform == "win32", "a Windows machine may have a real sensor")
    def test_no_sensor_api_off_windows(self):
        self.assertIsNone(DeviceImu.open())


if __name__ == "__main__":
    unittest.main()
