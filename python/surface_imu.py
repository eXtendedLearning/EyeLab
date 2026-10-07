"""Device IMU (Windows sensor API) for camera setup: tilt, level and motion.

On a Surface (or any Windows tablet) the accelerometer gives the gravity
direction in the device frame, hence the tilt of a camera *built into* the
device. Heading (yaw) is not observable this way, and the magnetometer is not
trustworthy next to lab steel, so only gravity-referenced angles are given.
The IMU says nothing about an external webcam on a tripod.

Optional dependency, Windows only:
    pip install winrt-runtime winrt-Windows.Devices.Sensors
Everything degrades to ``None`` when the package or the sensor is missing
(desktop PC, Linux CI).

Device frame (Windows convention, natural orientation = landscape on a
Surface): +X right, +Y up, +Z out of the screen towards the user. A device
lying face-up reads about (0, 0, -1) g, i.e. the reading points *down*.
Rear camera looks along -Z, front camera along +Z. Run
``python surface_imu.py`` once to confirm the signs on your device.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

CAMERA_AXES = {"rear": np.array([0.0, 0.0, -1.0]), "front": np.array([0.0, 0.0, 1.0])}


@dataclass
class Tilt:
    axis_from_down_deg: float      # camera axis vs vertical-down; = plate-normal angle on a level plate
    elevation_deg: float           # camera axis below horizontal (90 = straight down)
    roll_deg: float                # image rotation about the camera axis (0 = level)
    g: float                       # |reading| in g (about 1 when still)


def tilt_from_gravity(down, camera: str = "rear") -> Tilt:
    """Camera angles from a gravity reading ``down`` (device frame, any units)."""
    d = np.asarray(down, float)
    n = float(np.linalg.norm(d))
    if n == 0:
        raise ValueError("zero gravity reading")
    d = d / n
    axis = CAMERA_AXES[camera]
    a = math.degrees(math.acos(float(np.clip(axis @ d, -1.0, 1.0))))
    # Roll: gravity projected on the image plane vs the image's "down" (-Y);
    # the rear camera image is mirrored in X with respect to the device frame.
    sx = -1.0 if camera == "rear" else 1.0
    roll = math.degrees(math.atan2(sx * d[0], -d[1])) if abs(d[2]) < 0.99 else float("nan")
    return Tilt(a, 90.0 - a, roll, n)


class MotionMonitor:
    """'Still' when the gravity reading barely changes over a short window."""

    def __init__(self, window: int = 25, max_sd_g: float = 0.004):
        self.buf: deque = deque(maxlen=window)
        self.max_sd_g = max_sd_g

    def add(self, g) -> None:
        self.buf.append(np.asarray(g, float))

    @property
    def sd_g(self) -> float:
        if len(self.buf) < 3:
            return float("nan")
        return float(np.linalg.norm(np.std(np.array(self.buf), axis=0)))

    @property
    def still(self) -> bool:
        return len(self.buf) == self.buf.maxlen and self.sd_g <= self.max_sd_g


class DeviceImu:
    """Thin wrapper on ``Windows.Devices.Sensors.Accelerometer`` (polling)."""

    def __init__(self, accelerometer, camera: str = "rear"):
        self._acc = accelerometer
        self.camera = camera
        self.motion = MotionMonitor()

    @classmethod
    def open(cls, camera: str = "rear") -> Optional["DeviceImu"]:
        try:
            from winrt.windows.devices.sensors import Accelerometer  # type: ignore
        except Exception:
            return None
        try:
            acc = Accelerometer.get_default()
        except Exception:
            return None
        if acc is None:
            return None
        try:
            acc.report_interval = max(acc.minimum_report_interval, 16)
        except Exception:
            pass
        return cls(acc, camera)

    def gravity(self) -> Optional[np.ndarray]:
        try:
            r = self._acc.get_current_reading()
        except Exception:
            return None
        if r is None:
            return None
        g = np.array([r.acceleration_x, r.acceleration_y, r.acceleration_z], float)
        self.motion.add(g)
        return g

    def read(self) -> Optional[dict]:
        g = self.gravity()
        if g is None:
            return None
        t = tilt_from_gravity(g, self.camera)
        return {"camera": self.camera, "axis_from_down_deg": t.axis_from_down_deg,
                "elevation_deg": t.elevation_deg, "roll_deg": t.roll_deg, "g": t.g,
                "motion_sd_g": self.motion.sd_g, "still": self.motion.still,
                "raw_g": [float(v) for v in g]}


def main() -> int:
    imu = DeviceImu.open()
    if imu is None:
        print("No accelerometer reachable (not Windows, package missing, or no sensor).")
        print("On a Surface: pip install winrt-runtime winrt-Windows.Devices.Sensors")
        return 1
    print("Lay the device flat, screen up: raw z should read about -1 g and the rear")
    print("camera 'axis_from_down' about 0 deg. Ctrl+C to stop.")
    try:
        while True:
            r = imu.read()
            if r:
                gx, gy, gz = r["raw_g"]
                print(f"\rraw g ({gx:+.3f} {gy:+.3f} {gz:+.3f})  rear axis from down "
                      f"{r['axis_from_down_deg']:6.2f} deg  elevation {r['elevation_deg']:6.2f}  "
                      f"roll {r['roll_deg']:+7.2f}  {'still ' if r['still'] else 'moving'}", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
