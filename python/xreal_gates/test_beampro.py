import csv
import json
import tempfile
import unittest
from pathlib import Path

from xreal_gates import beampro


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


FRAME_HEADER = ["seq", "cam_ts", "res_w", "res_h", "plane_w", "plane_h", "drained", "unity_frame",
                "mono_ns", "boot_ns", "real_ns", "hmd_ns", "mono_after_ns", "unity_s", "saved"]


def _frames(n=60, age_ms=80.0, cam_unit="ns", cam_clock="boot", saved=None):
    rows = []
    boot0, mono0, real0 = 5_000_000_000_000, 4_000_000_000_000, 1_700_000_000_000_000_000
    for i in range(n):
        t = i * 33_333_333 + (i % 3) * 1_000_000
        boot, mono = boot0 + t + int(age_ms * 1e6), mono0 + t + int(age_ms * 1e6)
        exposure = (boot0 if cam_clock == "boot" else mono0) + t
        cam = exposure // beampro.UNITS[cam_unit]
        rows.append([i, int(cam), 1280, 720, 1280, 720, 0, i, mono, boot, real0 + t, 0, mono + 1000, i / 30.0,
                     (saved or {}).get(i, "")])
    return rows


class ClockDomainTest(unittest.TestCase):
    def test_boot_ns(self):
        rows = [dict(zip(FRAME_HEADER, map(str, r))) for r in _frames()]
        best, _ = beampro.clock_domain(rows)
        self.assertEqual((best.clock, best.unit), ("boot_ns", "ns"))
        self.assertAlmostEqual(best.median_ms, 81.0, delta=1.5)

    def test_mono_us(self):
        rows = [dict(zip(FRAME_HEADER, map(str, r))) for r in _frames(cam_unit="us", cam_clock="mono")]
        best, _ = beampro.clock_domain(rows)
        self.assertEqual((best.clock, best.unit), ("mono_ns", "us"))

    def test_no_match(self):
        rows = [dict(zip(FRAME_HEADER, map(str, r))) for r in _frames()]
        for r in rows:
            r["cam_ts"] = str(int(r["cam_ts"]) + 10**15)
        best, cands = beampro.clock_domain(rows)
        self.assertIsNone(best)
        self.assertTrue(cands)


class ImuTest(unittest.TestCase):
    def test_rates(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            rows = []
            for i in range(400):
                t = 10**12 + i * 2_500_000  # 400 Hz
                rows.append([4, t, t + 1_000_000, t, 0, 0, 0, 3])
            _write_csv(run / "imu.csv", ["type", "event_ns", "boot_ns", "mono_ns", "v0", "v1", "v2", "accuracy"], rows)
            line = beampro.imu_report(run)[0]
            self.assertIn("gyroscope", line)
            self.assertIn("400.0 Hz", line)
            self.assertIn("median 1.00 ms", line)


class TimecodeTest(unittest.TestCase):
    def test_decode_and_match(self):
        import cv2
        import numpy as np
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "frames").mkdir()
            saved, draws = {}, []
            frames = _frames(n=10, saved=None)
            for i in range(10):
                mid = 100 + i
                img = np.full((360, 640), 255, np.uint8)
                img[60:300, 200:440] = cv2.aruco.generateImageMarker(dictionary, mid, 240)
                img = img[::-1].copy()  # delivered bottom-up
                name = f"frames/y_{i:06d}.pgm"
                (run / name).write_bytes(b"P5\n640 360\n255\n" + img.tobytes())
                frames[i][-1] = name
                arrival = frames[i][9]
                draws.append([mid, mid, i, 0, arrival - 120_000_000, 0, 0, 0, 0, 0, 0])
            _write_csv(run / "eye_frames.csv", FRAME_HEADER, frames)
            _write_csv(run / "draws.csv", ["counter", "id", "unity_frame", "set_mono_ns", "set_boot_ns", "set_real_ns",
                                           "set_hmd_ns", "set_mono_after_ns", "set_unity_s", "eof_mono_ns", "eof_boot_ns"], draws)
            (run / "run.json").write_text(json.dumps({"probe": "timecode", "timecode_host": "test"}), encoding="utf-8")
            text = "\n".join(beampro.report(run))
            self.assertIn("decoded 10/10", text)
            self.assertIn("'flipped': 10", text)
            self.assertIn("median 120.0 ms", text)


class ReportPathTest(unittest.TestCase):
    def test_folder_without_run_json(self):
        with tempfile.TemporaryDirectory() as d:
            lines = beampro.report(Path(d) / "missing_inventory")
        self.assertEqual(len(lines), 1)
        self.assertIn("no run.json", lines[0])


if __name__ == "__main__":
    unittest.main()
