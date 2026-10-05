#!/usr/bin/env python3
"""Tests for Stage S3: IMU recording contract and the analysis maths.

The fake library fires the real ctypes callback the way the vendor thread
would, so the tests pin the ownership contract (every data handle read inside
the callback and destroyed exactly once) and cleanup on each failure path.
The analysis is tested on synthetic signals whose answers are known.
"""

from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import imu_analysis
import xreal_imu
import xreal_imu_probe
from xreal_native import NR_SUCCESS

RATE_HZ = 1000
N_SAMPLES = 50


class _Fn:
    def __init__(self, name, impl, log):
        self.name, self.impl, self.log = name, impl, log
        self.restype = self.argtypes = None

    def __call__(self, *args):
        self.log.append(self.name)
        return self.impl(*args)


class _FakeImuLib:
    """libnr_loader stand-in: S2 API + IMU, firing callbacks on NRImuStart."""

    def __init__(self, fail=(), raise_in=(), missing=()):
        self.log: list[str] = []
        self.destroyed: list[int] = []
        self._cb = None
        ok = lambda *a: NR_SUCCESS  # noqa: E731
        impls = {n: ok for n in xreal_imu.IMU_SIGNATURES}
        impls.update({
            "NRAPICreate": self._set(0xA1),
            "NRImuCreate": self._set(0xB2),
            "NRGetVersion": self._version,
            "NRImuSetCaptureCallback": self._set_cb,
            "NRImuStart": self._start,
            "NRImuDataGetHMDTimeNanos": self._ts(0),
            "NRImuDataGetHMDTimeNanosOnDevice": self._ts(7_000),
            "NRImuDataGetGyroscope": self._vec(0.001, 0.002, 0.003),
            "NRImuDataGetAccelerometer": self._vec(0.0, 0.0, 9.81),
            "NRImuDataGetMagnetometer": self._vec(1.0, 2.0, 3.0),
            "NRImuDataDestroy": self._data_destroy,
        })
        for name in fail:
            impls[name] = lambda *a: 5
        for name in raise_in:
            impls[name] = self._raise
        self._fns = {n: _Fn(n, f, self.log) for n, f in impls.items() if n not in missing}

    @staticmethod
    def _set(value):
        def impl(ptr):
            ptr.contents.value = value
            return NR_SUCCESS
        return impl

    @staticmethod
    def _version(_h, out):
        out.contents.major, out.contents.minor, out.contents.patch = 3, 1, 1
        return NR_SUCCESS

    def _set_cb(self, _imu, cb, _user):
        self._cb = cb
        return NR_SUCCESS

    def _start(self, imu):
        for i in range(N_SAMPLES):
            self._cb(imu.value if hasattr(imu, "value") else imu, 1000 + i, None)
        return NR_SUCCESS

    @staticmethod
    def _ts(offset):
        def impl(_imu, data, out):
            out.contents.value = (data - 1000) * (10**9 // RATE_HZ) + offset
            return NR_SUCCESS
        return impl

    @staticmethod
    def _vec(x, y, z):
        def impl(_imu, _data, out):
            out.contents.x, out.contents.y, out.contents.z = x, y, z
            return NR_SUCCESS
        return impl

    def _data_destroy(self, _imu, data):
        self.destroyed.append(data)
        return NR_SUCCESS

    @staticmethod
    def _raise(*_a):
        raise OSError("exception: access violation reading 0x0")

    def __getattr__(self, name):
        try:
            return self.__dict__["_fns"][name]
        except KeyError:
            raise AttributeError(name) from None


def _record(lib, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "imu.csv"
        report = xreal_imu.record(0.01, csv_path=csv_path, lib=lib, **kw)
        cols = imu_analysis.load_csv(csv_path) if csv_path.exists() else {}
    return report, cols


def _api_and_imu_calls(log):
    return [n for n in log if not n.startswith("NRImuData")]


class RecordingContractTests(unittest.TestCase):
    def test_full_sequence_order(self):
        lib = _FakeImuLib()
        report, _ = _record(lib)
        self.assertEqual(_api_and_imu_calls(lib.log), [
            "NRAPICreate", "NRAPIInitSetNetworkType", "NRAPIStart", "NRGetVersion",
            "NRImuCreate", "NRImuSetCaptureCallback", "NRImuStart",
            "NRImuStop", "NRImuDestroy", "NRAPIStop", "NRAPIDestroy",
        ])
        self.assertTrue(report.ok)
        self.assertEqual(report.samples, N_SAMPLES)

    def test_every_data_handle_destroyed_exactly_once(self):
        lib = _FakeImuLib()
        _record(lib)
        self.assertEqual(lib.destroyed, list(range(1000, 1000 + N_SAMPLES)))

    def test_samples_carry_all_fields(self):
        _, cols = _record(_FakeImuLib())
        self.assertEqual(len(cols["t_ns"]), N_SAMPLES)
        self.assertEqual(int(cols["t_dev_ns"][0] - cols["t_ns"][0]), 7_000)
        self.assertAlmostEqual(float(cols["az"][0]), 9.81, places=5)
        self.assertAlmostEqual(float(cols["gz"][0]), 0.003, places=6)

    def test_start_failure_still_cleans_up(self):
        lib = _FakeImuLib(fail=("NRImuStart",))
        report, _ = _record(lib)
        self.assertNotIn("NRImuStop", lib.log)
        self.assertEqual(lib.log[-3:], ["NRImuDestroy", "NRAPIStop", "NRAPIDestroy"])
        self.assertFalse(report.ok)
        self.assertIn("NRImuStart", report.error)

    def test_callback_registration_failure_still_cleans_up(self):
        lib = _FakeImuLib(fail=("NRImuSetCaptureCallback",))
        report, _ = _record(lib)
        self.assertNotIn("NRImuStart", lib.log)
        self.assertEqual(lib.log[-3:], ["NRImuDestroy", "NRAPIStop", "NRAPIDestroy"])
        self.assertFalse(report.ok)

    def test_imu_create_failure_skips_imu_but_stops_api(self):
        lib = _FakeImuLib(fail=("NRImuCreate",))
        report, _ = _record(lib)
        self.assertNotIn("NRImuDestroy", lib.log)
        self.assertEqual(lib.log[-2:], ["NRAPIStop", "NRAPIDestroy"])
        self.assertFalse(report.ok)

    def test_fault_inside_callback_is_contained_and_data_still_destroyed(self):
        lib = _FakeImuLib(raise_in=("NRImuDataGetGyroscope",))
        report, _ = _record(lib)
        self.assertIn("access violation", report.callback_exception)
        self.assertEqual(len(lib.destroyed), N_SAMPLES)
        self.assertFalse(report.ok)

    def test_getter_failure_is_counted(self):
        lib = _FakeImuLib(fail=("NRImuDataGetMagnetometer",))
        report, _ = _record(lib)
        self.assertEqual(report.getter_failures, N_SAMPLES)

    def test_no_samples_is_an_error(self):
        lib = _FakeImuLib()
        lib._fns["NRImuStart"].impl = lambda imu: NR_SUCCESS
        report, _ = _record(lib)
        self.assertEqual(report.samples, 0)
        self.assertIn("no samples", report.error)
        self.assertFalse(report.ok)

    def test_missing_imu_export_calls_nothing(self):
        lib = _FakeImuLib(missing=("NRImuDataDestroy",))
        report, _ = _record(lib)
        self.assertEqual(lib.log, [])
        self.assertIn("NRImuDataDestroy", report.missing_required)

    def test_hooks_run_around_recording(self):
        order = []
        lib = _FakeImuLib()
        _record(lib, on_started=lambda: order.append(("start", len(lib.log))),
                on_finished=lambda: order.append(("finish", len(lib.log))))
        self.assertEqual([o[0] for o in order], ["start", "finish"])
        self.assertLess(lib.log.index("NRImuStart"), order[0][1])

    def test_report_serialises(self):
        import json
        report, _ = _record(_FakeImuLib())
        self.assertTrue(json.loads(json.dumps(report.to_dict()))["ok"])


def _synthetic(n, rate=RATE_HZ, gyro=(0.0, 0.0, 0.0), acc=(0.0, 0.0, 9.80665),
               noise=0.0, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    idx = np.array([i for i in range(n) if i not in set(drop)])
    t = idx * (10**9 // rate)
    cols = {"t_ns": t.astype(np.int64), "t_dev_ns": (t + 123).astype(np.int64),
            "host_ns": (t + 5_000_000).astype(np.int64)}
    for i, a in enumerate("xyz"):
        cols["g" + a] = np.full(len(t), gyro[i]) + noise * rng.standard_normal(len(t))
        cols["a" + a] = np.full(len(t), acc[i]) + noise * rng.standard_normal(len(t))
        cols["m" + a] = np.zeros(len(t))
    return cols


class AnalysisTests(unittest.TestCase):
    def test_rate_and_clean_timing(self):
        t = imu_analysis.timing(_synthetic(1001))
        self.assertAlmostEqual(t["rate_hz"], 1000.0, delta=0.5)
        self.assertEqual((t["gaps"], t["non_monotonic"]), (0, 0))

    def test_gaps_and_missing_samples_counted(self):
        t = imu_analysis.timing(_synthetic(1000, drop=(100, 500, 501, 502)))
        self.assertEqual(t["gaps"], 2)
        self.assertEqual(t["missing_estimate"], 4)

    def test_delivery_jitter_zero_for_constant_latency(self):
        d = imu_analysis.delivery_jitter(_synthetic(500))
        self.assertAlmostEqual(d["jitter_std_ms"], 0.0, places=6)

    def test_rest_bias_units_and_gravity_axis(self):
        r = imu_analysis.rest(_synthetic(2000, gyro=(0.01, -0.02, 0.0),
                                         acc=(0.0, -9.80665, 0.0), noise=1e-3))
        self.assertAlmostEqual(r["gyro_bias"][0], 0.01, places=3)
        self.assertEqual(r["accel_unit"], "m/s^2")
        self.assertEqual(r["gravity_reaction_axis"], "-y")

    def test_accel_unit_in_g_and_unknown(self):
        self.assertEqual(imu_analysis.infer_accel_unit(1.01), "g")
        self.assertIsNone(imu_analysis.infer_accel_unit(4.0))

    def _turn_cols(self, rate_value, axis=2):
        n = 8000                             # 8 s at 1 kHz: 3 s still, 2 s turning, 3 s still
        cols = _synthetic(n, gyro=(0.002, 0.002, 0.002))
        w = np.zeros(n)
        w[3000:5000] = rate_value / 2.0      # integral = rate_value
        cols["g" + "xyz"[axis]] += w
        return cols

    def test_turn_in_rad_per_s(self):
        tr = imu_analysis.turn(self._turn_cols(math.pi / 2), 90.0)
        self.assertEqual(tr["dominant_axis"], "+z")
        self.assertEqual(tr["gyro_unit"], "rad/s")
        self.assertGreater(tr["dominant_share"], 0.99)

    def test_turn_in_deg_per_s_negative_axis(self):
        tr = imu_analysis.turn(self._turn_cols(-90.0, axis=0), 90.0)
        self.assertEqual(tr["dominant_axis"], "-x")
        self.assertEqual(tr["gyro_unit"], "deg/s")

    def test_turn_needs_a_still_lead(self):
        self.assertIn("error", imu_analysis.turn(_synthetic(5), 90.0))


class ProbeCliTests(unittest.TestCase):
    def test_analyse_existing_csv_without_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "imu.csv"
            cols = _synthetic(3000, acc=(0.0, 0.0, 9.80665))
            rows = zip(*(cols[c] for c in xreal_imu.CSV_COLUMNS))
            xreal_imu.write_csv(path, list(rows))
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                code = xreal_imu_probe.main(["--analyse", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("m/s^2", sink.getvalue())
        self.assertIn("+z", sink.getvalue())

    def test_failed_recording_exit_code(self):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), \
                patch.object(xreal_imu_probe, "run_isolated",
                             return_value={"ok": False, "fatal": "boom"}), \
                patch.object(xreal_imu_probe, "append_jsonl"):
            self.assertEqual(xreal_imu_probe.main(["--turn", "yaw"]), 1)
        self.assertIn("boom", sink.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
