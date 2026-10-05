#!/usr/bin/env python3
"""Tests for Stage S3b: the licence-gate discriminator and head-pose recording.

The fake library stands in for ``libnr_loader`` and lets each test choose what
every export returns, so the tests pin the things that matter when the real
run happens once, on hardware, in a lab: that a pose handle is destroyed
exactly once per iteration, that cleanup runs on every failure path, that the
gate verdict is a function of the three codes and nothing else, and that the
pose-layout maths says the right thing about a matrix whose convention we
already know.
"""

from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np

import headpose_analysis
import xreal_headpose
import xreal_result_codes
from xreal_headpose import classify_gate
from xreal_native import NR_SUCCESS

UNSUPPORTED = xreal_result_codes.LICENSE_FEATURE_UNSUPPORTED
SESSION = 0xA1
GROUP = 0xE1
PERCEPTION = 0x1000
TRACKING = 0xB2
POSE = 0xC3

#: A known-good row-major transform: identity rotation, translation (1, 2, 3).
#: Deliberately *not* the identity matrix - it stands in for a pose that carries
#: something.
ROW_MAJOR_IDENTITY = [1, 0, 0, 1,
                      0, 1, 0, 2,
                      0, 0, 1, 3,
                      0, 0, 0, 1]

#: What run 2 actually got from the glasses, 1925 times: a pose carrying nothing.
TRUE_IDENTITY = [1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1]


class _Fn:
    def __init__(self, name, impl, log):
        self.name, self.impl, self.log = name, impl, log
        self.restype = self.argtypes = None

    def __call__(self, *args):
        self.log.append(self.name)
        return self.impl(*args)


class _FakeLib:
    """libnr_loader stand-in covering both S3b signature tables."""

    def __init__(self, results=None, pose_floats=None, perception_ids=(10, 11),
                 start_fails_for=()):
        self.log: list[str] = []
        self.destroyed_poses: list[int] = []
        self.results = dict(results or {})
        self.pose_floats = list(pose_floats or ROW_MAJOR_IDENTITY)
        self.perception_ids = list(perception_ids)
        self.start_fails_for = set(start_fails_for)
        self._clock = 0
        self.created_perceptions: list[int] = []
        self.started_perceptions: list[int] = []
        self._t = 0
        names = set(xreal_headpose.HEADPOSE_SIGNATURES) | set(xreal_headpose.GATE_SIGNATURES)
        self._impls = {n: (lambda *a: NR_SUCCESS) for n in names}
        self._impls.update({
            "NRAPICreate": self._out(SESSION),
            "NRGetVersion": self._version,
            "NRHeadTrackingCreate": self._tracking_create,
            "NRHeadTrackingAcquireHeadPose": self._acquire,
            "NRHeadPoseGetPose": self._get_pose,
            "NRHeadPoseGetHMDTimeNanos": self._get_time,
            "NRHeadPoseGetTrackingReason": self._get_reason,
            "NRHeadPoseDestroy": self._pose_destroy,
            "NRPerceptionGroupCreate": self._out(GROUP, "NRPerceptionGroupCreate"),
            "NRPerceptionGroupGetCount": self._group_count,
            "NRPerceptionGroupGetPerceptionId": self._perception_id,
            "NRPerceptionGroupGetSupportedFeatures": self._features,
            "NRPerceptionCreate": self._perception_create,
            "NRPerceptionGetHMDTimeNanos": self._perception_clock,
            "NRPerceptionGetSupportedFeatures": self._perception_features,
            "NRPerceptionStart": self._perception_start,
            "NRImuCreate": self._out(0xD1, "NRImuCreate"),
            "NRGrayscaleCameraCreate": self._out(0xD2, "NRGrayscaleCameraCreate"),
            "NRHMDCreate": self._out(0xD3, "NRHMDCreate"),
        })

    # -- helpers ---------------------------------------------------------
    def _code(self, name):
        return self.results.get(name, NR_SUCCESS)

    def _out(self, value, name=None):
        def impl(ptr, *_):
            code = self._code(name) if name else NR_SUCCESS
            if code == NR_SUCCESS:
                ptr.contents.value = value
            return code
        return impl

    def _version(self, _h, ptr):
        ptr.contents.major, ptr.contents.minor, ptr.contents.patch = 3, 1, 1
        return NR_SUCCESS

    def _group_count(self, _group, ptr):
        code = self._code("NRPerceptionGroupGetCount")
        if code == NR_SUCCESS:
            ptr.contents.value = len(self.perception_ids)
        return code

    def _perception_id(self, _group, index, ptr):
        ptr.contents.value = self.perception_ids[int(getattr(index, "value", index))]
        return self._code("NRPerceptionGroupGetPerceptionId")

    def _features(self, _group, index, ptr):
        ptr.contents.value = 0b1011 << int(getattr(index, "value", index))
        return self._code("NRPerceptionGroupGetSupportedFeatures")

    def _perception_create(self, pid, ptr):
        pid = int(getattr(pid, "value", pid))
        code = self._code("NRPerceptionCreate")
        if code != NR_SUCCESS:
            return code
        self.created_perceptions.append(pid)
        ptr.contents.value = PERCEPTION + pid
        return NR_SUCCESS

    def _perception_clock(self, _perception, ptr):
        code = self._code("NRPerceptionGetHMDTimeNanos")
        if code != NR_SUCCESS:
            return code
        self._clock += 5_000_000          # 5 ms in ns
        ptr.contents.value = self._clock
        return NR_SUCCESS

    def _perception_features(self, _perception, ptr):
        ptr.contents.value = 0b110
        return self._code("NRPerceptionGetSupportedFeatures")

    def _perception_start(self, handle):
        handle = int(getattr(handle, "value", handle))
        pid = handle - PERCEPTION
        if pid in self.start_fails_for:
            return 1
        self.started_perceptions.append(pid)
        return self._code("NRPerceptionStart")

    def _tracking_create(self, _session, ptr):
        code = self._code("NRHeadTrackingCreate")
        if code == NR_SUCCESS:
            ptr.contents.value = TRACKING
        return code

    def _acquire(self, _session, _tracking, _time, ptr):
        code = self._code("NRHeadTrackingAcquireHeadPose")
        if code == NR_SUCCESS:
            ptr.contents.value = POSE
        return code

    def _get_pose(self, _session, _pose, ptr):
        for i, v in enumerate(self.pose_floats):
            ptr.contents[i] = float(v)
        return self._code("NRHeadPoseGetPose")

    def _get_time(self, _session, _pose, ptr):
        self._t += 10_000_000          # 10 ms in ns
        ptr.contents.value = self._t
        return NR_SUCCESS

    def _get_reason(self, _session, _pose, ptr):
        ptr.contents.value = 1
        return NR_SUCCESS

    def _pose_destroy(self, _session, pose):
        # bind() only sets argtypes as a plain attribute on the fake, so the
        # ctypes object arrives unconverted; take its value the way the real
        # library would see it.
        self.destroyed_poses.append(int(getattr(pose, "value", pose)))
        return NR_SUCCESS

    def __getattr__(self, name):
        impls = self.__dict__.get("_impls", {})
        if name not in impls:
            raise AttributeError(name)
        return _Fn(name, impls[name], self.__dict__["log"])


class GateVerdictTests(unittest.TestCase):
    """The verdict must depend on all three codes, not just the IMU one."""

    def test_empty_feature_list(self):
        v = classify_gate({"imu": UNSUPPORTED, "grayscale_camera": UNSUPPORTED,
                           "hmd": NR_SUCCESS})
        self.assertIn("empty", v)

    def test_per_feature_licence(self):
        v = classify_gate({"imu": UNSUPPORTED, "grayscale_camera": NR_SUCCESS,
                           "hmd": NR_SUCCESS})
        self.assertIn("per-feature", v)

    def test_imu_licensed(self):
        v = classify_gate({"imu": NR_SUCCESS, "grayscale_camera": NR_SUCCESS,
                           "hmd": NR_SUCCESS})
        self.assertIn("licensed", v)

    def test_control_failure_is_inconclusive(self):
        v = classify_gate({"imu": UNSUPPORTED, "grayscale_camera": UNSUPPORTED,
                           "hmd": 2})
        self.assertIn("inconclusive", v)
        self.assertIn("InvalidArgument", v)


class GateProbeTests(unittest.TestCase):
    def test_reproduces_the_2026_09_23_failure(self):
        lib = _FakeLib(results={"NRImuCreate": UNSUPPORTED,
                                "NRGrayscaleCameraCreate": UNSUPPORTED})
        report = xreal_headpose.probe_gate(lib=lib)
        self.assertEqual(report.codes["imu"], UNSUPPORTED)
        self.assertEqual(report.codes["hmd"], NR_SUCCESS)
        self.assertIn("empty", report.verdict)
        self.assertTrue(report.ok)
        # only the control succeeded, so only the control is destroyed
        self.assertIn("NRHMDDestroy", lib.log)
        self.assertNotIn("NRImuDestroy", lib.log)
        # the session is always closed
        self.assertEqual(lib.log[-2:], ["NRAPIStop", "NRAPIDestroy"])

    def test_per_feature_case_destroys_what_it_created(self):
        lib = _FakeLib(results={"NRImuCreate": UNSUPPORTED})
        report = xreal_headpose.probe_gate(lib=lib)
        self.assertIn("per-feature", report.verdict)
        self.assertIn("NRGrayscaleCameraDestroy", lib.log)

    def test_named_codes_in_payload(self):
        lib = _FakeLib(results={"NRImuCreate": UNSUPPORTED,
                                "NRGrayscaleCameraCreate": UNSUPPORTED})
        payload = xreal_headpose.probe_gate(lib=lib).to_dict()
        self.assertEqual(payload["named_codes"]["imu"],
                         "22 (LicenseFeatureUnsupported)")


class HeadPoseRecordingTests(unittest.TestCase):
    def test_single_iteration_is_fully_recorded(self):
        lib = _FakeLib()
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertTrue(report.ok)
        self.assertEqual(report.samples, 1)
        self.assertEqual(lib.destroyed_poses, [POSE])
        names = [c.name for c in report.calls]
        for expected in ("NRHeadTrackingCreate", "NRHeadTrackingAcquireHeadPose",
                         "NRHeadPoseGetPose", "NRHeadPoseGetTrackingReason",
                         "NRHeadPoseDestroy", "NRHeadTrackingDestroy"):
            self.assertIn(expected, names)

    def test_loop_destroys_one_pose_per_sample(self):
        lib = _FakeLib()
        report = xreal_headpose.record(duration_s=0.05, rate_hz=1000, lib=lib)
        self.assertGreater(report.samples, 1)
        self.assertEqual(len(lib.destroyed_poses), report.samples)
        self.assertEqual(report.destroy_failures, 0)

    def test_loop_calls_are_not_all_recorded(self):
        """A 100 Hz loop must not write one call record per iteration."""
        lib = _FakeLib()
        report = xreal_headpose.record(duration_s=0.05, rate_hz=1000, lib=lib)
        self.assertLess(len(report.calls), report.samples + 12)

    def test_acquire_failure_still_destroys_tracking(self):
        lib = _FakeLib(results={"NRHeadTrackingAcquireHeadPose": 10})
        report = xreal_headpose.record(duration_s=0.05, lib=lib)
        self.assertFalse(report.ok)
        self.assertEqual(report.acquire_code, 10)
        self.assertIn("candidate", report.error)
        self.assertIn("NRHeadTrackingDestroy", lib.log)
        self.assertEqual(lib.log[-2:], ["NRAPIStop", "NRAPIDestroy"])
        self.assertEqual(report.to_dict()["acquire_result"], "10 (TrackingNotRunning)")

    def test_every_time_candidate_is_tried(self):
        lib = _FakeLib(results={"NRHeadTrackingAcquireHeadPose": 2})
        report = xreal_headpose.record(duration_s=0.0, lib=lib, clock="zero",
                                       time_candidates=(0, 1, 2))
        self.assertEqual(lib.log.count("NRHeadTrackingAcquireHeadPose"), 3)
        self.assertEqual(report.acquire_failures, 3)

    def test_tracking_create_failure_stops_early(self):
        lib = _FakeLib(results={"NRHeadTrackingCreate": 2})
        report = xreal_headpose.record(duration_s=0.05, lib=lib)
        self.assertEqual(report.error, "NRHeadTrackingCreate failed")
        self.assertNotIn("NRHeadTrackingAcquireHeadPose", lib.log)
        self.assertEqual(lib.log[-2:], ["NRAPIStop", "NRAPIDestroy"])

    def test_csv_round_trip(self):
        lib = _FakeLib()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "headpose.csv"
            report = xreal_headpose.record(duration_s=0.02, rate_hz=1000,
                                           csv_path=path, lib=lib)
            self.assertEqual(report.csv_path, str(path))
            cols = headpose_analysis.load_csv(path)
            self.assertEqual(len(cols["m0"]), report.samples)
            self.assertEqual(cols["m3"][0], 1.0)


class PoseLayoutTests(unittest.TestCase):
    def test_row_major_identity_is_recognised(self):
        mats = np.array([ROW_MAJOR_IDENTITY] * 5, dtype=float)
        out = headpose_analysis.interpret(mats)
        self.assertIn("row-major", out["verdict"])

    def test_column_major_is_recognised(self):
        col = np.array(ROW_MAJOR_IDENTITY, dtype=float).reshape(4, 4).T.reshape(-1)
        out = headpose_analysis.interpret(np.array([col] * 5))
        self.assertIn("column-major", out["verdict"])

    def test_real_rotation_block_passes_orthonormality(self):
        c, s = np.cos(0.4), np.sin(0.4)
        m = [c, -s, 0, 0.1,  s, c, 0, 0.2,  0, 0, 1, 0.3,  0, 0, 0, 1]
        out = headpose_analysis.interpret(np.array([m] * 3))
        self.assertIn("row-major", out["verdict"])

    def test_nonsense_payload_is_not_forced_into_a_layout(self):
        rng = np.random.default_rng(0)
        out = headpose_analysis.interpret(rng.normal(size=(5, 16)))
        self.assertIn("no candidate layout fits", out["verdict"])

    def test_quaternion_payload(self):
        q = [0.0, 0.0, 0.0, 1.0] + [0.0] * 12
        out = headpose_analysis.interpret(np.array([q] * 4))
        self.assertIn("quaternion", out["verdict"])

    def test_static_pose_is_flagged(self):
        cols = {"host_ns": [i * 1e7 for i in range(5)],
                "hmd_ns": [i * 1e7 for i in range(5)],
                "tracking_reason": [1] * 5}
        for i in range(16):
            cols[f"m{i}"] = [ROW_MAJOR_IDENTITY[i]] * 5
        out = headpose_analysis.analyse(cols)
        self.assertTrue(out["pose_is_static"])
        self.assertEqual(out["timing"]["hmd"]["rate_hz"], 100.0)
        self.assertEqual(out["tracking_reasons"], {"1": 5})


class PerceptionLayerTests(unittest.TestCase):
    """The layer that produces the handle NRHeadTracking actually wants."""

    def test_enumeration_lists_every_perception(self):
        lib = _FakeLib(perception_ids=(10, 11, 12))
        report = xreal_headpose.probe_perception(lib=lib)
        self.assertTrue(report.ok)
        self.assertEqual([p["perception_id"] for p in report.perceptions],
                         [10, 11, 12])
        self.assertEqual(report.perceptions[0]["features_bits"], [0, 1, 3])
        self.assertIn("NRPerceptionGroupDestroy", lib.log)
        self.assertNotIn("NRPerceptionCreate", lib.log)

    def test_group_create_failure_is_named(self):
        lib = _FakeLib(results={"NRPerceptionGroupCreate": 1})
        report = xreal_headpose.probe_perception(lib=lib)
        self.assertFalse(report.ok)
        self.assertEqual(report.error, "NRPerceptionGroupCreate failed")
        self.assertEqual(lib.log[-2:], ["NRAPIStop", "NRAPIDestroy"])

    def test_pose_run_brings_up_the_perception_first(self):
        lib = _FakeLib()
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertTrue(report.ok)
        self.assertEqual(report.perception_id, 10)
        order = [n for n in lib.log if n.startswith(("NRPerception", "NRHeadTracking"))]
        self.assertEqual(order[:4], ["NRPerceptionGroupCreate",
                                     "NRPerceptionGroupGetCount",
                                     "NRPerceptionGroupGetPerceptionId",
                                     "NRPerceptionGroupGetSupportedFeatures"])
        self.assertLess(order.index("NRPerceptionStart"),
                        order.index("NRHeadTrackingCreate"))

    def test_teardown_is_the_reverse_of_setup(self):
        lib = _FakeLib()
        xreal_headpose.record(duration_s=0.0, lib=lib)
        tail = [n for n in lib.log if n in ("NRHeadTrackingDestroy",
                                            "NRPerceptionStop",
                                            "NRPerceptionDestroy",
                                            "NRPerceptionGroupDestroy",
                                            "NRAPIStop", "NRAPIDestroy")]
        self.assertEqual(tail, ["NRHeadTrackingDestroy", "NRPerceptionStop",
                                "NRPerceptionDestroy", "NRPerceptionGroupDestroy",
                                "NRAPIStop", "NRAPIDestroy"])

    def test_falls_through_to_a_perception_that_starts(self):
        lib = _FakeLib(perception_ids=(10, 11), start_fails_for=(10,))
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertTrue(report.ok)
        self.assertEqual(report.perception_id, 11)
        self.assertEqual(lib.created_perceptions, [10, 11])
        self.assertEqual(lib.started_perceptions, [11])

    def test_no_perception_starts(self):
        lib = _FakeLib(perception_ids=(10, 11), start_fails_for=(10, 11))
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertFalse(report.ok)
        self.assertIn("no perception", report.error)
        self.assertNotIn("NRHeadTrackingCreate", lib.log)

    def test_perception_id_can_be_pinned(self):
        lib = _FakeLib(perception_ids=(10, 11))
        report = xreal_headpose.record(duration_s=0.0, lib=lib, perception_id=11)
        self.assertEqual(report.perception_id, 11)
        self.assertEqual(lib.created_perceptions, [11])

    def test_the_2026_09_24_failure_is_reported_with_its_name(self):
        """NRHeadTrackingCreate -> 2 must surface as InvalidArgument."""
        lib = _FakeLib(results={"NRHeadTrackingCreate": 2})
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertEqual(report.error, "NRHeadTrackingCreate failed")
        bad = [c for c in report.calls if c.name == "NRHeadTrackingCreate"][0]
        self.assertEqual(xreal_result_codes.describe(bad.result),
                         "2 (InvalidArgument)")


class DeviceClockTests(unittest.TestCase):
    """Run 2 passed hmd_time_nanos = 0 and got identity poses; this is the fix."""

    def test_default_clock_is_the_perception_clock(self):
        lib = _FakeLib()
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertEqual(report.clock, "hmd")
        self.assertIn("NRPerceptionGetHMDTimeNanos", lib.log)
        self.assertIsNotNone(report.time_argument)
        self.assertGreater(report.time_argument, 0)

    def test_offset_is_added_to_the_device_clock(self):
        """Same fake clock sequence both times, so the delta is the offset."""
        base = xreal_headpose.record(duration_s=0.0, lib=_FakeLib()).time_argument
        offset = xreal_headpose.record(duration_s=0.0, lib=_FakeLib(),
                                       offset_ns=1_000_000).time_argument
        self.assertEqual(offset - base, 1_000_000)

    def test_clock_failure_is_named_not_silent(self):
        lib = _FakeLib(results={"NRPerceptionGetHMDTimeNanos": 1})
        report = xreal_headpose.record(duration_s=0.0, lib=lib)
        self.assertFalse(report.ok)
        self.assertEqual(report.error, "NRPerceptionGetHMDTimeNanos failed")

    def test_zero_clock_still_available_for_comparison(self):
        lib = _FakeLib()
        report = xreal_headpose.record(duration_s=0.0, lib=lib, clock="zero",
                                       time_candidates=(0,))
        self.assertEqual(report.time_argument, 0)

    def test_reason_names_are_in_the_payload(self):
        lib = _FakeLib()
        payload = xreal_headpose.record(duration_s=0.05, rate_hz=1000,
                                        lib=lib).to_dict()
        self.assertEqual(list(payload["tracking_reason_names"]),
                         ["1 (Initializing)"])


class VerdictTests(unittest.TestCase):
    """A run that records identity poses must not report success."""

    def test_identity_only_recording_is_not_ok(self):
        lib = _FakeLib(pose_floats=TRUE_IDENTITY)
        report = xreal_headpose.record(duration_s=0.05, rate_hz=1000, lib=lib)
        self.assertGreater(report.samples, 1)
        self.assertEqual(report.identity_samples, report.samples)
        self.assertFalse(report.delivered_data)
        self.assertFalse(report.ok)
        self.assertIn("identity transform", report.error)

    def test_a_moving_recording_is_ok(self):
        lib = _FakeLib(pose_floats=ROW_MAJOR_IDENTITY)
        report = xreal_headpose.record(duration_s=0.05, rate_hz=1000, lib=lib)
        self.assertEqual(report.identity_samples, 0)
        self.assertTrue(report.delivered_data)
        self.assertTrue(report.ok)
        self.assertTrue(report.to_dict()["delivered_data"])


class IdentityPoseTests(unittest.TestCase):
    def test_identity_is_recognised(self):
        ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.assertTrue(xreal_headpose.is_identity(ident))
        moved = list(ident)
        moved[3] = 0.25
        self.assertFalse(xreal_headpose.is_identity(moved))
        self.assertFalse(xreal_headpose.is_identity([1.0] * 4))


class SweepTests(unittest.TestCase):
    def test_every_perception_is_visited_and_compared(self):
        lib = _FakeLib(perception_ids=(1, 2, 3), pose_floats=TRUE_IDENTITY)
        report = xreal_headpose.sweep_perceptions(lib=lib)
        self.assertTrue(report.ok)
        self.assertEqual([r["perception_id"] for r in report.sweep], [1, 2, 3])
        for row in report.sweep:
            self.assertTrue(row["started"])
            self.assertEqual(row["reason"], "1 (Initializing)")
            self.assertTrue(row["identity_pose"])
            self.assertEqual(row["features_mask"], 0b110)

    def test_a_perception_that_will_not_start_is_recorded_not_skipped(self):
        lib = _FakeLib(perception_ids=(1, 2), start_fails_for=(1,))
        report = xreal_headpose.sweep_perceptions(lib=lib)
        self.assertEqual([r["started"] for r in report.sweep], [False, True])

    def test_a_moving_pose_is_distinguished_from_identity(self):
        lib = _FakeLib(perception_ids=(1,), pose_floats=ROW_MAJOR_IDENTITY)
        report = xreal_headpose.sweep_perceptions(lib=lib)
        self.assertFalse(report.sweep[0]["identity_pose"])

    def test_sweep_tears_down_each_perception(self):
        lib = _FakeLib(perception_ids=(1, 2))
        xreal_headpose.sweep_perceptions(lib=lib)
        self.assertEqual(lib.log.count("NRPerceptionStop"), 2)
        self.assertEqual(lib.log.count("NRPerceptionDestroy"), 2)
        self.assertEqual(lib.log.count("NRHeadTrackingDestroy"), 2)
        self.assertEqual(lib.log[-3:], ["NRPerceptionGroupDestroy", "NRAPIStop",
                                        "NRAPIDestroy"])


class ResultCodeTests(unittest.TestCase):
    def test_the_code_s3_actually_got(self):
        self.assertEqual(xreal_result_codes.describe(22),
                         "22 (LicenseFeatureUnsupported)")
        self.assertTrue(xreal_result_codes.is_licence_failure(22))

    def test_unrecovered_codes_are_not_invented(self):
        self.assertEqual(xreal_result_codes.describe(999), "999 (unnamed)")
        self.assertIsNone(xreal_result_codes.name(999))
        self.assertEqual(xreal_result_codes.describe(None), "not called")

    def test_the_block_is_sequential(self):
        names = xreal_result_codes.RESULT_NAMES
        self.assertEqual(sorted(names), list(range(xreal_result_codes.VERIFIED_MAX + 1)))


if __name__ == "__main__":
    unittest.main()
