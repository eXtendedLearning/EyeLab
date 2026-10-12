import unittest

import cv2
import numpy as np

from pose_estimator import LStructureDetector, PoseKalmanFilter, PoseResult
from pose_lock import (
    LOCK_COASTING,
    LOCK_LOCKED,
    LOCK_SEARCHING,
    POSE_LOCK_PRESETS,
    MarkerCarryover,
    PoseLock,
    PoseLockConfig,
    RoiRedetector,
    align_rvec,
    rotation_angle_between,
)


def _pose(n_markers, tvec=(0.0, 0.0, 0.3), rvec=(0.0, 0.0, 0.0), rms=0.5,
          ambiguity_ratio=None, ambiguity_separation_deg=None):
    return PoseResult(
        rvec=np.array(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.array(tvec, dtype=np.float64).reshape(3, 1),
        marker_ids=list(range(n_markers)),
        marker_count=n_markers,
        timestamp=0.0,
        rms_reproj_px=rms,
        ambiguity_ratio=ambiguity_ratio,
        ambiguity_separation_deg=ambiguity_separation_deg,
    )


class AlignRvecTests(unittest.TestCase):
    def test_flipped_representation_is_aligned(self):
        r = np.array([0.1, 0.2, 3.0]).reshape(3, 1)
        theta = np.linalg.norm(r)
        flipped = r * ((theta - 2 * np.pi) / theta)
        aligned = align_rvec(flipped, r)
        self.assertLess(rotation_angle_between(aligned, r), 1e-4)
        np.testing.assert_allclose(aligned, r, atol=1e-6)

    def test_near_zero_rotation_passthrough(self):
        r = np.zeros((3, 1))
        np.testing.assert_allclose(align_rvec(r, np.ones((3, 1))), r)


class PoseLockStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.lock = PoseLock(
            PoseKalmanFilter(),
            PoseLockConfig(coast_duration_s=0.3),
        )

    def _acquire(self, t=0.0):
        out = self.lock.process(_pose(3), now=t)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)
        return out

    def test_two_markers_rejected_while_searching(self):
        self.assertIsNone(self.lock.process(_pose(2), now=0.0))
        self.assertEqual(self.lock.state, LOCK_SEARCHING)

    def test_acquire_requires_three_then_two_sustains(self):
        self._acquire(t=0.0)
        out = self.lock.process(_pose(2), now=0.033)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)

    def test_two_marker_pose_with_translation_jump_is_rejected(self):
        self._acquire(t=0.0)
        jumped = _pose(2, tvec=(0.5, 0.0, 0.3))  # 0.5 m jump
        out = self.lock.process(jumped, now=0.033)
        # Rejected as measurement -> lock coasts instead of teleporting
        self.assertEqual(self.lock.state, LOCK_COASTING)
        self.assertTrue(out.coasted)
        self.assertEqual(self.lock.last_reject_reason, "jump")

    def test_full_count_pose_with_large_stable_rms_still_acquires(self):
        # Real board geometry can have a large stable baseline RMS; the
        # acquisition ceiling is generous enough not to block it.
        out = self.lock.process(_pose(3, rms=8.0), now=0.0)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)

    def test_full_count_pose_above_rms_ceiling_is_not_acquired(self):
        out = self.lock.process(_pose(3, rms=50.0), now=0.0)
        self.assertIsNone(out)
        self.assertEqual(self.lock.state, LOCK_SEARCHING)
        self.assertEqual(self.lock.last_reject_reason, "rms")

    def test_sustain_pose_with_high_rms_is_rejected(self):
        self._acquire(t=0.0)
        bad = _pose(2, rms=50.0)
        out = self.lock.process(bad, now=0.033)
        self.assertEqual(self.lock.state, LOCK_COASTING)
        self.assertTrue(out.coasted)
        self.assertEqual(self.lock.last_reject_reason, "rms")

    def test_sustain_rms_gate_adapts_to_baseline(self):
        # Acquire with a consistently high baseline RMS (e.g. 8 px)
        for i in range(5):
            self.lock.process(_pose(3, rms=8.0), now=0.033 * i)
        self.assertEqual(self.lock.state, LOCK_LOCKED)
        # 2-marker pose at ~baseline should pass (gate = 2.5x baseline)
        out = self.lock.process(_pose(2, rms=9.0), now=0.2)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)

    def test_coasting_bridges_short_dropout(self):
        self._acquire(t=0.0)
        out = self.lock.process(None, now=0.1)
        self.assertIsNotNone(out)
        self.assertTrue(out.coasted)
        self.assertEqual(self.lock.state, LOCK_COASTING)
        # Reacquire from coasting with a sustain-count pose
        out = self.lock.process(_pose(2), now=0.2)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)

    def test_coasting_expires_to_searching(self):
        self._acquire(t=0.0)
        out = self.lock.process(None, now=1.0)
        self.assertIsNone(out)
        self.assertEqual(self.lock.state, LOCK_SEARCHING)
        # After expiry, one 2-marker frame is not enough: it must be confirmed
        self.assertIsNone(self.lock.process(_pose(2), now=1.033))
        self.assertEqual(self.lock.last_reject_reason, "confirming")
        self.assertEqual(self.lock.state, LOCK_SEARCHING)

    def test_min_markers_property_tracks_state(self):
        self.assertEqual(self.lock.min_markers, 2)
        self._acquire(t=0.0)
        self.assertEqual(self.lock.min_markers, 1)

    def test_single_marker_sustains_lock(self):
        self._acquire(t=0.0)
        out = self.lock.process(_pose(1), now=0.033)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)

    def test_single_marker_rejected_while_searching(self):
        self.assertIsNone(self.lock.process(_pose(1), now=0.0))
        self.assertEqual(self.lock.state, LOCK_SEARCHING)

    def test_single_marker_pose_with_jump_is_rejected(self):
        self._acquire(t=0.0)
        jumped = _pose(1, tvec=(0.5, 0.0, 0.3))  # 0.5 m jump
        out = self.lock.process(jumped, now=0.033)
        self.assertEqual(self.lock.state, LOCK_COASTING)
        self.assertTrue(out.coasted)
        self.assertEqual(self.lock.last_reject_reason, "jump")


class AcquisitionConfirmationTests(unittest.TestCase):
    """Locks started from fewer than the trusted marker count (default 3)."""

    def _lock(self, **cfg):
        return PoseLock(PoseKalmanFilter(), PoseLockConfig(**cfg))

    def _feed(self, lock, poses, t0=0.0):
        outs = []
        for i, p in enumerate(poses):
            outs.append(lock.process(p, now=t0 + 0.033 * i))
        return outs

    def test_default_thresholds(self):
        cfg = PoseLockConfig()
        self.assertEqual(
            (cfg.sustain_min_markers, cfg.acquire_min_markers, cfg.trusted_min_markers),
            (1, 2, 3),
        )

    def test_two_markers_lock_after_confirmation_frames(self):
        lock = self._lock()
        n = lock.config.acquire_confirm_frames
        outs = self._feed(lock, [_pose(2) for _ in range(n)])
        self.assertTrue(all(o is None for o in outs[:-1]))
        self.assertIsNotNone(outs[-1])
        self.assertEqual(lock.state, LOCK_LOCKED)

    def test_three_markers_lock_at_once(self):
        lock = self._lock()
        self.assertIsNotNone(lock.process(_pose(3), now=0.0))
        self.assertEqual(lock.state, LOCK_LOCKED)

    def test_inconsistent_candidate_restarts_confirmation(self):
        lock = self._lock(acquire_confirm_frames=3)
        poses = [
            _pose(2),
            _pose(2, rvec=(0.5, 0.0, 0.0)),  # ~29 deg off: a mirror-pose flip
            _pose(2, rvec=(0.5, 0.0, 0.0)),
        ]
        outs = self._feed(lock, poses)
        self.assertTrue(all(o is None for o in outs))
        self.assertEqual(lock.state, LOCK_SEARCHING)
        self.assertIsNotNone(lock.process(_pose(2, rvec=(0.5, 0.0, 0.0)), now=0.2))
        self.assertEqual(lock.state, LOCK_LOCKED)

    def test_dropout_restarts_confirmation(self):
        lock = self._lock(acquire_confirm_frames=3)
        self._feed(lock, [_pose(2), _pose(2), None])
        self.assertIsNone(lock.process(_pose(2), now=0.2))
        self.assertEqual(lock.state, LOCK_SEARCHING)

    def test_ambiguous_candidate_cannot_start_lock(self):
        lock = self._lock(acquire_confirm_frames=1)
        out = lock.process(
            _pose(2, ambiguity_ratio=1.2, ambiguity_separation_deg=40.0), now=0.0,
        )
        self.assertIsNone(out)
        self.assertEqual(lock.last_reject_reason, "ambiguous")

    def test_low_ratio_with_close_solutions_is_not_ambiguous(self):
        # Face-on views: both mirror poses are close, so the choice is cheap.
        lock = self._lock(acquire_confirm_frames=1)
        out = lock.process(
            _pose(2, ambiguity_ratio=1.2, ambiguity_separation_deg=4.0), now=0.0,
        )
        self.assertIsNotNone(out)

    def test_ambiguity_blocks_acquisition_at_trusted_count_too(self):
        # Marker count is evidence, not a bypass: a planar solve whose mirror
        # pose is equally good and far away cannot start a lock.
        lock = self._lock()
        out = lock.process(
            _pose(3, ambiguity_ratio=1.0, ambiguity_separation_deg=40.0), now=0.0,
        )
        self.assertIsNone(out)
        self.assertEqual(lock.last_reject_reason, "ambiguous")

    def test_resolved_ambiguity_does_not_block_trusted_count(self):
        lock = self._lock()
        out = lock.process(
            _pose(3, ambiguity_ratio=6.0, ambiguity_separation_deg=40.0), now=0.0,
        )
        self.assertIsNotNone(out)

    def test_two_marker_pose_is_gated_once_locked(self):
        lock = self._lock()
        lock.process(_pose(3), now=0.0)
        out = lock.process(_pose(2, tvec=(0.5, 0.0, 0.3)), now=0.033)
        self.assertTrue(out.coasted)
        self.assertEqual(lock.last_reject_reason, "jump")

    def test_single_preset_locks_from_one_marker(self):
        lock = PoseLock(PoseKalmanFilter(), POSE_LOCK_PRESETS["single"])
        n = lock.config.acquire_confirm_frames
        outs = self._feed(lock, [_pose(1) for _ in range(n)])
        self.assertIsNotNone(outs[-1])
        self.assertEqual(lock.state, LOCK_LOCKED)

    def test_standard_preset_refuses_one_marker(self):
        lock = PoseLock(PoseKalmanFilter(), POSE_LOCK_PRESETS["standard"])
        outs = self._feed(lock, [_pose(1) for _ in range(10)])
        self.assertTrue(all(o is None for o in outs))
        self.assertEqual(lock.last_reject_reason, "count")

    def test_legacy3_preset_refuses_two_markers(self):
        lock = PoseLock(PoseKalmanFilter(), POSE_LOCK_PRESETS["legacy3"])
        outs = self._feed(lock, [_pose(2) for _ in range(10)])
        self.assertTrue(all(o is None for o in outs))

    def test_rms_ceiling_must_cover_sustain_floor(self):
        with self.assertRaises(ValueError):
            PoseLockConfig(max_reproj_rms_px=5.0, max_acquire_reproj_rms_px=4.0)

    def test_threshold_order_is_validated(self):
        with self.assertRaises(ValueError):
            PoseLockConfig(acquire_min_markers=4)
        with self.assertRaises(ValueError):
            PoseLockConfig(sustain_min_markers=3, acquire_min_markers=2)
        with self.assertRaises(ValueError):
            PoseLockConfig(acquire_confirm_frames=0)


class QualityGatesAtEveryMarkerCountTests(unittest.TestCase):
    """Audit F02: marker count must not bypass finite / depth / RMS / motion gates."""

    COUNTS = (1, 2, 3, 4, 6)

    def _locked(self):
        lock = PoseLock(PoseKalmanFilter(), PoseLockConfig(coast_duration_s=0.3))
        for i in range(10):
            self.assertIsNotNone(lock.process(_pose(3), now=0.033 * i))
        self.assertEqual(lock.state, LOCK_LOCKED)
        return lock

    def test_audit_case_three_markers_high_rms_and_jump_after_lock(self):
        lock = self._locked()
        out = lock.process(_pose(3, tvec=(0.3, 0.0, 0.3), rms=25.0), now=0.33)
        self.assertTrue(out.coasted)
        self.assertIn(lock.last_reject_reason, ("rms", "jump"))

    def test_high_rms_rejected_at_every_count_once_locked(self):
        for n in self.COUNTS:
            lock = self._locked()
            out = lock.process(_pose(n, rms=25.0), now=0.4)
            self.assertTrue(out.coasted, f"n={n}")
            self.assertEqual(lock.last_reject_reason, "rms", f"n={n}")

    def test_translation_jump_rejected_at_every_count_once_locked(self):
        for n in self.COUNTS:
            lock = self._locked()
            out = lock.process(_pose(n, tvec=(0.3, 0.0, 0.3)), now=0.4)
            self.assertTrue(out.coasted, f"n={n}")
            self.assertEqual(lock.last_reject_reason, "jump", f"n={n}")

    def test_rotation_jump_rejected_at_every_count_once_locked(self):
        for n in self.COUNTS:
            lock = self._locked()
            out = lock.process(_pose(n, rvec=(0.0, 1.0, 0.0)), now=0.4)  # ~57 deg
            self.assertTrue(out.coasted, f"n={n}")
            self.assertEqual(lock.last_reject_reason, "jump", f"n={n}")

    def test_non_finite_pose_rejected_at_every_count_and_state(self):
        bad = {
            "nan_t": dict(tvec=(np.nan, 0.0, 0.3)),
            "inf_t": dict(tvec=(0.0, np.inf, 0.3)),
            "nan_r": dict(rvec=(0.0, np.nan, 0.0)),
            "nan_rms": dict(rms=float("nan")),
        }
        for label, kw in bad.items():
            for n in self.COUNTS:
                fresh = PoseLock(PoseKalmanFilter())
                self.assertIsNone(fresh.process(_pose(n, **kw), now=0.0), f"{label} n={n}")
                self.assertEqual(fresh.last_reject_reason, "invalid", f"{label} n={n}")
                lock = self._locked()
                out = lock.process(_pose(n, **kw), now=0.4)
                self.assertTrue(out.coasted, f"{label} n={n}")
                self.assertEqual(lock.last_reject_reason, "invalid", f"{label} n={n}")
                self.assertTrue(np.all(np.isfinite(lock.kalman.coast()[1])))

    def test_negative_or_zero_depth_rejected_at_every_count(self):
        for z in (-0.3, 0.0):
            for n in self.COUNTS:
                fresh = PoseLock(PoseKalmanFilter())
                self.assertIsNone(fresh.process(_pose(n, tvec=(0.0, 0.0, z)), now=0.0))
                self.assertEqual(fresh.last_reject_reason, "invalid")
                lock = self._locked()
                out = lock.process(_pose(n, tvec=(0.0, 0.0, z)), now=0.4)
                self.assertTrue(out.coasted)
                self.assertEqual(lock.last_reject_reason, "invalid")

    def test_ambiguous_solve_rejected_at_acquisition_for_every_count(self):
        for n in self.COUNTS:
            lock = PoseLock(PoseKalmanFilter(), PoseLockConfig(acquire_min_markers=1))
            out = lock.process(
                _pose(n, ambiguity_ratio=1.1, ambiguity_separation_deg=50.0), now=0.0,
            )
            self.assertIsNone(out, f"n={n}")
            self.assertEqual(lock.last_reject_reason, "ambiguous", f"n={n}")

    def test_motion_gate_is_time_aware(self):
        # 0.2 m is a jump within 33 ms but plausible after 0.25 s at 1 m/s.
        quick = self._locked()
        last_accept = 0.033 * 9
        out = quick.process(_pose(3, tvec=(0.2, 0.0, 0.3)), now=last_accept + 0.036)
        self.assertTrue(out.coasted)
        slow = self._locked()
        slow.config = PoseLockConfig(coast_duration_s=1.0)  # keep the filter alive
        out = slow.process(_pose(3, tvec=(0.2, 0.0, 0.3)), now=last_accept + 0.25)
        self.assertFalse(out.coasted)

    def test_rejected_candidates_do_not_update_the_filter(self):
        lock = self._locked()
        before = lock.kalman.coast()[1].copy()
        lock.process(_pose(3, tvec=(0.3, 0.0, 0.3), rms=25.0), now=0.33)
        # coast() advances by prediction only; the 0.3 m outlier must be absent
        self.assertLess(abs(float(lock.kalman.coast()[1][0, 0]) - float(before[0, 0])), 0.01)

    def test_moved_specimen_is_reacquired_through_searching(self):
        lock = self._locked()
        moved = lambda: _pose(3, tvec=(0.4, 0.0, 0.3))
        t = 0.33
        states = []
        for _ in range(20):
            lock.process(moved(), now=t)
            states.append(lock.state)
            t += 0.033
        # rejected while the gates hold the old pose, then SEARCHING -> new lock
        self.assertIn(LOCK_COASTING, states)
        self.assertIn(LOCK_SEARCHING, states)
        self.assertEqual(lock.state, LOCK_LOCKED)
        np.testing.assert_allclose(
            lock.kalman.coast()[1].reshape(3)[0], 0.4, atol=0.05,
        )


class KalmanCoastTests(unittest.TestCase):
    def test_coast_advances_prediction(self):
        kf = PoseKalmanFilter()
        rvec = np.zeros((3, 1))
        # Feed a constant-velocity translation in x
        for i in range(10):
            tvec = np.array([[0.01 * i], [0.0], [0.3]])
            kf.update(rvec, tvec)
        _, t1 = kf.coast()
        _, t2 = kf.coast()
        self.assertGreater(float(t2[0, 0]), float(t1[0, 0]))


class MarkerCarryoverTests(unittest.TestCase):
    @staticmethod
    def _frame_with_square(x0, y0, size=40):
        img = np.full((240, 320), 255, dtype=np.uint8)
        cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), 0, -1)
        return img

    @staticmethod
    def _quad(x0, y0, size=40):
        return np.array(
            [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
            dtype=np.float32,
        ).reshape(1, 4, 2)

    def test_missing_marker_is_carried_over(self):
        carry = MarkerCarryover(max_age_frames=3)
        f1 = self._frame_with_square(100, 100)
        f2 = self._frame_with_square(103, 101)

        corners, ids = carry.process(f1, [self._quad(100, 100)], np.array([[7]]))
        self.assertEqual(ids.flatten().tolist(), [7])

        # Marker missed in frame 2 -> tracked forward
        corners, ids = carry.process(f2, [], None)
        self.assertIsNotNone(ids)
        self.assertEqual(ids.flatten().tolist(), [7])
        self.assertEqual(carry.last_carryover_count, 1)
        # Tracked corners should have moved roughly with the square
        c = corners[0].reshape(4, 2)
        self.assertAlmostEqual(float(c[0, 0]), 103.0, delta=2.5)

    def test_track_ages_out(self):
        carry = MarkerCarryover(max_age_frames=2)
        f = self._frame_with_square(100, 100)
        carry.process(f, [self._quad(100, 100)], np.array([[7]]))
        for _ in range(2):
            _, ids = carry.process(f, [], None)
            self.assertIsNotNone(ids)
        _, ids = carry.process(f, [], None)
        self.assertIsNone(ids)

    def test_redetection_resets_age(self):
        carry = MarkerCarryover(max_age_frames=1)
        f = self._frame_with_square(100, 100)
        carry.process(f, [self._quad(100, 100)], np.array([[7]]))
        carry.process(f, [], None)                                   # age 1
        carry.process(f, [self._quad(100, 100)], np.array([[7]]))    # refresh
        _, ids = carry.process(f, [], None)
        self.assertIsNotNone(ids)


class RoiRedetectorTests(unittest.TestCase):
    def test_recovers_small_marker_in_predicted_roi(self):
        marker_id = 5
        marker_size_m = 0.012
        fx = 800.0
        cam = np.array([[fx, 0, 320], [0, fx, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)
        # z chosen so the marker is ~36 px in the image (small)
        z = marker_size_m * fx / 36.0
        rvec = np.zeros((3, 1))
        tvec = np.array([[0.0], [0.0], [z]])

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        half = marker_size_m / 2.0
        obj = np.array(
            [[-half, half, 0], [half, half, 0],
             [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        board = cv2.aruco.Board([obj], dictionary, np.array([marker_id]))

        # Render the marker where the pose projects it
        proj, _ = cv2.projectPoints(obj, rvec, tvec, cam, dist)
        proj = proj.reshape(4, 2)
        side = int(round(proj[:, 0].max() - proj[:, 0].min()))
        x0, y0 = int(round(proj[:, 0].min())), int(round(proj[:, 1].min()))
        img = np.full((480, 640), 255, dtype=np.uint8)
        marker_px = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        img[y0:y0 + side, x0:x0 + side] = marker_px

        detector = cv2.aruco.ArucoDetector(dictionary)
        roi = RoiRedetector(board, detector, upscale=2.0)
        corners, ids = roi.recover(img, [marker_id], rvec, tvec, cam, dist)

        self.assertEqual(ids, [marker_id])
        centre = corners[0].reshape(4, 2).mean(axis=0)
        self.assertAlmostEqual(float(centre[0]), 320.0, delta=3.0)
        self.assertAlmostEqual(float(centre[1]), 240.0, delta=3.0)

    def test_no_recovery_without_board_geometry(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary)
        board = cv2.aruco.Board(
            [np.zeros((4, 3), dtype=np.float32)], dictionary, np.array([1]),
        )
        roi = RoiRedetector(board, detector)
        corners, ids = roi.recover(
            np.full((100, 100), 255, dtype=np.uint8),
            [99],  # not on the board
            np.zeros((3, 1)), np.array([[0.0], [0.0], [0.3]]),
            np.eye(3), np.zeros(5),
        )
        self.assertEqual(ids, [])


class MinMarkersOverrideTests(unittest.TestCase):
    def test_estimate_pose_relaxes_gate_with_override(self):
        calls = []

        class _FakeBoard:
            def matchImagePoints(self, corners, ids):
                calls.append(ids.copy())
                raise cv2.error("stop here")  # gate check is what we test

        detector = LStructureDetector(
            board=_FakeBoard(),
            board_marker_ids={0, 1, 2},
            min_board_markers=3,
        )
        corners = [np.zeros((1, 4, 2), dtype=np.float32) for _ in range(2)]
        ids = np.array([[0], [1]], dtype=np.int32)
        cam = np.eye(3)
        dist = np.zeros(5)

        # Default gate: 2 markers blocked before matchImagePoints
        self.assertIsNone(detector.estimate_pose(corners, ids, cam, dist))
        self.assertEqual(calls, [])

        # Relaxed gate: 2 markers reach the board solve
        self.assertIsNone(
            detector.estimate_pose(corners, ids, cam, dist, min_markers=2)
        )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
