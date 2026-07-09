import unittest

import cv2
import numpy as np

from pose_estimator import LStructureDetector, PoseKalmanFilter, PoseResult
from pose_lock import (
    LOCK_COASTING,
    LOCK_LOCKED,
    LOCK_SEARCHING,
    MarkerCarryover,
    PoseLock,
    PoseLockConfig,
    RoiRedetector,
    align_rvec,
    rotation_angle_between,
)


def _pose(n_markers, tvec=(0.0, 0.0, 0.3), rvec=(0.0, 0.0, 0.0), rms=0.5):
    return PoseResult(
        rvec=np.array(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.array(tvec, dtype=np.float64).reshape(3, 1),
        marker_ids=list(range(n_markers)),
        marker_count=n_markers,
        timestamp=0.0,
        rms_reproj_px=rms,
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

    def test_full_count_pose_accepted_regardless_of_rms(self):
        # Real board geometry can have a large stable baseline RMS; gating
        # acquisition on it would prevent the lock from ever engaging.
        out = self.lock.process(_pose(3, rms=50.0), now=0.0)
        self.assertIsNotNone(out)
        self.assertEqual(self.lock.state, LOCK_LOCKED)

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
        # After expiry, 2 markers are not enough to reacquire
        self.assertIsNone(self.lock.process(_pose(2), now=1.033))
        self.assertEqual(self.lock.state, LOCK_SEARCHING)

    def test_min_markers_property_tracks_state(self):
        self.assertEqual(self.lock.min_markers, 3)
        self._acquire(t=0.0)
        self.assertEqual(self.lock.min_markers, 2)


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
