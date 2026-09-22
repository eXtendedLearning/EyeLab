import time
import unittest
from unittest.mock import patch

import numpy as np

import cv2

import pose_estimator

from pose_estimator import (
    ARUCO_PREPROCESS_CLIP_LIMIT,
    ArucoDetectorTuning,
    ArucoPipeline,
    DETECTOR_TUNING_PRESETS,
    LStructureDetector,
    REFINE_DUTY_CYCLE,
    REFINE_IDLE_LIMIT,
    REFINE_MAX_REJECTED,
    ThreadedCapture,
    make_detector_parameters,
    preprocess_frame,
)
from registration import MarkerCorrespondence


class _FakeBoard:
    def __init__(self):
        self.calls = []

    def matchImagePoints(self, corners, ids):
        self.calls.append((corners, ids.copy()))
        obj_pts = np.zeros((12, 1, 3), dtype=np.float32)
        img_pts = np.zeros((12, 1, 2), dtype=np.float32)
        return obj_pts, img_pts


class BoardPoseMarkerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.eye(3, dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.corners = [np.zeros((1, 4, 2), dtype=np.float32) for _ in range(4)]

    def test_board_pose_requires_three_structure_markers(self):
        board = _FakeBoard()
        detector = LStructureDetector(
            board=board,
            board_marker_ids={0, 1, 2},
            min_board_markers=3,
        )

        pose = detector.estimate_pose(
            self.corners[:2],
            np.array([[0], [1]], dtype=np.int32),
            self.camera_matrix,
            self.dist_coeffs,
        )

        self.assertIsNone(pose)
        self.assertEqual(board.calls, [])

    def test_board_pose_uses_only_structure_marker_ids(self):
        board = _FakeBoard()
        detector = LStructureDetector(
            board=board,
            board_marker_ids={0, 1, 2},
            min_board_markers=3,
        )

        rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        tvec = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        with patch("pose_estimator.cv2.solvePnP", return_value=(True, rvec, tvec)), patch(
            "pose_estimator.cv2.solvePnPRefineLM",
            return_value=(rvec, tvec),
        ):
            pose = detector.estimate_pose(
                self.corners,
                np.array([[0], [44], [2], [1]], dtype=np.int32),
                self.camera_matrix,
                self.dist_coeffs,
            )

        self.assertIsNotNone(pose)
        self.assertEqual(pose.marker_ids, [0, 2, 1])
        self.assertEqual(pose.marker_count, 3)
        np.testing.assert_array_equal(board.calls[0][1], np.array([[0], [2], [1]], dtype=np.int32))


class PlanarPoseSolveTests(unittest.TestCase):
    """Single-marker (planar) board solves must resolve the two-fold ambiguity."""

    def setUp(self):
        self.detector = LStructureDetector()
        fx = 800.0
        self.cam = np.array(
            [[fx, 0, 320], [0, fx, 240], [0, 0, 1]], dtype=np.float64,
        )
        self.dist = np.zeros(5, dtype=np.float64)
        half = 0.006  # 12 mm marker
        self.obj = np.array(
            [[-half, half, 0], [half, half, 0],
             [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )

    def test_points_are_planar(self):
        self.assertTrue(LStructureDetector._points_are_planar(self.obj))
        nonplanar = self.obj.copy()
        nonplanar[0, 2] = 0.05
        self.assertFalse(LStructureDetector._points_are_planar(nonplanar))

    def test_planar_solve_with_prior_recovers_true_pose(self):
        true_rvec = np.array([[0.4], [0.3], [0.1]], dtype=np.float64)
        true_tvec = np.array([[0.01], [0.02], [0.30]], dtype=np.float64)
        img, _ = cv2.projectPoints(self.obj, true_rvec, true_tvec, self.cam, self.dist)

        rvec, tvec = self.detector._solve_board_pose(
            self.obj, img.reshape(-1, 2), self.cam, self.dist,
            prior=(true_rvec, true_tvec),
        )

        self.assertIsNotNone(rvec)
        from pose_lock import rotation_angle_between
        self.assertLess(rotation_angle_between(rvec, true_rvec), 2.0)
        np.testing.assert_allclose(tvec.flatten(), true_tvec.flatten(), atol=2e-3)


class DetectorParameterTests(unittest.TestCase):
    def test_default_detector_parameters_are_balanced_between_strict_and_forgiving(self):
        strict = DETECTOR_TUNING_PRESETS["strict"]
        balanced = DETECTOR_TUNING_PRESETS["balanced"]
        forgiving = DETECTOR_TUNING_PRESETS["forgiving"]
        params = make_detector_parameters()

        self.assertEqual(ARUCO_PREPROCESS_CLIP_LIMIT, 2.5)
        self.assertLess(strict.clip_limit, balanced.clip_limit)
        self.assertLess(balanced.clip_limit, forgiving.clip_limit)
        self.assertLess(strict.adaptive_thresh_win_size_max, balanced.adaptive_thresh_win_size_max)
        self.assertLess(balanced.adaptive_thresh_win_size_max, forgiving.adaptive_thresh_win_size_max)
        self.assertLess(forgiving.min_marker_perimeter_rate, balanced.min_marker_perimeter_rate)
        self.assertLess(balanced.min_marker_perimeter_rate, strict.min_marker_perimeter_rate)

        self.assertEqual(params.adaptiveThreshWinSizeMin, 3)
        self.assertEqual(params.adaptiveThreshWinSizeMax, 37)
        self.assertAlmostEqual(params.minMarkerPerimeterRate, 0.022)
        self.assertAlmostEqual(params.polygonalApproxAccuracyRate, 0.055)
        self.assertAlmostEqual(params.errorCorrectionRate, 0.7)
        self.assertEqual(params.cornerRefinementMethod, cv2.aruco.CORNER_REFINE_SUBPIX)

    def test_structure_detector_uses_default_balanced_parameters(self):
        detector = LStructureDetector()

        self.assertEqual(detector.det_params.adaptiveThreshWinSizeMax, 37)
        self.assertAlmostEqual(detector.det_params.minMarkerPerimeterRate, 0.022)

    def test_detector_tuning_can_be_applied_live(self):
        detector = LStructureDetector()
        strict = DETECTOR_TUNING_PRESETS["strict"]

        detector.set_detector_tuning(strict)

        self.assertEqual(detector.det_params.adaptiveThreshWinSizeMax, 23)
        self.assertAlmostEqual(detector.det_params.errorCorrectionRate, 0.6)

    def test_detector_records_raw_allowed_and_rejected_counts(self):
        class FakeArucoDetector:
            def detectMarkers(self, _gray):
                corners = [
                    np.zeros((1, 4, 2), dtype=np.float32),
                    np.ones((1, 4, 2), dtype=np.float32),
                ]
                ids = np.array([[1], [7]], dtype=np.int32)
                rejected = [np.zeros((1, 4, 2), dtype=np.float32) for _ in range(3)]
                return corners, ids, rejected

        detector = LStructureDetector(allowed_ids={7})
        detector.detector = FakeArucoDetector()

        _corners, ids = detector.detect(np.zeros((20, 20), dtype=np.uint8))

        np.testing.assert_array_equal(ids, np.array([[7]], dtype=np.int32))
        self.assertEqual(detector.last_raw_marker_count, 2)
        self.assertEqual(detector.last_allowed_marker_count, 1)
        self.assertEqual(detector.last_rejected_count, 3)


class ArucoPipelineModeTests(unittest.TestCase):
    def test_structure_board_mode_uses_full_detection_every_frame(self):
        correspondences = [
            MarkerCorrespondence(i, np.array([i * 0.04, 0.0, 0.0], dtype=np.float64))
            for i in range(3)
        ]

        pipeline = ArucoPipeline(
            board_correspondences=correspondences,
            use_optical_flow=True,
        )

        self.assertTrue(pipeline.uses_structure_board)
        self.assertFalse(pipeline.use_optical_flow)

    def test_pipeline_applies_detector_tuning_live(self):
        pipeline = ArucoPipeline(detector_tuning=ArucoDetectorTuning(error_correction_rate=0.5))

        pipeline.apply_detector_tuning(ArucoDetectorTuning(error_correction_rate=0.85))

        self.assertAlmostEqual(pipeline.l_detector.det_params.errorCorrectionRate, 0.85)

    def test_pipeline_applies_allowed_ids_live(self):
        pipeline = ArucoPipeline(allowed_ids={1, 2})

        pipeline.apply_allowed_ids({3, 4})

        self.assertEqual(pipeline.l_detector.allowed_ids, {3, 4})


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in: N good frames, then failures."""

    def __init__(self, good_frames: int):
        self.good_frames = good_frames
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def get(self, *_args):
        return 0.0

    def read(self):
        if self.good_frames > 0:
            self.good_frames -= 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        time.sleep(0.001)
        return False, None

    def release(self):
        self.released = True


class ThreadedCaptureStatsTests(unittest.TestCase):
    """The capture counters feeding the AR freeze diagnostics log."""

    def _capture(self, good_frames: int) -> ThreadedCapture:
        with patch("pose_estimator.open_camera", return_value=_FakeCapture(good_frames)):
            return ThreadedCapture(camera_index=0)

    def test_counts_reads_and_failures(self):
        capture = self._capture(good_frames=3)
        capture.start()
        deadline = time.perf_counter() + 0.3
        while capture._reads < 10 and time.perf_counter() < deadline:
            time.sleep(0.005)
        capture.stop()

        stats = capture.stats()
        self.assertGreaterEqual(stats["reads"], 10)
        self.assertGreaterEqual(stats["failures"], 1)
        self.assertGreaterEqual(stats["max_fail_streak"], 1)
        self.assertEqual(stats["fail_streak"], stats["max_fail_streak"])
        self.assertIsNotNone(stats["last_ok_age_ms"])
        self.assertFalse(stats["thread_alive"])
        self.assertFalse(stats["join_timed_out"])

    def test_gives_up_instead_of_spinning_on_a_dead_camera(self):
        capture = self._capture(good_frames=0)
        with patch.object(pose_estimator, "CAPTURE_GIVE_UP_S", 0.2):
            capture.start()
            deadline = time.perf_counter() + 3.0
            while capture._thread.is_alive() and time.perf_counter() < deadline:
                time.sleep(0.01)

        stats = capture.stats()
        self.assertFalse(stats["thread_alive"], "the reader must stop, not retry forever")
        self.assertIsNotNone(stats["error"])
        self.assertIn("no frames", stats["error"])
        # The bug this replaces managed 70.9 million failed reads in 30 s.
        self.assertLess(stats["reads"], 200, "failed reads must be paced, not spun")
        capture.stop()

    def test_accepts_an_already_opened_capture(self):
        cap = _FakeCapture(good_frames=1)
        capture = ThreadedCapture(camera_index=3, cap=cap)
        self.assertIs(capture.cap, cap)
        self.assertEqual(capture.camera_index, 3)

    def test_stats_before_start_are_empty(self):
        capture = self._capture(good_frames=1)
        stats = capture.stats()
        self.assertEqual(stats["reads"], 0)
        self.assertIsNone(stats["last_ok_age_ms"])
        self.assertFalse(stats["thread_alive"])


class TestRefineDutyCycle(unittest.TestCase):
    """refineDetectedMarkers is the priciest call in detect(); it must not run
    unconditionally once it has stopped recovering markers."""

    def _detector(self):
        return LStructureDetector(board=None)

    def test_runs_every_frame_while_still_recovering(self):
        det = self._detector()
        self.assertTrue(all(det._should_refine(4) for _ in range(REFINE_IDLE_LIMIT)))

    def test_duty_cycles_once_idle(self):
        det = self._detector()
        det._refine_idle_frames = REFINE_IDLE_LIMIT
        runs = sum(1 for _ in range(4 * REFINE_DUTY_CYCLE) if det._should_refine(4))
        self.assertEqual(runs, 4)

    def test_recovering_a_marker_restores_every_frame(self):
        det = self._detector()
        det._refine_idle_frames = REFINE_IDLE_LIMIT
        det._refine_idle_frames = 0  # what detect() does on a successful recovery
        self.assertTrue(all(det._should_refine(4) for _ in range(REFINE_DUTY_CYCLE)))

    def test_skipped_when_candidates_explode(self):
        det = self._detector()
        self.assertFalse(det._should_refine(REFINE_MAX_REJECTED + 1))


class TestClaheCache(unittest.TestCase):
    """CLAHE carries no state between calls, so one per frame was pure cost."""

    def setUp(self):
        pose_estimator._CLAHE_CACHE.clear()
        self.addCleanup(pose_estimator._CLAHE_CACHE.clear)
        self.gray = np.full((32, 32), 120, dtype=np.uint8)

    def test_same_clip_limit_builds_one_object(self):
        with patch.object(
            pose_estimator.cv2, "createCLAHE", wraps=cv2.createCLAHE
        ) as make:
            for _ in range(5):
                preprocess_frame(self.gray, clip_limit=2.5)
        self.assertEqual(make.call_count, 1)

    def test_distinct_clip_limits_get_distinct_objects(self):
        with patch.object(
            pose_estimator.cv2, "createCLAHE", wraps=cv2.createCLAHE
        ) as make:
            preprocess_frame(self.gray, clip_limit=2.0)
            preprocess_frame(self.gray, clip_limit=3.0)
            preprocess_frame(self.gray, clip_limit=2.0)
        self.assertEqual(make.call_count, 2)

    def test_output_is_unchanged_by_caching(self):
        first = preprocess_frame(self.gray, clip_limit=ARUCO_PREPROCESS_CLIP_LIMIT)
        second = preprocess_frame(self.gray, clip_limit=ARUCO_PREPROCESS_CLIP_LIMIT)
        expected = cv2.createCLAHE(
            clipLimit=ARUCO_PREPROCESS_CLIP_LIMIT, tileGridSize=(8, 8)
        ).apply(self.gray)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)


if __name__ == "__main__":
    unittest.main()
