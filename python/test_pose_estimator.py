import unittest
from unittest.mock import patch

import numpy as np

import cv2

from pose_estimator import (
    ARUCO_PREPROCESS_CLIP_LIMIT,
    ArucoDetectorTuning,
    ArucoPipeline,
    DETECTOR_TUNING_PRESETS,
    LStructureDetector,
    make_detector_parameters,
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


if __name__ == "__main__":
    unittest.main()
