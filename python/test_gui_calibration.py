import unittest
from unittest.mock import Mock, patch

import numpy as np

from gui_calibration import CalibrationWindow
from gui_common import marker_up_label, position_ui_to_m, position_m_to_ui


class _FakeWindow:
    def __init__(self):
        self.destroyed = False
        self.unbound = []

    def destroy(self):
        self.destroyed = True

    def unbind_all(self, sequence):
        self.unbound.append(sequence)


class CalibrationWindowFinishTests(unittest.TestCase):
    def test_position_ui_helpers_convert_centimetres_to_stored_metres(self):
        self.assertAlmostEqual(position_m_to_ui(0.125), 12.5)
        self.assertAlmostEqual(position_ui_to_m(12.5), 0.125)

    def test_marker_up_label_uses_face_and_roll(self):
        self.assertEqual(marker_up_label("+Z", 0.0), "+Y")
        self.assertEqual(marker_up_label("+Z", 90.0), "-X")

    def test_finish_accepts_opencv_charuco_five_value_return(self):
        window = CalibrationWindow.__new__(CalibrationWindow)
        callback = Mock()
        camera_matrix = np.eye(3, dtype=np.float64)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        window.callback = callback
        window.win = _FakeWindow()
        window.cap = Mock()
        window.status_var = Mock()
        window.board = object()
        window.image_size = (640, 480)
        window.min_frames = 15
        window.all_corners = [object()] * 15
        window.all_ids = [object()] * 15
        window._running = True

        with patch(
            "gui_calibration.cv2.aruco.calibrateCameraCharuco",
            return_value=(0.42, camera_matrix, dist_coeffs, [], []),
        ) as calibrate:
            window._finish()

        calibrate.assert_called_once_with(
            window.all_corners,
            window.all_ids,
            window.board,
            window.image_size,
            None,
            None,
        )
        window.cap.release.assert_called_once_with()
        self.assertEqual(window.win.unbound, ["<space>", "<Escape>"])
        self.assertTrue(window.win.destroyed)
        self.assertFalse(window._running)
        callback.assert_called_once_with(camera_matrix, dist_coeffs, 0.42, (640, 480))


if __name__ == "__main__":
    unittest.main()
