import tempfile
import unittest
from pathlib import Path

import numpy as np

from registration import (
    MarkerCorrespondence,
    load_marker_config,
    marker_axes_from_normal,
    marker_object_corners,
    normal_label,
    save_marker_config,
)


class MarkerOrientationTests(unittest.TestCase):
    def test_marker_axes_match_requested_normal(self):
        for normal in (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ):
            x_axis, y_axis, n_axis = marker_axes_from_normal(normal, roll_deg=37.0)
            self.assertTrue(np.allclose(np.cross(x_axis, y_axis), n_axis, atol=1e-9))
            self.assertTrue(np.allclose(n_axis, np.array(normal, dtype=float), atol=1e-9))

    def test_marker_corners_keep_center_and_size(self):
        center = np.array([0.2, -0.1, 0.05], dtype=float)
        corners = marker_object_corners(center, (0.0, 0.0, 1.0), 90.0, 0.012)
        self.assertTrue(np.allclose(corners.mean(axis=0), center, atol=1e-9))
        edge_lengths = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
        self.assertTrue(np.allclose(edge_lengths, [0.012] * 4, atol=1e-9))

    def test_marker_config_roundtrip_preserves_pose_fields(self):
        corr = MarkerCorrespondence(
            marker_id=4,
            unv_position=np.array([1.0, 2.0, 3.0], dtype=float),
            node_id=42,
            description="test marker",
            normal=np.array([0.0, -1.0, 0.0], dtype=float),
            roll_deg=90.0,
            marker_size_mm=20.0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "markers.json"
            save_marker_config(str(path), [corr])
            loaded = load_marker_config(str(path))[0]

        self.assertEqual(loaded.marker_id, 4)
        self.assertEqual(loaded.node_id, 42)
        self.assertEqual(normal_label(loaded.normal), "-Y")
        self.assertEqual(loaded.roll_deg, 90.0)
        self.assertEqual(loaded.marker_size_mm, 20.0)


if __name__ == "__main__":
    unittest.main()
