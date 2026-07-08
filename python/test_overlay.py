import unittest

import cv2
import numpy as np

import overlay


def _camera(fx=100.0, fy=100.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class ProjectNodesTests(unittest.TestCase):
    def setUp(self):
        self.K = _camera()
        self.dist = np.zeros(5, dtype=np.float64)

    def test_point_on_axis_maps_to_principal_point(self):
        nodes = [{"id": 1, "x": 0.0, "y": 0.0, "z": 1.0}]
        px = overlay.project_nodes(nodes, self.K, self.dist)
        self.assertEqual(px[1], (320, 240))

    def test_offset_point_uses_pinhole_projection(self):
        # x' = fx * X / Z + cx -> 100 * 0.5 / 1 + 320 = 370
        nodes = [{"id": 7, "x": 0.5, "y": -0.5, "z": 1.0}]
        px = overlay.project_nodes(nodes, self.K, self.dist)
        self.assertEqual(px[7], (370, 190))

    def test_node_transform_applied_before_projection(self):
        # Node sits at z=0 (behind/at camera); transform lifts it to z=1 in front.
        nodes = [{"id": 3, "x": 0.0, "y": 0.0, "z": 0.0}]
        shift = lambda p: p + np.array([0.0, 0.0, 1.0])
        px = overlay.project_nodes(nodes, self.K, self.dist, node_transform=shift)
        self.assertEqual(px[3], (320, 240))

    def test_node_transform_returning_none_is_skipped(self):
        nodes = [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 0.0, "y": 0.0, "z": 0.0},
        ]
        drop_even = lambda p: None  # registration not ready -> everything skipped
        self.assertEqual(overlay.project_nodes(nodes, self.K, self.dist,
                                               node_transform=drop_even), {})

    def test_empty_nodes_returns_empty(self):
        self.assertEqual(overlay.project_nodes([], self.K, self.dist), {})

    def test_extreme_projected_points_are_skipped_for_opencv_line_safety(self):
        K = _camera(fx=3_000_000_000.0, fy=3_000_000_000.0, cx=0.0, cy=0.0)
        nodes = [{"id": 1, "x": 1.0, "y": 0.0, "z": 1.0}]

        self.assertEqual(overlay.project_nodes(nodes, K, self.dist), {})

    def test_projected_points_can_be_passed_to_opencv_line(self):
        nodes = [{"id": 1, "x": 0.0, "y": 0.0, "z": 1.0}]
        point = overlay.project_nodes(nodes, self.K, self.dist)[1]
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.line(canvas, point, point, (255, 255, 255), 1, cv2.LINE_AA)


class WireframeSegmentsTests(unittest.TestCase):
    def test_only_fully_projected_edges_survive(self):
        node_px = {1: (0, 0), 2: (1, 1), 3: (2, 2)}
        edges = [[1, 2], [2, 3], [1, 4]]  # node 4 never projected
        segs = overlay.wireframe_segments(node_px, edges)
        self.assertEqual(segs, [((0, 0), (1, 1)), ((1, 1), (2, 2))])

    def test_no_edges_no_segments(self):
        self.assertEqual(overlay.wireframe_segments({1: (0, 0)}, []), [])


if __name__ == "__main__":
    unittest.main()
