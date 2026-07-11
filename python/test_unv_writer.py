import tempfile
import unittest
from pathlib import Path

from unv_to_json import UNVParser
from unv_writer import (
    polylines_to_trace_sequence,
    segments_to_polylines,
    write_unv,
)


class PolylineHelpersTests(unittest.TestCase):
    def test_segments_chain_into_polylines(self):
        chains = segments_to_polylines([[1, 2], [2, 3], [7, 8]])
        self.assertEqual(chains, [[1, 2, 3], [7, 8]])

    def test_polylines_pass_through(self):
        chains = segments_to_polylines([[1, 2, 3, 4]])
        self.assertEqual(chains, [[1, 2, 3, 4]])

    def test_trace_sequence_uses_pen_up_separator(self):
        seq = polylines_to_trace_sequence([[1, 2, 3], [7, 8]])
        self.assertEqual(seq, [1, 2, 3, 0, 7, 8])


class WriteUnvRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "out.unv"
        self.nodes = [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 0.1, "y": 0.0, "z": 0.0},
            {"id": 3, "x": 0.1, "y": 0.2, "z": 0.0},
            {"id": 4, "x": 0.0, "y": 0.0, "z": 0.3},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_nodes_and_lines(self):
        write_unv(self.path, self.nodes, [[1, 2], [2, 3], [1, 4]])
        result = UNVParser(self.path, validate_cs=False).parse()

        self.assertEqual(len(result["nodes"]), 4)
        got = {n["id"]: (n["x"], n["y"], n["z"]) for n in result["nodes"]}
        self.assertAlmostEqual(got[3][1], 0.2, places=9)
        self.assertAlmostEqual(got[4][2], 0.3, places=9)

        segments = {tuple(sorted(seg)) for seg in result["traceLines"]}
        self.assertEqual(segments, {(1, 2), (2, 3), (1, 4)})

    def test_round_trip_nodes_only(self):
        write_unv(self.path, self.nodes)
        result = UNVParser(self.path, validate_cs=False).parse()
        self.assertEqual(len(result["nodes"]), 4)

    def test_rejects_unknown_line_node(self):
        with self.assertRaises(ValueError):
            write_unv(self.path, self.nodes, [[1, 99]])

    def test_rejects_duplicate_node_ids(self):
        nodes = self.nodes + [{"id": 1, "x": 1.0, "y": 1.0, "z": 1.0}]
        with self.assertRaises(ValueError):
            write_unv(self.path, nodes)

    def test_rejects_empty_nodes(self):
        with self.assertRaises(ValueError):
            write_unv(self.path, [])


class EditorHelperTests(unittest.TestCase):
    def test_ray_plane_intersection(self):
        import numpy as np
        from gui_geometry_editor import ray_plane_intersection

        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        p = ray_plane_intersection(origin, direction, "XY", 0.0)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(float(p[2]), 0.0)

        # Parallel ray
        parallel = np.array([1.0, 0.0, 0.0])
        self.assertIsNone(ray_plane_intersection(origin, parallel, "XY", 0.0))

        # Offset plane on another axis
        p = ray_plane_intersection(
            np.array([0.5, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), "XZ", 0.25,
        )
        self.assertAlmostEqual(float(p[1]), 0.25)

    def test_click_ray_recovers_3d_point_on_plane(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import proj3d

        from gui_geometry_editor import click_ray, ray_plane_intersection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        fig.canvas.draw()
        proj = ax.get_proj()

        target = np.array([0.12, -0.2, 0.1])
        x2, y2, _ = proj3d.proj_transform(*target, proj)
        origin, direction = click_ray(x2, y2, proj)
        hit = ray_plane_intersection(origin, direction, "XY", 0.1)

        self.assertIsNotNone(hit)
        np.testing.assert_allclose(hit, target, atol=1e-9)
        plt.close(fig)

    def test_nearest_node_and_ids(self):
        import numpy as np
        from gui_geometry_editor import nearest_node_index, next_node_id, remove_node

        nodes = [{"id": 1}, {"id": 5}]
        display = np.array([[100.0, 100.0], [300.0, 300.0]])
        self.assertEqual(nearest_node_index(nodes, np.array([102.0, 101.0]), display), 0)
        self.assertIsNone(nearest_node_index(nodes, np.array([200.0, 200.0]), display))
        self.assertEqual(next_node_id([]), 1)
        self.assertEqual(next_node_id(nodes), 6)

        full = [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 1, "y": 0, "z": 0}]
        kept_nodes, kept_lines = remove_node(full, [[1, 2]], 1)
        self.assertEqual([n["id"] for n in kept_nodes], [2])
        self.assertEqual(kept_lines, [])


if __name__ == "__main__":
    unittest.main()
