import tempfile
import unittest
from pathlib import Path

from unv_to_json import UNVParser, UNVParseError


class _ParserFixture(unittest.TestCase):
    """UNVParser.__init__ requires an existing file; the parsing helpers under
    test operate on already-loaded pyuff dataset dicts, so an empty stub file
    is enough to construct the parser."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        stub = Path(self._tmp.name) / "stub.unv"
        stub.write_text("")
        self.parser = UNVParser(stub, validate_cs=False)

    def tearDown(self):
        self._tmp.cleanup()


class NodeParsingTests(_ParserFixture):
    def test_parses_ids_and_coordinates(self):
        ds = {
            "node_nums": [1, 2, 3],
            "x": [0.0, 1.5, 2.0],
            "y": [0.0, 0.0, 3.0],
            "z": [0.0, 0.0, 0.0],
            "coord_sys": [0, 0, 0],
            "disp_coord_sys": [0, 0, 0],
        }
        nodes = self.parser._parse_nodes(ds)
        self.assertEqual([n["id"] for n in nodes], [1, 2, 3])
        self.assertEqual(nodes[1]["x"], 1.5)
        self.assertEqual(nodes[2]["y"], 3.0)

    def test_empty_nodes_raises(self):
        with self.assertRaises(UNVParseError):
            self.parser._parse_nodes({"node_nums": []})


class TraceLineParsingTests(_ParserFixture):
    def test_pen_up_zero_splits_into_segments(self):
        # 0 is a "pen-up" break: [1,2,3, 0, 4,5] -> two disjoint polylines.
        edges = self.parser._parse_trace_lines({"nodes": [1, 2, 3, 0, 4, 5]})
        self.assertEqual(edges, [[1, 2], [2, 3], [4, 5]])

    def test_node_nums_fallback_when_nodes_absent(self):
        edges = self.parser._parse_trace_lines({"node_nums": [7, 8]})
        self.assertEqual(edges, [[7, 8]])

    def test_repeated_node_does_not_make_self_edge(self):
        edges = self.parser._parse_trace_lines({"nodes": [1, 1, 2]})
        self.assertEqual(edges, [[1, 2]])

    def test_empty_sequence_returns_no_edges(self):
        self.assertEqual(self.parser._parse_trace_lines({"nodes": []}), [])


if __name__ == "__main__":
    unittest.main()
