"""Dataset 2420 against the real pyuff reader (audit F01).

Every fixture is written with pyuff and parsed through ``UNVParser.parse()``,
so the key names, array shapes and validation path are those of the installed
reader, not of a hand-built dictionary.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyuff

from unv_to_json import UNVParseError, UNVParser


def _matrix(origin=(0.0, 0.0, 0.0)):
    return np.vstack([np.eye(3), np.asarray(origin, float).reshape(1, 3)])


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, def_cs, disp_cs, cs_sets=(), units=(1, 1.0), name="g.unv"):
        """cs_sets: iterable of (labels, types, origins) -> one 2420 block each."""
        n = len(def_cs)
        sets = [
            pyuff.prepare_164(
                units_code=units[0], units_description="test", temp_mode=1,
                length=units[1], force=1.0, temp=1.0, temp_offset=273.15,
            ),
            pyuff.prepare_2411(
                node_nums=np.arange(1, n + 1), def_cs=np.array(def_cs),
                disp_cs=np.array(disp_cs), color=np.zeros(n, int),
                x=np.arange(n, dtype=float), y=np.full(n, 90.0), z=np.zeros(n),
            ),
        ]
        for labels, types, origins in cs_sets:
            sets.append(pyuff.prepare_2420(
                Part_UID=1, Part_Name="p", CS_sys_labels=list(labels),
                CS_types=list(types), CS_colors=[8] * len(labels),
                CS_names=[f"cs{k}" for k in labels],
                CS_matrices=[_matrix(o) for o in origins],
            ))
        path = self.dir / name
        pyuff.UFF(str(path)).write_sets(sets, mode="overwrite")
        return path

    def parse(self, path, validate=True):
        return UNVParser(path, validate_cs=validate).parse()


class CoordinateSystemReadTests(_Base):
    def test_pyuff_dataset_2420_is_read_not_dropped(self):
        path = self.write([0, 0], [0, 0], [([1, 2], [0, 0], [(0, 0, 0), (0.5, 0, 0)])])
        data = self.parse(path)
        self.assertEqual([cs["id"] for cs in data["coordinateSystems"]], [1, 2])
        self.assertEqual(data["metadata"]["coordinateSystemCount"], 2)
        self.assertEqual({cs["type"] for cs in data["coordinateSystems"]}, {"rectangular"})

    def test_axes_and_origin_are_exposed(self):
        path = self.write([0], [0], [([3], [0], [(0.1, 0.2, 0.3)])])
        cs = self.parse(path)["coordinateSystems"][0]
        self.assertEqual(cs["xAxis"], [1.0, 0.0, 0.0])
        self.assertEqual(cs["zAxis"], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(cs["origin"], [0.1, 0.2, 0.3])

    def test_cs_origin_is_converted_to_metres_like_the_nodes(self):
        path = self.write([0], [0], [([3], [0], [(100.0, 0.0, 0.0)])], units=(2, 0.001))
        cs = self.parse(path)["coordinateSystems"][0]
        np.testing.assert_allclose(cs["origin"], [0.1, 0.0, 0.0])

    def test_multiple_2420_blocks_are_merged(self):
        path = self.write(
            [0], [2],
            [([1], [0], [(0, 0, 0)]), ([2], [0], [(0, 0, 0)])],
        )
        data = self.parse(path)
        self.assertEqual(sorted(cs["id"] for cs in data["coordinateSystems"]), [1, 2])


class CoordinateSystemRejectionTests(_Base):
    def test_cylindrical_definition_cs_is_rejected(self):
        path = self.write([7, 7], [0, 0], [([7], [1], [(0, 0, 0)])])
        for validate in (True, False):
            with self.assertRaises(UNVParseError, msg=f"validate={validate}"):
                self.parse(path, validate)

    def test_spherical_definition_cs_is_rejected(self):
        path = self.write([8, 8], [0, 0], [([8], [2], [(0, 0, 0)])])
        with self.assertRaises(UNVParseError):
            self.parse(path)

    def test_audit_case_cylindrical_cs7_with_undefined_displacement_cs99(self):
        # The reproduced F01 input: the old parser reported zero systems and
        # exposed an angular coordinate (90) as Cartesian y.
        path = self.write([7, 7], [99, 99], [([7], [1], [(0, 0, 0)])])
        with self.assertRaises(UNVParseError):
            self.parse(path)

    def test_multiple_definition_systems_are_rejected(self):
        path = self.write([0, 1], [0, 0], [([1], [0], [(0, 0, 0)])])
        with self.assertRaises(UNVParseError):
            self.parse(path)

    def test_undefined_displacement_cs_without_2420_is_rejected(self):
        path = self.write([0, 0], [99, 99])
        with self.assertRaises(UNVParseError):
            self.parse(path)

    def test_undefined_definition_cs_without_2420_is_rejected(self):
        path = self.write([5, 5], [0, 0])
        with self.assertRaises(UNVParseError):
            self.parse(path)

    def test_undefined_displacement_cs_with_other_systems_defined(self):
        path = self.write([0, 0], [99, 99], [([1], [0], [(0, 0, 0)])])
        with self.assertRaises(UNVParseError):
            self.parse(path)

    def test_defined_rectangular_systems_are_accepted(self):
        path = self.write([4, 4], [4, 4], [([4], [0], [(0, 0, 0)])])
        data = self.parse(path)
        self.assertEqual({n["exportCS"] for n in data["nodes"]}, {4})

    def test_global_frame_needs_no_2420(self):
        data = self.parse(self.write([0, 0], [0, 0]))
        self.assertEqual(data["coordinateSystems"], [])

    def test_legacy_key_dictionaries_still_parse(self):
        stub = self.dir / "stub.unv"
        stub.write_text("")
        parser = UNVParser(stub, validate_cs=False)
        cs = parser._parse_coord_systems({"cs_id": [2], "cs_type": [1]})
        self.assertEqual((cs[0]["id"], cs[0]["type"]), (2, "cylindrical"))


if __name__ == "__main__":
    unittest.main()
