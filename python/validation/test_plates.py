import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from registration import read_marker_config
from unv_to_json import UNVParser
from validation.plates import build_target, layout_from_provenance, plate_a, plate_l
from validation.render import render_page
from validation.target import load_target, roll_for_axes, save_target

MARGIN_MM = 8.0


def _boxes(layout):
    s = layout.square_mm
    for sd in layout.saddles:
        yield ("saddle", sd.id, sd.x - s, sd.y - s, sd.x + s, sd.y + s)
    for m in layout.markers:
        q = m.size_mm / 6  # one-cell quiet zone
        h = m.size_mm / 2 + q
        yield ("marker", m.id, m.x - h, m.y - h, m.x + h, m.y + h)


class TestLayouts(unittest.TestCase):
    def test_features_inside_printable_area_and_disjoint(self):
        for lay in (plate_a(), plate_l()):
            boxes = list(_boxes(lay))
            W, H = lay.page_mm
            for b in boxes:
                self.assertGreaterEqual(min(b[2], b[3], W - b[4], H - b[5]), MARGIN_MM, (lay.name, b))
            for i, a in enumerate(boxes):
                for b in boxes[i + 1:]:
                    overlap = not (a[4] <= b[2] or b[4] <= a[2] or a[5] <= b[3] or b[5] <= a[3])
                    self.assertFalse(overlap, (lay.name, a, b))
            if lay.fold_y is not None:
                for b in boxes:
                    self.assertTrue(b[5] < lay.fold_y - 5 or b[3] > lay.fold_y + 5, b)

    def test_printed_markers_sit_at_layout_positions(self):
        ppm = 600 / 25.4
        for lay in (plate_a(), plate_l()):
            img = render_page(lay, ppm)
            det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
            corners, ids, _ = det.detectMarkers(img)
            got = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
            self.assertEqual(sorted(got), sorted(m.id for m in lay.markers))
            for m in lay.markers:
                # OpenCV pixel centres are at integers: an edge at pixel boundary b is at b - 0.5
                ctr = (got[m.id].mean(axis=0) + 0.5) / ppm
                np.testing.assert_allclose(ctr, [m.x, m.y], atol=0.03)
                side = np.linalg.norm(got[m.id][1] - got[m.id][0]) / ppm
                self.assertAlmostEqual(side, m.size_mm, delta=0.1)

    def test_marker_orientation_matches_eyelab_corner_order(self):
        # Corner TL of each marker must be at page top-left of its square on every face.
        for lay in (plate_a(), plate_l()):
            t = build_target(lay)
            for m, pm in zip(t.markers, lay.markers):
                uv = t.to_face(m.face, m.corners()) * 1000.0
                self.assertLess(uv[0, 0], uv[1, 0])       # TL left of TR
                self.assertGreater(uv[0, 1], uv[3, 1])    # TL above BL (face v is page-up)
                self.assertAlmostEqual(roll_for_axes(m.normal, t.faces[m.face].ex), 0.0, places=9)

    def test_fold_geometry(self):
        t = build_target(plate_l())
        np.testing.assert_allclose(t.faces["V"].normal, [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(t.faces["H"].normal, [0, 0, 1], atol=1e-12)
        self.assertTrue(all(t.points[i][2] > 0.02 for i in t.face_ids("V")))
        self.assertTrue(all(t.points[i][1] < -0.02 for i in t.face_ids("H")))
        t80 = build_target(plate_l(), fold_interior_deg=80.0)
        n = t80.faces["V"].normal
        self.assertAlmostEqual(np.degrees(np.arccos(np.dot(n, [0, 0, 1]))), 180 - 80, places=6)
        # face coordinates are print-only ground truth: unchanged by the fold angle
        np.testing.assert_allclose(t80.face_coords(t80.ids()), t.face_coords(t.ids()), atol=1e-12)


class TestExports(unittest.TestCase):
    def test_measured_scale_and_derived_files(self):
        lay = plate_a()
        nominal = build_target(lay)
        t = build_target(lay, 1.002, 0.999, uncertainty_mm=0.1, measured=True)
        a, b = nominal.array(nominal.ids()), t.array(t.ids())
        np.testing.assert_allclose(b[:, 0], a[:, 0] * 1.002, atol=1e-12)
        np.testing.assert_allclose(b[:, 1], a[:, 1] * 0.999, atol=1e-12)
        with tempfile.TemporaryDirectory() as d:
            save_target(t, Path(d) / "t.json")
            t.export_marker_config(Path(d) / "m.json")
            t.export_unv(Path(d) / "g.unv")
            t2 = load_target(Path(d) / "t.json")
            self.assertEqual(layout_from_provenance(t2.provenance).name, "A")
            np.testing.assert_allclose(t2.array(t2.ids()), b)
            cfg = read_marker_config(str(Path(d) / "m.json"))
            self.assertEqual([c.marker_id for c in cfg.markers], [30, 31, 32])
            for c, m in zip(cfg.markers, t.markers):
                np.testing.assert_allclose(c.unv_position, m.center)       # metres
                self.assertAlmostEqual(c.marker_size_mm, m.size_m * 1000.0)
            geo = UNVParser(Path(d) / "g.unv").parse()
            nodes = {n["id"]: [n["x"], n["y"], n["z"]] for n in geo["nodes"]}
            for i in t.ids():
                np.testing.assert_allclose(nodes[int(i)], t.points[i], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
