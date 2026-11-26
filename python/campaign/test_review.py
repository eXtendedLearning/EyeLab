"""Manual review: a dirt fleck taken for a hit is rejected, a faint dot the detector missed is pinned, and the review
survives re-analysis of the same file but not a new version of it."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from campaign import review as Rv
from campaign import sheets as S
from campaign import store as St

PPM = 600 / 25.4 / 2   # a 300 dpi scan


def _scan(sh, dots, path):
    """dots: list of (x_mm, y_mm, grey level, diameter mm) drawn on the rendered page, scanner-like blur and noise."""
    page = S.render(sh, PPM).astype(np.float32)
    for x, y, g, d in dots:
        cv2.circle(page, (int(round(x * PPM * 16)), int(round(y * PPM * 16))), int(round(d / 2 * PPM * 16)), g, -1,
                   cv2.LINE_AA, 4)
    page = cv2.GaussianBlur(page, (0, 0), 0.6) + np.random.default_rng(1).normal(0, 1.5, page.shape)
    cv2.imwrite(str(path), np.clip(page, 0, 255).astype(np.uint8))
    old = time.time() - 30
    os.utime(path, (old, old))


class TestReview(unittest.TestCase):
    def test_reject_pin_and_versions(self):
        sh = S.sheet("M", 44)
        T = {t.n: (t.x, t.y) for t in sh.targets}
        dots = [(T[n][0] + 0.8, T[n][1] - 0.5, 30, 1.0) for n in T if n not in (1, 2)]
        faint = (T[1][0] - 1.2, T[1][1] + 0.7)
        dots.append((faint[0], faint[1], 200, 1.0))           # too faint for the automatic rule
        dirt = (T[2][0] + 4.0, T[2][1] + 3.0)
        dots.append((dirt[0], dirt[1], 60, 0.8))              # a fleck the detector takes for target 2's dot
        with tempfile.TemporaryDirectory() as tmp:
            root = St.ensure_layout(Path(tmp))
            img = root / "scans" / "M044_scan.png"
            _scan(sh, dots, img)
            src = St.discover(root)[0]
            r0 = St.analyse(root, src)
            self.assertNotIn("error", r0, r0.get("error"))
            self.assertIn(1, r0["missing"])
            self.assertNotIn(2, r0["missing"])                # the fleck was taken for a hit

            rj = St.result_path(root, src)
            from campaign import hits as Hm
            _, rect = Hm.analyse(img, "M044")
            px, py = Rv.refine_click(rect, faint[0] + 0.5, faint[1] - 0.4, 10.0)   # a click half a mm off
            self.assertLess(np.hypot(px - faint[0], py - faint[1]), 0.15)
            Rv.save(rj, src.key, src.version, "M044", {1: {"status": "pinned", "hit": (px, py)},
                                                       2: {"status": "absent", "hit": None},
                                                       3: {"status": "confirmed", "hit": (T[3][0] + 0.8,
                                                                                          T[3][1] - 0.5)}},
                    note="fleck at 2")
            r1 = St.analyse(root, src)
            self.assertEqual(r1["manual"]["pinned"], 1)
            self.assertEqual(r1["manual"]["absent"], 1)
            self.assertEqual(r1["manual"]["rejected_dots"], 1)
            self.assertEqual(r1["missing"], [2])
            self.assertEqual(r1["found"], 11)
            self.assertEqual(r1["extra_dots"], 0)            # the rejected fleck is not an extra dot either
            # the review file is not taken for a result
            self.assertEqual(len(St.load_results(root)), 1)

            # a new version of the file (rescan) does not inherit the review
            time.sleep(1.1)
            _scan(sh, dots, img)
            os.utime(img, None)                                # a rescan arrives now
            src2 = St.discover(root)[0]
            self.assertNotEqual(src2.version, src.version)
            r2 = St.analyse(root, src2)
            self.assertTrue(r2["manual"]["stale"])
            self.assertIn(1, r2["missing"])
            self.assertTrue(any("older version" in n for n in r2["notes"]))
            rv = json.loads(Rv.review_path(rj).read_text(encoding="utf-8"))
            self.assertEqual(rv["note"], "fleck at 2")

    def test_review_path(self):
        self.assertEqual(Rv.review_path(Path("r/scans_a.png.json")).name, "scans_a.png.review.json")


if __name__ == "__main__":
    unittest.main()
