"""Synthetic photos of campaign sheets: known dot offsets must come back to well under 0.1 mm."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from campaign import hits as Hm
from campaign import sheets as S


def _photo(sh, offsets, ppm=300 / 25.4, dot_mm=1.0, seed=0, persp=0.06, blur=0.8, noise=3.0):
    """Render, draw dots at target + offset (model frame mm: x right, y up), warp like a hand-held photo."""
    page = S.render(sh, ppm).astype(np.float32)
    for t, (dx, dy) in zip(sh.targets, offsets):
        c = ((t.x + dx) * ppm, (t.y - dy) * ppm)
        cv2.circle(page, (int(round(c[0] * 16)), int(round(c[1] * 16))), int(round(dot_mm / 2 * ppm * 16)), 40,
                   -1, cv2.LINE_AA, 4)
    h, w = page.shape
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-persp, persp, (4, 2)).astype(np.float32) * np.float32([w, h])
    dst -= dst.min(axis=0) - 40
    Hw = cv2.getPerspectiveTransform(src, dst)
    size = (int(dst[:, 0].max() + 40), int(dst[:, 1].max() + 40))
    img = cv2.warpPerspective(page, Hw, size, borderValue=200)
    img = cv2.GaussianBlur(img, (0, 0), blur) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestHits(unittest.TestCase):
    def _run(self, sh, **kw):
        rng = np.random.default_rng(5)
        offs = [tuple(v) for v in rng.uniform(-6, 6, (len(sh.targets), 2))]
        img = _photo(sh, offs, **kw)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "photo.png"
            cv2.imwrite(str(p), img)
            res, _ = Hm.analyse(p)
        self.assertEqual(res.sheet, sh.name)
        self.assertFalse(res.extra_dots, res.extra_dots)
        errs = []
        for h, o in zip(res.hits, offs):
            self.assertIsNotNone(h.hit, h.n)
            errs.append(np.hypot(h.err[0] - o[0], h.err[1] - o[1]))
        return max(errs), res

    def test_m_sheet_photo(self):
        worst, res = self._run(S.sheet("M", 41))
        self.assertLess(worst, 0.12, res.fit_rms_mm)

    def test_p_sheet_photo_with_printed_crosses(self):
        sh = S.sheet("P", 11)
        img_offs = [(0.3, -0.2)] * len(sh.targets)   # small offsets: dots inside the open centres
        img = _photo(sh, img_offs)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "photo.png"
            cv2.imwrite(str(p), img)
            res, _ = Hm.analyse(p)
        for h in res.hits:
            self.assertIsNotNone(h.hit, h.n)
            self.assertLess(np.hypot(h.err[0] - 0.3, h.err[1] + 0.2), 0.1)

    def test_p_sheet_printed_in_another_font_gives_no_false_hits(self):
        # 0.5.000's PDFs were rendered by OpenCV 5, whose text is wider than 4.x's Hershey: the printed numbers must
        # not come back as dots (they did: 8 false hits on a clean P011 under 4.13).
        sh = S.sheet("P", 11)
        orig = S._text_scale
        S._text_scale = lambda h, ppm: (orig(h, ppm)[0] * 1.3, orig(h, ppm)[1] + 2)
        try:
            img = _photo(sh, [])
        finally:
            S._text_scale = orig
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "clean.png"
            cv2.imwrite(str(p), img)
            res, _ = Hm.analyse(p)
        self.assertEqual([h.n for h in res.hits if h.hit is not None], [])
        self.assertEqual(res.extra_dots, [])

    def test_eye_like_low_resolution(self):
        # Eye at about 30 cm: ~0.5 mm/px; a 1 mm dot is 2 px. Expect a first estimate, not the final number.
        worst, res = self._run(S.sheet("M", 42), ppm=2.0, blur=0.6, noise=2.0, persp=0.03)
        self.assertLess(worst, 0.6, res.fit_rms_mm)

    def test_digest_guard(self):
        self.assertEqual(Hm.load_sheet("M041").name, "M041")


if __name__ == "__main__":
    unittest.main()
