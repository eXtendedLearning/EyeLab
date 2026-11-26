"""The campaign data folder: discovery, analysis once per file version, Eye estimate from a trial run, overview."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from campaign import sheets as S
from campaign import store as St
from campaign.test_hits import _photo


def _offsets(n, seed=3):
    return [tuple(v) for v in np.random.default_rng(seed).uniform(-4, 4, (n, 2))]


def _fake_run(root: Path, sheet, offsets, n_frames=4):
    """A Beam Pro trial as pulled: run.json, live.csv with review-look rows, Eye-like PGM frames."""
    d = root / "runs" / "BeamProRuns" / "20261126_120000_mark"
    (d / "frames").mkdir(parents=True)
    rows = ["seq,step,markers,saved"]
    for k in range(n_frames):
        img = _photo(sheet, offsets, ppm=2.6, blur=0.6, noise=2.0, persp=0.03, seed=k)   # ~0.38 mm/px
        name = f"frames/y_{k:06d}.pgm"
        cv2.imwrite(str(d / name), img)
        rows.append(f"{k},review,5,{name}")
    rows.append("99,tgt_01,5,")
    (d / "live.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    meta = {"variant": "mark", "campaign_sheet": f"{sheet.name} digest {S.digest(sheet)}",
            "campaign_result": f"{sheet.name},P01,A,all,0.5-default,20261126_120000_mark,x,12,13,1,0,2026-11-26 12:00:00"}
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    old = time.time() - 10
    for p in [d / "live.csv", d / "run.json"]:
        os.utime(p, (old, old))
    return d


class TestStore(unittest.TestCase):
    def setUp(self):
        # synthetic frames are not from the Beam Pro Eye: do not undistort them with its calibration
        self._cam = St.EYE_CAMERA
        St.EYE_CAMERA = Path("no_such_camera.json")

    def tearDown(self):
        St.EYE_CAMERA = self._cam

    def test_far_review_look_is_not_a_hit_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = St.ensure_layout(Path(tmp))
            m = S.sheet("M", 45)
            d = _fake_run(root, m, _offsets(len(m.targets)), n_frames=2)
            for f in (d / "frames").iterdir():   # as at 48 cm: about 0.75 mm/px
                cv2.imwrite(str(f), _photo(m, _offsets(len(m.targets)), ppm=1.35, blur=0.6, noise=2.0, persp=0.03))
            r = St.analyse(root, St.pending(root, settle_s=0)[0])
            self.assertTrue(r.get("too_far"), r)
            res = St.load_results(root)
            self.assertEqual(St.pooled(root, res, St.trials(root)), [])
            self.assertTrue(St.by_sheet(res, St.trials(root))[0]["eye"]["too_far"])
    def test_folder_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = St.ensure_layout(Path(tmp) / "data")
            for sub in St.SUBDIRS:
                self.assertTrue((root / sub).is_dir())
            m = S.sheet("M", 43)
            offs = _offsets(len(m.targets))
            cv2.imwrite(str(root / "scans" / "scan_001.png"), _photo(m, offs, ppm=600 / 25.4 / 2, persp=0.0,
                                                                    blur=0.5, noise=1.0))
            cv2.imwrite(str(root / "photos" / "IMG_1.jpg"), _photo(m, offs))
            (root / "scans" / "scan_002.pdf").write_bytes(b"%PDF-1.4")
            _fake_run(root, m, offs)
            old = time.time() - 10
            for p in (root / "scans").iterdir():
                os.utime(p, (old, old))
            for p in (root / "photos").iterdir():
                os.utime(p, (old, old))

            self.assertEqual(St.pending(root), [])          # just arrived (ctime is now): still settling
            todo = St.pending(root, settle_s=0)
            self.assertEqual(sorted(s.kind for s in todo), ["eye", "photo", "scan", "scan"])
            for s in todo:
                St.analyse(root, s)
            self.assertEqual(St.pending(root, settle_s=0), [])          # analysed once per file version

            res = {r["key"]: r for r in St.load_results(root)}
            scan = res["scans/scan_001.png"]
            self.assertEqual((scan["sheet"], scan["found"], scan["targets"]), ("M043", 12, 12))
            self.assertLess(scan["fit_rms_mm"], 0.25)   # synthetic render rounds features to whole pixels; real prints 0.013
            self.assertIn("PDF", res["scans/scan_002.pdf"]["error"])
            eye = next(r for r in res.values() if r["kind"] == "eye")
            self.assertEqual(eye["sheet"], "M043")
            self.assertEqual(eye["frames"], 4)
            self.assertGreaterEqual(eye["found"], 10)
            # the error vectors come back against the drawn offsets (stats are against the nominal points)
            import csv

            def worst(r):
                with open(Path(r["_json"]).with_suffix(".csv"), newline="", encoding="utf-8") as f:
                    got = [(h["dx_mm"], h["dy_mm"]) for h in csv.DictReader(f)]
                return max(np.hypot(float(g[0]) - o[0], float(g[1]) - o[1]) for g, o in zip(got, offs) if g[0])
            self.assertLess(worst(scan), 0.12)
            self.assertLess(worst(eye), 0.6)   # Eye-like 0.45 mm/px, median of 4 frames

            trials = St.trials(root)
            self.assertEqual(trials["M043"]["participant"], "P01")
            rows = St.by_sheet(list(res.values()), trials)
            self.assertEqual([r["sheet"] for r in rows], ["M043"])
            self.assertEqual(set(k for k in ("scan", "photo", "eye") if k in rows[0]), {"scan", "photo", "eye"})
            pooled = St.pooled(root, list(res.values()), trials)
            self.assertEqual({(g["participant"], g["condition"], g["kind"]) for g in pooled},
                             {("P01", "A", "scan"), ("P01", "A", "photo"), ("P01", "A", "eye")})

            # a changed file is analysed again
            p = root / "photos" / "IMG_1.jpg"
            cv2.imwrite(str(p), _photo(m, offs, seed=9))
            self.assertEqual([s.key for s in St.pending(root, settle_s=0)], ["photos/IMG_1.jpg"])

    def test_run_pulled_before_its_frames_is_analysed_again(self):
        # adb pull keeps the device's (old) mtimes and live.csv can land before frames/ is complete
        with tempfile.TemporaryDirectory() as tmp:
            root = St.ensure_layout(Path(tmp))
            m = S.sheet("M", 44)
            offs = _offsets(len(m.targets))
            d = _fake_run(root, m, offs, n_frames=3)
            frames = sorted((d / "frames").iterdir())
            hold = Path(tmp) / "hold"
            hold.mkdir()
            for f in frames:
                f.rename(hold / f.name)
            src = St.pending(root, settle_s=0)[0]
            self.assertIn("no review-look frames", St.analyse(root, src)["error"])
            self.assertEqual(St.pending(root, settle_s=0), [])
            for f in hold.iterdir():
                f.rename(d / "frames" / f.name)
            again = St.pending(root, settle_s=0)
            self.assertEqual([s.key for s in again], [src.key])
            self.assertNotIn("error", St.analyse(root, again[0]))

    def test_sheet_name_in_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = St.ensure_layout(Path(tmp))
            (root / "scans" / "day1").mkdir()
            f = root / "scans" / "day1" / "P012_scan.png"
            f.write_bytes(b"")
            self.assertEqual(St.discover(root)[0].sheet_hint, "P012")


if __name__ == "__main__":
    unittest.main()
