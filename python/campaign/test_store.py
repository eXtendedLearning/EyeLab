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
        img = _photo(sheet, offsets, ppm=2.2, blur=0.6, noise=2.0, persp=0.03, seed=k)
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

            todo = St.pending(root)
            self.assertEqual(sorted(s.kind for s in todo), ["eye", "photo", "scan", "scan"])
            for s in todo:
                St.analyse(root, s)
            self.assertEqual(St.pending(root), [])          # analysed once per file version

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
            os.utime(p, (old + 5, old + 5))
            self.assertEqual([s.key for s in St.pending(root)], ["photos/IMG_1.jpg"])

    def test_sheet_name_in_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = St.ensure_layout(Path(tmp))
            (root / "scans" / "day1").mkdir()
            f = root / "scans" / "day1" / "P012_scan.png"
            f.write_bytes(b"")
            self.assertEqual(St.discover(root)[0].sheet_hint, "P012")


if __name__ == "__main__":
    unittest.main()
