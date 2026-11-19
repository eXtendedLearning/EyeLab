import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xreal_gates import b3a, beampro

BENCH = ["frame", "threads_req", "threads", "rep", "markers", "rejected", "board_markers", "pose_ok", "planar", "points",
         "clahe_ms", "detect_ms", "pose_ms", "total_ms", "r0", "r1", "r2", "t0", "t1", "t2", "rms_px", "amb_ratio", "amb_deg"]
CORN = ["frame", "id", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]


def _w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class B3aTest(unittest.TestCase):
    def test_set_and_bench_parity(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            K = np.array([[620.0, 0, 640], [0, 620, 360], [0, 0, 1]])
            calib = d / "calib.json"
            calib.write_text(json.dumps({"K": K.tolist(), "dist": [0, 0, 0, 0, 0]}), encoding="utf-8")
            set_dir = d / "set"
            s = b3a.make_set(set_dir, calib, b3a.DEFAULT_MARKERS, empty=1)
            self.assertEqual(s["ids"], [30, 31, 32])
            self.assertEqual(len(s["tuning"]), 14)
            py = b3a.PythonPipeline(s)
            name = s["frames"][0]                      # 0.35 m, facing the camera
            r = py.process(b3a.read_pgm(set_dir / name))
            self.assertEqual(sorted(r["ids"]), [30, 31, 32])
            truth = json.loads((set_dir / "truth.json").read_text(encoding="utf-8"))[name]
            self.assertLess(np.linalg.norm(r["pose"][1] - truth["tvec"]) * 1000, 2.0)

            # a device run that reports exactly what Python finds: parity must be exact
            run = d / "run"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({"probe": "detect", "plugin": "test"}), encoding="utf-8")
            rv, tv = r["pose"]
            row = [name, 1, 1, 0, 3, 0, 3, 1, 1, 12, 1.0, 5.0, 0.5, 6.5, *rv, *tv, 0.1, "", ""]
            row_n = [name, -1, 8, 0, 3, 0, 3, 1, 1, 12, 1.0, 3.0, 0.5, 4.5, *rv, *tv, 0.1, "", ""]
            _w(run / "bench.csv", BENCH, [row, row_n])
            _w(run / "bench_corners.csv", CORN, [[name, i, *r["corners"][i].reshape(-1)] for i in r["ids"]])
            lines = b3a.bench_report(run, set_dir)
            text = "\n".join(lines)
            self.assertIn("ids equal 1/1", text)
            self.assertIn("corners max 0.0000 px", text)
            self.assertIn("pose max 0.000 mm", text)
            self.assertIn("default (8)", text)
            self.assertIn("== run  detect", "\n".join(beampro.report(run)))


    def test_bottom_up_eye_frames(self):
        """Real Eye frames come bottom-up: the set flips them upright, and the live report warns when a run did not."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            K = np.array([[620.0, 0, 640], [0, 620, 360], [0, 0, 1]])
            calib = d / "calib.json"
            calib.write_text(json.dumps({"K": K.tolist(), "dist": [0, 0, 0, 0, 0]}), encoding="utf-8")
            ids, corners = b3a.board()
            upright = b3a.render(K, np.zeros(5), ids, corners, np.array([np.pi, 0, 0]), np.array([0.0, 0.0, 0.4]), seed=7)
            stored = upright[::-1].copy()
            self.assertEqual(b3a._detect_ids(stored, ids), [])          # mirrored markers do not decode
            raw = d / "raw"
            b3a.write_pgm(raw / "y_000010.pgm", stored)
            s = b3a.make_set(d / "set", calib, b3a.DEFAULT_MARKERS, empty=0, eye_plate=raw, bottom_up=True)
            self.assertIn("frames/eyeplate_y_000010.pgm", s["frames"])
            self.assertEqual(b3a._detect_ids(b3a.read_pgm(d / "set/frames/eyeplate_y_000010.pgm"), ids), [30, 31, 32])

            run = d / "run"
            b3a.write_pgm(run / "frames/y_000010.pgm", stored)
            live = ("seq,step,threads,cam_ts,arrival_mono_ns,start_mono_ns,end_mono_ns,skipped,markers,rejected,board_markers,"
                    "pose_ok,planar,points,clahe_ms,detect_ms,pose_ms,total_ms,r0,r1,r2,t0,t1,t2,rms_px,amb_ratio,amb_deg,ids,saved\n"
                    "10,t2_still,2,1000000,96000000,97000000,140000000,0,0,50,0,0,0,0,3,40,0,43,,,,,,,,,,,frames/y_000010.pgm\n")
            (run / "live.csv").write_text(live, encoding="utf-8")
            (run / "frames.csv").write_text("step,threads,unity_frame,dt_ms,present,mono_ns,eye_ms\n"
                                            "t2_still,2,1,16.6,1,1,0.8\n", encoding="utf-8")
            text = "\n".join(b3a.live_report(run, d / "set"))
            self.assertIn("WARNING", text)
            self.assertIn("main-thread Eye work", text)


if __name__ == "__main__":
    unittest.main()
