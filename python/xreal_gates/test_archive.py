import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from xreal_gates.archive import import_folder, import_zip, match_log
from xreal_gates.summary import summarize


def _make_run(base: Path, name: str, payload: str = "x") -> Path:
    run = base / name
    (run / "snapshots").mkdir(parents=True)
    (run / "events.csv").write_text(
        "t_unity,frame,event,result,detail\n"
        "1.0,1,gate_start,G3,\n"
        "2.0,2,tracking_ready,WAIT,\n"
        "4.5,3,tracking_ready,READY,\n"
        f"10.0,4,gate_result,COMPLETED,{payload}\n", encoding="utf-8")
    (run / "camera.csv").write_text(
        "t_unity,frame,cam_ts,res_w,res_h,y_mean,y_mad_prev,dup_ts,drained,source\n"
        + "".join(f"{5 + i / 30:.4f},{i},{1_000_000_000 + i * 33_333_333},1280,720,80,,0,,vendor\n" for i in range(150)),
        encoding="utf-8")
    (run / "perf.csv").write_text(
        "t_unity,frame,unity_fps,cpu_process_pct,cpu_cores_busy,cpu_system_utility_pct,cpu_performance_pct,"
        "gc_mb,cam_fps,cam_lag_growth_s,tracking_reason,step\n"
        "5,1,60,10,1,20,200,1,30,0,0,static\n6,2,60,10,1,20,60,1,30,0,0,static\n", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"args": {"gate": "G3", "label": "g3c"},
                                              "build": {"utc_build": "2026-10-14T09:00:00Z"}}), encoding="utf-8")
    return run


class MatchLogTest(unittest.TestCase):
    def test_same_label_within_window(self):
        runs = ["20261014_105312_G3_g3c", "20261014_105438_G3_g3c_drain"]
        self.assertEqual(match_log("player_g3c_20261014_105309.log", runs), "20261014_105312_G3_g3c")
        self.assertEqual(match_log("player_g3c_drain_20261014_105435.log", runs), "20261014_105438_G3_g3c_drain")

    def test_rejects_other_label_or_far_time(self):
        runs = ["20261014_105312_G3_g3c"]
        self.assertIsNone(match_log("player_g2_20261014_105309.log", runs))
        self.assertIsNone(match_log("player_g3c_20261014_100000.log", runs))
        self.assertIsNone(match_log("player_vendor_20261014_105309.log", runs))


class ImportTest(unittest.TestCase):
    def test_cumulative_zips_are_deduplicated_and_logs_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            export = tmp / "export"
            _make_run(export, "20261014_105312_G3_g3c")
            (export / "player_g3c_20261014_105309.log").write_text("log", encoding="utf-8")
            (export / "player_vendor_20261014_110000.log").write_text("vendor", encoding="utf-8")
            zpath = tmp / "GateRuns_CODY_20261014_110003.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                for p in export.rglob("*"):
                    z.write(p, p.relative_to(export))
            root = tmp / "GateRuns"

            first = import_zip(zpath, root, move_to_imported=False)
            self.assertEqual(first.added, ["20261014_105312_G3_g3c"])
            self.assertTrue((root / "runs" / "20261014_105312_G3_g3c" / "player.log").exists())
            self.assertEqual(first.logs_unmatched, ["player_vendor_20261014_110000.log"])

            second = import_zip(zpath, root)  # same cumulative content again
            self.assertEqual(second.skipped, ["20261014_105312_G3_g3c"])
            self.assertEqual(second.added, [])
            self.assertTrue((root / "_imported" / zpath.name).exists())
            self.assertFalse(zpath.exists())

    def test_larger_copy_replaces_and_keeps_player_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = tmp / "GateRuns"
            _make_run(tmp / "a", "20261014_105312_G3_g3c")
            (tmp / "a" / "player_g3c_20261014_105309.log").write_text("log", encoding="utf-8")
            import_folder(tmp / "a", root)
            _make_run(tmp / "b", "20261014_105312_G3_g3c", payload="x" * 5000)
            report = import_folder(tmp / "b", root)
            self.assertEqual(report.replaced, ["20261014_105312_G3_g3c"])
            self.assertEqual((root / "runs" / "20261014_105312_G3_g3c" / "player.log").read_text(encoding="utf-8"), "log")


class SummaryTest(unittest.TestCase):
    def test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "20261014_105312_G3_g3c")
            s = summarize(run)
            self.assertEqual(s.result, "COMPLETED")
            self.assertEqual(s.camera_frames, 150)
            self.assertAlmostEqual(s.camera_fps_median, 30, delta=1)
            self.assertAlmostEqual(s.lag_end_s, 0.0, places=2)
            self.assertAlmostEqual(s.tracking_ready_after_s, 2.5)
            self.assertEqual(s.clock_min_pct, 60)


if __name__ == "__main__":
    unittest.main()
