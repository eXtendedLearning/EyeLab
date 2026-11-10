"""T5.1 Phase 1: per-stage latency table, timecode pairing, the HMD clock check, and the timecode target."""

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from xreal_gates.stages import STAGES, align_clocks, hmd_clock, report, stage_table, timecode_latency
from xreal_gates.timecode import TimecodeTarget, draw_time, frame_for, index_draws, load_draws

COLS = ["seq", "cam_ts", "hmd_arrived_ns", "q_frame_prev", "q_frame", "q_arrived", "q_copy0", "q_copy1", "q_send0",
        "q_send1", "q_recv", "q_det0", "q_det1", "q_reply", "q_reply_recv", "q_applied", "q_shown", "age_ms",
        "queue_ms", "hold_ms", "base_ms", "ok", "hmd_send_ns", "native_cap", "native_now", "unity_head_now"]

# One frame's path, ms after its arrival: hold 10, copy 2, send 1, transport 1, parse 2, detect 20, reply 3,
# apply wait 8, display 16.
STEP_MS = {"q_frame": -12, "q_arrived": 0, "q_copy0": 10, "q_copy1": 12, "q_send0": 12, "q_send1": 13, "q_recv": 14,
           "q_det0": 16, "q_det1": 36, "q_reply": 37, "q_reply_recv": 39, "q_applied": 47, "q_shown": 63}


def _rows(n=20):
    rows = []
    for i in range(n):
        t = 100.0 + i * 0.125
        r = {c: "" for c in COLS}
        r.update({k: f"{t + v / 1000:.6f}" for k, v in STEP_MS.items()})
        r.update(seq=str(i + 1), cam_ts=str(int(t * 1e9)), q_frame_prev=f"{t - 0.029:.6f}", age_ms="120",
                 queue_ms="5", hold_ms="10", base_ms="105", ok="1")
        rows.append(r)
    return rows


def _write(path: Path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


class StageTableTest(unittest.TestCase):
    def test_stage_medians(self):
        table = dict(stage_table(_rows()))
        expect = {"hold": 10, "copy": 2, "send": 1, "to service": 1, "parse": 2, "detection": 20, "reply": 3,
                  "apply wait": 8, "display": 16}
        for name, ms in expect.items():
            self.assertAlmostEqual(table[name][0], ms, delta=0.01, msg=name)
        self.assertEqual(len([s for s in STAGES]), 10)

    def test_report_runs_on_a_run_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "20261110_101500_G5_g5_plateA_3dof"
            run.mkdir(parents=True)
            _write(run / "frame_timing.csv", _rows(), COLS)
            lines = report(run)
            self.assertTrue(lines[0].endswith("QPC clock"))
            self.assertTrue(any("detection" in ln for ln in lines))
            self.assertTrue(any("HMD clock: not logged" in ln for ln in lines))


class ClockAlignTest(unittest.TestCase):
    def test_process_relative_unity_clock_is_shifted(self):
        # 0.3.008: Unity's q_* count from process start; hmd_arrived_ns (QPC since boot) at the same arrival.
        rows = _rows()
        off = 1_654_890.825
        for r in rows:
            r["hmd_arrived_ns"] = str(int(round((float(r["q_arrived"]) + off) * 1e9)))
            r["cam_ts"] = str(int(round((float(r["q_arrived"]) + off - 0.150) * 1e9)))
            for k in ("q_recv", "q_det0", "q_det1"):   # the service already logs absolute QPC
                r[k] = f"{float(r[k]) + off:.6f}"
        shift = align_clocks(rows)
        self.assertAlmostEqual(shift, off, places=4)
        table = dict(stage_table(rows))
        self.assertAlmostEqual(table["to service"][0], 1, delta=0.05)
        self.assertAlmostEqual(table["cam_ts -> arrival (HMD clock, exact)"][0], 150, delta=0.05)
        self.assertEqual(align_clocks(rows), 0.0)          # already absolute: no second shift


class TimecodeTest(unittest.TestCase):
    def test_draw_matching_and_latency(self):
        # Draws every 1/60 s from t=99.0; ids wrap at 1000. Frame i shows the draw made 70 ms before its arrival.
        draws = [(99.0 + k / 60, 99.0 + k / 60 + 0.002, k % 1000) for k in range(1200)]
        idx = index_draws(draws)
        rows = _rows()
        service = []
        for r in rows:
            t_arr = float(r["q_arrived"])
            k = int(np.floor((t_arr - 0.070 - 99.0) * 60 + 1e-9))
            service.append({"seq": r["seq"], "target": "timecode", "marker_ids": str(k % 1000)})
        out = timecode_latency(rows, service, draws)
        med = out["draw -> arrival"][0]
        self.assertGreaterEqual(med, 70.0)
        self.assertLess(med, 70.0 + 1000 / 60 + 0.01)
        self.assertEqual(out["frames read"][2], len(rows))
        # The id that wrapped (k and k+1000 share it) resolves to the latest draw before arrival.
        self.assertAlmostEqual(draw_time(draws, idx, 5, 99.0 + 1005 / 60 + 0.01)[0], 99.0 + 1005 / 60, places=6)

    def test_load_draws_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "timecode_20261110_101455.csv"
            p.write_text("qpc_draw,qpc_shown,counter,id\n2.0,2.001,1,1\n1.0,1.001,0,0\n", encoding="utf-8")
            self.assertEqual([d[2] for d in load_draws([p])], [0, 1])

    def test_target_reads_the_id(self):
        img = frame_for(1523, 640, 480)                     # id 523
        est = TimecodeTarget().estimate(img, np.eye(3), np.zeros(5))
        self.assertFalse(est.ok)
        self.assertEqual(est.marker_ids, [523])
        self.assertEqual(TimecodeTarget().overlay(), {"segments": [], "points": []})
        blank = np.full((480, 640), 255, np.uint8)
        self.assertEqual(TimecodeTarget().estimate(blank, np.eye(3), np.zeros(5)).marker_ids, [])


class HmdClockTest(unittest.TestCase):
    def test_same_clock_detected(self):
        cam = [{"cam_ts": str(int(10e9 + i * 33_333_333)), "hmd_ns": str(int(10e9 + i * 33_333_333 + 95e6 + (i % 5) * 4e6))}
               for i in range(60)]
        hc = hmd_clock(cam)
        self.assertTrue(hc["same_clock"])
        self.assertAlmostEqual(hc["min_s"], 0.095, places=3)
        self.assertIsNone(hmd_clock([{"cam_ts": "1", "hmd_ns": "0"}]))


if __name__ == "__main__":
    unittest.main()
