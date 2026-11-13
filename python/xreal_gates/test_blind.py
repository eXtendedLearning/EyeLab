"""Blind time per block kind and camera health (``python -m xreal_gates blind``)."""

import csv
import tempfile
import unittest
from pathlib import Path

from xreal_gates.blind import blind_fractions, camera_health, kind_of, report


def _ev(t, event, result="", detail=""):
    return {"t_unity": str(t), "frame": "0", "event": event, "result": result, "detail": detail}


def _lock(t):
    return _ev(t, "lock", "UPDATE", "target=plateA seq=1 world=0,0,0")


class BlindTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(kind_of("cmp1_euro_still"), "still")
        self.assertEqual(kind_of("cmp2_latest_move"), "move")
        self.assertEqual(kind_of("cmp3_hysteresis_rate"), "rate")
        self.assertEqual(kind_of("static"), "still")
        self.assertIsNone(kind_of("align"))

    def test_gaps_over_one_second_count(self):
        ev = [_ev(0, "step", "cmp1_euro_still"), _ev(20, "step", "cmp1_euro_move"), _ev(30, "end")]
        # still: locks every 0.5 s except a 3 s hole at 10-13 s -> 3 s blind of 20.
        ev += [_lock(0.25 + 0.5 * i) for i in range(40) if not 10 <= 0.25 + 0.5 * i <= 13]
        # move: no lock at all -> 10 s blind of 10.
        fr = blind_fractions(ev)
        self.assertAlmostEqual(fr["still"][1], 20.0)
        self.assertAlmostEqual(fr["still"][0], 3.5, delta=0.01)   # 9.75 -> 13.25
        self.assertEqual(fr["move"], (10.0, 10.0))

    def test_demo_without_blocks_counts_from_first_lock(self):
        from xreal_gates.blind import blind_after_first_lock
        # locks at 10, 10.5, 11, then a 4 s hole, 15, and the run ends at 15.5 -> 4 s blind of 5.5.
        ev = [_ev(0, "demo", "ALL_OFF"), _lock(10), _lock(10.5), _lock(11), _lock(15), _ev(15.5, "gate_result")]
        blind, total = blind_after_first_lock(ev)
        self.assertAlmostEqual(blind, 4.0)
        self.assertAlmostEqual(total, 5.5)
        self.assertIsNone(blind_after_first_lock([_ev(0, "demo", "ALL_OFF")]))

    def test_report_and_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            ev = [_ev(0, "step", "static"), _lock(0.5), _lock(1.0), _ev(5, "end"),
                  _ev(2, "stream", "lag", "n=1 frames=60 sent=10 on_s=3.50 lag_s=1.01"),
                  _ev(4, "stream", "window", "n=2 frames=60 sent=10 on_s=4.50 lag_s=0.20")]
            with (run / "events.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["t_unity", "frame", "event", "result", "detail"])
                w.writeheader()
                w.writerows(ev)
            with (run / "perf.csv").open("w", newline="", encoding="utf-8") as fh:
                fh.write("t_unity,cam_fps,cpu_performance_pct\n")
                for i in range(10):
                    fh.write(f"{i},{20 - i},{220 - 3 * i}\n")
            h = camera_health(run)
            self.assertEqual(h["windows"], 2)
            self.assertEqual(h["ends"], {"lag": 1, "window": 1})
            self.assertAlmostEqual(h["median_window_s"], 4.0)
            self.assertEqual(len(h["cam_fps_fifths"]), 5)
            lines = report(run)
            self.assertIn("still 80 % of 5 s", lines[0])


if __name__ == "__main__":
    unittest.main()
