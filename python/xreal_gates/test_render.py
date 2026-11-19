import csv
import json
import tempfile
import unittest
from pathlib import Path

from xreal_gates import beampro, render

HEADER = ["setting", "step", "unity_frame", "dt_ms", "present", "dropped", "m2p_ms", "extended", "early", "teared",
          "scale", "msaa", "mono_ns", "boot_ns", "real_ns", "hmd_ns", "mono_after_ns", "unity_s"]
EV = ["event", "detail", "mono_ns", "boot_ns", "real_ns", "hmd_ns", "mono_after_ns", "unity_s"]


def _w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _frames(setting, fps, seconds, t0_ns, present_two_every=0):
    rows, step = [], 1e9 / fps
    for i in range(int(fps * seconds)):
        t = int(t0_ns + i * step)
        present = 2 if present_two_every and i % present_two_every == 0 else 1
        rows.append([setting, setting + "_still", i, 1000 / fps, present, 60, 33.0, i // 10, 0, "", 0.8, 1,
                     t, t, 0, t, t, t / 1e9])
    return rows


class RenderTest(unittest.TestCase):
    def test_fps_pacing_and_ratings(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "run.json").write_text(json.dumps({"probe": "render"}), encoding="utf-8")
            rows = _frames("now", 45, 10, 0, present_two_every=3) + _frames("lean_06", 60, 10, int(20e9))
            _w(run / "render.csv", HEADER, rows)
            _w(run / "events.csv", EV, [
                ["rating", "now:steady", 1, 1, 0, 1, 1, 0],
                ["rating", "now:flicker or judder", 2, 2, 0, 2, 2, 0],
                ["rating", "lean_06:steady", 3, 3, 0, 3, 3, 0],
            ])
            r = render.ratings(run)
            self.assertEqual(r["now"], ["steady", "flicker or judder"])
            by = {}
            for row in beampro.load_csv(run / "render.csv"):
                by.setdefault(row["setting"], []).append(row)
            now, lean = render.setting_stats(by["now"]), render.setting_stats(by["lean_06"])
            self.assertAlmostEqual(now["fps"], 45, delta=0.5)
            self.assertAlmostEqual(lean["fps"], 60, delta=0.5)
            self.assertAlmostEqual(now["present_2plus"], 1 / 3, delta=0.02)
            self.assertEqual(lean["present_2plus"], 0)
            self.assertAlmostEqual(now["over_refresh"], 1.0)   # 22 ms > 1.25 x 16.7 ms
            self.assertAlmostEqual(lean["over_refresh"], 0.0)
            self.assertAlmostEqual(now["extended"], 44.0)
            lines = beampro.report(run)
            text = "\n".join(lines)
            self.assertIn("flicker or judder (taps: steady, flicker or judder)", text)
            self.assertIn("lean_06", text)


if __name__ == "__main__":
    unittest.main()
