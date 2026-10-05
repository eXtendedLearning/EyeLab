import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2

from camera_utils import lock_focus
from validation.campaign import Campaign
from validation.focus import focus_supported
from validation.plan import TESTS, default_plan, instructions
from validation.runner import capture_cmd


class TestPlan(unittest.TestCase):
    def test_ids_unique_and_every_spec_has_instructions(self):
        plan = default_plan()
        ids = [s["id"] for s in plan]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({s["test"] for s in plan}, set(TESTS))
        self.assertEqual(len(plan), 27 + 6 + 18 + 12)
        for s in plan:
            self.assertGreaterEqual(len(instructions(s)), 3, s["id"])
        self.assertIn("B1_A_d600_a30_r2", ids)
        self.assertIn("B3_A_p05_d600_a30_r1", ids)

    def test_repetitions_are_the_outer_loop(self):
        b1 = [s for s in default_plan() if s["test"] == "B1"]
        self.assertEqual([s["rep"] for s in b1[:9]], [1] * 9)
        self.assertEqual(b1[9]["rep"], 2)


class TestCampaignState(unittest.TestCase):
    def test_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            c = Campaign.open(Path(d) / "runs")
            first = c.next_pending()
            self.assertEqual(first["id"], c.plan[0]["id"])
            self.assertTrue(any("calibration" in p for p in c.ready_problems(first)))
            self.assertTrue(any("plate A" in p for p in c.ready_problems(first)))

            c.session_path(first["id"]).mkdir(parents=True)
            moved = c.reject_folder(first["id"])
            self.assertTrue(moved.exists() and "_rejected" in str(moved))
            c.mark(first["id"], "done", path="x")
            self.assertEqual(Campaign.open(Path(d) / "runs").status(first["id"]), "done")
            self.assertNotEqual(c.next_pending()["id"], first["id"])
            c.reopen(first["id"])
            self.assertEqual(c.status(first["id"]), "pending")
            self.assertEqual(c.counts()["B1"]["total"], 27)

    def test_calibration_record_and_change_detection(self):
        from calibrate import save_calibration
        with tempfile.TemporaryDirectory() as d:
            cal = Path(d) / "cam.yaml"
            import numpy as np
            save_calibration(str(cal), np.eye(3), np.zeros((1, 5)), (1280, 720), 0.31, 5, 7, 0.025, 0.019)
            c = Campaign.open(Path(d) / "runs")
            info = c.record_calibration(str(cal))
            self.assertAlmostEqual(info["rms_px"], 0.31)
            self.assertFalse(c.calibration_changed())
            save_calibration(str(cal), np.eye(3) * 2, np.zeros((1, 5)), (1280, 720), 0.4, 5, 7, 0.025, 0.019)
            self.assertTrue(c.calibration_changed())

    def test_step_pairs_use_measured_positions(self):
        with tempfile.TemporaryDirectory() as d:
            c = Campaign.open(Path(d) / "runs")
            b3 = [s for s in c.plan if s["test"] == "B3" and s["rep"] == 1]
            for s in b3:
                p = s["condition"]["position_mm"]
                c.mark(s["id"], "done", measured_position_mm=p + (0.03 if p == 10 else 0.0))
            pairs = c.step_pairs()
            self.assertEqual([p["nominal_mm"] for p in pairs], [5, 10, 20])
            self.assertAlmostEqual(pairs[1]["known_mm"], 10.03)

    def test_capture_command(self):
        with tempfile.TemporaryDirectory() as d:
            c = Campaign.open(Path(d) / "runs")
            c.state["calibration"] = {"path": "config/camera_params.yaml", "digest": "x"}
            c.state["plates"]["A"]["measured"] = "t.json"
            c.state["focus"] = 120.0
            cmd = capture_cmd(c, c.plan[0])
            self.assertIn("--focus", cmd)
            self.assertIn("test=B1", cmd)
            self.assertIn("rep=1", cmd)
            json.dumps(cmd)


class _Cap:
    def __init__(self, honours: bool):
        self.honours, self.p = honours, {cv2.CAP_PROP_AUTOFOCUS: 1.0, cv2.CAP_PROP_FOCUS: -1.0}

    def set(self, k, v):
        if self.honours:
            self.p[k] = float(v)
        return self.honours

    def get(self, k):
        return self.p[k]


class TestFocus(unittest.TestCase):
    def test_lock_focus_reports_what_the_camera_did(self):
        ok = lock_focus(_Cap(True), 120)
        self.assertEqual((ok["autofocus"], ok["focus"]), (0.0, 120.0))
        self.assertTrue(focus_supported(ok))
        ignored = lock_focus(_Cap(False), 120)
        self.assertFalse(focus_supported(ignored))
        self.assertEqual(SimpleNamespace(**ignored).requested_focus, 120.0)


if __name__ == "__main__":
    unittest.main()
