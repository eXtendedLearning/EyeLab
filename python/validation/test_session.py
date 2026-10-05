import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validation.recorder import SessionRecorder
from validation.session import load_session


class TestSessionRoundtrip(unittest.TestCase):
    def test_recorder_loader_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            pose = {"rvec": [0.1, 0, 0], "tvec": [0, 0, 0.5], "marker_ids": [30, 31, 32],
                    "coasted": False, "rms_reproj_px": 0.2}
            with SessionRecorder(Path(d) / "s", np.eye(3), np.zeros(5), (64, 48), "webcam",
                                 condition={"distance_mm": 700}, extra_meta={"target": "x"}) as rec:
                rec.record(np.zeros((48, 64), np.uint8), 1.0, pose=pose, measured=pose,
                           lock_state="locked", markers_seen=[30, 31, 32],
                           projected={"1": (10.5, 20.25)}, drawn={"1": (10, 20)})
                rec.record(np.zeros((48, 64), np.uint8), 2.0,
                           pose={**pose, "marker_ids": [], "coasted": True}, lock_state="coasting")
                rec.record(None, 3.0)
            s = load_session(Path(d) / "s")
            self.assertEqual(s.meta["length_unit"], "m")
            self.assertEqual(s.meta["target"], "x")
            r0, r1, r2 = s.records
            self.assertEqual(r0.n_markers_measured, 3)
            self.assertTrue(r1.pose.coasted)
            self.assertIsNone(r1.measured)
            self.assertIsNone(r2.pose)
            self.assertIsNone(r2.frame)
            np.testing.assert_allclose(r0.projected["1"], [10.5, 20.25])
            np.testing.assert_allclose(r0.measured.tvec, [0, 0, 0.5])
            self.assertEqual(s.read_gray(r0).shape, (48, 64))
            self.assertEqual(len(s.poses("measured")[1]), 1)
            self.assertEqual(len(s.poses("pose")[1]), 2)

    def test_first_draft_sessions_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "meta.json").write_text(json.dumps({"schema_version": 1, "length_unit": "mm"}))
            (Path(d) / "log.jsonl").write_text("")
            with self.assertRaises(ValueError):
                load_session(d)


if __name__ == "__main__":
    unittest.main()
