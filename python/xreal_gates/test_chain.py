import json
import tempfile
import unittest
from pathlib import Path

from xreal_gates.chain import fingerprint, report

HEADER = "t_unity,frame,event,result,detail\n"


def _run(root: Path, name: str, fx: float, offset_end: str) -> Path:
    run = root / name
    run.mkdir()
    (run / "pose_service.json").write_text(json.dumps({
        "targets": [{"name": "plateA", "source": "x/plate_A.target.json"}],
        "K": [[fx, 0, 640], [0, fx, 360], [0, 0, 1]], "dist": [0.1, 0, 0, 0, 0]}), encoding="utf-8")
    (run / "events.csv").write_text(
        HEADER
        + "1,1,render_fps,30,vSyncCount=2\n"
        + "2,2,switch_tracking,True,MODE_3DOF after=MODE_3DOF\n"
        + '3,3,rgb_from_head,OK,"pos=0,0,0.03 rot=0,0,0,1"\n'
        + '4,4,view_offset,LOADED,"none saved (zero): x +0.0 y +0.0 z +0.0 mm, yaw +0.0 pitch +0.0 roll +0.0 deg"\n'
        + "5,5,camera_mode,stream,flush_lag=0.3s\n"
        + "6,6,lock_filter,euro,start\n"
        + f'7,7,view_offset,SAVED,"C:/x/view_offset.json: {offset_end}"\n', encoding="utf-8")
    return run


class ChainTest(unittest.TestCase):
    def test_fingerprint_and_offset_may_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _run(Path(tmp), "a", 623.9, "x +6.0 y -18.0 z +0.0 mm, yaw +0.0 pitch +0.0 roll +0.0 deg")
            b = _run(Path(tmp), "b", 623.9, "x +0.0 y +0.0 z +0.0 mm, yaw +0.0 pitch +0.0 roll +0.0 deg")
            fp = fingerprint(a)
            self.assertEqual(fp["tracking"], "MODE_3DOF")
            self.assertEqual(fp["filter"], "euro")
            self.assertTrue(fp["offset_end"].startswith("x +6.0 y -18.0"))
            self.assertIn("IDENTICAL", report([a, b])[-1])

    def test_different_lens_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _run(Path(tmp), "a", 623.9, "x +0.0 y +0.0 z +0.0 mm, yaw +0.0 pitch +0.0 roll +0.0 deg")
            b = _run(Path(tmp), "b", 607.2, "x +0.0 y +0.0 z +0.0 mm, yaw +0.0 pitch +0.0 roll +0.0 deg")
            self.assertIn("DIFFERS in lens", report([a, b])[-1])


if __name__ == "__main__":
    unittest.main()
