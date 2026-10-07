"""End to end on synthetic frames: EyeLab's real ArucoPipeline -> capture ->
recorder -> analysis. Checks units, frames, marker geometry and corner order
agree between EyeLab and the validation package."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibrate import save_calibration
from validation import capture
from validation.plates import build_target, plate_a
from validation.recorder import SessionRecorder
from validation.report import analyze_session
from validation.synth import render_scene, view_pose
from validation.target import save_target

K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])


class _FrameSource:
    """Stands in for ThreadedCapture: the same read()/error/stop surface."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.error = None
        self.is_opened = True

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def stats(self):
        return {}

    def stop(self):
        pass



class TestCaptureEndToEnd(unittest.TestCase):
    def test_pipeline_capture_and_analysis_agree(self):
        layout = plate_a()
        t = build_target(layout, uncertainty_mm=0.05)
        rv, tv = view_pose(t, 0.55, 25.0)
        rng = np.random.default_rng(3)
        clean = render_scene(layout, t, K, rv, tv)
        # frames 0-19: all markers; 20-29: markers 31, 32 covered (B1b single-marker sustain)
        covered = clean.copy()
        for m in t.markers[1:]:
            uv = cv2.projectPoints(m.corners(), rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
            cv2.fillConvexPoly(covered, np.round(uv).astype(np.int32), 230.0)
            cv2.polylines(covered, [np.round(uv).astype(np.int32)], True, 230.0, 12)
        frames = []
        for i in range(30):
            g = np.clip((clean if i < 20 else covered) + rng.normal(0, 1.0, clean.shape), 0, 255)
            frames.append(cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_calibration(str(d / "cam.yaml"), K, np.zeros((1, 5)), (1280, 720), 0.2, 5, 7,
                             0.025, 0.019)
            save_target(t, d / "t.json")
            pipe = capture.build_pipeline(t, str(d / "cam.yaml"))
            pipe._capture = _FrameSource(frames)
            meta = capture.session_meta(pipe, t, d / "t.json", d / "cam.yaml", "balanced")
            n = capture.run(pipe, lambda size: SessionRecorder(d / "s", pipe.camera_matrix,
                                                               pipe.dist_coeffs, size, "synthetic",
                                                               {"test": "e2e"}, meta),
                            t, n_frames=30, idle_timeout_s=0.5,
                            condition={"test": "B1", "distance_mm": 550, "angle_deg": 25})
            self.assertEqual(n, 30)
            first = json.loads((d / "s" / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
            sm = analyze_session(d / "s", d / "t.json", d / "report", plots=False)

        # setup assistant: the logged geometry is the one the scene was rendered with
        setup = first["extra"]["setup"]
        self.assertAlmostEqual(setup["distance_mm"], 550.0, delta=2.0)
        self.assertAlmostEqual(setup["angle_deg"], 25.0, delta=0.3)
        self.assertLess(setup["centre_offset"], 0.02)

        det = sm["detection"]
        self.assertEqual(det["measured_rate"], 1.0)
        self.assertEqual(det["markers_in_measured_pose"], {"1": 10, "3": 20})
        reg = sm["registration"]
        self.assertEqual(reg["points_detected"], reg["points_total"])
        # Three markers on clean synthetic frames: EyeLab is well below 1 mm on the plate.
        three, one = reg["err_mm_by_markers"]["3"], reg["err_mm_by_markers"]["1"]
        self.assertLess(three["p95"], 0.2)
        # One 20 mm marker sustaining the lock is markedly worse even on ideal frames
        # (~1 mm median at 0.55 m on 2026-09-25): a property of EyeLab that the
        # B1b test exists to quantify, not an analysis artefact.
        self.assertGreater(one["median"], 3 * three["median"])
        # measured pose is metres: ~0.55 m from the plate centre
        self.assertAlmostEqual(np.linalg.norm(sm["jitter_measured"]["t_mean_mm"]), 550.0, delta=150.0)
        self.assertEqual(sm["target_digest"], sm["target_digest_at_capture"])


if __name__ == "__main__":
    unittest.main()
