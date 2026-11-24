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

    def test_render_margin_split(self):
        """Late Unity frames are split by whether the detection worker was busy at that frame."""
        rows = [{"start_mono_ns": "100", "end_mono_ns": "200"}, {"start_mono_ns": "300", "end_mono_ns": "400"}]
        fr = [{"dt_ms": "16.7", "mono_ns": "0"},
              {"dt_ms": "25.0", "mono_ns": "150"},    # busy, late
              {"dt_ms": "16.7", "mono_ns": "350"},    # busy
              {"dt_ms": "16.7", "mono_ns": "250"},    # idle
              {"dt_ms": "30.0", "mono_ns": "500"}]    # idle, late
        line = b3a._render_margin(fr, rows, 60.0)
        self.assertIn("worker busy on 50 %", line)
        self.assertIn("late when busy 50.0 % (2), idle 50.0 % (2)", line)

    def test_ambiguity_matches_python_solver(self):
        """The report's ambiguity reads IPPE's errors as float64: the ratio (second over best) is never below 1."""
        K = np.array([[624.0, 0, 640], [0, 624, 360], [0, 0, 1]])
        ids, corners = b3a.board()
        obj = np.asarray(corners, float).reshape(-1, 3)
        import cv2
        for rvec, tvec in ((np.array([np.pi, 0, 0]), np.array([0, 0, 0.65])),
                           (np.array([np.pi + 0.6, 0.4, 0]), np.array([0, 0, 0.35]))):
            img = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
            img = img + np.random.default_rng(1).normal(0, 0.3, img.shape)
            a = b3a.ambiguity(obj, img, K, np.zeros(5))
            self.assertIsNotNone(a)
            self.assertGreater(a[0], 1.0)              # sorted by error: second over best is never below 1
        self.assertEqual(b3a.high_pass(np.ones((5, 3))).tolist(), np.zeros((5, 3)).tolist())

    def test_targets_and_roi(self):
        """T6.7: the set's targets, and detection in a region gives the full frame's markers in frame pixels."""
        ts = {t["name"]: t for t in b3a.targets()}
        self.assertEqual(ts["plate_A"]["ids"], [30, 31, 32])
        self.assertEqual(ts["plate_L"]["ids"], [40, 41, 42, 43])
        self.assertEqual(ts["board25"]["ids"], list(range(17)))           # 5 x 7 ChArUco: 17 markers
        c = np.asarray(ts["board25"]["corners"]).reshape(-1, 4, 3)
        self.assertAlmostEqual(float(np.linalg.norm(c[0][1] - c[0][0])), 0.02597, places=6)   # gui_common printed marker side (caliper)
        self.assertEqual(b3a.pose_lock_settings(), {"min_ambiguity_ratio": 3.0, "ambiguity_min_separation_deg": 10.0,
                                                    "max_acquire_reproj_rms_px": 15.0})
        s = {"ids": [30], "corners": [0.0] * 12}
        self.assertEqual(b3a.target_of(s, "nothing")["name"], "plate_A")

        K = np.array([[624.0, 0, 640], [0, 624, 360], [0, 0, 1]])
        ids, corners = b3a.board()
        img = b3a.render(K, np.zeros(5), ids, corners, np.array([np.pi, 0, 0]), np.array([0.05, 0.02, 0.5]), seed=3)
        s = {"tuning": b3a.tuning_list(), "dictionary": 0, "ids": ids, "corners": corners.reshape(-1).tolist(),
             "K": K.reshape(-1).tolist(), "dist": [0, 0, 0, 0, 0]}
        py = b3a.PythonPipeline(s)
        full = py.process(img)
        self.assertEqual(sorted(full["ids"]), [30, 31, 32])
        pts = np.concatenate(list(full["corners"].values()))
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        box = (int(x0) - 60, int(y0) - 60, int(x1 - x0) + 120, int(y1 - y0) + 120)
        roi = py.process(img, roi=box)
        self.assertEqual(sorted(roi["ids"]), [30, 31, 32])
        rx, ry, rw, rh = roi["roi"]                                       # ABI 4: grown to multiples of 8, same origin
        self.assertEqual((rx, ry, rw % 8, rh % 8), (box[0], box[1], 0, 0))
        self.assertTrue(box[2] <= rw < box[2] + 8 and box[3] <= rh < box[3] + 8)
        self.assertEqual(py.process(img, roi=(1235, 677, 100, 100))["roi"], (1232, 672, 48, 48))   # edge: moved back inside
        self.assertEqual(b3a.PythonPipeline(s, snap8=False).process(img, roi=box)["roi"], box)  # ABI 3 runs
        # ABI 4 also scales ArUco's perimeter rates by frame / region: the pixel limits stay the frame's
        d = py._scaled_detector(1280 / max(rw, rh)).getDetectorParameters()
        self.assertAlmostEqual(d.minMarkerPerimeterRate * max(rw, rh),
                               py.detector.getDetectorParameters().minMarkerPerimeterRate * 1280, 6)
        # CLAHE's 8x8 tiles cover the crop, not the frame, so a few corners move (about 1 px here); against the true
        # projection the region is no worse than the full frame
        import cv2
        rv, tv = np.array([np.pi, 0, 0]), np.array([0.05, 0.02, 0.5])
        truth = {i: cv2.projectPoints(c, rv, tv, K, np.zeros(5))[0].reshape(4, 2) for i, c in zip(ids, corners)}
        err = lambda res: float(np.mean([np.abs(res["corners"][i] - truth[i]).mean() for i in ids]))
        self.assertLess(err(roi), err(full) + 0.1)
        self.assertLess(float(np.mean([np.abs(roi["corners"][i] - full["corners"][i]).mean() for i in ids])), 0.5)
        self.assertLess(float(np.linalg.norm(roi["pose"][1] - full["pose"][1])) * 1000, 5.0)
        self.assertEqual(py.process(img, roi=(5, 5, 10, 10))["roi"], (0, 0, 1280, 720))   # too small: whole frame
        self.assertIsNone(b3a._roi({"roi_x": "0", "roi_y": "0", "roi_w": "1280", "roi_h": "720"}))
        self.assertEqual(b3a._roi({"roi_x": "10", "roi_y": "20", "roi_w": "300", "roi_h": "200"}), (10, 20, 300, 200))
        self.assertIsNone(b3a._roi({}))

    def test_mode_and_still_lines(self):
        def row(seq, acc, rv, roi=("100", "100", "320", "240")):
            return {"seq": str(seq), "pose_ok": "1", "accepted": acc, "prior_used": "0", "solution": "0", "mode": "roi",
                    "target": "plate_L", "nice": "10", "cap_skipped": "0", "roi_x": roi[0], "roi_y": roi[1], "roi_w": roi[2],
                    "roi_h": roi[3], "r0": str(rv[0]), "r1": str(rv[1]), "r2": str(rv[2]), "t0": "0", "t1": "0", "t2": "0.6"}
        import cv2
        Rg = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
        good = cv2.Rodrigues(Rg)[0].ravel()
        bad = cv2.Rodrigues(Rg @ cv2.Rodrigues(np.radians([30.0, 0.0, 0.0]))[0])[0].ravel()   # normal 30 deg off
        sel = [row(i, "1", good) for i in range(12)] + [row(12, "0", bad), row(13, "0", bad, ("0", "0", "1280", "720"))]
        line = b3a._mode_line(sel)
        self.assertIn("mode roi on plate_L, nice 10: region on 13/14 frames", line)
        self.assertIn("trusted 12/14 poses", line)
        still = b3a._still_lines(sel, {})
        self.assertIn("wrong solution (normal > 10 deg from the median): 2/14 poses; among trusted 0/12", still[0])
        # with cam_ts the first second is skipped (5 of 14 rows at 0.2 s apart)
        for i, r in enumerate(sel):
            r["cam_ts"] = str(int(i * 0.2e9))
        still = b3a._still_lines(sel, {})
        self.assertIn("first 1 s skipped): 2/9 poses", still[0])
        # with the head pose the normal is taken in the room: a head turned so that the "bad" camera normal is the
        # good one in the room makes those frames right
        F = np.diag([1.0, -1.0, 1.0])
        Rb = cv2.Rodrigues(bad)[0]

        S = np.diag([1.0, 1.0, -1.0])

        def head(R):
            """A Unity-handed head rotation, stored as the native pose (z flipped: S R S)."""
            m = np.eye(4)
            m[:3, :3] = S @ R @ S
            return {"ok": "1", **{f"m{i}": repr(float(v)) for i, v in enumerate(m.T.ravel())}}
        # H @ F @ n_bad = F @ n_good: H = F @ (Rg Rb^T) @ F rotates the bad normal onto the good one
        turned = F @ Rg @ Rb.T @ F
        heads = {r["seq"]: head(F @ F if r["r0"] == str(good[0]) else turned) for r in sel}
        still = b3a._still_lines(sel, heads)
        self.assertIn("wrong solution (room normal > 10 deg from the median, first 1 s skipped): 0/9 poses", still[0])

    def test_overlay_chain(self):
        """The room pose from an Eye pose, the native head pose and rgb_from_head, as the device's PoseOverlay."""
        import cv2
        rng = np.random.default_rng(3)
        S, F = np.diag([1.0, 1.0, -1.0]), np.diag([1.0, -1.0, 1.0])

        def rot(v):
            return cv2.Rodrigues(np.asarray(v, float))[0]
        Rw, tw = rot([0.3, -0.2, 0.1]), np.array([0.1, 1.2, 0.8])            # the target in the room (Unity)
        Rr, tr = rot([0.02, -0.01, 0.005]), np.array([0.01, 0.03, 0.02])     # Eye camera in the head (Unity)
        qx, qy, qz, qw = 0.0, np.sin(0.01), 0.0, np.cos(0.01)
        self.assertTrue(np.allclose(b3a.quat_matrix(qx, qy, qz, qw), rot([0, 0.02, 0])))
        for _ in range(5):
            RH, tH = rot(rng.normal(0, 0.4, 3)), rng.normal(0, 0.5, 3)        # head (Unity)
            # camera pose in Unity: world = H rgb C  ->  C = (H rgb)^-1 world
            RC = (RH @ Rr).T @ Rw
            tC = (RH @ Rr).T @ (tw - tH - RH @ tr)
            rv = cv2.Rodrigues(F @ RC @ F)[0].ravel()
            tv = F @ tC
            Rn, tn = S @ RH @ S, S @ tH                                      # native: z flipped
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = Rn, tn
            h = {f"m{i}": str(x) for i, x in enumerate(M.T.ravel())}
            R, t = b3a.target_in_room(rv, tv, b3a.head_unity(h), (Rr, tr))
            self.assertTrue(np.allclose(R, Rw, atol=1e-9) and np.allclose(t, tw, atol=1e-9))
        meta = {"rgb_from_head": f"ok; p 0.01 0.03 0.02; q {float(qx)!r} {float(qy)!r} {float(qz)!r} {float(qw)!r}"}
        R, t, ok = b3a.rgb_from_head(meta)
        self.assertTrue(ok and np.allclose(t, [0.01, 0.03, 0.02]) and np.allclose(R, rot([0, 0.02, 0])))
        self.assertFalse(b3a.rgb_from_head({})[2])

    def test_view_offset_chain(self):
        """T6.10: with a head-frame view offset, world = H off rgb C; an overlay.csv row carries it, a T6.9 row has none."""
        import cv2
        F = np.diag([1.0, -1.0, 1.0])

        def rot(v):
            return cv2.Rodrigues(np.asarray(v, float))[0]
        Rw, tw = rot([0.1, 0.4, -0.2]), np.array([0.2, 1.1, 0.6])
        Rr, tr = rot([0.01, 0.02, 0.0]), np.array([0.001, 0.005, 0.028])
        RH, tH = rot([0.2, -0.3, 0.05]), np.array([0.0, 1.5, 0.1])
        ang = np.radians(1.3) / 2                                             # 1.3 deg yaw, 2 mm up
        row = {"off_tx_mm": "0", "off_ty_mm": "2", "off_tz_mm": "0",
               "off_qx": "0", "off_qy": repr(float(np.sin(ang))), "off_qz": "0", "off_qw": repr(float(np.cos(ang)))}
        off = b3a.view_offset(row)
        self.assertTrue(np.allclose(off[0], rot([0, np.radians(1.3), 0])) and np.allclose(off[1], [0, 0.002, 0]))
        self.assertAlmostEqual(b3a.offset_angle_deg(off), 1.3, places=6)
        Ro, to = off
        RC = (RH @ Ro @ Rr).T @ Rw                                           # world = H (Ro (rgb C) + to)
        tC = (RH @ Ro @ Rr).T @ (tw - tH - RH @ (Ro @ tr + to))
        rv, tv = cv2.Rodrigues(F @ RC @ F)[0].ravel(), F @ tC
        R, t = b3a.target_in_room(rv, tv, (RH, tH), (Rr, tr), off)
        self.assertTrue(np.allclose(R, Rw, atol=1e-9) and np.allclose(t, tw, atol=1e-9))
        R0, t0 = b3a.target_in_room(rv, tv, (RH, tH), (Rr, tr))
        self.assertGreater(np.linalg.norm(t0 - tw), 0.005)                   # without it the drawing is off
        self.assertIsNone(b3a.view_offset({"seq": "1", "px": "0"}))

    def test_scale_fit(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0, 0.2, (50, 3))                     # head part moves
        b = -a / 0.9 + rng.normal(0, 1e-4, (50, 3))         # camera translation 1/0.9 too long: s = 0.9 keeps it still
        s, spread = b3a.scale_fit(a, b)
        self.assertAlmostEqual(s, 0.9, places=3)
        self.assertLess(spread, 0.5)

    def test_refinement_flips(self):
        dev = {7: np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])}
        py = {"ids": [7], "corners": {7: dev[7] + [[0.0, 0.0], [0.5, 0.0], [0.0, 0.3], [0.0, 0.0]]}, "pose": None,
              "raw": {7: dev[7] + [[0.0, 0.0], [0.5, 0.0], [0.7, 0.7], [0.0, 0.0]]}}
        c = b3a.compare(dev, None, py)
        self.assertEqual((c["differ"], c["flips"]), (2, 1))   # corner 1: python kept the unrefined one
        self.assertIn("differing corners 2, of which 1 refinement flips", b3a._parity_line("x", [c]))


if __name__ == "__main__":
    unittest.main()


class UpdateSetTest(unittest.TestCase):
    """T6.13: a new camera calibration replaces the set's model and re-renders the synthetic frames."""

    def test_update_set_swaps_the_camera_model(self):
        import json
        import tempfile
        from pathlib import Path
        from xreal_gates import b3a
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            old = tmp / "old" / "eye_calibration.json"
            old.parent.mkdir()
            old.write_text(json.dumps({"K": [[624, 0, 646], [0, 626, 357], [0, 0, 1]], "dist": [0, 0, 0, 0, 0]}))
            new = tmp / "cameras" / "beampro_eye.json"
            new.parent.mkdir()
            new.write_text(json.dumps({"K": [[790, 0, 640], [0, 790, 360], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                                       "run": "20261124_100000_camcal"}))
            out = tmp / "set"
            b3a.make_set(out, old, empty=0)
            before = (out / "frames" / "plate_00.pgm").read_bytes()
            s = b3a.update_set(out, new)
            self.assertEqual(s["K"][0], 790)
            self.assertEqual(s["calibration"], "cameras/beampro_eye.json 20261124_100000_camcal")
            self.assertNotEqual((out / "frames" / "plate_00.pgm").read_bytes(), before)
            self.assertEqual(json.loads((out / "set.json").read_text())["K"][0], 790)
            board25 = [t for t in s["targets"] if t["name"] == "board25"][0]
            c = np.asarray(board25["corners"]).reshape(-1, 4, 3)[board25["ids"].index(8)].mean(axis=0)
            np.testing.assert_allclose(c[:2], [2.5 * 0.02597 * 25 / 19, 3.5 * 0.02597 * 25 / 19], atol=1e-9)   # the printed board
