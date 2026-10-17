import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from validation.detect import apply_overrides
from validation.gt_review import (AuditWalk, DecisionStore, MissingWalk, Review, RunSource,
                                  SessionWalk, audit_keys, load_overrides, reanalyse, review_dir,
                                  scan, status_for, summarize, upper_bound_95)
from validation.plates import build_target, plate_a
from validation.synth import generate_session, view_pose

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _campaign(root: Path, n_sessions=2, n_frames=3):
    layout = plate_a()
    t = build_target(layout)
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    rv, tv = view_pose(t, 0.6, 20.0)
    (root / "sessions").mkdir(parents=True)
    (root / "campaign.json").write_text(json.dumps({"plates": {}}))
    for j in range(n_sessions):
        generate_session(root / "sessions" / f"B1_S{j}", layout, t, K, rv, tv, n_frames, seed=j)
        m = root / "sessions" / f"B1_S{j}" / "meta.json"
        meta = json.loads(m.read_text())
        meta["condition"] = {"test": "B1", "plate": "A"}
        m.write_text(json.dumps(meta))


class TestGtReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "validation_runs"
        _campaign(cls.root)
        cls.zip = cls.tmp / "runs.zip"
        with zipfile.ZipFile(cls.zip, "w") as z:
            for p in cls.root.rglob("*"):
                z.write(p, p.relative_to(cls.tmp).as_posix())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _review(self, src=None):
        return Review(RunSource(src or self.root), CONFIG)

    def test_folder_and_zip_agree(self):
        a, b = RunSource(self.root), RunSource(self.zip)
        self.assertEqual(a.session_ids(), ["B1_S0", "B1_S1"])
        self.assertEqual(a.session_ids(), b.session_ids())
        name = a.log("B1_S0")[0]["frame"]
        np.testing.assert_array_equal(a.frame_gray("B1_S0", name), b.frame_gray("B1_S0", name))

    def test_auto_points_and_snap(self):
        r = self._review()
        i = r.session("B1_S0").frames[0][0]
        fv = r.frame("B1_S0", i)
        for pid in ("1", "13", "20"):
            a = fv.auto(pid)
            self.assertLess(np.linalg.norm(a - fv.eyelab[pid]), 0.3)     # unbiased synthetic pose
            s = r.snap(fv, pid, a + [2.0, -1.5])
            self.assertIsNotNone(s)
            self.assertLess(np.linalg.norm(s - a), 0.2)
        self.assertLess(r.residual_px(fv, "7"), 0.3)

    def test_audit_sample_is_reproducible_and_spread(self):
        r = self._review()
        k1 = audit_keys(r, ["B1_S0", "B1_S1"], 7, seed=3)
        k2 = audit_keys(r, ["B1_S0", "B1_S1"], 7, seed=3)
        self.assertEqual(k1, k2)
        self.assertEqual(len(set(k1)), 7)
        counts = sorted(sum(k[0] == s for k in k1) for s in ("B1_S0", "B1_S1"))
        self.assertEqual(counts, [3, 4])

    def test_audit_walk_records_and_resumes(self):
        r = self._review()
        csv_path = self.tmp / "audit.csv"
        w = AuditWalk(r, DecisionStore(csv_path), ["B1_S0", "B1_S1"], 4, seed=1)
        it = w.next()
        w.decide(it, status_for(it, it.proposal), it.proposal, False)
        it2 = w.next()
        w.decide(it2, "invalid", None, False)
        w2 = AuditWalk(r, DecisionStore(csv_path), ["B1_S0", "B1_S1"], 4, seed=1)
        self.assertEqual(w2.done(), 2)
        self.assertNotIn((it.sid, it.i, it.pid), [(n.sid, n.i, n.pid) for n in [w2.next()]])

    def test_session_walk_trusts_a_stable_carry(self):
        r = Review(RunSource(self.root), CONFIG)
        frames = r.session("B1_S1").frames
        last = r.frame("B1_S1", frames[-1][0])
        last.loc.points["5"] = last.loc.points["5"] + np.array([3.0, 0.0])
        w = SessionWalk(r, DecisionStore(self.tmp / "loose.csv"), "B1_S1")
        it, n = w.next(), 0
        while it is not None:
            n += 1
            w.decide(it, status_for(it, it.proposal), it.proposal, False)
            it = w.next()
        self.assertEqual(n, 20)                    # only the first frame is shown
        self.assertEqual(w.auto_disagree, 1)
        row = w.store.get(("B1_S1", frames[-1][0], "5"))
        self.assertEqual(row["status"], "carried")
        self.assertGreater(float(row["d_px"]), 2.5)

    def test_session_walk_carries_decisions(self):
        r = self._review()
        csv_path = self.tmp / "session.csv"
        w = SessionWalk(r, DecisionStore(csv_path), "B1_S0", strict=True)
        frames = r.session("B1_S0").frames
        # perturb the automatic point 5 in the last frame: it must be shown again
        last = r.frame("B1_S0", frames[-1][0])
        last.loc.points["5"] = last.loc.points["5"] + np.array([3.0, 0.0])
        shown = []
        it = w.next()
        while it is not None:
            shown.append((it.i, it.pid))
            w.decide(it, status_for(it, it.proposal), it.proposal, False)
            it = w.next()
        first = frames[0][0]
        self.assertEqual(sum(i == first for i, _ in shown), 20)
        self.assertEqual([s for s in shown if s[0] != first], [(frames[-1][0], "5")])
        store = DecisionStore(csv_path)
        self.assertEqual(len(store.rows), 20 * len(frames))
        self.assertEqual(store.get(("B1_S0", frames[1][0], "9"))["status"], "carried")
        # the reshown point was proposed at the carried (true) position: correction recorded
        self.assertEqual(store.get(("B1_S0", frames[-1][0], "5"))["status"], "corrected")
        self.assertIsNone(SessionWalk(r, store, "B1_S0", strict=True).next())   # resume: nothing left
        # an audit over the reviewed session proposes the reviewed truth
        a = AuditWalk(r, DecisionStore(self.tmp / "aud.csv"), ["B1_S0"], 5, seed=2, truth=store)
        it = a.next()
        self.assertEqual(it.ref_kind, "reviewed")
        np.testing.assert_allclose(it.proposal, it.ref)

    def test_summary_and_bound(self):
        self.assertAlmostEqual(upper_bound_95(0, 300), 1 - 0.05 ** (1 / 300), places=4)
        store = DecisionStore(self.tmp / "sum.csv")
        base = {"session": "B1_S0", "mode": "audit", "test": "B1", "ref_kind": "auto", "auto_u": "1",
                "auto_v": "1", "final_u": "1", "final_v": "1"}
        store.put_many([dict(base, frame=0, point_id="1", status="ok", d_px="0"),
                        dict(base, frame=0, point_id="2", status="corrected", d_px="2.0"),
                        dict(base, frame=0, point_id="3", status="corrected", d_px="0.2"),
                        dict(base, frame=0, point_id="4", status="invalid", d_px="", final_u="")])
        s = summarize(store, 0.5)[0]
        self.assertEqual((s["judged"], s["failures"], s["invalid"]), (3, 1, 1))

    def test_apply_overrides_refits_the_homography(self):
        r = Review(RunSource(self.root), CONFIG)
        i = r.session("B1_S1").frames[1][0]
        fv = r.frame("B1_S1", i)
        truth = fv.auto("8").copy()
        fv.loc.points["8"] = truth + [4.0, 0.0]                       # a wrong automatic point
        t = r.session("B1_S1").target
        apply_overrides(fv.loc, t, fv.K, fv.dist, {"8": fv.loc.points["8"]})
        self.assertGreater(fv.loc.residual_px["F"].max(), 1.0)
        apply_overrides(fv.loc, t, fv.K, fv.dist, {"8": truth, "9": None})
        self.assertNotIn("9", fv.loc.points)
        self.assertLess(fv.loc.residual_px["F"].max(), 0.3)

    def test_overrides_merge_and_reanalysis_from_zip(self):
        folder = self.tmp / "rev"
        src = RunSource(self.zip)
        r = Review(src, CONFIG)
        i0 = r.session("B1_S0").frames[0][0]
        base = {"session": "B1_S0", "frame": i0, "test": "B1", "ref_kind": "auto"}
        DecisionStore(folder / "session.csv").put_many([
            dict(base, point_id="3", mode="session", status="corrected", final_u="1", final_v="1"),
            dict(base, point_id="4", mode="session", status="invalid")])
        DecisionStore(folder / "audit.csv").put(
            dict(base, point_id="3", mode="audit", status="corrected", final_u="2", final_v="2"))
        ov = load_overrides(folder)
        np.testing.assert_allclose(ov["B1_S0"][i0]["3"], [2, 2])         # the audit wins
        self.assertIsNone(ov["B1_S0"][i0]["4"])
        # re-analysis straight from the zip: a reviewed truth equal to the automatic one
        a = r.frame("B1_S0", i0).auto("3")
        DecisionStore(folder / "audit.csv").put(
            dict(base, point_id="3", mode="audit", status="ok", final_u=f"{a[0]:.4f}", final_v=f"{a[1]:.4f}"))
        out = reanalyse(src, r, folder, self.tmp / "reports", ["B1"], plots=False, progress=lambda m: None)
        sm = json.loads((out / "B1" / "sessions_B1_S0" / "summary.json").read_text())
        self.assertEqual(sm["registration"]["qa"]["ground_truth_points_reviewed"], 2)
        self.assertEqual(sm["registration"]["points_detected"], 20 * len(r.session("B1_S0").frames) - 1)
        self.assertTrue((out / "B1" / "conditions.csv").exists())

    def test_missing_walk_proposes_the_point_from_other_frames(self):
        r = Review(RunSource(self.root), CONFIG)
        frames = r.session("B1_S1").frames
        fv = r.frame("B1_S1", frames[2][0])
        truth = fv.loc.points.pop("17")
        rows = scan(r, ["B1_S1"], self.tmp / "scan.csv", progress=lambda m: None)
        self.assertEqual(sum(1 for x in rows if not x["auto_u"]), 1)
        w = MissingWalk(r, DecisionStore(self.tmp / "miss.csv"), rows)
        it = w.next()
        self.assertEqual((it.i, it.pid), (frames[2][0], "17"))
        self.assertLess(np.linalg.norm(it.proposal - truth), 0.3)
        w.decide(it, status_for(it, it.proposal), it.proposal, True)
        self.assertEqual(w.store.get(("B1_S1", frames[2][0], "17"))["status"], "corrected")
        self.assertIsNone(w.next())

    def test_review_dir_layout(self):
        self.assertEqual(review_dir(Path("x/imports/camp.zip")), Path("x/imports/camp/gt_review"))
        self.assertEqual(review_dir(Path("x/validation_runs")), Path("x/validation_runs/gt_review"))


if __name__ == "__main__":
    unittest.main()
