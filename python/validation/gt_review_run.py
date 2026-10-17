"""``python -m validation gt-review`` and the campaign menu's ground-truth review.

Sources: the local campaign folder, or a campaign recorded on another machine and
copied as a zip into ``validation_runs/imports/`` (``IMPORTS``). Each source keeps its
decisions in its own review folder (``gt_review.review_dir``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

from .gt_review import (AuditWalk, DecisionStore, MissingWalk, Review, RunSource, SessionWalk,
                        reanalyse, reports_dir, review_dir, scan, summarize)

IMPORTS_HELP = ("Campaigns recorded on another machine: zip the whole validation_runs folder of "
                "that machine and copy the .zip into")


def imports_dir(py_root: Path) -> Path:
    return py_root / "validation_runs" / "imports"


def tests_of(src: RunSource, tests: List[str]) -> List[str]:
    return [s for s in src.session_ids() if s.split("_")[0] in tests]


def print_summary(stores: List[DecisionStore], tol_px: float) -> None:
    rows = [r for st in stores for r in summarize(st, tol_px)]
    if not rows:
        print("no decisions yet")
        return
    print(f"\nfailure = final truth more than {tol_px} px from the judged point "
          "(ref auto: the analysis' detection; reviewed: a session review)")
    print(f"{'mode':8s} {'test':5s} {'ref':8s} {'judged':>6s} {'ok':>5s} {'carr.':>6s} {'corr.':>5s} "
          f"{'inval':>5s} {'fail %':>7s} {'u95 %':>6s} {'med d px':>8s}")
    pct = lambda v: "   -" if math.isnan(v) else f"{100 * v:6.1f}"
    for r in rows:
        print(f"{r['mode']:8s} {r['test']:5s} {r['ref']:8s} {r['judged']:6d} {r['ok']:5d} {r['carried']:6d} "
              f"{r['corrected']:5d} {r['invalid']:5d} {pct(r['failure_rate']):>7s} "
              f"{pct(r['failure_rate_u95']):>6s} {r['fail_d_px_median']:8.2f}")


def choose_source(py_root: Path) -> Optional[Path]:
    imp = imports_dir(py_root)
    imp.mkdir(parents=True, exist_ok=True)
    local = py_root / "validation_runs"
    options = ([local] if (local / "campaign.json").exists() else []) + sorted(imp.glob("*.zip"))
    print(f"\n{IMPORTS_HELP}\n  {imp}\n")
    if not options:
        print("No campaign found (no validation_runs/campaign.json, no zip in imports).")
        return None
    for j, p in enumerate(options, 1):
        label = "local campaign" if p == local else "imported"
        print(f"  {j}  {p.name:40s} {label}")
    print("  0  back")
    c = input("source > ").strip()
    if not c.isdigit() or not 1 <= int(c) <= len(options):
        return None
    return options[int(c) - 1]


MENU = """
 Ground-truth review: {name}
 decisions: {folder}
  1  Random audit B1 ({n_b1} points)
  2  Random audit B1b ({n_b1b} points)
  3  Review one session (fix one frame, carried to the others)
  4  Points without a truth (all sessions)
  5  Re-analyse with the reviewed truth
  6  Summary of decisions
  0  Back"""


class Runner:
    def __init__(self, source: Path, py_root: Path, args):
        self.path = Path(source)
        self.src = RunSource(self.path)
        self.folder = Path(args.out) if args.out else review_dir(self.path)
        self.review = Review(self.src, py_root / "config")
        self.args = args

    @property
    def sessions_store(self):
        return DecisionStore(self.folder / "session.csv")

    @property
    def audit_store(self):
        return DecisionStore(self.folder / "audit.csv")

    def gui(self, walk, title: str) -> None:
        from .gt_review_gui import ReviewApp
        ReviewApp(walk, title).run()
        print_summary([self.sessions_store, self.audit_store], self.args.tol_px)

    def audit(self, tests: List[str], n: int) -> None:
        sids = tests_of(self.src, tests)
        if not sids:
            print(f"no {'+'.join(tests)} sessions in {self.path.name}")
            return
        walk = AuditWalk(self.review, self.audit_store, sids, n, self.args.seed, truth=self.sessions_store)
        self.gui(walk, f"Ground-truth audit {'+'.join(tests)} (n={n}, seed {self.args.seed})")

    def session(self, sid: Optional[str]) -> None:
        store = self.sessions_store
        if sid is None:
            sids = self.src.session_ids()
            for j, s in enumerate(sids, 1):
                si = self.review.session(s)
                tot = len(si.frames) * len(si.point_ids)
                n = sum(1 for k in store.rows if k[0] == s)
                print(f"{j:4d}  {s:24s} points decided {n:5d}/{tot} "
                      f"({len(si.frames)} frames x {len(si.point_ids)} points)")
            k = input("session number (empty = back) > ").strip()
            if not k.isdigit() or not 1 <= int(k) <= len(sids):
                return
            sid = sids[int(k) - 1]
        walk = SessionWalk(self.review, store, sid, self.args.tol_px, self.args.show_all, self.args.strict)
        self.gui(walk, f"Ground-truth review {sid}")

    def missing(self) -> None:
        sids = tests_of(self.src, self.args.tests_missing)
        print(f"scanning the automatic truth of {len(sids)} sessions (cached after the first run)...")
        rows = scan(self.review, sids, self.folder / "scan.csv")
        walk = MissingWalk(self.review, self.sessions_store, rows, also_decided=self.audit_store,
                           max_residual_px=self.args.max_residual_px)
        left = walk.total() - walk.done()
        print(f"{walk.total()} points without a usable automatic truth, {left} still to decide")
        if left:
            self.gui(walk, f"Points without a truth ({self.path.name})")

    def reanalyse(self) -> None:
        out = Path(self.args.reports) if self.args.reports else reports_dir(self.path)
        tests = sorted({s.split("_")[0] for s in self.src.session_ids()})
        print(f"re-analysing {', '.join(tests)} with the reviewed truth -> {out}")
        reanalyse(self.src, self.review, self.folder, out, tests, every=1, plots=not self.args.no_plots)
        print(f"reports in {out} (conditions.csv, sessions.csv, one folder per session)")

    def menu(self) -> None:
        while True:
            print(MENU.format(name=self.path.name, folder=self.folder, n_b1=self.args.n,
                              n_b1b=self.args.n_b1b))
            c = input("> ").strip()
            try:
                if c == "1":
                    self.audit(["B1"], self.args.n)
                elif c == "2":
                    self.audit(["B1b"], self.args.n_b1b)
                elif c == "3":
                    self.session(None)
                elif c == "4":
                    self.missing()
                elif c == "5":
                    self.reanalyse()
                elif c == "6":
                    print_summary([self.sessions_store, self.audit_store], self.args.tol_px)
                elif c in ("0", "q", ""):
                    return
            except KeyboardInterrupt:
                print("\n(interrupted)")


def main(args, py_root: Path) -> int:
    while True:
        source = Path(args.source) if args.source else choose_source(py_root)
        if source is None:
            return 0
        r = Runner(source, py_root, args)
        if args.mode is None:
            r.menu()
        elif args.mode == "audit":
            r.audit(args.tests, args.n)
        elif args.mode == "session":
            r.session(args.session)
        elif args.mode == "missing":
            r.missing()
        elif args.mode == "reanalyse":
            r.reanalyse()
        elif args.mode == "summary":
            print_summary([r.sessions_store, r.audit_store], args.tol_px)
        if args.source or args.mode is not None:
            return 0


def defaults(**kw):
    """Arguments when called from the campaign menu (no command line)."""
    from types import SimpleNamespace
    base = dict(source=None, out=None, mode=None, tests=["B1"], n=300, n_b1b=120, seed=20261016,
                session=None, tol_px=0.5, show_all=False, strict=False, tests_missing=["B1", "B1b", "B2L"],
                max_residual_px=math.inf, reports=None, no_plots=False)
    base.update(kw)
    return SimpleNamespace(**base)
