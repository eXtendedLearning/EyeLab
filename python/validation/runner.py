"""Menu-driven campaign runner: ``python -m validation campaign`` (validation.bat).

Thin orchestration only: state lives in ``campaign.Campaign``, the session
matrix in ``plan``, and every measurement step runs the same CLI commands a
user could type (``capture``, ``analyze``, ``apply-measurements``,
``calibrate.py``) as a subprocess, so a crash in one never loses the campaign.
Step-by-step guide: docs/validation-campaign.md.
"""
from __future__ import annotations

import csv
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .campaign import CALIBRATION, PRINTABLES, Campaign
from .plan import TESTS, instructions

PY = [sys.executable]
LINE = "-" * 78


# ── small console helpers ──
def ask(prompt: str, default: str = "") -> str:
    try:
        v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:          # input closed: nobody is there to answer
        print()
        raise SystemExit(0)
    return v or default


def ask_float(prompt: str, default: Optional[float] = None) -> Optional[float]:
    while True:
        v = ask(prompt, "" if default is None else f"{default:g}")
        if v == "" and default is None:
            return None
        try:
            return float(v.replace(",", "."))
        except ValueError:
            print("  a number, please (decimal point or comma)")


def ask_readings(prompt: str, n: int = 3) -> List[float]:
    out = []
    for i in range(n):
        v = ask_float(f"  {prompt}, reading {i + 1}/{n} [mm]")
        if v is None:
            break
        out.append(v)
    return out


def run(cmd: List[str]) -> int:
    print(f"\n> {' '.join(str(c) for c in cmd)}\n")
    return subprocess.call([str(c) for c in cmd])


def pause():
    ask("\nENTER to continue")


# ── status ──
def header(c: Campaign) -> None:
    s = c.state
    cal = s.get("calibration")
    cal_txt = (f"calibration {cal['recorded'][:10]} RMS {cal['rms_px']:.2f} px"
               if cal else "calibration NOT recorded")
    if c.calibration_changed():
        cal_txt += " (FILE CHANGED since recorded!)"
    focus = "not fixed" if s.get("focus") is None else f"{s['focus']:g}"
    plates = " | ".join(f"plate {k} {'measured' if v['measured'] else 'NOT measured'}"
                        for k, v in s["plates"].items())
    print("\n" + "=" * 78)
    print(f" EyeLab validation campaign   ({c.root})")
    print(f" camera {s['camera_index']} | focus {focus} | {cal_txt}")
    print(f" {plates}")
    counts = c.counts()
    print(" " + "   ".join(f"{t} {v['done']}/{v['total']}" + (f" ({v['skipped']} skipped)" if v["skipped"] else "")
                           for t, v in counts.items()) + "   (done/total)")
    nxt = c.next_pending()
    print(f" next session: {nxt['id'] if nxt else 'none - campaign complete'}")
    print("=" * 78)


MENU = """ Setup                               Measure
  1  Self-test (~15 s)                 6  Next session
  2  Camera index and focus            7  Pick / redo a session
  3  Calibrate camera                Results
  4  Plate measurements                8  Analyse
  5  Visual check in the EyeLab GUI    9  Step-error table (B3)
                                       0  Quit"""


# ── setup actions ──
def self_test(c: Campaign) -> None:
    run(PY + ["-m", "unittest", "discover", "-s", "validation", "-t", ".", "-p", "test_*.py"])
    pause()


def camera_and_focus(c: Campaign) -> None:
    from .focus import find_focus, focus_supported
    idx = ask_float("Camera index (0 = first camera; the GUI lists them)", c.state["camera_index"])
    c.state["camera_index"] = int(idx)
    c.save()
    print("\nFocus: put plate A at ~600 mm, press A to let autofocus settle on it, then L to")
    print("lock there (or +/- by hand), ENTER to accept. Q keeps the current setting.")
    if ask("Open the focus window? (Y/n)", "y").lower() != "y":
        return
    r = find_focus(c.state["camera_index"])
    if r is None:
        return
    if focus_supported(r["readback"]):
        c.state["focus"] = r["focus"]
        print(f"Focus fixed at {r['focus']:g}. Calibration and every capture will use it.")
    else:
        c.state["focus"] = None
        print("This camera/driver ignores focus control (read-back "
              f"{r['readback']}). Disable autofocus in the camera's own software, or use")
        print("a fixed-focus camera. Do not change zoom/focus during the campaign.")
    c.save()
    if c.state.get("calibration"):
        print("NOTE: focus changed after calibration -> calibrate again (menu 3).")
    pause()


def calibrate(c: Campaign) -> None:
    print(LINE)
    print("Calibration (ChArUco A4, 25/19 mm). Check its 100 mm ruler first: if the print")
    print("is scaled, reprint at 100 %. Then hold the board (glued on card) in front of")
    print("the camera and press SPACE when corners are drawn, 20-25 views:")
    print("  centre, four corners and edges of the image, near (~300 mm) and far (~900 mm),")
    print("  tilted 20-45 deg left/right/up/down. Sharp, no motion blur. ESC to finish.")
    print("Aim for RMS < 0.5 px.")
    print(LINE)
    if ask("Start? (Y/n)", "y").lower() != "y":
        return
    cal = Path(CALIBRATION)
    if cal.exists():
        backup = c.root / "calibration_backups" / f"camera_params_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cal, backup)
        print(f"previous calibration backed up to {backup}")
    cmd = PY + ["calibrate.py", "--live", "--camera", str(c.state["camera_index"]),
                "--output", str(cal), "--min-frames", "20"]
    if c.state.get("focus") is not None:
        cmd += ["--focus", f"{c.state['focus']:g}"]
    if run(cmd) != 0 or not cal.exists():
        print("calibration did not complete")
        pause()
        return
    old = (c.state.get("calibration") or {}).get("digest")
    info = c.record_calibration(str(cal))
    print(f"\nRMS {info['rms_px']:.3f} px at {info['image_size'][0]}x{info['image_size'][1]}"
          + ("  -> above 0.5 px: consider repeating with sharper, more varied views" if info["rms_px"] > 0.5 else ""))
    stale = [sid for sid, e in c.state["sessions"].items()
             if e.get("status") == "done" and e.get("calibration_digest") not in (None, info["digest"])]
    if old and stale:
        print(f"{len(stale)} sessions were recorded with a previous calibration. They are not")
        print("comparable with new ones: redo them (menu 7) before pooling results.")
    pause()


def plate_measurements(c: Campaign) -> None:
    plate = ask("Plate to enter (A or L)", "A").upper()
    if plate not in c.state["plates"]:
        return
    print(LINE)
    print(f"Plate {plate}: caliper on each scale bar from one edge of the first tick to the")
    print("SAME edge of the last tick (bleed cancels). Three readings each; re-seat the")
    print("caliper every time. Nominal: X 200 mm, Y " + ("150" if plate == "A" else "75") + " mm.")
    print(LINE)
    u_cal = ask_float("Caliper accuracy [mm] (digital caliper: ~0.02-0.03)", 0.03)
    xs = ask_readings("X bar")
    ys = ask_readings("Y bar")
    if not xs or not ys:
        return
    fold = 90.0
    if plate == "L":
        fold = ask_float("Measured interior angle between the two faces [deg]", 90.0)
    bx, by = statistics.mean(xs), statistics.mean(ys)
    nominal = {"x": 200.0, "y": 150.0 if plate == "A" else 75.0}
    for name, v in (("X", bx / nominal["x"]), ("Y", by / nominal["y"])):
        if abs(v - 1) > 0.01:
            print(f"WARNING: {name} scale off by {100 * (v - 1):+.2f} %: the page was probably "
                  "printed with fit-to-page. Reprint at 100 % before measuring.")

    def rel(r, length):
        sd = statistics.stdev(r) / math.sqrt(len(r)) if len(r) > 1 else u_cal
        return math.hypot(u_cal, sd) / length

    u_rel = max(rel(xs, nominal["x"]), rel(ys, nominal["y"]))
    # farthest reference point ~130 mm from the plate centre; 0.02 mm print
    # placement; 0.05 mm flatness of mounted paper
    suggest = math.sqrt((u_rel * 130.0) ** 2 + 0.02 ** 2 + 0.05 ** 2)
    print(f"\nX {bx:.3f} mm, Y {by:.3f} mm. Suggested point uncertainty {suggest:.3f} mm")
    print("(scale reading + print placement 0.02 + paper flatness 0.05).")
    unc = ask_float("Standard uncertainty to record [mm]", round(suggest, 3))
    nominal_path = c.state["plates"][plate]["nominal"]
    cmd = PY + ["-m", "validation", "apply-measurements", "--target", nominal_path,
                "--bar-x-mm", f"{bx:.4f}", "--bar-y-mm", f"{by:.4f}", "--fold-deg", f"{fold:g}",
                "--uncertainty-mm", f"{unc:g}"]
    if run(cmd) != 0:
        pause()
        return
    c.state["plates"][plate]["measured"] = str(PRINTABLES / f"plate_{plate}.measured.target.json")
    c.state["plates"][plate]["readings"] = {"x_mm": xs, "y_mm": ys, "fold_deg": fold,
                                            "caliper_mm": u_cal, "uncertainty_mm": unc}
    c.save()
    pause()


def visual_check(c: Campaign) -> None:
    for k, v in c.state["plates"].items():
        if v["measured"]:
            stem = Path(v["measured"]).name.replace(".target.json", "")
            print(f"plate {k}: Markers tab -> load {PRINTABLES / (stem + '.markers.json')}")
            print(f"         Geometry  -> load {PRINTABLES / (stem + '.unv')}")
    print("Start AR: the yellow wireframe must sit on the saddle centres. Close the GUI to")
    print("come back here (it holds the camera).")
    if ask("Launch the GUI? (Y/n)", "y").lower() == "y":
        run(PY + ["eyelab_gui.py"])


# ── sessions ──
def quick_check(path: Path, spec: dict) -> List[str]:
    from .metrics import detection_stats
    from .session import load_session
    s = load_session(path)
    d = detection_stats(s.records)
    imgs = sum(1 for r in s.records if r.frame)
    out = [f"frames {d['n_frames']} (images {imgs}) | raw pose {100 * d['measured_rate']:.0f} % | "
           f"coasted {100 * d['coasted_rate']:.0f} % | markers in pose {d['markers_in_measured_pose']}"]
    want = {"B1": "3", "B1b": "1", "B2L": "4", "B3": "3"}.get(spec["test"])
    mk = d["markers_in_measured_pose"]
    if d["n_frames"] < spec["frames"]:
        out.append("WARNING: fewer frames than planned (aborted?)")
    if d["measured_rate"] < 0.95:
        out.append("WARNING: pose missing in >5 % of frames - check light, focus, framing")
    if want and mk and max(mk, key=mk.get) != want:
        out.append(f"WARNING: most frames used {max(mk, key=mk.get)} marker(s), expected {want}")
    return out


def capture_cmd(c: Campaign, spec: dict) -> List[str]:
    cond = {"test": spec["test"], "plate": spec["plate"], "rep": spec["rep"], **spec["condition"]}
    cmd = PY + ["-m", "validation", "capture", "--target", c.target_for(spec["plate"]),
                "--calibration", c.state["calibration"]["path"],
                "--out", str(c.session_path(spec["id"])), "--camera", str(c.state["camera_index"]),
                "--frames", str(spec["frames"]), "--condition"] + [f"{k}={v}" for k, v in cond.items()]
    if c.state.get("focus") is not None:
        cmd += ["--focus", f"{c.state['focus']:g}"]
    return cmd


def run_session(c: Campaign, spec: dict) -> None:
    sid = spec["id"]
    problems = c.ready_problems(spec)
    if c.calibration_changed():
        problems.append(f"{CALIBRATION} changed since it was recorded: run menu 3 "
                        "(or restore the recorded file) so sessions are not mixed")
    if problems:
        print("\nCannot capture yet:\n  - " + "\n  - ".join(problems))
        pause()
        return
    idx = [s["id"] for s in c.plan].index(sid) + 1
    while True:
        print("\n" + LINE)
        print(f" Session {idx}/{len(c.plan)}: {sid}   [{spec['test']}: {TESTS[spec['test']]}]")
        print(LINE)
        for line in instructions(spec):
            print(f"  - {line}")
        print("  In the preview: SPACE starts recording, Q aborts.")
        a = ask("\nENTER = open camera | S = skip with note | B = back to menu", "").lower()
        if a == "b":
            return
        if a == "s":
            c.mark(sid, "skipped", note=ask("Reason"))
            return
        moved = c.reject_folder(sid)
        if moved:
            print(f"previous attempt moved to {moved}")
        c.count_attempt(sid)
        rc = run(capture_cmd(c, spec))
        path = c.session_path(sid)
        if rc != 0 or not (path / "log.jsonl").exists():
            print("\nNo session recorded.")
            c.reject_folder(sid)
            if ask("Try again? (Y/n)", "y").lower() == "y":
                continue
            return
        print()
        for line in quick_check(path, spec):
            print(f"  {line}")
        a = ask("\nK = keep | R = redo now | D = discard, back to menu", "k").lower()
        if a == "r":
            continue
        if a == "d":
            c.reject_folder(sid)
            return
        extra = {"path": str(path), "calibration_digest": c.state["calibration"]["digest"]}
        if spec["test"] == "B3":
            extra["measured_position_mm"] = ask_float(
                "Caliper-measured position for this session [mm]",
                float(spec["condition"]["position_mm"]))
        note = ask("Note (optional)")
        if note:
            extra["note"] = note
        c.mark(sid, "done", **extra)
        return


def next_session(c: Campaign) -> None:
    spec = c.next_pending()
    if spec is None:
        print("All sessions done or skipped.")
        pause()
        return
    run_session(c, spec)


def pick_session(c: Campaign) -> None:
    test = ask("Test (" + ", ".join(TESTS) + ")", "B1")
    specs = [s for s in c.plan if s["test"] == test]
    for i, s in enumerate(specs, 1):
        e = c.state["sessions"].get(s["id"], {})
        print(f"  {i:3d}  {s['id']:28s} {c.status(s['id']):8s} {e.get('note', '')}")
    n = ask_float("Number (empty = back)")
    if n is None or not 1 <= int(n) <= len(specs):
        return
    spec = specs[int(n) - 1]
    if c.status(spec["id"]) != "pending":
        if ask(f"{spec['id']} is {c.status(spec['id'])}. Redo it? (y/N)", "n").lower() != "y":
            return
        c.reopen(spec["id"])
    run_session(c, spec)


# ── results ──
def analyse(c: Campaign) -> None:
    which = ask("Test to analyse (" + ", ".join(TESTS) + ", or ALL)", "ALL").upper()
    tests = [t for t in TESTS if which == "ALL" or t.upper() == which]
    every = ask_float("Analyse every k-th stored image (1 = all; 2-3 is faster)", 1)
    for t in tests:
        sessions = c.done_sessions(t)
        if not sessions:
            print(f"{t}: no sessions yet")
            continue
        plate = next(s["plate"] for s in c.plan if s["test"] == t)
        out = c.reports_dir / t
        run(PY + ["-m", "validation", "analyze", *map(str, sessions), "--target",
                  c.target_for(plate), "--out", str(out), "--every", str(int(every))])
        print(f"{t}: reports in {out} (conditions.csv, sessions.csv, one folder per session)")
    pause()


def step_table(c: Campaign) -> None:
    from .metrics import step_error
    from .session import load_session
    pairs = c.step_pairs()
    if not pairs:
        print("No complete B3 pairs yet (need position 0 and another position in one repetition).")
        pause()
        return
    rows = []
    for p in pairs:
        r = step_error(load_session(p["a"]), load_session(p["b"]), p["known_mm"])
        rows.append({"rep": p["rep"], "nominal_mm": p["nominal_mm"], "known_mm": p["known_mm"],
                     "estimated_mm": r["estimated_mm"], "error_mm": r["error_mm"]})
    out = c.reports_dir / "B3"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "steps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'step':>6} {'reps':>5} {'mean error':>11} {'sd':>8}   [mm]")
    for nom in sorted({r["nominal_mm"] for r in rows}):
        e = [r["error_mm"] for r in rows if r["nominal_mm"] == nom]
        sd = statistics.stdev(e) if len(e) > 1 else float("nan")
        print(f"{nom:>6} {len(e):>5} {statistics.mean(e):>+11.3f} {sd:>8.3f}")
    print(f"\nwritten {out / 'steps.csv'}")
    pause()


ACTIONS = {"1": self_test, "2": camera_and_focus, "3": calibrate, "4": plate_measurements,
           "5": visual_check, "6": next_session, "7": pick_session, "8": analyse, "9": step_table}


def main(root: str = "validation_runs") -> int:
    here = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != here:
        print(f"run from {here} (validation.bat does this)")
        return 2
    c = Campaign.open(root)
    while True:
        header(c)
        print(MENU)
        choice = ask("\nChoice", "6" if c.next_pending() else "0")
        if choice == "0":
            return 0
        action = ACTIONS.get(choice)
        if action is None:
            continue
        try:
            action(c)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:  # keep the campaign menu alive; the state is on disk
            print(f"\nERROR: {e!r}")
            pause()
