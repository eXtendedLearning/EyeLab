"""Menu-driven campaign runner: ``python -m validation campaign`` (validation.bat).

Thin orchestration only: state lives in ``campaign.Campaign``, the session
matrix in ``plan``, and every measurement step runs the same CLI commands a
user could type (``capture``, ``analyze``, ``apply-measurements``,
``calibrate.py``) as a subprocess, so a crash in one never loses the campaign.
Step-by-step guide: docs/validation-campaign.md.
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import eyelab_paths as paths

from . import uncertainty as unc
from .campaign import Campaign
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
                        + (f" (u {v['readings']['uncertainty_mm']:.2f} mm)" if v.get("readings") else "")
                        for k, v in s["plates"].items())
    inst = c.instrument_mm
    print("\n" + "=" * 78)
    print(f" EyeLab validation campaign   ({c.root})")
    print(f" camera {s['camera_index']} profile '{s.get('camera')}' | IMU {s.get('imu', 'off')} | focus {focus}")
    print(f" {cal_txt}")
    print(f" instrument {'+-%g mm' % inst if inst is not None else 'not set'} | {plates}")
    counts = c.counts()
    print(" " + "   ".join(f"{t} {v['done']}/{v['total']}" + (f" ({v['skipped']} skipped)" if v["skipped"] else "")
                           for t, v in counts.items()) + "   (done/total)")
    nxt = c.next_pending()
    print(f" next session: {nxt['id'] if nxt else 'none - campaign complete'}")
    print("=" * 78)


MENU = """ Setup                               Measure
  1  Self-test (~15 s)                 6  Next session
  2  Camera, profile, IMU, focus      7  Pick / redo a session
  3  Calibrate camera                Results
  4  Plate measurements                8  Analyse
  5  Visual check in the EyeLab GUI    9  Step-error table (B3)
                                      10  Ground-truth review (GUI; also
                                          campaigns zipped into
                                          validation_runs/imports/)
                                       0  Quit"""


# ── setup actions ──
def self_test(c: Campaign) -> None:
    run(PY + ["-m", "unittest", "discover", "-s", "validation", "-t", ".", "-p", "test_*.py"])
    pause()


def camera_and_focus(c: Campaign) -> None:
    from .focus import find_focus, focus_supported
    idx = ask_float("Camera index (0 = first camera; the GUI lists them)", c.state["camera_index"])
    c.state["camera_index"] = int(idx)
    cams = paths.list_cameras()
    print("Camera profiles (one calibration each, shared with the EyeLab GUI): "
          + (", ".join(cams) if cams else "none yet"))
    name = ask("Profile for this camera (e.g. desktop_c920, surface_rear)",
               c.state.get("camera") or paths.active_camera())
    old_profile = c.state.get("camera")
    c.state["camera"] = paths.slug(name)
    paths.set_active_camera(c.state["camera"])
    imu = ask("Is the camera built into this device with an IMU? off / rear / front",
              c.state.get("imu", "off")).lower()
    c.state["imu"] = imu if imu in ("off", "rear", "front") else "off"
    if c.state["imu"] != "off":
        from surface_imu import DeviceImu
        if DeviceImu.open(c.state["imu"]) is None:
            print("No IMU reachable here (pip install winrt-runtime winrt-Windows.Devices.Sensors"
                  " on the Surface). IMU set to off.")
            c.state["imu"] = "off"
    c.save()
    if old_profile and old_profile != c.state["camera"] and c.state.get("calibration"):
        print(f"NOTE: the campaign calibration belongs to '{old_profile}'. Calibrate this "
              "profile (menu 3) before capturing.")
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
    cal = c.camera_file
    print(LINE)
    print(f"Calibration of camera profile '{cal.stem}' -> {cal}")
    print("Board: the EyeLab ChArUco board, 5x7 squares, 25 mm squares, 19 mm markers,")
    print("DICT_4X4_50 (the same one the GUI uses; any print of it works). It must be FLAT:")
    print("glued on card or foam board. Its absolute print scale does not matter for the")
    print("intrinsics (it only scales the board's distance), but X and Y must be equal:")
    print("compare the 100 mm ruler in both directions if you can.")
    print("Hold it in front of the camera and press SPACE when corners are drawn, 20-25 views:")
    print("  centre, four corners and edges of the image, near (~300 mm) and far (~900 mm),")
    print("  tilted 20-45 deg left/right/up/down. Sharp, no motion blur. ESC to finish.")
    print("Aim for RMS < 0.5 px.")
    print(LINE)
    if ask("Start? (Y/n)", "y").lower() != "y":
        return
    backup = paths.backup_calibration(cal)
    if backup:
        print(f"previous calibration backed up to {backup}")
    cal.parent.mkdir(parents=True, exist_ok=True)
    cmd = PY + ["calibrate.py", "--live", "--camera", str(c.state["camera_index"]),
                "--output", str(cal), "--min-frames", "20"]
    if c.state.get("focus") is not None:
        cmd += ["--focus", f"{c.state['focus']:g}"]
    if run(cmd) != 0 or not cal.exists():
        print("calibration did not complete")
        pause()
        return
    paths.set_active_camera(cal.stem)
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


def ask_instrument(c: Campaign) -> Optional[float]:
    cur = c.instrument_mm
    print("Length instrument: its accuracy as +-a mm (digital caliper ~0.02-0.03,")
    print("steel ruler / tape 0.5-1). Taken as rectangular: u = a / sqrt(3).")
    a = ask_float("Instrument accuracy +-a [mm]", cur)
    if a is None or a <= 0:
        return None
    c.state["instrument"] = {"accuracy_mm": float(a)}
    c.save()
    return float(a)


def plate_measurements(c: Campaign) -> None:
    from .plates import layout_from_provenance
    from .target import load_target
    plate = ask("Plate to enter (A or L)", "A").upper()
    if plate not in c.state["plates"]:
        return
    nominal_path = c.nominal_for(plate)
    nominal = load_target(nominal_path)
    bars = {b.name: b.length_mm for b in layout_from_provenance(nominal.provenance).bars}
    print(LINE)
    print(f"Plate {plate}: measure each scale bar from one edge of the first tick to the")
    print("SAME edge of the last tick (bleed cancels). Three readings each; re-seat the")
    print(f"instrument every time. Nominal: X {bars['x']:g} mm, Y {bars['y']:g} mm.")
    print(LINE)
    a = ask_instrument(c)
    if a is None:
        return
    xs = ask_readings("X bar")
    ys = ask_readings("Y bar")
    if not xs or not ys:
        return
    bx, by = statistics.mean(xs), statistics.mean(ys)
    for name, v in (("X", bx / bars["x"]), ("Y", by / bars["y"])):
        if abs(v - 1) > 0.01:
            print(f"WARNING: {name} scale off by {100 * (v - 1):+.2f} %: the page was probably "
                  "printed with fit-to-page. Reprint at 100 % before measuring.")
    fold, u_fold = 90.0, None
    if plate == "L":
        fold, u_fold = fold_angle(c, nominal_path, a)
        if fold is None:
            return
    budget = unc.point_uncertainty(nominal, unc.reading_u(xs, a), unc.reading_u(ys, a),
                                   bars["x"], bars["y"], u_fold)
    print(f"\nX {bx:.3f} mm, Y {by:.3f} mm. Reference-point standard uncertainty:")
    for k, v in budget["budget"].items():
        print(f"  {k:20s} {v:.3f} mm")
    print(f"  {'combined u':20s} {budget['u_mm']:.3f} mm   (U, k=2: {budget['U_mm']:.3f} mm)")
    print(f"  {unc.verdict(budget['u_mm'])}")
    print("  (uniform print scale cancels in the on-face metrics; only X/Y anisotropy counts)")
    if budget["tur"] < unc.MIN_TUR:
        print("  The sessions stay valid measurements, but cannot demonstrate sub-mm accuracy:")
        print("  report every error with this U. A digital caliper makes the anisotropy term")
        print("  negligible.")
    u_rec = ask_float("Standard uncertainty to record [mm]", round(budget["u_mm"], 3))
    cmd = PY + ["-m", "validation", "apply-measurements", "--target", nominal_path,
                "--bar-x-mm", f"{bx:.4f}", "--bar-y-mm", f"{by:.4f}", "--fold-deg", f"{fold:.3f}",
                "--uncertainty-mm", f"{u_rec:g}"]
    if run(cmd) != 0:
        pause()
        return
    c.state["plates"][plate]["measured"] = f"targets/plate_{plate}.measured.target.json"
    c.state["plates"][plate]["readings"] = {
        "x_mm": xs, "y_mm": ys, "fold_deg": fold, "u_fold_deg": u_fold, "instrument_mm": a,
        "uncertainty_mm": u_rec, "budget_mm": budget["budget"], "tur_1mm": budget["tur"]}
    c.save()
    pause()


def fold_angle(c: Campaign, nominal_path: str, a_mm: float):
    """Plate L interior angle: from images (recommended) or typed in."""
    print("\nFold angle of plate L. E = estimate it from the camera (recommended: about")
    print("0.1 deg, no instrument), T = type a measured value.")
    if ask("E / T", "E").upper() != "E":
        v = ask_float("Measured interior angle between the two faces [deg]", 90.0)
        a_deg = ask_float("Its accuracy +-[deg] (protractor ~1)", 1.0)
        return v, unc.rect_u(a_deg)
    if not c.state.get("calibration") or c.calibration_changed():
        print("Needs a current calibration first (menu 3).")
        pause()
        return None, None
    out = c.root / "fold" / time.strftime("%Y%m%d_%H%M%S")
    print("Plate L flat face on the table, both faces in view, 400-700 mm away. Press SPACE,")
    print("then move the camera slowly around (left, right, higher, lower) for ~6 s.")
    cmd = PY + ["-m", "validation", "capture", "--target", nominal_path,
                "--calibration", str(c.calibration_path()), "--out", str(out),
                "--camera", str(c.state["camera_index"]), "--frames", "180", "--save-every", "6",
                "--condition", "test=fold", "plate=L"]
    if c.state.get("focus") is not None:
        cmd += ["--focus", f"{c.state['focus']:g}"]
    if run(cmd) != 0 or not (out / "log.jsonl").exists():
        print("no images recorded")
        pause()
        return None, None
    from .fold import estimate_session
    from .session import load_session
    from .target import load_target
    r = estimate_session(load_session(out), load_target(nominal_path))
    if not r["n_frames"]:
        print("could not fit the fold in any image (both faces need >= 4 saddles)")
        pause()
        return None, None
    print(f"interior angle {r['interior_deg']:.3f} deg from {r['n_frames']} images "
          f"(spread {r['sd_frames_deg']:.3f} deg, u {r['u_deg']:.3f} deg, "
          f"reprojection {r['median_rms_px']:.3f} px)")
    v = ask_float("Interior angle to use [deg]", round(r["interior_deg"], 3))
    return v, r["u_deg"]


def visual_check(c: Campaign) -> None:
    measured = [(k, v) for k, v in c.state["plates"].items() if v["measured"]]
    if not measured:
        print("No measured plate yet (menu 4).")
        pause()
        return
    k = measured[0][0] if len(measured) == 1 else ask("Plate to show (" + "/".join(m[0] for m in measured) + ")",
                                                       measured[0][0]).upper()
    stem = f"plate_{k}.measured"
    markers = paths.MARKER_CONFIGS_DIR / f"{stem}.markers.json"
    geometry = paths.GEOMETRY_DIR / f"{stem}.unv"
    print(f"plate {k}: markers  {markers}")
    print(f"         geometry {geometry}")
    print(f"         camera   {c.camera_file}")
    if ask("Make these the GUI's active markers, geometry and camera? (Y/n)", "y").lower() == "y":
        paths.save_active(markers=markers, geometry=geometry, camera=c.camera_file.stem)
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
                "--calibration", str(c.calibration_path()),
                "--out", str(c.session_path(spec["id"])), "--camera", str(c.state["camera_index"]),
                "--frames", str(spec["frames"]), "--condition"] + [f"{k}={v}" for k, v in cond.items()]
    if c.state.get("focus") is not None:
        cmd += ["--focus", f"{c.state['focus']:g}"]
    if c.state.get("imu", "off") != "off":
        cmd += ["--imu", c.state["imu"]]
    return cmd


def run_session(c: Campaign, spec: dict) -> None:
    sid = spec["id"]
    problems = c.ready_problems(spec)
    if c.calibration_changed():
        problems.append(f"{c.calibration_path()} changed since it was recorded: run menu 3 "
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
        if spec["test"] == "B3" and spec["condition"]["position_mm"] == 0:
            a = c.instrument_mm
            if a is None or unc.tur(unc.step_uncertainty(a)) < unc.MIN_TUR:
                print(f"  ! B3 sets 5-20 mm steps with the instrument (+-{a if a else '?'} mm): "
                      f"step u = {unc.step_uncertainty(a or 1.0):.2f} mm,")
                print("    too coarse to test a 1 mm goal. Use a caliper/gauge blocks, or skip B3.")
        print("  In the preview the setup panel turns green when distance/angle/centring")
        print("  are in band. SPACE starts recording, Q aborts.")
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
                "Measured position for this session [mm]",
                float(spec["condition"]["position_mm"]))
            extra["instrument_mm"] = c.instrument_mm
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
        gt = c.root / "gt_review"
        extra = ["--gt-review", str(gt)] if gt.exists() else []
        run(PY + ["-m", "validation", "analyze", *map(str, sessions), "--target",
                  c.target_for(plate), "--out", str(out), "--every", str(int(every)), *extra])
        if extra:
            print(f"{t}: ground truth corrected with the review in {gt}")
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
        u_known = unc.step_uncertainty(c.instrument_mm) if c.instrument_mm else float("nan")
        rows.append({"rep": p["rep"], "nominal_mm": p["nominal_mm"], "known_mm": p["known_mm"],
                     "known_u_mm": u_known, "estimated_mm": r["estimated_mm"],
                     "error_mm": r["error_mm"], "tur_1mm": unc.tur(u_known)})
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
    u_known = rows[0]["known_u_mm"]
    print(f"\nknown-step standard uncertainty {u_known:.3f} mm -> {unc.verdict(u_known)}")
    print(f"written {out / 'steps.csv'}")
    pause()


def ground_truth_review(c: Campaign) -> None:
    from .gt_review_run import defaults, main as review_main
    review_main(defaults(), Path(__file__).resolve().parent.parent)


ACTIONS = {"1": self_test, "2": camera_and_focus, "3": calibrate, "4": plate_measurements,
           "5": visual_check, "6": next_session, "7": pick_session, "8": analyse, "9": step_table,
           "10": ground_truth_review}


def main(root: str = "validation_runs") -> int:
    here = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != here:
        print(f"run from {here} (validation.bat does this)")
        return 2
    for line in paths.migrate_legacy_layout():
        print(f"config migrated: {line}")
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
