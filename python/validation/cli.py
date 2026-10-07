"""python -m validation <command>. Protocol: docs/validation.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _condition(pairs) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m validation",
                                description="EyeLab validation (see docs/validation.md)")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-plates", help="printable PDFs + nominal target/markers/UNV")
    m.add_argument("--out", default=None,
                   help="PDF folder (default docs/validation_printables); target/markers/UNV go "
                        "to python/config/{targets,markers,geometry}")
    m.add_argument("--flat", action="store_true",
                   help="write every file into --out instead (experiments, nothing in config/)")
    m.add_argument("--layout", nargs="+", default=["A", "L"], choices=["A", "L"])
    m.add_argument("--marker-size-mm", type=float, default=20.0)
    m.add_argument("--dpi", type=int, default=600)

    a = sub.add_parser("apply-measurements",
                       help="rescale a nominal target with the caliper-measured scale bars")
    a.add_argument("--target", required=True, help="nominal plate_<L>.target.json")
    a.add_argument("--bar-x-mm", type=float, required=True)
    a.add_argument("--bar-y-mm", type=float, required=True)
    a.add_argument("--fold-deg", type=float, default=90.0,
                   help="measured interior angle between the faces (plate L)")
    a.add_argument("--uncertainty-mm", type=float, required=True,
                   help="standard uncertainty of a reference-point position, mm")
    a.add_argument("--out", default=None,
                   help="write the measured files into this folder instead of python/config/")

    c = sub.add_parser("capture", help="record a session with EyeLab's pipeline (headless)")
    c.add_argument("--target", required=True)
    c.add_argument("--calibration", default=None,
                   help="camera calibration YAML (default: the active camera profile, "
                        "python/config/cameras/<profile>.yaml)")
    c.add_argument("--out", required=True, help="new session directory")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--frames", type=int, default=300)
    c.add_argument("--save-every", type=int, default=3,
                   help="write every k-th frame image (poses are logged every frame)")
    c.add_argument("--tuning", default="balanced", choices=["strict", "balanced", "forgiving"])
    c.add_argument("--platform", default="webcam")
    c.add_argument("--condition", nargs="*", metavar="KEY=VALUE",
                   help="e.g. test=B1 distance_mm=700 angle_deg=30 rep=1")
    c.add_argument("--focus", type=float, default=None,
                   help="disable autofocus and fix this focus value (same as at calibration)")
    c.add_argument("--no-preview", action="store_true")
    c.add_argument("--no-wait", action="store_true", help="start recording immediately")
    c.add_argument("--imu", default="off", choices=["off", "rear", "front"],
                   help="read the device IMU (camera built into a Surface: rear/front camera)")

    f = sub.add_parser("fold-angle", help="plate L: interior fold angle from recorded images")
    f.add_argument("sessions", nargs="+")
    f.add_argument("--target", required=True, help="plate L target (nominal or measured)")
    f.add_argument("--every", type=int, default=1)

    z = sub.add_parser("analyze", help="analyse recorded sessions")
    z.add_argument("sessions", nargs="+")
    z.add_argument("--target", required=True)
    z.add_argument("--out", required=True)
    z.add_argument("--search-radius", type=float, default=5.0, help="px, around the independent seed")
    z.add_argument("--every", type=int, default=1, help="analyse every k-th posed frame image")
    z.add_argument("--no-plots", action="store_true")

    s = sub.add_parser("step", help="B3: step error between two static sessions")
    s.add_argument("session_a")
    s.add_argument("session_b")
    s.add_argument("--known-mm", type=float, required=True)
    s.add_argument("--which", default="measured", choices=["measured", "pose"])

    g = sub.add_parser("campaign", help="guided measurement campaign (menu)")
    g.add_argument("--root", default="validation_runs", help="campaign folder (default: %(default)s)")

    y = sub.add_parser("synth", help="synthetic session with a known injected error")
    y.add_argument("--layout", default="A", choices=["A", "L"])
    y.add_argument("--out", required=True)
    y.add_argument("--frames", type=int, default=10)
    y.add_argument("--distance-mm", type=float, default=600.0)
    y.add_argument("--tilt-deg", type=float, default=0.0)
    y.add_argument("--bias-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    y.add_argument("--rot-noise-deg", type=float, default=0.0)
    y.add_argument("--trans-noise-mm", type=float, default=0.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .plates import LAYOUTS, build_target, layout_from_provenance
    from .target import load_target

    if args.cmd == "make-plates":
        import eyelab_paths
        from .printable import write_plate_bundle
        pdf_dir = Path(args.out) if args.out else eyelab_paths.PRINTABLES_DIR
        for name in args.layout:
            paths = write_plate_bundle(LAYOUTS[name](marker_size_mm=args.marker_size_mm),
                                       pdf_dir, args.dpi, flat=args.flat)
            print("\n".join(f"{k:8s} {v}" for k, v in paths.items()))

    elif args.cmd == "apply-measurements":
        from .printable import export_derived
        nominal = load_target(args.target)
        layout = layout_from_provenance(nominal.provenance)
        bars = {b.name: b.length_mm for b in layout.bars}
        sx, sy = args.bar_x_mm / bars["x"], args.bar_y_mm / bars["y"]
        t = build_target(layout, sx, sy, args.fold_deg, args.uncertainty_mm, measured=True)
        paths = export_derived(t, f"plate_{layout.name}.measured", out_dir=args.out)
        print(f"scale x {sx:.5f}, y {sy:.5f} (deviation {1e3 * (sx - 1):+.2f} / "
              f"{1e3 * (sy - 1):+.2f} per mille)")
        print("\n".join(f"{k:8s} {v}" for k, v in paths.items()))

    elif args.cmd == "capture":
        from . import capture
        from .recorder import SessionRecorder
        import eyelab_paths
        from surface_imu import DeviceImu
        t = load_target(args.target)
        if args.calibration is None:
            args.calibration = str(eyelab_paths.active_calibration_file())
        imu = DeviceImu.open(args.imu) if args.imu != "off" else None
        if args.imu != "off" and imu is None:
            print("IMU requested but not available: continuing without it")
        pipe = capture.build_pipeline(t, args.calibration, args.camera, args.tuning)
        cap, cam = capture.open_camera_locked(args.camera, args.focus)
        print(f"camera {args.camera} via {cam['backend']}: autofocus={cam['autofocus']:g} "
              f"focus={cam['focus']:g}")
        pipe.start(cap=cap)
        try:
            meta = capture.session_meta(pipe, t, args.target, args.calibration, args.tuning, cam)
            meta["imu"] = {"camera": args.imu} if imu is not None else None
            cond = _condition(args.condition)
            n = capture.run(
                pipe, lambda size: SessionRecorder(args.out, pipe.camera_matrix, pipe.dist_coeffs,
                                                   size, args.platform, cond, meta),
                t, args.frames, args.save_every, preview=not args.no_preview,
                wait=not args.no_wait and not args.no_preview, condition=cond, imu=imu)
        finally:
            pipe.stop()
        print(f"{n} frames -> {args.out}")

    elif args.cmd == "analyze":
        from .report import aggregate, analyze_session
        out = Path(args.out)
        sums = []
        for sp in args.sessions:
            name = "_".join(Path(sp).resolve().parts[-2:])
            sm = analyze_session(sp, args.target, out / name, args.search_radius, args.every,
                                 not args.no_plots)
            sums.append(sm)
            r, d = sm["registration"], sm["detection"]
            print(f"{name}: measured {d.get('measured_rate', float('nan')):.3f} | err_mm rmse "
                  f"{r['err_mm'].get('rmse', float('nan')):.3f} p95 {r['err_mm'].get('p95', float('nan')):.3f}"
                  f" | overlay jitter p95 {r['overlay_jitter_mm'].get('p95', float('nan')):.3f} mm")
        aggregate(sums, out)

    elif args.cmd == "fold-angle":
        from .fold import estimate_session
        from .session import load_session
        t = load_target(args.target)
        if "V" not in t.faces:
            raise SystemExit("fold-angle needs the folded plate L target")
        res = [estimate_session(load_session(sp), t, args.every) for sp in args.sessions]
        print(json.dumps(res, indent=2))
        good = [r for r in res if r["n_frames"]]
        if good:
            print(f"interior angle {np.mean([r['interior_deg'] for r in good]):.3f} deg "
                  f"(u {max(r['u_deg'] for r in good):.3f} deg) -> apply-measurements --fold-deg")

    elif args.cmd == "step":
        from .metrics import step_error
        from .session import load_session
        r = step_error(load_session(args.session_a), load_session(args.session_b), args.known_mm,
                       args.which)
        print(json.dumps(r, indent=2))

    elif args.cmd == "campaign":
        from .runner import main as campaign_main
        return campaign_main(args.root)

    elif args.cmd == "synth":
        from .synth import generate_session, view_pose
        layout = LAYOUTS[args.layout]()
        t = build_target(layout)
        K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
        rv, tv = view_pose(t, args.distance_mm / 1000.0, args.tilt_deg)
        generate_session(args.out, layout, t, K, rv, tv, args.frames, args.rot_noise_deg,
                         args.trans_noise_mm, args.bias_mm)
        from .target import save_target
        save_target(t, Path(args.out) / "target.json")
        print(f"{args.out} (analyse with --target {Path(args.out) / 'target.json'})")
    return 0
