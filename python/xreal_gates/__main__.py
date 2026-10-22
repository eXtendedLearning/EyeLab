"""CLI for the XREAL PC-beta gates. Run from ``python/``.

- ``import [zip|folder ...]``: de-duplicate exported runs into ``GateRuns/runs`` (default: ``_inbox``).
- ``summary [run ...]``: one line per run.
- ``calibrate RUN|--latest-in DIR [--out YAML]``: Eye intrinsics from a ``-gate calib`` run.
- ``anchor RUN --target plateA|plateL|board25 [--calib YAML]``: offline lock test of an XR run.
- ``serve --target plateA[,board25] [--calib YAML] [--port 8765]``: the G5 and demo pose service (ADR-001 Step C).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .archive import default_root, import_folder, import_inbox, import_zip
from .summary import HEADER, summarize
from .targets import TARGET_NAMES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m xreal_gates")
    parser.add_argument("--root", type=Path, default=default_root(), help="GateRuns root (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_import = sub.add_parser("import", help="import exported run zips or folders")
    p_import.add_argument("sources", nargs="*", type=Path)
    p_import.add_argument("--keep-zip", action="store_true", help="leave imported zips in place")
    p_sum = sub.add_parser("summary", help="one line per run")
    p_sum.add_argument("runs", nargs="*", type=Path)
    p_cal = sub.add_parser("calibrate", help="Eye intrinsics from a calibration run")
    p_cal.add_argument("run", nargs="?", type=Path)
    p_cal.add_argument("--latest-in", type=Path, help="use the newest *_Calib_* run in this folder")
    p_cal.add_argument("--out", type=Path, help="YAML to write (default: eye_calibration.yaml in the run)")
    p_cal.add_argument("--copy-to", type=Path, help="also copy the YAML here (the pose service default)")
    p_anc = sub.add_parser("anchor", help="offline lock test of an XR run's snapshots")
    p_anc.add_argument("run", type=Path)
    p_anc.add_argument("--target", required=True, choices=TARGET_NAMES)
    p_anc.add_argument("--calib", type=Path, help="calibration YAML (default: repo calibration, else vendor)")
    p_srv = sub.add_parser("serve", help="G5 pose service on ws://127.0.0.1:PORT")
    p_srv.add_argument("--target", required=True,
                       help=f"one or more of {', '.join(TARGET_NAMES)}, comma-separated (demo: plateA,board25)")
    p_srv.add_argument("--calib", type=Path, help="calibration YAML (default: repo calibration, else vendor)")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.add_argument("--idle-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    root: Path = args.root

    if args.cmd == "import":
        if not args.sources:
            results = import_inbox(root)
            if not results:
                print(f"nothing in {root / '_inbox'}")
            for name, report in results:
                print(f"== {name}")
                print("\n".join("   " + line for line in report.lines()))
            return 0
        for src in args.sources:
            report = (import_zip(src, root, move_to_imported=not args.keep_zip)
                      if src.suffix.lower() == ".zip" else import_folder(src, root))
            print(f"== {src}")
            print("\n".join("   " + line for line in report.lines()))
        return 0

    if args.cmd == "calibrate":
        return _calibrate(args)
    if args.cmd == "anchor":
        from .anchor import analyse, report
        from .eyecam import load_camera
        from .targets import make_target
        camera = load_camera(args.calib if args.calib and args.calib.exists() else None)
        print(f"camera: {camera.describe()}")
        target = make_target(args.target)
        print(f"target: {target.label}")
        for line in report(analyse(args.run, target, camera)):
            print(line)
        return 0
    if args.cmd == "serve":
        from .pose_service import serve
        return serve(args.target, args.calib if args.calib and args.calib.exists() else None,
                     args.port, args.idle_timeout)

    runs = args.runs or sorted(p for p in (root / "runs").iterdir() if p.is_dir())
    print(HEADER)
    for run in runs:
        print(summarize(run).row())
    return 0


def _calibrate(args) -> int:
    import shutil
    from .calib import calibrate_run, describe, latest_calib_run
    run = args.run or (latest_calib_run(args.latest_in) if args.latest_in else None)
    if run is None:
        print("no calibration run given or found")
        return 1
    print(f"calibrating from {run}")
    try:
        result = calibrate_run(run, args.out)
    except ValueError as e:
        print(f"FAILED: {e}")
        return 1
    for line in describe(result):
        print(line)
    if args.copy_to:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.yaml_path, args.copy_to)
        shutil.copy2(result.report_path, args.copy_to.with_suffix(".json"))
        print(f"copied to {args.copy_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
