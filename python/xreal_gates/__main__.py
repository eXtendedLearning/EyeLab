"""CLI: ``python -m xreal_gates import [zip|folder ...]`` and ``python -m xreal_gates summary [run ...]``.

Run from ``python/``. Without arguments, ``import`` takes every zip in ``GateRuns/_inbox`` and
``summary`` lists every run in ``GateRuns/runs``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .archive import default_root, import_folder, import_inbox, import_zip
from .summary import HEADER, summarize


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m xreal_gates")
    parser.add_argument("--root", type=Path, default=default_root(), help="GateRuns root (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_import = sub.add_parser("import", help="import exported run zips or folders")
    p_import.add_argument("sources", nargs="*", type=Path)
    p_import.add_argument("--keep-zip", action="store_true", help="leave imported zips in place")
    p_sum = sub.add_parser("summary", help="one line per run")
    p_sum.add_argument("runs", nargs="*", type=Path)
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

    runs = args.runs or sorted(p for p in (root / "runs").iterdir() if p.is_dir())
    print(HEADER)
    for run in runs:
        print(summarize(run).row())
    return 0


if __name__ == "__main__":
    sys.exit(main())
