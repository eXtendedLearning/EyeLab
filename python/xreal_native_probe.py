#!/usr/bin/env python3
"""Stage S2 — bring up the XREAL C API standalone and read its version.

Run on the Windows machine the glasses are plugged into, with Tier B staged
(``vendor/README.md``). Levels are cumulative; start at the bottom:

    python python/xreal_native_probe.py --level load      # load + resolve, no NR call
    python python/xreal_native_probe.py                   # + create, version, destroy
    python python/xreal_native_probe.py --level start     # + the host's start/stop

``--entry api`` loads ``libnr_api.dll`` directly instead of going through
``libnr_loader.dll`` the way the vendor's own Unity plugin does; use it only
to tell a loader problem from an API problem.

The native work runs in a child process (``native_isolation``), so a fault in
the vendor DLL is reported with its exit code instead of killing this process,
and a hang is cut off by ``--timeout``. Every run appends a line to
``python/.logs/xreal_native_probe.jsonl``, including whatever the vendor
library printed, which is often the most informative part of a failure.

Exit status: 0 if the requested level fully succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from native_isolation import append_jsonl, emit, run_json_worker

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / ".logs" / "xreal_native_probe.jsonl"

#: ``start`` may bring up tracking threads and the 6DoF plugin; give it room.
DEFAULT_TIMEOUT_S = {"load": 30.0, "version": 30.0, "start": 90.0}

_WORKER_FLAG = "--_worker"

START_WARNING = (
    "level 'start' runs NRAPIStart: the glasses may switch display mode or "
    "begin tracking. Unplugging and replugging restores them. Nothing is flashed."
)


def _run_worker(argv: list[str]) -> int:
    """Child entry: ``--_worker LEVEL ENTRY [DIR]``."""
    sys.path.insert(0, str(HERE))
    import xreal_native

    level, entry = argv[0], argv[1]
    directory = Path(argv[2]) if len(argv) > 2 and argv[2] else None
    emit(xreal_native.run(level=level, entry=entry, directory=directory).to_dict())
    return 0


def run_isolated(level: str, entry: str, directory: Optional[Path],
                 timeout_s: float) -> dict:
    argv = [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG, level, entry]
    if directory is not None:
        argv.append(str(directory))
    return run_json_worker(
        argv, timeout_s, label=f"S2 '{level}'",
        hang_hint="unplug and replug the glasses, then retry one level lower",
    )


def _render(payload: dict) -> str:
    lines = ["XREAL native API - Stage S2", "=" * 28, ""]
    if payload.get("fatal"):
        lines.append(f"FAILED: {payload['fatal']}")
        for key in ("stderr", "stdout"):
            if payload.get(key, "").strip():
                lines += ["", f"{key} (tail):", payload[key].strip()[-1500:]]
        return "\n".join(lines)

    lines.append(f"level             {payload.get('level')}  (entry: {payload.get('entry')})")
    lines.append(f"directory         {payload.get('directory') or '(injected)'}")
    pre = payload.get("preflight") or {}
    if pre:
        lines.append(f"preflight         {'ok' if pre.get('ok') else 'FAILED'}")
        if not pre.get("platform_ok"):
            lines.append("  not Windows: run this on the machine with the glasses")
        if not pre.get("python_64bit", True):
            lines.append("  32-bit Python: these are x64 DLLs")
        if pre.get("missing_files"):
            lines.append(f"  not staged:     {', '.join(pre['missing_files'])}")
        if pre.get("missing_system"):
            lines.append(f"  not on system:  {', '.join(pre['missing_system'])}"
                         "  (vulkan-1 comes with the GPU driver)")
    lines.append(f"loaded            {'yes' if payload.get('loaded') else 'no'}"
                 + (f"  {payload['library_path']}" if payload.get("library_path") else ""))

    resolved = payload.get("resolved") or {}
    if resolved:
        present = sum(1 for v in resolved.values() if v)
        lines.append(f"exports resolved  {present}/{len(resolved)}")
        if payload.get("missing_required"):
            lines.append(f"  MISSING:        {', '.join(payload['missing_required'])}")

    calls = payload.get("calls") or []
    if calls:
        lines += ["", "calls:"]
        for c in calls:
            outcome = (f"error {c['error']}" if c.get("error")
                       else f"-> {c.get('result')}")
            mark = "ok " if c.get("succeeded") else "!! "
            args = ", ".join(str(a) for a in c.get("args") or [])
            lines.append(f"  {mark}{c['name']}({args}) {outcome}  [{c.get('elapsed_ms')} ms]")
    lines.append("")
    if payload.get("handle"):
        lines.append(f"handle            {payload['handle']}")
    if payload.get("level") == "start":
        lines.append(f"started           {'yes' if payload.get('started') else 'no'}")
    lines.append(f"NR version        {payload.get('version') or '-'}")
    if payload.get("error"):
        lines.append(f"error             {payload['error']}")
    if payload.get("native_output"):
        lines += ["", "vendor library output (tail):", payload["native_output"][-1500:]]
    lines += ["", f"verdict           {'OK' if payload.get('ok') else 'incomplete'}",
              f"elapsed           {payload.get('elapsed_s', '?')}s",
              f"logged to         {LOG_PATH}"]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == [_WORKER_FLAG]:
        return _run_worker(argv[1:])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", choices=("load", "version", "start"), default="version")
    parser.add_argument("--entry", choices=("loader", "api"), default="loader")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Tier B directory (default: $EYELAB_XREAL_DIR or vendor/xreal/win-x64)")
    parser.add_argument("--timeout", type=float, default=None, help="seconds")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)

    if args.level == "start" and not args.json:
        print(f"note: {START_WARNING}\n")
    timeout = args.timeout or DEFAULT_TIMEOUT_S[args.level]
    payload = run_isolated(args.level, args.entry, args.dir, timeout)
    payload.setdefault("level", args.level)
    payload.setdefault("entry", args.entry)
    append_jsonl(LOG_PATH, payload)

    print(json.dumps(payload, indent=2) if args.json else _render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
