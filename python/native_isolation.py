#!/usr/bin/env python3
"""Run a native-code worker in a child process and read back one JSON payload.

Shared by the XREAL probes (``xreal_probe.py`` for Stage 0, ``xreal_native_probe.py``
for S2). Every call into an undocumented vendor DLL goes through here, because:

* a native call that hangs is bounded by a timeout instead of wedging the caller;
* a native call that faults (access violation, abort) costs one child process,
  and its exit code and stderr tail are reported instead of lost;
* the same code is headed for the GUI, where a blocking native call on the
  tkinter thread is the failure recorded in ``docs/ar-freeze-diagnostics.md``.

**Framing.** Vendor libraries write their own logs to stdout, which would
corrupt a bare-JSON protocol. The worker therefore emits its payload as a
single line prefixed with :data:`SENTINEL`; the parent takes the *last* such
line and ignores everything else, which it keeps as ``native_output`` for
diagnosis. Output without a sentinel is still parsed as plain JSON, so a
worker that predates the framing keeps working.

Nothing here imports ctypes or touches hardware; it is platform-independent.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

SENTINEL = "@@EYELAB-JSON@@ "

#: How much of the child's own stdout/stderr to keep in a payload.
OUTPUT_TAIL_CHARS = 4000


def emit(payload: dict[str, Any]) -> None:
    """Worker side: write the payload as one framed line and flush."""
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()


def extract_payload(stdout: str) -> tuple[Optional[dict], str]:
    """Parent side: return ``(payload, other_output)``.

    Raises ``json.JSONDecodeError`` if there is no sentinel and the whole
    output is not valid JSON either.
    """
    idx = stdout.rfind(SENTINEL)
    if idx < 0:
        return json.loads(stdout), ""
    line_end = stdout.find("\n", idx)
    line = stdout[idx + len(SENTINEL): line_end if line_end >= 0 else None]
    other = (stdout[:idx] + (stdout[line_end:] if line_end >= 0 else "")).strip()
    return json.loads(line), other


def run_json_worker(
    argv: Sequence[str],
    timeout_s: float,
    *,
    label: str = "probe",
    hang_hint: str = "unplug and replug the glasses and retry",
) -> dict[str, Any]:
    """Run ``argv`` as a child process and return its payload. Never raises.

    On failure the dict has ``ok: False`` and a human-readable ``fatal``.
    """
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "fatal": (
                f"{label} did not finish within {timeout_s:.0f}s. A native call "
                f"is hanging; {hang_hint}."
            ),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except OSError as exc:
        return {"ok": False, "fatal": f"could not start {label}: {exc}",
                "elapsed_s": round(time.monotonic() - started, 3)}

    elapsed = round(time.monotonic() - started, 3)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0 or not stdout.strip():
        detail = stderr.strip().splitlines()
        tail = detail[-1] if detail else describe_exit_code(completed.returncode)
        return {
            "ok": False,
            "fatal": f"{label} subprocess failed: {tail}",
            "returncode": completed.returncode,
            "stderr": stderr[-OUTPUT_TAIL_CHARS:],
            "stdout": stdout[-OUTPUT_TAIL_CHARS:],
            "elapsed_s": elapsed,
        }

    try:
        payload, other = extract_payload(stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "fatal": f"{label} produced unparseable output: {exc}",
            "stdout": stdout[:OUTPUT_TAIL_CHARS],
            "elapsed_s": elapsed,
        }
    if not isinstance(payload, dict):
        return {"ok": False, "fatal": f"{label} payload is not an object",
                "elapsed_s": elapsed}

    payload["elapsed_s"] = elapsed
    native = "\n".join(p for p in (other, stderr.strip()) if p)
    if native:
        payload["native_output"] = native[-OUTPUT_TAIL_CHARS:]
    return payload


#: Windows NTSTATUS values a crashing native call typically ends the process with.
_NTSTATUS = {
    0xC0000005: "access violation (0xC0000005) - a native call dereferenced a bad pointer",
    0xC0000135: "DLL not found (0xC0000135) - a dependency is missing",
    0xC0000139: "entry point not found (0xC0000139)",
    0xC0000409: "stack buffer overrun / fail-fast (0xC0000409)",
    0xC000001D: "illegal instruction (0xC000001D)",
    0x80000003: "breakpoint (0x80000003) - the library hit an assertion",
}


def describe_exit_code(code: int) -> str:
    """Name the common native-crash exit codes; fall back to the number."""
    unsigned = code & 0xFFFFFFFF
    if unsigned in _NTSTATUS:
        return _NTSTATUS[unsigned]
    if code < 0:
        return f"killed by signal {-code}"
    return f"exit code {code}"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append ``payload`` plus a UTC timestamp. Best-effort; never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass
