#!/usr/bin/env python3
"""
AR freeze diagnostics: phase tracking, stall watchdog, hard-freeze stack dumps.

Why this module exists
----------------------
The AR overlay runs on the tkinter main thread as a chain of ``root.after``
callbacks: camera read, ArUco detection, pose solve, wireframe projection and
the PhotoImage upload all happen there. If any one of them blocks, the GUI
stops repainting, Windows leaves the stale window black and reports "not
responding", and nothing is raised -- no traceback reaches the session log or
the console. A frozen process cannot describe its own freeze, so the
diagnosis has to be written from outside the blocked thread.

Two independent observers, deliberately redundant:

1. ``ARWatchdog`` runs a daemon thread that samples a liveness timestamp the
   main thread refreshes. When the main thread has been silent for longer
   than ``stall_s`` it writes a ``stall`` record naming the phase the main
   thread last entered, how long it has been there, and the Python stack of
   every thread. This reports any block that releases the GIL, which covers
   every OpenCV call and all I/O.

2. A ``faulthandler`` timer, re-armed by that same thread *only while the
   main thread is healthy*. If the interpreter stops altogether -- a native
   deadlock, or a C call holding the GIL -- nothing re-arms it, so it fires
   from its own C thread and dumps every thread's stack without needing the
   GIL. That is the only mechanism that can report a hard freeze.

The instrumentation is passive: it starts one daemon thread and writes files.
It changes no runtime behaviour, so a freeze that reproduces without it
reproduces with it.

Output, in ``.logs/`` (gitignored), one pair of files per run:

    ar_debug_<ts>.jsonl   structured records, one JSON object per line
    ar_stacks_<ts>.log    raw faulthandler dumps (plain text, not JSON)

Record kinds in the JSONL:

    session     run metadata, written at start
    event       one-off marker (app_start, ar_start, ar_stopped, ...)
    phase_slow  a phase that took longer than ``slow_phase_ms``
    heartbeat   periodic AR-loop aggregate (stage timings, capture health)
    stall       main thread silent for > ``stall_s``, with all thread stacks
    resumed     the main thread came back, with the total blocked time

Use:

    wd = ARWatchdog(LOG_DIR).start()
    wd.attach(root.after)             # liveness tick from the tk event loop
    with wd.phase("frame.process"):
        ...
    wd.frame(markers=3, pose=True)    # once per AR frame
    wd.stop()

The module is stdlib-only and imports neither cv2 nor tkinter, so it is
usable from the headless EyeLab Service (ADR-001, Step C) and testable
without a display.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# ── Defaults ─────────────────────────────────────────────────────────────────

STALL_S = 2.0            # main-thread silence before a stall record is written
HARD_FREEZE_S = 10.0     # silence before faulthandler dumps native-side stacks
POLL_S = 0.25            # watchdog sampling period
HEARTBEAT_S = 2.0        # AR-loop aggregate period
SLOW_PHASE_MS = 250.0    # a single phase above this is logged on its own
STALL_REPEAT_S = 10.0    # re-report an ongoing stall this often
MAX_STACK_LINES = 14     # deepest frames kept per thread in a stall record


def _rss_mb() -> Optional[float]:
    """Resident memory in MiB, or None when it cannot be read cheaply."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            class _MemCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _MemCounters()
            counters.cb = ctypes.sizeof(_MemCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb,
            )
            if not ok:
                return None
            return round(counters.WorkingSetSize / (1024 * 1024), 1)
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
    except Exception:
        return None


def _thread_stacks(limit: int = MAX_STACK_LINES) -> dict[str, list[str]]:
    """Python stack of every live thread, newest frames last."""
    names = {t.ident: t.name for t in threading.enumerate()}
    stacks: dict[str, list[str]] = {}
    for ident, frame in sys._current_frames().items():
        label = f"{names.get(ident, 'thread')}-{ident}"
        lines = [
            " | ".join(part.strip() for part in entry.strip().split("\n"))
            for entry in traceback.format_stack(frame)
        ]
        stacks[label] = lines[-limit:]
    return stacks


class ARWatchdog:
    """Observes the main thread and records where it stalls.

    All public methods are safe to call after :meth:`stop`; they become
    no-ops. Diagnostics must never be able to break the application.
    """

    def __init__(
        self,
        log_dir: Path | str,
        *,
        stall_s: float = STALL_S,
        hard_freeze_s: float = HARD_FREEZE_S,
        poll_s: float = POLL_S,
        heartbeat_s: float = HEARTBEAT_S,
        slow_phase_ms: float = SLOW_PHASE_MS,
        stamp: Optional[str] = None,
    ):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = log_dir / f"ar_debug_{stamp}.jsonl"
        self.stacks_path = log_dir / f"ar_stacks_{stamp}.log"

        self.stall_s = float(stall_s)
        self.hard_freeze_s = float(hard_freeze_s)
        self.poll_s = float(poll_s)
        self.heartbeat_s = float(heartbeat_s)
        self.slow_phase_ms = float(slow_phase_ms)

        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)
        self._stacks = open(self.stacks_path, "a", encoding="utf-8", buffering=1)
        self._write_lock = threading.Lock()

        # (sequence, phase, perf_counter) rebound as one tuple by the observed
        # thread and only read by the watchdog thread, so no lock is needed on
        # the hot path.
        self._alive: tuple[int, str, float] = (0, "init", time.perf_counter())

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._owns_faulthandler = False

        self._frames = 0
        self._stages: dict[str, list[float]] = {}   # name -> [count, total_ms, max_ms]
        self._hb_frames = 0
        self._hb_t0 = time.perf_counter()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> "ARWatchdog":
        if self._running:
            return self
        self._running = True
        if not faulthandler.is_enabled():
            faulthandler.enable(file=self._stacks, all_threads=True)
            self._owns_faulthandler = True
        self.event(
            "session",
            pid=os.getpid(),
            python=sys.version.split()[0],
            platform=sys.platform,
            stall_s=self.stall_s,
            hard_freeze_s=self.hard_freeze_s,
            stacks_file=self.stacks_path.name,
            faulthandler_owned=self._owns_faulthandler,
        )
        self.tick("idle")
        self._thread = threading.Thread(target=self._watch, name="ar-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self, reason: str = "normal") -> None:
        if not self._running:
            return
        self._running = False
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        self.event("watchdog_stop", reason=reason, frames=self._frames)
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=max(1.0, self.poll_s * 4))
        if self._owns_faulthandler:
            try:
                faulthandler.disable()
            except Exception:
                pass
            self._owns_faulthandler = False
        with self._write_lock:
            for handle in (self._jsonl, self._stacks):
                try:
                    handle.close()
                except Exception:
                    pass

    # ── Main-thread API (hot path) ───────────────────────────────────────

    def tick(self, phase: str = "idle") -> None:
        """Mark the observed thread alive, and say what it is doing."""
        seq = self._alive[0]
        self._alive = (seq + 1, phase, time.perf_counter())

    def attach(self, schedule: Callable[[int, Callable[[], None]], Any],
               interval_ms: int = 500) -> None:
        """Drive :meth:`tick` from an event loop.

        ``schedule`` takes ``(delay_ms, callback)`` -- i.e. ``root.after``.
        This is what detects a freeze outside an instrumented phase.
        """
        def _beat() -> None:
            if not self._running:
                return
            self.tick("idle")
            try:
                schedule(interval_ms, _beat)
            except Exception:
                pass

        schedule(interval_ms, _beat)

    @contextmanager
    def phase(self, name: str):
        """Time one stage of work and name it while it runs."""
        start = time.perf_counter()
        self.tick(name)
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            slot = self._stages.get(name)
            if slot is None:
                self._stages[name] = [1.0, elapsed_ms, elapsed_ms]
            else:
                slot[0] += 1.0
                slot[1] += elapsed_ms
                slot[2] = max(slot[2], elapsed_ms)
            self.tick("idle")
            if elapsed_ms >= self.slow_phase_ms:
                self.event("phase_slow", phase=name, ms=round(elapsed_ms, 1))

    def frame(self, **fields: Any) -> None:
        """Close one AR frame; emits a heartbeat every ``heartbeat_s``."""
        self._frames += 1
        self._hb_frames += 1
        self.tick("idle")
        now = time.perf_counter()
        elapsed = now - self._hb_t0
        if elapsed >= self.heartbeat_s:
            stages = {
                name: {
                    "n": int(slot[0]),
                    "avg_ms": round(slot[1] / slot[0], 1),
                    "max_ms": round(slot[2], 1),
                }
                for name, slot in self._stages.items()
            }
            self._stages.clear()
            self.event(
                "heartbeat",
                frames=self._frames,
                ui_fps=round(self._hb_frames / elapsed, 1) if elapsed > 0 else None,
                stages=stages,
                threads=threading.active_count(),
                rss_mb=_rss_mb(),
                **fields,
            )
            self._hb_frames = 0
            self._hb_t0 = now

    def event(self, kind: str, **fields: Any) -> None:
        """Write one record. Never raises."""
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
        }
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:
            line = json.dumps({"ts": record["ts"], "kind": kind, "error": "unserialisable"})
        with self._write_lock:
            try:
                self._jsonl.write(line + "\n")
                self._jsonl.flush()
            except Exception:
                pass

    # ── Watchdog thread ──────────────────────────────────────────────────

    def _watch(self) -> None:
        stalled_seq: Optional[int] = None
        stall_start = 0.0
        last_report = 0.0
        while self._running:
            seq, phase, marked = self._alive
            now = time.perf_counter()
            silent = now - marked
            if silent > self.stall_s:
                if stalled_seq != seq:
                    stalled_seq, stall_start, last_report = seq, marked, 0.0
                if last_report == 0.0 or now - last_report >= STALL_REPEAT_S:
                    last_report = now
                    self.event(
                        "stall",
                        phase=phase,
                        blocked_ms=round(silent * 1000.0, 1),
                        frames=self._frames,
                        rss_mb=_rss_mb(),
                        stacks=_thread_stacks(),
                    )
                # Deliberately not re-arming the faulthandler timer: if this
                # is a hard freeze it must be allowed to fire.
            else:
                if stalled_seq is not None:
                    self.event(
                        "resumed",
                        phase=phase,
                        blocked_ms=round((marked - stall_start) * 1000.0, 1),
                    )
                    stalled_seq = None
                self._arm_faulthandler()
            time.sleep(self.poll_s)

    def _arm_faulthandler(self) -> None:
        try:
            faulthandler.dump_traceback_later(
                self.hard_freeze_s, repeat=True, file=self._stacks, exit=False,
            )
        except Exception:
            pass
