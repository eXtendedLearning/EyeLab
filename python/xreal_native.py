#!/usr/bin/env python3
"""Tier B native NRSDK (``libnr_api``) bindings — Stage S2 of ADR-003.

S2 asks one question: **can the XREAL C API be brought up standalone — no
Nebula, no Unity, no graphics context — and report its version?** This module
answers it with the smallest possible surface, and grows by adding signatures,
not by guessing them.

What it calls, and why each call is safe to make
------------------------------------------------
Every signature below was recovered statically from the Nebula for Windows
build (2026-02-05); none is guessed. Two independent sources were used — the
export thunks inside ``libnr_api.dll`` itself (which registers carry which
argument, and at what width) and the call sites in ``XREALXRPlugin.dll``, the
vendor's own Unity plugin, which is the canonical host of this API. The
evidence is recorded in ``docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md`` §9.

The call order is the host's, not an invention:

* ``XREALXRPlugin`` sets ``SetDllDirectoryA(<plugins dir>)`` and loads
  **``libnr_loader.dll``**, not ``libnr_api.dll``; it resolves every ``NRAPI*``
  symbol from the loader. :data:`DEFAULT_ENTRY` follows it.
* ``[NativeAPI] Start`` runs ``NRAPIInitSetNetworkType(h, 0)`` → ``NRAPIStart(h)``
  → ``NRGetVersion(h, &v)``. Shutdown is ``NRAPIStop`` → ``NRAPIDestroy``.
* The host **never calls** ``NRAPIInitSetStandalone``. ADR-003 assumed it
  would be needed; the binary says otherwise, so it is resolved and reported
  but not called (:data:`RESOLVE_ONLY_EXPORTS`).

Levels (see :data:`LEVELS`), each a strict superset of the previous one:

``load``     preflight, load the entry DLL, resolve exports. No NR call.
``version``  ``NRAPICreate`` → ``NRGetVersion`` → ``NRAPIDestroy``. No start,
             so no sensor, display or glasses-state change is requested.
``start``    the host's start sequence, then stop and destroy. This is the
             first level that may make the glasses do something visible
             (switch display mode, begin tracking). Opt-in.

Nothing here prints or exits; ``xreal_native_probe.py`` owns I/O and runs this
in a child process, because a fault inside a vendor DLL must cost a
subprocess, not the caller. The DLLs are vendor binaries with no
redistribution licence: stage them yourself (``vendor/README.md``).
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import POINTER, Structure, c_int32, c_uint64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from xreal_glasses import vendor_dir

ENV_DIR = "EYELAB_XREAL_DIR"

#: Everything Tier B needs beside ``libnr_glasses_api.dll`` (Tier A). One
#: definition; ``vendor/README.md`` repeats it for humans.
TIER_B_FILES = (
    "libnr_api.dll",
    "libnr_loader.dll",
    "libnr_plugin_6dof.dll",
    "ov_utils.dll",
    "avcodec-60.dll",
    "avformat-60.dll",
    "avutil-58.dll",
    "swresample-4.dll",
    "swscale-7.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)

#: Imported statically by ``libnr_api.dll`` but supplied by Windows or the GPU
#: driver, never staged. A missing one makes the load fail with the unhelpful
#: "module not found" (error 126), so preflight names it instead.
SYSTEM_DEPENDENCIES = ("vulkan-1.dll", "D3DCOMPILER_47.dll")

ENTRY_POINTS = {"loader": "libnr_loader.dll", "api": "libnr_api.dll"}
DEFAULT_ENTRY = "loader"

LEVELS = ("load", "version", "start")

#: ``NRResult`` success. The only value the host code tests for. Other values
#: are reported as raw integers: naming them would be guessing.
NR_SUCCESS = 0

NRHandle = c_uint64


class NRVersion(Structure):
    """Three 32-bit integers, logged by the host as ``%d.%d.%d``."""

    _fields_ = [("major", c_int32), ("minor", c_int32), ("patch", c_int32)]


@dataclass(frozen=True)
class Signature:
    restype: Any
    argtypes: tuple
    evidence: str


#: The only functions this module ever calls.
SIGNATURES: dict[str, Signature] = {
    "NRAPICreate": Signature(
        c_int32, (POINTER(NRHandle),),
        "thunk forwards rcx as the out-pointer; host passes &wrapper->handle"),
    "NRGetVersion": Signature(
        c_int32, (NRHandle, POINTER(NRVersion)),
        "rcx handle, rdx out; writes 3 dwords at +0/+4/+8; returns 1 if out is NULL"),
    "NRAPIInitSetNetworkType": Signature(
        c_int32, (NRHandle, c_int32),
        "rcx handle, edx 32-bit value; host passes 0 right before NRAPIStart"),
    "NRAPIStart": Signature(c_int32, (NRHandle,), "rcx handle only"),
    "NRAPIStop": Signature(c_int32, (NRHandle,), "rcx handle only"),
    "NRAPIDestroy": Signature(c_int32, (NRHandle,), "rcx handle only"),
}

#: Resolved and reported, never called.
RESOLVE_ONLY_EXPORTS = (
    "NRAPIInitSetStandalone",        # the host never calls it; semantics unknown
    "NRAPIInitSetGraphicContextType",  # (h, int) — enum values not recovered
    "NRAPIInitSetRenderMode",        # (h, int) — host passes 1, conditionally
    "NRAPIInitSetGlassesControl",    # (h, bool, const char* config)
    "NRAPIInitSetLicenseData",       # (h, const char*, int)
    "NRAPIInitSetChannelIdentifier",
    "NRAPIPause",
    "NRAPIResume",
    "NRGetVersionExt",               # a stub: `mov eax, 4; ret` in this build
)


class NativeLibraryError(RuntimeError):
    """The Tier B library could not be found or loaded."""


@dataclass
class Preflight:
    directory: str
    platform_ok: bool
    python_64bit: bool
    missing_files: list[str] = field(default_factory=list)
    missing_system: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.platform_ok and self.python_64bit
                and not self.missing_files and not self.missing_system)


@dataclass
class CallRecord:
    name: str
    args: list[Any]
    result: Optional[int] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.result == NR_SUCCESS


@dataclass
class NativeReport:
    level: str
    entry: str
    platform: str
    directory: Optional[str] = None
    preflight: Optional[Preflight] = None
    library_path: Optional[str] = None
    loaded: bool = False
    resolved: dict[str, bool] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    handle: Optional[str] = None
    started: bool = False
    version: Optional[str] = None
    error: Optional[str] = None

    @property
    def missing_required(self) -> list[str]:
        return [n for n in SIGNATURES if not self.resolved.get(n, False)]

    @property
    def ok(self) -> bool:
        if not self.loaded or self.missing_required or self.error:
            return False
        if self.level == "load":
            return True
        if self.version is None:
            return False
        return self.level == "version" or self.started

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["missing_required"] = self.missing_required
        out["ok"] = self.ok
        if self.preflight is not None:
            out["preflight"]["ok"] = self.preflight.ok
        for rec, raw in zip(self.calls, out["calls"]):
            raw["succeeded"] = rec.succeeded
        return out


# ---------------------------------------------------------------------------
# Discovery, preflight, load
# ---------------------------------------------------------------------------


def find_directory() -> Path:
    override = os.environ.get(ENV_DIR)
    return Path(override) if override else vendor_dir()


def _system_dll_present(name: str) -> bool:
    """Probe for a system DLL without running its DllMain."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p
    kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    load_library_as_datafile = 0x00000002
    handle = kernel32.LoadLibraryExW(name, None, load_library_as_datafile)
    if not handle:
        return False
    kernel32.FreeLibrary(handle)
    return True


def preflight(directory: Path,
              dll_present: Optional[Callable[[str], bool]] = None) -> Preflight:
    """Check everything that would otherwise surface as an opaque load error."""
    result = Preflight(
        directory=str(directory),
        platform_ok=sys.platform == "win32",
        python_64bit=ctypes.sizeof(ctypes.c_void_p) == 8,
    )
    result.missing_files = [n for n in TIER_B_FILES if not (directory / n).is_file()]
    if dll_present is None and result.platform_ok:
        dll_present = _system_dll_present
    if dll_present is not None:
        result.missing_system = [n for n in SYSTEM_DEPENDENCIES if not dll_present(n)]
    return result


def load(directory: Path, entry: str = DEFAULT_ENTRY) -> tuple[Any, Path]:
    """Load the entry DLL the way the vendor's own host does."""
    if entry not in ENTRY_POINTS:
        raise NativeLibraryError(f"unknown entry {entry!r}; use one of {sorted(ENTRY_POINTS)}")
    path = directory / ENTRY_POINTS[entry]
    if not path.is_file():
        raise NativeLibraryError(f"{path} not found; stage Tier B (see vendor/README.md)")
    if sys.platform == "win32":
        # The loader LoadLibrary()s libnr_api and plugins by bare name. Those
        # nested loads use the classic search order, which add_dll_directory
        # does not affect; SetDllDirectoryW does, and it is what the host uses.
        ctypes.windll.kernel32.SetDllDirectoryW(str(directory))
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass
    try:
        return ctypes.CDLL(str(path)), path
    except OSError as exc:
        raise NativeLibraryError(f"failed to load {path}: {exc}") from exc


def bind(lib: Any) -> tuple[dict[str, Any], dict[str, bool]]:
    """Resolve exports; apply signatures to the ones this module calls."""
    functions: dict[str, Any] = {}
    resolved: dict[str, bool] = {}
    for name in tuple(SIGNATURES) + RESOLVE_ONLY_EXPORTS:
        try:
            fn = getattr(lib, name)
        except AttributeError:
            resolved[name] = False
            continue
        resolved[name] = True
        if name in SIGNATURES:
            fn.restype = SIGNATURES[name].restype
            fn.argtypes = list(SIGNATURES[name].argtypes)
            functions[name] = fn
    return functions, resolved


# ---------------------------------------------------------------------------
# The S2 sequence
# ---------------------------------------------------------------------------


class _Recorder:
    """Calls a bound function and records what happened, never raising.

    On Windows, ctypes turns a structured exception inside the callee (an
    access violation, say) into ``OSError``; that is recorded as the call's
    error rather than propagated, so cleanup still runs.
    """

    def __init__(self, functions: dict[str, Any], report: NativeReport):
        self._functions = functions
        self._report = report

    def __call__(self, name: str, *args: Any, shown: Optional[list] = None) -> CallRecord:
        record = CallRecord(name=name, args=shown if shown is not None else list(args))
        started = time.perf_counter()
        try:
            record.result = int(self._functions[name](*args))
        except Exception as exc:  # noqa: BLE001 - any failure is data here
            record.error = f"{type(exc).__name__}: {exc}"
        record.elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        self._report.calls.append(record)
        return record


def _read_version(call: _Recorder, handle: NRHandle, report: NativeReport) -> None:
    out = NRVersion()
    rec = call("NRGetVersion", handle, ctypes.pointer(out), shown=["handle", "&version"])
    if rec.succeeded:
        report.version = f"{out.major}.{out.minor}.{out.patch}"


def execute(level: str, functions: dict[str, Any], report: NativeReport) -> None:
    """Run the calls for ``level`` against already-bound functions."""
    if level == "load":
        return
    call = _Recorder(functions, report)
    handle = NRHandle(0)
    created = call("NRAPICreate", ctypes.pointer(handle), shown=["&handle"])
    if not created.succeeded:
        report.error = "NRAPICreate failed; nothing else was called"
        return
    report.handle = f"{handle.value:#x}"
    try:
        if level == "version":
            _read_version(call, handle, report)
            return
        # level == "start": the host's exact order (see module docstring).
        call("NRAPIInitSetNetworkType", handle, 0, shown=["handle", 0])
        report.started = call("NRAPIStart", handle, shown=["handle"]).succeeded
        if report.started:
            _read_version(call, handle, report)
            call("NRAPIStop", handle, shown=["handle"])
    finally:
        call("NRAPIDestroy", handle, shown=["handle"])


def run(level: str = "version", entry: str = DEFAULT_ENTRY,
        directory: Optional[Path] = None, lib: Any = None) -> NativeReport:
    """Whole S2 probe. Never raises for expected failures.

    ``lib`` injects an already-loaded library (tests); preflight and loading
    are then skipped.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; use one of {LEVELS}")
    report = NativeReport(level=level, entry=entry, platform=sys.platform)

    if lib is None:
        directory = directory or find_directory()
        report.directory = str(directory)
        report.preflight = preflight(directory)
        if not report.preflight.ok:
            report.error = "preflight failed; nothing was loaded"
            return report
        try:
            lib, path = load(directory, entry)
        except NativeLibraryError as exc:
            report.error = str(exc)
            return report
        report.library_path = str(path)

    report.loaded = True
    functions, report.resolved = bind(lib)
    if report.missing_required:
        report.error = "required exports missing: " + ", ".join(report.missing_required)
        return report
    execute(level, functions, report)
    return report
