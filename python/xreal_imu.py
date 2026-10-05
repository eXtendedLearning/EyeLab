#!/usr/bin/env python3
"""Stage S3 — the glasses' IMU stream through the native NRSDK (ADR-003).

S3 asks: **does the IMU deliver samples to our process, at what rate, with
what timing, and in which units and axes?** This module only records; the
numbers are computed by ``imu_analysis.py``, so recording (native, risky,
Windows-only) and analysis (pure, testable anywhere) stay separate.

Signatures — all recovered statically, evidence in
``docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md`` §10:

* ``NRImuCreate(uint64* out)`` — the vendor host passes ``&wrapper->handle``.
* ``NRImuSetCaptureCallback(uint64 imu, cb, void* user)`` then
  ``NRImuStart(uint64 imu)``; ``NRImuStop`` / ``NRImuDestroy`` take the handle.
* The callback is ``void cb(uint64 imu, uint64 data, void* user)``: the
  library's trampoline forwards exactly those three registers.
* Getters ``(uint64 imu, uint64 data, out*)``: gyroscope, accelerometer and
  magnetometer write three ``float32``; both timestamps write one ``uint64``.
* **Ownership:** the vendor host reads every field *inside* the callback and
  then calls ``NRImuDataDestroy(imu, data)``. This module does the same; the
  data handle is never kept past the callback.

Sequence: the S2 start (``xreal_native.started_api``) → ``NRImuCreate`` →
``NRImuSetCaptureCallback`` → ``NRImuStart`` → record → ``NRImuStop`` →
``NRImuDestroy`` → S2 stop/destroy. Every step is recorded like S2's calls,
and cleanup runs on every failure path.

Units are **not** assumed. The accelerometer magnitude at rest and the gyro
integral over a known turn decide them, in ``imu_analysis``.
"""

from __future__ import annotations

import csv
import ctypes
import sys
import threading
import time
from ctypes import CFUNCTYPE, POINTER, Structure, c_float, c_int32, c_uint64, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import xreal_native
from xreal_native import (
    DEFAULT_ENTRY,
    NR_SUCCESS,
    NativeReport,
    NRHandle,
    Recorder,
    Signature,
    bind,
    open_library,
    started_api,
)


class NRVector3f(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]


#: ``void cb(uint64 imu_handle, uint64 data_handle, void* user_data)``
IMU_CALLBACK = CFUNCTYPE(None, c_uint64, c_uint64, c_void_p)

_GETTER_VEC = (NRHandle, NRHandle, POINTER(NRVector3f))
_GETTER_NS = (NRHandle, NRHandle, POINTER(c_uint64))

IMU_SIGNATURES: dict[str, Signature] = {
    **xreal_native.SIGNATURES,
    "NRImuCreate": Signature(c_int32, (POINTER(NRHandle),), "rcx out-pointer only"),
    "NRImuSetCaptureCallback": Signature(
        c_int32, (NRHandle, IMU_CALLBACK, c_void_p), "rcx imu, rdx cb, r8 user"),
    "NRImuStart": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRImuStop": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRImuDestroy": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRImuDataGetHMDTimeNanos": Signature(c_int32, _GETTER_NS, "writes one qword"),
    "NRImuDataGetHMDTimeNanosOnDevice": Signature(c_int32, _GETTER_NS, "writes one qword"),
    "NRImuDataGetGyroscope": Signature(c_int32, _GETTER_VEC, "writes 3 x float32"),
    "NRImuDataGetAccelerometer": Signature(c_int32, _GETTER_VEC, "writes 3 x float32"),
    "NRImuDataGetMagnetometer": Signature(c_int32, _GETTER_VEC, "writes 3 x float32"),
    "NRImuDataDestroy": Signature(c_int32, (NRHandle, NRHandle), "host calls it after reading"),
}

CSV_COLUMNS = ("host_ns", "t_ns", "t_dev_ns", "gx", "gy", "gz",
               "ax", "ay", "az", "mx", "my", "mz")

#: Callback objects must outlive every native thread that might call them.
#: The worker is a throwaway process, so holding them for its lifetime is right.
_KEEPALIVE: list[Any] = []


class SampleSink:
    """Receives IMU callbacks on the vendor's thread; never raises into it."""

    def __init__(self, functions: dict[str, Any]):
        f = functions
        self._t = f["NRImuDataGetHMDTimeNanos"]
        self._t_dev = f["NRImuDataGetHMDTimeNanosOnDevice"]
        self._gyro = f["NRImuDataGetGyroscope"]
        self._acc = f["NRImuDataGetAccelerometer"]
        self._mag = f["NRImuDataGetMagnetometer"]
        self._destroy = f["NRImuDataDestroy"]
        self.rows: list[tuple] = []
        self.getter_failures = 0
        self.destroy_failures = 0
        self.first_exception: Optional[str] = None
        self.accepting = True
        self._lock = threading.Lock()

    def __call__(self, imu: int, data: int, _user: Any) -> None:
        host_ns = time.perf_counter_ns()
        try:
            if self.accepting:
                t, t_dev = c_uint64(0), c_uint64(0)
                g, a, m = NRVector3f(), NRVector3f(), NRVector3f()
                codes = (
                    self._t(imu, data, ctypes.pointer(t)),
                    self._t_dev(imu, data, ctypes.pointer(t_dev)),
                    self._gyro(imu, data, ctypes.pointer(g)),
                    self._acc(imu, data, ctypes.pointer(a)),
                    self._mag(imu, data, ctypes.pointer(m)),
                )
                with self._lock:
                    if any(c != NR_SUCCESS for c in codes):
                        self.getter_failures += 1
                    self.rows.append((host_ns, t.value, t_dev.value, g.x, g.y, g.z,
                                      a.x, a.y, a.z, m.x, m.y, m.z))
        except Exception as exc:  # noqa: BLE001 - must not propagate into C
            if self.first_exception is None:
                self.first_exception = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if self._destroy(imu, data) != NR_SUCCESS:
                    self.destroy_failures += 1
            except Exception as exc:  # noqa: BLE001
                if self.first_exception is None:
                    self.first_exception = f"NRImuDataDestroy: {type(exc).__name__}: {exc}"

    def snapshot(self) -> list[tuple]:
        with self._lock:
            return list(self.rows)


@dataclass
class ImuReport(NativeReport):
    duration_s: float = 0.0
    samples: int = 0
    getter_failures: int = 0
    destroy_failures: int = 0
    callback_exception: Optional[str] = None
    csv_path: Optional[str] = None
    imu_handle: Optional[str] = None

    @property
    def ok(self) -> bool:
        return (self.loaded and not self.missing_required and not self.error
                and self.started and self.samples > 0 and self.callback_exception is None)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["ok"] = self.ok
        return out


def write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def _record_with_imu(call: Recorder, report: ImuReport, duration_s: float,
                     sink: SampleSink, on_started: Optional[Callable[[], None]],
                     on_finished: Optional[Callable[[], None]]) -> None:
    imu = NRHandle(0)
    if not call("NRImuCreate", ctypes.pointer(imu), shown=["&imu"]).succeeded:
        report.error = "NRImuCreate failed"
        return
    report.imu_handle = f"{imu.value:#x}"
    try:
        callback = IMU_CALLBACK(sink)
        _KEEPALIVE.append(callback)
        if not call("NRImuSetCaptureCallback", imu, callback, None,
                    shown=["imu", "callback", None]).succeeded:
            report.error = "NRImuSetCaptureCallback failed"
            return
        if not call("NRImuStart", imu, shown=["imu"]).succeeded:
            report.error = "NRImuStart failed"
            return
        try:
            if on_started is not None:
                on_started()
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                time.sleep(0.05)
            if on_finished is not None:
                on_finished()
        finally:
            call("NRImuStop", imu, shown=["imu"])
    finally:
        sink.accepting = False
        call("NRImuDestroy", imu, shown=["imu"])


def record(duration_s: float, csv_path: Optional[Path] = None,
           entry: str = DEFAULT_ENTRY, directory: Optional[Path] = None,
           lib: Any = None, on_started: Optional[Callable[[], None]] = None,
           on_finished: Optional[Callable[[], None]] = None) -> ImuReport:
    """Record ``duration_s`` of IMU samples. Never raises for expected failures.

    ``on_started`` / ``on_finished`` run on the calling thread right after
    ``NRImuStart`` succeeds and right before ``NRImuStop`` (the CLI beeps, so
    a turn can be timed without watching a screen).
    """
    report = ImuReport(level="imu", entry=entry, platform=sys.platform,
                       duration_s=duration_s, required=list(IMU_SIGNATURES))
    lib = open_library(report, entry, directory, lib)
    if lib is None:
        return report
    functions, report.resolved = bind(lib, IMU_SIGNATURES, xreal_native.RESOLVE_ONLY_EXPORTS)
    if report.missing_required:
        report.error = "required exports missing: " + ", ".join(report.missing_required)
        return report

    call = Recorder(functions, report)
    sink = SampleSink(functions)
    with started_api(call, report) as api:
        if api is not None:
            _record_with_imu(call, report, duration_s, sink, on_started, on_finished)

    rows = sink.snapshot()
    report.samples = len(rows)
    report.getter_failures = sink.getter_failures
    report.destroy_failures = sink.destroy_failures
    report.callback_exception = sink.first_exception
    if csv_path is not None and rows:
        write_csv(csv_path, rows)
        report.csv_path = str(csv_path)
    if report.started and not rows and not report.error:
        report.error = "IMU started but delivered no samples"
    return report
