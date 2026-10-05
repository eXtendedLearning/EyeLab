#!/usr/bin/env python3
"""Stage S3b - head pose through the native NRSDK, and the licence-gate probe.

Why this module exists
----------------------
ADR-003 planned S3 as "IMU stream, then ``NRHeadTracking`` if it initialises".
The 2026-09-23 run stopped at ``NRImuCreate`` with result 22: every ``NRImu*``
export in ``libnr_api.dll`` first calls a licence gate that looks the module
name ``"imu"`` up in a list of licensed features and returns
``LicenseFeatureUnsupported`` when it is missing. The ``--gate`` run then showed
that list to be **empty** - ``imu`` and ``grayscale_camera`` both refused, the
ungated control fine. Raw IMU is out until a licence exists; the head-pose
family is not gated, so this module takes the fused pose instead.

The 2026-09-24 correction
-------------------------
The first version of this module passed the ``NRAPICreate`` handle to
``NRHeadTrackingCreate`` and got 2 ``InvalidArgument``, with the SDK naming the
reason itself: ``NRHeadTracking Session Invalid Argument!``. The handle these
calls want is a **perception** handle, and the perception layer that produces it
is in :mod:`xreal_perception`, recovered the same way as everything else here.
The full sequence is now:

    NRAPICreate -> InitSetNetworkType(0) -> NRAPIStart
      NRPerceptionGroupCreate -> GetCount -> per index GetPerceptionId +
        GetSupportedFeatures
        NRPerceptionCreate(id) -> NRPerceptionStart
          NRHeadTrackingCreate(perception, &tracking)
            loop: AcquireHeadPose -> GetHMDTimeNanos / GetPose /
                  GetTrackingReason -> NRHeadPoseDestroy
          NRHeadTrackingDestroy
        NRPerceptionStop -> NRPerceptionDestroy
      NRPerceptionGroupDestroy
    NRAPIStop -> NRAPIDestroy

Three levels, each a strict superset of the one before:

``gate``        one create in each of ``imu`` (gated), ``grayscale_camera``
                (gated) and ``NRHMDCreate`` (ungated control), in a single
                session. Their three codes together say whether the licensed
                feature list is empty or merely lacks ``imu``. Nothing started.

``perception``  bring-up, then enumerate the perception group and stop. This is
                the SDK listing which tracking modes these glasses offer, with
                their feature bits. Nothing is created or started beyond the
                group itself, so it is the cheapest useful run.

``pose``        the whole sequence above, polling head pose for a duration.

Signatures - recovered from the vendor's own caller, not guessed
----------------------------------------------------------------
``XREALXRPlugin.dll``'s ``NativePerception`` resolves these exports into a table
at ``[this+8]`` and calls them with handles the object holds; the export thunks
in ``libnr_api.dll`` confirm the arity by which registers they read. Note the
resolver stores the **previous** ``GetProcAddress`` result, so a naive read of
it shifts every slot by one; the slots below are confirmed against four
independent call sites.

* ``NRHeadTrackingCreate(uint64 perception, uint64* out_tracking)`` - slot
  +0xe0, called immediately after ``[NativePerception] Start Finish`` with
  ``rcx = [this+0x98]`` (the perception handle) and ``rdx = &[this+0xa0]``.
* ``NRHeadTrackingAcquireHeadPose(uint64 perception, uint64 tracking,
  uint64 hmd_time_nanos, uint64* out_pose)`` - slot +0xf0; four registers in
  the thunk, four set in ``[NativePerception] GetHeadPose: %llu``.
* ``NRHeadPoseGetPose(uint64 perception, uint64 pose, float out[16])`` - slot
  +0x110; the caller zeroes **64 bytes** at the out-pointer and the thunk
  copies four xmmwords back, returning early if it is NULL. The 16 floats are
  recorded raw; ``headpose_analysis`` decides what they are.
* ``NRHeadPoseGetHMDTimeNanos`` +0x138, ``GetTrackingReason`` +0x118 (one
  dword), ``NRHeadPoseDestroy`` +0x120, ``NRHeadTrackingDestroy`` +0xe8.

Still not known: what ``hmd_time_nanos`` wants, so the probe takes a list of
candidates and starts with 0.

Ownership: a pose handle is acquired, read and destroyed inside one loop
iteration, exactly as the vendor's wrapper does. It is never kept.

Logging: the first iteration is recorded call-by-call like S2, so the JSONL
keeps one fully annotated example; the rest of the loop calls the bound
functions directly and only counts failures, because 20 s at 100 Hz would
otherwise write 10000 call records.
"""

from __future__ import annotations

import csv
import ctypes
import sys
import time
from ctypes import POINTER, c_float, c_int32, c_uint64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import xreal_native
import xreal_perception
import xreal_result_codes
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
from xreal_perception import (
    enumerate_group,
    group_scope,
    hmd_time_ns,
    started_perception,
    supported_features,
)
from xreal_result_codes import describe, describe_reason

#: ``NRHeadPoseGetPose`` writes 64 bytes. Kept raw; see the module docstring.
POSE_FLOATS = 16
NRPoseRaw = c_float * POSE_FLOATS

_POSE_ARGS = (NRHandle, NRHandle, POINTER(NRPoseRaw))

HEADPOSE_SIGNATURES: dict[str, Signature] = {
    **xreal_native.SIGNATURES,
    **xreal_perception.PERCEPTION_SIGNATURES,
    "NRHeadTrackingCreate": Signature(
        c_int32, (NRHandle, POINTER(NRHandle)),
        "rcx perception, rdx out; vendor passes [this+0x98] and &[this+0xa0]"),
    "NRHeadTrackingDestroy": Signature(
        c_int32, (NRHandle, NRHandle), "rcx perception, rdx tracking"),
    "NRHeadTrackingAcquireHeadPose": Signature(
        c_int32, (NRHandle, NRHandle, c_uint64, POINTER(NRHandle)),
        "rcx perception, rdx tracking, r8 hmd_time_nanos, r9 &out_pose"),
    "NRHeadPoseGetPose": Signature(
        c_int32, _POSE_ARGS, "rcx perception, rdx pose, r8 out - 64 bytes"),
    "NRHeadPoseGetHMDTimeNanos": Signature(
        c_int32, (NRHandle, NRHandle, POINTER(c_uint64)), "writes one qword"),
    "NRHeadPoseGetTrackingReason": Signature(
        c_int32, (NRHandle, NRHandle, POINTER(c_int32)), "writes one dword"),
    "NRHeadPoseDestroy": Signature(
        c_int32, (NRHandle, NRHandle), "rcx perception, rdx pose"),
}

#: Exports reported but never called, from both layers.
RESOLVE_ONLY = (xreal_native.RESOLVE_ONLY_EXPORTS
                + xreal_perception.PERCEPTION_RESOLVE_ONLY)

#: The gate discriminator needs none of the perception layer.
GATE_SIGNATURES: dict[str, Signature] = {
    **xreal_native.SIGNATURES,
    "NRImuCreate": Signature(c_int32, (POINTER(NRHandle),), 'gated on "imu"'),
    "NRImuDestroy": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRGrayscaleCameraCreate": Signature(
        c_int32, (POINTER(NRHandle),), 'gated on "grayscale_camera"'),
    "NRGrayscaleCameraDestroy": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRHMDCreate": Signature(c_int32, (POINTER(NRHandle),), "ungated control"),
    "NRHMDDestroy": Signature(c_int32, (NRHandle,), "rcx only"),
}

#: ``(label, create, destroy, gated)`` - the order they are called in.
GATE_PROBES = (
    ("imu", "NRImuCreate", "NRImuDestroy", True),
    ("grayscale_camera", "NRGrayscaleCameraCreate", "NRGrayscaleCameraDestroy", True),
    ("hmd", "NRHMDCreate", "NRHMDDestroy", False),
)

CSV_COLUMNS = ("host_ns", "hmd_ns", "tracking_reason") + tuple(
    f"m{i}" for i in range(POSE_FLOATS))

#: Poll target. The delivered rate is what the recording measures.
DEFAULT_RATE_HZ = 100.0

#: ``hmd_time_nanos`` values to try, in order, until one acquire succeeds.
#: Only used by ``clock="zero"``.
DEFAULT_TIME_CANDIDATES = (0,)

#: Where the acquire timestamp comes from. ``"hmd"`` reads the perception's own
#: clock before each acquire; ``"zero"`` passes a literal from the candidate
#: list. Run 2 (2026-09-25) used ``"zero"`` and got 1925 identity poses stamped
#: 0 with tracking reason 1 (Initializing), so ``"hmd"`` is now the default.
CLOCKS = ("hmd", "zero")
DEFAULT_CLOCK = "hmd"

#: Give up on a loop that has stopped producing rather than spin for 20 s.
MAX_CONSECUTIVE_FAILURES = 20

#: The 4x4 identity - what every sample of runs 2 and 3 contained.
_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def is_identity(floats, tol: float = 1e-6) -> bool:
    """Whether a pose payload is the identity transform - i.e. carries nothing."""
    if len(floats) != POSE_FLOATS:
        return False
    return all(abs(a - b) <= tol for a, b in zip(floats, _IDENTITY))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class GateReport(NativeReport):
    """Three creates in one session; the verdict is a pure function of them."""

    codes: dict[str, Optional[int]] = field(default_factory=dict)
    verdict: str = ""

    @property
    def ok(self) -> bool:
        # "ok" means the discriminator ran, not that the IMU is licensed: the
        # ungated control must succeed for the other two codes to mean anything.
        return (self.loaded and not self.missing_required and not self.error
                and self.started and self.codes.get("hmd") == NR_SUCCESS)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["ok"] = self.ok
        out["named_codes"] = {k: describe(v) for k, v in self.codes.items()}
        return out


@dataclass
class PerceptionReport(NativeReport):
    """What the glasses offer: one entry per perception in the group."""

    group_handle: Optional[str] = None
    perceptions: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.loaded and not self.missing_required and not self.error
                and self.started and bool(self.perceptions))

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["ok"] = self.ok
        return out


@dataclass
class HeadPoseReport(NativeReport):
    duration_s: float = 0.0
    samples: int = 0
    acquire_failures: int = 0
    getter_failures: int = 0
    destroy_failures: int = 0
    group_handle: Optional[str] = None
    perceptions: list = field(default_factory=list)
    perception_id: Optional[int] = None
    perception_handle: Optional[str] = None
    tracking_handle: Optional[str] = None
    tracking_reasons: dict[str, int] = field(default_factory=dict)
    time_argument: Optional[int] = None
    acquire_code: Optional[int] = None
    clock: str = DEFAULT_CLOCK
    offset_ns: int = 0
    perception_clock_ns: Optional[int] = None
    perception_features_mask: Optional[int] = None
    identity_samples: int = 0
    csv_path: Optional[str] = None

    @property
    def delivered_data(self) -> bool:
        """At least one pose that is not the do-nothing identity.

        Run 2 recorded 1925 samples with no failures and reported ``ok``, which
        was misleading: every pose was the identity, so the run delivered
        nothing. Recording samples is not the same as receiving data.
        """
        return self.samples > 0 and self.identity_samples < self.samples

    @property
    def ok(self) -> bool:
        return (self.loaded and not self.missing_required and not self.error
                and self.started and self.delivered_data)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["ok"] = self.ok
        out["acquire_result"] = describe(self.acquire_code)
        out["tracking_reason_names"] = {describe_reason(int(k)): v
                                        for k, v in self.tracking_reasons.items()}
        out["delivered_data"] = self.delivered_data
        return out


@dataclass
class SweepReport(NativeReport):
    """One acquire through every perception in the group, for comparison."""

    group_handle: Optional[str] = None
    clock: str = DEFAULT_CLOCK
    sweep: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.loaded and not self.missing_required and not self.error
                and self.started and bool(self.sweep))

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["ok"] = self.ok
        return out


def classify_gate(codes: dict[str, Optional[int]]) -> str:
    """What the three codes jointly mean. Pure; the probe only prints it."""
    imu = codes.get("imu")
    gray = codes.get("grayscale_camera")
    hmd = codes.get("hmd")
    unsupported = xreal_result_codes.LICENSE_FEATURE_UNSUPPORTED
    if hmd != NR_SUCCESS:
        return ("inconclusive: the ungated control NRHMDCreate also failed with "
                f"{describe(hmd)}, so the gate is not the only thing in the way")
    if imu == NR_SUCCESS:
        return "the imu feature is licensed in this session - the S3 IMU path is open"
    if imu == unsupported and gray == unsupported:
        return ("licensed-feature list is empty: every gated family is refused, so "
                "no licence is in force in this session")
    if imu == unsupported and gray == NR_SUCCESS:
        return ("per-feature licence: grayscale_camera is licensed and imu is not, "
                "so a list exists and excludes imu")
    return (f"inconclusive: imu={describe(imu)}, "
            f"grayscale_camera={describe(gray)}, hmd={describe(hmd)}")


# ---------------------------------------------------------------------------
# Shared bring-up
# ---------------------------------------------------------------------------


def _open(report: NativeReport, entry: str, directory: Optional[Path],
          lib: Any, signatures: dict[str, Signature]) -> Optional[dict]:
    """Preflight, load, bind. ``None`` if the report already says why not."""
    lib = open_library(report, entry, directory, lib)
    if lib is None:
        return None
    functions, report.resolved = bind(lib, signatures, RESOLVE_ONLY)
    if report.missing_required:
        report.error = "required exports missing: " + ", ".join(report.missing_required)
        return None
    return functions


# ---------------------------------------------------------------------------
# Level: gate
# ---------------------------------------------------------------------------


def probe_gate(entry: str = DEFAULT_ENTRY, directory: Optional[Path] = None,
               lib: Any = None) -> GateReport:
    """Call one create in each family and destroy whatever was created."""
    report = GateReport(level="gate", entry=entry, platform=sys.platform,
                        required=list(GATE_SIGNATURES))
    functions = _open(report, entry, directory, lib, GATE_SIGNATURES)
    if functions is None:
        return report
    call = Recorder(functions, report)
    with started_api(call, report) as api:
        if api is None:
            return report
        for label, create, destroy, _gated in GATE_PROBES:
            handle = NRHandle(0)
            rec = call(create, ctypes.pointer(handle), shown=[f"&{label}"])
            report.codes[label] = rec.result
            if rec.succeeded:
                call(destroy, handle, shown=[label])
    report.verdict = classify_gate(report.codes)
    return report


# ---------------------------------------------------------------------------
# Level: perception
# ---------------------------------------------------------------------------


def probe_perception(entry: str = DEFAULT_ENTRY, directory: Optional[Path] = None,
                     lib: Any = None) -> PerceptionReport:
    """Enumerate the perception group and stop. Creates nothing else."""
    report = PerceptionReport(level="perception", entry=entry,
                              platform=sys.platform,
                              required=list(HEADPOSE_SIGNATURES))
    functions = _open(report, entry, directory, lib, HEADPOSE_SIGNATURES)
    if functions is None:
        return report
    call = Recorder(functions, report)
    with started_api(call, report) as api:
        if api is None:
            return report
        with group_scope(call) as group:
            if group is None:
                report.error = "NRPerceptionGroupCreate failed"
                return report
            report.group_handle = f"{group.value:#x}"
            report.perceptions = [i.to_dict() for i in enumerate_group(call, group)]
    if not report.perceptions and not report.error:
        report.error = "the perception group reported no entries"
    return report


# ---------------------------------------------------------------------------
# Level: pose
# ---------------------------------------------------------------------------


def _read_pose(functions: dict[str, Any], perception: NRHandle, pose: NRHandle
               ) -> tuple[int, int, list, int]:
    """Read one pose. Returns ``(hmd_ns, reason, floats, getter_failures)``."""
    hmd = c_uint64(0)
    reason = c_int32(0)
    raw = NRPoseRaw()
    codes = (
        functions["NRHeadPoseGetHMDTimeNanos"](perception, pose, ctypes.pointer(hmd)),
        functions["NRHeadPoseGetPose"](perception, pose, ctypes.pointer(raw)),
        functions["NRHeadPoseGetTrackingReason"](perception, pose,
                                                 ctypes.pointer(reason)),
    )
    failures = sum(1 for c in codes if c != NR_SUCCESS)
    return hmd.value, reason.value, list(raw), failures


def _time_source(functions: dict[str, Any], perception: NRHandle, clock: str,
                 offset_ns: int, candidate: int) -> Callable[[], Optional[int]]:
    """What to pass as ``hmd_time_nanos``, per acquire."""
    if clock == "hmd":
        return lambda: (lambda ns: None if ns is None else ns + offset_ns)(
            hmd_time_ns(functions, perception))
    return lambda: candidate


def _first_iteration(call: Recorder, perception: NRHandle, tracking: NRHandle,
                     time_ns: int) -> tuple[Optional[tuple], Optional[int], int]:
    """One fully recorded iteration: the JSONL keeps an annotated example.

    Returns ``(row, result_code, getter_failures)``; ``row`` is ``None`` when
    the acquire itself failed, which is the case worth reading the code for.
    """
    pose = NRHandle(0)
    rec = call("NRHeadTrackingAcquireHeadPose", perception, tracking, time_ns,
               ctypes.pointer(pose),
               shown=["perception", "tracking", time_ns, "&pose"])
    if not rec.succeeded:
        return None, rec.result, 0
    hmd, reason, raw = c_uint64(0), c_int32(0), NRPoseRaw()
    failures = 0
    for fname, out, shown in (
            ("NRHeadPoseGetHMDTimeNanos", ctypes.pointer(hmd), "&hmd_ns"),
            ("NRHeadPoseGetPose", ctypes.pointer(raw), "&pose[16]"),
            ("NRHeadPoseGetTrackingReason", ctypes.pointer(reason), "&reason")):
        if not call(fname, perception, pose, out,
                    shown=["perception", "pose", shown]).succeeded:
            failures += 1
    if not call("NRHeadPoseDestroy", perception, pose,
                shown=["perception", "pose"]).succeeded:
        failures += 1
    row = (time.perf_counter_ns(), hmd.value, reason.value) + tuple(raw)
    return row, rec.result, failures


def _loop(functions: dict[str, Any], report: HeadPoseReport, perception: NRHandle,
          tracking: NRHandle, time_source: Callable[[], Optional[int]],
          duration_s: float, interval_s: float, rows: list) -> None:
    acquire = functions["NRHeadTrackingAcquireHeadPose"]
    destroy = functions["NRHeadPoseDestroy"]
    reasons: dict[str, int] = {}
    consecutive = 0
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        pose = NRHandle(0)
        time_ns = time_source()
        if time_ns is None:
            report.acquire_failures += 1
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                report.error = "the perception clock stopped answering"
                break
            time.sleep(interval_s)
            continue
        if acquire(perception, tracking, time_ns, ctypes.pointer(pose)) != NR_SUCCESS:
            report.acquire_failures += 1
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                report.error = (f"{consecutive} consecutive acquire failures; "
                                "stopped early")
                break
            time.sleep(interval_s)
            continue
        consecutive = 0
        host_ns = time.perf_counter_ns()
        hmd_ns, reason, floats, failures = _read_pose(functions, perception, pose)
        report.getter_failures += failures
        if destroy(perception, pose) != NR_SUCCESS:
            report.destroy_failures += 1
        if is_identity(floats):
            report.identity_samples += 1
        rows.append((host_ns, hmd_ns, reason) + tuple(floats))
        key = str(reason)
        reasons[key] = reasons.get(key, 0) + 1
        time.sleep(interval_s)
    report.tracking_reasons = reasons


def _with_tracking(call: Recorder, functions: dict[str, Any],
                   report: HeadPoseReport, perception: NRHandle,
                   duration_s: float, interval_s: float, time_candidates: tuple,
                   clock: str, offset_ns: int, rows: list,
                   on_started: Optional[Callable[[], None]],
                   on_finished: Optional[Callable[[], None]]) -> None:
    tracking = NRHandle(0)
    created = call("NRHeadTrackingCreate", perception, ctypes.pointer(tracking),
                   shown=["perception", "&tracking"])
    if not created.succeeded:
        report.error = "NRHeadTrackingCreate failed"
        return
    report.tracking_handle = f"{tracking.value:#x}"
    try:
        row = None
        candidates = (None,) if clock == "hmd" else time_candidates
        for candidate in candidates:
            source = _time_source(functions, perception, clock, offset_ns,
                                  candidate or 0)
            stamp = source()
            if stamp is None:
                report.error = "NRPerceptionGetHMDTimeNanos failed"
                return
            row, code, failures = _first_iteration(call, perception, tracking,
                                                   stamp)
            report.time_argument = stamp
            report.acquire_code = code
            report.getter_failures += failures
            if row is not None:
                break
            report.acquire_failures += 1
        if row is None:
            report.error = ("NRHeadTrackingAcquireHeadPose failed for every "
                            "hmd_time_nanos candidate")
            return
        rows.append(row)
        if is_identity(row[3:]):
            report.identity_samples += 1
        if on_started is not None:
            on_started()
        _loop(functions, report, perception, tracking,
              _time_source(functions, perception, clock, offset_ns,
                           report.time_argument or 0),
              duration_s, interval_s, rows)
        if on_finished is not None:
            on_finished()
    finally:
        call("NRHeadTrackingDestroy", perception, tracking,
             shown=["perception", "tracking"])


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def record(duration_s: float, csv_path: Optional[Path] = None,
           entry: str = DEFAULT_ENTRY, directory: Optional[Path] = None,
           lib: Any = None, rate_hz: float = DEFAULT_RATE_HZ,
           time_candidates: tuple = DEFAULT_TIME_CANDIDATES,
           perception_id: Optional[int] = None,
           clock: str = DEFAULT_CLOCK, offset_ns: int = 0,
           on_started: Optional[Callable[[], None]] = None,
           on_finished: Optional[Callable[[], None]] = None) -> HeadPoseReport:
    """Poll head pose for ``duration_s``. Never raises for expected failures.

    ``perception_id`` pins one perception; the default enumerates the group and
    takes the first one that creates and starts.
    """
    report = HeadPoseReport(level="headpose", entry=entry, platform=sys.platform,
                            duration_s=duration_s, clock=clock, offset_ns=offset_ns,
                            required=list(HEADPOSE_SIGNATURES))
    functions = _open(report, entry, directory, lib, HEADPOSE_SIGNATURES)
    if functions is None:
        return report

    interval_s = 0.0 if rate_hz <= 0 else 1.0 / rate_hz
    rows: list = []
    call = Recorder(functions, report)
    with started_api(call, report) as api:
        if api is not None:
            _run_perception(call, functions, report, duration_s, interval_s,
                            tuple(time_candidates), perception_id, clock,
                            offset_ns, rows, on_started, on_finished)

    report.samples = len(rows)
    if csv_path is not None and rows:
        write_csv(csv_path, rows)
        report.csv_path = str(csv_path)
    if report.started and not rows and not report.error:
        report.error = "head tracking started but delivered no pose"
    elif rows and report.identity_samples == len(rows) and not report.error:
        report.error = (f"all {len(rows)} poses were the identity transform: the "
                        "calls succeed but the tracker is not producing")
    return report


def _run_perception(call: Recorder, functions: dict[str, Any],
                    report: HeadPoseReport, duration_s: float, interval_s: float,
                    time_candidates: tuple, perception_id: Optional[int],
                    clock: str, offset_ns: int, rows: list,
                    on_started: Optional[Callable[[], None]],
                    on_finished: Optional[Callable[[], None]]) -> None:
    """Group -> enumerate -> first perception that starts -> head tracking."""
    with group_scope(call) as group:
        if group is None:
            report.error = "NRPerceptionGroupCreate failed"
            return
        report.group_handle = f"{group.value:#x}"
        infos = enumerate_group(call, group)
        report.perceptions = [i.to_dict() for i in infos]
        if perception_id is not None:
            candidates = [perception_id]
        else:
            candidates = [i.perception_id for i in infos
                          if i.perception_id is not None]
        if not candidates:
            report.error = ("the perception group offered no usable id; run "
                            "--perception to see what it reported")
            return
        for pid in candidates:
            with started_perception(call, pid) as perception:
                if perception is None:
                    continue
                report.perception_id = pid
                report.perception_handle = f"{perception.value:#x}"
                report.perception_clock_ns = hmd_time_ns(functions, perception)
                report.perception_features_mask = supported_features(functions,
                                                                     perception)
                _with_tracking(call, functions, report, perception, duration_s,
                               interval_s, time_candidates, clock, offset_ns,
                               rows, on_started, on_finished)
                return
        report.error = ("no perception in the group could be created and started "
                        f"(tried {candidates})")


# ---------------------------------------------------------------------------
# Level: sweep
# ---------------------------------------------------------------------------

def _probe_one_perception(call: Recorder, functions: dict[str, Any],
                          perception: NRHandle, clock: str, offset_ns: int
                          ) -> dict[str, Any]:
    """Create head tracking, take one pose, tear it down. Never raises."""
    out: dict[str, Any] = {
        "clock_ns": hmd_time_ns(functions, perception),
        "features_mask": supported_features(functions, perception),
    }
    tracking = NRHandle(0)
    created = call("NRHeadTrackingCreate", perception, ctypes.pointer(tracking),
                   shown=["perception", "&tracking"])
    out["head_tracking_create"] = describe(created.result)
    if not created.succeeded:
        return out
    try:
        source = _time_source(functions, perception, clock, offset_ns, 0)
        stamp = source()
        out["time_argument"] = stamp
        if stamp is None:
            out["acquire"] = "clock unavailable"
            return out
        pose = NRHandle(0)
        rec = call("NRHeadTrackingAcquireHeadPose", perception, tracking, stamp,
                   ctypes.pointer(pose), shown=["perception", "tracking", stamp,
                                                "&pose"])
        out["acquire"] = describe(rec.result)
        if not rec.succeeded:
            return out
        hmd_ns, reason, floats, failures = _read_pose(functions, perception, pose)
        out.update({"hmd_ns": hmd_ns, "reason": describe_reason(reason),
                    "identity_pose": is_identity(floats),
                    "getter_failures": failures})
        functions["NRHeadPoseDestroy"](perception, pose)
    finally:
        call("NRHeadTrackingDestroy", perception, tracking,
             shown=["perception", "tracking"])
    return out


def sweep_perceptions(entry: str = DEFAULT_ENTRY, directory: Optional[Path] = None,
                      lib: Any = None, clock: str = DEFAULT_CLOCK,
                      offset_ns: int = 0) -> SweepReport:
    """One pose through every perception in the group, side by side.

    Run 2 took the first id the group offered (1) and got an identity pose with
    tracking reason ``Initializing``. Twenty ids were on offer and all reported a
    feature mask of 0, so "the first one" was a guess. This tries them all in one
    session and prints the comparison, which is the cheapest way to find out
    whether any of them tracks.
    """
    report = SweepReport(level="sweep", entry=entry, platform=sys.platform,
                         clock=clock, required=list(HEADPOSE_SIGNATURES))
    functions = _open(report, entry, directory, lib, HEADPOSE_SIGNATURES)
    if functions is None:
        return report
    call = Recorder(functions, report)
    with started_api(call, report) as api:
        if api is None:
            return report
        with group_scope(call) as group:
            if group is None:
                report.error = "NRPerceptionGroupCreate failed"
                return report
            report.group_handle = f"{group.value:#x}"
            for info in enumerate_group(call, group):
                if info.perception_id is None:
                    continue
                row: dict[str, Any] = {"index": info.index,
                                       "perception_id": info.perception_id,
                                       "group_features_mask": info.features_mask}
                with started_perception(call, info.perception_id) as perception:
                    row["started"] = perception is not None
                    if perception is not None:
                        row.update(_probe_one_perception(call, functions, perception,
                                                         clock, offset_ns))
                report.sweep.append(row)
    if not report.sweep and not report.error:
        report.error = "the perception group offered nothing to sweep"
    return report
