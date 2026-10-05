#!/usr/bin/env python3
"""The NRSDK perception layer - the handle everything else in S3b hangs off.

Why this module exists
----------------------
The first S3b run (2026-09-24) failed at ``NRHeadTrackingCreate`` -> 2
``InvalidArgument``, and the SDK's own log named the reason:

    [ERROR] [NRSDK] NRHeadTracking Session Invalid Argument!

The probe had passed the ``NRAPICreate`` handle. It is not the right one. The
vendor host's ``NativePerception`` object holds a *perception* handle and
passes that; audit section 11 recorded the handle's origin as the one thing
not recovered statically. It is recovered now: between ``NRAPIStart`` and
``NRHeadTrackingCreate`` the vendor brings up a whole family that ADR-003
never mentioned, and none of it is behind the licence gate.

The order, from ``XREALXRPlugin.dll``'s function table (slots +0x10..+0x68,
read with the resolver's off-by-one corrected against four independent call
sites):

1. ``NRPerceptionGroupCreate(&group)``
2. ``NRPerceptionGroupGetCount(group, &count)``
3. per index: ``NRPerceptionGroupGetPerceptionId(group, i, &id)`` and
   ``NRPerceptionGroupGetSupportedFeatures(group, i, &mask)``
4. ``NRPerceptionCreate(id, &perception)`` - note the id is **32-bit**: the
   thunk does ``mov edx, ecx``, not ``mov rdx, rcx``
5. ``NRPerceptionStart(perception)``

and only then ``NRHeadTrackingCreate(perception, &tracking)``. Teardown runs
in reverse: ``NRPerceptionStop`` -> ``NRPerceptionDestroy`` ->
``NRPerceptionGroupDestroy``.

Step 3 is worth having for its own sake: it is the SDK telling us which
tracking modes this pair of glasses actually offers, and with which feature
bits. That enumeration is the answer to "what is available", and it costs one
short run with nothing started.

Signatures, from the export thunks in ``libnr_api.dll`` (which registers each
one reads) cross-checked against the vendor's call sites:

===================================================  ==========================================
``NRPerceptionGroupCreate``                          ``int32 (uint64* out_group)``
``NRPerceptionGroupGetCount``                        ``int32 (uint64 group, int32* out_count)``
``NRPerceptionGroupGetPerceptionId``                 ``int32 (uint64 group, int32 index, uint32* out_id)``
``NRPerceptionGroupGetSupportedFeatures``            ``int32 (uint64 group, int32 index, uint64* out_mask)``
``NRPerceptionGroupDestroy``                         ``int32 (uint64 group)``
``NRPerceptionCreate``                               ``int32 (int32 perception_id, uint64* out_perception)``
``NRPerceptionStart`` / ``Stop`` / ``Pause`` / ``Resume`` / ``Destroy``  ``int32 (uint64 perception)``
``NRPerceptionGetHMDTimeNanos``                      ``int32 (uint64 perception, uint64* out_ns)``
===================================================  ==========================================

``NRPerceptionGroupGetDescription`` would give each perception a human-readable
name, but its buffer-length argument is not pinned down, so it is resolved and
reported and never called - a buffer written by a vendor DLL on a guessed
length is not worth the risk.

Everything here takes an ``xreal_native.Recorder``, so each call lands in the
caller's report exactly like S2's.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from ctypes import POINTER, c_int32, c_uint32, c_uint64
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Optional

from xreal_native import NRHandle, Recorder, Signature

PERCEPTION_SIGNATURES: dict[str, Signature] = {
    "NRPerceptionGroupCreate": Signature(
        c_int32, (POINTER(NRHandle),), "rcx out-pointer only"),
    "NRPerceptionGroupGetCount": Signature(
        c_int32, (NRHandle, POINTER(c_int32)), "rcx group, rdx out"),
    "NRPerceptionGroupGetPerceptionId": Signature(
        c_int32, (NRHandle, c_int32, POINTER(c_uint32)),
        "rcx group, edx index (32-bit), r8 out"),
    "NRPerceptionGroupGetSupportedFeatures": Signature(
        c_int32, (NRHandle, c_int32, POINTER(c_uint64)),
        "rcx group, edx index (32-bit), r8 out"),
    "NRPerceptionGroupDestroy": Signature(
        c_int32, (NRHandle,), "rcx only"),
    "NRPerceptionCreate": Signature(
        c_int32, (c_int32, POINTER(NRHandle)),
        "ecx perception id (32-bit: thunk does mov edx, ecx), rdx out"),
    "NRPerceptionStart": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRPerceptionStop": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRPerceptionDestroy": Signature(c_int32, (NRHandle,), "rcx only"),
    "NRPerceptionGetHMDTimeNanos": Signature(
        c_int32, (NRHandle, POINTER(c_uint64)), "rcx perception, rdx out"),
    "NRPerceptionGetSupportedFeatures": Signature(
        c_int32, (NRHandle, POINTER(c_uint64)), "rcx perception, rdx out"),
}

#: Resolved and reported, never called - see the module docstring.
PERCEPTION_RESOLVE_ONLY = (
    "NRPerceptionGroupGetDescription",   # buffer length argument not pinned down
    "NRPerceptionPause",
    "NRPerceptionResume",
    "NRPerceptionGroupCheckPerceptionType",
    "NRConfigCreate",
)

#: A group with more entries than this means the count is not a count.
MAX_PERCEPTIONS = 64


@dataclass
class PerceptionInfo:
    index: int
    perception_id: Optional[int] = None
    features_mask: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        mask = self.features_mask
        out["features_bits"] = ([i for i in range(64) if mask >> i & 1]
                                if mask is not None else None)
        return out


@contextmanager
def group_scope(call: Recorder) -> Iterator[Optional[NRHandle]]:
    """``NRPerceptionGroupCreate`` around a block; always destroys the group."""
    group = NRHandle(0)
    created = call("NRPerceptionGroupCreate", ctypes.pointer(group),
                   shown=["&group"])
    if not created.succeeded:
        yield None
        return
    try:
        yield group
    finally:
        call("NRPerceptionGroupDestroy", group, shown=["group"])


def enumerate_group(call: Recorder, group: NRHandle) -> list[PerceptionInfo]:
    """Every perception the glasses offer, with its id and feature mask."""
    count = c_int32(0)
    if not call("NRPerceptionGroupGetCount", group, ctypes.pointer(count),
                shown=["group", "&count"]).succeeded:
        return []
    total = max(0, min(int(count.value), MAX_PERCEPTIONS))
    found: list[PerceptionInfo] = []
    for index in range(total):
        info = PerceptionInfo(index=index)
        pid = c_uint32(0)
        if call("NRPerceptionGroupGetPerceptionId", group, index,
                ctypes.pointer(pid), shown=["group", index, "&id"]).succeeded:
            info.perception_id = int(pid.value)
        else:
            info.error = "NRPerceptionGroupGetPerceptionId failed"
        mask = c_uint64(0)
        if call("NRPerceptionGroupGetSupportedFeatures", group, index,
                ctypes.pointer(mask), shown=["group", index, "&features"]).succeeded:
            info.features_mask = int(mask.value)
        found.append(info)
    return found


@contextmanager
def started_perception(call: Recorder, perception_id: int
                       ) -> Iterator[Optional[NRHandle]]:
    """Create and start one perception; always stop and destroy it.

    Yields the handle once ``NRPerceptionStart`` succeeded, else ``None``.
    """
    perception = NRHandle(0)
    created = call("NRPerceptionCreate", perception_id,
                   ctypes.pointer(perception), shown=[perception_id, "&perception"])
    if not created.succeeded:
        yield None
        return
    started = False
    try:
        started = call("NRPerceptionStart", perception, shown=["perception"]).succeeded
        yield perception if started else None
    finally:
        if started:
            call("NRPerceptionStop", perception, shown=["perception"])
        call("NRPerceptionDestroy", perception, shown=["perception"])

def hmd_time_ns(functions: dict[str, Any], perception: NRHandle) -> Optional[int]:
    """The perception's own clock, or ``None`` if the call failed.

    This exists because run 2 (2026-09-25) passed ``hmd_time_nanos = 0`` to
    ``NRHeadTrackingAcquireHeadPose``, got 1925 identity poses stamped 0, and the
    SDK exposes this getter for exactly the value the acquire wants. Not recorded
    through a ``Recorder``: it is called once per loop iteration.
    """
    out = c_uint64(0)
    fn = functions.get("NRPerceptionGetHMDTimeNanos")
    if fn is None:
        return None
    if fn(perception, ctypes.pointer(out)) != 0:
        return None
    return int(out.value)


def supported_features(functions: dict[str, Any], perception: NRHandle) -> Optional[int]:
    """The feature mask of a *started* perception, not of a group entry.

    Every group entry reported mask 0 in run 2; this asks the live object
    instead, which is a different call on a different handle.
    """
    out = c_uint64(0)
    fn = functions.get("NRPerceptionGetSupportedFeatures")
    if fn is None:
        return None
    if fn(perception, ctypes.pointer(out)) != 0:
        return None
    return int(out.value)
