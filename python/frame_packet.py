"""Immutable frame packets and the frame-source contract (audit F03).

A bare image carries no identity: a consumer that polls faster than the camera
delivers cannot tell a new frame from the one it already processed, and cannot
tell a frame from a device that stopped delivering minutes ago. A
``FramePacket`` binds the image to where it came from and when the host
received it, so every later stage (pose, recording, presentation) can state
*which* frame it used.

Timing semantics, deliberately limited to what is actually known:

- ``received_at`` is the HOST ``time.perf_counter()`` value taken when the read
  returned. It is NOT the exposure time; the interval between exposure and
  receipt (sensor, driver, USB, decode) is unmeasured.
- ``device_time_s`` / ``device_clock`` are filled only by sources that expose a
  device capture timestamp, together with the name of that clock. Never mix
  clocks without an explicit mapping.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One delivered image with its identity. The image is made read-only."""
    image: np.ndarray
    source_id: str
    session_id: str          # changes whenever the source is (re)opened
    sequence: int            # strictly increasing within a session
    received_at: float       # host time.perf_counter() at receipt (not exposure)
    image_mode: str = "bgr8"
    device_time_s: Optional[float] = None
    device_clock: Optional[str] = None

    def __post_init__(self) -> None:
        self.image.setflags(write=False)

    def age_s(self, now: Optional[float] = None) -> float:
        """Host-clock seconds since receipt."""
        return (time.perf_counter() if now is None else now) - self.received_at


@runtime_checkable
class FrameSource(Protocol):
    """What the pipeline needs from a camera or a replay.

    ``read_packet`` returns the newest packet or None when nothing has been
    delivered yet. It MAY return the same packet again when no newer frame has
    arrived; consumers detect that by ``(session_id, sequence)`` and must not
    treat it as a new measurement.
    """
    error: Optional[str]

    def read_packet(self) -> Optional[FramePacket]: ...

    def stop(self) -> None: ...


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class ReplayFrameSource:
    """Plays packets (or bare images) back in order, one per ``read_packet``.

    Bare images become packets of one session, numbered from 1 and stamped with
    the host time at which they are delivered (a replay behaves like a live
    camera), or at ``t0 + i * frame_interval_s`` when ``t0`` is given.
    ``FramePacket`` objects are delivered exactly as supplied, which lets tests
    inject duplicates, delays, gaps, out-of-order sequences and session changes.
    After the last item ``repeat_last=True`` (the camera behaviour) keeps
    returning the final packet; otherwise None.
    """

    def __init__(
        self,
        items: Iterable,
        source_id: str = "replay",
        frame_interval_s: float = 1.0 / 30.0,
        t0: Optional[float] = None,
        repeat_last: bool = False,
    ) -> None:
        self.error: Optional[str] = None
        self.is_opened = True
        self.repeat_last = repeat_last
        self.source_id = source_id
        self.session_id = new_session_id()
        self._items = list(items)
        self._dt = frame_interval_s
        self._t0 = t0
        self._pos = 0
        self._last: Optional[FramePacket] = None

    def read_packet(self) -> Optional[FramePacket]:
        if self._pos < len(self._items):
            item = self._items[self._pos]
            if isinstance(item, FramePacket):
                packet = item
            else:
                received = (
                    time.perf_counter() if self._t0 is None
                    else self._t0 + self._pos * self._dt
                )
                packet = FramePacket(
                    np.asarray(item), self.source_id, self.session_id,
                    self._pos + 1, received,
                )
            self._pos += 1
            self._last = packet
            return packet
        return self._last if self.repeat_last else None

    def stats(self) -> dict:
        return {}

    def stop(self) -> None:
        self.is_opened = False


def make_packets(
    images: Sequence[np.ndarray],
    source_id: str = "test",
    session_id: Optional[str] = None,
    first_sequence: int = 1,
    t0: float = 0.0,
    dt_s: float = 1.0 / 30.0,
) -> list[FramePacket]:
    """Convenience for tests: consecutive packets with a synthetic host clock."""
    sid = session_id or new_session_id()
    return [
        FramePacket(np.asarray(img), source_id, sid, first_sequence + i, t0 + i * dt_s)
        for i, img in enumerate(images)
    ]
