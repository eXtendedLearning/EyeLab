"""Frame identity, uniqueness and freshness (audit F03)."""
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from frame_packet import FramePacket, ReplayFrameSource, make_packets, new_session_id
from pose_estimator import ArucoPipeline, PoseResult, ThreadedCapture
from pose_lock import LOCK_LOCKED, LOCK_SEARCHING


def _img(value=0):
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _pipeline(source, **kw):
    pipe = ArucoPipeline(**kw)
    pipe._capture = source
    return pipe


def _fresh(seq, session="s1", age=0.0, value=0):
    return FramePacket(_img(value), "test", session, seq, time.perf_counter() - age)


class FramePacketTests(unittest.TestCase):
    def test_image_is_read_only(self):
        p = _fresh(1)
        with self.assertRaises(ValueError):
            p.image[0, 0, 0] = 1

    def test_packet_is_immutable(self):
        p = _fresh(1)
        with self.assertRaises(Exception):
            p.sequence = 2

    def test_age_uses_the_host_clock(self):
        p = _fresh(1, age=0.25)
        self.assertGreaterEqual(p.age_s(), 0.25)


class PipelineUniquenessTests(unittest.TestCase):
    def test_frozen_source_is_processed_exactly_once(self):
        # The audit case: one injected frame, 100 polls.
        src = ReplayFrameSource([_img()], repeat_last=True)
        pipe = _pipeline(src)
        results = [pipe.process_frame() for _ in range(100)]
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(pipe.frame_counts["processed"], 1)
        self.assertEqual(pipe.frame_counts["idle_polls"], 99)

    def test_frozen_source_updates_the_filter_once(self):
        src = ReplayFrameSource([_img()], repeat_last=True)
        pipe = _pipeline(src)
        with patch.object(pipe.pose_lock, "process", wraps=pipe.pose_lock.process) as spy:
            for _ in range(100):
                pipe.process_frame()
        # board-less pipeline does not use the lock; the Kalman path runs only
        # on a measured pose, so what matters is that no frame is re-processed
        self.assertEqual(pipe.frame_counts["processed"], 1)
        self.assertLessEqual(spy.call_count, 1)

    def test_duplicates_delayed_and_out_of_order_frames(self):
        seqs = [1, 2, 2, 3, 3, 2, 5, 4, 6]
        packets = [_fresh(s) for s in seqs]
        src = ReplayFrameSource(packets)
        pipe = _pipeline(src)
        got = []
        for _ in seqs:
            r = pipe.process_frame()
            if r is not None:
                got.append(r.sequence)
        self.assertEqual(got, [1, 2, 3, 5, 6])
        # repeats of 2 and 3 are idle polls; the late 2 and 4 are out of order;
        # 5 skipped a number (4), which is counted as missed.
        self.assertEqual(pipe.frame_counts["idle_polls"], 2)
        self.assertEqual(pipe.frame_counts["out_of_order"], 2)
        self.assertEqual(pipe.frame_counts["missed"], 1)
        self.assertEqual(len(set(got)), len(got))

    def test_result_carries_the_frame_identity(self):
        src = ReplayFrameSource(make_packets([_img(7)], t0=time.perf_counter()))
        pipe = _pipeline(src)
        r = pipe.process_frame()
        self.assertEqual(r.sequence, 1)
        self.assertEqual(r.source_id, "test")
        self.assertTrue(r.session_id)
        self.assertTrue(np.all(r.frame == 7))
        self.assertFalse(r.frame.flags.writeable)

    def test_stale_frame_is_dropped_and_counted(self):
        src = ReplayFrameSource([_fresh(1, age=5.0), _fresh(2)])
        pipe = _pipeline(src, max_frame_age_s=0.5)
        self.assertIsNone(pipe.process_frame())
        self.assertEqual(pipe.last_skip_reason, "stale")
        self.assertEqual(pipe.process_frame().sequence, 2)
        self.assertEqual(pipe.frame_counts["stale"], 1)

    def test_age_deadline_does_not_need_reader_progress(self):
        # A blocked reader never delivers again; the consumer-side age check
        # must still reject what it last delivered once it has aged out.
        old = _fresh(1, age=0.0)
        src = ReplayFrameSource([old], repeat_last=True)
        pipe = _pipeline(src, max_frame_age_s=0.05)
        self.assertIsNotNone(pipe.process_frame())
        time.sleep(0.1)
        self.assertIsNone(pipe.process_frame())

    def test_session_change_resets_history_and_restarts_sequences(self):
        packets = [_fresh(5, session="a"), _fresh(6, session="a"), _fresh(1, session="b")]
        pipe = _pipeline(ReplayFrameSource(packets))
        seqs, trackers = [], []
        for _ in packets:
            seqs.append(pipe.process_frame().sequence)
            trackers.append(pipe.of_tracker)
        self.assertEqual(seqs, [5, 6, 1])
        self.assertIs(trackers[0], trackers[1])        # same session: state kept
        self.assertIsNot(trackers[1], trackers[2])     # a -> b: tracker rebuilt
        self.assertEqual(pipe.frame_counts["sessions"], 2)

    def test_frozen_source_cannot_hold_a_lock(self):
        pipe = _pipeline(ReplayFrameSource([_fresh(1, age=10.0)], repeat_last=True),
                         max_frame_age_s=0.5)
        pose = PoseResult(
            rvec=np.zeros((3, 1)), tvec=np.array([[0.0], [0.0], [0.3]]),
            marker_ids=[0, 1, 2], marker_count=3, rms_reproj_px=0.5,
        )
        pipe.pose_lock.process(pose, now=time.perf_counter() - 10.0)
        self.assertEqual(pipe.pose_lock.state, LOCK_LOCKED)
        for _ in range(3):
            self.assertIsNone(pipe.process_frame())
        self.assertEqual(pipe.pose_lock.state, LOCK_SEARCHING)


class _SlowCap:
    def __init__(self):
        self.n = 0

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0.0

    def read(self):
        time.sleep(0.003)
        self.n += 1
        return True, np.full((4, 4, 3), self.n % 255, dtype=np.uint8)

    def release(self):
        pass


class ThreadedCapturePacketTests(unittest.TestCase):
    def test_sequences_increase_and_repeat_until_a_new_frame(self):
        capture = ThreadedCapture(camera_index=2, cap=_SlowCap()).start()
        try:
            deadline = time.perf_counter() + 2.0
            first = None
            while first is None and time.perf_counter() < deadline:
                first = capture.read_packet()
            self.assertIsNotNone(first)
            self.assertEqual(first.source_id, "camera:2")
            again = capture.read_packet()
            self.assertGreaterEqual(again.sequence, first.sequence)
            time.sleep(0.05)
            later = capture.read_packet()
            self.assertGreater(later.sequence, first.sequence)
            self.assertEqual(later.session_id, first.session_id)
            self.assertGreater(later.received_at, first.received_at)
        finally:
            capture.stop()

    def test_each_capture_has_its_own_session(self):
        a = ThreadedCapture(camera_index=0, cap=_SlowCap())
        b = ThreadedCapture(camera_index=0, cap=_SlowCap())
        self.assertNotEqual(a.session_id, b.session_id)

    def test_legacy_read_returns_a_writable_copy(self):
        capture = ThreadedCapture(camera_index=0, cap=_SlowCap()).start()
        try:
            deadline = time.perf_counter() + 2.0
            img = None
            while img is None and time.perf_counter() < deadline:
                img = capture.read()
            img[0, 0, 0] = 99  # must not raise, must not alias the packet
            self.assertNotEqual(int(capture.read_packet().image[0, 0, 0]), 99)
        finally:
            capture.stop()


if __name__ == "__main__":
    unittest.main()
