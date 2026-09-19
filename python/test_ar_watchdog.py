#!/usr/bin/env python3
"""Tests for the AR freeze watchdog (stdlib only -- no cv2, no tkinter)."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from ar_watchdog import ARWatchdog


def read_records(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def kinds(records: list[dict]) -> list[str]:
    return [r["kind"] for r in records]


class ARWatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _watchdog(self, **kwargs) -> ARWatchdog:
        params = dict(
            stall_s=0.1,
            hard_freeze_s=30.0,
            poll_s=0.02,
            heartbeat_s=0.05,
            slow_phase_ms=50.0,
        )
        params.update(kwargs)
        wd = ARWatchdog(self.log_dir, **params)
        self.addCleanup(wd.stop)
        return wd.start()

    def test_session_record_written_on_start(self) -> None:
        wd = self._watchdog()
        wd.stop()
        records = read_records(wd.jsonl_path)
        self.assertEqual(records[0]["kind"], "session")
        self.assertIn("pid", records[0])
        self.assertIn("watchdog_stop", kinds(records))

    def test_every_line_is_valid_json(self) -> None:
        wd = self._watchdog()
        with wd.phase("frame.process"):
            pass
        wd.event("odd_types", path=Path("x"), value=object())
        wd.frame(markers=2)
        wd.stop()
        records = read_records(wd.jsonl_path)          # raises if a line is broken
        self.assertIn("odd_types", kinds(records))

    def test_stall_names_the_blocking_phase(self) -> None:
        wd = self._watchdog()
        with wd.phase("frame.overlay"):
            time.sleep(0.4)
        time.sleep(0.1)
        wd.stop()
        records = read_records(wd.jsonl_path)
        stalls = [r for r in records if r["kind"] == "stall"]
        self.assertTrue(stalls, "a phase held past stall_s must produce a stall record")
        self.assertEqual(stalls[0]["phase"], "frame.overlay")
        self.assertGreater(stalls[0]["blocked_ms"], 90.0)
        self.assertTrue(stalls[0]["stacks"], "stall record must carry thread stacks")
        self.assertIn("resumed", kinds(records))

    def test_no_stall_while_ticking(self) -> None:
        wd = self._watchdog()
        deadline = time.time() + 0.4
        while time.time() < deadline:
            wd.tick()
            time.sleep(0.01)
        wd.stop()
        self.assertNotIn("stall", kinds(read_records(wd.jsonl_path)))

    def test_slow_phase_logged_separately(self) -> None:
        wd = self._watchdog(stall_s=5.0, slow_phase_ms=20.0)
        with wd.phase("frame.display"):
            time.sleep(0.05)
        wd.stop()
        slow = [r for r in read_records(wd.jsonl_path) if r["kind"] == "phase_slow"]
        self.assertEqual([r["phase"] for r in slow], ["frame.display"])

    def test_heartbeat_aggregates_stage_timings(self) -> None:
        wd = self._watchdog(stall_s=5.0, heartbeat_s=0.0)
        with wd.phase("frame.process"):
            time.sleep(0.01)
        wd.frame(markers=4, capture={"reads": 10})
        wd.stop()
        beats = [r for r in read_records(wd.jsonl_path) if r["kind"] == "heartbeat"]
        self.assertEqual(len(beats), 1)
        self.assertEqual(beats[0]["markers"], 4)
        self.assertEqual(beats[0]["capture"]["reads"], 10)
        self.assertIn("frame.process", beats[0]["stages"])
        self.assertGreaterEqual(beats[0]["stages"]["frame.process"]["max_ms"], 5.0)

    def test_attach_reschedules_itself(self) -> None:
        wd = self._watchdog()
        calls: list[int] = []

        def schedule(delay_ms: int, callback) -> None:
            calls.append(delay_ms)
            if len(calls) < 3:
                callback()

        wd.attach(schedule, interval_ms=250)
        self.assertEqual(calls, [250, 250, 250])

    def test_calls_after_stop_are_inert(self) -> None:
        wd = self._watchdog()
        wd.stop()
        wd.stop()                      # idempotent
        wd.event("after_stop")         # must not raise
        with wd.phase("late"):
            pass
        wd.frame()
        self.assertNotIn("after_stop", kinds(read_records(wd.jsonl_path)))


if __name__ == "__main__":
    unittest.main()
