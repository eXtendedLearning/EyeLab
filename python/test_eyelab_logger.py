#!/usr/bin/env python3
"""Tests for session-log retention (``eyelab_logger.prune_logs``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eyelab_logger import KEEP_FILE, pinned_logs, prune_logs


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_text("{}\n", encoding="utf-8")


class PruneLogsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def names(self):
        return sorted(p.name for p in self.dir.iterdir())

    def test_keeps_newest_per_family(self):
        _touch(self.dir, *(f"eyelab_2026090{i}_120000.jsonl" for i in range(1, 8)))
        _touch(self.dir, "ar_debug_20260901_120000.jsonl", "ar_debug_20260902_120000.jsonl")
        removed = prune_logs(self.dir, keep=3)
        self.assertEqual(len(removed), 4 + 0)
        self.assertEqual(
            [n for n in self.names() if n.startswith("eyelab_")],
            ["eyelab_20260905_120000.jsonl", "eyelab_20260906_120000.jsonl",
             "eyelab_20260907_120000.jsonl"])
        self.assertEqual(len([n for n in self.names() if n.startswith("ar_debug_")]), 2)

    def test_pinned_files_survive_and_do_not_count(self):
        _touch(self.dir, *(f"ar_debug_2026090{i}_120000.jsonl" for i in range(1, 6)))
        (self.dir / KEEP_FILE).write_text(
            "# evidence cited in docs\nar_debug_20260901_120000.jsonl  # freeze\n", encoding="utf-8")
        prune_logs(self.dir, keep=2)
        self.assertEqual(
            [n for n in self.names() if n.startswith("ar_debug_")],
            ["ar_debug_20260901_120000.jsonl", "ar_debug_20260904_120000.jsonl",
             "ar_debug_20260905_120000.jsonl"])

    def test_single_append_logs_are_never_touched(self):
        _touch(self.dir, "xreal_probe.jsonl", "xreal_native_probe.jsonl", "notes.txt")
        self.assertEqual(prune_logs(self.dir, keep=0), [])
        self.assertEqual(len(self.names()), 3)

    def test_keep_file_parsing(self):
        (self.dir / KEEP_FILE).write_text("a.jsonl\n\n  # only a comment\nb.log # why\n",
                                          encoding="utf-8")
        self.assertEqual(pinned_logs(self.dir), {"a.jsonl", "b.log"})
        self.assertEqual(pinned_logs(self.dir / "missing"), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
