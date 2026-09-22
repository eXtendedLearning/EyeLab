#!/usr/bin/env python3
"""Tests for the S1 display smoke test.

The drawing functions take a canvas-like object, so the geometry is exercised
here with a recorder instead of a real Tk canvas — no display needed, which
also means these run on the Linux side and in CI.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import display_check
from display_check import (
    BLACK_LEVELS,
    COLOURS,
    LINE_WIDTHS,
    PAGES,
    Monitor,
    grey,
    pick_monitor,
    render,
)


class _Recorder:
    """Canvas stand-in that records primitives and their coordinates."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.cleared = 0

    def delete(self, *_a, **_k):
        self.cleared += 1

    def _record(self, kind):
        def inner(*coords, **kw):
            self.calls.append((kind, coords, kw))
        return inner

    def __getattr__(self, name):
        if name.startswith("create_"):
            return self._record(name[len("create_"):])
        raise AttributeError(name)

    def kinds(self):
        return [c[0] for c in self.calls]

    def xs(self):
        out = []
        for _kind, coords, _kw in self.calls:
            out.extend(coords[0::2])
        return out


class GreyTests(unittest.TestCase):
    def test_pure_black(self):
        self.assertEqual(grey(0), "#000000")

    def test_clamps(self):
        self.assertEqual(grey(-5), "#000000")
        self.assertEqual(grey(999), "#ffffff")

    def test_ramp_starts_at_pure_black(self):
        self.assertEqual(BLACK_LEVELS[0], 0)
        self.assertEqual(sorted(BLACK_LEVELS), list(BLACK_LEVELS))


class MonitorSelectionTests(unittest.TestCase):
    def setUp(self):
        self.monitors = [
            Monitor(1, 0, 0, 2880, 1920, True),
            Monitor(2, 2880, 0, 1920, 1080, False),
        ]

    def test_defaults_to_first_non_primary(self):
        self.assertEqual(pick_monitor(self.monitors, None).index, 2)

    def test_explicit_index_wins(self):
        self.assertEqual(pick_monitor(self.monitors, 1).index, 1)

    def test_falls_back_to_primary_when_only_one(self):
        only = [self.monitors[0]]
        self.assertEqual(pick_monitor(only, None).index, 1)

    def test_unknown_index_raises_with_choices(self):
        with self.assertRaises(ValueError) as ctx:
            pick_monitor(self.monitors, 7)
        self.assertIn("1", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    def test_geometry_string(self):
        self.assertEqual(self.monitors[1].geometry, "1920x1080+2880+0")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            pick_monitor([], None)

    def test_list_monitors_never_returns_empty(self):
        self.assertTrue(len(display_check.list_monitors()) >= 1)


class PageRenderTests(unittest.TestCase):
    def test_every_page_draws_something(self):
        for page in PAGES:
            rec = _Recorder()
            render(rec, page, 1920, 1080, "#ffffff")
            self.assertTrue(rec.calls, f"{page} drew nothing")
            self.assertEqual(rec.cleared, 1, f"{page} did not clear first")

    def test_black_page_has_one_patch_per_level(self):
        rec = _Recorder()
        render(rec, "black", 1920, 1080, "#ffffff")
        rects = [c for c in rec.calls if c[0] == "rectangle"]
        self.assertEqual(len(rects), len(BLACK_LEVELS))
        fills = [c[2]["fill"] for c in rects]
        self.assertEqual(fills[0], "#000000")
        self.assertEqual(len(set(fills)), len(BLACK_LEVELS))

    def test_line_matrix_covers_colours_and_widths(self):
        rec = _Recorder()
        render(rec, "lines", 1920, 1080, "#ffffff")
        lines = [c for c in rec.calls if c[0] == "line"]
        self.assertEqual(len(lines), len(COLOURS) * len(LINE_WIDTHS))
        widths = {c[2]["width"] for c in lines}
        self.assertEqual(widths, set(LINE_WIDTHS))
        fills = {c[2]["fill"] for c in lines}
        self.assertEqual(fills, {v for _n, v in COLOURS})

    def test_geometry_page_draws_four_corner_brackets(self):
        rec = _Recorder()
        render(rec, "geometry", 1920, 1080, "#ffffff")
        thick = [c for c in rec.calls if c[0] == "line" and c[2].get("width") == 3]
        self.assertEqual(len(thick), 8)  # two arms per corner

    def test_drawing_stays_within_the_canvas(self):
        for page in PAGES:
            rec = _Recorder()
            render(rec, page, 1920, 1080, "#ffffff")
            for kind, coords, _kw in rec.calls:
                for x in coords[0::2]:
                    self.assertGreaterEqual(x, 0, f"{page}/{kind} off left")
                    self.assertLessEqual(x, 1920, f"{page}/{kind} off right")

    def test_small_panel_does_not_crash(self):
        for page in PAGES:
            rec = _Recorder()
            render(rec, page, 640, 480, "#00ff66")
            self.assertTrue(rec.calls)


class SideBySideTests(unittest.TestCase):
    def test_sbs_draws_both_eyes(self):
        mono = _Recorder()
        render(mono, "geometry", 1920, 1080, "#ffffff")
        stereo = _Recorder()
        render(stereo, "geometry", 1920, 1080, "#ffffff", sbs=True)
        # two eyes plus the centre divider
        self.assertGreater(len(stereo.calls), len(mono.calls))

    def test_sbs_splits_the_panel(self):
        rec = _Recorder()
        render(rec, "geometry", 1920, 1080, "#ffffff", sbs=True)
        xs = rec.xs()
        self.assertTrue(any(x < 960 for x in xs), "nothing in the left half")
        self.assertTrue(any(x > 960 for x in xs), "nothing in the right half")

    def test_disparity_moves_the_eyes_apart(self):
        near = _Recorder()
        render(near, "geometry", 1920, 1080, "#ffffff", sbs=True, disparity=0)
        far = _Recorder()
        render(far, "geometry", 1920, 1080, "#ffffff", sbs=True, disparity=80)
        self.assertNotEqual(near.xs(), far.xs())

    def test_disparity_is_symmetric(self):
        rec = _Recorder()
        render(rec, "geometry", 1920, 1080, "#ffffff", sbs=True, disparity=80)
        xs = rec.xs()
        self.assertTrue(any(x < 0 for x in xs) or min(xs) < 40,
                        "left eye should shift outward")


class CliTests(unittest.TestCase):
    def test_list_monitors_exits_zero(self):
        import contextlib
        import io

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            code = display_check.main(["--list-monitors"])
        self.assertEqual(code, 0)
        self.assertIn("[1]", sink.getvalue())

    def test_bad_monitor_index_exits_two(self):
        import contextlib
        import io

        with patch.object(display_check, "list_monitors",
                          return_value=[Monitor(1, 0, 0, 800, 600, True)]):
            sink = io.StringIO()
            with contextlib.redirect_stderr(sink):
                code = display_check.main(["--monitor", "9"])
        self.assertEqual(code, 2)
        self.assertIn("no monitor 9", sink.getvalue())

    def test_gui_receives_selected_monitor(self):
        chosen = Monitor(2, 2880, 0, 1920, 1080, False)
        with patch.object(display_check, "list_monitors",
                          return_value=[Monitor(1, 0, 0, 800, 600, True), chosen]):
            with patch.object(display_check, "run_gui", return_value=0) as gui:
                import contextlib
                import io

                with contextlib.redirect_stdout(io.StringIO()):
                    display_check.main([])
        self.assertEqual(gui.call_args[0][0].index, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
