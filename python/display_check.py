#!/usr/bin/env python3
"""Stage S1 — display smoke test for the XREAL glasses used as a plain monitor.

This answers three questions that need no pose, no registration and no XREAL
SDK, and that are cheaper to answer now than after the sensor work:

1. **Is pure black really transmissive?** These are additive optics: ``#000000``
   should disappear and let the specimen through. Near-black usually does not.
   Page ``black`` steps from ``#000000`` upwards so you can find the level at
   which the panel starts to glow and wash out the scene behind it.
2. **Is a thin bright wireframe legible against a real specimen?** Page ``lines``
   is a matrix of colours against line widths. Hold it over grey steel under
   lab lighting and see which combination survives.
3. **Is stereo convergence comfortable at working distance?** ``--sbs`` renders
   the pattern twice side-by-side with an adjustable horizontal disparity, for
   when the glasses are switched into a 3840x1080 side-by-side mode.

Page ``grid`` is the realistic case: a perspective wireframe of the kind the
UNV overlay will actually draw. Page ``geometry`` checks that the full panel is
visible and nothing is cropped by the optics.

Usage::

    python python/display_check.py --list-monitors
    python python/display_check.py --monitor 2
    python python/display_check.py --monitor 2 --sbs --disparity 40

Keys: ``space``/``right`` next page, ``left`` previous, ``c`` cycle colour,
``[`` / ``]`` adjust disparity in SBS mode, ``f`` toggle fullscreen,
``Esc``/``q`` quit.

tkinter only — no OpenCV, no numpy, so it starts instantly and cannot be
blocked by a camera backend. Nothing here talks to the glasses over USB; they
are just a monitor.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

PAGES = ("black", "lines", "grid", "geometry")

#: Candidate overlay colours, brightest-neutral first.
COLOURS = (
    ("white", "#ffffff"),
    ("green", "#00ff66"),
    ("cyan", "#00e5ff"),
    ("amber", "#ffb000"),
    ("red", "#ff3b30"),
)

#: Steps for the black-level ramp, in 8-bit grey levels.
BLACK_LEVELS = (0, 2, 4, 6, 8, 12, 16, 24, 32)

LINE_WIDTHS = (1, 2, 3, 5)


@dataclass(frozen=True)
class Monitor:
    """A display in virtual-desktop coordinates."""

    index: int
    x: int
    y: int
    width: int
    height: int
    primary: bool

    @property
    def geometry(self) -> str:
        return f"{self.width}x{self.height}+{self.x}+{self.y}"

    def describe(self) -> str:
        tag = " (primary)" if self.primary else ""
        return (
            f"[{self.index}] {self.width}x{self.height} "
            f"at {self.x:+d},{self.y:+d}{tag}"
        )


def list_monitors() -> list[Monitor]:
    """Enumerate monitors in virtual-desktop coordinates.

    Uses ``EnumDisplayMonitors`` on Windows because tkinter only ever reports
    the primary display, which is exactly the wrong one here. Elsewhere, and
    if the API call fails, returns a single synthetic entry so the tool still
    runs.
    """
    if sys.platform != "win32":
        return [Monitor(index=1, x=0, y=0, width=1920, height=1080, primary=True)]

    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
    except Exception:  # pragma: no cover - defensive
        return [Monitor(index=1, x=0, y=0, width=1920, height=1080, primary=True)]

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        ]

    found: list[Monitor] = []
    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
        ctypes.POINTER(RECT),
        ctypes.c_double,
    )

    def _callback(hmonitor, _hdc, _rect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(ctypes.c_ulonglong(hmonitor), ctypes.byref(info)):
            r = info.rcMonitor
            found.append(
                Monitor(
                    index=len(found) + 1,
                    x=r.left,
                    y=r.top,
                    width=r.right - r.left,
                    height=r.bottom - r.top,
                    primary=bool(info.dwFlags & 1),
                )
            )
        return 1

    try:
        user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_callback), 0)
    except Exception:  # pragma: no cover - defensive
        pass

    if not found:
        return [Monitor(index=1, x=0, y=0, width=1920, height=1080, primary=True)]
    return found


def pick_monitor(monitors: Sequence[Monitor], requested: Optional[int]) -> Monitor:
    """Choose a monitor: the requested index, else the first non-primary, else primary.

    Defaulting to a non-primary display is the useful behaviour here — the
    glasses are almost never the primary monitor, and putting a fullscreen
    black window on the laptop panel helps nobody.
    """
    if not monitors:
        raise ValueError("no monitors")
    if requested is not None:
        for monitor in monitors:
            if monitor.index == requested:
                return monitor
        raise ValueError(
            f"no monitor {requested}; available: "
            + ", ".join(str(m.index) for m in monitors)
        )
    for monitor in monitors:
        if not monitor.primary:
            return monitor
    return monitors[0]


def grey(level: int) -> str:
    level = max(0, min(255, level))
    return f"#{level:02x}{level:02x}{level:02x}"


# --------------------------------------------------------------------------
# Drawing. Each page takes a canvas-like object so the geometry is testable
# without a display: the tests pass a recorder that captures the calls.
# --------------------------------------------------------------------------


def draw_black_levels(canvas: Any, w: int, h: int, colour: str) -> None:
    """Grey ramp from pure black upwards, plus a reference hairline."""
    count = len(BLACK_LEVELS)
    patch_w = w // (count + 1)
    patch_h = max(60, h // 6)
    top = h // 2 - patch_h // 2
    left = (w - patch_w * count) // 2

    for i, level in enumerate(BLACK_LEVELS):
        x0 = left + i * patch_w
        canvas.create_rectangle(
            x0, top, x0 + patch_w, top + patch_h, fill=grey(level), outline=""
        )
        canvas.create_text(
            x0 + patch_w // 2,
            top + patch_h + 24,
            text=str(level),
            fill=colour,
            anchor="n",
        )

    canvas.create_text(
        w // 2,
        top - 60,
        text="black level — find where the panel starts to glow",
        fill=colour,
        anchor="s",
    )
    canvas.create_line(0, h - 80, w, h - 80, fill=colour, width=1)
    canvas.create_text(
        w // 2, h - 60, text="1 px reference hairline", fill=colour, anchor="n"
    )


def draw_line_matrix(canvas: Any, w: int, h: int, colour: str) -> None:
    """Colour x line-width legibility matrix."""
    rows = len(COLOURS)
    row_h = h // (rows + 2)
    label_x = w // 12
    start_x = w // 6
    span = w - start_x - w // 12

    canvas.create_text(
        w // 2, row_h // 2, text="line legibility — colour x width", fill=colour
    )
    for r, (name, value) in enumerate(COLOURS):
        y = row_h * (r + 1) + row_h // 2
        canvas.create_text(label_x, y, text=name, fill=value, anchor="w")
        seg = span // len(LINE_WIDTHS)
        for c, width in enumerate(LINE_WIDTHS):
            x0 = start_x + c * seg
            canvas.create_line(x0, y, x0 + seg - 30, y, fill=value, width=width)
            canvas.create_text(
                x0 + seg - 14, y, text=str(width), fill=value, anchor="w"
            )


def draw_wireframe(canvas: Any, w: int, h: int, colour: str) -> None:
    """A perspective grid and box — the realistic overlay case."""
    cx, cy = w // 2, int(h * 0.58)
    horizon = int(h * 0.34)

    # Chosen so the outermost near-line lands exactly on the panel edge: a
    # receding floor grid should reach the edges, but drawing hundreds of
    # pixels beyond them is wasted work and makes the layout untestable.
    lanes = 6
    near_step = w // (lanes * 2)
    far_step = max(1, w // (lanes * 10))

    for i in range(-lanes, lanes + 1):
        canvas.create_line(
            cx + i * far_step, horizon, cx + i * near_step, h, fill=colour, width=1
        )

    depth = 10
    near_half, far_half = lanes * near_step, lanes * far_step
    for i in range(1, depth + 1):
        t = (i / depth) ** 2.2
        y = horizon + int((h - horizon) * t)
        half = int(far_half + (near_half - far_half) * t)
        canvas.create_line(cx - half, y, cx + half, y, fill=colour, width=1)

    size = min(w, h) // 6
    ox, oy = cx, cy - size
    off = size // 2
    front = [
        (ox - size, oy - size),
        (ox + size, oy - size),
        (ox + size, oy + size),
        (ox - size, oy + size),
    ]
    back = [(x + off, y - off) for x, y in front]
    for quad in (front, back):
        for a in range(4):
            x0, y0 = quad[a]
            x1, y1 = quad[(a + 1) % 4]
            canvas.create_line(x0, y0, x1, y1, fill=colour, width=2)
    for (x0, y0), (x1, y1) in zip(front, back):
        canvas.create_line(x0, y0, x1, y1, fill=colour, width=1)

    canvas.create_text(
        w // 2, 40, text="wireframe over a real specimen", fill=colour, anchor="n"
    )


def draw_geometry(canvas: Any, w: int, h: int, colour: str) -> None:
    """Corner brackets, centre cross, circle and a scale bar — FOV / crop check."""
    inset, arm = 40, 160
    for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = inset if sx == 0 else w - inset
        y = inset if sy == 0 else h - inset
        dx = arm if sx == 0 else -arm
        dy = arm if sy == 0 else -arm
        canvas.create_line(x, y, x + dx, y, fill=colour, width=3)
        canvas.create_line(x, y, x, y + dy, fill=colour, width=3)

    cx, cy = w // 2, h // 2
    canvas.create_line(cx - 90, cy, cx + 90, cy, fill=colour, width=1)
    canvas.create_line(cx, cy - 90, cx, cy + 90, fill=colour, width=1)
    radius = min(w, h) // 3
    canvas.create_oval(
        cx - radius, cy - radius, cx + radius, cy + radius, outline=colour, width=1
    )

    bar_y = h - 120
    canvas.create_line(cx - 250, bar_y, cx + 250, bar_y, fill=colour, width=2)
    for i in range(-5, 6):
        x = cx + i * 50
        canvas.create_line(x, bar_y - 10, x, bar_y + 10, fill=colour, width=1)
    canvas.create_text(
        cx, bar_y + 30, text="500 px, ticks every 100 px", fill=colour, anchor="n"
    )
    canvas.create_text(
        cx,
        60,
        text="all four brackets visible = nothing cropped",
        fill=colour,
        anchor="n",
    )


PAGE_RENDERERS: dict[str, Callable[[Any, int, int, str], None]] = {
    "black": draw_black_levels,
    "lines": draw_line_matrix,
    "grid": draw_wireframe,
    "geometry": draw_geometry,
}


def render(
    canvas: Any,
    page: str,
    width: int,
    height: int,
    colour: str,
    sbs: bool = False,
    disparity: int = 0,
) -> None:
    """Draw one page, once or twice for side-by-side stereo.

    In SBS the panel is split in half and each eye's copy is nudged inward or
    outward by ``disparity // 2``, so positive disparity pushes the image
    further away and negative brings it closer.
    """
    canvas.delete("all")
    if not sbs:
        PAGE_RENDERERS[page](canvas, width, height, colour)
        return

    half = width // 2
    shift = disparity // 2
    for eye, origin in enumerate((0, half)):
        sign = -1 if eye == 0 else 1
        _OffsetCanvas(canvas, origin + sign * shift).render(
            page, half, height, colour
        )
    canvas.create_line(half, 0, half, height, fill="#101010", width=1)


class _OffsetCanvas:
    """Translates every draw call by a fixed x offset. Keeps the pages simple."""

    def __init__(self, canvas: Any, dx: int):
        self._canvas = canvas
        self._dx = dx

    def render(self, page: str, width: int, height: int, colour: str) -> None:
        PAGE_RENDERERS[page](self, width, height, colour)

    def _shift(self, coords: Sequence[int]) -> list[int]:
        return [c + self._dx if i % 2 == 0 else c for i, c in enumerate(coords)]

    def create_line(self, *coords, **kw):
        return self._canvas.create_line(*self._shift(coords), **kw)

    def create_rectangle(self, *coords, **kw):
        return self._canvas.create_rectangle(*self._shift(coords), **kw)

    def create_oval(self, *coords, **kw):
        return self._canvas.create_oval(*self._shift(coords), **kw)

    def create_text(self, *coords, **kw):
        return self._canvas.create_text(*self._shift(coords), **kw)

    def delete(self, *args, **kw):  # pragma: no cover - never called per-eye
        return None


def run_gui(monitor: Monitor, args: argparse.Namespace) -> int:  # pragma: no cover
    import tkinter as tk

    state = {
        "page": PAGES.index(args.page) if args.page in PAGES else 0,
        "colour": 0,
        "disparity": args.disparity,
        "fullscreen": True,
    }

    root = tk.Tk()
    root.title("EyeLab — display check")
    root.configure(bg="black")
    root.geometry(monitor.geometry)
    root.overrideredirect(False)
    root.attributes("-fullscreen", True)

    canvas = tk.Canvas(
        root, bg="black", highlightthickness=0, bd=0, width=monitor.width,
        height=monitor.height,
    )
    canvas.pack(fill="both", expand=True)

    def redraw(_event=None):
        width = canvas.winfo_width() or monitor.width
        height = canvas.winfo_height() or monitor.height
        colour = COLOURS[state["colour"]][1]
        render(
            canvas,
            PAGES[state["page"]],
            width,
            height,
            colour,
            sbs=args.sbs,
            disparity=state["disparity"],
        )
        hud = (
            f"{PAGES[state['page']]}  |  {COLOURS[state['colour']][0]}  |  "
            f"{width}x{height}"
        )
        if args.sbs:
            hud += f"  |  disparity {state['disparity']:+d}px"
        hud += "   [space] page  [c] colour  [ ] disparity  [Esc] quit"
        canvas.create_text(20, 20, text=hud, fill="#606060", anchor="nw")

    def step_page(delta):
        state["page"] = (state["page"] + delta) % len(PAGES)
        redraw()

    def cycle_colour(_e=None):
        state["colour"] = (state["colour"] + 1) % len(COLOURS)
        redraw()

    def nudge(delta):
        state["disparity"] += delta
        redraw()

    def toggle_fullscreen(_e=None):
        state["fullscreen"] = not state["fullscreen"]
        root.attributes("-fullscreen", state["fullscreen"])
        redraw()

    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    root.bind("<space>", lambda e: step_page(1))
    root.bind("<Right>", lambda e: step_page(1))
    root.bind("<Left>", lambda e: step_page(-1))
    root.bind("c", cycle_colour)
    root.bind("bracketleft", lambda e: nudge(-4))
    root.bind("bracketright", lambda e: nudge(4))
    root.bind("[", lambda e: nudge(-4))
    root.bind("]", lambda e: nudge(4))
    root.bind("f", toggle_fullscreen)
    canvas.bind("<Configure>", redraw)

    root.after(60, redraw)
    root.mainloop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fullscreen display smoke test for the XREAL glasses."
    )
    parser.add_argument("--list-monitors", action="store_true")
    parser.add_argument("--monitor", type=int, default=None,
                        help="1-based index; default is the first non-primary")
    parser.add_argument("--page", default=PAGES[0], choices=PAGES)
    parser.add_argument("--sbs", action="store_true",
                        help="side-by-side stereo (set the glasses to a 3D mode)")
    parser.add_argument("--disparity", type=int, default=0,
                        help="horizontal disparity in px; + pushes away, - pulls near")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    monitors = list_monitors()
    if args.list_monitors:
        for monitor in monitors:
            print(monitor.describe())
        return 0

    try:
        monitor = pick_monitor(monitors, args.monitor)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"using {monitor.describe()}  — Esc to quit")
    return run_gui(monitor, args)


if __name__ == "__main__":
    raise SystemExit(main())
