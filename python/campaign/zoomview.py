"""A zoomable, pannable image view for tkinter (campaign GUI, T7.0 §14.5).

Mouse wheel zooms about the cursor, a left or middle drag pans, a click without a drag is reported to ``on_click``
with image coordinates. Fit resizes the image to the window, Centre centres it at the current zoom, 1:1 shows one
image pixel per screen pixel. Overlays are canvas items drawn by ``on_redraw`` callbacks with :meth:`to_canvas`.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageTk

CLICK_PX = 4          # a press and release closer than this is a click, not a drag
ZOOM_STEP = 1.25
MAX_SCALE = 40.0      # screen px per image px


class ZoomView(ttk.Frame):
    def __init__(self, master, toolbar: bool = True, background: str = "#222"):
        super().__init__(master)
        self.bar = ttk.Frame(self)
        if toolbar:
            self.bar.pack(fill="x")
            for text, cmd in (("Fit", self.fit), ("Centre", self.centre), ("1:1", self.one_to_one),
                              ("+", lambda: self.zoom_by(ZOOM_STEP)), ("-", lambda: self.zoom_by(1 / ZOOM_STEP))):
                ttk.Button(self.bar, text=text, width=6 if len(text) > 1 else 3, command=cmd).pack(side="left", padx=1)
            self.zoom_label = ttk.Label(self.bar, text="", width=9)
            self.zoom_label.pack(side="left", padx=6)
        else:
            self.zoom_label = None
        self.canvas = tk.Canvas(self, background=background, highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.image: Optional[Image.Image] = None
        self._pyramid: List[Image.Image] = []
        self._tk = None
        self.scale = 1.0          # screen px per image px
        self.off = (0.0, 0.0)     # canvas position of image (0, 0)
        self._fitted = True       # keep fitting on resize until the user zooms or pans
        self._press = None
        self.on_redraw: List[Callable[["ZoomView"], None]] = []
        self.on_click: Optional[Callable[[float, float, tk.Event], None]] = None
        c = self.canvas
        c.bind("<Configure>", lambda e: self.fit() if self._fitted else self.redraw())
        c.bind("<MouseWheel>", self._wheel)                  # Windows, macOS
        c.bind("<Button-4>", lambda e: self._wheel(e, 120))  # X11
        c.bind("<Button-5>", lambda e: self._wheel(e, -120))
        for b in (1, 2):
            c.bind(f"<ButtonPress-{b}>", self._down)
            c.bind(f"<B{b}-Motion>", self._drag)
            c.bind(f"<ButtonRelease-{b}>", lambda e, b=b: self._up(e, b))

    # ------------------------------------------------------------------ image
    def set_image(self, image: Optional[Image.Image], keep_view: bool = False):
        self.image = image
        self._pyramid = []
        if image is not None:
            im = image
            while min(im.size) > 600:
                im = im.reduce(2)
                self._pyramid.append(im)
        if keep_view and image is not None and not self._fitted:
            self.redraw()
        else:
            self.fit()

    def clear(self):
        self.set_image(None)

    # ------------------------------------------------------------------ transforms
    def to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.scale + self.off[0], y * self.scale + self.off[1]

    def to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - self.off[0]) / self.scale, (cy - self.off[1]) / self.scale

    def _size(self) -> Tuple[int, int]:
        return max(20, self.canvas.winfo_width()), max(20, self.canvas.winfo_height())

    # ------------------------------------------------------------------ view commands
    def fit(self):
        self._fitted = True
        if self.image is None:
            self.redraw()
            return
        w, h = self._size()
        iw, ih = self.image.size
        self.scale = min(w / iw, h / ih)
        self.off = ((w - iw * self.scale) / 2, (h - ih * self.scale) / 2)
        self.redraw()

    def centre(self):
        if self.image is None:
            return
        self._fitted = False
        w, h = self._size()
        iw, ih = self.image.size
        self.off = ((w - iw * self.scale) / 2, (h - ih * self.scale) / 2)
        self.redraw()

    def one_to_one(self):
        self.zoom_to(None, None, 1.0)

    def zoom_by(self, f: float, cx: Optional[float] = None, cy: Optional[float] = None):
        if self.image is None:
            return
        w, h = self._size()
        cx = w / 2 if cx is None else cx
        cy = h / 2 if cy is None else cy
        iw, ih = self.image.size
        lo = min(w / iw, h / ih) / 4
        new = min(MAX_SCALE, max(lo, self.scale * f))
        ix, iy = self.to_image(cx, cy)
        self.scale = new
        self.off = (cx - ix * new, cy - iy * new)
        self._fitted = False
        self.redraw()

    def zoom_to(self, x: Optional[float], y: Optional[float], scale: float):
        """Centre image point (x, y) (default: the point at the window centre) at ``scale`` screen px per px."""
        if self.image is None:
            return
        w, h = self._size()
        if x is None:
            x, y = self.to_image(w / 2, h / 2)
        self.scale = min(MAX_SCALE, scale)
        self.off = (w / 2 - x * self.scale, h / 2 - y * self.scale)
        self._fitted = False
        self.redraw()

    # ------------------------------------------------------------------ events
    def _wheel(self, e, delta=None):
        d = e.delta if delta is None else delta
        self.zoom_by(ZOOM_STEP ** (d / 120), e.x, e.y)

    def _down(self, e):
        self._press = (e.x, e.y, self.off, False)

    def _drag(self, e):
        if not self._press:
            return
        x0, y0, off, moved = self._press
        if not moved and abs(e.x - x0) + abs(e.y - y0) < CLICK_PX:
            return
        self._press = (x0, y0, off, True)
        self.off = (off[0] + e.x - x0, off[1] + e.y - y0)
        self._fitted = False
        self.redraw()

    def _up(self, e, button):
        p, self._press = self._press, None
        if p and not p[3] and button == 1 and self.on_click and self.image is not None:
            ix, iy = self.to_image(e.x, e.y)
            self.on_click(ix, iy, e)

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.zoom_label is not None:
            self.zoom_label.config(text=f"{self.scale * 100:.0f} %" if self.image is not None else "")
        if self.image is None:
            return
        w, h = self._size()
        iw, ih = self.image.size
        # visible part of the image, in image px
        x0, y0 = self.to_image(0, 0)
        x1, y1 = self.to_image(w, h)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(iw), x1), min(float(ih), y1)
        if x1 > x0 and y1 > y0:
            # the coarsest pyramid level still at least as fine as the screen
            src, f = self.image, 1
            for k, im in enumerate(self._pyramid):
                if self.scale * (2 ** (k + 1)) <= 1.0:
                    src, f = im, 2 ** (k + 1)
            box = (int(x0 / f), int(y0 / f), max(int(x0 / f) + 1, int(-(-x1 // f))),
                   max(int(y0 / f) + 1, int(-(-y1 // f))))
            crop = src.crop(box)
            ox, oy = self.to_canvas(box[0] * f, box[1] * f)
            ow = max(1, round((box[2] - box[0]) * f * self.scale))
            oh = max(1, round((box[3] - box[1]) * f * self.scale))
            crop = crop.resize((ow, oh), Image.NEAREST if self.scale >= 2 else Image.BILINEAR)
            self._tk = ImageTk.PhotoImage(crop)
            c.create_image(ox, oy, image=self._tk, anchor="nw")
        for cb in self.on_redraw:
            cb(self)
