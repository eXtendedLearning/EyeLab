"""Tk front end of ``gt_review``: one point at a time, zoomed.

Layout: large zoom on the point (left); on the right a thumbnail of the whole
frame with the zoom area outlined, the item's data and the key list.

  click         set the truth here (snapped to the X-junction unless snap is off)
  arrows        nudge 0.25 px (Shift: 1 px)
  Space/Enter   confirm and go to the next point (next frame after the last point)
  x             point not judgeable (occluded, blurred, ambiguous)
  r             back to the automatic point
  s             snap on/off
  + / - / wheel zoom      c  recentre      Backspace  previous item      Esc  back to the menu
                                                                     (decisions are saved at once)
"""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .gt_review import Item, status_for

ZOOM_PX = 720
THUMB_W = 400
C_AUTO, C_EYELAB, C_CUR, C_BOX = "#3cc83c", "#e63ce6", "#ffd23c", "#ffd23c"


class ReviewApp:
    def __init__(self, walk, title: str = "EyeLab ground-truth review"):
        self.walk = walk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="#1b1e22")
        self.half = 16                          # zoom half-width, image px
        self.snap_on = True
        self.item: Optional[Item] = None
        self.cur: Optional[np.ndarray] = None
        self.cur_snapped = False
        self.centre = np.zeros(2)
        self._build()
        self._bind()
        self._advance(self.walk.next())

    # ── layout ──
    def _build(self):
        self.zoom = tk.Canvas(self.root, width=ZOOM_PX, height=ZOOM_PX, bg="black",
                              highlightthickness=0, cursor="crosshair")
        self.zoom.grid(row=0, column=0, rowspan=3, padx=8, pady=8)
        self.thumb = tk.Canvas(self.root, width=THUMB_W, height=int(THUMB_W * 9 / 16), bg="black",
                               highlightthickness=0)
        self.thumb.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=8)
        mono = ("Consolas", 10)
        self.info = tk.Label(self.root, justify="left", anchor="nw", font=mono, fg="#e8ebee",
                             bg="#1b1e22", width=58, wraplength=THUMB_W + 60)
        self.info.grid(row=1, column=1, sticky="nw", padx=(0, 8))
        keys = ("click set   arrows nudge   Space confirm   x invalid\n"
                "r auto   s snap   +/- zoom   c centre   Bksp back   Esc menu")
        legend = "o green: automatic   + magenta: EyeLab   o yellow: your truth"
        tk.Label(self.root, text=legend + "\n\n" + keys, justify="left", anchor="sw", font=mono,
                 fg="#9aa4ae", bg="#1b1e22").grid(row=2, column=1, sticky="sw", padx=(0, 8), pady=8)

    def _bind(self):
        r = self.root
        self.zoom.bind("<Button-1>", self._click)
        self.zoom.bind("<MouseWheel>", lambda e: self._zoom(-1 if e.delta > 0 else 1))
        self.zoom.bind("<Button-4>", lambda e: self._zoom(-1))
        self.zoom.bind("<Button-5>", lambda e: self._zoom(1))
        for k in ("<space>", "<Return>"):
            r.bind(k, lambda e: self._confirm())
        r.bind("<BackSpace>", lambda e: self._advance(self.walk.back(), keep=True))
        r.bind("x", lambda e: self._invalid())
        r.bind("r", lambda e: self._reset())
        r.bind("s", lambda e: self._toggle_snap())
        r.bind("c", lambda e: self._recentre())
        for k in ("plus", "equal", "KP_Add"):
            r.bind(f"<{k}>", lambda e: self._zoom(-1))
        for k in ("minus", "KP_Subtract"):
            r.bind(f"<{k}>", lambda e: self._zoom(1))
        for k, d in (("Left", (-1, 0)), ("Right", (1, 0)), ("Up", (0, -1)), ("Down", (0, 1))):
            r.bind(f"<{k}>", lambda e, d=d: self._nudge(d, 0.25))
            r.bind(f"<Shift-{k}>", lambda e, d=d: self._nudge(d, 1.0))
        r.bind("<Escape>", lambda e: self.root.destroy())
        r.protocol("WM_DELETE_WINDOW", self.root.destroy)

    # ── state ──
    def _advance(self, item: Optional[Item], keep: bool = False):
        if item is None:
            if keep:
                return
            self.item = None
            self.zoom.delete("all")
            self.zoom.create_text(ZOOM_PX / 2, ZOOM_PX / 2, fill="white", font=("Segoe UI", 16),
                                  text="Nothing left to review here.\nDecisions saved. Esc returns to the menu.")
            self._info()
            return
        self.item = item
        self.cur = None if item.proposal is None else np.asarray(item.proposal, float)
        self.cur_snapped = False
        fv = self._fv()
        ref = self.cur if self.cur is not None else fv.eyelab.get(item.pid)
        self.centre = np.asarray(ref if ref is not None else np.array(fv.gray.shape[::-1]) / 2, float)
        self._thumb_img = None
        self._render()

    def _fv(self):
        return self.walk.review.frame(self.item.sid, self.item.i)

    def _confirm(self):
        if self.item is None:
            return
        self.walk.decide(self.item, status_for(self.item, self.cur), self.cur, self.cur_snapped)
        self._advance(self.walk.next())

    def _invalid(self):
        if self.item is None:
            return
        self.walk.decide(self.item, "invalid", None, False)
        self._advance(self.walk.next())

    def _reset(self):
        if self.item is not None:
            self.cur = None if self.item.auto is None else self.item.auto.copy()
            self.cur_snapped = False
            self._render()

    def _toggle_snap(self):
        self.snap_on = not self.snap_on
        self._info()

    def _recentre(self):
        if self.cur is not None:
            self.centre = self.cur.copy()
            self._render()

    def _zoom(self, step: int):
        self.half = int(np.clip(self.half + 4 * step, 6, 80))
        self._render()

    def _nudge(self, d, px):
        if self.item is None or self.cur is None:
            return
        self.cur = self.cur + np.array(d, float) * px
        self.cur_snapped = False
        self._render()

    def _click(self, e):
        if self.item is None:
            return
        uv = self._to_img(e.x, e.y)
        snapped = self.walk.review.snap(self._fv(), self.item.pid, uv) if self.snap_on else None
        self.cur, self.cur_snapped = (snapped, True) if snapped is not None else (uv, False)
        self._snap_failed = self.snap_on and snapped is None
        self._render()

    # ── geometry: canvas <-> image (pixel centres on integers) ──
    def _scale(self):
        return ZOOM_PX / (2 * self.half)

    def _origin(self):
        return np.round(self.centre) - self.half

    def _to_canvas(self, uv):
        s = self._scale()
        o = self._origin()
        return (uv[0] - o[0] + 0.5) * s, (uv[1] - o[1] + 0.5) * s

    def _to_img(self, x, y):
        s = self._scale()
        o = self._origin()
        return np.array([x / s + o[0] - 0.5, y / s + o[1] - 0.5])

    # ── drawing ──
    def _render(self):
        if self.item is None:
            return
        fv = self._fv()
        g = fv.gray
        h = self.half
        o = self._origin().astype(int)
        pad = np.pad(g, h + 2, mode="constant")
        crop = pad[o[1] + h + 2:o[1] + 3 * h + 2, o[0] + h + 2:o[0] + 3 * h + 2]
        big = cv2.resize(crop, (ZOOM_PX, ZOOM_PX), interpolation=cv2.INTER_NEAREST)
        self._zoom_img = ImageTk.PhotoImage(Image.fromarray(big))
        z = self.zoom
        z.delete("all")
        z.create_image(0, 0, anchor="nw", image=self._zoom_img)
        s = self._scale()
        r = max(6.0, 0.6 * s)
        a = self.item.auto
        if a is not None:
            x, y = self._to_canvas(a)
            z.create_oval(x - r, y - r, x + r, y + r, outline=C_AUTO, width=2)
        e = fv.eyelab.get(self.item.pid)
        if e is not None:
            x, y = self._to_canvas(e)
            z.create_line(x - r, y, x + r, y, fill=C_EYELAB, width=2)
            z.create_line(x, y - r, x, y + r, fill=C_EYELAB, width=2)
        if self.cur is not None:
            x, y = self._to_canvas(self.cur)
            z.create_line(x - 3 * r, y, x - r * 0.4, y, fill=C_CUR, width=1)
            z.create_line(x + r * 0.4, y, x + 3 * r, y, fill=C_CUR, width=1)
            z.create_line(x, y - 3 * r, x, y - r * 0.4, fill=C_CUR, width=1)
            z.create_line(x, y + r * 0.4, x, y + 3 * r, fill=C_CUR, width=1)
            z.create_oval(x - r * 0.4, y - r * 0.4, x + r * 0.4, y + r * 0.4, outline=C_CUR, width=2)
        self._render_thumb(fv)
        self._info()

    def _render_thumb(self, fv):
        H, W = fv.gray.shape
        k = THUMB_W / W
        if self._thumb_img is None:
            small = cv2.resize(fv.gray, (THUMB_W, int(H * k)), interpolation=cv2.INTER_AREA)
            self._thumb_img = ImageTk.PhotoImage(Image.fromarray(small))
        t = self.thumb
        t.delete("all")
        t.create_image(0, 0, anchor="nw", image=self._thumb_img)
        o = self._origin()
        x0, y0 = o * k
        x1, y1 = (o + 2 * self.half) * k
        t.create_rectangle(x0 - 2, y0 - 2, x1 + 2, y1 + 2, outline=C_BOX, width=2)

    def _info(self):
        w = self.walk
        what = {"audit": "sampled points", "session": "points of this session",
                "missing": "points without a truth"}.get(w.mode, "points")
        lines = [f"{w.mode}: {what} decided {w.done()} / {w.total()}   snap {'on' if self.snap_on else 'OFF'}"]
        if w.mode == "session":
            s0 = w.review.session(w.sid)
            lines.append(f"(session = {len(s0.frames)} frames x {len(s0.point_ids)} points; "
                         "carried points count as decided)")
        it = self.item
        if it is not None:
            s = w.review.session(it.sid)
            c = s.condition
            fv = self._fv()
            frames = [i for i, _ in s.frames]
            pids = s.point_ids
            cond = "  ".join(f"{k} {v}" for k, v in c.items() if k not in ("plate",))
            fmt = lambda p: "-" if p is None else f"({p[0]:8.2f}, {p[1]:8.2f})"
            e = fv.eyelab.get(it.pid)
            d = lambda p, q: "" if p is None or q is None else f"  d {np.linalg.norm(np.asarray(p) - q):5.2f} px"
            lines += ["", it.sid, cond,
                      f"frame {fv.name}  ({frames.index(it.i) + 1}/{len(frames)})",
                      f"point {it.pid}  ({pids.index(it.pid) + 1}/{len(pids)})   "
                      f"square ~{w.review.square_px(fv, it.pid):.0f} px",
                      "",
                      f"automatic {fmt(it.auto)}"
                      + ("" if it.auto is None else f"  plane res {w.review.residual_px(fv, it.pid):.2f} px"),
                      *([f"reviewed  {fmt(it.ref)}{d(it.ref, it.auto)}"] if it.ref_kind == "reviewed" else []),
                      f"EyeLab    {fmt(e)}{d(e, it.auto)}",
                      f"truth     {fmt(self.cur)}{d(self.cur, it.auto)}"
                      + ("  snapped" if self.cur_snapped else ""),
                      f"status if confirmed: {status_for(it, self.cur)}"]
            if getattr(self, "_snap_failed", False):
                lines.append("snap found no X-junction: raw click kept")
                self._snap_failed = False
            if it.note:
                lines.append(it.note)
            if getattr(w, "auto_carried", 0):
                lines.append(f"carried automatically so far: {w.auto_carried} "
                             f"(automatic point disagreed in {w.auto_disagree})")
        self.info.configure(text="\n".join(lines))

    def run(self):
        self.root.mainloop()
