"""Manual analysis of one scan or photo (T7.0 §14.5): check every target, confirm, pin or reject its dot.

The page is shown rectified (10 px per mm) with, per target, the nominal cross (blue), the dot in use (red, with the
error line) and every candidate the detector saw: accepted (orange) and too faint to count (grey dashed). Select a
target in the list (the view zooms to it), then:

- **Confirm** (Enter): the automatic dot is right;
- **Pin** (P, then click on the image): the dot is here; a click near a candidate takes that candidate, elsewhere the
  centre of the dark spot under the click; Shift+click keeps the exact click;
- **Not present** (N): no dot for this target (an automatic "dot" there was dirt);
- **Reset** (R): back to the automatic result.

Space goes to the next target not yet reviewed. **Save** writes ``results/<stem>.review.json`` and re-analyses the
image, so the overview uses the reviewed hits. Logic in ``campaign.review``.
"""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from . import review as Rv
from . import store as St
from .zoomview import ZoomView

PPMM = 10.0           # rectified pixels per mm (hits.analyse default; the overlay is drawn in page mm)
PICK_MM = 1.5         # a pin click this close to a candidate takes the candidate
WEAK_INK_MM2 = 0.05   # candidates down to this ink are shown (grey) so a faint real dot can be picked
VIEW_MM = 30.0        # field of view when zooming to a target
COL = {"nom": "#3a7bff", "hit": "#ff3030", "auto": "#ff9a1f", "weak": "#9a9a9a", "sel": "#ffe14a"}


def _calib_for(root: Path, src: St.Source) -> Optional[str]:
    """The same undistortion the automatic analysis uses (store.analyse)."""
    if src.kind == "photo":
        c = Path(root) / "photos" / "camera.yaml"
        return str(c) if c.exists() else None
    return None


class ManualWindow:
    def __init__(self, master, root: Path, src: St.Source, hide_numbers: bool = False,
                 on_saved: Optional[Callable[[], None]] = None):
        self.master = master
        self.root, self.src, self.hide, self.on_saved = Path(root), src, hide_numbers, on_saved
        self.version = getattr(src, "version", "")
        self.result_json = St.result_path(self.root, src)
        self.win = tk.Toplevel(master)
        self.win.title(f"Manual analysis - {src.key}")
        self.win.geometry("1500x900")
        self.res = None            # hits.Result of the automatic analysis (no review applied)
        self.sh = None
        self.auto: Dict[int, Optional[Tuple[float, float]]] = {}
        self.nom: Dict[int, Tuple[float, float]] = {}
        self.cands: List[Tuple[float, float, bool]] = []   # x, y (page mm), accepted
        self.dec: Dict[int, dict] = {}
        self.dirty = False
        self.pin_mode = False
        self.q: "queue.Queue" = queue.Queue()   # worker threads never touch Tk: they post here
        self._build()
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.status.set("analysing the image...")
        threading.Thread(target=self._load, daemon=True).start()
        self._poll()

    def _poll(self):
        try:
            while True:
                f = self.q.get_nowait()
                f()
        except queue.Empty:
            pass
        try:
            if self.win.winfo_exists():
                self.win.after(100, self._poll)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ layout
    def _build(self):
        pw = ttk.PanedWindow(self.win, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=6, pady=6)
        self.view = ZoomView(pw)
        right = ttk.Frame(pw, padding=(6, 0))
        pw.add(self.view, weight=4)
        pw.add(right, weight=1)
        self.view.on_redraw.append(self._draw)
        self.view.on_click = self._click

        cols = [("n", "target", 50), ("status", "status", 110), ("dx", "dx mm", 60), ("dy", "dy mm", 60),
                ("err", "err mm", 60)]
        self.tree = ttk.Treeview(right, columns=[c[0] for c in cols], show="headings", height=16,
                                 selectmode="browse")
        for cid, head, w in cols:
            self.tree.heading(cid, text=head)
            self.tree.column(cid, width=w, anchor="center")
        for tag, colour in (("confirmed", "#106010"), ("pinned", "#0a4fb0"), ("absent", "#8a2be2"),
                            ("nodot", "#b00000")):
            self.tree.tag_configure(tag, foreground=colour)
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._focus_selected())

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=6)
        for i, (text, cmd) in enumerate((("Confirm  (Enter)", self.confirm), ("Pin  (P, then click)", self.pin),
                                         ("Not present  (N)", self.absent), ("Reset to auto  (R)", self.reset),
                                         ("Next unreviewed  (Space)", self.next_unreviewed),
                                         ("Confirm all remaining", self.confirm_all))):
            ttk.Button(btns, text=text, command=cmd).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        self.show_weak = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Show faint candidates (grey)", variable=self.show_weak,
                        command=self.view.redraw).pack(anchor="w")
        self.summary = tk.StringVar()
        ttk.Label(right, textvariable=self.summary, justify="left", font=("Consolas", 9)).pack(anchor="w", pady=6)
        ttk.Label(right, text="Note:").pack(anchor="w")
        self.note = tk.Text(right, height=3, width=40)
        self.note.pack(fill="x")
        self.note.bind("<<Modified>>", self._note_changed)
        sv = ttk.Frame(right)
        sv.pack(fill="x", pady=8)
        ttk.Button(sv, text="Save review", command=self.save).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(sv, text="Close", command=self.close).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Label(right, foreground="#555", wraplength=330, justify="left", text=(
            "Blue: nominal target.  Red: the dot in use, with its error.  Orange: candidates the detector "
            "accepted.  Grey dashed: too faint to count automatically.\nWheel: zoom.  Drag: pan.  "
            "Pin: P, then click (near a candidate takes it; elsewhere the centre of the dark spot; Shift+click "
            "the exact point).  Esc cancels a pin.")).pack(anchor="w")
        self.status = tk.StringVar()
        ttk.Label(self.win, textvariable=self.status, foreground="#235", padding=(6, 0, 6, 4)).pack(anchor="w")

        w = self.win
        w.bind("<Return>", self._key(self.confirm))
        w.bind("<KP_Enter>", self._key(self.confirm))
        for k, f in (("n", self.absent), ("p", self.pin), ("r", self.reset)):
            w.bind(f"<KeyPress-{k}>", self._key(f))
            w.bind(f"<KeyPress-{k.upper()}>", self._key(f))
        w.bind("<space>", self._key(self.next_unreviewed))
        w.bind("<Escape>", lambda e: self._set_pin(False))

    def _key(self, f):
        def h(e):
            if e.widget is self.note:   # typing in the note
                return None
            f()
            return "break"
        return h

    # ------------------------------------------------------------------ load
    def _load(self):
        from . import hits as Hm
        try:
            res, rect = Hm.analyse(self.src.path, self.src.sheet_hint, calib=_calib_for(self.root, self.src))
            sh = Hm.load_sheet(res.sheet)
            acc = Hm.find_dots(rect, sh, PPMM, res.mm_per_px)
            try:
                weak = Hm.find_dots(rect, sh, PPMM, res.mm_per_px, min_ink_mm2=WEAK_INK_MM2)
            except TypeError:   # a find_dots without the ink rule
                weak = []
            out = (res, rect, sh, acc, weak, None)
        except Exception as e:
            out = (None, None, None, [], [], f"{type(e).__name__}: {e}")
        self.q.put(lambda: self._loaded(*out))

    def _loaded(self, res, rect, sh, acc, weak, err):
        if err:
            self.status.set("cannot analyse this image: " + err)
            return
        self.res, self.rect, self.sh = res, rect, sh
        self.auto = {h.n: h.hit for h in res.hits}
        self.nom = {h.n: h.nom for h in res.hits}
        acc_xy = [(d[0], d[1]) for d in acc]
        self.cands = [(x, y, True) for x, y in acc_xy]
        self.cands += [(d[0], d[1], False) for d in weak if all(math.dist((d[0], d[1]), a) > 0.2 for a in acc_xy)]
        rv = Rv.load(self.result_json)
        if rv is not None:
            if rv.get("source_version") == self.version and rv.get("sheet", res.sheet) == res.sheet:
                for k, d in rv.get("targets", {}).items():
                    self.dec[int(k)] = {"status": d["status"], "hit": None if d["hit"] is None else tuple(d["hit"])}
                self.note.insert("1.0", rv.get("note", ""))
                self.status.set(f"sheet {res.sheet}: review of {rv.get('reviewed')} loaded")
            else:
                self.status.set(f"sheet {res.sheet}: the saved review is for an older version of this file; "
                                "starting from the automatic result")
        else:
            self.status.set(f"sheet {res.sheet}: {sum(h is not None for h in self.auto.values())}/{len(self.auto)} "
                            f"dots found automatically, {sum(c[2] for c in self.cands)} accepted candidates, "
                            f"{sum(not c[2] for c in self.cands)} faint")
        self.note.edit_modified(False)
        self.dirty = False
        self.view.set_image(Image.fromarray(rect))
        self._fill_tree()
        first = next((n for n in sorted(self.nom) if n not in self.dec), min(self.nom))
        self.tree.selection_set(str(first))

    # ------------------------------------------------------------------ state
    def final(self, n: int) -> Optional[Tuple[float, float]]:
        d = self.dec.get(n)
        if d is None:
            return self.auto.get(n)
        return None if d["status"] == "absent" else d["hit"]

    def _status_text(self, n: int) -> Tuple[str, str]:
        d = self.dec.get(n)
        if d is None:
            return ("auto", "") if self.auto.get(n) is not None else ("auto: no dot", "nodot")
        return d["status"], d["status"]

    def _err(self, n: int):
        f = self.final(n)
        if f is None:
            return None
        nx, ny = self.nom[n]
        return f[0] - nx, -(f[1] - ny)

    def _fill_tree(self):
        sel = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for n in sorted(self.nom):
            st, tag = self._status_text(n)
            e = self._err(n)
            vals = (n, st, "", "", "") if e is None or self.hide else \
                (n, st, f"{e[0]:+.2f}", f"{e[1]:+.2f}", f"{math.hypot(*e):.2f}")
            self.tree.insert("", "end", iid=str(n), values=vals, tags=(tag,))
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0])
        self._update_summary()

    def _update_summary(self):
        from . import hits as Hm
        errs = [e for e in (self._err(n) for n in self.nom) if e is not None]
        counts = {s: sum(1 for d in self.dec.values() if d["status"] == s) for s in Rv.STATUSES}
        lines = [f"reviewed {len(self.dec)}/{len(self.nom)}: " +
                 ", ".join(f"{v} {k}" for k, v in counts.items()),
                 f"hits in use {len(errs)}/{len(self.nom)}" + (" (unsaved changes)" if self.dirty else "")]
        st = Hm.stats(errs)
        if st.get("n") and not self.hide:
            lines.append(f"bias ({st['bias_x_mm']:+.2f}, {st['bias_y_mm']:+.2f}) mm  median {st['median_mm']:.2f}"
                         f"  R95 {st['r95_mm']:.2f}  max {st['max_mm']:.2f}")
        self.summary.set("\n".join(lines))

    def _selected(self) -> Optional[int]:
        s = self.tree.selection()
        return int(s[0]) if s else None

    def _changed(self, n: Optional[int] = None, advance: bool = False):
        self.dirty = True
        self._fill_tree()
        self.view.redraw()
        if advance:
            self.next_unreviewed()

    # ------------------------------------------------------------------ actions
    def confirm(self):
        n = self._selected()
        if n is None or self.res is None:
            return
        cur = self.final(n)
        if cur is None:
            self.status.set(f"target {n} has no dot: pin it (P, then click) or mark it not present (N)")
            return
        self.dec[n] = {"status": "confirmed", "hit": cur}
        self._changed(n, advance=True)

    def absent(self):
        n = self._selected()
        if n is None or self.res is None:
            return
        self.dec[n] = {"status": "absent", "hit": None}
        self._changed(n, advance=True)

    def reset(self):
        n = self._selected()
        if n is not None and n in self.dec:
            del self.dec[n]
            self._changed(n)

    def pin(self):
        if self._selected() is not None and self.res is not None:
            self._set_pin(True)

    def _set_pin(self, on: bool):
        self.pin_mode = on
        self.view.canvas.config(cursor="crosshair" if on else "fleur")
        n = self._selected()
        self.status.set(f"PIN target {n}: click on its dot (Shift+click: exact point; Esc: cancel)" if on else "")

    def confirm_all(self):
        if self.res is None:
            return
        left = [n for n in self.nom if n not in self.dec and self.auto.get(n) is not None]
        if not left:
            return
        if messagebox.askyesno("Confirm all remaining", f"Confirm the automatic dot of the {len(left)} targets "
                               "not reviewed yet? (Targets without a dot are left as they are.)", parent=self.win):
            for n in left:
                self.dec[n] = {"status": "confirmed", "hit": self.auto[n]}
            self._changed()

    def next_unreviewed(self):
        if not self.nom:
            return
        order = sorted(self.nom)
        cur = self._selected()
        start = order.index(cur) + 1 if cur in order else 0
        for n in order[start:] + order[:start]:
            if n not in self.dec:
                self.tree.selection_set(str(n))
                self.tree.see(str(n))
                return
        self.status.set("every target is reviewed: Save review")

    def _click(self, ix: float, iy: float, e):
        if not self.pin_mode:
            return
        n = self._selected()
        if n is None:
            return
        x, y = ix / PPMM, iy / PPMM
        how = "exact click"
        if not (e.state & 0x0001):   # Shift held: keep the exact click
            best = min(self.cands, key=lambda c: math.dist((c[0], c[1]), (x, y)), default=None)
            if best is not None and math.dist((best[0], best[1]), (x, y)) <= PICK_MM:
                x, y, how = best[0], best[1], "candidate"
            else:
                x, y = Rv.refine_click(self.rect, x, y, PPMM)
                how = "centre of the dark spot"
        self.dec[n] = {"status": "pinned", "hit": (x, y)}
        self._set_pin(False)
        self.status.set(f"target {n} pinned at ({x:.2f}, {y:.2f}) mm ({how})")
        self._changed(n)

    # ------------------------------------------------------------------ drawing
    def _focus_selected(self):
        n = self._selected()
        if n is None or n not in self.nom:
            return
        self._set_pin(False)
        w, h = self.view._size()
        x, y = self.nom[n]
        f = self.final(n)
        if f is not None:   # centre between the target and its dot when they are far apart
            x, y = (x + f[0]) / 2, (y + f[1]) / 2
        self.view.zoom_to(x * PPMM, y * PPMM, min(w, h) / (VIEW_MM * PPMM))

    def _draw(self, view: ZoomView):
        if self.res is None:
            return
        c = view.canvas
        s = view.scale * PPMM            # screen px per mm
        P = lambda x, y: view.to_canvas(x * PPMM, y * PPMM)
        for x, y, ok in self.cands:
            if not ok and not self.show_weak.get():
                continue
            cx, cy = P(x, y)
            r = max(5.0, 1.6 * s)
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=COL["auto"] if ok else COL["weak"],
                          width=1, dash=() if ok else (3, 3))
        sel = self._selected()
        for n, (nx, ny) in self.nom.items():
            cx, cy = P(nx, ny)
            a = max(6.0, 3.0 * s)
            colour = COL["sel"] if n == sel else COL["nom"]
            c.create_line(cx - a, cy, cx + a, cy, fill=colour, width=2 if n == sel else 1)
            c.create_line(cx, cy - a, cx, cy + a, fill=colour, width=2 if n == sel else 1)
            c.create_text(cx + a + 2, cy - a - 2, text=str(n), fill=colour, anchor="sw",
                          font=("Segoe UI", 10 if n == sel else 9, "bold"))
            f = self.final(n)
            if f is not None:
                hx, hy = P(*f)
                r = max(4.0, 0.9 * s)
                c.create_line(cx, cy, hx, hy, fill=COL["hit"], width=2)
                c.create_oval(hx - r, hy - r, hx + r, hy + r, outline=COL["hit"], width=2)
            d = self.dec.get(n)
            if d and d["status"] == "absent":
                c.create_text(cx, cy + a + 2, text="not present", fill="#c070ff", anchor="n", font=("Segoe UI", 9))

    # ------------------------------------------------------------------ save / close
    def _note_changed(self, _e=None):
        if self.note.edit_modified():
            self.dirty = True
            self._update_summary()

    def save(self) -> bool:
        if self.res is None:
            return False
        note = self.note.get("1.0", "end").strip()
        Rv.save(self.result_json, self.src.key, self.version, self.res.sheet,
                {n: {"status": d["status"], "hit": d["hit"]} for n, d in self.dec.items()}, note)
        self.dirty = False
        self.note.edit_modified(False)
        self.status.set("saved; re-analysing with the review...")
        self._update_summary()

        # re-analysed here, on the GUI thread (a few seconds): the window may close right after Save
        self.win.config(cursor="watch")
        self.win.update_idletasks()
        try:
            r = St.analyse(self.root, self.src)
        finally:
            self.win.config(cursor="")
        self.status.set(r.get("error") or f"saved: {r.get('found')}/{r.get('targets')} hits in use")
        if self.on_saved:
            self.on_saved()
        return True

    def close(self):
        if self.dirty:
            a = messagebox.askyesnocancel("Unsaved review", "Save the review before closing?", parent=self.win)
            if a is None:
                return
            if a:
                self.save()
        self.win.destroy()
