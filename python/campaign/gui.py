"""Campaign hits GUI (T7.0 §4.2): watches the campaign data folder, analyses every new scan, photo or Beam Pro trial,
and shows the results as they arrive. Start with ``campaign.bat`` or ``python -m campaign gui``.

All logic is in ``campaign.store``; this file only shows it.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import store as St

SETTINGS = Path.home() / ".eyelab_campaign_gui.json"
POLL_MS = 2000
KIND_NAME = {"scan": "scan", "photo": "photo", "eye": "Eye (review look)"}


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(d: dict) -> None:
    try:
        SETTINGS.write_text(json.dumps(d, indent=1), encoding="utf-8")
    except OSError:
        pass


class Worker(threading.Thread):
    """Analyses pending sources one at a time; tells the GUI through a queue."""

    def __init__(self, root: Path, q: "queue.Queue"):
        super().__init__(daemon=True)
        self.root, self.q = root, q
        self.wake = threading.Event()
        self.stop = False
        self.paused = False

    def run(self):
        while not self.stop:
            if not self.paused:
                try:
                    todo = St.pending(self.root)
                except Exception as e:   # folder vanished, permission…
                    self.q.put(("status", f"cannot read the folder: {e}"))
                    todo = []
                for i, src in enumerate(todo):
                    if self.stop or self.paused:
                        break
                    self.q.put(("status", f"analysing {src.key}  ({i + 1}/{len(todo)})"))
                    t0 = time.time()
                    r = St.analyse(self.root, src)
                    msg = r.get("error") or f"{r.get('found')}/{r.get('targets')} hits, " \
                                            f"median {St.fmt(r.get('stats', {}).get('median_mm'))} mm"
                    self.q.put(("done", f"{src.key}: {msg}  ({time.time() - t0:.1f} s)"))
                if todo:
                    self.q.put(("status", "watching"))
            self.wake.wait(POLL_MS / 1000)
            self.wake.clear()


class App:
    def __init__(self, tkroot: tk.Tk, data_root: Path, version: str):
        self.tk = tkroot
        self.version = version
        self.q: "queue.Queue" = queue.Queue()
        self.worker = None
        self.results = []
        self.trials = {}
        self.preview_img = None
        self.preview_path = None
        self.selected_json = None
        self.sig = None
        tkroot.title(f"EyeLab {version} - campaign hits")
        tkroot.geometry("1500x880")
        self._build()
        self.set_root(data_root)
        self.tk.after(300, self._tick)
        tkroot.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ layout
    def _build(self):
        top = ttk.Frame(self.tk, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Data folder:").pack(side="left")
        self.root_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.root_var, width=70, state="readonly").pack(side="left", padx=4)
        for text, cmd in (("Change...", self.choose_root), ("Open folder", lambda: self.open_dir("")),
                          ("Open scans", lambda: self.open_dir("scans")),
                          ("Open photos", lambda: self.open_dir("photos")),
                          ("Open runs", lambda: self.open_dir("runs"))):
            ttk.Button(top, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Re-analyse selected", command=self.reanalyse_selected).pack(side="left", padx=2)
        ttk.Button(top, text="Re-analyse all", command=self.reanalyse_all).pack(side="left", padx=2)
        self.hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Hide numbers (participant present)", variable=self.hide_var,
                        command=self.refresh).pack(side="left", padx=8)

        self.status = tk.StringVar(value="starting")
        bar = ttk.Frame(self.tk, padding=(6, 0))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status, foreground="#235").pack(side="left")
        self.counts = tk.StringVar()
        ttk.Label(bar, textvariable=self.counts).pack(side="right")

        pw = ttk.PanedWindow(self.tk, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=3)
        pw.add(right, weight=2)

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)
        self.t_sources = self._tree(self.nb, "Images and trials", [
            ("key", "source", 330), ("kind", "type", 110), ("sheet", "sheet", 60), ("trial", "participant / cond", 120),
            ("hits", "hits", 60), ("fit", "fit rms mm", 75), ("mmpx", "mm/px", 60), ("bias", "bias x, y mm", 100),
            ("median", "median", 60), ("r95", "R95", 55), ("max", "max", 55), ("extra", "extra", 45),
            ("state", "state", 260)])
        self.t_sheets = self._tree(self.nb, "By sheet", [
            ("sheet", "sheet", 60), ("participant", "participant", 80), ("condition", "cond", 50),
            ("display", "display", 60), ("scan", "scan: hits / median / R95", 170),
            ("photo", "photo: hits / median / R95", 170), ("eye", "Eye: hits / median / R95", 170),
            ("photo_scan", "photo - scan", 85), ("eye_scan", "Eye - scan", 80), ("bad", "bad", 40)])
        self.t_pooled = self._tree(self.nb, "Summary", [
            ("participant", "participant", 80), ("condition", "cond", 50), ("display", "display", 60),
            ("kind", "source", 120), ("sheets", "sheets", 55), ("n", "hits", 50), ("bx", "bias x", 60),
            ("by", "bias y", 60), ("sx", "sd x", 55), ("sy", "sd y", 55), ("median", "median", 60),
            ("rmse", "RMSE", 55), ("r95", "R95", 55), ("max", "max", 55)])
        for t in (self.t_sources, self.t_sheets):
            t.bind("<<TreeviewSelect>>", self._on_select)
            t.bind("<Double-1>", self._open_check)

        self.canvas = tk.Canvas(right, background="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show_preview())
        self.canvas.bind("<Double-1>", self._open_check)
        self.detail = tk.Text(right, height=16, font=("Consolas", 9), wrap="none")
        self.detail.pack(fill="x")
        ttk.Label(right, text="Blue: nominal target. Red: dot found and its error. Orange: dot not assigned. "
                              "Double-click: open the full image.", foreground="#555").pack(anchor="w")

    def _tree(self, nb, title, cols):
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        t = ttk.Treeview(f, columns=[c[0] for c in cols], show="headings", selectmode="extended")
        for cid, head, w in cols:
            t.heading(cid, text=head)
            t.column(cid, width=w, anchor="w" if cid in ("key", "state") else "center", stretch=cid in ("key", "state"))
        sy = ttk.Scrollbar(f, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sy.set)
        t.pack(side="left", fill="both", expand=True)
        sy.pack(side="right", fill="y")
        t.tag_configure("err", foreground="#b00")
        t.tag_configure("pending", foreground="#888")
        return t

    # ------------------------------------------------------------------ folder
    def set_root(self, path: Path):
        path = St.ensure_layout(Path(path))
        if self.worker:
            self.worker.stop = True
            self.worker.wake.set()
        self.data_root = path
        self.root_var.set(str(path))
        s = _load_settings()
        s["root"] = str(path)
        _save_settings(s)
        self.worker = Worker(path, self.q)
        self.worker.start()
        self.sig = None
        self.refresh()

    def choose_root(self):
        d = filedialog.askdirectory(initialdir=str(self.data_root), title="Campaign data folder")
        if d:
            self.set_root(Path(d))

    def open_dir(self, sub):
        p = self.data_root / sub
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(str(p))

    # ------------------------------------------------------------------ actions
    def _selected_keys(self):
        keys = []
        for t in (self.t_sources, self.t_sheets):
            for iid in t.selection():
                if t is self.t_sources:
                    keys.append(iid)
                else:
                    keys += [r["key"] for r in self.results if r.get("sheet") == iid]
        return keys

    def reanalyse_selected(self):
        self._forget(self._selected_keys())

    def reanalyse_all(self):
        if messagebox.askyesno("Re-analyse all", "Analyse every image and trial again?"):
            self._forget([s.key for s in St.discover(self.data_root)])

    def _forget(self, keys):
        for s in St.discover(self.data_root):
            if s.key in keys:
                p = St.result_path(self.data_root, s)
                for q in (p, p.with_suffix(".csv"), p.with_name(p.stem + "_check.jpg")):
                    try:
                        q.unlink()
                    except OSError:
                        pass
        self.worker.wake.set()
        self.refresh()

    # ------------------------------------------------------------------ refresh
    def _tick(self):
        changed = False
        try:
            while True:
                kind, msg = self.q.get_nowait()
                if kind == "status":
                    self.status.set(msg)
                else:
                    self.status.set("done: " + msg)
                    changed = True
        except queue.Empty:
            pass
        sig = self._signature()
        if changed or sig != self.sig:
            self.sig = sig
            self.refresh()
        self.tk.after(POLL_MS // 2, self._tick)

    def _signature(self):
        """Changes when a result, a source or the trial map changes (cheap: names and mtimes)."""
        items = []
        for sub in ("results",):
            d = self.data_root / sub
            if d.exists():
                items += [(e.name, e.stat().st_mtime) for e in os.scandir(d) if e.name.endswith(".json")]
        try:
            items += [(s.key, s.mtime) for s in St.discover(self.data_root)]
        except OSError:
            pass
        items += [(p.name, p.stat().st_mtime) for p in self.data_root.glob("sheets_used*.csv")]
        return hash(tuple(sorted(items)))

    def refresh(self):
        root = self.data_root
        self.results = St.load_results(root)
        self.trials = St.trials(root)
        by_key = {r["key"]: r for r in self.results}
        hide = self.hide_var.get()
        sources = St.discover(root)
        sel = set(self.t_sources.selection())
        self.t_sources.delete(*self.t_sources.get_children())
        n_err = n_pend = 0
        for s in sources:
            r = by_key.get(s.key)
            cur = r is not None and r.get("source_version") == s.version
            sheet = (r or {}).get("sheet") or s.sheet_hint or ""
            tr = self.trials.get(sheet, {})
            trial = f"{tr.get('participant', '')} {tr.get('condition', '')}".strip()
            if not cur:
                n_pend += 1
                vals = (s.key, KIND_NAME[s.kind], sheet, trial, "", "", "", "", "", "", "", "",
                        s.refused or "waiting for analysis")
                tag = "err" if s.refused else "pending"
            elif "error" in r:
                n_err += 1
                vals = (s.key, KIND_NAME[s.kind], sheet, trial, "", "", "", "", "", "", "", "", r["error"])
                tag = "err"
            else:
                st = r.get("stats", {})
                num = (lambda v: "-" if hide else St.fmt(v))
                bias = "-" if hide or not st.get("n") else f"{st['bias_x_mm']:+.2f}, {st['bias_y_mm']:+.2f}"
                state = []
                if r.get("missing"):
                    state.append(f"no dot: {r['missing']}")
                if r.get("frames"):
                    state.append(f"{r['frames']} frames")
                if r.get("too_far"):
                    state.insert(0, "TOO FAR for dots")
                if r.get("fit_rms_mm", 0) > 0.1 and s.kind != "eye":
                    state.append("fit rms > 0.1 mm: check flatness / lens")
                state += r.get("notes", [])[:1]
                vals = (s.key, KIND_NAME[s.kind], sheet, trial, f"{r.get('found')}/{r.get('targets')}",
                        St.fmt(r.get("fit_rms_mm"), 3), St.fmt(r.get("mm_per_px"), 3), bias,
                        num(st.get("median_mm")), num(st.get("r95_mm")), num(st.get("max_mm")),
                        r.get("extra_dots", ""), "; ".join(state) or "ok")
                tag = ""
            self.t_sources.insert("", "end", iid=s.key, values=vals, tags=(tag,))
        for k in sel:
            if self.t_sources.exists(k):
                self.t_sources.selection_add(k)

        sel = set(self.t_sheets.selection())
        self.t_sheets.delete(*self.t_sheets.get_children())
        for row in St.by_sheet(self.results, self.trials):
            def cell(k):
                c = row.get(k)
                if not c:
                    return ""
                if c.get("too_far"):
                    return f"too far ({St.fmt(c['mm_per_px'])} mm/px)"
                return f"{c['found']}/{c['targets']}" + ("" if hide else f"  {St.fmt(c['median'])} / {St.fmt(c['r95'])}")

            def diff(k):
                a, b = row.get(k), row.get("scan")
                if hide or not a or not b or a.get("too_far") or a["bias"][0] is None or b["bias"][0] is None:
                    return ""
                return f"{a['bias'][0] - b['bias'][0]:+.2f}, {a['bias'][1] - b['bias'][1]:+.2f}"
            self.t_sheets.insert("", "end", iid=row["sheet"], values=(
                row["sheet"], row.get("participant", ""), row.get("condition", ""), row.get("display", ""),
                cell("scan"), cell("photo"), cell("eye"), diff("photo"), diff("eye"), row.get("bad_marks", "")))
        for k in sel:
            if self.t_sheets.exists(k):
                self.t_sheets.selection_add(k)

        self.t_pooled.delete(*self.t_pooled.get_children())
        if not hide:
            for g in St.pooled(root, self.results, self.trials):
                self.t_pooled.insert("", "end", values=(
                    g["participant"], g["condition"], g["display"], KIND_NAME.get(g["kind"], g["kind"]), g["sheets"],
                    g["n"], St.fmt(g.get("bias_x_mm")), St.fmt(g.get("bias_y_mm")), St.fmt(g.get("sd_x_mm")),
                    St.fmt(g.get("sd_y_mm")), St.fmt(g.get("median_mm")), St.fmt(g.get("rmse_mm")),
                    St.fmt(g.get("r95_mm")), St.fmt(g.get("max_mm"))))
        self.counts.set(f"{len(sources)} sources, {len(self.results)} analysed, {n_pend} waiting, {n_err} failed, "
                        f"{len(self.trials)} trials known")
        self._show_detail()

    # ------------------------------------------------------------------ preview
    def _on_select(self, event):
        t = event.widget
        sel = t.selection()
        if not sel:
            return
        iid = sel[0]
        if t is self.t_sources:
            r = next((r for r in self.results if r["key"] == iid), None)
        else:   # prefer the scan, then the photo, then the Eye
            cands = [r for r in self.results if r.get("sheet") == iid and "error" not in r]
            order = {"scan": 0, "photo": 1, "eye": 2}
            r = min(cands, key=lambda r: order.get(r["kind"], 9)) if cands else None
        self.selected_json = r["_json"] if r else None
        self._show_detail()
        self._show_preview()

    def _show_detail(self):
        self.detail.delete("1.0", "end")
        if not self.selected_json or not Path(self.selected_json).exists():
            self.detail.insert("end", "Select a row.\n\nPut scans in scans\\ (PNG or TIFF, 600 dpi, greyscale), "
                                      "photos in photos\\, pulled Beam Pro runs in runs\\.\nNew files are analysed "
                                      "automatically.")
            return
        r = json.loads(Path(self.selected_json).read_text(encoding="utf-8"))
        hide = self.hide_var.get()
        lines = [f"{r['key']}  ({KIND_NAME.get(r['kind'], r['kind'])}, analysed {r.get('analysed')})"]
        if "error" in r:
            lines.append("ERROR: " + r["error"])
        else:
            tr = self.trials.get(r.get("sheet"), {})
            lines.append(f"sheet {r.get('sheet')}  participant {tr.get('participant', '?')}  condition "
                         f"{tr.get('condition', '?')}  display {tr.get('display', '?')}  bad marks "
                         f"{tr.get('bad_marks', '?')}")
            lines.append(f"frame fit {r.get('fit_points')} points, rms {St.fmt(r.get('fit_rms_mm'), 3)} mm, "
                         f"{St.fmt(r.get('mm_per_px'), 3)} mm/px; {r.get('found')}/{r.get('targets')} hits, "
                         f"{r.get('extra_dots')} unassigned dots")
            csv_path = Path(self.selected_json).with_suffix(".csv")
            if csv_path.exists() and not hide:
                lines.append(f"{'n':>3} {'nom x':>7} {'nom y':>7} {'dx':>7} {'dy':>7} {'err':>6}")
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for h in csv.DictReader(f):
                        lines.append(f"{h['target']:>3} {h['nom_x_mm']:>7} {h['nom_y_mm']:>7} "
                                     f"{(h['dx_mm'] or '-'):>7} {(h['dy_mm'] or '-'):>7} {(h['err_mm'] or 'no dot'):>6}")
            for n in r.get("notes", []):
                lines.append("note: " + n)
        self.detail.insert("end", "\n".join(lines))

    def _check_path(self):
        if not self.selected_json:
            return None
        p = Path(self.selected_json)
        p = p.with_name(p.stem + "_check.jpg")
        return p if p.exists() else None

    def _show_preview(self):
        from PIL import Image, ImageTk
        self.canvas.delete("all")
        p = self._check_path()
        if p is None or self.hide_var.get():
            return
        w, h = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        im = Image.open(p)
        im.thumbnail((w, h))
        self.preview_img = ImageTk.PhotoImage(im)
        self.canvas.create_image(w // 2, h // 2, image=self.preview_img)

    def _open_check(self, _e=None):
        p = self._check_path()
        if p:
            os.startfile(str(p))

    def _close(self):
        if self.worker:
            self.worker.stop = True
            self.worker.wake.set()
        self.tk.destroy()


def main(root: str = None) -> int:
    try:
        from eyelab_version import __version__ as version
    except ImportError:
        version = "?"
    data_root = Path(root) if root else Path(_load_settings().get("root", St.DEFAULT_ROOT))
    tkroot = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(tkroot, data_root, version)
    tkroot.mainloop()
    return 0
