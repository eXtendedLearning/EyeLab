"""Import exported gate-run zips into one de-duplicated tree.

Layout under the GateRuns root (gitignored)::

    _inbox/        transfer zips copied from the Surface (GateRuns_<host>_<ts>.zip)
    _imported/     zips after a successful import (redundant; safe to delete)
    runs/<run_id>/ one folder per run: camera.csv, pose.csv, events.csv, perf.csv,
                   run.json, native.log, snapshots/, plus the matching player.log
    _unmatched/    player logs with no run folder (vendor sample, crashed starts)

A run folder is ``<yyyymmdd>_<HHMMSS>_<gate>_<label>``; a player log is
``player_<label>_<yyyymmdd>_<HHMMSS>.log`` and is written by the launcher a few
seconds *before* the harness creates its run folder. Older Surface exports were
cumulative, so the same run arrives many times: an existing run is kept unless
the incoming copy is strictly larger (a later, more complete flush).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

RUN_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_(?P<gate>G1|G2|G3|manual)_(?P<label>.+?)(?:_\d+)?$")
LOG_RE = re.compile(r"^player_(?P<label>.+)_(?P<ts>\d{8}_\d{6})\.log$")

#: The launcher starts the player, which creates its run folder this many seconds later at most.
MAX_LOG_TO_RUN_S = 120


@dataclass
class ImportReport:
    added: List[str] = field(default_factory=list)
    replaced: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    logs_attached: List[str] = field(default_factory=list)
    logs_unmatched: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        return [
            f"runs added:     {len(self.added)}",
            f"runs replaced:  {len(self.replaced)}",
            f"runs skipped:   {len(self.skipped)} (already present)",
            f"logs attached:  {len(self.logs_attached)}",
            f"logs unmatched: {len(self.logs_unmatched)}",
        ]


def default_root() -> Path:
    """GateRuns of the gate project, relative to this file (python/xreal_gates/)."""
    return Path(__file__).resolve().parents[2] / "unity_projects" / "xreal_pc_gates" / "GateRuns"


def _ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d_%H%M%S")


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def find_runs(folder: Path) -> List[Path]:
    """Run folders anywhere below ``folder`` (exports may nest them)."""
    return sorted(p for p in folder.rglob("*") if p.is_dir() and RUN_RE.match(p.name))


def find_logs(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("player_*.log") if p.is_file() and LOG_RE.match(p.name))


def match_log(log_name: str, run_names: Iterable[str]) -> Optional[str]:
    """Run folder that a player log belongs to: same label, run started 0..120 s after the log."""
    m = LOG_RE.match(log_name)
    if not m:
        return None
    log_ts = _ts(m.group("ts"))
    best, best_dt = None, None
    for name in run_names:
        r = RUN_RE.match(name)
        if not r or r.group("label") != m.group("label"):
            continue
        dt = (_ts(r.group("ts")) - log_ts).total_seconds()
        if 0 <= dt <= MAX_LOG_TO_RUN_S and (best_dt is None or dt < best_dt):
            best, best_dt = name, dt
    return best


def import_folder(source: Path, root: Path, report: Optional[ImportReport] = None) -> ImportReport:
    """Import every run folder and player log found below ``source`` into ``root``."""
    report = report or ImportReport()
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for run in find_runs(source):
        dest = runs_dir / run.name
        if not dest.exists():
            shutil.copytree(run, dest)
            report.added.append(run.name)
        elif _size(run) > _size(dest) - _size_of(dest / "player.log"):
            kept_log = None
            if (dest / "player.log").exists():
                kept_log = (dest / "player.log").read_bytes()
            shutil.rmtree(dest)
            shutil.copytree(run, dest)
            if kept_log is not None:
                (dest / "player.log").write_bytes(kept_log)
            report.replaced.append(run.name)
        else:
            report.skipped.append(run.name)

    known = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    for log in find_logs(source):
        run = match_log(log.name, known)
        if run is not None:
            target = runs_dir / run / "player.log"
            if not target.exists() or log.stat().st_size > target.stat().st_size:
                shutil.copy2(log, target)
            report.logs_attached.append(f"{log.name} -> {run}")
        else:
            unmatched = root / "_unmatched"
            unmatched.mkdir(parents=True, exist_ok=True)
            if not (unmatched / log.name).exists():
                shutil.copy2(log, unmatched / log.name)
            report.logs_unmatched.append(log.name)
    return report


def _size_of(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def import_zip(zip_path: Path, root: Path, move_to_imported: bool = True) -> ImportReport:
    """Unpack ``zip_path`` to a temporary folder and import it; then move the zip to ``_imported/``."""
    with tempfile.TemporaryDirectory(prefix="xreal_gates_") as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        report = import_folder(Path(tmp), root)
    if move_to_imported:
        done = root / "_imported"
        done.mkdir(parents=True, exist_ok=True)
        target = done / zip_path.name
        if target.resolve() != zip_path.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(zip_path), str(target))
    return report


def import_inbox(root: Path) -> List[tuple]:
    """Import every zip in ``root/_inbox``; returns (zip name, report) pairs."""
    inbox = root / "_inbox"
    results = []
    for z in sorted(inbox.glob("*.zip")) if inbox.exists() else []:
        results.append((z.name, import_zip(z, root)))
    return results
