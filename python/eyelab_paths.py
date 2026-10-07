"""Single source of every EyeLab configuration path (GUI, calibrate.py, validation).

Everything EyeLab and the validation scripts read or write as configuration
lives under ``python/config``::

    config/
      active.json                 machine-local selection (git-ignored)
      cameras/<camera>.yaml       one intrinsic calibration per camera profile
      cameras/history/            automatic backups before a calibration is replaced
      markers/*.markers.json      marker configs (structures, plates, hammer)
      geometry/*.unv              geometries (structures, plates)
      targets/*.target.json       validation ground truth (nominal and measured)

An intrinsic calibration is valid for one camera at one resolution and focus,
so calibrations are kept per *camera profile* (a name such as
``desktop_c920`` or ``surface_rear``) instead of in one shared file. The
active profile, marker config and geometry are remembered in ``active.json``,
so the GUI and the validation runner always start from the same files.

Paths stored in ``active.json`` (and in the validation campaign) are relative
to ``config/`` when they are inside it, so the repository can move.

No tkinter or OpenCV import here: safe to use from tests and CLI tools.
"""
from __future__ import annotations

import json
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional

PYTHON_DIR = Path(__file__).resolve().parent
REPO_DIR = PYTHON_DIR.parent
CONFIG_DIR = PYTHON_DIR / "config"
CAMERAS_DIR = CONFIG_DIR / "cameras"
CAMERA_HISTORY_DIR = CAMERAS_DIR / "history"
MARKER_CONFIGS_DIR = CONFIG_DIR / "markers"
GEOMETRY_DIR = CONFIG_DIR / "geometry"
TARGETS_DIR = CONFIG_DIR / "targets"
ACTIVE_FILE = CONFIG_DIR / "active.json"

PRINTABLES_DIR = REPO_DIR / "docs" / "validation_printables"   # PDFs only
TEST_ASSETS_DIR = REPO_DIR / "test_assets"                       # unit-test fixtures
VALIDATION_RUNS_DIR = PYTHON_DIR / "validation_runs"

DEFAULT_MARKER_CONFIG = MARKER_CONFIGS_DIR / "marker_config.json"
HAMMER_MARKER_CONFIG = MARKER_CONFIGS_DIR / "hammer_marker_config.json"

ACTIVE_SCHEMA = "eyelab-active/1"
_ACTIVE_KEYS = ("camera", "markers", "geometry")


def ensure_dirs() -> None:
    for d in (CAMERAS_DIR, CAMERA_HISTORY_DIR, MARKER_CONFIGS_DIR, GEOMETRY_DIR, TARGETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── camera profiles ──────────────────────────────────────────────────────────

def slug(name: str) -> str:
    """File-safe camera profile name: lowercase letters, digits, '_' and '-'."""
    s = re.sub(r"[^a-z0-9_-]+", "_", str(name).strip().lower()).strip("_")
    if not s:
        raise ValueError(f"invalid camera profile name: {name!r}")
    return s


def default_camera_name() -> str:
    """``<hostname>_cam0``: different machines get different profiles by default."""
    try:
        host = socket.gethostname()
    except OSError:
        host = "this_pc"
    return slug(f"{host}_cam0")


def camera_file(name: str) -> Path:
    return CAMERAS_DIR / f"{slug(name)}.yaml"


def list_cameras() -> List[str]:
    if not CAMERAS_DIR.exists():
        return []
    return sorted(p.stem for p in CAMERAS_DIR.glob("*.yaml"))


def active_camera() -> str:
    """Selected profile; else the only existing one; else the machine default."""
    name = load_active().get("camera")
    if name:
        return slug(name)
    cams = list_cameras()
    return cams[0] if len(cams) == 1 else default_camera_name()


def active_calibration_file() -> Path:
    return camera_file(active_camera())


def set_active_camera(name: str) -> Path:
    save_active(camera=slug(name))
    return camera_file(name)


def backup_calibration(path: Path) -> Optional[Path]:
    """Copy an existing calibration to ``cameras/history`` before it is replaced."""
    path = Path(path)
    if not path.exists():
        return None
    CAMERA_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    dst = CAMERA_HISTORY_DIR / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    n = 1
    while dst.exists():
        dst = CAMERA_HISTORY_DIR / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}_{n}.yaml"
        n += 1
    shutil.copy2(path, dst)
    return dst


# ── active selection ─────────────────────────────────────────────────────────

def to_stored(path) -> str:
    """Path as stored in JSON: relative to config/ when inside it, POSIX separators."""
    p = Path(path)
    try:
        return p.resolve().relative_to(CONFIG_DIR.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def from_stored(value) -> Optional[Path]:
    if not value:
        return None
    p = Path(str(value).replace("\\", "/"))
    return p if p.is_absolute() else CONFIG_DIR / p


def load_active() -> Dict[str, str]:
    try:
        d = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: d[k] for k in _ACTIVE_KEYS if isinstance(d.get(k), str) and d[k]}


def save_active(**updates) -> Dict[str, str]:
    """Update the remembered selection. ``markers``/``geometry`` take paths."""
    d = load_active()
    for k, v in updates.items():
        if k not in _ACTIVE_KEYS:
            raise KeyError(k)
        if v is None:
            d.pop(k, None)
        else:
            d[k] = slug(v) if k == "camera" else to_stored(v)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ACTIVE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": ACTIVE_SCHEMA, **d}, indent=2), encoding="utf-8")
    tmp.replace(ACTIVE_FILE)
    return d


def active_marker_config() -> Path:
    return from_stored(load_active().get("markers")) or DEFAULT_MARKER_CONFIG


def active_geometry() -> Optional[Path]:
    return from_stored(load_active().get("geometry"))


# ── one-off migration of the old flat layout ─────────────────────────────────

def migrate_legacy_layout() -> List[str]:
    """Move files from the pre-2026-10 flat ``config/`` into the tree. Idempotent.

    - ``config/camera_params.yaml`` -> ``cameras/<machine default>.yaml``
      (selected as active camera);
    - ``config/*.json`` marker configs -> ``markers/``;
    - ``validation_runs/calibration_backups/*`` -> ``cameras/history/``.
    Existing destination files are never overwritten. Returns what was done.
    """
    done: List[str] = []
    ensure_dirs()
    old_cal = CONFIG_DIR / "camera_params.yaml"
    if old_cal.exists():
        dst = camera_file(default_camera_name())
        if not dst.exists():
            old_cal.replace(dst)
            if not load_active().get("camera"):
                save_active(camera=dst.stem)
            done.append(f"{old_cal.name} -> cameras/{dst.name}")
    for f in sorted(CONFIG_DIR.glob("*.json")):
        if f.name == ACTIVE_FILE.name:
            continue
        dst = MARKER_CONFIGS_DIR / f.name
        if not dst.exists():
            f.replace(dst)
            done.append(f"{f.name} -> markers/{f.name}")
    old_backups = VALIDATION_RUNS_DIR / "calibration_backups"
    if old_backups.is_dir():
        for f in sorted(old_backups.glob("*.yaml")):
            dst = CAMERA_HISTORY_DIR / f.name
            if not dst.exists():
                f.replace(dst)
                done.append(f"calibration_backups/{f.name} -> cameras/history/{f.name}")
    return done
