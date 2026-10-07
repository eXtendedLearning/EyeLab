"""Campaign state: what has been measured, with which calibration and plates.

``<root>/campaign.json`` (default root ``python/validation_runs``)::

    {"schema": "eyelab-validation-campaign/1", "created": ...,
     "camera_index": 0, "focus": null | float,
     "camera": "<profile>",               # config/cameras/<profile>.yaml
     "imu": "off" | "rear" | "front",     # device IMU coupled to the camera
     "instrument": {"name", "accuracy_mm"} | null,   # +-a of the length instrument
     "calibration": {"path", "camera", "digest", "rms_px", "image_size", "recorded"} | null,
     "plates": {"A": {"nominal", "measured", "readings"}, "L": {...}},
     "plan": [...],                       # see plan.py; editable by hand
     "sessions": {"<id>": {"status": "done" | "skipped", "path", "when",
                           "calibration_digest", "attempts", "note", ...}}}

Calibration and plate files live in the shared config tree (``eyelab_paths``)
and are stored relative to ``python/config``; session folders are relative to
``python/`` (the runner's working directory). Campaign files written before
the config tree existed are migrated on open. No file is ever deleted: a
redone session's folder moves to ``sessions/_rejected/``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import eyelab_paths as paths

from .plan import STEP_POSITIONS_MM, default_plan

SCHEMA = "eyelab-validation-campaign/1"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Campaign:
    def __init__(self, root, state: dict):
        self.root = Path(root)
        self.state = state

    # ── persistence ──
    @classmethod
    def open(cls, root, reps: int = 3) -> "Campaign":
        root = Path(root)
        f = root / "campaign.json"
        if f.exists():
            state = json.loads(f.read_text(encoding="utf-8"))
            if state.get("schema") != SCHEMA:
                raise ValueError(f"{f}: unknown schema {state.get('schema')!r}")
            c = cls(root, state)
            if c._migrate():
                c.save()
            return c
        root.mkdir(parents=True, exist_ok=True)
        state = {"schema": SCHEMA, "created": now(), "camera_index": 0, "focus": None,
                 "camera": paths.active_camera(), "imu": "off", "instrument": None,
                 "calibration": None,
                 "plates": {k: {"nominal": f"targets/plate_{k}.target.json",
                                "measured": None, "readings": None} for k in ("A", "L")},
                 "plan": default_plan(reps), "sessions": {}}
        c = cls(root, state)
        c.save()
        return c

    def _migrate(self) -> bool:
        """Bring a campaign written before the shared config tree up to date."""
        s, changed = self.state, False
        for k, v in (("camera", paths.active_camera()), ("imu", "off"), ("instrument", None)):
            if k not in s:
                s[k], changed = v, True
        for plate in s.get("plates", {}).values():
            for key in ("nominal", "measured"):
                v = plate.get(key)
                if v and not paths.from_stored(v).exists() and "validation_printables" in v.replace("\\", "/"):
                    name = Path(v.replace("\\", "/")).name
                    if (paths.TARGETS_DIR / name).exists():
                        plate[key], changed = f"targets/{name}", True
        cal = s.get("calibration")
        if cal and not paths.from_stored(cal["path"]).exists():
            from .target import file_digest
            cand = paths.camera_file(s["camera"])
            if cand.exists() and file_digest(cand) == cal.get("digest"):
                cal.update(path=paths.to_stored(cand), camera=cand.stem)
                changed = True
        return changed

    def save(self) -> None:
        f = self.root / "campaign.json"
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(f)

    # ── paths ──
    @property
    def sessions_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def session_path(self, sid: str) -> Path:
        return self.sessions_dir / sid

    # ── plan / status ──
    @property
    def plan(self) -> List[dict]:
        return self.state["plan"]

    def spec(self, sid: str) -> dict:
        return next(s for s in self.plan if s["id"] == sid)

    def status(self, sid: str) -> str:
        return self.state["sessions"].get(sid, {}).get("status", "pending")

    def next_pending(self) -> Optional[dict]:
        return next((s for s in self.plan if self.status(s["id"]) == "pending"), None)

    def counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for s in self.plan:
            c = out.setdefault(s["test"], {"done": 0, "skipped": 0, "pending": 0, "total": 0})
            c[self.status(s["id"])] += 1
            c["total"] += 1
        return out

    def target_for(self, plate: str) -> Optional[str]:
        """Absolute path of the measured target of ``plate`` (None until measured)."""
        p = paths.from_stored(self.state["plates"][plate]["measured"])
        return str(p) if p else None

    def nominal_for(self, plate: str) -> str:
        return str(paths.from_stored(self.state["plates"][plate]["nominal"]))

    def calibration_path(self) -> Optional[Path]:
        cal = self.state.get("calibration")
        return paths.from_stored(cal["path"]) if cal else None

    @property
    def camera_file(self) -> Path:
        return paths.camera_file(self.state.get("camera") or paths.active_camera())

    @property
    def instrument_mm(self) -> Optional[float]:
        inst = self.state.get("instrument")
        return float(inst["accuracy_mm"]) if inst else None

    def ready_problems(self, spec: dict) -> List[str]:
        """Why a session cannot be captured yet (empty when it can)."""
        problems = []
        if not self.state.get("calibration"):
            problems.append("no calibration recorded for this campaign (menu: calibrate)")
        elif not self.calibration_path().exists():
            problems.append(f"calibration file missing: {self.calibration_path()}")
        if not self.target_for(spec["plate"]):
            problems.append(f"plate {spec['plate']} not measured yet (menu: plate measurements)")
        return problems

    def calibration_changed(self) -> bool:
        """True when the calibration file no longer matches the one recorded."""
        from .target import file_digest
        cal = self.state.get("calibration")
        p = self.calibration_path()
        return bool(cal) and p.exists() and file_digest(p) != cal["digest"]

    # ── updates ──
    def record_calibration(self, path: str) -> dict:
        import cv2
        from .target import file_digest
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        info = {"path": paths.to_stored(path), "camera": Path(path).stem, "digest": file_digest(path),
                "rms_px": float(fs.getNode("rms_error").real()),
                "image_size": [int(fs.getNode("image_width").real()),
                               int(fs.getNode("image_height").real())],
                "recorded": now()}
        fs.release()
        self.state["calibration"] = info
        self.save()
        return info

    def mark(self, sid: str, status: str, **extra) -> None:
        entry = self.state["sessions"].setdefault(sid, {"attempts": 0})
        entry.update(status=status, when=now(), **extra)
        self.save()

    def count_attempt(self, sid: str) -> None:
        entry = self.state["sessions"].setdefault(sid, {"attempts": 0})
        entry["attempts"] = entry.get("attempts", 0) + 1
        self.save()

    def reject_folder(self, sid: str) -> Optional[Path]:
        """Move an existing session folder aside (never deleted)."""
        src = self.session_path(sid)
        if not src.exists():
            return None
        dst_dir = self.sessions_dir / "_rejected"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{sid}__{time.strftime('%Y%m%d_%H%M%S')}"
        src.rename(dst)
        return dst

    def reopen(self, sid: str) -> None:
        """Back to pending (its folder is moved aside on the next capture)."""
        self.state["sessions"].pop(sid, None)
        self.save()

    # ── analysis groups ──
    def done_sessions(self, test: str) -> List[Path]:
        return [self.session_path(s["id"]) for s in self.plan
                if s["test"] == test and self.status(s["id"]) == "done"
                and self.session_path(s["id"]).exists()]

    def step_pairs(self) -> List[dict]:
        """B3 pairs (position 0 vs position p) within each repetition, with the
        caliper-measured positions where they were entered."""
        by_rep: Dict[int, Dict[int, dict]] = {}
        for s in self.plan:
            if s["test"] == "B3" and self.status(s["id"]) == "done":
                by_rep.setdefault(s["rep"], {})[s["condition"]["position_mm"]] = s
        pairs = []
        for rep, pos in sorted(by_rep.items()):
            if 0 not in pos:
                continue
            base = pos[0]
            for p in STEP_POSITIONS_MM[1:]:
                if p not in pos:
                    continue
                e0 = self.state["sessions"][base["id"]]
                e1 = self.state["sessions"][pos[p]["id"]]
                known = e1.get("measured_position_mm", p) - e0.get("measured_position_mm", 0.0)
                pairs.append({"rep": rep, "nominal_mm": p, "known_mm": known,
                              "a": self.session_path(base["id"]), "b": self.session_path(pos[p]["id"])})
        return pairs
