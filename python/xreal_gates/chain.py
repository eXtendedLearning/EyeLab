"""Is every run using the same registration chain? ``python -m xreal_gates chain RUN [RUN ...]``.

Everything between a camera pixel and a drawn point, per run, from its logs: lens calibration
(``pose_service.json``), RGB-camera-from-head pose, tracking mode, render rate, camera mode, target
definition, lock filter, and the view offset at the start and the end. Fields that must agree are
compared across the runs and any difference is flagged; the view offset and tracking mode are
listed but allowed to differ.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

#: Fields that define the chain; a difference among them means the runs are not comparable.
MUST_MATCH = ("lens", "rgb_from_head", "render_fps", "camera_mode", "targets")
ALSO_SHOWN = ("tracking", "filter", "cam_latency", "offset_start", "offset_end")


def _rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _offset(detail: str) -> str:
    """The ``x .. y .. z .. mm, yaw .. pitch .. roll .. deg`` part of a view_offset detail."""
    i = detail.rfind("x ")
    return detail[i:] if i >= 0 else detail


def fingerprint(run_dir: Path) -> Dict[str, str]:
    run_dir = Path(run_dir)
    events = _rows(run_dir / "events.csv")
    fp: Dict[str, str] = {"run": run_dir.name}
    fp["lens"] = "?"
    service = run_dir / "pose_service.json"
    if service.exists():
        d = json.loads(service.read_text(encoding="utf-8"))
        lens = json.dumps({"K": d.get("K"), "dist": d.get("dist")}, sort_keys=True)
        k = d.get("K") or [[0, 0, 0], [0, 0, 0]]
        fp["lens"] = f"fx {k[0][0]:.1f} fy {k[1][1]:.1f} [{hashlib.sha256(lens.encode()).hexdigest()[:8]}]"
        names = d.get("targets") or [{"name": d.get("target", "?"), "source": d.get("target_source", "?")}]
        fp["targets"] = ", ".join(f"{t.get('name')} ({Path(str(t.get('source', ''))).name})" for t in names)
    for key in ("rgb_from_head", "tracking", "render_fps", "camera_mode", "filter", "offset_start", "offset_end"):
        fp.setdefault(key, "?")
    fp["cam_latency"] = "0 ms (pose at sending; before 0.3.005)"
    fp.setdefault("targets", "?")
    for e in events:
        kind, result, detail = e["event"], e["result"], e["detail"]
        if kind == "rgb_from_head":
            fp["rgb_from_head"] = f"{result} {detail}"
        elif kind == "switch_tracking" and result == "True":
            fp["tracking"] = detail.split()[0]
        elif kind == "render_fps":
            fp["render_fps"] = str(result)
        elif kind == "camera_mode" and fp["camera_mode"] == "?":
            fp["camera_mode"] = result
        elif kind == "cam_latency":
            fp["cam_latency"] = f"{result} ms"
        elif kind == "lock_filter" and result != "ADJUST":
            fp["filter"] = result
        elif kind == "view_offset":
            if result == "LOADED":
                fp["offset_start"] = fp["offset_end"] = _offset(detail)
            elif result in ("CHANGED", "SAVED"):
                fp["offset_end"] = _offset(detail)
    return fp


def report(run_dirs) -> List[str]:
    fps = [fingerprint(r) for r in run_dirs]
    lines = []
    for fp in fps:
        lines.append(fp["run"])
        for key in MUST_MATCH + ALSO_SHOWN:
            lines.append(f"  {key:14s} {fp[key]}")
    if len(fps) > 1:
        differs = [k for k in MUST_MATCH if len({fp[k] for fp in fps}) > 1]
        lines.append("chain: " + ("IDENTICAL across these runs (" + ", ".join(MUST_MATCH) + ")" if not differs
                                  else "DIFFERS in " + ", ".join(differs)))
    return lines
