"""Per-session analysis (JSON, CSV, plots) and per-condition aggregation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .metrics import (FIELDS, detection_stats, overlay_jitter_mm, per_node, pose_jitter,
                      projected_jitter_px, registration_errors)
from .session import load_session
from .stats import across_units, summarize
from .target import load_target, target_digest
from .uncertainty import expanded, tur

NAN = float("nan")


def _builtin(o):
    if isinstance(o, dict):
        return {str(k): _builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_builtin(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def _jitter(s, which):
    rv, tv = s.poses(which)
    if len(tv) < 2:
        return None
    j = pose_jitter(rv, tv)
    return {"n": j["n"], "t_mean_mm": j["t_mean_mm"], "t_sd_xyz_mm": j["t_sd_xyz_mm"],
            "t_radial_dev_mm": summarize(j["t_radial_dev_mm"]),
            "rot_dev_deg": summarize(j["rot_dev_deg"])}, j


def achieved_setup(records) -> dict:
    """Median setup geometry over the recorded frames (setup assistant log):
    what the condition actually was, as opposed to its nominal value."""
    out = {}
    for k in ("distance_mm", "angle_deg", "elevation_deg", "centre_offset"):
        v = [r.extra["setup"][k] for r in records if "setup" in r.extra and k in r.extra["setup"]]
        if v:
            out[k] = float(np.median(v))
    imu = [r.extra["imu"] for r in records if "imu" in r.extra]
    if imu:
        out["imu_axis_from_down_deg"] = float(np.median([i["axis_from_down_deg"] for i in imu]))
        out["imu_still_rate"] = float(np.mean([bool(i.get("still")) for i in imu]))
    return out


def analyze_session(session_path, target_path, out_dir, search_radius: float = 5.0,
                    every: int = 1, plots: bool = True) -> dict:
    s = load_session(session_path)
    t = load_target(target_path)
    digest = target_digest(target_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"session": str(s.path), "platform": s.meta.get("platform"),
               "condition": s.meta.get("condition", {}), "target": t.name, "target_digest": digest,
               "target_digest_at_capture": s.meta.get("target_digest"),
               "ground_truth_uncertainty_mm": t.uncertainty_mm,
               "ground_truth_U_mm": expanded(t.uncertainty_mm),
               "ground_truth_tur_1mm": tur(t.uncertainty_mm) if t.uncertainty_mm == t.uncertainty_mm else NAN,
               "achieved_setup": achieved_setup(s.records),
               "target_measured": bool(t.provenance.get("measured")),
               "detection": detection_stats(s.records)}
    jit = {}
    for which in ("measured", "pose"):
        r = _jitter(s, which)
        if r is not None:
            summary[f"jitter_{which}"], jit[which] = r
    summary["projected_jitter_px"] = summarize(projected_jitter_px(s.records))

    rows, qa = registration_errors(s, t, search_radius, every)
    with open(out / "registration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    nodes = per_node(rows)
    if nodes:
        with open(out / "per_node.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(nodes[0]))
            w.writeheader()
            w.writerows(nodes)
    det = [r for r in rows if r.detected]
    by_m, by_face = defaultdict(list), defaultdict(list)
    for r in det:
        by_m["coasted" if r.coasted else str(r.n_markers)].append(r.err_mm)
        by_face[r.face].append(r.err_mm)
    summary["registration"] = {
        "points_total": len(rows), "points_detected": len(det),
        "err_px": summarize([r.err_px for r in det]),
        "err_px_raw": summarize([r.err_px_raw for r in det]),
        "err_px_drawn": summarize([r.err_px_drawn for r in det]),
        "err_mm": summarize([r.err_mm for r in det]),
        "err_mm_raw": summarize([r.err_mm_raw for r in det]),
        "err_mm_by_markers": {k: summarize(v) for k, v in sorted(by_m.items())},
        "err_mm_by_face": {k: summarize(v) for k, v in sorted(by_face.items())},
        "overlay_jitter_mm": summarize(overlay_jitter_mm(rows)),
        "max_node_bias_mm": max((n["bias_mm"] for n in nodes), default=float("nan")),
        "qa": {"frames_analysed": qa["frames_analysed"], "frames_no_marker": qa["frames_no_marker"],
               "homography_residual_px": summarize(qa["homography_residual_px"]),
               "seed_offset_px": summarize([r.seed_offset_px for r in det])},
    }
    if digest != s.meta.get("target_digest") and s.meta.get("target_digest"):
        summary["warning"] = ("target file differs from the one used at capture "
                              "(expected after apply-measurements; check it is the same plate)")
    (out / "summary.json").write_text(json.dumps(_builtin(summary), indent=2), encoding="utf-8")
    if plots:
        _plots(out, det, jit, nodes, t)
    return summary


def _plots(out: Path, det, jit, nodes, target) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for a, key, lab in ((ax[0], "err_px", "overlay error [px]"),
                        (ax[1], "err_mm", "overlay error on face [mm]")):
        v = np.array([getattr(r, key) for r in det], float)
        v = v[np.isfinite(v)]
        if len(v):
            a.hist(v, bins=40, color="#4a6fa5")
        a.set_xlabel(lab)
        a.set_ylabel("node-frames")
    fig.tight_layout()
    fig.savefig(out / "registration_hist.png", dpi=150)
    plt.close(fig)

    if jit:
        fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
        for which, style in (("measured", "-"), ("pose", "--")):
            if which in jit:
                ax[0].plot(jit[which]["t_radial_dev_mm"], style, lw=0.8, label=which)
                ax[1].plot(jit[which]["rot_dev_deg"], style, lw=0.8, label=which)
        ax[0].set_ylabel("translation dev. [mm]")
        ax[1].set_ylabel("rotation dev. [deg]")
        ax[1].set_xlabel("posed frame")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "jitter.png", dpi=150)
        plt.close(fig)

    if nodes:
        faces = sorted({n["face"] for n in nodes})
        fig, axs = plt.subplots(1, len(faces), figsize=(5 * len(faces), 4), squeeze=False)
        for a, face in zip(axs[0], faces):
            ns = [n for n in nodes if n["face"] == face]
            uv = target.face_coords([n["point_id"] for n in ns]) * 1000.0
            b = np.array([[n["bias_du_mm"], n["bias_dv_mm"]] for n in ns])
            scale = 10.0
            q = a.quiver(uv[:, 0], uv[:, 1], b[:, 0] * scale, b[:, 1] * scale, angles="xy",
                         scale_units="xy", scale=1, color="#b5452b")
            a.quiverkey(q, 0.85, 1.04, scale, "1 mm", labelpos="E", coordinates="axes")
            a.scatter(uv[:, 0], uv[:, 1], s=8, c="k")
            ext = np.vstack([uv, uv + b * scale])
            lo, hi = ext.min(axis=0) - 10.0, ext.max(axis=0) + 10.0
            a.set_xlim(lo[0], hi[0])
            a.set_ylim(lo[1], hi[1])
            a.set_aspect("equal")
            a.set_title(f"face {face}: mean error x{scale:g}")
            a.set_xlabel("u [mm]")
            a.set_ylabel("v [mm]")
        fig.tight_layout()
        fig.savefig(out / "bias_map.png", dpi=150)
        plt.close(fig)


def _cond_key(cond: dict) -> str:
    return json.dumps({k: v for k, v in sorted(cond.items()) if k != "rep"})


def aggregate(summaries, out_dir) -> None:
    """``sessions.csv``: one row per session. ``conditions.csv``: repositioned
    repetitions pooled per condition, the session being the statistical unit."""
    out = Path(out_dir)
    cols = ["session", "platform", "condition", "measured_rate", "coasted_rate",
            "t_radial_dev_mm_median", "rot_dev_deg_median", "err_px_median", "err_mm_rmse",
            "err_mm_median", "err_mm_p95", "err_mm_raw_rmse", "overlay_jitter_mm_p95",
            "max_node_bias_mm", "homography_residual_px_median", "gt_U_mm",
            "achieved_distance_mm", "achieved_angle_deg", "achieved_elevation_deg"]

    def row(s):
        j = s.get("jitter_measured", {})
        r = s["registration"]
        return [s["session"], s["platform"], json.dumps(s["condition"]),
                s["detection"].get("measured_rate"), s["detection"].get("coasted_rate"),
                j.get("t_radial_dev_mm", {}).get("median"), j.get("rot_dev_deg", {}).get("median"),
                r["err_px"].get("median"), r["err_mm"].get("rmse"), r["err_mm"].get("median"),
                r["err_mm"].get("p95"), r["err_mm_raw"].get("rmse"),
                r["overlay_jitter_mm"].get("p95"), r["max_node_bias_mm"],
                r["qa"]["homography_residual_px"].get("median"), s.get("ground_truth_U_mm"),
                s.get("achieved_setup", {}).get("distance_mm"),
                s.get("achieved_setup", {}).get("angle_deg"),
                s.get("achieved_setup", {}).get("elevation_deg")]

    with open(out / "sessions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in summaries:
            w.writerow(row(s))

    groups = defaultdict(list)
    for s in summaries:
        groups[(s["platform"], _cond_key(s["condition"]))].append(s)
    metrics = {"err_mm_rmse": lambda s: s["registration"]["err_mm"].get("rmse"),
               "err_mm_p95": lambda s: s["registration"]["err_mm"].get("p95"),
               "overlay_jitter_mm_p95": lambda s: s["registration"]["overlay_jitter_mm"].get("p95"),
               "measured_rate": lambda s: s["detection"].get("measured_rate")}
    with open(out / "conditions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["platform", "condition", "n_reps"] +
                   [f"{k}_{st}" for k in metrics for st in ("mean", "sd")])
        for (plat, cond), ss in sorted(groups.items()):
            vals = []
            for f in metrics.values():
                a = across_units([f(s) if f(s) is not None else float("nan") for s in ss])
                vals += [a.get("mean"), a.get("sd")]
            w.writerow([plat, cond, len(ss)] + vals)
