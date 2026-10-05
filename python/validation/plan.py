"""The default measurement campaign: which sessions, in which order, set up how.

A plan is a list of plain dicts (stored in ``campaign.json`` so it can be
edited by hand): ``{"id", "test", "plate", "rep", "condition", "frames"}``.
Setup instructions are generated from the spec, not stored, so they stay in
step with the code. Repetitions are the outer loop on purpose: every
repetition re-creates each condition from scratch (the plate is lifted and set
down again), which is what makes between-session spread meaningful.
"""
from __future__ import annotations

from typing import Dict, List

TESTS = {
    "B1": "static pose: jitter, detection, registration (plate A, 3 markers)",
    "B1b": "single-marker sustain (plate A, lock with 3, then 2 covered)",
    "B2L": "two-face registration (plate L)",
    "B3": "step error: known translations of plate A",
}
STEP_POSITIONS_MM = (0, 5, 10, 20)


def default_plan(reps: int = 3) -> List[dict]:
    plan: List[dict] = []

    def add(test, plate, rep, frames, **cond):
        tag = "_".join(f"{k[0]}{_fmt(v)}" for k, v in cond.items() if k != "markers_visible")
        plan.append({"id": f"{test}_{plate}_{tag}_r{rep}", "test": test, "plate": plate,
                     "rep": rep, "frames": frames, "condition": dict(cond)})

    for rep in range(1, reps + 1):
        for d in (400, 600, 900):
            for a in (0, 30, 50):
                add("B1", "A", rep, 300, distance_mm=d, angle_deg=a)
    for rep in range(1, reps + 1):
        for a in (0, 30):
            add("B1b", "A", rep, 300, distance_mm=600, angle_deg=a, markers_visible=1)
    for rep in range(1, reps + 1):
        for d in (500, 800):
            for e in (25, 45, 65):
                add("B2L", "L", rep, 300, distance_mm=d, elevation_deg=e)
    for rep in range(1, reps + 1):
        for p in STEP_POSITIONS_MM:
            add("B3", "A", rep, 150, position_mm=p, distance_mm=600, angle_deg=30)
    return plan


def _fmt(v) -> str:
    return f"{v:02d}" if isinstance(v, int) and v < 10 else f"{v:g}"


def instructions(spec: dict) -> List[str]:
    """Setup card shown before a session."""
    c, t, rep = spec["condition"], spec["test"], spec["rep"]
    first_rep_note = ("Repetition 1." if rep == 1 else
                      f"Repetition {rep}: build this condition again from scratch (lift the plate, "
                      "set it down, re-set distance and angle). Do not just record again.")
    if t in ("B1", "B1b"):
        lines = [
            first_rep_note,
            f"Plate A. Angle between camera axis and plate normal: {c['angle_deg']} deg "
            "(phone inclinometer on the plate board or on the camera).",
            f"Distance camera front -> plate centre: {c['distance_mm']} mm (tape, +-10 mm is fine).",
            "Aim the camera at the plate centre. All saddles and markers inside the image.",
            "Even, diffuse light; no glare on the paper. Do not touch camera or plate while recording.",
        ]
        if t == "B1":
            lines.append("All three markers (30, 31, 32) visible. Wait for 'lock locked', then SPACE.")
        else:
            lines += ["Wait for 'lock locked' with all three markers.",
                      "Then cover markers 31 and 32 with white card without touching the plate.",
                      "Check the preview still says 'lock locked' and 'markers 1', then SPACE.",
                      "If the lock drops: uncover, wait for the lock, try again."]
        return lines
    if t == "B2L":
        return [
            first_rep_note,
            "Plate L on the table: flat face H on the table, upright face V facing the camera.",
            f"Camera elevation: {c['elevation_deg']} deg below horizontal (45 = looking along "
            "the bisector of the two faces; inclinometer on the camera).",
            f"Distance camera front -> centre of the fold line: {c['distance_mm']} mm.",
            "Markers 40-43 and all 20 saddles inside the image. Wait for 'lock locked', then SPACE.",
        ]
    if t == "B3":
        p = c["position_mm"]
        lines = [
            "Plate A flat on the table, one edge against a straightedge clamped to the table.",
            f"Camera fixed on the tripod, ~{c['distance_mm']} mm, ~{c['angle_deg']} deg. "
            "DO NOT move the camera during the 4 positions of a repetition.",
        ]
        if p == 0:
            lines += [f"Repetition {rep}, position 0: slide the plate against the end stop "
                      "(a block clamped on the straightedge)."]
        else:
            lines += [f"Position {p} mm: slide the plate along the straightedge so that the gap "
                      f"between end stop and plate is {p} mm (caliper depth rod or gauge blocks).",
                      "Write the caliper reading in the note if it is not exactly the nominal value."]
        lines.append("Wait for 'lock locked', then SPACE.")
        return lines
    return [f"{t}: {c}"]


def plan_index(plan: List[dict]) -> Dict[str, dict]:
    return {s["id"]: s for s in plan}
