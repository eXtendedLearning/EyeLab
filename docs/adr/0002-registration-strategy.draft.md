# ADR-002 — Registration strategy: board solve for pose, Kabsch for bootstrap and QA

- **Status:** Proposed — awaiting decision
- **Date:** 2026-09-18
- **Deciders:** Giacomo (Jack)
- **Relates to:** finding S5 in `docs/AUDIT-2026-09-18.md`

---

## Context

EyeLab computes the same quantity — the rigid transform from the UNV/structure frame to the camera frame — by two independent routes, and has done since the board path was added.

**Route A, "legacy centre registration" (`registration.SpatialRegistration`).** Each detected marker is solved independently with `solvePnP` to get its centre in the camera frame. Those centres are matched to the marker positions in the UNV frame, and Kabsch/Procrustes finds the rigid transform that best maps one point set onto the other. Three or more non-collinear markers are needed.

**Route B, "board solve" (`LStructureDetector.estimate_pose`).** `board_from_correspondences()` builds an OpenCV `Board` whose object points are the marker *corners expressed in the UNV frame*. One `solvePnP` over all visible corners at once returns the transform directly. The board pose **is** the registration.

Today route B drives the overlay whenever a structure board is configured, and route A runs every frame anyway and is discarded (`_draw_registered_wireframe` takes the board branch and returns first). Route A is additionally broken for flat specimens by the conditioning-gate defect (finding S2), so on a plate it returns `None` regardless.

The question this ADR settles is not "which is faster" — both are far below the frame budget (Kabsch is 0.069 ms, the per-marker solves 0.06 ms for six markers, against 3–12 ms for detection). It is whether route A does anything route B cannot, and therefore whether to repair it or delete it.

## What the measurements say

All figures from 4–5 markers on a ~80–100 mm layout at 0.5 m, 0.2 px corner noise, 120–300 trials. Reproduce with `docs/audit_verify.py` and the probes in the audit.

**1. For the live pose, route B is roughly 10× more accurate.**

| | RMS node error |
|---|---|
| Kabsch on per-marker centres | 3.83 mm (p95 6.49) |
| Joint board solve | 0.39 mm (p95 0.70) |

Same pixels, same noise. Two reasons. Route A is a two-stage estimator — pixels → per-marker pose → Kabsch — and each stage discards information; collapsing a marker to its centre throws away the constraint that its four corners are rigidly related to every other marker's corners. And Kabsch minimises an *unweighted isotropic 3D* cost, while the noise on a per-marker centre is strongly anisotropic: depth uncertainty for a small marker is far larger than lateral uncertainty, so treating a 4 mm depth error and a 0.4 mm lateral error as equally important is the wrong metric. Route B minimises reprojection error in the image plane, which is where the noise actually lives.

**2. For diagnosing a mis-measured marker, route B is also better.**

Deliberately mis-measuring one marker by 5 mm in the model, then asking each method which marker is at fault:

| | culprit | others | contrast | names the right marker |
|---|---|---|---|---|
| Kabsch residual (mm) | 3.89 | 2.10 | 1.9× | 80 % of frames |
| Board per-marker reprojection, converted to mm | 2.60 | 1.08 | 2.4× | 100 % of frames |

So the intuition that "Kabsch gives you physically meaningful millimetre residuals and the board solve only gives pixels" does not survive: per-marker reprojection residuals convert to millimetres at the board's depth, and separate the culprit more cleanly.

**3. But route A is blind to marker roll, and route B is not.**

This is the one real asymmetry. Route B needs marker *corners*, so it needs each marker's surface normal **and** its roll angle about that normal. Route A needs only marker *centres*, which do not depend on roll at all. Today the model defaults roll to 0°, and the operator is expected to fill it in.

Model assumes roll = 0 while markers are physically applied at some roll error:

| roll error | joint board solve | Kabsch on centres |
|---|---|---|
| 0° | 0.43 mm | 2.65 mm |
| 15° | 1.24 mm | 2.79 mm |
| 30° | 2.60 mm | 2.96 mm |
| 45° | 4.03 mm | 2.68 mm |
| 90° | 10.36 mm | 2.85 mm |

Kabsch is flat; the board solve degrades and crosses over at roughly 35°. Nobody applies adhesive markers to a flange at a measured roll angle, so in practice the board path has been running with an unquantified roll error this whole time.

**4. The roll does not have to be measured — it can be recovered.**

Each detected marker's own `solvePnP` already returns its orientation in the camera frame (`DetectedMarker.rvec`, computed every frame). Given any rough structure→camera transform, that orientation can be expressed in the structure frame, which yields the marker's roll directly. Kabsch supplies exactly such a rough transform without needing roll — so route A bootstraps route B:

| markers applied at arbitrary roll, operator measured none | RMS node error |
|---|---|
| Board solve with model roll = 0 (today) | 9.85 mm |
| Kabsch on centres (roll-free, but has a noise floor) | 2.75 mm |
| **Kabsch bootstrap → recover roll → board solve** | **0.44 mm** |

Roll was recovered to within 0.59° with no operator measurement. The combined path is 6× better than Kabsch alone and 22× better than the current board path, and it reaches essentially the same accuracy as a board whose roll was measured perfectly (0.43 mm).

## Decision

**Keep both solvers, and reassign their roles.**

1. **The board solve is the only path to the live pose.** Remove the per-frame Kabsch call from `_ar_loop` and the "legacy centre registration" branch from `_draw_registered_wireframe`. There is no operating point at which route A produces a better overlay than a roll-corrected route B.

2. **Kabsch becomes the bootstrap, not a competitor.** Add a short "learn board" pass, run once when a structure's markers are first placed: collect marker observations over a few seconds and several viewpoints, use Kabsch on the centres to get a roll-free initial transform, recover each marker's roll (and, where it is not obvious, its face normal), and write the results back into that structure's marker config. From then on the board solve runs with a correct model.

3. **Kabsch also stays as marker-placement QA**, reported in millimetres alongside the board's per-marker reprojection residuals. Its residuals are the *weaker* discriminator, but it is a genuinely independent estimate that does not assume the board model is right — which is exactly the assumption under test when the operator is checking their own measurements. Two independent numbers disagreeing is a stronger signal than one number being large.

4. **Repair Kabsch properly** now that it has a real job: fix the conditioning gate to a rank-2 collinearity test (S2, so that flat specimens work at all), implement the distance-weighted variant already specified in `.docs/THEORY.md` §7.4, and add outlier rejection so one bad correspondence cannot corrupt the fit.

## Consequences

**Good.** The overlay gets roughly an order of magnitude more accurate on any structure whose markers were not applied at a measured roll — that is, all of them. The operator's job gets *smaller*, not larger: they place markers and assign them to UNV nodes, and the system works out the rest. One code path owns the live pose, so there is no longer a silently-dead second path to mislead a reader. `SpatialRegistration` keeps a defensible purpose, so repairing it is worth the effort.

**Costs.** The "learn board" pass is new work — a UI affordance, a multi-view collection loop, and persistence of the recovered roll/normal into the marker config (the config format already carries `rollDeg` and `normal` per marker, so no schema change). Until it exists, the board path keeps its current roll sensitivity; an interim mitigation is to surface the estimated roll error as a diagnostic so the operator at least knows.

**Risks.** Roll recovery leans on per-marker orientation from a single small marker, which is the noisiest quantity in the system; it must be averaged over many frames and viewpoints, and markers seen at a very oblique angle should be excluded. The bootstrap also inherits the planar two-fold ambiguity, so the "learn board" pass should require genuinely varied viewpoints and reject a solution whose recovered rolls are not consistent across frames.

**Effect on the XREAL port.** This keeps the pose contract unchanged — one `(rvec, tvec)` per frame over the ADR-001 Step C WebSocket — and moves the board-model refinement entirely to the Python service side, where it can be done once per structure and cached. Nothing here constrains the Unity side.

**Longer term.** The principled endpoint is a small bundle adjustment: refine the marker positions *and* rolls jointly across many views, rather than trusting the operator's caliper measurements at all. The bootstrap above is the first step toward that and would slot in as its initialisation. Worth revisiting once marker-placement error is measured rather than assumed — it is currently the dominant error source in the whole system, and nothing in the repo quantifies it.
