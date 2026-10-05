# EyeLab validation — baseline protocol

How good are EyeLab's overlays today? This protocol measures the webcam
pipeline against printed ground truth: pose jitter, detection, overlay
registration error in pixels and in millimetres on the part, and step error.
The XREAL port reuses the same targets, recorder and analysis once a camera
feed exists (ADR-003).

Code: `python/validation/` (`python -m validation --help`). Printables:
`docs/validation_printables/`. Replaces the first draft that lived in
`docs/validation_tmp/` (review: units in mm against the repo invariant, ground
truth tied to separately glued markers, and a detector seeded by EyeLab's own
projection that reported a 15 mm error as 0.1 mm).

## Principles

1. **One definition of the plate.** A plate layout (`validation/plates.py`)
   generates the PDF, the ground-truth target, the EyeLab marker config and the
   UNV geometry. Markers and reference points are printed together, so their
   relative positions are known from the print, not from caliper work on
   marker centres (which are not physical features) or on marker roll (ADR-002).
2. **Calipers only check the printer.** Each plate carries an X and a Y scale
   bar. Their measured lengths rescale the layout per axis
   (`apply-measurements`).
3. **Ground truth is found independently of EyeLab.** The analysis detects the
   plate's markers itself, fits a face-plane homography, and grows it over the
   saddle points (sub-pixel, X-junction verified). EyeLab's output is only
   compared against it, never used to find it. Synthetic tests check that a
   30 mm error is reported as 30 mm.
4. **Units.** Metres everywhere inside, mm only in `*_mm` fields (AGENTS.md).
5. **The session is the statistical unit.** Frames of one static pose are not
   independent; spread between poses comes from repositioned repetitions.

## What each number means

| Metric | Definition | Source |
|---|---|---|
| `measured_rate`, `coasted_rate` | frames with a raw pose / with a coasted (predicted) pose | `FrameResult.measured_pose`, `pose.coasted` |
| `jitter_measured`, `jitter_pose` | spread of the static pose about its mean: translation (mm), rotation (deg, about the chordal mean). Raw solve vs Kalman output. | recorded poses |
| `err_px` | image distance between EyeLab's projection of a node (float) and the detected printed point | per node-frame |
| `err_px_drawn` | same for the integer pixels `overlay.project_nodes` draws (rounding adds ~0.4 px) | |
| `err_mm` | where EyeLab's projection lands on the face plane vs the true point, via that frame's face homography. In-plane overlay error as the camera sees it: what decides whether the operator hits the right node. Depth error is invisible to it by construction. | |
| `err_mm_raw` | `err_mm` for the raw (unfiltered) pose | |
| `overlay_jitter_mm` | spread of EyeLab's on-face position of each node about its session mean | |
| `per_node.csv`, `bias_map.png` | mean error vector per node: systematic error (calibration, model, scale) vs noise | |
| `homography_residual_px` | fit residual of the detected saddles. QA: must stay far below the errors being measured | |
| `step error` | estimated minus jig-known translation between two static sessions | `step` command |

RMSE is reported with median, p95 and max (THEORY §9.2). Project targets:
registration RMS ≤ 5 mm (static), working aim < 1 mm per node.

## Ground-truth uncertainty

Enter the standard uncertainty of a reference-point position as
`--uncertainty-mm`. It combines the scale-bar reading (caliper uncertainty
divided by the bar length, times the distance of a point from the plate
centre, ≤ ~130 mm), print placement (~0.02 mm at 600 dpi) and flatness of the
mounted paper. For plate A on glass or aluminium, 0.05–0.1 mm is realistic.
For a pass/fail claim at a tolerance T the ground truth should be at least 4×
better (test-uncertainty ratio ≥ 4:1, ANSI/NCSL Z540.3-2006; uncertainty per
JCGM 100:2008, GUM): 0.1 mm supports statements down to ~0.4 mm.

For plate L the per-face ground truth comes from the print alone, so it does
not depend on the fold. The model EyeLab registers to does: enter the measured
interior angle (`--fold-deg`). An error in that angle appears as a real
registration error, as it would on a mis-modelled structure.

## Printables (`docs/validation_printables/`)

| File | Use |
|---|---|
| `plate_A.pdf` | planar plate: 20 saddles (5×4, 40 mm), markers 30/31/32 (20 mm, DICT_4X4_50). Tests B1, B1b, B2, B3. |
| `plate_L.pdf` | folded plate: 2×(5×2) saddles, markers 40/41 on the flat face, 42/43 on the upright one. Non-coplanar markers, grazing face. Test B2-L. |
| `charuco_calibration_A4.pdf` | ChArUco 5×7, 25/19 mm, from `calibrate.py --generate` (A4 portrait). Step 0. |
| `plate_*.target.json` | nominal ground truth, input to `apply-measurements` |
| `plate_*.markers.json`, `plate_*.unv` | nominal marker config and geometry for the GUI (visual check only) |

Regenerate with `python -m validation make-plates --out ../docs/validation_printables`
(`--marker-size-mm 40` for the 512×378 XREAL Eye feed).

Printing: A4 landscape, **100 % / actual size**, no fit or shrink, best quality,
matte paper. Mount plate A flat on glass, aluminium or ≥ 5 mm foam board
(spray adhesive, no bubbles). For plate L, valley-fold along the edge ticks,
printed side inside, and bond each half to a rigid 90° bracket with the fold
on the bracket edge (3D-printed or machined; cardboard only for a trial).

## Procedure

For the campaign itself use `validation.bat` (menu) and the step-by-step guide
`docs/validation-campaign.md`; the commands below are what the menu runs.

Run everything from `python/` in the project venv (the repo's `venv/`, with
`pip install -r python/requirements.txt`; the validation package needs nothing
beyond it).
Session data go under `python/validation_runs/` (gitignored).

**0. Calibration.** Print the ChArUco board, measure its 100 mm ruler, and
calibrate at the resolution you will test (`calibrate.py --live ...`, RMS
target < 0.5 px, THEORY §9.3). Fix focus and zoom; autofocus changes the
intrinsics. Validation results hold only for this calibration.

**1. Plates.** Print, mount, then measure each scale bar with calipers from
one edge of the first tick to the same edge of the last, three times, and
average:

```
python -m validation apply-measurements --target ../docs/validation_printables/plate_A.target.json --bar-x-mm 199.86 --bar-y-mm 149.93 --uncertainty-mm 0.08
python -m validation apply-measurements --target ../docs/validation_printables/plate_L.target.json --bar-x-mm ... --bar-y-mm ... --fold-deg 90.3 --uncertainty-mm 0.1
```

This writes `plate_A.measured.target.json` (+ `.markers.json`, `.unv`). Use
the measured files from here on. One measured file per physical plate; a
reprint is a new plate.

**2. Visual check (optional).** Load `plate_A.measured.markers.json` and
`plate_A.measured.unv` in the GUI: the wireframe must sit on the saddles.

**3. Capture.** Camera on a tripod, plate static, lighting constant, no
autofocus. Each run opens a preview; SPACE starts recording, Q aborts.

```
python -m validation capture --target ../docs/validation_printables/plate_A.measured.target.json --calibration config/camera_params.yaml --out validation_runs/sessions/B1_A_d600_a30_r1 --frames 300 --condition test=B1 plate=A distance_mm=600 angle_deg=30 rep=1
```

The pipeline is EyeLab's (`ArucoPipeline`, same board, pose lock, ROI
re-detection and detector presets as the GUI; `--tuning` to change preset).
It logs every frame's raw pose, drawn pose, lock state and float and drawn
projections, and stores the raw image (greyscale) of every `--save-every`-th
frame (default 3, ~100 images and a few tens of MB per 300-frame session).

| Test | Plate | Conditions | Per session |
|---|---|---|---|
| B1 jitter, detection, registration | A | distance 400 / 600 / 900 mm × angle 0 / 30 / 50° × 3 reps (reposition between reps) | 300 frames |
| B1b single-marker sustain | A | as B1 at 600 mm; acquire the lock with all 3 markers, cover 31 and 32 with white card, then SPACE | 300 frames |
| B2-L two-face registration | L | 500 / 800 mm × view from the face bisector, ±20° × 3 reps | 300 frames |
| B3 step error | A | plate slid along a straightedge by 5 / 10 / 20 mm (gauge blocks or caliper-set stop), camera fixed, 2 static sessions per step, 3 reps | 150 frames |

**4. Analysis.**

```
python -m validation analyze validation_runs/sessions/B1_* --target ../docs/validation_printables/plate_A.measured.target.json --out validation_runs/reports/B1
python -m validation step validation_runs/sessions/B3_s10_r1_a validation_runs/sessions/B3_s10_r1_b --known-mm 10
```

Per session: `summary.json`, `registration.csv`, `per_node.csv`,
`registration_hist.png`, `jitter.png`, `bias_map.png`. Over sessions:
`sessions.csv` and `conditions.csv` (repetitions pooled per condition, mean
and SD across sessions). `--every k` analyses every k-th frame image.

## Reading the results

- **Single-marker sustain is expected to be worse.** One 20 mm marker gives a
  pose with a large lever arm to nodes 100–200 mm away, and a planar target
  has a two-fold pose ambiguity (Schweighofer & Pinz, IEEE TPAMI 28(12),
  2006; IPPE: Collins & Bartoli, IJCV 109(3), 2014). On ideal synthetic frames
  at 0.55 m EyeLab's single-marker error is ~1 mm median against ~0.04 mm with
  three markers (`test_capture_e2e`).
- **Bias vs noise.** A consistent pattern in `bias_map.png` (all arrows one
  way, or growing with distance from the plate centre) points to calibration
  or scale, not to detection noise. A reprojection residual cannot reveal a
  scale error (AGENTS.md invariant): the step test and the plate's absolute
  geometry can.
- **Filter.** `err_mm` vs `err_mm_raw` and `jitter_pose` vs `jitter_measured`
  separate the pose lock's smoothing from the measurement.
- **QA.** If `homography_residual_px` is not well below the `err_px` being
  reported, or many nodes are undetected, fix the image (focus, blur, glare)
  before interpreting anything.

## Not covered yet

- Latency (glass-to-glass) and the display path: `timings_ms` records
  processing time only.
- XREAL: the recorder and analysis are platform-agnostic, but no Eye camera
  feed is available yet (CONTEXT, native route status).
- Structures other than the plates: any planar-faced part can be added as a
  new layout, or as a target JSON with measured faces.

## Self-check

`python -m unittest discover -p "test_*.py"` includes `validation/test_*`:
layout geometry and PDF placement, detection on rendered scenes (plate A to
60°, plate L with a grazing face), recovery of injected errors from 0 to 30 mm,
schema/units, and an end-to-end run of the real `ArucoPipeline` on rendered
frames.
