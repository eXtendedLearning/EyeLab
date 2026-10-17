# EyeLab validation campaign — step-by-step guide

Operational guide for the baseline campaign. Why the protocol looks like this
and what each metric means: `docs/validation.md`.

Everything runs from one menu: double-click **`validation.bat`** in the repo
root. It sets up the venv, then shows the campaign status and the steps.
Progress is saved after every action in `python/validation_runs/campaign.json`,
so you can close it at any time and continue later (another day too).

```
 Setup                               Measure
  1  Self-test (~15 s)                 6  Next session
  2  Camera, profile, IMU, focus      7  Pick / redo a session
  3  Calibrate camera                Results
  4  Plate measurements                8  Analyse
  5  Visual check in the EyeLab GUI    9  Step-error table (B3)
```

## 0. Before you start

**Where the files are.** The campaign uses the same configuration files as
the EyeLab GUI, all under `python/config/` (`python/eyelab_paths.py`):

| Folder | Content |
|---|---|
| `cameras/<profile>.yaml` | one calibration per camera profile (e.g. `desktop_c920`, `surface_rear`); `cameras/history/` keeps every replaced one |
| `markers/` | marker configs: structures, `plate_A.markers.json`, `plate_A.measured.markers.json`, … |
| `geometry/` | UNV geometries: structures, `plate_A.unv`, `plate_A.measured.unv`, … |
| `targets/` | validation ground truth, nominal and measured |
| `active.json` | which profile / markers / geometry the GUI opens with (local, git-ignored) |

`docs/validation_printables/` holds only the plate PDFs. The ChArUco
calibration board is the one EyeLab already uses (5×7, 25 mm squares, 19 mm
markers, DICT_4X4_50): there is no separate validation board. A calibration
made in the GUI and one made from this menu are the same file.

**Instrument and the 1 mm goal.** Every ground truth here is only as good as
the instrument that checks the prints. The menu asks for its accuracy (±a mm)
and prints the uncertainty budget and the test-uncertainty ratio (TUR) against
a 1 mm goal; TUR ≥ 4 is the usual requirement. With a ±1 mm ruler: plate A
≈ 0.33 mm (TUR 1.5), plate L ≈ 0.5 mm, B3 steps ≈ 0.8 mm (TUR 0.6, skip B3).
With a digital caliper (±0.03 mm): plate A ≈ 0.06 mm (TUR 9), plate L ≈
0.13 mm with the fold angle measured from images (TUR 4). A uniform print scale
error does not matter (it cancels in the on-face metrics); only the X/Y
difference does.

**Equipment**

- the three prints: plate A on a rigid flat board (glass, aluminium, ≥ 5 mm
  foam board); plate L folded on a rigid 90° bracket; the ChArUco board on card;
- the webcam on a tripod. The head must tilt down to at least 65° below
  horizontal; straight down (90°) is needed for the 0° plate-A sessions in
  rig 1 (see below);
- a length instrument for the scale bars: a digital caliper if possible (see
  above); no tape or inclinometer is needed, the preview shows distance and
  angle;
- a metal straightedge (≥ 300 mm), two clamps and a small block as an end
  stop (test B3); gauge blocks if you have them;
- two pieces of white card about 40 × 40 mm (test B1b);
- diffuse light (daylight from a window, not direct sun; or a lamp bounced off
  a wall). No glare on the paper.

**Rules for the whole campaign**

- Use one camera at 1280×720, with zoom and focus never touched. Changing
  focus or zoom changes the calibration (step 2). If you recalibrate, the
  sessions you already have stop being comparable with the new ones: the menu
  tells you how many, and they should be redone (menu 7).
- Close the EyeLab GUI and any other camera application before menu items 2,
  3, 6 and 7: only one program can hold the camera.
- Commit or stash the code before starting. Every session records the git
  commit and whether the tree had local changes.
- `python/validation_runs/` is git-ignored and holds the raw data for the
  thesis. Copy it to OneDrive or another disk at the end of each day.

**Time.** About 1 h for setup (steps 1-4), then roughly 3-4 h of sessions: B1
1.5 h, B1b 20 min, B2-L 1 h, B3 30 min. Two half-days work well, as long as the
camera, focus and calibration stay the same.

## 1. Self-test (menu 1)

Runs the validation test suite (about 15 s). It must end with `OK`. If it
fails, stop and report the output: the analysis is not trustworthy.

## 2. Camera, profile, IMU and focus (menu 2)

1. Enter the camera index. 0 is the first camera; the GUI's camera list shows
   the others.
1. Enter the camera profile name. Each physical camera (and resolution/focus
   setting) gets its own profile, so the desktop webcam and the Surface camera
   never overwrite each other's calibration. The profile becomes the active
   one for the GUI too.
1. IMU: `rear` or `front` only when the camera is built into the device you
   are running on (Surface). On the desktop, or with a USB webcam, `off`.
   Check the signs once with `python surface_imu.py` (device flat, screen up:
   raw z ≈ −1 g, rear axis from down ≈ 0°).
2. In the focus window, hold or place plate A about 600 mm from the camera.
   Press **A** to switch autofocus on and let it settle on the plate. Then
   press **L** to lock the focus there and **ENTER** to accept. The sharpness
   number helps with fine-tuning (**+**/**−**).
3. The menu reports whether the camera honoured the lock. Some webcams ignore
   it. If yours does, switch autofocus off in the camera's own software and
   make sure nothing changes it during the campaign.

From now on, calibration and every capture use this fixed focus value.

## 3. Calibration (menu 3)

1. Use the EyeLab ChArUco board (5×7, 25/19 mm), flat on card or foam board.
   Its absolute print scale does not affect the intrinsics (Zhang 2000: a
   scaled board only changes the board's estimated distance), so no ruler
   check is required; what matters is flatness and equal X and Y scale.
2. Start calibration. It writes `config/cameras/<profile>.yaml`; the previous
   one is backed up first to `config/cameras/history/`.
3. Hold the board in front of the camera and press **SPACE** when the corners
   are drawn. Take 20-25 views covering:
   - the centre, the four corners and the edges of the image;
   - near (~300 mm) and far (~900 mm);
   - tilted 20-45° left, right, up and down.
   Keep the board still and sharp for each view; avoid motion blur. Press
   **ESC** to finish.
4. Target: **RMS < 0.5 px**. If it is higher, repeat with sharper and more
   varied views.

## 4. Plate measurements (menu 4)

Do this for plate A, then for plate L.

1. Enter the instrument's accuracy ±a (digital caliper ≈ 0.02-0.03 mm, steel
   ruler 0.5-1 mm). It is stored for the campaign and also used for B3.
2. Measure the **X** scale bar three times, then the **Y** bar three times.
   Each time, measure from one edge of the first tick to the *same* edge
   of the last tick (left edge to left edge), and re-seat the instrument
   between readings. Nominal: X 200 mm; Y 150 mm on plate A, 75 mm on plate L.
3. Plate L only: the fold angle. **E** (recommended) estimates it from the
   camera: plate L in view at 400-700 mm, SPACE, then move the camera slowly
   around for ~6 s. Both faces' saddles are fitted jointly with the angle as
   an unknown (`validation/fold.py`); expect ~0.1° uncertainty, against ~0.6°
   for a protractor. **T** types in a measured value instead.
4. The menu prints the uncertainty budget (anisotropy, print placement, paper
   flatness, fold angle), the combined u, U (k = 2) and the TUR against 1 mm.
   Accept the suggested value or enter your own.

The menu writes `plate_A.measured.*` / `plate_L.measured.*` into
`config/targets`, `config/markers` and `config/geometry`, and every session
uses those files. A deviation above 1 % triggers a warning: that means the
page was printed fit-to-page. Reprint it.

## 5. Visual check (menu 5, optional but recommended once)

The menu can make the measured plate's marker config and geometry (and the
campaign's camera profile) the GUI's active ones, so the GUI opens with them
loaded. Press Start AR. The yellow wireframe must sit on the saddle centres. Close the GUI to go
back to the menu.

## 6. Sessions (menu 6, repeatedly)

**Next session** walks through the plan in order. Each session shows a setup
card, for example:

```
 Session 4/63: B1_A_d600_a00_r1   [B1: static pose: jitter, detection, registration]
  - Repetition 1.
  - Plate A. Angle between camera axis and plate normal: 0 deg ...
  - Distance camera front -> plate centre: 600 mm ...
```

The flow for each session:

1. Build the setup on the card.
2. Press **ENTER** to open the camera preview.
3. In the preview, adjust the camera with the **setup panel** (top left):
   distance, angle or elevation and centring, each against its target band.
   Green means in band; the title line says `SETUP IN BAND`. With an IMU, an
   extra line shows the gravity-referenced angle (it must agree on a level
   plate) and `MOVING` while the device is not still. Wait until it says
   `lock locked` with the expected number of markers. The yellow circles
   should sit on the saddles.
4. Press **SPACE** to record. Don't touch anything for the 10-20 s it takes.
5. The preview closes and a quick check is printed: raw pose rate, coasted
   rate, markers in the pose, and warnings.
6. Choose **K** (keep), **R** (redo now) or **D** (discard). A redone attempt
   is moved to `sessions/_rejected/`, never deleted.

Redo the session when the quick check warns, when something moved during the
recording, or when the lock was lost.

### The setups

*Distance* is from the camera to the plate centre (plate L: the centre of the
fold line), along the viewing direction; ±10 mm is fine. *Angle* is between
the viewing direction and the perpendicular to the plate. Both are read live
from the plate pose in the preview's setup panel, so no tape or inclinometer
is needed. They are a positioning aid only: the analysis detects the plate on
its own, and the geometry actually achieved is logged per frame and reported
(`achieved_*` columns in `sessions.csv`).

**Rig 1 (recommended): plate flat on the table, camera tilted.**

| angle | camera looks down at |
|---|---|
| 0° | 90° below horizontal (straight down) |
| 30° | 60° below horizontal |
| 50° | 40° below horizontal |

**Rig 2: plate standing, camera level.** Use it if the tripod cannot look
straight down. Prop the plate's board vertically against a heavy block and
keep the camera level; 0° means facing the plate. For 30° and 50°, lean the
board back until the setup panel shows the angle. (The IMU angle is
gravity-referenced and does not apply to this rig.)

Use the same rig for the whole of a test.

**B1 — static pose (27 sessions).** Plate A at 400, 600 and 900 mm, at 0°,
30° and 50°, three repetitions. Repetitions are the outer loop: after the 9
conditions of repetition 1, the menu starts again at 400 mm / 0° for
repetition 2. Rebuild each setup from scratch: lift the plate, put it back,
and re-set the distance and angle. That is what makes the repetitions
independent.

**B1b — single-marker sustain (6 sessions).** Plate A at 600 mm, 0° and 30°.
1. Let the lock engage with all three markers.
2. Lay the white cards over markers 31 and 32 without touching the plate.
3. The preview must still say `lock locked` and `markers 1`; then press
   **SPACE**.
4. If the lock drops, uncover the markers, wait for the lock, and try again.

**B2-L — two-face plate (18 sessions).** Plate L with the flat face H on the
table and the upright face V facing the camera. Camera elevation 25°, 45° and
65° below horizontal (45° looks along the bisector of the two faces), at 500
and 800 mm.

**B3 — step error (12 sessions = 3 repetitions × 4 positions).**
1. Lay plate A flat. Clamp the straightedge to the table along the plate's
   bottom edge, and clamp the end-stop block at its left end.
2. Set the camera at about 600 mm and 30° (rig 1: 60° below horizontal).
   **Do not move it for the 4 positions of a repetition.**
3. Position 0: plate against the stop.
4. Positions 5, 10 and 20 mm: slide the plate along the straightedge until the
   gap between stop and plate equals the step. Set it with the caliper depth
   rod or gauge blocks. **B3 needs ~±0.1 mm or better:** with a ±1 mm ruler the
   step uncertainty is ~0.8 mm and the test cannot resolve a 1 mm goal (the
   menu warns). Skip B3 until a caliper is available.
5. After each session, enter the gap you actually measured.
6. Between repetitions you may move the camera; position 0 is recorded again.

The four tests fill the 63 sessions in the order above. **Pick / redo**
(menu 7) jumps to any session, for example to do all of B3 while the
straightedge is clamped.

## 7. Pilot check (after the first 3 B1 sessions)

Before spending hours on sessions, run **Analyse** (menu 8) on `B1`, then open
`python/validation_runs/reports/B1/sessions.csv`.

| Column | Expected |
|---|---|
| `homography_residual_px_median` | below ~0.1 px. If not, the image is soft (focus, motion, compression) or the plate is not flat. |
| `measured_rate` | close to 1 |
| `err_mm_rmse` | plausible, sub-mm to a few mm |

In each session folder, `summary.json` → `registration.points_detected`
should be close to `points_total` (20 per frame). Missing points at
900 mm / 50° are possible: note it, don't force it.

If any of these look wrong, stop and send me the `reports/B1` folder before
continuing.

## 8. Results (menus 8 and 9)

- **Analyse** writes `python/validation_runs/reports/<test>/`. Each session
  folder holds `summary.json`, `registration.csv`, `per_node.csv` and three
  plots (error histogram, jitter, bias map). Two tables compare sessions:
  `sessions.csv` has one row per session, and `conditions.csv` pools the
  repetitions per condition (mean and SD across sessions; this is the table
  for the thesis). `--every 2` or `3` speeds the analysis up with little loss.
- **Step-error table** prints and writes `reports/B3/steps.csv`, giving the
  error per step as mean and SD over the repetitions.

How to read them (bias vs noise, single-marker vs three, raw vs filtered):
`docs/validation.md`, "Reading the results".

## 9. Ground-truth review (menu 10)

The analysis finds the printed saddles on its own; they are the ground truth.
When that detection is wrong (B1b, where one 20 mm marker seeds the search) or
missing, menu 10 lets you look at each point in a zoomed window and fix it.
Also available as `python -m validation gt-review` (`--help` for flags).

**Sources.** The local campaign (`python/validation_runs/`), or a campaign
recorded on another machine: zip that machine's whole `validation_runs`
folder and copy the `.zip` into **`python/validation_runs/imports/`**. It is
read straight from the zip, no need to extract it. Everything the review
writes goes next to the source:

| Source | Decisions | Re-analysis (entry 5) |
|---|---|---|
| local campaign | `validation_runs/gt_review/` | `validation_runs/reports_reviewed/` |
| `imports/<name>.zip` | `imports/<name>/gt_review/` | `imports/<name>/reports/` |

`session.csv` holds the session and missing-point decisions, `audit.csv` the
random audits, `scan.csv` a cache of the automatic detection. All of
`validation_runs/` is git-ignored.

**Entries.**

1. *Random audit B1* (300 points) and 2. *B1b* (120): a fixed-seed random
   sample. Zero failures in 300 points bound the failure rate below 1 % at 95 %
   (Clopper-Pearson; Hanley & Lippman-Hand, JAMA 249(13), 1983). Where a
   session was reviewed, the audit checks the reviewed truth instead.
3. *Review one session*: fix the first frame; each point is carried to the
   following frames and snapped there, and a frame comes back only when the
   carried point no longer snaps in place. A session is frames x points
   (100 x 20 = 2000 points), so the counter shows points, not images.
4. *Points without a truth*: every point the analysis did not find, in all
   sessions, proposed at its position in the session's other frames.
5. *Re-analyse with the reviewed truth*: the usual reports, with the corrected
   points. Menu 8 also uses the local campaign's review when it exists.
6. *Summary*: failure rate of the automatic truth per test, with its bound.

**Keys.** Click sets the truth (snapped to the X-junction; `s` toggles snap),
arrows nudge 0.25 px (Shift: 1 px), Space confirms and moves on, `x` marks a
point that cannot be judged, `r` returns to the automatic point, Backspace
goes back, Esc returns to the menu. Every decision is saved at once.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Camera ... delivered no frames" | Another program holds the camera (GUI, Teams, browser). Close it, or replug the webcam. |
| "calibration file changed since recorded" | the profile's `config/cameras/<profile>.yaml` was replaced outside the menu (e.g. a GUI calibration of the same profile). Run menu 3 again, or restore the backup from `config/cameras/history/`. |
| Lock never engages | Markers too small or soft at this distance or angle. Check focus and light. At 900 mm / 50° it may simply be the limit: skip with a note (menu 6 → S). |
| B1b: lock drops when the cards go on | The cards moved the plate, or a card covers marker 30. Uncover, re-lock, retry. |
| Quick check: "fewer frames than planned" | Recording was aborted (Q). Redo. |
| Session in the wrong place / wrong setup | Menu 7 → pick it → redo. The old attempt is kept in `_rejected/`. |
