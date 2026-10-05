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
  2  Camera index and focus            7  Pick / redo a session
  3  Calibrate camera                Results
  4  Plate measurements                8  Analyse
  5  Visual check in the EyeLab GUI    9  Step-error table (B3)
```

## 0. Before you start

**Equipment**

- the three prints: plate A on a rigid flat board (glass, aluminium, ≥ 5 mm
  foam board); plate L folded on a rigid 90° bracket; the ChArUco board on card;
- the webcam on a tripod. The head must tilt down to at least 65° below
  horizontal; straight down (90°) is needed for the 0° plate-A sessions in
  rig 1 (see below);
- digital caliper, tape measure, phone with an inclinometer app;
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

## 2. Camera index and focus (menu 2)

1. Enter the camera index. 0 is the first camera; the GUI's camera list shows
   the others.
2. In the focus window, hold or place plate A about 600 mm from the camera.
   Press **A** to switch autofocus on and let it settle on the plate. Then
   press **L** to lock the focus there and **ENTER** to accept. The sharpness
   number helps with fine-tuning (**+**/**−**).
3. The menu reports whether the camera honoured the lock. Some webcams ignore
   it. If yours does, switch autofocus off in the camera's own software and
   make sure nothing changes it during the campaign.

From now on, calibration and every capture use this fixed focus value.

## 3. Calibration (menu 3)

1. Check the 100 mm ruler on the ChArUco print with the caliper. If it is not
   100 mm (±0.3), the page was scaled: reprint at 100 %.
2. Start calibration. The previous `config/camera_params.yaml` is backed up
   first, to `validation_runs/calibration_backups/`.
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

1. Enter the caliper's accuracy (the default 0.03 mm is right for a typical
   digital caliper).
2. Measure the **X** scale bar three times, then the **Y** bar three times.
   Each time, put the jaws from one edge of the first tick to the *same* edge
   of the last tick (left edge to left edge), and re-seat the caliper between
   readings. Nominal: X 200 mm; Y 150 mm on plate A, 75 mm on plate L.
3. Plate L only: enter the measured interior angle between the two faces
   (engineer's square or inclinometer; 90 if it sits flush on a square).
4. Accept the suggested ground-truth uncertainty (typically ~0.06 mm) or enter
   your own.

The menu writes `plate_A.measured.*` / `plate_L.measured.*` in
`docs/validation_printables/`, and every session uses those files. A
deviation above 1 % triggers a warning: that means the page was printed
fit-to-page. Reprint it.

## 5. Visual check (menu 5, optional but recommended once)

The GUI opens. Load the `plate_A.measured.markers.json` marker config and the
`plate_A.measured.unv` geometry (the menu prints the paths), then press Start
AR. The yellow wireframe must sit on the saddle centres. Close the GUI to go
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
3. In the preview, wait until it says `lock locked` with the expected number
   of markers. The yellow circles should sit on the saddles.
4. Press **SPACE** to record. Don't touch anything for the 10-20 s it takes.
5. The preview closes and a quick check is printed: raw pose rate, coasted
   rate, markers in the pose, and warnings.
6. Choose **K** (keep), **R** (redo now) or **D** (discard). A redone attempt
   is moved to `sessions/_rejected/`, never deleted.

Redo the session when the quick check warns, when something moved during the
recording, or when the lock was lost.

### The setups

*Distance* is from the camera's front glass to the plate centre (plate L: the
centre of the fold line), along the viewing direction, measured with a tape;
±10 mm is fine. *Angle* is between the viewing direction and the
perpendicular to the plate; set it with the phone inclinometer.

**Rig 1 (recommended): plate flat on the table, camera tilted.**

| angle | camera looks down at |
|---|---|
| 0° | 90° below horizontal (straight down) |
| 30° | 60° below horizontal |
| 50° | 40° below horizontal |

**Rig 2: plate standing, camera level.** Use it if the tripod cannot look
straight down. Prop the plate's board vertically against a heavy block and
keep the camera level; 0° means facing the plate. For 30° and 50°, lean the
board back by that angle, measured with the inclinometer on the board.

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
   rod or gauge blocks.
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

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Camera ... delivered no frames" | Another program holds the camera (GUI, Teams, browser). Close it, or replug the webcam. |
| "calibration file changed since recorded" | `config/camera_params.yaml` was replaced outside the menu (e.g. a GUI calibration). Run menu 3 again, or restore the backup from `validation_runs/calibration_backups/`. |
| Lock never engages | Markers too small or soft at this distance or angle. Check focus and light. At 900 mm / 50° it may simply be the limit: skip with a note (menu 6 → S). |
| B1b: lock drops when the cards go on | The cards moved the plate, or a card covers marker 30. Uncover, re-lock, retry. |
| Quick check: "fewer frames than planned" | Recording was aborted (Q). Redo. |
| Session in the wrong place / wrong setup | Menu 7 → pick it → redo. The old attempt is kept in `_rejected/`. |
