# AGENTS.md — EyeLab

Entry point for any coding agent (Claude Code, Codex, Cursor) working in this repo.
Current build: `python python/eyelab_version.py` (the only authority; this file deliberately does not state it).
Read this file first and in full. It is short on purpose; everything else is linked, not duplicated.

## What EyeLab is

An XR spatial-localization aid for Experimental Modal Analysis impact-hammer testing. It parses Simcenter Testlab `.unv` geometry, registers it to the physical specimen via ArUco markers, and overlays the measurement wireframe on a live camera feed. Phase 1 (webcam, Python/OpenCV) is feature-complete in `python/`. Phase 2 (Unity + XREAL One Pro + Eye on an XREAL Beam Pro, Android APK) is scaffolded in `eyelab_xreal/` and not yet implemented. The PC-beta gate project (`unity_projects/xreal_pc_gates/`, Surface) proved the correction chain and is paused (2026-11-10). **New work must also run on One Pro + Eye + Beam Pro;** the hand-over is `docs/BEAMPRO-HANDOVER-2026-11-10.md`.

## Hardware in hand — read before any XREAL work

**`docs/HARDWARE.md`** is the single source for the machines and the glasses, and what each can do. In short: XREAL One Pro + Eye; desktop `lagann-0526` (RTX 5060, no iGPU) has USB data to the glasses but **cannot drive them as a display**; the **Surface Pro 7** drives them as a display (USB-C DP Alt Mode) and has a built-in webcam. A Thunderbolt DP-in card for the desktop was an open purchase decision (ADR-003, open question 6); **decided 2026-10-14: not buying** — the pipeline is made light enough for the Surface and the Beam Pro instead (`docs/HARDWARE.md`). **A Beam Pro arrives around 2026-11-17** and becomes the target host. Update that file whenever hardware or a verified capability changes.

## Document precedence

When two documents disagree, the one higher in this list wins. When a document disagrees with observed runtime behaviour, **the runtime wins** — correct the document in the same change, and say so in the commit message.

1. **This file** — working rules.
2. **`CONTEXT.md`** — verified facts: XREAL SDK 3.1.0 surface, resolved decisions, open-issues board. Check its `Last updated` stamp before trusting the Status section.
3. **`docs/adr/`** — architecture decisions. Append-only: to reverse one, write a new ADR and mark the old `Status: Superseded by ADR-NNNN`. Never edit the prose of an accepted ADR.
4. **`docs/tasks/`** — implementation plans with "Done when" gates.
5. **`.docs/THEORY.md`** — the scientific reference (camera model, ArUco, PnP, Kabsch, SLAM, statistics). Accurate and worth trusting, with one known exception noted below.
6. **`.docs/PROJECT.md`, `.docs/TASKS.md`** — scope and the WP1–WP6 plan. Some hardware and version details have drifted; treat as intent, not fact.
7. **`.docs/tasks/T1.2 … T4.1`** — original per-task specs, 2026-04-02. **Superseded design material.** Read for background only; do not implement from them without checking against 2–4 first.

`.docs/` is gitignored (Siemens collaboration scope, dates, prices). Never move its content into a tracked file, and never link to it from a public doc as if a reader could follow the link.

## Invariants — do not silently break these

- **Units are metres, everywhere, past the parser boundary.** `MarkerCorrespondence.unv_position`, board object points, overlay nodes, every `tvec`. Millimetres appear only in user-facing fields named `*_mm` and are converted at the edge.
- **One physical constant, one definition.** Marker edge size, ChArUco square/marker length, and the camera resolution a calibration was taken at are each defined once. If you need one in a second place, import it.
- **A pure scale error reprojects perfectly.** Reprojection RMS cannot detect a wrong marker size, a wrong calibration resolution, or a wrong unit factor. Never use a reprojection residual as evidence that a scale is right; check an absolute distance instead.
- **`IArucoPoseBridge` (Unity, ADR-001 Phase 2) is load-bearing.** No non-bridge code may depend on the WebSocket implementation directly.
- **Class names in `docs/tasks/` are contracts.** `EwmaStructureAnchor`, `CoordinateConverter`, `IStructureAnchor` and friends are cross-referenced from CONTEXT.md; renaming silently breaks those references.
- **Process each camera frame once.** Frames are `python/frame_packet.py` `FramePacket`s (session_id, strictly increasing sequence, host `received_at`, which is not exposure time). `ArucoPipeline` processes each (session, sequence) at most once, drops packets older than `max_frame_age_s` independently of reader progress, and resets filters on a session change. Never re-run detection on a repeated frame or treat reader progress as proof of freshness.
- **Pose acceptance is phase-dependent.** Acquisition requires validity, the RMS ceiling and an unambiguous solve at every marker count; sustained tracking uses the adaptive RMS gate and time-aware jump limits; recovery goes through COASTING/SEARCHING with a filter reset (`python/pose_lock.py`). Do not reintroduce unconditional acceptance for 3+ markers.
- **Doc defect fixed (2026-10-08):** `.docs/THEORY.md` §6.3 now states the correct quaternion map for `R → F R F⁻¹`, `F = diag(1,−1,1)`: `(w, x, y, z) → (w, −x, y, −z)`. `.docs/tasks/T*.md` guides carry superseded banners; `.docs/PROJECT.md` and `.docs/TASKS.md` are intent, not completion evidence (see `docs/CAPABILITIES.md`).

## Python

- Work in a venv. `python/requirements.txt` pins the floors; `opencv-contrib-python` is held `<5.0` because OpenCV 5 removed `cv2.aruco.calibrateCameraCharuco`, still used by `calibrate.py` and `gui_calibration.py`. Lift the pin only together with the migration to `board.matchImagePoints` + `cv2.calibrateCamera`.
- Tests live beside their modules, not in a `tests/` directory. Run from `python/`:
  ```
  python -m unittest discover -p "test_*.py"
  ```
- CI (`.github/workflows/ci.yml`) runs `ruff check --select E9,F63,F7,F82` plus the suite on Python 3.11. The narrow ruleset is deliberate — real errors only, no style churn. Do not widen it in passing; if you want a new rule, propose it as its own change.
- Target Python 3.9+ in source; CI validates on 3.11.
- Prefer modules over growth in `eyelab_gui.py`. Anything with no tkinter dependency belongs outside it — the Step C service (ADR-001) needs to import that logic headlessly.

## Unity

`eyelab_xreal/` targets Unity 6000.4.5f1, URP, an Android APK. **The XREAL Beam Pro is the reference host (Jack, 2026-11-10; this replaces the 2026-05-08 "generic APK, not Beam-Pro-specific" decision).**

- Beam-Pro-specific settings are allowed when the Beam Pro needs them: manifest entries, permissions, ABI, graphics and performance settings, its input (touchscreen, controller mode).
- Record each one, with its reason, in CONTEXT.md.
- Other Android hosts are not a goal; do not break them without a reason.

The XREAL SDK lives at `eyelab_xreal/Packages/com.xreal.xr/` and is gitignored (no redistribution licence for modified vendor binaries): install XREAL XR Plugin 3.1.0 locally, then run `tools/patch_xreal_sdk.ps1`. Namespace is `Unity.XR.XREAL`; **no legacy `NR*` names exist in this version** — if a doc uses them, it predates the verified surface in CONTEXT.md.

**What a local agent may do (2026-11-10, Jack)**, for `eyelab_xreal/`, the gate project and any Unity project under `unity_projects/`:

- **Unity CLI in batch mode:** open or create the project, compile, run EditMode tests, build Windows players and **Android APKs**.
- **Sign** with the Unity debug key or a local keystore outside the repo. Never commit keystores or passwords. APKs are build artefacts: gitignored, under the project's `Builds/` or `build/`.
- **Deploy to the Beam Pro over `adb`**, once Jack has connected it and it shows in `adb devices`:
  - `adb install -r` of an EyeLab APK;
  - start and stop the EyeLab app (`am start`, `am force-stop`);
  - `pm grant` of permissions the EyeLab manifest declares;
  - `adb logcat`;
  - `adb pull` and `adb push` inside the EyeLab app's own storage (run logs out; config and calibration files in).
- **The loop:** agent builds and installs → Jack runs the protocol wearing the glasses → agent pulls the logs and analyses them.

**Still Jack's:**

- Editor Play mode (needs his go-ahead); entering XR on the desktop (forbidden).
- Wearing the glasses and running a protocol.
- Anything on the device outside the EyeLab app: enabling USB debugging or pairing, `adb root`/remount, system settings, uninstalling or touching other apps, factory reset.
- Firmware or OS updates of the glasses or the Beam Pro.
- USB-configuration changes on the glasses.
- Deciding to change the ADR-001 bridge (Step C → A → B).

*History:* 2026-10-09 to 2026-11-10 the Unity CLI was allowed only for the PC gate project (`docs/XREAL-PC-GATES-G1-G3.md` §1), and an agent could not build an APK or deploy. The rules above replace that.

## Commits

- Style: plain lowercase descriptive messages (`geometry bugfix`), **not** conventional commits.
- **Every commit subject starts with the build it bumped to:** `0.2.014 geometry bugfix` (then plain lowercase text). Scheme `MAJOR.MINOR.BUILD`; BUILD is +1 per commit, reset on a MINOR change. The number lives in `python/eyelab_version.py` and **nowhere else as an original** — never hand-edit it or copy it into a doc or launcher; run `python tools/bump_version.py` (it stamps README, CHANGELOG and the Unity `bundleVersion`; `check` verifies). Full policy, agent hand-off rules and known limits: **`docs/VERSIONING.md`**.
- Enable the commit guard once per clone: `git config core.hooksPath tools/githooks`.
- Releases are tagged only at a MINOR cut (`v0.3.000`). Bumping MINOR means a `CHANGELOG.md` heading and a tag; do not encode a release in a commit message alone.
- Touch `CHANGELOG.md` and `CONTEXT.md`'s `Last updated` stamp with substantive changes; the stamp reads `Last updated: DATE, build X.Y.ZZZ (...)`.
- Line endings are normalised by `.gitattributes` (`* text=auto`), for a Windows checkout with Linux agents. If EOL noise reappears, renormalise — never commit CRLF.

## Sandbox hazards (agents running against a mounted copy of this repo)

These have cost real work before. Read them before writing anything.

- **The mount can serve stale, tail-truncated reads of files edited by the current session.** Truncation occurs at the file's pre-edit byte size: new content up to that offset is visible, the tail is not. Fresh `Write`s sync correctly; `Edit`s may not. Verify line counts against the Windows-side file before trusting a large file you just edited. `eyelab_gui.py` has lost its `main()` this way.
- **Never run two agent sessions against this repo at once.**
- **Do not commit from the sandbox.** `git status` / `git diff` can report clean while the worktree genuinely differs from HEAD, and `.git/index` can corrupt (`bad signature 0x00000000`). Hand changes to Jack, who commits and pushes from Windows. There are no push credentials in the sandbox.
- `tkinter` is usually absent, so `test_gui_calibration` fails locally and passes in CI. That single failure is expected; anything else is not.
- Purge `python/__pycache__` before running tests against edited sources, or run with `python3 -B` from a copy.

## Open work

`CONTEXT.md` holds the live open-issues board. The two largest known items are the T2.6 XREAL bring-up (`docs/tasks/T2.6-implementation-checklist.md` — treat its code stubs as intent, re-verify the SDK surface before copying) and the findings in the 2026-09-18 audit, which are not yet applied.
