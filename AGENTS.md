# AGENTS.md — EyeLab

Entry point for any coding agent (Claude Code, Codex, Cursor) working in this repo.
Read this file first and in full. It is short on purpose; everything else is linked, not duplicated.

## What EyeLab is

An XR spatial-localization aid for Experimental Modal Analysis impact-hammer testing. It parses Simcenter Testlab `.unv` geometry, registers it to the physical specimen via ArUco markers, and overlays the measurement wireframe on a live camera feed. Phase 1 (webcam, Python/OpenCV) is feature-complete in `python/`. Phase 2 (Unity + XREAL One Pro + Eye, generic Android APK) is scaffolded in `eyelab_xreal/` and not yet implemented.

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
- **Known doc defect:** `.docs/THEORY.md` §6.3 states the OpenCV→Unity quaternion rule as "flip the signs of y and z". That is wrong and inconsistent with the matrix rule given in the same section. The correct map for `R → F R F⁻¹`, `F = diag(1,−1,1)` is `(w, x, y, z) → (w, −x, y, −z)`. Fix the doc when you first touch this.

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

`eyelab_xreal/` targets Unity 6000.4.5f1, URP, generic Android APK — **not** Beam-Pro-specific. The XREAL SDK lives at `eyelab_xreal/Packages/com.xreal.xr/` and is gitignored (no redistribution licence for modified vendor binaries): install XREAL XR Plugin 3.1.0 locally, then run `tools/patch_xreal_sdk.ps1`. Namespace is `Unity.XR.XREAL`; **no legacy `NR*` names exist in this version** — if a doc uses them, it predates the verified surface in CONTEXT.md.

An agent cannot launch the Editor, enter Play mode, build an APK, or deploy. The loop is: agent writes C# → Jack runs it in Unity → agent reads `Logs/Editor.log` or `adb logcat`.

## Commits

- Style: plain lowercase descriptive messages (`geometry bugfix`), **not** conventional commits.
- Version lives in `python/eyelab_version.py` and nowhere else. Bumping it means updating `CHANGELOG.md` (Keep a Changelog) and tagging. Do not encode a release in a commit message alone.
- Touch `CHANGELOG.md` and `CONTEXT.md`'s `Last updated` stamp with substantive changes.
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
