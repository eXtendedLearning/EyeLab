# XREAL PC beta — gate experiment G1–G3 (build on desktop, run on Surface)

> Created 2026-10-09. Executable guide for **Claude Code on the desktop
> (`lagann-0526`) with the Unity CLI** and for **Jack on the Surface Pro 7**.
> It turns the bounded procedure in
> [`XREAL-PC-BETA-2026-10-09.md`](XREAL-PC-BETA-2026-10-09.md#next-experiment-isolated-unity-camera-sample)
> into scripted, logged, pass/fail gates. Background and ADR reconciliation:
> [`AUDIT-2026-10-08.md`](AUDIT-2026-10-08.md#xreal-reassessment---9-october-2026).
> Record every outcome in [`CAPABILITIES.md`](CAPABILITIES.md).

## 0. Purpose and scope

The three gates decide whether the XREAL PC beta (SDK 3.1.2 preview) can be the
host runtime for EyeLab on Windows:

| Gate | Question | Why it matters |
|---|---|---|
| **G1** | Does the Eye deliver live RGB frames to a Windows Unity player? | Without Eye frames there is no ArUco registration from the wearer's viewpoint (ADR-003). |
| **G2** | Does the player get a valid, changing 6DoF head pose? | Without 6DoF the overlay cannot stay on the specimen when markers leave view; September's native probes stalled in `Initializing`. |
| **G3** | Do G1 and G2 work **at the same time**? | Both depend on the same Eye sensor. Registration needs a frame *and* the head pose at its capture time. |

**These are functional gates, not accuracy measurements.** Passing them says
nothing about the 1 mm target. Out of scope here (G4 and later): camera
intrinsics/distortion validation, camera-to-head extrinsic, capture-time pose
interpolation, per-eye projection, the `IArucoPoseBridge` Windows client, UNV
wireframe rendering, plate A/L registration.

The gate project is **not** EyeLab. It is a separate, throw-away-able Unity
project. `eyelab_xreal/` (SDK 3.1.0, Android) is not touched.

## 1. Roles, permissions and hard limits

| Actor | Machine | Allowed |
|---|---|---|
| Claude Code + Unity CLI | Desktop | Create the gate project, edit its manifest/settings/scripts, run the Editor in batch mode, compile, build Windows players, write the analysis scripts, analyse returned logs, update docs. |
| Claude Code | Desktop | **Optional D7 only, with Jack's explicit go-ahead each time:** Play-mode smoke test of G1 (camera only) with the glasses on desktop USB. |
| Jack | Surface | Copy the build, run the gates, perform the motion protocol, return the logs. |

This is a scoped exception to the AGENTS.md rule that agents do not launch the
Editor or build (see AGENTS.md, *Unity*). It covers the gate project only.

**Never, on either machine:**

1. Accept, start or script a firmware update. Do not call `NativeOTA` update
   methods. Record the firmware string; change nothing.
2. Change the glasses' USB configuration or use the unofficial NCM/UVC routes
   (on hold since 2026-10-06, see `claude/xreal-access-routes-2026-10-06.md` in the
   project space).
3. Enter XR on the desktop. `HelloMR.StartXRAsync` changes EDID
   (`NR_EDID_3840_1080_60`), extends monitors and changes the primary monitor;
   the desktop cannot display on the glasses anyway (`HARDWARE.md`).
4. Modify files inside the vendor packages or commit vendor binaries, imported
   samples, builds or run logs. Commits follow AGENTS.md (*Commits*, *Sandbox
   hazards*).
5. Touch `eyelab_xreal/`.

**Two-strike rule.** If the same step fails twice with no new information in
the logs, stop, write down the stage and log excerpt, and ask Jack.

## 2. Facts this guide relies on (verified 2026-10-09 by source inspection)

Paths are relative to `vendor/xreal/pc-beta-3.1.2/<package>/package/`.
Re-verify before relying on a line number; the API names are what matter.

| Fact | Source |
|---|---|
| PC beta needs Unity 2022.3 or Unity 6, `com.xreal.xr` 3.1.2, `com.xreal.xr.experimental` 3.1.2, `com.cysharp.unitask`; enable the XREAL loader for Standalone with *Initialize XR on Startup* **off**; import only the Standalone samples. Play mode renders to the Game view, not the glasses. "Hand tracking and AR features are not supported on PC yet." 6DoF on One series requires the Eye. | [XREAL PC SDK (Beta) guide](https://github.com/dengxian-xreal/xreal.github.io/blob/main/docs/14_PC%20SDK%20%28Beta%29.md) |
| Plane, image-tracking and anchor subsystems compile only under `UNITY_ANDROID && XR_ARFOUNDATION`. Do not install AR Foundation in the gate project. | core `Runtime/Scripts/Android/AR Features/XREAL{Plane,ImageTracking,Anchor}Subsystem.cs`, line 1 |
| `XREALSessionManager` exists only under `UNITY_STANDALONE && UNITASK`; `UNITASK` is defined by the experimental asmdef when `com.cysharp.unitask` resolves. | experimental `Runtime/Standalone/XREALSessionManager.cs` line 1; `Runtime/Unity.XR.XREAL.Experimental.asmdef` `versionDefines` |
| Session start for One series: TCP preflight to `169.254.2.1:52999` (10 tries, 1 s), `XREALPlugin.CreateSession(DirectPresent)`, then poll `GetGlassesStartupState` until `NR_GLASSES_STARTUP_STATE_END`. Power-save and disconnect callbacks tear the session down. | `XREALSessionManager.cs` `CreateSession`, `Start`, `Disconnect` |
| `RGBCamera.unity` contains the `XRRigSetup` prefab (session manager, left/right/centre cameras), the `HelloMR` prefab (Enter XR flow, tracking buttons) and an `XREALRGBCameraTexture` component. | experimental `Samples~/Standalone/RGBCamera/RGBCamera.unity` (`m_SourcePrefab` GUIDs `b3af8874…`, `b62fa05f…`) |
| Camera API: `XREALRGBCameraTexture.StartCapture()` (true iff `StartRGBCameraDataCapture() != ulong.MaxValue`), `StopCapture()`, `IsCapturing`, event `OnRGBCameraUpdate`, `GetYUVFormatTextures()` (Y/U/V `Alpha8`, readable), `GetResolution()`, `GetTimeStamp()` (raw `ulong`, unit undocumented). Frames are polled once per Unity `Update`. | core `Runtime/Scripts/Android/Camera Features/XREALRGBCameraTexture.cs`; `Runtime/Scripts/XREALRGBCamera.cs` (compiled for `UNITY_STANDALONE`) |
| Tracking: `XREALPlugin.SwitchTrackingTypeAsync(TrackingType.MODE_6DOF)` returns `Task<bool>`; `XREALPlugin.GetTrackingType()`. `GetTrackingReason()` is `internal`, not callable from user code. | core `Runtime/Scripts/XREALPlugin.cs`; `Android/AR Features/XREALSessionSubsystem.cs` |
| Vendor calibration claims available at runtime: `XREALPlugin.GetCameraIntrinsic(XREAL_COMPONENT_RGB_CAMERA, ref focal, ref principal)`, `GetDevicePoseFromHead(XREAL_COMPONENT_RGB_CAMERA, ref pose)`, `GetDeviceResolution(...)`. | core `Runtime/Scripts/XREALPlugin.cs` (`XREALComponent` enum) |
| Version strings: `XREALExperimentalPlugin.GetGlassesSystemVersion()`, `GetGlassesSWVersion()`, `GetGlassesHWVersion()`, `GetGlassesDspVersion()`. | experimental `Runtime/XREALExperimentalPlugin.cs` |
| Project validator rule for Windows: graphics API must be **Direct3D11** only; *Initialize XR on Startup* must be off. | core `Editor/XREALProjectValidator.cs` |
| Windows native set to expect in a build: `XREALXRPlugin.dll`, `libnr_api.dll`, `libnr_loader.dll`, `libnr_glasses_api.dll`, `libnr_plugin_6dof.dll`, `ov_utils.dll`, `avcodec-60.dll`, `avformat-60.dll`, `avutil-58.dll`, `swresample-4.dll`, `swscale-7.dll`. | experimental `Runtime/Plugins/X64/` |
| Standalone samples have no asmdef → they compile into `Assembly-CSharp`. Harness scripts that call `HelloMR` must therefore also live in `Assembly-CSharp` (no asmdef). | `Samples~/Standalone/` listing |

## 3. Layout

```
EyeLab/
├── vendor/xreal/pc-beta-3.1.2/              # gitignored; tarballs + extracted packages
├── unity_projects/xreal_pc_gates/            # NEW gate project (Unity 6000.6.5f1, Built-in RP; see §11)
│   ├── Packages/manifest.json                # tracked
│   ├── ProjectSettings/                      # tracked
│   ├── Assets/EyeLabGates/                   # tracked: harness + editor scripts + scene
│   ├── Assets/Samples/…/Standalone/          # gitignored: vendor sample copy
│   ├── Builds/                               # gitignored (**/[Bb]uilds/)
│   └── GateRuns/                             # gitignored: logs returned from the Surface
└── python/xreal_gates/                       # NEW analysis package (tracked)
```

Add to `.gitignore` (the `**/Library/`, `**/[Bb]uilds/`, `**/Logs/` patterns
already cover the rest):

```
unity_projects/xreal_pc_gates/Assets/Samples/
unity_projects/xreal_pc_gates/Assets/Samples.meta
unity_projects/xreal_pc_gates/GateRuns/
```

**Built-in render pipeline, deliberately.** The beta guide treats URP as an
extra configuration step (materials, `XREALEmbeddingRenderFeature` last in the
renderer). The gates test runtime access, not rendering; one fewer variable.
URP is re-checked when EyeLab rendering is built (G4+).

## 4. Phase D — desktop (Claude Code + Unity CLI)

Absolute paths: repo `C:\GitHub_Uni\EyeLab`, project
`C:\GitHub_Uni\EyeLab\unity_projects\xreal_pc_gates`.

### D0 — Preflight

```
unity version
unity skill show                 # read once: authoritative agent usage of this CLI
unity doctor
unity editors --json             # planned 6000.4.5f1; built with 6000.6.5f1 (§11)
unity status                     # no Editor may have the gate project open during batch runs
git --version                    # UniTask resolves from a git URL
```

- Confirm the Editor install has **Windows Build Support (Mono)**
  (`unity modules --help` / `unity install-modules --help` to list). Normally
  bundled with the Windows Editor.
- Verify the tarball SHA-256 against `XREAL-PC-BETA-2026-10-09.md`
  (PowerShell `Get-FileHash -Algorithm SHA256`). Mismatch → stop.
- Discover exact flags with `unity <command> --help` before first use
  (`build`, `run`, `open`, `projects`, `recompile`). Where this guide gives a raw
  Editor command line, it is the fallback if the CLI does not expose the option.
  The Editor executable path comes from `unity editors --json`.

### D1 — Create the project

Prefer the CLI's project creation (`unity projects --help`, `unity templates
--help`). Fallback:

```
"<Editor>\Unity.exe" -batchmode -quit -createProject "C:\GitHub_Uni\EyeLab\unity_projects\xreal_pc_gates" -logFile "C:\GitHub_Uni\EyeLab\unity_projects\xreal_pc_gates.create.log"
```

Then apply the `.gitignore` lines from §3.

**Done when** the folder has `Assets/`, `Packages/manifest.json`,
`ProjectSettings/ProjectVersion.txt` = the Editor used (`6000.6.5f1`, §11), and the log has no errors.

### D2 — Packages

Edit `Packages/manifest.json` dependencies (paths are resolved relative to the
`Packages/` folder):

```json
"com.xreal.xr": "file:../../../vendor/xreal/pc-beta-3.1.2/com.xreal.xr.tar.gz",
"com.xreal.xr.experimental": "file:../../../vendor/xreal/pc-beta-3.1.2/com.xreal.xr.experimental.tar.gz",
"com.cysharp.unitask": "https://github.com/Cysharp/UniTask.git?path=src/UniTask/Assets/Plugins/UniTask",
"com.unity.ugui": "<keep or add the version Unity 6000.4 ships>"
```

`com.unity.xr.management` (≥ 4.4.1) and `com.unity.xr.core-utils` come in as
dependencies of `com.xreal.xr`. **Do not add AR Foundation, XR Hands or XR
Interaction Toolkit.**

**Done when** a batch open (`unity run` or `-batchmode -quit -projectPath …`)
resolves all packages: `Packages/packages-lock.json` lists `com.xreal.xr` 3.1.2,
`com.xreal.xr.experimental` 3.1.2 and `com.cysharp.unitask`, and the log has no
package-resolution errors.

### D3 — Import the Standalone sample

Copy, **with `.meta` files** (they carry the GUIDs the scene references):

```
vendor/xreal/pc-beta-3.1.2/com.xreal.xr.experimental/package/Samples~/Standalone
  → unity_projects/xreal_pc_gates/Assets/Samples/XREAL XR Plugin Experimental/3.1.2/Standalone
```

This is what Package Manager's *Import* button does. Import no other sample.

**Done when** `Assets/Samples/…/Standalone/{HelloMR,Prefabs,RGBCamera}` exist
with their `.meta` files.

### D4 — Project configuration (editor script, batch mode)

Write `Assets/EyeLabGates/Editor/ProjectSetup.cs` with a static
`EyeLabGates.EditorTools.ProjectSetup.Apply()` and run it with
`-executeMethod EyeLabGates.EditorTools.ProjectSetup.Apply -quit`. It must be
idempotent and log every value it sets. It sets:

| Setting | Value | Reason |
|---|---|---|
| Active build target | `StandaloneWindows64` | Surface target. |
| XR Plug-in Management, Standalone | loader `Unity.XR.XREAL.XREALXRLoader` assigned; `InitManagerOnStart = false` | Beta guide + validator. The session manager starts XR on demand. |
| Graphics APIs (Win64) | `UseDefaultGraphicsAPIs = false`, `[Direct3D11]` | Validator rule. |
| Scripting backend | Mono, x86_64 | Faster builds; IL2CPP is not required for the gate. |
| Active Input Handling | **Both** | Sample UI uses the legacy `StandaloneInputModule`; harness uses legacy `Input`. Set via `SerializedObject` on `ProjectSettings/ProjectSettings.asset`, property `activeInputHandler = 2`. |
| Company / Product | `EyeLab` / `XrealPcGates` | Fixes `Application.persistentDataPath` = `%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates`. |
| Run in background, visible in background | true | Focus moves when XR changes monitors. |
| Fullscreen mode | Windowed, 1280×720, resizable | Matches `HelloMR.Start`. |

XR Management API to use (verify against
`Library/PackageCache/com.unity.xr.management@*/Editor/` before writing):
`EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey, out XRGeneralSettingsPerBuildTarget)`
(create the asset under `Assets/XR/` and `EditorBuildSettings.AddConfigObject`
if absent), the per-target `XRGeneralSettings` for `BuildTargetGroup.Standalone`,
`InitManagerOnStart = false`, and
`XRPackageMetadataStore.AssignLoader(settings.AssignedSettings, "Unity.XR.XREAL.XREALXRLoader", BuildTargetGroup.Standalone)`.

**Fallback (one manual action):** if the loader cannot be assigned from script
after two attempts, Jack opens the project once and ticks *Project Settings ›
XR Plug-in Management › Standalone › XREAL*, leaving *Initialize XR on Startup*
unchecked. Then continue in batch mode.

**Done when** a batch run logs the loader as assigned for Standalone,
`InitManagerOnStart=false`, graphics API `Direct3D11`, and the XREAL project
validator reports no failing Standalone rule (read `XREALProjectValidator.cs`
to see which rules are active when the loader is active).

### D5 — Harness and scenes

Implement §5. Then an editor script
`EyeLabGates.EditorTools.GateSceneBuilder.Build()` that:

1. Copies the sample scene to `Assets/EyeLabGates/Scenes/GateHarness.unity`
   (`AssetDatabase.CopyAsset`), opens it, adds one `GateHarness` GameObject with
   the runtime components from §5, wires the `XREALRGBCameraTexture` and
   `HelloMR` references found in the scene, saves.
2. Leaves the original `RGBCamera.unity` untouched; it is the **vendor control**
   build.

### D6 — Compile check

Batch open with `-quit` (or `unity recompile` against a running Editor) and
scan the log for `error CS`. **Done when** zero compile errors; list any XREAL
warnings in the D-phase report.

### D7 — Optional desktop Play smoke test (Jack's go-ahead required)

Purpose: cheap early signal on G1 only. Glasses on desktop USB, Nebula and any
XREAL app closed. Open the Editor, enter Play in `GateHarness`, start camera
capture with the harness key `C` (or gate `G1` default in the Editor). **Never
press X / Enter XR on the desktop.** Record the outcome as host = desktop. A
desktop failure is not a Windows failure (the DP path is absent); a desktop
success does not replace the Surface gate.

### D8 — Build

`EyeLabGates.EditorTools.BuildGates.BuildAll()` (run with `unity build` if it
supports a custom method/scene list, else `-executeMethod … -quit`) produces:

```
Builds/Win64/GateHarness/XrealPcGates.exe          # scene: GateHarness.unity
Builds/Win64/VendorRGBCamera/VendorRGBCamera.exe   # scene: unmodified RGBCamera.unity (control)
```

After building, the script (or Claude Code) must:

1. Verify each `*_Data/Plugins/x86_64/` contains all eleven DLLs listed in §2.
   Any missing → stop; inspect the plugin importer platform settings.
2. Write `Builds/Win64/BUILD_INFO.txt`: UTC time, `git rev-parse HEAD`, dirty
   flag, Editor version, package versions from `packages-lock.json`, tarball
   SHA-256.
3. Copy `run_gates.bat` (§6) into `Builds/Win64/`.
4. Zip `Builds/Win64/` to `Builds/XrealPcGates_<yyyymmdd_HHmm>.zip`.

**Done when** both executables exist, the DLL check passes and the zip exists.
Tell Jack the zip path.

## 5. Harness specification

All runtime scripts in `Assets/EyeLabGates/Runtime/`, **no asmdef** (see §2).
Namespace `EyeLabGates`. One responsibility per file. Invariant culture for all
number formatting. Use `Time.realtimeSinceStartupAsDouble` as `t_unity` and
`Time.frameCount` as `frame` in every CSV.

| File | Responsibility |
|---|---|
| `GateArgs.cs` | Parse command line: `-gate G1\|G2\|G3\|manual` (Editor default `manual`), `-order xrfirst\|camfirst` (G3, default `xrfirst`), `-duration <s>` (defaults G1 60, G2 180, G3 180), `-snapshotEvery <s>` (default 2), `-label <text>`, `-logRoot <dir>` (default `persistentDataPath/GateRuns`). |
| `GateRunLog.cs` | Create `GateRuns/<yyyymmdd_HHmmss>_<gate>_<label>/`; write `run.json`, `camera.csv`, `pose.csv`, `events.csv`, `snapshots/`. Buffered, flushed every 1 s and on quit/`OnApplicationQuit`. |
| `DeviceInfoProbe.cs` | After glasses connect: write to `run.json` the device type, the four version strings, `GetCameraIntrinsic`, `GetDevicePoseFromHead` and `GetDeviceResolution` for `XREAL_COMPONENT_RGB_CAMERA`, `GetDevicePoseFromHead` for both displays, `Screen` info, monitor count, plus args and build info. Store each call's bool return next to its value. These are vendor claims, recorded for G4, not trusted. |
| `CameraProbe.cs` | Subscribe to `XREALRGBCameraTexture.OnRGBCameraUpdate`. Per event append to `camera.csv`: `t_unity, frame, cam_ts, res_w, res_h, y_mean, y_mad_prev, dup_ts`. `y_mean` and `y_mad_prev` (mean absolute difference to the previous frame) on a 64×48 subsample of the Y plane read with `GetRawTextureData<byte>()` (texture is `Alpha8`, readable). Every `snapshotEvery` s save the full Y plane as binary **PGM** (`P5`) in `snapshots/<cam_ts>.pgm` (lossless, no PNG/Alpha8 conversion issues). |
| `PoseProbe.cs` | Every `Update` and in `Application.onBeforeRender`: `InputDevices.GetDeviceAtXRNode(XRNode.Head)` → `isValid`, `CommonUsages.trackingState`, `devicePosition`, `deviceRotation`; the main camera's world pose; `XREALPlugin.GetTrackingType()`; loader active (`XREALUtility.GetActiveLoader() != null`). Append to `pose.csv` with a `phase` column (`update`/`render`). |
| `GateController.cs` | State machine for the automatic gates, keyboard control, HUD, event log. Uses `XREALSessionManager.OnGlassesConnect/OnGlassesDisconnect/OnStartSubsystems/OnStopSubsystems` and `XREALCallbackHandler.OnXREALGlassesPowerSave` to log lifecycle events. Enters/exits XR **only** by calling the scene's `HelloMR.OnClickSwitchXRButton()` (reuses the vendor flow). |

Keyboard (also printed in the HUD):

| Key | Action |
|---|---|
| `C` | Start/stop camera capture |
| `X` | Enter/exit XR (`HelloMR.OnClickSwitchXRButton`) |
| `3` / `6` | `SwitchTrackingTypeAsync(MODE_3DOF / MODE_6DOF)`; log the returned bool and `GetTrackingType()` after |
| `R` | Recenter (`XRInputSubsystem.TryRecenter()`, as `HelloMR`) |
| `M` | Write `mark,<n>` to `events.csv` (motion protocol, §6) |
| `Esc` | Stop capture, exit XR, flush, quit |

Automatic sequences (every step writes an `events.csv` row with its result):

- **G1** — wait for glasses (timeout 30 s → `NO_GLASSES`, quit) → device info →
  `StartCapture` → run `duration` → three restart cycles (`StopCapture`, 2 s,
  `StartCapture`, 10 s) → `StopCapture` → quit. **No XR.**
- **G2** — wait for glasses → device info → Enter XR → wait `OnStartSubsystems`
  (timeout 20 s) → `MODE_6DOF` → run `duration` (Jack performs §6 S2 with `M`
  marks) → Exit XR → quit.
- **G3** — as G2, plus `StartCapture` after 6DoF (`xrfirst`) or before Enter XR
  (`camfirst`); `StopCapture` before Exit XR.

HUD (`OnGUI`, top-left): gate, elapsed, camera fps over the last 2 s, last
`cam_ts`, resolution, tracking type, head position (mm) and yaw/pitch/roll (deg),
last event. In XR the main window is shown on the glasses; the HUD is
diagnostic only.

## 6. Phase S — Surface (Jack)

### `run_gates.bat` (written by Claude Code in D8)

> **Superseded 2026-10-13 by §13** (guided protocols, new menu). Kept for the record.

A menu launcher in the build root. Each option runs
`GateHarness\XrealPcGates.exe -gate … -label <option> -logFile "<runroot>\player_<gate>_<timestamp>.log"`
so that every run keeps its own Unity player log next to the CSVs.

```
1  G1   camera only (no XR)
2  G2   6DoF (Enter XR)
3  G3   camera + 6DoF, XR first
4  G3b  camera + 6DoF, camera first
5  Manual (keyboard)
6  Vendor RGBCamera sample (control, unmodified)
7  Open run folder
8  Zip all runs for transfer
Q  Quit
```

### S0 — Preparation

1. Unzip the build to `C:\EyeLabGates\` (short path, no spaces).
2. Surface on mains power. Close Nebula and every XREAL app (Task Manager).
   Decline any firmware-update prompt.
3. Glasses connected directly to the Surface USB-C port, no hub.
4. Screenshot *Settings › System › Display* — Enter XR changes monitor
   arrangement and primary monitor; this is how you restore it.
5. Put plate A or plate L at ~0.5 m on a textured table, good light. The
   snapshots then double as material for later calibration checks.
6. Lay a tape on the floor: mark **A** and mark **B** 500 mm to the right of A.
   Pick two reference directions 90° apart (e.g. two walls of a room corner).

### S1 — G1 (option 1, ~90 s)

Look at the plate. Press `M` at each phase change:
static 20 s → `M` → slow head motion 20 s → `M` → cover the Eye with your hand
5 s → `M` → uncover. The automatic restart cycles follow; stay still.
If no frames: run option 6 (vendor control) once, then option 4 (G3b) — the
camera may only start inside an XR session. Note the outcome of each.

### S2 — G2 (option 2, wear the glasses)

Wear them for the whole run (power-save can tear the session down). After the
6DoF switch:

1. Static 30 s, head still → `M`.
2. Translation ×3: stand on A facing forward → `M` → side-step to B keeping
   height and facing, hold 5 s → `M` → back to A, hold 5 s → `M`.
3. Rotation ×3: face reference 1 → `M` → turn to reference 2, hold 5 s → `M` →
   back, hold 5 s → `M`.
4. Cover the Eye 5 s while still → `M` → uncover → `M`.
5. Press `3` (3DoF), side-step once A→B→A with `M` marks, press `6`.

### S3 — G3 (options 3 and 4)

Repeat the S1 phases and one S2 translation repetition in each run.

### S4 — Return the logs

Option 8 → zip → copy to the desktop into
`unity_projects/xreal_pc_gates/GateRuns/`. Tell Claude Code which runs to
analyse and anything you noticed (flicker, freezes, disconnects, display
changes).

## 7. Acceptance criteria

Thresholds are **functional**: the reference (tape steps, room corner) has
centimetre-level uncertainty, so these detect "working vs not working", not
accuracy.

**G1 — PASS when all hold in at least one run:**

1. `StartCapture` returned true and frames arrived: median unique-`cam_ts` rate
   ≥ 20 Hz over the static and motion phases (Eye nominal video 30 fps,
   `CONTEXT.md`).
2. `cam_ts` strictly increasing; no gap > 0.5 s outside covered/restart windows.
3. Live content: median `y_mad_prev` in the motion phase ≥ 2× the static phase;
   `y_mean` in the covered phase ≤ 50 % of the uncovered phases; Jack confirms one
   snapshot shows the plate.
4. Resolution non-zero and constant within each capture session (record it).
5. 3/3 restart cycles deliver a frame within 5 s of `StartCapture`.

*Partial* = 1 holds but 3 or 5 fails. *Fail* = no frames; record the stage:
no glasses / session not created / `StartCapture` false / true but no frames.

**G2 — PASS when all hold:**

1. `SwitchTrackingTypeAsync(MODE_6DOF)` returned true and `GetTrackingType()`
   reports `MODE_6DOF`.
2. Head device valid, `trackingState` has Position and Rotation, pose is not
   constant/identity (the September failure signature).
3. Each translation rep: |Δp| = 500 ± 75 mm; return-to-A residual ≤ 50 mm.
4. Each rotation rep: 90 ± 10° about the vertical axis.
5. While moving, pose changes on ≥ 90 % of rendered frames.

Record without threshold: static jitter (std, mm/deg) and drift over 30 s;
behaviour with the Eye covered; behaviour in 3DoF (position should stay fixed).

**G3 — PASS when** G1 criteria 1–3 and G2 criteria 1–3 hold in the same run
(≥ 120 s) for at least one ordering. Record: camera rate vs G1, pose behaviour
vs G2, which orderings work, and the `cam_ts` ↔ `t_unity` relation (slope →
timestamp unit; spread of receipt delay). `InputDevices` exposes no pose
timestamp, so capture-time pose lookup stays an **open G4 item** regardless.

## 8. Analysis and recording (Claude Code, desktop)

Package `python/xreal_gates/` (repo `venv`; tests beside modules per AGENTS.md):

| Module | Content |
|---|---|
| `load.py` | Read a run folder into dataframes + `run.json`. |
| `segments.py` | Split by `M` marks and controller events into phases. |
| `metrics.py` | Frame rate, gaps, `y_mad`/`y_mean` ratios, restart latency, displacement/rotation per rep, jitter/drift, clock slope. |
| `report.py` | Markdown verdict per criterion (value, threshold, PASS/FAIL). |
| `__main__.py` | `python -m xreal_gates analyze <run_dir>... --out <report.md>` |
| `test_metrics.py` | Synthetic logs: a passing run, a frozen-timestamp run, an identity-pose run. |

Then:

1. Save reports as `GateRuns/<run>/report.md` (gitignored) and copy the verdict
   summary (no raw data) into `CAPABILITIES.md`: update row 11 and add rows for
   *PC beta Eye frames*, *PC beta 6DoF*, *PC beta concurrent camera + 6DoF*,
   each with host, firmware string from `run.json`, label `observed`, date.
2. Record the firmware string in `HARDWARE.md` (closes CAPABILITIES row 4 if
   the version call returned a value).
3. `CHANGELOG.md` (Unreleased) and the `CONTEXT.md` *Last updated* stamp.

## 9. Outcome matrix → next step

| G1 | G2 | G3 | Meaning | Next |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | Windows beta can host EyeLab's acquisition + tracking. | G4: validate vendor intrinsics against a ChArUco calibration of the delivered mode; camera-to-head extrinsic; capture-time pose. Then the Windows `IArucoPoseBridge` client + Python service (ADR-001), plate L first. |
| ✓ | ✓ | ✗ | Camera and 6DoF are mutually exclusive or degrade each other. | Measure switch cost; evaluate a register-then-hold mode. Ask XREAL. |
| ✓ | ✗ | – | Frames, no world tracking. | Restricted marker-visible mode only (PnP per frame, no persistence). Report to XREAL. |
| ✗ | ✓ | – | World tracking, no camera. | No registration on Windows; manual-placement demo only. Android host (Beam Pro / listed phone) for the camera. |
| ✗ | ✗ | – | PC beta route closed on this firmware. | Send logs to XREAL (Discord thread of 2026-10-06); Android host. Do not resume unofficial routes on this basis. |

## 10. Troubleshooting

| Symptom | Check |
|---|---|
| `XREALSessionManager` / `Cysharp` not found | UniTask not resolved → `UNITASK` undefined. Check `packages-lock.json` and git access. |
| Log repeats `Connection to XREAL One glasses failed!` | NCM adapter on the Surface: `Get-NetAdapter`, `Get-NetIPAddress` should show `169.254.2.10`. Nebula still running? |
| Session created, `StartupState` never reaches `END` | Record; try vendor control build; power-cycle the glasses by unplugging once. |
| `StartCapture` true, no `OnRGBCameraUpdate` | Record; try G3b (camera inside XR session). |
| Disconnect / power-save events mid-run | Wear the glasses; check the glasses' sleep setting in their menu (do not change firmware). |
| Monitors rearranged after XR | Restore from the S0 screenshot (*Display settings*), or `Win+P`. |
| DLL missing in build | Plugin importer: `Runtime/Plugins/X64/*.dll` must be enabled for Standalone Windows x86_64. |

## 11. D-phase report (2026-10-09, Claude Code on `lagann-0526`)

D0–D6 and D8 are done; D7 (optional desktop Play smoke test) was not run and
still needs Jack's go-ahead. Nothing was run against the glasses; no XR was
entered on the desktop; no firmware or USB configuration was touched.

**Deviations from the plan above (runtime wins):**

1. **Editor 6000.6.5f1, not 6000.4.5f1.** Only 6000.6.5f1 is installed on the
   desktop (with Windows Standalone support); 6000.4.5f1 is not. Jack chose
   6000.6.5f1 for the throw-away gate project. `eyelab_xreal/` stays on
   6000.4.5f1. The Built-in RP 3D template is deprecated from Unity 6.5 but
   still present in 6.6. The template's extra packages (Input System, Visual
   Scripting, Timeline, code-coverage tooling) were left in place; none of AR
   Foundation, XR Hands or XRI was added.
2. **`XREALSettings` asset created by script.** `XREALXRLoader.Start` calls
   `XREALSettings.GetSettings().IsDeviceSupported()` without a null check; the
   settings object is normally created by the XR Plug-in Management UI.
   `ProjectSetup` creates `Assets/XR/Settings/XREALSettings.asset` with vendor
   defaults (`SinglePassInstanced`, `InitialTrackingType = MODE_6DOF`, both
   device categories) and registers it as the config object.
3. **`HelloMR.m_UseSecondWindow` set to `false` in the harness scene only.**
   The prefab ships `true`. In a player, `HelloMR.StartXRAsync` then calls
   `XREALExperimentalPlugin.Restart()` when no 3840×1080 display was
   enumerated at startup; `Restart()` relaunches the exe **without arguments**
   and `TerminateProcess`es the running one, which would kill an automatic gate
   run. The harness uses the main window (fullscreen on the glasses after Enter
   XR). The **vendor control build keeps `true`**: expect it to restart itself
   on Enter XR; that is vendor behaviour, not a harness fault.
4. **XR is refused in the Editor.** `GateController` blocks `X` and the G2/G3
   sequences in Play mode (desktop safety, §1); the Editor always runs `manual`.
5. **Extra `run.json` probe.** Device info is probed at glasses connect
   (`device_connect`) and again after `OnStartSubsystems` (`device_xr`).
6. **`OnGlassesConnect` does not imply a session.** The vendor
   `XREALSessionManager.Update` fires `OnGlassesConnect` even when
   `CreateSession()` returned false (TCP preflight failed or startup state
   unreadable). The harness therefore copies every Unity log line containing
   `XREAL` (and every error/exception) into `events.csv` as `unity_log`, so the
   failure stage in §7 can be read from the session manager's own messages.

**Log schema (for `python/xreal_gates/`, §8).** Times are
`Time.realtimeSinceStartupAsDouble` (s), positions metres, invariant culture.

| File | Columns |
|---|---|
| `camera.csv` | `t_unity,frame,cam_ts,res_w,res_h,y_mean,y_mad_prev,dup_ts` (`y_mad_prev` empty on the first frame of each capture session) |
| `pose.csv` | `t_unity,frame,phase,valid,tracking_state,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,cam_pos_x,cam_pos_y,cam_pos_z,cam_rot_x,cam_rot_y,cam_rot_z,cam_rot_w,tracking_type,loader_active` (`tracking_state` = `InputTrackingState` flags; Position = 1, Rotation = 2; `tracking_type` empty when no loader is active; camera = rig `CenterCamera`, found by `MainCamera` tag) |
| `events.csv` | `t_unity,frame,event,result,detail`. Events: `gate_start`, `wait_glasses`, `glasses_connect`/`_disconnect`, `device_info`, `start_capture`/`stop_capture` (result = returned bool), `step` (result = step name: `run`, `restart_<i>_off`, `restart_<i>_on`), `enter_xr`, `enter_xr_wait`, `start_subsystems`/`stop_subsystems`, `switch_tracking` (result = returned bool, detail `after=<GetTrackingType>`), `tracking_type_changed`, `mark` (result = n), `key`, `recenter`, `power_save`, `native_disconnect`, `xreal_error`, `unity_log`, `gate_result` (`COMPLETED`, `NO_GLASSES`, `XR_TIMEOUT`, `XR_REFUSED`, `ABORTED_BY_USER`), `quit` |
| `run.json` | sections `args`, `build` (from the baked `EyeLabGatesBuildInfo` resource), `host`, `device_connect`, `device_xr` |
| `snapshots/<cam_ts>.pgm` | full Y plane, binary PGM, row 0 = first row of the native buffer |

**Results.**

| Step | Result |
|---|---|
| D0 | CLI 1.0.0-beta.12 (beta.13 available, not installed), signed in, Personal licence active; git 2.54; both tarball SHA-256 match `XREAL-PC-BETA-2026-10-09.md`. |
| D1 | `unity projects create … --template com.unity.template.3d --no-cloud`. First attempt failed in the API updater (`MovedFromExtractor` exit 1, no output; the same command succeeds standalone) — transient; the second attempt succeeded. |
| D2 | `com.xreal.xr` 3.1.2 and `com.xreal.xr.experimental` 3.1.2 (local tarballs), `com.cysharp.unitask` 2.5.11 (git `a77a1b4a`), `com.unity.xr.management` 4.7.0, `com.unity.xr.core-utils` 2.6.0, `com.unity.xr.legacyinputhelpers` 3.0.1. No resolution errors. Import errors only for the vendor's XRI prefabs (`XR Interaction Setup.prefab`, `XR Interaction Hands Setup.prefab`: missing nested XRI prefab), expected without XRI. |
| D3 | Standalone sample copied with 11 `.meta` files. |
| D4 | `ProjectSetup.Apply`: Win64 target, D3D11 only, Mono, input handling Both, `EyeLab`/`XrealPcGates`, windowed 1280×720 resizable, run/visible in background; loader `Unity.XR.XREAL.XREALXRLoader` assigned via `XRPackageMetadataStore.AssignLoader`; `InitManagerOnStart=false`; both active XREAL validator rules pass. Idempotent (re-run inside D8). |
| D5 | `GateSceneBuilder.Build` → `Assets/EyeLabGates/Scenes/GateHarness.unity`; verified wiring; vendor scene verified unmodified. |
| D6 | 0 compile errors, 0 warnings in `Assembly-CSharp(-Editor)` (Unity does not print package compiler warnings). |
| D7 | Not run (needs go-ahead). |
| D8 | Both players built (one warning each: `XREAL/XREALEmbeddingURP shader is not supported`, expected on Built-in RP). DLL check 11/11 for both (`libonigwrap.dll` is also present, from Unity). `BUILD_INFO.txt`, `run_gates.bat` (CRLF), zip `Builds/XrealPcGates_<yyyymmdd_HHmm>.zip`. |

Re-run everything with:

```
unity run C:\GitHub_Uni\EyeLab\unity_projects\xreal_pc_gates -l <log> -- -executeMethod EyeLabGates.EditorTools.BuildGates.BuildAll
```

(`unity run` adds `-batchmode -quit` itself and rejects a forwarded `-quit`.)

## 12. Surface results (2026-10-10, 2026-10-13)

Full numbers: `GateRuns/CODY_0526_20261013_143648/report.md` (gitignored). Status rows:
`CAPABILITIES.md` 4, 11, 16-18. Analysis was done with a quick script; the §8
package is still to be written.

**2026-10-10: nothing ran.** Windows Smart App Control refused `libnr_api.dll`
(error 4551, `An Application Control policy has blocked this file`) in every
player, including the vendor control → `NotFindRuntime`, no session, black XR
display, frozen HUD. All XREAL DLLs are unsigned. Jack turned Smart App Control
off on 2026-10-13 (`HARDWARE.md`).

**2026-10-13:**

| Gate | Result | Missing for a formal verdict |
|---|---|---|
| G1 | Frames: 30.0 Hz, 1280×720, `cam_ts` in ns, strictly increasing, no skipped frames, max gap 67 ms; snapshots show a live upright scene | S1 marks (static/motion/covered), restart cycles (capture was stopped after 17 s by the sample's on-screen Stop button), plate in view |
| G2 | 6DoF: switch true, pose valid with Position+Rotation, changing on 99 % of render frames; TrackingReason Initializing → Scanning → None in ~4 s | S2 translation/rotation reps with marks, ≥180 s |
| G3 xrfirst | **Degrades.** Camera 23.8 Hz but every frame is delivered in order, so latency grows ~0.2 s per second (+6.8 s after 33 s); TrackingReason never reaches None; pose plausible for ~16 s, then runs away to 23 m and freezes | – (fails as run) |
| G3b camfirst | **Fails.** Camera 29.6 Hz; perception start errors (`GlassesDisconnect`, `InvalidArgument`); pose constant for the whole run **while reported valid with Position+Rotation** | – |

Outcome-matrix row (§9): **G1 ✓, G2 ✓, G3 ✗ on this host** — camera and 6DoF
degrade each other. What the data does not yet separate: SDK behaviour vs.
Surface CPU load (H.26x decode, SLAM and 3840×1080 rendering on an Iris Plus
machine), and the effect of the start order.

Lessons for the harness and for EyeLab:

- `InputDevices` `isValid`/`trackingState` are **not** health signals (G3b). Use
  pose change plus the native TrackingReason (`NotTrackingReason`: 0 None,
  1 Initializing, 2 Relocalizing, 3 InsufficientLight, 4 InsufficientFeatures,
  5 ExcessiveMotion, 6 Unsupported, 7 CameraUnavailable, 8 Scanning; visible in
  the player log as `TrackingReason changed`).
- The vendor camera texture acquires at most one frame per Unity `Update` and the
  native layer queues, so a slow consumer accumulates latency instead of dropping
  frames. EyeLab must read the newest frame, never the oldest.
- The vendor calibration calls (`GetCameraIntrinsic`, `GetDevicePoseFromHead`,
  `GetDeviceResolution`) return `false` before XR starts and plausible values
  inside XR (RGB fx≈fy≈607.3 px at 1280×720; RGB camera ≈28 mm ahead of the head
  origin; displays ±31.6 mm). Vendor claims for G4, not validated.
- Showing the Eye image on the glasses (the G3 observation) is uncalibrated video
  see-through. It proves the data path. Registering an optical overlay needs the
  camera-to-head extrinsic and per-eye projection validated against a ChArUco
  calibration of the delivered mode (G4, ADR-003).

**Proposed next runs (not yet implemented):**

1. **G3c:** camera started only after TrackingReason reaches None, with the
   reason logged per frame (the API is `internal`; read it by reflection).
2. **Latest-frame reader:** drain the native queue each `Update` and log the
   backlog, to test whether latency can be bounded.
3. **G3 in 3DoF:** camera + XR rendering without SLAM, plus process CPU time in
   the logs, to separate render/decode load from SLAM.
4. **Full S1/S2 protocols** for formal G1/G2 verdicts.
5. Harness fixes: abort with `NO_SESSION` when the session fails; log each
   distinct native error once with a count; disable the sample's on-screen
   Play/Stop buttons in the harness scene.

## 13. Guided harness (build of 2026-10-13) — operator instructions

What changed after §12, and exactly what to do on the Surface.

### What changed in the harness

- **Guided protocols.** G1, G2 and G3 now run the §6 protocols as timed steps.
  The current instruction is shown in **yellow** with a countdown, the next one
  in grey, and every new step is announced by a **short beep**. No `M` presses
  are needed (`M` still adds an extra mark). Steps are logged as `step` events,
  which is what the analysis segments on.
- **Tracking state on screen.** The HUD shows `TRACKING: READY` (green) or the
  reason it is not ready (red: `INITIALIZING`, `SCANNING`, `INSUFFICIENTLIGHT`,
  ...), read from the native `GetTrackingReason` export. A **long high beep**
  sounds when tracking becomes ready.
- **New G3 variants:** G3c (camera only after READY), G3c-drain (same with a
  latest-frame reader), G3-3DoF (camera with XR rendering but no 6DoF tracking),
  and the 13 Oct G3-xr ordering for comparison. G3b (camera before XR) is
  dropped from the menu: it failed on 13 Oct.
- **Live diagnostics:** camera fps, **camera lag growth** (how far behind the
  image is compared with the start of capture), frames drained, process CPU
  in cores.
- **HUD in both eyes** in XR (the 3840×1080 window is side by side), larger font.
- **Fixes:** the sample's on-screen Play/Stop buttons are hidden (a click
  stopped the 13 Oct G1 capture); `NO_SESSION` stops the run when the native
  session was not created; repeated native errors are logged once and counted
  (`repeat_summary` at the end) instead of thousands of rows.

### Do I click anything?

**No.** Start a gate from `run_gates.bat`; from then on the harness connects,
enters XR, switches tracking, starts the camera and steps through the protocol
by itself. Do not click in the app window (the sample's remaining "Enter XR"
and DoF buttons would interfere). Keys: **Esc** aborts (only if something is
clearly wrong; an aborted run is still logged), **M** adds an extra mark. Let
every run finish on its own: the formal criteria need the full length.

### How do I know tracking is ready (G2, G3c)?

After Enter XR and the 6DoF switch the harness shows, in the glasses:

> LOOK AROUND SLOWLY at a well-lit, textured part of the room until TRACKING turns READY. (60 s)

with the red tracking line below it, typically `INITIALIZING`, then `SCANNING`.
`SCANNING` means the tracker wants to see the environment: turn your head
slowly left and right over furniture, shelves, posters (not a blank wall, a
window or only the floor). Do not cover the Eye. When the tracker reports
normal tracking for one continuous second:

- the line turns **green `TRACKING: READY`**;
- you hear a **long high beep**;
- the yellow instruction switches to the first protocol step (with its short beep).

On 13 Oct this took about 4 s in G2. If it is not ready after 60 s, G3c stops
with `TRACKING_TIMEOUT` (G2 carries on; the run then documents the failure).
If tracking drops later, the line turns red again; keep following the steps.

### Before the session (once)

1. Copy and unzip the new build (`Builds/XrealPcGates_20261013_1513.zip`) to
   `C:\EyeLabGates\`, replacing the old folder. Smart App Control stays off.
2. Optional but tidy: move the old run folders out of
   `%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\GateRuns` (menu `O`), so
   the transfer zip holds only the new runs.
3. Mains power; Nebula and every XREAL app closed; glasses straight into the
   Surface USB-C; decline any firmware update.
4. Scene: plate A or L at about 0.5 m on a textured table in good light; tape
   marks **A** and **B** on the floor 500 mm apart (B to the right of A); two
   reference directions 90° apart (e.g. two walls of a room corner); some
   textured objects in view for tracking.

### Run order and what to do in each

Run them in this order. Wear the glasses for every run (in G1 holding them is
fine, but the Eye must see the plate).

| # | Run | What happens automatically | What you do |
|---|---|---|---|
| 1 | **G1** (~1.5 min, no XR) | connect, camera on, 4 guided steps, 3 automatic camera restarts, quit. The HUD is on the **Surface screen** (no XR) | Look at the plate: **still 20 s, slow head motion 20 s, cover the Eye 5 s, uncover 10 s**, then stay still during the restarts. Follow the beeps. |
| 2 | **G2** (~4 min) | connect, Enter XR (screens rearrange), 6DoF, wait for READY, 40 guided steps incl. an automatic 3DoF/6DoF switch, exit XR, quit | Put the glasses on when told; look around until READY. Then: **still on A 30 s**; **3 translations** (on A, side-step right to B, hold, back to A, hold); **3 rotations** (reference 1, turn 90° to reference 2, hold, back, hold); **cover/uncover**; after the switch to 3DoF **one more translation**. Keep height and facing during side-steps. |
| 3 | **G3c** (~3 min) | as G2 up to READY, then camera on, 11 guided steps, camera off, quit | Look around until READY, then: still at the plate 20 s, slow motion 20 s, cover 5 s, uncover 10 s, one translation A-B-A, look around slowly 40 s, still 20 s. Watch the HUD **lag** value and the TRACKING line; note anything odd. |
| 4 | **G3c-drain** (~3 min) | same as 3 with the latest-frame reader | Same as 3. |
| 5 | **G3-3DoF** (~2.5 min) | XR in 3DoF (no tracking wait), camera on, same steps | Same steps as 3 (side-steps do not move the pose in 3DoF: expected). |
| 6 | **G3-xr** (~2.5 min) | the 13 Oct ordering: camera right after the 6DoF switch | Same steps as 3. |
| – | Manual / Vendor | only if something needs a closer look | – |

Then menu **Z** zips all runs. Copy the zip to the desktop into
`unity_projects/xreal_pc_gates/GateRuns/` (not into `Builds/`, which every
rebuild cleans) and note what you saw (lag, flicker, tracking drops,
disconnects).

### What the comparison will tell us

| If ... | then ... |
|---|---|
| G3c keeps READY and the lag stays bounded | the 13 Oct failure was the start order; concurrent use is feasible on the Surface |
| lag grows in G3c but not in G3c-drain | the backlog is in the application read path; EyeLab must drain to the newest frame |
| lag grows in both and `drained` stays 0 | the backlog is upstream (decoder/transport): host CPU or SDK limit |
| G3-3DoF is fine but G3c is not | 6DoF tracking plus camera is the limit, not rendering |
| CPU (cores busy) near the Surface core count when it fails | host-bound; a faster host (desktop + DP-in, ADR-003 Q6) is the next test |

### Log schema changes (supersede the §11 table where they differ)

- `camera.csv` adds `drained` (frames discarded before this one in the same
  Update; empty for the vendor reader) and `source` (`vendor`/`drain`).
- `pose.csv` adds `tracking_reason` (native `NotTrackingReason`, 0 = ready;
  empty when XR is off).
- New `perf.csv` (1 Hz): `t_unity,frame,unity_fps,cpu_process_pct,cpu_cores_busy,gc_mb,cam_fps,tracking_reason,step`.
- New events: `session` (`OK`/`NO_SESSION`), `tracking_reason` (on change),
  `tracking_ready` (`WAIT`/`READY`/`TIMEOUT`), `protocol` (`begin`/`end`),
  `step` (result = step id), `repeat_summary` (result = count). `xreal_error`
  and `unity_log` now appear once per distinct message.
- `-order` accepts `tracked` (default), `xrfirst`, `3dof`, `camfirst`;
  `-reader vendor|drain`. `-duration` only applies to manual. New `gate_result`
  values: `NO_SESSION`, `TRACKING_TIMEOUT`.

## 14. Surface runs of 2026-10-14 and the voice-guided build

### What ran

All 14 Oct runs used the **9 Oct build** (`run.json` `build.utc_build`
`2026-10-09T13:54:41Z`): the §13 guided build had not been copied to the
Surface. Hence no guidance, no beeps and the small top-left HUD. The launcher
and the HUD now show the build time so this is visible at once.

Even unguided, the runs were long enough to add facts (`GateRuns/CODY_0526_20261014_095954/`):

| Run | Camera | 6DoF | Notes |
|---|---|---|---|
| G1 | 30 fps (median), 1280×720, **no lag growth over 96 s**; 3/3 restarts deliver the first frame after **0.9 s** | – | **completed**; phases unmarked, so G1 criterion 3 is still not assessable |
| G2 | – | tracking ready **24 s** after XR start (Scanning before), then ready for the rest of the 180 s run (two short InsufficientLight episodes); pose range 512 mm in x, yaw 186° | the full-screen sample preview showed **solid green** (YUV shader with no frames) across both eyes; the process ended during Exit XR before the final log flush (no `gate_result`) |
| G3 (camera right after the 6DoF switch) | rate falls over time: **23 → 17 → 10 → 9 → 8 fps** per 30 s; every frame still delivered in order, so the **lag grows to 102 s after 178 s** ("a couple of minutes old", operator) | **ready after 2.6 s and stayed ready** (brief InsufficientLight) for 3 min; pose sane (range 0.9 m) | unlike 13 Oct, no runaway: the 13 Oct divergence is not reproducible |
| G3b (camera before XR) | 29 → 27 → 21 → 10 fps; lag 17.5 s after 105 s | `SwitchTrackingType(6DOF)` false, pose constant (same as 13 Oct) | camera-first breaks 6DoF: **reproduced** |

Unity's render rate stayed at 57–60 fps in all XR runs while the camera rate
fell, so the main thread is not the bottleneck: the camera pipeline upstream of
the acquisition API (decoder/transport) loses throughput over minutes once XR
runs, and nothing drops old frames. Thermal throttling of the fanless Surface
is a candidate; the new `perf.csv` columns test it.

Status after 14 Oct: **G1 frames: observed (complete run). G2 6DoF: observed
over 3 min. Concurrent camera + 6DoF: tracking holds, but the camera stream
becomes unusable within a minute on the Surface (unbounded latency).** The
register-then-hold idea (§9, row 2) or a bounded-latency camera path is
required; the G3c-flush run tests the cheapest form.

### What changed in the build of 2026-10-14

- **Spoken cues** (Windows speech, English voice if installed, else the
  default voice): every step is announced ("Step right to B", "Hold",
  "Cover the camera", "Tracking ready", "Tracking lost", "Finished"). Sound goes
  to the Windows default output device.
- **Beeps:** a beep at every step start, **3-2-1 ticks** before the end of
  steps of 8 s or longer, a long high beep when tracking becomes ready.
- **Large banner in the middle of each eye:** step title in large letters,
  countdown (red in the last 3 s), progress bar, next step; the whole banner
  **flashes yellow** for 0.8 s at every step change. Tracking line below it
  (green READY / red reason). Diagnostics in small print at the bottom left,
  including the **build time**.
- **No more green screen:** the sample's full-screen preview is hidden; a small
  grey preview of the Eye image (with fps and lag) sits at the bottom centre of
  each eye, only while frames arrive.
- **G3c-flush** (menu 5): restarts the camera whenever its lag has grown by
  more than 1 s (`-flushLag 1`), to test whether a restart clears the backlog
  (G1: restart costs 0.9 s). The lag reference is now kept across restarts.
- `perf.csv` adds whole-system CPU utility and **CPU performance** (clock as a
  percentage of nominal; a fall under load means throttling) and the camera
  lag; the log is flushed immediately when the gate result is written.

### Operator procedure (supersedes the run table of §13)

1. Unzip `Builds/XrealPcGates_20261014_1024.zip` to `C:\EyeLabGates\`
   **replacing the old folder**, and start `run_gates.bat` from there. The menu
   header must read `build 2026-10-14T08:24:36Z`; the same time appears at the
   bottom left of the app. If it shows `2026-10-09` or `2026-10-13`, the old
   build is running.
2. Windows sound on, volume up. The cues play on the default output device;
   if that is the glasses you hear them in the glasses.
3. Run in this order: **1 G1** (sanity check that voice and banner work),
   **2 G2**, **3 G3c**, **5 G3c-flush**, **4 G3c-drain**, **6 G3-3DoF**. Let each
   finish; just do what the voice says. Esc only to abort.
4. Menu **Z**, copy the zip to the desktop into
   `unity_projects/xreal_pc_gates/GateRuns/`.

"A"/"B" are floor tape marks 500 mm apart (B to the right of A); "the board"
is the 25 mm ChArUco board at about 0.5–1 m; reference 1/2 are two directions
90° apart.

### Log schema changes (in addition to §13)

- `perf.csv`: `t_unity,frame,unity_fps,cpu_process_pct,cpu_cores_busy,cpu_system_utility_pct,cpu_performance_pct,gc_mb,cam_fps,cam_lag_growth_s,tracking_reason,step`.
- New events: `speech` (`READY` with the voice name, `UNAVAILABLE`, `STDERR`),
  `perf_counters` (`OK`/`UNAVAILABLE`), `camera_flush`.
- `step` detail is `<seconds>s <TITLE>: <detail>`; step ids are unchanged.
- `-flushLag <s>` (default 0 = off).

## 15. Voice-guided runs of 2026-10-14, host throttling, and the build/run workflow

### Results (build `2026-10-14T08:24:36Z`)

Voice cues, beeps and the banner worked as intended (operator). Per run
(`python -m xreal_gates summary`):

| Run | Result | Camera | Tracking | CPU clock (median / min, % of nominal) |
|---|---|---|---|---|
| G1 | COMPLETED | 30 fps, no lag | – (no XR) | 230 / 95 |
| G2 | COMPLETED (211 s protocol) | – | native log: ready 1.4 s after XR start | **63 / 52** |
| G3c, G3c-drain, G3c-flush | TRACKING_TIMEOUT, no camera frames | – | HUD "n/a" (bug below) | 58-60 / 43-51 |
| G3-3DoF | COMPLETED | **8 fps, lag 101 s after 137 s** | 3DoF | 59 / 44 |

Three findings:

1. **"Tracking n/a" was a harness bug.** The Windows `XREALXRPlugin.dll` does
   not export `GetTrackingReason` (its export table has 260 entries; the call
   threw and the harness stopped asking). G3c therefore waited 60 s for a READY
   that could never be reported and stopped before starting the camera. The
   DLL does export `SetNativeLogCallback` and formats
   `TrackingReason changed: %d -> %d` itself, so the new build installs that
   callback (the vendor does so only in the Editor), parses the reason, and
   writes every native line to `native.log` in the run folder. If no report
   arrives within 15 s the run continues with `tracking_ready,ASSUMED`. Also
   noted: the DLL exports `GetHeadPoseAtTime` and `GetHMDTimeNanos`, the
   candidates for capture-time pose lookup (G4).
2. **The Surface throttles.** `% Processor Performance` falls from about 220 %
   (turbo) to **45-60 % of nominal** about 50 s after XR starts and stays there
   for every later run, with only a brief recovery between runs. System load
   drops at the same time, so the CPU is held back, not saturated. Unity's own
   frame rate sags (38-58 fps). On the throttled CPU the Eye stream decodes at
   7-9 fps instead of 30; every frame is still delivered in order, so the image
   falls behind by 0.75 s per second (**101 s after 137 s**, matching the
   operator's "over 90 s"). G1 without XR on a cool Surface: 30 fps, no lag.
   This is the cause of the G3 latency on this host; the camera path itself is
   not slow. Whether it is thermal or power limiting is not separated (mains
   power and Windows power mode to be confirmed).
   **Corrected in §16:** throttling makes it much worse, but it is not the only
   cause. With 6DoF running, the host-side decoder falls behind even at full
   clock (24-28 fps).
3. **The Eye image buffer is bottom-up.** On a G1 snapshot, ArUco detection
   (DICT_4X4_50) finds 16 markers after a vertical flip, 2 spurious ones as
   stored and after a 180° rotation. So the preview's flip was backwards (the
   operator saw it upside down) and **all PGM snapshots up to build
   `2026-10-14T08:24:36Z` are upside down**: flip them vertically before any
   calibration use (a mirrored image calibrates to the wrong handedness).
   The §12 claim that the snapshots were upright was wrong.

### Changes in builds `2026-10-14T09:18:55Z` and `2026-10-14T10:03:14Z`

- Tracking reason from the native log callback (above); `native.log` per run;
  `native_log` events (`INSTALLED`/`UNAVAILABLE`, `SUMMARY` with line counts).
- Snapshots written **upright** (rows reversed); `run.json` `build.snapshot_orientation`.
- Preview: correct orientation, **colour** (BT.601 from the Y/U/V planes),
  45 % of each eye's width, bottom centre.
- HUD: red **SURFACE THROTTLED** line when the CPU clock is below 80 % of nominal.
- **Cool-down check** (launcher `K`, `-gate cooldown`, no glasses needed): every 10 s it loads
  two CPU threads for 3.5 s and reads the clock (an idle CPU clocks down by itself, so the
  clock is only meaningful under load); two readings at or above 150 % of nominal end it with
  "Cool enough" (cool Surface: about 220 %, throttled: 45-60 %). Probes are logged as
  `cooldown_probe` events, so cooling methods (radiator, fan) can be compared. Gave up after 10 min.
- Tracking wait shortened to 30 s (ready was reported 1.4-24 s after XR start); with no tracking
  report at all the run continues after 15 s.
- G3c-30fps (`-renderFps 30`, vSyncCount 2): halves the XR render load, to test
  whether a lighter load avoids throttling.
- Launcher: new order (G1, G3c, G3c-30fps, G3c-flush, G2, G3-3DoF, G3c-drain),
  outdated-build warning, export of new runs only.

### Build and run workflow (desktop ↔ Surface)

The repo carries the **source**; a build is a 150 MB artefact and is not in git.
After every rebuild both must reach the Surface:

1. Desktop: `BuildGates.BuildAll` writes `Builds/XrealPcGates_<yyyymmdd_HHmm>.zip`
   and the tracked pointer `unity_projects/xreal_pc_gates/LATEST_BUILD.txt`.
   Commit and push (the pointer travels with the source).
2. Surface: `git pull`; copy the newest zip from the desktop (USB stick or
   share) and unzip it over `C:\EyeLabGates\`.
3. The launcher compares its own `BUILD_INFO.txt` with `LATEST_BUILD.txt` in the
   pulled repo (found via `EYELAB_REPO`, `C:\github_repos\EyeLab` or
   `C:\GitHub_Uni\EyeLab`) and prints a framed **WARNING** when they differ.
   The build time is also shown at the bottom left of the app.

Retention:

- `Builds/` keeps only the newest zip; `Builds/archive/` keeps the two before it;
  older ones are deleted by `BuildAll` (each is reproducible from the
  `source_git_head` in its `BUILD_INFO.txt`). Run zips found in `Builds/` are
  moved to `GateRuns/_inbox/`.
- Surface: menu **Z** zips only runs not exported yet and moves them to
  `GateRuns\_exported\` (earlier exports were cumulative).
- Desktop: copy run zips into `GateRuns/_inbox/`, then from `python/`:
  `python -m xreal_gates import` (de-duplicates into `GateRuns/runs/<run>/`,
  attaches each player log as `player.log`, moves the zip to `_imported/`,
  which is safe to delete) and `python -m xreal_gates summary`.
  Session reports live in `GateRuns/sessions/`.

### Next Surface session

Before running: mains power, Windows power mode **Best performance**, Surface
upright on its kickstand in a cool spot (a desk fan on its back helps), and
run the cool-down check (`K`) between XR runs and start the next run when it
says "Cool enough". Order: **G1** (sanity: colour, upright preview), **G3c**,
**G3c-30fps**, **G3c-flush**, then G2 and the rest if time allows.
(Superseded by §16.)

## 16. Afternoon runs of 2026-10-14 (radiator) and the light-overlay gate G4

### Results (build `2026-10-14T10:03:14Z`, aluminium radiator on the Surface's back)

`python -m xreal_gates summary` now also prints the seconds at full CPU clock after
Enter XR (until the clock stays below 150 % of nominal for 3 s), the rate at which
the SDK *received* Eye frames, the hottest thermal zone and the G4 bursts.

| Run (in order) | Rest before | Full clock after XR | Decoded camera fps | Lag at end |
|---|---|---|---|---|
| G3c (after a 2 h rest) | 2 h | **96 s**, then ~140 % | 24-25 while at full clock, 16-18 after | 45 s |
| G3c-30fps | ~1 min | 47 s | 27-28, then 13-19 | 47 s |
| G3c-flush | ~1 min | 9 s | 8-11 | bounded at 2-3 s (27 restarts) |
| G2 | ~1 min | 7 s | – | – |
| G3-3DoF | ~1 min | throttled from the start | 7-8 | 105 s |
| G3c-drain | 7.5 min | 12 s | 7 | 101 s |

Same morning without the radiator: G2 49 s. Findings:

1. **The Eye stream is decoded on the host CPU, and the decoder is the
   bottleneck.** The SDK's native log counts frames received over USB (`rgb recv`):
   **29.8 fps in every run**, including the throttled ones, at 60-77 KB per
   1280x720 frame (a compressed stream). What reaches the application is the
   decoder's output:
   - 30 fps with no XR (G1);
   - 24-25 fps next to 6DoF at full clock;
   - 27-28 fps at a 30 fps render rate;
   - 7-10 fps when throttled.

   Every frame is delivered in order, so the lag grows whenever the decoder is
   slower than 30 fps. This corrects §15, which blamed throttling alone: on this
   host a continuous Eye stream next to 6DoF never stays real time.
2. **No application-side reader can skip the backlog.** The drain reader never
   found more than one frame waiting (`drained` = 0 in all 1088 rows), and its
   frames arrived exactly 33.5 ms apart. The queue sits inside the SDK, before
   the acquisition API.
3. **Restarting the camera empties that queue.** In G3c-flush, the lag after each
   restart fell back to 0.0-0.2 s, and the first frame arrived 0.1-1.1 s after
   the start. So a fresh frame on demand costs about 1 s and leaves the decoder
   idle in between. G4 is built on this.
4. **The radiator helps once, then saturates.** It doubled the first run's
   full-clock time (49 → 96 s), and the throttled clock stayed near 140 % instead
   of 60 %. After that the stored heat dominates: 7.5 min of rest gave only 12 s.
5. **The cool-down check (K) of 0.2.006 was wrong.**
   - Its 3.5 s probe read 205-233 % every time, even straight after a run that
     ended at 60 %. The CPU's short turbo budget refills within seconds of idling.
   - So it said "Cool enough" after 23 s, and the next runs throttled within
     7-15 s.
   - Fixed: the probe now lasts 20 s and is judged on its last 5 s, and the
     thermal zones are logged.
6. **6DoF is good when it runs.** In G2 the 90° turns read as 85-88°. The
   side-steps between floor marks 500 mm apart read as 0.36-0.47 m, but the
   operator may have missed slides because of the flipped view (item 8). So this
   is not yet a scale result; the camera is the better scale check later.
7. **The tracker wants motion while it scans.** Both tracking timeouts sat in
   `Scanning` while the operator held still. G2 became ready only at t = 96 s (89 s after Enter
   XR), during a side-step, so its 30 s "hold still" phase was recorded untracked.
   The wait now prompts "Step slowly side to side" every 8 s while not ready.
8. **G2 showed the whole XR view upside down in the glasses** (operator). None of
   the recorded data is flipped:
   - All camera snapshots of all seven camera runs are upright, by ArUco count
     as stored versus flipped.
   - The G2 head pose stayed level throughout.

   The flip therefore happened after the app. The suspect is Windows rotating
   the display (the Surface sits against the radiator, and auto-rotate follows
   its accelerometer). The new build logs every display's rotation at start and
   after Enter XR, and again whenever it changes. It also saves screenshots of
   what the app drew. Turn on Rotation lock on the Surface.

### Direction (Jack, 2026-10-14)

- **The target is the Beam Pro, so the pipeline must get lighter.** No desktop
  DP-in card will be bought (`HARDWARE.md`): it would be personal money, could
  not be reproduced, and could not be shown outside the lab.
- **The glasses show only an outline or grid with points**, calibrated and
  world-locked on the target, and no video.
- **Camera frames are only analysed**, not displayed.
- **Continuous 6DoF is not a requirement.**

### G4 — light overlay (build `2026-10-14T20:21:48Z`)

G4 starts the "rendering + calibration" gate that §1 and §9 left for later. Its
first part measures whether a light enough runtime is sustainable on the
Surface. Its burst snapshots of the board feed the remaining G4 items offline:
vendor intrinsics against ChArUco, and capture-time pose.

- **Display.** XR rendered at 30 fps (`-renderFps 60` overrides). After tracking
  is ready, the operator looks at the board and a **world-locked 60 x 40 cm
  grid** appears 1 m along the gaze:
  - 10 cm cells, 4 mm lines, 2 cm points at the corners and the centre;
  - one static mesh, one draw call per eye;
  - `P` re-places it.

  The HUD shrinks to a banner at the top, with no camera preview and no YUV
  conversion.
- **Camera bursts.** Every `-burstEvery` s (default 2) the camera starts. It
  stops after `-burstFrames` unique frames (default 3), or after 3 s without
  them. The last frame is saved as an upright PGM, and a `burst` event records
  the frames, the time to the first frame and the time on. `-burstEvery 0`
  never starts the camera (control).
- **Tracking.** `-order tracked` is 6DoF and waits for READY (continuing after
  30 s if needed). `-order 3dof` is 3DoF: the grid follows head rotation only,
  so it slides when you step. That is the case where camera registration would
  correct it later.
- **Protocol, 293 s:**

  | Step | Seconds |
  |---|---|
  | hold still on the board | 30 |
  | move head slowly | 30 |
  | translation A→B→A | 23 |
  | look away and back | 20 |
  | walk a slow loop, end on A | 40 |
  | hold still | 30 |
  | move head | 30 |
  | **sustain** hold | 60 |
  | look around, end on the board | 30 |

  The questions are whether the grid stays on the board and whether the clock
  holds over 5 minutes.
- **Launcher.**

  | Key | Option |
  |---|---|
  | 2 | G4-6DoF |
  | 3 | G4-3DoF |
  | 4 | G4-nocam |
  | 1 | G1 |
  | 5 | G2 |
  | 6-9, D | G3 variants (kept for comparing cooling) |
  | M | manual |
  | K | cool-down |

Also in this build:

- **Cool-down check.** Each check runs 2 busy threads for 20 s, judges the mean
  clock of the last 5 s against 150 %, then rests 20 s. One passing check is
  enough. Gives up after 10 min.
- **Tracking wait.** A movement prompt every 8 s while the tracker reports
  Scanning or Initializing.
- **Diagnostics.** `display` events (each display's mode, position and rotation;
  `CHANGED` when it changes). Screenshots in `screens/`: the first step, every
  30 s at a step change, tracking ready, and grid placed.
- **Logging.** `perf.csv` gains `temp_max_c,temps` (ACPI thermal zones; empty
  where Windows exposes none). `run.json` args gain `burst_every_s` and
  `burst_frames`.

### Next Surface session (fan instead of radiator)

Before running: Rotation lock on, mains power, Best performance, fan on the
back, and the new build (the launcher shows `[OK]`). Order:

1. **1 G1**: sanity check.
2. **K**.
3. **2 G4-6DoF**: does the grid stay on the board, and does the clock hold?
4. **K**.
5. **3 G4-3DoF**.
6. **K**.
7. **4 G4-nocam**.
8. **K**.
9. **6 G3c**: the full-video comparison with the radiator's 96 s.

Then **Z**. In each G4 run, note whether the grid slid, jumped or stayed put.
If any view is upside down, note which run and whether the Surface's own screen
was flipped too.
