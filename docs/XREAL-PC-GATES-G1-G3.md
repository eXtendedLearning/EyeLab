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

## 17. G4 with the fan (2026-10-16), markers in the burst frames, and 0.3.000: Eye calibration + G5 locked overlay

### Results (build `2026-10-14T20:21:48Z`, fan on the Surface's back)

| Run | CPU clock median / min | Full clock after Enter XR | Camera |
|---|---|---|---|
| G1 | 229 / 99 % | – (no XR) | 27 fps |
| G4-6DoF | 222 / 182 % | **whole run (315 s)** | 144/144 bursts complete, first frame 1.15 s after start |
| G4-3DoF | 223 / 135 % | **whole run (308 s)** | 145/145 bursts complete |
| G4-nocam | 212 / 126 % | 248 s | – |
| G3c (last, full video) | 111 / 73 % | 14 s | 12 fps, lag +72 s |

- **The light runtime is sustainable on the Surface with a fan.** Camera bursts cost
  nothing measurable against no camera, and the operator judged the 6DoF grid very
  stable. Full video still throttles.
- **Every burst frame is current.** The lag growth stays within 0.04 s, because each
  burst restarts the decoder's queue.
- **The camera is on about 60 % of the time.** Each burst lasts about 1.2 s, almost
  all of it waiting for the first frame.
- **The display and screenshots behaved.** Every display reported rotation 0 for
  the whole session (Rotation lock was on). The screenshots show the grid in both
  eyes and the HUD, so screenshots work in XR.
- **Temperature zones are logged.** The hottest one rose from 40 °C to 53 °C over
  the session.
- **K is still optimistic for full video.** It passed every time, and G3c then
  throttled after 14 s. For G4 and G5 it is adequate.

### Markers in the burst frames (offline, `python -m xreal_gates anchor`)

The burst snapshots were never shown; the camera only took them. Detecting the 25 mm
ChArUco board in them (IDs 0-16) and chaining each pose with the 6DoF head pose and the
vendor's RGB-camera-from-head pose (§16, frame chain in `python/xreal_gates/anchor.py`):

- **Detection.** The board was found in 59 % of the 6DoF run's frames (48 % in
  3DoF); the misses are mostly frames where the operator looked away. Markers are
  11-22 px wide at 0.7-0.9 m: the Eye's lens is wide, about 93° across, so 1 px is
  about 1.3 mm at that distance. The PnP reprojection error is 0.4 px.
- **Within a still phase the board holds.** Its world position varies by 6.6 mm
  (median) within a still phase.
- **Between phases it does not.** Its position differs by 10-110 mm between phases,
  with the largest offsets when the board sits off-centre in the image. This is the
  signature of an uncalibrated lens: the SDK supplies fx, fy, cx and cy but no
  distortion.
- **Duplicate IDs cause outliers.** A cluster 0.5 m to the side is almost certainly
  the 20 mm board, which carries the same IDs. **Only one marker set per ID may be in
  view.**
- **The board's size is not observable from its image** (a pure scale error
  reprojects perfectly); 25 mm is assumed.

### What 0.3.000 adds

**Eye calibration (launcher `C`, `-gate calib`).** Camera only, no XR, one snapshot
every 0.5 s for about 2 minutes. Wearing the glasses with the head still, the operator
holds the **25 mm board** and moves it into the zone outlined in yellow on the large
preview (Surface screen): centre at arm's length, close, the four corners, the four
edges, and tilted both ways, then a slow sweep.

The launcher then runs `python -m xreal_gates calibrate --latest-in <runs>` on the
Surface:

- **View selection.** It detects the board in every snapshot and picks up to 60 views
  spread over image position and board size.
- **Model.** It fits OpenCV's standard 5-coefficient model, and the rational model
  if that lowers the RMS by more than 15 %.
- **Output.** It writes `eye_calibration.yaml` with a JSON report: coverage,
  comparison with the vendor values, and the worst views. The YAML goes into the run
  folder and into
  `%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\calibration\`, where G5 reads it.
- **Refusals.** It writes nothing below 35 % image coverage, or when fx or fy is more
  than 25 % from the vendor's. The 16 Oct G1 snapshots, with the board in one place,
  gave RMS 0.37 px and fx 4344 px, which would have scaled every pose by about 7.

**G5 locked overlay (launcher `A` plate A, `L` plate L, `B` 25 mm board).** G4's light
runtime plus ADR-001's pose bridge.

- **Pose service.** The launcher starts `python -m xreal_gates serve --target ...` in
  a minimised window: Step C on the same machine, `ws://127.0.0.1:8765`, stdlib only.
- **Connection.** The gate connects before entering XR; a missing service ends the
  run as `NO_SERVICE`.
- **Overlay geometry.** The service answers with the camera model it uses (the
  calibration, else the vendor values, which the gate announces as a warning) and the
  target's geometry from the single definitions:
  - plates: `config/targets/plate_*.target.json`, the measured file when present,
    giving saddle grid, markers and page outline;
  - board: `calibrate.make_charuco_board`, giving squares and inner corners.
- **Sending frames.** The last frame of each 2 s burst is sent, as a raw Y plane,
  only while the head turns slower than 10°/s and moves slower than 8 cm/s. The head
  pose at receipt then stands in for the unmeasured pose at exposure.
- **Pose.** The service locates the target:
  - plates: markers plus the printed saddles, found by the validation detector, with
    at least **two markers** in view (saddles seeded from one 20 mm marker are often
    wrong; validation B1b, 15 Oct);
  - board: ChArUco corners.

  It solves PnP with IPPE plus LM refinement for planar targets, and refuses an RMS
  above 2 px or an ambiguous planar solution. It logs every frame to
  `pose_service.csv` in the run folder. Each (session, sequence) is processed once.
- **Placing the overlay.** The gate chains each pose into the world frame (`F R F`,
  then the RGB-camera-from-head pose, then the head pose). `LockAnchor` filters it:
  - the first lock places the overlay;
  - a pose within 3 cm and 3° joins an 8-pose mean;
  - three consistent outliers relock;
  - while **frozen**, poses are compared and logged but not applied.
- **Logging.** Every pose is a `lock` event: kind, jump in mm and degrees, markers,
  points, RMS, service and round-trip ms, world position. The burst frames are saved
  for `anchor`.
- **Protocol (220 s), target about 70 cm in front of A:**

  | Step | Seconds |
  |---|---|
  | acquire | 20 |
  | hold still: do the points sit on the printed crosses? | 30 |
  | move head | 20 |
  | step left | 15 |
  | step right | 15 |
  | closer, about 40 cm | 15 |
  | back to A | 15 |
  | **frozen**: walk around and come back | 40 |
  | **frozen**: hold still, has it drifted? | 20 |
  | live again, hold still | 30 |

Verified on the desktop:

- **Python.** Synthetic renders of each target give the pose within 0.5-5 mm and 1°
  (`test_g5.py`). Calibration recovers known intrinsics from synthetic views and
  refuses a one-place capture (`test_calib.py`).
- **Transport.** The real .NET `ClientWebSocket` exchanged hello and a 1280x720
  frame with the service: plate A at 599.9 mm for 600 mm, in 40 ms round trip.
- **Unity.** The players compile with 0 errors.
- **Not yet run on the glasses:** the Unity player's bridge, the drawn target
  overlay, and the frozen/live logic.

Known limits:

- **Timing.** The head pose is taken at frame receipt, not exposure. Capture-time
  pose (`GetHeadPoseAtTime`) is still open.
- **Extrinsic.** The RGB-camera-from-head pose is the vendor's, uncalibrated.
- **Eye-to-display calibration.** What the operator sees also depends on XREAL's
  display calibration; the frozen phase and "points on the crosses" check that end
  to end. It is not a camera-only measurement (ADR-003).

### Session order (Surface, fan, Rotation lock on, `git pull` first)

The calibration and G5 need the repo's Python with OpenCV on the Surface (the launcher
looks for `venv\Scripts\python.exe` in the repo, then `python` on PATH).

1. **C** (calibration): the 20 mm board out of sight.
   - Fill each yellow zone, holding each position still.
   - The console prints RMS, coverage and the comparison with the vendor values. If
     it refuses, repeat C, reaching further into the corners.
2. **B** (G5 on the 25 mm board): the board fixed upright on a table or wall, about
   70 cm from A.
3. **A** (G5 on plate A).
4. **L** (G5 on plate L): its fold at the front, both faces visible.
5. **K** between runs if the Surface feels warm.
6. **Z**.

For each G5 run, note:

- whether the points sat on the printed crosses, and by how much they missed (mm by
  eye);
- whether the overlay drifted during the frozen steps;
- whether the relock snapped it back.

## 18. First G5 runs (17-21 Oct, metroXRAINE 2026), the cause of the floating overlay, G5 rev. 2 and the demo gate

### What ran (build `2026-10-16T16:19:31Z`)

- **Calibration.** The Calib run of 21 Oct used 201 of 226 snapshots and covered 98 % of
  the image. RMS 0.47 px; standard model (rational: 0.46 px).
  - fx 623.9, fy 625.8, cx 646.4, cy 357.2 against the vendor's 607.2, 607.4, 638.3,
    359.2: the vendor focal length is **2.7 % short**, about 19 mm at 0.7 m.
  - k1 -0.079, k2 +0.128, k3 -0.066.
  - The two Calib runs of 17 Oct (one aborted) were superseded.
- **G5, plate A.** 17, 18 Oct (three runs, all aborted early) and 21 Oct (complete).
- **G5, 25 mm board.** 21 Oct (complete, after two aborted starts). Plate L was not
  available.
- **Operator report.** The overlay was never fixed. It floated 10-15 cm in front of the
  target and swung up and down and left and right, while its *rotation* was good.
  Freezing moved it metres away.

### Cause: 6DoF position drift, made worse by the anchor filter

- **The camera side is sound.** Over 30 s of holding still (board run): board in the
  camera frame steady (0.58-0.59 m, image position stable to about 1 cm), rotation to
  under 1°, reprojection 0.2-0.6 px.
- **The headset position drifted.** Over the same 30 s it moved 10 cm forward and 5 cm
  up, while its rotation stayed steady. During plate A's 40 s frozen walk the reported
  head position ranged over 7 m, then jumped back 3.4 m when tracking relocalised.
  Tracking reported `InsufficientLight` episodes in the frozen phases.
  - Rotation comes from the IMU; position comes from visual SLAM, which the venue (people
    moving, booth lighting) degraded.
  - A world-fixed object is rendered at a fixed place in a frame that moves with the
    drift, so it floats and swings.
- **No other chain explains it.** A search over frame conventions, signs and a head-pose
  delay of 0-0.3 s found no better chain than the implemented one.
- **The filter amplified it.**
  - Each target pose was averaged over the last 8 poses, and poses more than 3 cm off
    were rejected: 30 of 69 on plate A, 33 of 73 on the board.
  - So the overlay stayed at a stale, drifted position: median 48-56 mm, p90 230-315 mm
    from the latest camera pose.
  - Following the latest pose instead would leave only the drift between two locks:
    median 17-20 mm, p90 32-55 mm, at 2 s spacing, from the same logs.
- **"Floating in front".** This is the forward drift plus the stale anchor. A residual
  depth offset from the display's own eye model (IPD and virtual image distance) cannot
  be excluded yet; ADR-003 calls that eye-to-display calibration.

### G5 rev. 2 (build `2026-10-22T09:52:28Z`)

- **Follow the latest lock** (`LockAnchor`).
  - A pose within 15 mm and 1.5° is blended in by half, which hides per-frame noise.
  - Up to 10 cm, it replaces the anchor.
  - A larger jump is applied only once the next pose agrees within 3 cm and 3°.
  - Frozen: compared and logged, not applied.
- **Faster updates.** A burst every 1.2 s, using its 2nd frame. The motion gate is
  relaxed to 15°/s and 0.15 m/s; it skipped 35-46 % of bursts on 21 Oct.
- **3DoF variants** (plate A, board). Rotation-only tracking cannot drift in position;
  stepping sideways is corrected at the next lock.
- **Several targets at once.** The pose service takes `--target plateA,board25` (the IDs
  are disjoint); each target has its own anchor and colour (green, blue).
- **Protocol 3 min.** The frozen steps are now "hold still" (does it drift while you
  stand?) and one side-step; the 40 s walk is gone.

### Demo gate (launcher `D`)

No protocol; it runs until Esc. It shows the following, in 6DoF:

- **A world grid**, 60 x 40 cm, 1.5 m ahead.
- **Plate A and the 25 mm board**, each with its outline, markers and points locked by
  the camera.
- **The Eye's view** in the bottom-right corner: the latest burst frame, about once a
  second.

Operator keys on the Surface:

| Key | Action |
|---|---|
| G | grid on/off |
| P | place the grid 1.5 m along the wearer's gaze |
| E | Eye view on/off |
| T | target overlays on/off |
| Esc | end |

The pose service starts with both targets. If 6DoF drifts in the room, the targets
still correct themselves at every lock. The world grid does not, and drifting is then
what it shows.

### Launcher

The menu is now two levels:

- **Main menu.** D demo, C calibrate, 1 G1, 2 G2, then 3 / 4 / 5 open the G3, G4 and
  G5 submenus. K, M, V, O, Z, Q as before.
- **Submenus.** Each lists its variants and has B for back.
- **G5 submenu:**
  1. plate A
  2. plate L
  3. board
  4. plate A, 3DoF
  5. board, 3DoF
  6. plate A + board

### Next session

Order: **5 → 1** (plate A, 6DoF), **5 → 4** (plate A, 3DoF), **5 → 3** (board), then
**D**.

Note the following:

- whether the overlay now stays on the target while standing;
- how far it moves during the frozen still step;
- whether 3DoF looks steadier than 6DoF in the same room.

The calibration of 21 Oct stays valid (same glasses, same resolution).

## 19. G5 rev. 2 runs (22 Oct), the camera that died, stream mode and the view offset (0.3.002)

### What ran (build `2026-10-22T09:52:28Z`)

Plate A only: G5 6DoF (menu 5 → 1) and G5 3DoF (5 → 4).

- **6DoF: a failure for the operator.** The overlay drifted until it left the field of
  view, and the operator described a freeze. Unity itself held 26.5 fps throughout, so
  the app did not freeze; the overlay stopped being corrected.
  - **The camera died 11 s into the run.** With a burst every 1.2 s, the camera was
    restarted 0.18 s after a stop. Burst 3 delivered no frames. From then on every start
    failed: 138 attempts, `xreal_error Failure NativeRGBCamera+Start`.
  - So only two locks ever arrived, and 6DoF drifted uncorrected for three minutes.
  - Those two locks show the drift again: the head "moved" 9 cm in 2 s while plate A
    stayed 0.723 m from the camera.
- **3DoF: much better.**
  - The camera survived (128 bursts, one empty), at a lock every 1.43 s (round trip
    70 ms).
  - Standing still, corrections were a median 2.6 mm (max 20 mm): per-lock noise of a
    few mm.
  - Frozen and standing, it drifted 6.8 mm.
  - Stepping produced the expected 0.4-1 m jumps (3DoF has no position), each
    confirmed and applied at the next lock.
- **Operator:**
  1. the corrections came too slowly;
  2. the grid was steady in space but not on the plate as seen by the eyes. The camera
     sees from a different point than the eyes, and the display model is generic.

### Rev. 3 (build `2026-10-22T11:25:28Z`)

- **Stream camera mode** (default for G5, the demo and align; `-camMode burst` restores
  bursts).
  - The camera stays on. The newest frame goes to the service whenever it is free and
    the head is nearly still, up to 8 per second; about 70 ms per frame was measured.
  - The camera restarts when its frames fall 0.3 s behind (the SDK decoder queue, §16)
    or every 8 s.
  - The overlay is corrected several times a second instead of every 1.4 s.
  - **If the CPU clock stays below 110 % for 10 s**, the gate switches to bursts and
    says so. Continuous decoding is what heated the Surface in G3.
- **Camera rest and recovery.**
  - At least 0.6 s between a stop and the next start.
  - After a start that fails, or a start with no frames, it backs off 2, 4, 8 s, up to
    10 s, and retries. The HUD shows `CAMERA RESTARTING`.
  - Default burst period 1.5 s.
- **View offset** (head frame, saved).
  - The disagreement between where the camera puts the target and where the wearer sees
    the drawing is a small rigid error fixed to the head. Its sources are the vendor's
    RGB-camera-from-head pose and the display model against this wearer's eyes.
  - The new `ViewOffset` sits between the head and the camera: a translation in mm plus
    a rotation in degrees. The wearer sets it by eye and it is saved to
    `%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\calibration\view_offset.json`.
  - G5, the demo and align load it. Because it lives in the head frame it stays valid
    when the wearer moves.
  - Keys, in any locking gate:

    | Key | Action |
    |---|---|
    | arrows | move (2 mm; Shift 10 mm) |
    | PgUp / PgDn | farther / nearer |
    | Ctrl + arrows | yaw / pitch (0.25°; Shift 1°) |
    | Ctrl + PgUp / PgDn | roll |
    | Backspace | zero |
    | Enter | save |

  - A change re-places every target at once from its last camera pose; the HUD shows
    the offset (amber while unsaved).
- **Align gate** (G5 menu 7 plate A, 8 board): 3DoF, stream, no protocol, until Esc. Ends
  with a warning if the offset was not saved.
- **Demo `D` now runs in 3DoF**, the robust choice where 6DoF drifts; G5 menu 9 is the
  6DoF demo. The demo's Eye view refreshes twice a second.
- **Fewer log lines.** Motion-skipped frames are logged at most once a second, with a
  count. One sent frame per second is kept as a snapshot for `anchor`.

### Session 0.3.002

Same room, plate A first.

1. **5 → 7, ALIGN on plate A.**
   - Stand on A, about 70 cm from the plate, look at it and hold still.
   - Move the drawing with the arrows until the points sit on the printed crosses: first
     left/right and up/down, then PgUp/PgDn for depth. Check by moving the head a little.
   - Press **Enter**.
2. **5 → 4, G5 plate A, 3DoF.** Does the drawing now sit on the crosses? Are corrections
   fast enough?
3. **5 → 1, G5 plate A, 6DoF.** Does the camera stay alive, and does the stream mode keep
   up with the drift?
4. **5 → 8, ALIGN on the 25 mm board.** Check that the saved offset also fits the board.
   Do not save a different one unless it is clearly better.
5. **D**, the demo.

If the offset that fits at 70 cm is wrong at 40 cm, note it: that points to a rotation
error, not a translation.

## 20. Stream-mode runs (22 Oct evening), the jitter, and the lock-filter comparison (0.3.003)

### What ran (build `2026-10-22T11:25:28Z`)

Align twice, G5 3DoF and G5 6DoF, all on plate A.

- **Stream mode worked.** The camera stayed alive in all four runs and restarted on lag
  about every 4 s. Corrections came at **7.4 Hz** (6 Hz in 6DoF), against 0.7 Hz before.
- **Align.** The first run saved x +6, y −18 mm. The second reset it and saved zero,
  which G5 6DoF then used. The operator: align works, but the offset was not the
  problem.
- **Operator: G5 3DoF and 6DoF now jitter constantly.**
  - In the 3DoF still phase each correction moved the drawing a median 3.2 mm and 0.6°
    (p90 9 mm and 1.5°), seven times a second.
  - The camera-derived world position scatters about 5 mm at the plate centre and
    10 mm at its corners. About 4 mm of that is frame-to-frame noise from a small target
    (markers about 16 px). A slow ~7 mm wander remains after any low-pass filter: a
    static, view-dependent registration error.
  - In 6DoF the slow part is the drift, which must be followed.

### Literature, and what 0.3.003 does with it

Jitter against lag is the classic dynamic-registration trade-off: Azuma 1997; Holloway
1997; Azuma & Bishop 1994 for optical see-through.

- **Temporal filtering.**
  - One Euro filter (Casiez, Roussel & Vogel, CHI 2012): a speed-adaptive low-pass.
  - Double exponential smoothing (LaViola 2003).
  - Kalman filtering (Welch & Bishop, SCAAT 1997).
- **Anchors with hysteresis.** As in ARKit and ARCore: hold the pose, correct only when
  the error clearly exceeds the noise.
- **Static error is calibration**, not filtering: SPAAM (Tuceryan & Navab 2000); survey
  by Grubert et al., IEEE TVCG 2018.

0.3.003 implements the first two and compares them.

- **Three lock filters run side by side** on every camera pose (`LockAnchor.cs`). Common
  rule: a jump over 10 cm needs a second, agreeing pose; it then snaps.
  - `latest`: the 0.3.002 policy, kept as the baseline.
  - `euro`: One Euro on position and rotation. Minimum cutoff 0.1 Hz; β 2 per m/s and
    0.015 per °/s; derivative cutoff 1 Hz.
  - `hysteresis`: an exponential estimate (τ 0.6 s) of the measurements. The anchor
    moves to it only when it stays more than 5 mm or 0.6° away for 3 consecutive poses.
- **Measurement weight:** more points and lower reprojection RMS count for more.
- **One filter is shown, gliding.** The shown pose glides every frame towards it (time
  constant 0.05, 0.06 or 0.3 s), instead of stepping at the camera rate.
- **Keys.** `F` switches the shown filter, `]` makes it smoother, `[` more responsive;
  the HUD names the filter and its setting.
- **Logging.** Every pose writes a `filters` event: the measurement and all three
  outputs. The lock event carries the target's image position and extent (`cx_px`,
  `cy_px`, `span_px`) for the static-error study.

### Offline comparison on the 22 Oct data (`python -m xreal_gates filters RUN`)

The tool replays the measurements through Python ports of the filters, including the
glide.

- **jitter:** path speed of the shown plate corners, mm/s.
- **error:** distance from the ±1 s median of the measurements, mm.
- **rev3:** the 0.3.002 behaviour.

| Step | rev3 | euro | hysteresis |
|---|---|---|---|
| 3DoF still, jitter | 13.1 mm/s | **4.2 mm/s** | **3.5 mm/s** |
| 3DoF still, error | 3.0 mm | 3.2 mm | 5.1 mm |
| 3DoF "live" (after the frozen step), jitter | 18.4 mm/s | 6.9 mm/s | 6.5 mm/s |
| 3DoF "live", error | 3.9 mm | 5.2 mm | 7.2 mm |
| 6DoF still, error | 12.5 mm | 14.5 mm | 25.6 mm |

- **The defaults were chosen from a parameter sweep** so that the two candidates shake
  about equally while still; the comparison is then decided by accuracy and feel.
- **The sweep shows the trade-off.** Euro at 0.05 Hz without β is calmer (1.9 mm/s) but
  5 mm off, and lags drift (36 mm in 6DoF). Hysteresis at 8 mm holds 7 mm off.
- **6DoF:** drift is followed best by the faster settings; hysteresis lags it most.

### Comparison subgate (G5 menu `A` 3DoF, `C` 6DoF; `-compare`)

**Blind.** The three filters get random letters (`METHOD A/B/C`, logged as
`compare_mapping`), so the wearer does not know which is shown.

**Blocks.** Each of six blocks shows one method:

- 20 s holding still;
- 10 s moving the head slowly;
- 6 s to **rate it 1-5 on the keyboard** (1 unusable, 3 acceptable, 5 rock steady;
  `rating` events).

Round 2 repeats round 1 in reverse order. About 4 min with the acquire step.

`python -m xreal_gates filters RUN` reports both:

- the objective scores for all three filters, computed on the same data whichever was
  shown;
- the mean rating per method.

### Session 0.3.003

1. **5 → A**: compare in 3DoF. Rate every block, even when unsure.
2. **5 → C**: compare in 6DoF.
3. **5 → 4 or 5 → 1** with the winner. If needed, tune it live with `[` and `]`; the
   setting is logged.

## 21. Why the offset did not carry over, alignment inside the run, and the menu cleanup (0.3.004)

### Operator report (0.3.003, no logs of this short test)

- With a correct alignment in the Align gate, the following comparison gate showed a
  big offset. With no offset, the comparison was only a little off.
- In the blind comparison, methods A and B looked rock solid most of the time and C
  wiggled.

### Is the calibration the same in every gate? Yes

- **Code.** Align and G5 shared one path: `StartLocking`, `WorldPose`, the same view
  offset loaded at the same point.
- **Logs.** `python -m xreal_gates chain` on the 22 Oct Align and G5 3DoF runs reports
  the chain **identical**: lens calibration (hash 8d239fe5), RGB-camera-from-head pose,
  tracking mode, render rate, camera mode and target file.
- **The offset is repeatable within Align.** It reached x +6, y −18 mm in two independent
  runs (about 1.4° at 0.77 m), with the plate in nearly the same part of the view as in
  G5 (within 3-4°).

So the two gates compute the same thing, and what changes is not in the logs: the
**eye-to-display geometry**. This is a hypothesis; nothing logged can show it directly.

- How the glasses sit on the nose.
- Posture while reaching for the keyboard.
- The app closing and restarting between the two gates, with the glasses possibly
  touched.

A shift of about 1.4° is small for an eyewear fit. The optical see-through literature
treats this as the known validity limit of a display calibration under HMD slippage
(Grubert et al. 2018).

### What 0.3.004 changes

- **Alignment is a step inside every G5 run** (`AlignStep`), so it uses the same process,
  posture and glasses position as the test.
  1. The run locks on the target.
  2. Untimed: align the drawing with the keys, starting from the saved offset. **Enter**
     saves and continues.
  3. 6 s hands-off check.

  The keys stay active for the whole run, and every change is logged with its time. The
  run warns if the offset ends unsaved.
- **The Align gate is removed.** The demo keeps the keys but has no blocking step, since
  guests come and go.
- **`python -m xreal_gates chain RUN...`** prints each run's chain and states whether it
  is identical across runs.
- **`python -m xreal_gates filters`** adds a mean over all `_still` blocks per method.

### Menu

| Key | Entry |
|---|---|
| D | demo |
| C | calibrate the Eye |
| 1 | G5 plate A, 3DoF |
| 2 | G5 plate A, 6DoF |
| 3 | G5 25 mm board, 3DoF |
| 4 | G5 plate L, 3DoF |
| 5 | compare filters, 3DoF |
| 6 | compare filters, 6DoF |
| K | cool-down |
| O | open run folder |
| Z | zip runs |
| X | older gates |
| Q | quit |

Older gates (X) keeps, unchanged, for repeat measurements: G1, G2, the G3 and G4
variants, the other G5 combinations, the 6DoF demo, manual and the vendor sample.

### Session 0.3.004

1. Glasses on, and do not touch them from here on. Run **C** only if the glasses or the
   camera changed; the 21 Oct calibration stays valid.
2. **5**: compare in 3DoF. Align, Enter, hands-off check, then rate every block.
3. **6**: compare in 6DoF, the same way.
4. **Z**. Then on the desktop:

   ```
   python -m xreal_gates import
   python -m xreal_gates chain <run5> <run6>
   python -m xreal_gates filters <run5> <run6>
   ```

The result to look for: the same offset in both runs would mean the glasses stayed put; a
different one, with an identical chain, would support the eye-to-display explanation.
If it holds, the next step is a SPAAM-style several-point alignment.

## 22. Compare runs of 1-2 Nov, the 160 ms frame age, and 0.3.005

### What ran (build `2026-11-01T14:25:10Z`)

Plate A: compare 3DoF, compare 6DoF, demo. Also the 1 Nov tests on the older build.

- **Operator: 3DoF super stable but blind to position; 6DoF less stable but in all six
  DoF.**
- **The 3DoF compare was not a fair test.**
  - The overheating fallback switched it to bursts 40 s in: the CPU clock was 83-97 %
    on a cold Surface (36 °C) under light load. Updates were then about 2 Hz.
  - Its ratings (3.5 for every method) and scores do not separate the methods.
- **The blind order was identical in both runs** (euro = A, latest = B, hysteresis = C,
  order 1,2,0,0,2,1). `new System.Random()` is not a trustworthy seed.
- **The 6DoF compare** ended at 289 s, without the app's quit sequence.

  | 6DoF, still blocks | latest | euro | hysteresis |
  |---|---|---|---|
  | jitter (shown corners) | 11.8 mm/s | 3.1 mm/s | 3.0 mm/s |
  | error | 1.5 mm | 1.9 mm | 3.0 mm |
  | rating | 2.5 | 3 | 4 |

- **The in-run alignment was consistent:** y −16 mm in both runs, x +8..+10 mm. Depth
  varied (−14 and 0 mm): it is hard to judge by eye.
- **The headset's 6DoF position noise while still is about 0.6 mm rms**, not the
  dominant error.

### The frame age

The frame was paired with the head pose at sending. Re-chaining the target with the head
pose L earlier, `python -m xreal_gates latency RUN`, minimises the lock-to-lock jumps
while turning at **L = 140-200 ms** in all five stream-mode runs (3DoF and 6DoF):

| | L = 0 | L ≈ 160 ms |
|---|---|---|
| lock-to-lock jump while turning | 10-11 mm | 5-7 mm |
| lock-to-lock jump while still | 2.1 mm | 1.6 mm |

**For comparison, the AR latency literature:**

- Motion-to-photon about 20 ms or less (Oculus guidance; recent head-mounted AR work).
- Detection thresholds down to about 3 ms in sensitive users (Jerald & Whitton, IEEE VR
  2009; Ellis & Adelstein).
- Optical see-through is stricter than VR because reality has no delay (Azuma 1997;
  Lincoln et al., TVCG 2016).

**What that means here.** The 160 ms is not the gate's motion-to-photon delay: XREAL
renders head motion from its own tracking. It is the age of the camera correction.

- For a fixed target it is harmless once each frame is paired with the head pose of its
  capture, which 0.3.005 does.
- It matters for anything that moves, and it limits how fast and how often the camera can
  correct the drift of the head tracking.
- The operator's assessment is that head tracking alone cannot hold millimetres, so
  reducing it is the work of 0.4.000 (`docs/tasks/T5.1-eye-latency.md`).

### Changes in 0.3.005 (build `2026-11-03T09:26:40Z`)

- **Frame age compensation.**
  - The head pose is taken `cam_latency` before sending (default 160 ms, `-camLatency s`,
    keys `,` `.` ±10 ms), interpolated from 1 s of head history.
  - Logged as `cam_latency`; `chain` reports it.
  - `latency RUN` on a 0.3.005 run estimates the residual on top of it.
- **Motion gate relaxed to 45°/s and 0.4 m/s,** so corrections continue during normal
  head motion.
- **Overheating fallback only when hot.**
  - Hot means the hottest zone ≥ 65 °C for 10 s; without temperatures, a clock below
    110 % at ≥ 60 % system load.
  - It returns to streaming after 30 s below 60 °C (or clock ≥ 180 %).
- **Blind order seeded from a GUID,** with the seed logged in `compare_mapping`.
- **Alignment keys are angle first.**

  | Key | Action |
  |---|---|
  | arrows | yaw / pitch (0.1°; Shift 0.5°) |
  | Ctrl + arrows | shift (2 mm; Shift 10 mm) |
  | Ctrl + PgUp / PgDn | roll |
  | Alt + PgUp / PgDn | depth |

  A fixed tilt of the display against the eyes is an angle, and holds at every distance.

### Session 0.3.005

Glasses on and untouched; furnished, evenly lit, static room; textured background behind
the plate.

1. **5**: compare in 3DoF. It now stays in stream mode: check that the HUD never shows
   "SURFACE HOT" on a cool Surface.
2. **6**: compare in 6DoF.
3. **Z**. On the desktop, run `chain`, `filters` and `latency` on both runs.

## 23. Compare runs of 4-6 Nov (0.3.005): the frame age is a sawtooth, not a constant

### What ran (build `2026-11-03T09:26:40Z`)

Plate A, Surface with fan:

- 4 Nov: demo in 3DoF (46 s, then 12.6 min).
- 6 Nov: compare in 3DoF (299 s), then compare in 6DoF (432 s).

All four completed normally.

- **No camera death.**
- **No overheating fallback:** the clock stayed at 219 % in both G5 runs.
- **`chain`: identical** across the two compare runs (lens fx 623.9, RGB-from-head, 30 fps,
  stream, plate A measured target).

### Lock filters

| still blocks (mean) | latest | One Euro | hysteresis |
|---|---|---|---|
| 3DoF jitter / error | 9.3 mm/s / 1.0 mm | 5.2 mm/s / 1.4 mm | 4.9 mm/s / 2.6 mm |
| 3DoF without block cmp3 | 5.2 mm/s | 1.3 mm/s | 0.9 mm/s |
| 3DoF rating (1-5) | 4.0 | 3.5 | 4.0 |
| 6DoF jitter / error | 3.3 mm/s / 0.7 mm | **1.2 mm/s / 1.3 mm** | 1.1 mm/s / 2.7 mm |
| 6DoF rating (1-5) | 2.5 | **4.0** | 3.5 |
| 6DoF, 0.3.004 (2 Nov) | 11.8 mm/s / 1.5 mm | 3.1 mm/s / 1.9 mm | 3.0 mm/s / 3.0 mm |

Block cmp3 of the 3DoF run (24-30 mm/s for every method) is the operator moving, not a
filter effect.

- **6DoF improved about threefold** over 0.3.004 for every method. The change is the
  capture-time head pose (`cam_latency`) and the relaxed motion gate.
- **One Euro stays the default.**
  - It ties hysteresis on jitter but has half its error: hysteresis holds a pose up to
    5 mm stale.
  - It is rated best in 6DoF, the mode EyeLab needs.
- **Latest is acceptable only in 3DoF.**

### Alignment: the view offset is an angle

- **The 3DoF run kept the saved offset:** x +8, y −16 mm, aligned in 43 s.
- **The 6DoF run re-aligned from the angle keys** (182 s, 90 key presses) and saved yaw
  +0.5°, pitch +1.2°, with no shift.
- **At the median plate distance of 0.73 m these are the same correction:** 6.4 mm right,
  15.3 mm down.
- So the offset is a boresight of about 1.3° between what the Eye sees and what the
  operator's eye sees through the display. Keep it in angle form: an angle holds at every
  distance, a shift only at the distance it was set.

### Coverage: every stream restart costs about 2 s without a lock

- **The decoder does not keep up even on a cool Surface at full clock.**
  - The Eye produces 29.8 fps; Unity receives 16-21 fps (`perf.csv` `cam_fps`).
  - So the SDK queue grows 0.07-0.11 s per second of streaming.
  - The stream is restarted when it is 0.3 s behind: every 4.5-6 s.
- **Each restart:** about 0.8 s off, then about 1.05 s to the first lock (median). The
  lock-to-lock gap across a restart is 2.0 s.
- **In the still blocks, 28 % (3DoF) and 42 % (6DoF) of the time lies in lock gaps longer
  than 1 s.**
  - That is the largest loss of correction in the current pipeline, larger than the
    detection time.
  - Gaps in the move blocks (about 80 %) are expected: the plate leaves the view.
- **Pose service:** detection median 51-54 ms (p90 79-87); round trip 65-73 ms (p90
  91-103).
  - About 4.8 frames/s are sent, below the 8 Hz cap: one request in flight at a time.
  - Median 3 markers, 20 points, 0.50 px RMS, plate at 0.45-0.95 m.

### The frame age is not constant

`python -m xreal_gates latency` now also reports a per-frame age.

- **The queue age** is `t_host − cam_ts` (service receipt time minus device capture clock)
  above its rolling ±20 s minimum. The minimum is the freshest frame. The two clocks drift
  only about 20 ms in 5 min, and the rolling minimum absorbs that.
- **Inside every stream window the queue age ramps from 0 to about 0.3-0.48 s.** Median
  150-180 ms, p90 about 320 ms.
- So the fixed `cam_latency` of 160 ms was the average of a sawtooth, wrong by up to about
  ±150 ms per frame.

Each frame is paired with the head pose `base + queue age` before sending, scanning
`base`:

| run | fixed delay, best | per-frame age, best |
|---|---|---|
| 2 Nov 6DoF (0.3.004) | 4.8 mm at 180 ms | **2.6 mm** at base 70 ms |
| 6 Nov 3DoF | 5.4 mm at 300 ms | **3.0 mm** at base 80 ms |
| 6 Nov 6DoF | 5.0 mm at 310 ms | **2.6 mm** at base 70 ms |

The figures are the median lock-to-lock jump while turning. The jump while still drops too,
from 1.2-1.6 to 0.9-1.0 mm.

- The 2 Nov 3DoF run fell back to bursts and is not comparable.
- **Findings:**
  - The freshest frame is about 70-80 ms old when sent, and the base is the same in all
    three runs.
  - Per-frame pairing removes about 45 % of the turning error. It needs no new
    measurement: `CameraProbe` already computes `t − cam_ts` per frame (`LagGrowth`).
- **What this changes for the latency work.** For a fixed structure, the age of a correction
  costs little once it is known exactly: SLAM drift over 0.3 s is about 1 mm standing.
  What costs accuracy is:
  - **a wrong age:** fixed by the per-frame age;
  - **blind time:** the restarts;
  - **the decoder's throughput.**

  `docs/tasks/T5.1-eye-latency.md` is reordered accordingly.

### Overheating: the zone temperature does not see the throttle

- **In the 12.6 min demo the clock fell** from 210 % to 130 % (min 84 %) in the last
  third. Received camera frames fell from 13 to 7 fps.
- **The hottest thermal zone stayed at 52 °C,** under the 65 °C rule, so the fallback
  never triggered. When temperatures exist the rule never looks at the clock.
- This was a missed fallback, not a false one.
- The signal that matters is received camera frames per second and queue growth, not a
  zone temperature. See T5.1.

### Other observations

- **6DoF reported InsufficientLight 8 times** (the room at 14:51); 3DoF is not affected.
- **Most failed frames are "no plate marker"** (plate out of view) and "1 marker, need 2"
  (plate at the edge of the view).
- **Unity renders at 26.5-27 fps** against the 30 fps target in every run.

## 24. T5.1 toward 0.4.000: per-frame frame age (Phase 0, 0.3.007), stage timing (Phase 1, 0.3.008)

`docs/tasks/T5.1-eye-latency.md` is worked phase by phase; this section collects each phase's change
and its Surface results, and becomes the 0.4.000 write-up at Phase 4.

### Phase 0: each frame paired with its own age (build `2026-11-07T17:03:44Z`, EyeLab 0.3.007)

**What changed in the gate player.**

- **`CameraProbe`** keeps, for every unique frame, `D = t_arrived − cam_ts·1e-9` and a causal
  20 s minimum of `D` (a monotonic deque, carried across stream restarts). The frame's
  **queue age** is `D − D_min`: how far behind the freshest frame of the last 20 s it arrived.
- **`TrySendLockFrame`** pairs the frame with the head pose
  `age = base + queue age + hold` before sending, where `hold = now − t_arrived` is the time the
  frame waited in Unity. `base` is `-camLatency` (now default **90 ms**) and keys `,` `.` (±10 ms).
  - Equivalently the capture instant is `cam_ts` mapped to the host clock through the freshest
    frame's offset, minus `base`: the base is what happens before arrival of a fresh frame
    (exposure, on-glasses encode, USB, FFmpeg decode, texture upload).
  - With no queue age (frame not the probe's latest) the age falls back to `base` alone
    (`age_src=base_only` in the lock line).
- **Head history 2.0 s** (was 1.0 s), so frames up to about 1.9 s old still find a pose. A pose
  older than the history uses the oldest sample and is counted (`head_clamped`, per lock and in
  `lock_summary`).
- **Logged per lock** (`lock` events): `t_send` (exact), `age_ms`, `base_ms`, `queue_ms`, `hold_ms`.
  `cam_latency` now logs the base; `chain` shows it as "base N ms + per-frame queue age".
- **HUD:** `frame age 90+Q ms` with Q the current queue age.

**Why 90 ms and not the 70-80 ms of §23.** The §23 base was measured against the pose service's
receipt time (`t_host`), which already includes our copy and send. Unity measures from the frame's
arrival and adds the hold separately. `latency` now replays the Unity estimator on old runs from
`camera.csv` (every arrived frame) and finds the best base at 85-100 ms:

| run (0.3.004-0.3.005) | as run | service queue age, best | Unity estimator, best |
|---|---|---|---|
| 2 Nov 6DoF | 12.0 mm (age 0) | 2.6 mm at 70 ms | **1.9 mm** at 90 ms |
| 6 Nov 3DoF | 6.9 mm (fixed 160 ms) | 3.0 mm at 80 ms | **2.4 mm** at 100 ms |
| 6 Nov 6DoF | 5.7 mm (fixed 160 ms) | 2.6 mm at 70 ms | **1.5 mm** at 85 ms |

Median lock-to-lock jump while turning; the jump while still drops to 0.8-0.9 mm. The Unity
estimator does better than the service-side one because the arrival time is not blurred by the
copy, the send and the one-Unity-frame delay of the old `t_send` estimate. (The "as run" figures are
from today's `latency`, which slerps the head pose; T5.1's baseline table quotes 5.0-6.2 mm from an
earlier variant. The Phase 0 gate, ≤ 3 mm, is unchanged.)

**`python -m xreal_gates latency RUN`** now scans three pairings: `logged` (what the run did: age 0,
a fixed `cam_latency`, or from 0.3.007 each lock's `age_ms`), `service` (§23) and `unity` (the
estimator above, rebuilt from `camera.csv`). On a 0.3.007 run the `logged` residual should be within
about ±10 ms of 0 and `unity` should agree with `logged` at the logged base.

### Session 0.3.007 (Phase 0 gate)

Same set-up as §22: glasses on and untouched; furnished, evenly lit, static room; textured
background behind plate A; fan on the Surface; launcher shows `[OK]` for the new build.

1. **5**: compare in 3DoF. Align as usual, then rate the blocks. In the move blocks, turn the head
   at a normal pace while keeping the plate in view for part of each turn.
2. **K** until "Cool enough".
3. **6**: compare in 6DoF.
4. **Z**, and bring the zip to the desktop (`GateRuns/_inbox`).

Do not touch `,` `.` in these runs (the base must stay 90 ms for the comparison).

**Done when** (T5.1 Phase 0), against the baseline: the median turning jump as run ≤ 3 mm in both
runs, and the One Euro still-block error no worse than 1.3 mm (6DoF). Desktop:
`python -m xreal_gates import`, then `latency`, `filters` and `chain` on both runs.

### Phase 0 results (runs of 7 Nov, build `2026-11-07T17:03:44Z`)

Five G5 compare runs, two wearers besides Jack. `latency` (0.3.008: "still" also requires a head
speed ≤ 0.03 m/s, so a wearer drifting sideways no longer counts as still):

| run | DoF | turning jump as run | still | best extra delay | logged age, median (p10-p90) |
|---|---|---|---|---|---|
| 181513 | 3DoF, short (n = 26) | 2.5 mm | 1.5 mm | +20 ms | 228 ms (122-359) |
| 181714 | 3DoF | **2.1 mm** | 1.1 mm | +20 ms | 250 ms (128-377) |
| 182425 | 6DoF | **2.3 mm** | 1.0 mm | +20 ms | 223 ms (118-371) |
| 183223 | 3DoF, aborted (throttled) | 2.0 mm | 1.0 mm | +10 ms | 224 ms (111-389) |
| 183833 | 3DoF, aborted (throttled) | 2.9 mm | 1.2 mm | +10 ms | 233 ms (121-380) |

Baseline (6 Nov, same tool): 6.9 mm (3DoF), 5.7 mm (6DoF).

- **Turning jump: passed.** It is 2.0-2.9 mm in every run, at or below the 3 mm gate.
- **The logged age leaves a residual of +10 to +20 ms.** The Unity estimator replayed on the same
  runs puts its best base at 100-110 ms. **0.3.008 sets the base to 105 ms.**
- **Still error, 6DoF: not yet shown.**
  - One Euro still-block error was 3.9 mm in 182425 (gate: ≤ 1.3 mm); latest 2.0 mm, hysteresis 6.5 mm.
  - It is not caused by Phase 0. Re-pairing the same run with the old fixed 160 ms gives the same
    scatter: 7.2 mm against 7.3 mm.
  - The cause is the wearer. The head moved about three times more in the still blocks than in
    the 6 Nov run.
  - The 3DoF run 181714 is at 1.6 mm (One Euro), and its still scatter is 2.83 mm with the
    per-frame age against 3.14 mm re-paired at a fixed 160 ms.
  - **So this part of the gate is repeated in the 0.3.008 session**, with Jack holding still.
- **Other.**
  - 183223 and 183833 were aborted while the Surface throttled (clock 152 % and 112 %; camera 8-13 fps).
  - 182425 had 72 stream windows at 14 fps.
  - Alignment took 91-144 s and up to 165 key presses. One tester set it as a shift (x +40,
    y −12 mm), another as a yaw (−5.2 deg).
  - Ratings: One Euro 4.5 (3DoF) and 3.5 (6DoF), the best or tied best in both runs.

### Tester feedback (7 Nov) and what 0.3.008 changes

1. **"Too much writing in view."** The locking gates (G5, demo) now show a compact banner: the
   title, a countdown and one short grey line, with a thin amber frame for a moment when it changes.
   Removed from view:
   - the gate header and "next";
   - the tracking line while tracking is fine;
   - the lock line while locked;
   - the offset line, except while aligning or unsaved;
   - the diagnostics block.

   **H** brings them all back.
2. **"Rotate on x/y/z when aligning."** **Tab** switches two modes:
   - **MOVE:** arrows move the drawing left/right/up/down. This is stored as a boresight yaw/pitch,
     0.1 deg, Shift 0.5 deg: §23 showed the offset is an angle. PgUp/PgDn move it farther/closer
     (2 mm, Shift 10 mm).
   - **ROTATE:** the drawing turns about its own centre (the mean of the target's drawn points),
     about the head's axes. Left/Right turn it, Up/Down tip it, PgUp/PgDn roll it (0.2 deg, Shift 1 deg).
     - With the offset `x' = R x + t` and the shown centre `c' = R c + t`, a step `dq` sets
       `R' = dq R` and `t' = dq (t − c') + c'`, so the centre stays put.
     - The result is stored back as yaw/pitch/roll and x/y/z, in the same `view_offset.json`.

   Keys repeat while held (after 0.35 s, every 60 ms). Backspace zeroes, Enter saves. The old Ctrl/Alt
   combinations are gone. Each change is logged with its mode (`view_offset CHANGED move: ...`, `align_mode`).
3. **"Move your head slowly" is too generic.** Every move block is now a sequence of named
   movements of 4 s each, shown as the title, spoken, and logged as a `cue` event inside the
   unchanged step id:
   - **3DoF:** turn left and right, look up and down, tilt the head left and right.
   - **6DoF:** the same, then slide left and right, rise and lower, lean closer and back.

   The compare move blocks are 12 s (3DoF) and 24 s (6DoF), up from 10 s. The G5 `move_head` step
   uses the same cues. The step ids, and so `filters` and `latency`, are unchanged.
4. **"Make the drawing smaller so the target shows."**
   - Target lines are 1.2 mm (were 2.5 mm) and points 5 mm (were 8 mm).
   - **V** cycles lines and points / points only / lines only.
   - The demo grid is 40 × 30 cm with 2 mm lines (was 60 × 40 cm, 4 mm).

### Phase 1: every stage on one clock (build `2026-11-09T11:35:30Z`, EyeLab 0.3.008)

No behaviour change except the base (105 ms) and the feedback items above. Everything below is
logging.

**The clock is the Windows performance counter (QPC).** Unity reads it as
`Stopwatch.GetTimestamp()/Frequency`. The pose service reads it as `time.perf_counter()`, which on
Windows is the same counter in the same unit: on the desktop the two agree to 1 µs. So Unity and
service times subtract directly.

> **Correction (10 Nov, from the 9 Nov Surface runs):** in the player, Mono's `Stopwatch` counts from
> process start, not boot, so 0.3.008's Unity times sit a constant offset behind the service's. The
> HMD clock *is* boot QPC in ns. `stages` shifts 0.3.008 runs, and 0.3.009 reads
> `QueryPerformanceCounter` directly (§25).

**`frame_timing.csv`**: one row per frame sent to the service, posed or not. It is written at the
start of the Unity frame after the one that applied the result.
`python -m xreal_gates stages RUN` turns it into this table:

| stage | from → to | how |
|---|---|---|
| poll interval | `q_frame_prev` → `q_frame` | Unity frame starts around the arrival. The vendor texture polls `TryGetRGBCameraFrame` once per Update, so the SDK had the frame somewhere in this interval |
| texture | `q_frame` → `q_arrived` | frame start to our `FrameArrived` handler (vendor `TryAcquireLatestImage` and three texture uploads) |
| hold | `q_arrived` → `q_copy0` | waiting in Unity until the sender takes it |
| copy | `q_copy0` → `q_copy1` | Y plane out of the texture |
| send | `q_send0` → `q_send1` | building and sending the WebSocket message |
| to service | `q_send1` → `q_recv` | service receipt (`pose_service.csv` and the reply carry `q_recv`, `q_det0`, `q_det1`) |
| parse | `q_recv` → `q_det0` | frame packet built |
| detection | `q_det0` → `q_det1` | all targets estimated |
| reply | `q_det1` → `q_reply_recv` | reply back in the bridge |
| apply wait | `q_reply_recv` → `q_applied` | until the next Update applies it |
| display | `q_applied` → `q_shown` | to the next Unity frame start (rendered and presented in between) |

The table also gives the totals copy → shown and frame age + copy → shown. The row also holds the
frame's `age_ms`, `queue_ms`, `hold_ms`, `base_ms` and `ok`.

**Native probes (read-only).** Two exports of `XREALXRPlugin.dll` that the C# wrapper does not
bind are called through `DllImport`. The vendor binary is not modified. The signatures were read
from the x64 export code:

- `bool GetHMDTimeNanos(uint64_t*)`: the HMD clock. It is logged at every frame's arrival
  (`camera.csv` `hmd_ns`), at every send (`hmd_send_ns`), and once per second in `clock.csv`, next to QPC.
  - **If `cam_ts` is on this clock,** `hmd_ns − cam_ts` is each frame's exact age at arrival,
    and its minimum is the true base. That would replace the fitted base.
  - `stages` checks this: small, positive and slowly varying means the same clock.
- `bool GetHeadPoseAtTime(uint64_t time_ns, float[16])`: a 64-byte pose matrix, called at HMD-now
  (`native_now`) and at `cam_ts` (`native_cap`). `stages` finds which reading of the 16 floats
  matches Unity's head rotation (`unity_head_now`), using relative rotations between rows.
- **Every call is wrapped.** The first exception disables that call for the rest of the run;
  `native_clock` logs how many calls succeeded.

**`camera.csv`** gains `qpc` (arrival), `hmd_ns`, `q_frame`, `q_frame_prev`.

**Optical timecode (launcher T).**
- A fullscreen marker on the Surface screen, whose id is a draw counter mod 1000
  (`python -m xreal_gates timecode`). Each draw is logged with its QPC time to
  `GateRuns/timecode/timecode_<stamp>.csv`; **Z** exports it and `import` files it.
- G5 then runs in 3DoF on the target `timecode`. The service reads the id in every frame it is
  sent, and there is no alignment and no overlay.
- `stages` pairs each frame's id with the latest draw of that id before the frame's arrival.
  **Arrival minus draw is the camera-to-app latency, independent of the head.**
- It includes the Surface panel's own presentation delay (one or two refreshes), so it is an upper
  bound on exposure → arrival.
- Protocol (80 s): look at the screen 10 s, hold still 40 s, the three 3DoF movements, hold still 20 s.

### Session 0.3.008 (Phase 1 gate, and the repeat of Phase 0's still check)

Same set-up as §22. Launcher shows `[OK]` for build `2026-11-09T11:35:30Z`. Jack wears the
glasses for 1-3; other testers may use the demo afterwards.

1. **T**: timecode. Lay the Surface flat or tilt it about 50 cm in front of you, screen towards
   you, brightness high. Look at the black and white square for the whole run.
   - If the app window covers it, click the square once.
   - The run ends by itself (about 2.5 min with XR start). Then **K** until "Cool enough".
2. **5**: compare in 3DoF.
   - Align with the new keys: Tab, MOVE first and then ROTATE if the drawing is tilted, Enter.
   - In the still blocks hold really still, standing, eyes on the plate. In the move blocks
     follow the spoken movements.
   - Then **K**.
3. **6**: compare in 6DoF, same as 2. This repeats Phase 0's still check.
4. Optional: **D** with the other testers, to judge the compact display, Tab alignment and V
   styles. No metrics.
5. **Z**, and bring the zip to the desktop (`GateRuns/_inbox`).

Do not touch `,` `.` (the base must stay 105 ms).

**Done when.**
- **Phase 1:** `python -m xreal_gates stages` lists every stage's median and p90 over the two
  compare runs (stream mode, more than 3 min each). The timecode run gives camera → arrival.
  The native-clock lines say whether `cam_ts` is on the HMD clock.
- **Phase 0, still part:** the One Euro still-block error in the 6DoF run is ≤ 1.3 mm.
- Desktop: `python -m xreal_gates import`, then `stages`, `latency`, `filters` on the three G5 runs.

## 25. Session of 9 Nov (0.3.008): the camera timestamp is the PC clock; T5.1 Phase 2 begins (0.3.009)

### What ran (build `2026-11-09T11:35:30Z`, EyeLab 0.3.008)

Three gate runs, then two demo runs.

| run | what |
|---|---|
| `20261109_144141_G5_g5_timecode_3dof` | timecode, 183 frames sent, 23 with the marker read |
| `20261109_144348_G5_g5_plateA_3dof_compare` | compare, 3DoF, 1786 frames sent, 1307 measurements |
| `20261109_145026_G5_g5_plateA_compare` | compare, 6DoF, 1469 frames sent, 857 measurements |
| `20261109_145700_Demo_demo_3dof`, `20261109_145754_Demo_demo_3dof` | demo, no metrics |

Jack's verdict: "all great improvements". The compact display, Tab alignment, named moves and
thinner drawing stay. **The blind letters and the second round go** (0.3.009): a tester needs to
know which method is on, and a repeat is just another run.

### The HMD clock is QPC, and `cam_ts` is on it

This is the main result of Phase 1.

- `GetHMDTimeNanos` returned true on every call (`native_clock`), and
  `hmd_ns − cam_ts` at arrival is small, positive and slowly varying: min 78-104 ms, median
  185-267 ms. **`cam_ts` is on the HMD clock.**
- **The HMD clock is the Windows performance counter, in ns since boot.** The pose service's
  `time.perf_counter()` and `cam_ts` lie on one timeline: a frame with `cam_ts` 1 654 912.34 s was
  received by the service at 1 654 912.54 s.
- **Unity's side was not.** Mono's `Stopwatch.GetTimestamp()` counts from process start in the
  player. It sat a constant 1 654 890.825 s behind the HMD clock, the same to 1 µs over a run.
  - This corrects §24, which said Unity and service QPC agree to 1 µs. That was measured in the
    Editor on the desktop, not in the player.
  - `stages` now shifts 0.3.008 runs by `hmd_arrived_ns − q_arrived`. 0.3.009 reads
    `QueryPerformanceCounter` directly (`qpc` event `ABSOLUTE`).
- **So every frame's age is known exactly:** HMD-now − `cam_ts`. No fitted base.
- **The timecode confirms that `cam_ts` is the exposure.** The marker id a frame shows was drawn
  36-68 ms before its `cam_ts` (median 44 ms, n = 23). That is the Surface's presentation delay
  (one compositor frame plus 0-17 ms to the next refresh, and the panel) plus part of a camera
  frame.
- **`GetHeadPoseAtTime` works.** Read as column-major with z flipped, it matches Unity's head
  rotation to 0.00-0.07° (n = 1469-1786). The pairing keeps Unity's own head history, which is the
  same pose.

### Stage table (Phase 1 gate)

Median / p90 in ms, from `python -m xreal_gates stages`, on the two compare runs (stream mode,
more than 3 min each).

| stage | 3DoF | 6DoF | how |
|---|---|---|---|
| **cam_ts → arrival** (exposure, encode, USB, decode queue, decode, upload) | **187 / 345** | **185 / 355** | HMD clock, exact |
| of which the minimum (no queue) | 84 | 78 | HMD clock |
| poll interval | 33.4 / 33.6 | 33.3 / 33.6 | Unity frame starts |
| texture | 0.4 / 0.5 | 0.4 / 0.5 | QPC |
| hold | 0.9 / 33.4 | 0.9 / 33.6 | QPC |
| copy | 0.2 / 0.4 | 0.2 / 0.4 | QPC |
| send | 4.7 / 6.3 | 5.0 / 7.1 | QPC |
| to service | 3.4 / 5.9 | 4.0 / 7.4 | QPC, both processes |
| parse | 1.0 / 1.7 | 1.1 / 2.0 | QPC (service) |
| **detection** | **55.7 / 89.6** | **55.3 / 98.8** | QPC (service) |
| reply | 1.0 / 1.6 | 1.0 / 1.7 | QPC, both processes |
| apply wait | 10.4 / 31.1 | 10.7 / 32.3 | QPC |
| display | 32.5 / 33.3 | 32.5 / 33.3 | QPC |
| **cam_ts → shown (total)** | **336 / 494** | **331 / 509** | HMD clock + QPC, exact |

Timecode run (3DoF, the Surface screen): cam_ts → arrival 248 / 400 ms, draw → arrival 284 / 444 ms
(n = 23 frames with the marker read; the other 160 did not read it). Detection of
the timecode marker took 18 ms, against 55 ms for the plate.

**Where the time goes.** Of the 330 ms from exposure to the overlay, about 80 ms is the fixed
camera path, **about 105 ms (median) is the SDK decode queue**, 55 ms is the plate detection and
about 45 ms is Unity's frame cadence (apply wait and display). Our copy, send and transport add
about 10 ms. **Phase 1 is done.**

### Frame age: exact age against the estimate

`latency` replays the pairing with the exact age (`hmd_send_ns − cam_ts`), trying a residual on
top:

| run | as run (estimated, base 105) | exact age, residual 0 | best residual of exact |
|---|---|---|---|
| 3DoF | 2.5 mm turning, 1.1 still | 2.0 mm, 1.0 | −20 ms: 1.8 mm |
| 6DoF | 2.1 mm, 1.1 | **1.8 mm, 1.0** | 0 ms: 1.8 mm |

The 0.3.008 estimate ran 13 ms too old (median), in line with the Unity estimator's best base of
80-90 ms on these runs. The exact age needs no fit and lands within 0-20 ms of the best. **0.3.009
uses it** (`age_src=hmd` per lock), with the estimate as the fallback if `GetHMDTimeNanos` ever
fails. Keys `,` `.` now trim the age by 10 ms (default 0).

### Lock filters (named in 0.3.009; these runs were still blind)

`filters`, mean over the still blocks (jitter mm/s / error mm):

| | latest | One Euro | hysteresis | ratings (1-5) |
|---|---|---|---|---|
| 3DoF | 6.7 / 0.8 | **1.1 / 0.8** | 0.6 / 1.3 | euro 3.5, hysteresis 3.0, latest 2.5 |
| 6DoF | 7.6 / 1.8 | 2.5 / 2.6 | 2.8 / 3.5 | euro 2.0, hysteresis 3.0, latest 3.0 |

- **3DoF beats the baseline** (One Euro 1.2 mm/s / 1.3 mm): 1.1 / 0.8.
- **6DoF does not:** the One Euro error is 2.6 mm against ≤ 1.3. The measurements themselves
  scatter twice as much (latest error 1.8 mm against 0.8 in 3DoF), so the filter is not the cause.
  The 6DoF run was **blind half the still time and the camera was starved** (next section). One
  Euro then glides between sparse, older locks.
- Phase 0's still check therefore stays open. It is repeated after the blind-time change, in the
  0.3.009 session.

### Blind time: the decoder starves when the Surface warms

Share of each block type spent in lock gaps over 1 s:

| | still | move | rate | camera fps (median, by fifth of the run) | restarts | median window |
|---|---|---|---|---|---|---|
| 3DoF 9 Nov | **22 %** (baseline 28) | 76 % | 32 % | 22-23.5 | 47 | 7.2 s |
| 6DoF 9 Nov | **50 %** (baseline 42) | 81 % | 52 % | 23 → 13 → 11 | 82 | 3.5 s |
| 6DoF 6 Nov | 42 % | 79 % | 56 % | 17-20 → 11 | 86 | 3.9 s |

- In the 6DoF run the CPU clock fell from 220 % to 190 % after the first minute. **Decoded frames
  fell from 23 to 11 per second**, the queue grew three times faster, and the 0.3 s threshold
  restarted the camera every 3.5 s. **Each restart costs about 2 s without locks.**
- The zone temperature stayed at 52 °C, so the heat rule did not fire. That matches §23.
- **The 0.3 s threshold was there for accuracy.** An older frame was paired with a wrongly aged
  head pose. With the exact age it is paired correctly, so 0.3.009 raises the threshold to
  **1.0 s** and the forced window from 8 s to **30 s** (`-streamLag`, `-streamWindow`). The head
  history (2.0 s) still covers the oldest frame.
- **The trade-off:** a 1.0 s threshold is expected to let the median frame age grow (from about 0.2 s).
  For a fixed target that is correct, but the overlay reacts later to a bumped plate. The session
  measures still-phase error and blind time against the 9 Nov runs.

### Phase 2 gates against these numbers

| gate | Phase 2 target | 9 Nov |
|---|---|---|
| still-phase time in gaps > 1 s | ≤ 14 % (3DoF), ≤ 21 % (6DoF) | 22 %, 50 % |
| median exposure → applied | ≤ ½ of Phase 1: ≤ 150 ms | 300 ms (shown 331-336) |
| still error / jitter | no worse than 1.3 mm / 1.2 mm/s | 3DoF met; 6DoF 2.6 / 2.5 |

**The two halves of the latency gate pull apart.** A higher restart threshold cuts the blind time
but lengthens the decode queue, which is the largest part of the latency. Under the current decoder
both can be met only if the decoder keeps up. That needs a cooler, lighter host, less of our own CPU
(detection is 55 ms per frame at 5-6 frames/s, about a third of a core), or Phase 3. **0.3.009 does the
blind time first, as T5.1 orders.** The detection fast path (markers and PnP first, saddles later or
only when still) is next and also frees CPU for the decoder.

### Changes in 0.3.009 (build `2026-11-10T10:02:25Z`)

- **Exact frame age.** HMD-now − `cam_ts` + trim; the fallback is base 105 + queue + hold.
  - Logged per lock (`age_src`, `trim_ms`) and in `frame_timing.csv` (`age_src`, `trim_ms`, `qpc_abs`).
  - HUD with H: "frame age N ms (glasses clock, trim 0 ms , .)".
- **Absolute QPC in Unity** (`QueryPerformanceCounter` via kernel32). Unity, service and camera
  times subtract directly. `stages` also shifts 0.3.008 runs.
- **Stream restarts:** threshold 1.0 s (was 0.3), forced window 30 s (was 8). Arguments
  `-streamLag S` and `-streamWindow S`.
- **Compare, named and once.** Each block names its method on the banner and in speech (ONE EURO,
  HYSTERESIS, LATEST), in that fixed order, one block each. Ratings stay. Step ids are
  `cmpN_<method>_still|move|rate`.
- **Demo (Jack, 10 Nov): everything starts off.** Grid, plate and board overlays and the Eye view
  are off at start; G, T and E turn them on. **F switches the lock method** (One Euro → hysteresis →
  latest), and the banner names the method and what is on. The demo always used the active lock
  filter; it now says which.
- **`stages`:** exact lines (cam_ts → arrival, → sent, → shown) and the timecode's draw → `cam_ts`.
- **`latency`:** an exact-age scan, and residuals from −60 ms.

### Session 0.3.009 (Phase 2, blind time)

Same set-up as §22. Launcher shows `[OK]` for build `2026-11-10T10:02:25Z`. Start cool (**K** first).

1. **5**: compare in 3DoF.
   - Align (Tab, MOVE then ROTATE, Enter). Hold really still in the still blocks, follow the moves.
   - The banner names each method: rate each 1-5 as you see it.
   - Then **K**.
2. **6**: compare in 6DoF, the same way. This is Phase 0's still check again, with fewer restarts.
   - Then **K**.
3. Optional: **D** for 5-10 minutes (heat: does the camera keep up, does the overlay keep locking?).
4. **Z**, and bring the zip to `GateRuns/_inbox`.

Leave `,` `.` alone (trim 0). No timecode run is needed this time.

> **After this session the PC-only (Surface) work stops** (Jack, 10 Nov). A Beam Pro is due in about a
> week, and new work goes to features that also run on XREAL One Pro + Eye + Beam Pro. The
> hand-over is `docs/BEAMPRO-HANDOVER-2026-11-10.md`. This session is still worth running: its
> numbers are the PC reference the Beam Pro is compared against.

**Done when** (desktop: `import`, then `stages`, `latency`, `filters` on both runs):

- the still-phase time in lock gaps over 1 s is at or below 14 % (3DoF) and 21 % (6DoF);
- the One Euro still error in 6DoF is ≤ 1.3 mm, and 3DoF stays ≤ 1.3 mm;
- the turning jump stays ≤ 3 mm with `age_src=hmd`;
- the frame age and decode queue at 1.0 s are recorded, for the latency half of Phase 2.

## 26. Session of 11 Nov (0.3.009): blind time and the exact age

This is the last Surface session. Its numbers are the PC reference for the Beam Pro gates
(`docs/BEAMPRO-HANDOVER-2026-11-10.md`, B5).

### What ran (build `2026-11-10T10:02:25Z`, EyeLab 0.3.009)

Plate A, Surface with fan, Jack wearing. `summary` shows the build for all five runs.

| run | what |
|---|---|
| `20261111_134231_G5_g5_timecode_3dof` | timecode (not asked for), 342 frames sent, 18 with the marker read |
| `20261111_134428_G5_g5_plateA_3dof_compare` | compare, 3DoF, 224 s, 1349 frames sent, 1008 measurements |
| `20261111_134823_G5_g5_plateA_compare` | compare, 6DoF, 218 s, 1119 frames sent, 761 measurements |
| `20261111_135213_Demo_demo_3dof` | demo, 70 s, **no result: the glasses disconnected** |
| `20261111_135339_Demo_demo_3dof` | demo, 143 s, completed |

- **The clock checks pass in the three gate runs and the second demo.** Each has `qpc ABSOLUTE`
  with a live `hmd_ns`, and every lock carries `age_src=hmd` (1121, 843 and 1102 lock events).
  `GetHMDTimeNanos` answered 6542 of 6545 and 5179 of 5182 calls. `head_clamped=0`.
- **The first demo is excluded.** `NativePerception+Start` reported `GlassesDisconnect` at start,
  `hmd_ns` was 0, and all 337 lock events used the fallback (`age_src=per_frame`). The glasses
  dropped off at 25.6 s. The fallback worked as designed, but the run measures nothing.
- **Compare was named and ran once:** `compare_mapping NAMED order=euro,hysteresis,latest`. So
  there are 60 s of still time per run, against 120 s on 6 and 9 Nov, and one rating per method.
- **No camera death.** Every stream window ended on `lag`; none reached the 30 s limit.

### Results against 6 and 9 Nov

| measure | 6 Nov 3DoF / 6DoF | 9 Nov 3DoF / 6DoF | **11 Nov 3DoF / 6DoF** | Phase 2 target |
|---|---|---|---|---|
| still-block time in lock gaps > 1 s (`blind`) | 28 % / 42 % | 22 % / 50 % | **6 % / 24 %** | ≤ 14 % / ≤ 21 % |
| camera fps by fifth (`blind`) | - / 17-20 → 11 | 22-23.5 / 23 → 11 | 24 flat / 23.5-24 → 19.5-21 | (record) |
| CPU clock by fifth (`blind`) | - | 220 flat / 220 → 190 | 222 flat / 220 → 173-185 | (record) |
| stream windows, median length (`blind`) | 48 / 86 windows | 47, 7.2 s / 82, 3.5 s | **10, 23.0 s / 21, 7.1 s** | fewer, longer |
| One Euro still jitter / error (`filters`) | 1.2 mm/s / 1.3 mm (6DoF) | 1.1 / 0.8 and 2.5 / 2.6 | **0.9 / 0.6 and 1.8 / 1.7** | ≤ 1.2 mm/s / ≤ 1.3 mm |
| turning jump as run (`latency`) | 5.7-6.9 mm (fixed age) | 2.5 / 2.1 mm; exact age 2.0 / 1.8 | **1.8 / 1.3 mm** (exact age, as run) | ≤ 3 mm |
| best residual on the exact age (`latency`) | - | −20 / 0 ms | **−10 / 0 ms** | within ±20 ms |
| cam_ts → arrival, median / p90 (`stages`) | - | 187 / 345 and 185 / 355 ms | 569 / 981 and 545 / 1018 ms | (record) |
| cam_ts → sent, median / p90 (`stages`) | - | 205 / 364 ms | 581 / 1003 and 558 / 1026 ms | (record) |
| cam_ts → shown, median (`stages`) | - | 336 / 331 ms | **701 / 680 ms** (p90 1132 / 1126) | ≤ 150 ms (latency half) |
| detection, median / p90 (`stages`) | 54 ms | 55.7 / 89.6 and 55.3 / 98.8 ms | 51.8 / 81.1 and 54.8 / 85.4 ms | (record) |

The stages we own are unchanged from 9 Nov: copy 0.2 ms, send 4.7-4.9, to service 2.7-3.6,
apply wait 10.6-11.7, display 32.5 (medians).

### 1. Did the 1.0 s threshold and the 30 s window cut the blind time? Yes

- **3DoF: 22 % → 6 %.** The gate (≤ 14 %) is met. Per still block: 9 %, 0 %, 9 %.
- **6DoF: 50 % → 24 %.** It is half of 9 Nov but misses the gate (≤ 21 %) by 3 points. Per still
  block: 19 %, 28 %, 25 %.
- **Move blocks:** 60 % and 66 % (9 Nov: 76 % and 81 %). **Rate blocks:** 10 % and 34 % (32 % and
  52 %).
- **The gain is fewer restarts, not cheaper ones.** A restart still costs a median 1.85 s (3DoF)
  and 1.99 s (6DoF) between locks, as on 9 Nov (1.87 and 2.11 s). There were 9 and 20 restarts,
  against 41 and 75 with locks either side on 9 Nov.
- **The queue reaches 1.0 s in a median 23 s (3DoF) and 7 s (6DoF).** The 30 s window never
  applied.
- **Are the runs comparable?**
  - *3DoF: yes.* Both days ran at full clock for the whole run (220-222 %), with 22-24 camera fps.
  - *6DoF: in kind, not exactly.* Both runs lost clock after the first minute (to 190 % on 9 Nov,
    to 173-185 % on 11 Nov). The camera held up better on 11 Nov: 18.4 fps inside the stream
    windows (median; lowest window 14.3) against 15.1 (lowest 3.9). `blind`'s 11 fps for 9 Nov
    also counts the time the camera was off between windows. Part of the 6DoF gain may therefore
    be a less starved decoder rather than the threshold.
  - The 3DoF run is cool and the 6DoF run warm on both days, so 3DoF against 6DoF is not
    like-for-like on either.

### 2. What did the longer windows cost? Latency doubled; accuracy did not suffer

- **Frame age at sending (`cam_ts → sent`):** 581 / 1003 ms (3DoF) and 558 / 1026 ms (6DoF),
  median / p90, against 205 / 364 ms on 9 Nov. That is 2.8 times older.
- **Total (`cam_ts → shown`):** 701 and 680 ms median, against 336 and 331 ms. The decode queue is
  now about 470 ms of it (median), up from about 105 ms.
- **The still error does not grow with frame age.** Locks in the still blocks, binned by the
  logged `age_ms`; the error is the distance of the plate's corners from the ±1 s median of the
  measurements (as in `filters`), median in mm:

  | frame age | 3DoF: n, raw, One Euro | 6DoF: n, raw, One Euro |
  |---|---|---|
  | under 250 ms | 56, 0.68, 0.34 | 87, 0.90, 1.74 |
  | 250-500 ms | 77, 0.86, 0.48 | 60, 0.66, 1.48 |
  | 500-750 ms | 72, 0.90, 0.39 | 66, 0.98, 1.34 |
  | over 750 ms | 139, 0.81, 0.52 | 88, 0.76, 1.25 |

  A frame 0.75-1.0 s old gives the same error as a fresh one. For a fixed target the age costs
  nothing once it is exact. It still costs reaction time for anything that moves.
- These bins come from a one-off script over the `lock` and `filters` events, not from a
  `xreal_gates` command.

### 3. Is the exact age right? Yes

- **Best residual: −10 ms (3DoF) and 0 ms (6DoF)**, inside ±20 ms.
- **Turning jump as run: 1.8 mm (n = 154) and 1.3 mm (n = 86)**, under the 3 mm gate and the
  lowest so far, with frames up to 1.0 s old. Still: 0.9 and 0.8 mm.
- A fixed age would be wrong by hundreds of ms here: the jump is 6.4-7.1 mm at +160 ms.
- **Timecode:** the marker was drawn 40 ms (median, p90 53, n = 18) before the `cam_ts` of the
  frame showing it. 9 Nov gave 44 ms. `cam_ts` is the exposure.
- `GetHeadPoseAtTime` matches Unity's head to 0.00-0.06° (column-major, z flipped), as before.

### 4. The 6DoF still check (Phase 0): not met, and the measurements are not the cause

- **One Euro in 6DoF: 1.8 mm/s and 1.7 mm**, against ≤ 1.2 mm/s and ≤ 1.3 mm. It improved on
  9 Nov (2.5 / 2.6) and 7 Nov (3.9 mm error). Per block the error is 1.6, 1.8 and 1.7 mm.
- **The raw measurements are good.** The `latest` error is 0.9 mm, half of 9 Nov's 1.8 mm and
  close to the 6 Nov baseline (0.7 mm).
- **What One Euro lags is a slow movement of the measurement, not noise.** Within a still block
  the measured plate position wanders 2.8-3.8 mm (median distance from the block median) in
  6DoF, against 1.0-1.5 mm in 3DoF. One Euro at its 0.1 Hz minimum cutoff follows that late.
  The error does not fall in fresher frames (table above), so the frame age is not the cause
  either.
- **The head was not quite as still as in the run that passed.** `pose.csv`, median head speed in
  the still blocks:

  | 6DoF run | head speed | One Euro still error |
  |---|---|---|
  | 6 Nov (baseline) | 2.0-2.8 mm/s | 1.3 mm |
  | 9 Nov | 2.4-6.0 mm/s | 2.6 mm |
  | 11 Nov | 2.6-3.6 mm/s | 1.7 mm |

  In 3DoF the head turned 0.24-0.47°/s in the still blocks.
- **Correction (12 Nov): that head speed is mostly tracker drift, not the wearer.** The plate is
  fixed, so with a perfect tracker its measured world position stays put however the head moves.
  It does not. Over 2 s inside the still blocks, the plate's world position moves with the
  tracker's head position, in the same direction:

  | 6DoF run | tracker head move over 2 s, median | plate-in-world move, median | correlation | share along the head move |
  |---|---|---|---|---|
  | 6 Nov | 2.2 mm | 2.7 mm | 0.94 | 0.84 |
  | 7 Nov | 5.0 mm | 4.9 mm | 0.87 | 0.80 |
  | 9 Nov | 4.1 mm | 3.7 mm | 0.98 | 0.90 |
  | 11 Nov | 2.5 mm | 2.6 mm | 0.81 | 0.90 |

  So 80-90 % of the head movement the tracker reports in a still block did not happen: the
  camera, which sees the plate to under 1 mm, does not see it. The 6DoF position drifts by a few
  mm over seconds while standing still, and the overlay has to follow it.
- **No One Euro setting meets both still gates on these runs.** A sweep of the minimum cutoff
  (0.1-0.8 Hz) and β (2-10) on the 11 Nov 6DoF run trades one for the other: 1.8 mm/s and 1.7 mm
  at the default, 2.0 and 1.3 at β 10, 2.3 and 1.3 at 0.2 Hz, 3.1 and 0.9 at 0.5 Hz. The drift
  itself moves the measurement by about 1.3 mm/s, so following it closely is already over the
  jitter gate. The default stays.
- **Verdict.** In three sessions since 6 Nov the 6DoF still gate was not repeated: 3.9, 2.6,
  1.7 mm. The cause is the tracker's position drift, not the filter, the frame age or the wearer.
  The 6 Nov baseline was the run with the least drift, so the gate (1.3 mm, 1.2 mm/s) was set on
  a good day. The check goes to the Beam Pro (B5) together with a drift measurement of its own:
  glasses at rest, 6DoF, plate in view.
- These two results come from one-off scripts over the `lock` events and the `filters` replay.
- **3DoF meets it:** 0.9 mm/s and 0.6 mm, the best run so far.

### 5. The named comparison: ratings and measurements agree

`filters`, mean over the still blocks (jitter mm/s / error mm), and Jack's one rating per method:

| | latest | One Euro | hysteresis |
|---|---|---|---|
| 3DoF | 6.4 / 0.5 | **0.9 / 0.6** | 0.2 / 1.4 |
| 3DoF rating | 3 | **4** | 3 |
| 6DoF | 5.6 / 0.9 | **1.8 / 1.7** | 1.6 / 3.7 |
| 6DoF rating | 3 | **4** | 3 |

- **One Euro is rated best in both modes.** It is near the calmest and near the most accurate.
- **Latest** is the most accurate and shakes three to seven times more. **Hysteresis** is the
  calmest and sits about twice as far off.
- With the names shown, the ratings are the same in both runs. The blind ratings of 9 Nov were
  not (One Euro 3.5 in 3DoF, 2.0 in 6DoF).
- **One Euro stays the default.** One rating per method is thin evidence, but nothing here
  argues for a change.

### 6. Demo: too short for the heat question

- **The completed demo ran 2.4 min, not 5-10.** Camera 21.5 → 19.5 fps and clock 223 → 177 % by
  fifth; 13 stream windows of 10.8 s (median), all ended on `lag`.
- **35 % of the time after the first lock was in gaps over 1 s.** The wearer looks around
  freely, so this is an upper bound on the camera's own blind time.
- **Keys:** T (targets on) at 26 s; F three times (hysteresis at 50 s, latest at 59 s, One Euro
  again at 73 s); V four times (points, lines, both, points). The grid and the Eye view stayed
  off. Esc ended it.
- Whether the camera keeps up over 5-10 minutes with the 1.0 s threshold is **not answered** on
  the Surface.

### Other observations

- **The alignment differed between the two runs.** 3DoF was saved at x −18.9 mm, yaw −0.3°
  (82 s); 6DoF at x +31.6 mm, yaw +0.5° (31 s). Pitch +0.9° and roll +1.0° did not change. The
  cause is not investigated.
- **6DoF reported InsufficientLight five times**, all in the move blocks.
- **Most failed frames are "1 marker, need 2"** (109 and 79): the plate at the edge of the view.
- Unity rendered at 27 fps. The hottest zone read 51-52 °C while the clock fell, as in §23.

### What this settles for the Beam Pro

- **Frame age.** The exact age holds on the PC up to 1 s of age. On the Beam Pro, B1's clock
  question is only whether `cam_ts` is on the same clock as `GetHMDTimeNanos` there.
- **Restart threshold.** 1.0 s cut the blind time to a quarter (3DoF) and a half (6DoF) with no
  loss of still accuracy or turning accuracy. It is the starting setting if B1 shows a decoder
  queue on Android. Its price is a correction 0.6-1.0 s old.
- **Latency.** On this decoder, blind time and latency cannot both be met. Exposure → shown is
  now 680-700 ms against a target of 150 ms. Whether Android has the queue at all is B1.
- **PC reference for B5:** 3DoF still 0.9 mm/s and 0.6 mm; 6DoF still 1.8 mm/s and 1.7 mm;
  turning jump 1.8 and 1.3 mm; still blind time 6 % and 24 %; detection 52-55 ms.

### Commands (from `python/`, `R` = `../unity_projects/xreal_pc_gates/GateRuns/runs`)

```
python -m xreal_gates import --keep-zip ../unity_projects/xreal_pc_gates/GateRuns/_imported/GateRuns_CODY_0526_20261111_135614.zip
python -m xreal_gates summary
python -m xreal_gates blind   R/20261111_134428_G5_g5_plateA_3dof_compare R/20261111_134823_G5_g5_plateA_compare R/20261111_135339_Demo_demo_3dof
python -m xreal_gates stages  R/20261111_134428_G5_g5_plateA_3dof_compare R/20261111_134823_G5_g5_plateA_compare
python -m xreal_gates latency R/20261111_134428_G5_g5_plateA_3dof_compare R/20261111_134823_G5_g5_plateA_compare
python -m xreal_gates filters R/20261111_134428_G5_g5_plateA_3dof_compare R/20261111_134823_G5_g5_plateA_compare
python -m xreal_gates stages  R/20261111_134231_G5_g5_timecode_3dof
```

`blind` now also reports a run without protocol blocks (the demo): the share of time in lock
gaps over 1 s from the first lock (0.3.010). The per-block blind time, the restart cost, the
head speed and the age bins come from a one-off script over `events.csv` and `pose.csv`.
