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
