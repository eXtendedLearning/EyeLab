# Beam Pro hand-over (2026-11-10, build 0.3.009)

> Hand-over for a new agent session. The PC gate work (`unity_projects/xreal_pc_gates/`, Surface
> Pro 7, XREAL PC beta 3.1.2) stops here. **Next target: XREAL One Pro + XREAL Eye + XREAL Beam
> Pro.** Jack picks the Beam Pro up in about a week (expected around 2026-11-17).
>
> From now on, build only features that will also run in that configuration. Do not add more
> Surface- or Windows-only machinery.

## Prompt to paste into the new session

```text
You are taking over EyeLab at build 0.3.009 (check: python python/eyelab_version.py).
Read, in this order and in full: AGENTS.md, docs/BEAMPRO-HANDOVER-2026-11-10.md, CONTEXT.md
(Status, Deployment, Resolved decisions, Open issues), docs/HARDWARE.md, docs/adr/0001 and 0003,
docs/tasks/T2.6-implementation-checklist.md, then docs/XREAL-PC-GATES-G1-G3.md §20-25.

Situation: the XREAL PC-beta gate project on the Surface proved the whole correction chain:
Eye frame, ArUco/plate pose, the head pose at the frame's own capture time, a lock filter, a
world-locked overlay aligned by eye. A Beam Pro arrives in about a week and is now the
reference host. From now on we build only what also runs on XREAL One Pro + Eye + Beam Pro
(Android). The PC-only work is paused.

Your jobs, in order:
0. Analyse the last Surface session, build 0.3.009. Its zip is in
   unity_projects/xreal_pc_gates/GateRuns/_imported (or _inbox): newest
   GateRuns_*_<date>.zip with runs from build 2026-11-10T10:02:25Z. Follow the handover's
   section "First task: the 0.3.009 Surface session". Write the results into
   docs/XREAL-PC-GATES-G1-G3.md as §26 and close T5.1 in docs/tasks/T5.1-eye-latency.md. They
   are the PC reference the Beam Pro is measured against. Stop and report before step 1.
1. Classify the gate project's code and findings as portable or PC-only, using the handover's
   tables. Verify each claim against the code.
2. Propose (an ADR if it reverses an accepted one) where the portable core lives. Then extract
   it so that eyelab_xreal (Android) and the gate project both use it, behind interfaces
   for the camera source, clock and pose bridge, with tests.
3. Write the Beam Pro bring-up plan as docs/tasks/T6.x files with "Done when" gates (B0-B5 in
   the handover), plus the scripts and log formats so `python -m xreal_gates` can analyse
   Beam Pro runs as it does Surface runs.
4. You may build APKs and deploy them to the Beam Pro over adb, within AGENTS.md "Unity". Ask
   me before anything listed there as mine: Play mode, anything on the device outside the
   EyeLab app, firmware or OS updates, USB changes on the glasses, and changing the ADR-001
   bridge staircase.

Work in small commits that follow AGENTS.md (version prefix via tools/bump_version.py, plain
lowercase messages, commit and push to main only when I ask). Do not run at the same time as
another agent session on this repo. Stop after each gate for my results.
```

## First task: the 0.3.009 Surface session

> **Done (2026-11-11, build 0.3.010).** The session ran on 11 Nov; the results are guide §26 and
> the closing status in `docs/tasks/T5.1-eye-latency.md`. The reference numbers for B5 are in
> §26, "What this settles for the Beam Pro".

Jack runs build 0.3.009 (launcher `[OK]` for `2026-11-10T10:02:25Z`) on the Surface and drops the
exported zip in `GateRuns/_imported` or `GateRuns/_inbox`. The protocol is guide §25, "Session
0.3.009":

- **5**: compare in 3DoF;
- **6**: compare in 6DoF;
- optionally **D**, a 5-10 minute demo;
- **K** between runs.

What the build changed is in §25 too:

- the frame age comes from the HMD clock (`age_src=hmd`);
- Unity times are absolute QPC;
- the stream restarts at 1.0 s of lag (was 0.3) and at most every 30 s (was 8);
- the filter compare is named (ONE EURO, HYSTERESIS, LATEST) and runs once;
- the demo starts with everything off, and F switches the method.

### Commands (from `python/`)

```bash
python -m xreal_gates import --keep-zip ../unity_projects/xreal_pc_gates/GateRuns/_imported/<zip>
python -m xreal_gates summary
python -m xreal_gates blind  <3dof compare run> <6dof compare run> [<demo run>]
python -m xreal_gates stages <3dof compare run> <6dof compare run>
python -m xreal_gates latency <3dof compare run> <6dof compare run>
python -m xreal_gates filters <3dof compare run> <6dof compare run>
```

- If the zip is already in `_inbox`, plain `import` takes it and moves it to `_imported`.
- Runs land in `GateRuns/runs/<stamp>_G5_g5_plateA[_3dof]_compare` and `..._Demo_demo_3dof`.
- Check that `summary`'s build column says `2026-11-10T10:02:25Z` for every run.
- Check that the events carry `qpc ABSOLUTE` and that every lock event has `age_src=hmd`.
  If not, the HMD clock failed and the fallback ran: say so before comparing anything.

### Reference: 9 Nov (0.3.008) and 6 Nov (0.3.005), same commands

| measure | 6 Nov 3DoF / 6DoF | 9 Nov 3DoF / 6DoF | Phase 2 target |
|---|---|---|---|
| still-block time in lock gaps > 1 s (`blind`) | 28 % / 42 % | 22 % / 50 % | ≤ 14 % / ≤ 21 % |
| camera fps by fifth (`blind`) | - / 17-20 → 11 | 22-23.5 / 23 → 11 | (record) |
| stream windows, median length (`blind`) | 48 / 86 windows | 47, 7.2 s / 82, 3.5 s | fewer, longer |
| One Euro still jitter / error (`filters`) | baseline 1.2 mm/s / 1.3 mm (6DoF) | 1.1 / 0.8 and 2.5 / 2.6 | ≤ 1.2 mm/s / ≤ 1.3 mm |
| turning jump as run (`latency`) | 5.7-6.9 mm (fixed age) | 2.5 / 2.1 mm; exact age 2.0 / 1.8 | ≤ 3 mm |
| best residual on the exact age (`latency`) | - | -20 / 0 ms | within ±20 ms |
| cam_ts → arrival, median / p90 (`stages`) | - | 187 / 345 and 185 / 355 ms | (record) |
| cam_ts → shown, median (`stages`) | - | 336 / 331 ms | ≤ 150 ms (latency half) |
| detection, median / p90 (`stages`) | 54 ms | 55.7 / 89.6 and 55.3 / 98.8 ms | (record) |

### Questions the analysis must answer

1. **Did the 1.0 s threshold and 30 s window cut the blind time?** Compare `blind` for still,
   move and rate blocks, the number and length of the stream windows, and why they ended. Was the
   Surface as warm (camera fps and clock by fifth)? A cool 3DoF run against a hot 6DoF run is not
   a like-for-like comparison: say which runs are comparable.
2. **What did the longer windows cost?**
   - Frame age: `stages` `cam_ts → sent`, median and p90, against 9 Nov's 205 / 364 ms.
   - Total latency: `cam_ts → shown`.
   - Does the still error grow with frame age? Bin the locks by `age_ms` from the lock events
     and compare the filter error per bin.
3. **Is the exact age right?** Check that `latency`'s best residual is near 0, and that the
   turning jump is still ≤ 3 mm.
4. **The 6DoF still check (Phase 0, still open).** Is the One Euro still error ≤ 1.3 mm?
   - If not, are the raw measurements scattered (`latest` error), or is the filter the problem?
   - Is the head still (`pose.csv` speed in the still blocks)?
5. **The named comparison.** Give the per-method ratings next to `filters`' numbers. Do the
   ratings agree with the measurements? Keep One Euro as the default unless the data says
   otherwise.
6. **Demo (if run):** camera fps and blind time over 5-10 minutes; the `key` events show what
   was turned on and which method was used.

### What to write

- **Guide §26**, "Session of <date> (0.3.009): blind time and the exact age":
  - what ran;
  - the table above with a 0.3.009 column;
  - the answers to 1-6, with the numbers and the commands that produced them.
- **T5.1:** a closing status under Phase 2. It is paused, not done. State which Phase 2 gates
  were met on the Surface and which are left for the Beam Pro.
- **CAPABILITIES rows 19-20:** update if a capability changed.
- **CONTEXT `Last updated`, CHANGELOG.**
- **This hand-over's "Proven on the Surface" table**, if 0.3.009 changes a number in it.

Then use the conclusions in the next steps:

- **Frame age.** If the exact age holds on the PC, the Beam Pro's first clock question (B1) is
  only whether `cam_ts` is on the same clock as `GetHMDTimeNanos` there.
- **Restart threshold.** If 1.0 s helped without hurting accuracy, it is the starting setting
  if B1 shows a decoder queue on Android.

## Where things stand

### Proven on the Surface (PC beta 3.1.2, Windows)

Evidence: `docs/XREAL-PC-GATES-G1-G3.md` §12-25 and `docs/CAPABILITIES.md` rows 16-20. Run logs
are under `unity_projects/xreal_pc_gates/GateRuns/runs/` on the desktop (gitignored).

| result | numbers (best runs) | where |
|---|---|---|
| Eye calibration with the 25 mm ChArUco board | RMS 0.47 px, fx 623.9 at 1280x720; vendor fx is 2.7 % short | §20-22, `python/xreal_gates/calib.py` |
| Plate A pose from markers plus printed saddle points | detection 55 ms median per frame on the Surface | `python/xreal_gates/targets.py` |
| Each frame paired with the head pose at its own capture time | turning jump 1.3-1.8 mm with the exact age, frames up to 1 s old (was 5.7-6.9 with a fixed age) | §24-26 |
| Lock filters: latest, One Euro (default), hysteresis | One Euro still blocks: 0.9 mm/s, 0.6 mm (3DoF); 1.8 mm/s, 1.7 mm (6DoF, gate 1.3 mm not met) | §20, §25-26, `LockAnchor.cs` |
| Stream restarts at 1.0 s of decoder lag | still time without a lock 6 % (3DoF), 24 % (6DoF); exposure → overlay 680-700 ms | §26 |
| View offset (boresight) set by eye | about 1.3°, kept as an angle; Tab MOVE/ROTATE alignment | §23-24, `ViewOffset.cs` |
| Testers' UX: compact HUD, named movement cues, thin drawing, demo with toggles | liked ("all great improvements", 9 Nov) | §24-25 |

### Measured, and Windows-specific

- **On Windows, `cam_ts` is on the HMD clock, and the HMD clock is QPC** (ns since boot). On
  Android the clock domain must be measured again; do not assume it.
- **The Eye stream arrives compressed and the PC SDK decodes it on the CPU** (FFmpeg DLLs).
  - The decoder falls behind, and its queue is the largest latency stage: about 105 ms of the
    330 ms from exposure to overlay.
  - The stream restarts, heat rules and blind-time work all exist because of this.
  - On Android the SDK ships `libmedia_codec.so`. Whether it decodes in hardware and keeps up is
    **unknown**; measure it first (B1).
- The Surface throttles under XR load. Fans and cooldown checks are Surface workarounds.

### Not done anywhere

- No Android or Beam Pro run of anything: the main project `eyelab_xreal/` is a scaffold (empty
  folders and a sample scene; the empty APK built once on 2026-05-11).
- No measurement of the Python pose service over Wi-Fi (on the Surface it was localhost: copy,
  send and transport about 10 ms).
- No on-device ArUco (ADR-001 Steps A and B).
- No `.unv` geometry overlay in the XR app: the gates draw test targets only.

## Portable or PC-only

Use this table as a starting point and verify every row in the code. Files are under
`unity_projects/xreal_pc_gates/Assets/EyeLabGates/Runtime/` unless a path is given.

| item | verdict | notes |
|---|---|---|
| `IArucoPoseBridge.cs`, `WebSocketPoseBridge.cs` | **portable** (load-bearing interface, AGENTS.md) | Step C over Wi-Fi to a PC; must cope with packet loss and reconnects (T2.6 §1.2 protocol) |
| `LockAnchor.cs` (three filters, acquisition and tracking phases) | **portable** | pure C#; mirrors `python/pose_lock.py` and `python/xreal_gates/lockfilters.py` |
| `ViewOffset.cs` | **portable** | per wearer and device; storage path must be Android-safe |
| head-pose history and pairing at `now − age` (in `GateController.cs`) | **portable, extract** | needs a clock abstraction: the frame age source differs per platform |
| frame age from the HMD clock (`Clocks.cs` `NativeClock`) | **verify on Android** | Windows `DllImport` of `XREALXRPlugin` exports; on Android the plugin is `libXREALXRPlugin.so`; re-check the exports and the clock domain |
| `CameraSource.cs` interface, `VendorCameraSource.cs` (`XREALRGBCameraTexture`) | **portable interface**; source verify | YUV_420_888 on both. The PC frames were bottom-up; check orientation and resolution on Android |
| `GateProtocol.cs` (steps, named cues), `Speaker.cs`, `GateHud.cs` compact mode, `GateOverlay.cs` styles | **portable**, except `Speaker` | `Speaker` drives Windows `System.Speech` through PowerShell: Android needs its own TTS behind the same calls. The HUD needs a phone-screen or controller UI on the Beam Pro, not a keyboard |
| `GateRunLog.cs` (events.csv, frame_timing.csv, …) | **portable** | keep the formats so `python -m xreal_gates` works on Beam Pro runs; needs an export path from the device |
| `GateController.cs` | **split** | 2 200 lines mixing gate logic, keyboard input and Windows concerns. Extract the portable services; leave the gate shell behind |
| Stream restarts on decoder lag, `UpdateHeat`, `PerfProbe` thermal zones, cooldown gate | **PC-only** | revisit only if B1 shows the same decoder queue on Android |
| `Clocks.Qpc` (kernel32), `run_gates.bat`, Surface launcher, timecode on the Surface screen | **PC-only** | the timecode *method* is portable: show the marker on any second screen |
| Python pose service `python/xreal_gates/pose_service.py`, `targets.py`, `calib.py` | **portable as Step C** | runs on a PC; the Beam Pro talks to it over Wi-Fi |
| `stages`, `latency`, `filters`, `chain`, `anchor` analysis tools | **portable** | given the same log columns |

### Verified against the code (2026-11-12, build 0.3.010)

The table above is the 10 Nov starting point, kept as written. This is the result of checking
each row in the code. "3.1.0" is the Android SDK installed in `eyelab_xreal/Packages/com.xreal.xr`;
"3.1.2" is the PC beta the gate project uses. Nothing here has run on Android: "compiles" and
"exists" are read from source, not observed.

**Verdicts that changed**

| item | 10 Nov | verified | why |
|---|---|---|---|
| `WebSocketPoseBridge.cs` | portable | **rework for Wi-Fi** | see "The bridge" below |
| `GateArgs.cs` | (not listed) | **PC-only input** | reads `Environment.GetCommandLineArgs()`; an Android app has none. Needs intent extras or a config file pushed with `adb` |
| XR start-up in `GateController` (`Connect`, `StartXR`, `EnterXR`, the glasses-connect events) | part of "split" | **PC-only** | uses `XREALSessionManager.OnGlassesConnect` / `OnStartSubsystems`, which exist only in `com.xreal.xr.experimental` (PC beta). 3.1.0 has no such events |
| `DeviceInfoProbe.cs` | (not listed) | **mixed** | `XREALExperimentalPlugin` version and monitor calls are PC beta only; the `XREALPlugin` calls (intrinsics, pose from head, feature flags) exist in 3.1.0 |
| `DisplayProbe.cs` | (not listed) | **PC-only** | `user32.dll` |
| `Clocks.Qpc` | PC-only | **PC-only source, but it does not break Android** | the kernel32 calls are wrapped and fall back to `Stopwatch`; `Clocks.Absolute` then reads false |
| Python pose service | portable as Step C | **portable with changes** | see "The bridge" below |

**Verdicts confirmed**

| item | verdict | evidence |
|---|---|---|
| `IArucoPoseBridge.cs` | portable | `System` and `UnityEngine` only; the signature is ADR-001's (`IsAvailable`, `EstimatePoseAsync(YGrayImage, ulong, CancellationToken)`, `OnLatencyMeasured`) |
| `LockAnchor.cs` | portable | `System` and `UnityEngine` maths only; time is passed in, not read |
| `ViewOffset.cs` | portable | `JsonUtility` and `System.IO` under `Application.persistentDataPath`. Confirm at B0 that this path can be pulled with `adb` |
| head history and pairing (`RecordHead`, `HeadAt`, `HeadSpeed`, `WorldPose`, `CvRotationToUnity`) | portable, extract | Unity maths and `Time.realtimeSinceStartupAsDouble`; the only platform inputs are the frame age and the head transform |
| `NativeClock` (`GetHMDTimeNanos`, `GetHeadPoseAtTime`) | verify on Android | `DllImport("XREALXRPlugin")` resolves to `libXREALXRPlugin.so`. The 3.1.0 Android library contains both names, and `SetNativeLogCallback`. Signatures and clock domain are not checked. Every call is wrapped, so a failure falls back to the estimated age |
| `CameraSource.cs`, `VendorCameraSource.cs`, `DrainingCameraSource.cs` | portable; verify the source | every vendor call they use exists in 3.1.0 (`GetTimeStamp`, `GetResolution`, `GetYUVFormatTextures`, `OnRGBCameraUpdate`, `IsCapturing`, `StartCapture`, `StopCapture`, `TryAcquireLatestImage`, `TryGetRGBCameraDataPlane`, …) |
| `CameraProbe.cs`, `PoseProbe.cs` | portable | `CameraProbe` needs the clock abstraction (`Clocks`, `NativeClock`); `PoseProbe` uses `InputDevices` and `XREALPlugin.GetTrackingType`, both in 3.1.0 |
| `NativeTracking.cs` | verify on Android | the native log callback already carries `AOT.MonoPInvokeCallback`, as IL2CPP needs |
| `GateProtocol.cs`, `GateOverlay.cs` | portable | no platform calls. `GateOverlay` falls back to `Shader.Find("Unlit/Color")`: keep the material assigned so the shader is in the APK |
| `GateHud.cs` | portable drawing, PC input | IMGUI (`OnGUI`) runs on Android; the side-by-side split is decided from the screen's aspect and must be checked on the Beam Pro |
| `Speaker.cs` | PC-only | starts `powershell.exe` |
| `GateRunLog.cs` | portable | plain CSV through `System.IO`; the default root is `persistentDataPath/GateRuns` |
| `PerfProbe.cs` | PC-only | `pdh.dll` counters and ACPI thermal zones. The camera-rate columns are computed from our own counters and carry over |
| stream restarts, `UpdateHeat`, cooldown, bursts | PC-only | all inside `GateController` (`StreamLoop`, `UpdateHeat`, `RunCooldown`, `BurstLoop`) |
| keyboard input | PC-only | 30 `Input` / `KeyCode` uses in `GateController`, 3 in `Speaker`; none elsewhere |
| `Editor/BuildGates.cs`, `ProjectSetup.cs`, `GateSceneBuilder.cs` | PC build tooling | build Windows players; the batch-mode pattern is reusable for the APK |
| `blind`, `filters`, `latency`, `chain`, `anchor` | portable | read `events.csv`, `pose.csv`, `run.json` only. `import` takes a pulled folder (`archive.import_folder`) |
| `stages` | portable with one assumption | it subtracts Unity and service times directly. That holds only when both run on one machine's clock |

**The bridge: what Step C over Wi-Fi needs** (`WebSocketPoseBridge.cs`, `python/xreal_gates/pose_service.py`, `ws.py`)

- **The address is fixed** to `ws://127.0.0.1:port` on both sides. The service binds loopback only.
- **Frames go as raw Y planes,** 0.9 MB each: about 5 MB/s at the 5.5-6.4 frames/s the Surface
  sent. That was chosen for localhost.
- **No timeout, no reconnect, no session restart.** One request in flight; a lost reply blocks
  the next frame for good.
- **`hello` sends `run_dir`,** a path on the Unity machine, and the service writes
  `pose_service.csv` and `pose_service.json` into it. Over Wi-Fi the service needs its own log
  folder, and `import` has to join the two.
- **`hello` hard-codes 1280x720.** `TrySendLockFrame` hard-codes `BottomUp = true` at the send
  site instead of asking the camera source.
- **The service stamps stages with its own `perf_counter`.** On another machine that is another
  clock, so the stage table needs a clock-offset estimate in the protocol.
- **This is not the protocol the documents specify.** CONTEXT.md "Frame transport protocol (v0)"
  and T2.6 §1.2 define a 20-byte `ELAB` header, JPEG, `no_pose`, sessions and reconnects. The
  gate project speaks its own: a 4-byte length, a JSON header, raw Y, several targets per reply.
  T2.6 §2.1 also sketches a different `IArucoPoseBridge` from ADR-001's. Step 2 has to pick one
  of each.
- `ClientWebSocket` and `JsonUtility` are expected to work under IL2CPP on Android, but that is
  not observed. The main project already references `NativeWebSocket` and Newtonsoft JSON.

**Other things the check found**

- **`HeadAt` clamps.** When the history does not reach back to the capture time it uses the
  oldest sample and counts `head_clamped` (0 in every run so far). T2.6 §5.1 and CONTEXT open
  issue 3 say such a pose is rejected, not clamped. There is no tracking-epoch reset either.
- **The gate project has no C# tests and no assembly definition:** everything compiles into
  `Assembly-CSharp`. The filters are tested only through their Python ports
  (`test_lockfilters.py`). The main project already has `com.unity.test-framework`.
- **Two Unity versions and two SDKs.** `eyelab_xreal` is on 6000.4.5f1 with 3.1.0; the gate
  project is on 6000.6.5f1 with 3.1.2. The 3.1.2 package also ships Android plugins
  (`libXREALXRPlugin.so`, `nr_algo.aar`), so it may serve both targets. Not tested.
- **`GateController.cs` is 2 191 lines.** The portable part is roughly lines 876-1200 (pose
  chain, head history, send and apply) plus the alignment maths (`RotateAboutTarget`,
  `TryTargetCentreInHead`).

**Findings: which carry over**

| finding | carries to the Beam Pro? |
|---|---|
| Eye calibration method, 25 mm board, intrinsics | yes, if the Android stream has the same resolution and crop (B4) |
| pairing each frame with the head pose of its capture | yes; the age source is B1's question |
| lock filters and their trade-off, One Euro default | yes |
| view offset as a boresight angle, set by eye | yes; per wearer, set again on the device |
| compact HUD, named cues, thin drawing, demo toggles | yes, with new input |
| 6DoF position drifts a few mm over seconds while standing still (§26) | to measure again: tracking may run differently on the Beam Pro |
| `cam_ts` on the HMD clock, HMD clock = QPC | Windows-specific; B1 |
| decoder queue, 2 s restart cost, 680-700 ms latency at the 1.0 s threshold | PC-specific; B1 says whether a queue exists |
| Surface throttling, fan, cooldown, zone temperatures | PC-specific |
| rows bottom-up | PC observation; check on Android |
| stage times (detection 52-55 ms on the Surface's CPU) | the service will run on another PC over Wi-Fi; B3 |

## Decisions that stand until Jack changes them

- **The Beam Pro is the reference host** (AGENTS.md "Unity", 2026-11-10; it replaces the
  2026-05-08 "generic APK" decision).
  - Beam-Pro-specific manifest entries, permissions, input and tuning are allowed when needed.
  - Record each in CONTEXT.md with its reason.
  - Do not break other Android hosts without a reason.
- **ADR-001 staircase C → A → B.** The pivot from C to A is quality-driven: median end-to-end
  latency over 150 ms, accuracy worse than 10 mm RMS, or unreliable Wi-Fi.
  - **The Beam Pro is the first time Step C runs over Wi-Fi. Measure it before pivoting.**
  - The Surface measurement is not that number: it ran on localhost, with a CPU decoder in the
    way.
- **ADR-003 (proposed):** the pose camera must be the glasses' own camera (the Eye), rigidly
  coupled to the display.
- **AGENTS.md invariants:** metres past the parser, one definition per physical constant, each
  frame processed once (`python/frame_packet.py`), phase-dependent pose acceptance
  (`python/pose_lock.py`). Reprojection RMS never proves a scale.
- **SDK.** The main project pins XREAL XR Plugin 3.1.0 for Android, installed locally and
  patched with `tools/patch_xreal_sdk.ps1`; the gate project uses the PC beta 3.1.2. Check
  whether a newer Android SDK exists and whether 3.1.0 vs 3.1.2 matters for the Eye on Android.
  Do not modify or redistribute vendor binaries.
- **Unity versions:** `eyelab_xreal/` is on 6000.4.5f1; the gate project was built with
  6000.6.5f1 (`docs/XREAL-PC-GATES-G1-G3.md` §11). Pick one for shared code and say why.
- **What an agent may do** (AGENTS.md "Unity", 2026-11-10):
  - **May:** use the Unity CLI in batch mode for any project here, build APKs and Windows players,
    and run EditMode tests.
  - **May, once Jack has connected the Beam Pro:** `adb install -r` an EyeLab APK, start and stop
    it, grant its declared permissions, read `logcat`, and pull and push inside the app's own
    storage.
  - **Jack's:** Play mode; wearing the glasses and running protocols; anything on the device
    outside the EyeLab app; firmware and OS updates; USB changes on the glasses; changing the
    ADR-001 bridge.
  - **Signing:** debug key or a local keystore outside the repo (`*.keystore` and `*.jks` are
    gitignored).

## Beam Pro bring-up gates (to write as `docs/tasks/T6.x`)

The same idea as the PC gates G1-G5: one question per gate, a protocol, logs the Python tools
read, and a "Done when".

| gate | question | done when (proposal) |
|---|---|---|
| **B0** | Does the APK install (agent, over adb) and start on the Beam Pro with the glasses as display? | the empty or minimal APK runs; SDK version, firmware and Beam Pro OS recorded in `HARDWARE.md` and `CAPABILITIES.md` |
| **B1** | Eye frames on Android: rate, resolution, format, orientation, `cam_ts` clock domain, decode lag over 5 min | `camera.csv` from the device; the lag growth says whether a decoder queue exists on Android |
| **B2** | 6DoF and 3DoF head pose on the Beam Pro, alone and with B1's camera | tracking ready time, lost events, and camera rate with tracking on |
| **B3** | Step C over Wi-Fi: frame to the PC service and pose back | per-stage table like `stages` (send, network, detection, reply); median and p90; packet loss over 10 min. **This decides C vs A** |
| **B4** | Eye calibration on the Android stream | `calib` RMS; intrinsics compared with the Surface calibration (same camera, possibly a different stream) |
| **B5** | Locked overlay on plate A, with alignment and lock filters | the compare protocol (named, once) gives still error and jitter, and a turning jump, against the Surface 9 Nov reference (§25) |

The Surface 0.3.009 session (guide §25) is still worth one run: its numbers are the PC reference
for B5.

## Practical notes

- **Never run two agent sessions on this repo at once** (AGENTS.md sandbox hazards).
- **Logs on the device:** keep runs in the app's own storage so the agent can `adb pull` them;
  teach `python -m xreal_gates import` to take a pulled folder (it already takes folders). Keep
  `events.csv`, `perf.csv`, `pose.csv`, `camera.csv` and `frame_timing.csv` column-compatible, so
  `blind`, `stages`, `latency` and `filters` work unchanged.
- **Input on the Beam Pro:** there is no keyboard. Alignment, method switching (F in the PC demo)
  and ratings need the Beam Pro's touchscreen or controller mode. The demo's toggles (grid,
  overlays, Eye view, method: all off at start since 0.3.009) are the UX to carry over.
- **Power:** the Beam Pro has its own battery and a charge-through port; record whether long runs
  need power, as the Surface needed mains and a fan.
- **Confidential material:** `.docs/` is gitignored (Siemens scope, dates, prices). Never copy it
  into tracked files.
