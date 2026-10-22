# EyeLab — domain context (XREAL phase)

> Living glossary and shared-mental-model notes. Captures terms, decisions, and open issues surfaced during grilling. Authoritative scope/plan lives in `.docs/PROJECT.md`, theory in `.docs/THEORY.md`, task plan in `.docs/TASKS.md`, decisions in `.docs/adr/`. This file holds **deltas the project plan does not capture** plus the live "open question" board.
>
> Last updated: 2026-10-22, build 0.3.002 (G5 rev. 2 runs: 6DoF failed because the RGB camera died after a too-fast restart, 3DoF stable to a few mm but slow and not aligned for the eyes; rev. 3: stream camera mode with lag-triggered restarts, camera rest and recovery, head-frame view offset set by eye and saved, align gate, 3DoF demo; `docs/XREAL-PC-GATES-G1-G3.md` §19. Earlier: 2026-10-22, build 0.3.001 (first G5 runs at metroXRAINE: Eye calibration good, overlay floated because 6DoF position drifted at the venue while camera poses were steady; G5 rev. 2 follows the latest lock, 3DoF variants, multi-target pose service, demo gate, launcher submenus; `docs/XREAL-PC-GATES-G1-G3.md` §18. Earlier: 2026-10-17, build 0.3.000 (MINOR cut: validation campaign tooling plus the XREAL gate work below. G4 light overlay sustainable on the Surface with a fan; ChArUco detected in Eye burst frames, world lock holds within a still phase but needs an Eye calibration; Eye calibration capture and G5 camera-locked overlay via the ADR-001 pose bridge, not yet run on the glasses; `docs/XREAL-PC-GATES-G1-G3.md` §17. Earlier: 2026-10-14, build 0.2.007 (radiator runs: Eye stream arrives at 30 fps but is decoded on the host CPU, which falls behind next to 6DoF even at full clock; camera restarts empty the queue; desktop DP-in card not bought; light-overlay gate G4 with camera bursts; `docs/XREAL-PC-GATES-G1-G3.md` §16. Earlier: 2026-10-14, build 0.2.005 (gate runs with the voice-guided build: Surface CPU throttles to 45-60 % of nominal ~50 s into XR, causing the minutes-long camera lag; Eye buffer bottom-up; tracking reason now from the native log; build/run archive workflow; `docs/XREAL-PC-GATES-G1-G3.md` §15. Earlier: 2026-10-14, build 0.2.004 (Surface gate runs of 14 Oct: G1 complete at 30 fps, G2 6DoF held 3 min, concurrent camera + 6DoF keeps tracking but camera lag grows to minutes; voice-guided harness rebuilt; `docs/XREAL-PC-GATES-G1-G3.md` §14. Earlier: 2026-10-13, build 0.2.003 (gate harness rebuilt with guided protocols, tracking-ready wait and G3 variants, `docs/XREAL-PC-GATES-G1-G3.md` §13; XREAL PC-beta gate runs on the Surface: Smart App Control blocked the unsigned runtime until turned off; then Eye frames 30 Hz and 6DoF each observed, camera + 6DoF together degrade on the Surface; `docs/XREAL-PC-GATES-G1-G3.md` §12, `docs/CAPABILITIES.md` rows 16-18. Earlier: 2026-10-12, build 0.2.002 (CI fixes: NumPy 2.4 scalar indexing in the pose regression test; version source parsing and writing supports LF/CRLF with regression coverage. Earlier: 2026-10-10 build numbering `MAJOR.MINOR.BUILD` introduced, source `python/eyelab_version.py`, policy `docs/VERSIONING.md`. Earlier: 2026-10-09 gate project `unity_projects/xreal_pc_gates/` built on the desktop per `docs/XREAL-PC-GATES-G1-G3.md` D0–D8 with Unity 6000.6.5f1; harness and vendor-control Windows players ready for the Surface; no glasses run yet, D7 not run. Earlier: gate guide added: Unity CLI build on desktop, G1-G3 runs on Surface. Earlier the same day: all three ADRs reconciled with the PC beta in the revised `docs/AUDIT-2026-10-08.md`. Downloaded 3.1.2 preview contains a Standalone RGB sample; four core DLLs are identical to the September runtime, while the Unity wrapper differs. Desktop NCM/TCP preflight succeeds, but camera/pose operation remains untested. Windows Unity plus a local Python bridge is a candidate deployment, not implemented EyeLab. Emulator categorical exclusion corrected; details in `docs/XREAL-PC-BETA-2026-10-09.md`. Main Unity SDK remains 3.1.0.))))))
>
> 2026-10-08 audit state: F01 Dataset 2420 parsing, F02 pose-lock gates, F03 immutable FramePacket/freshness, F06 planar Kabsch, F11 test lifecycle/Windows CI and documentation findings F05/F08/F13/F14 corrected. F04, F07, F09, F10, F12 remain open. No Beam Pro/compatible Android host. Final display is optical see-through wireframe/points; video is diagnostic. Camera colour is optional; evaluate >=50 mm point spacing, the 1 mm accuracy target and an experimentally determined working range.
>
> Previous updates: 2026-10-06 (one shared config tree `python/config/{cameras,markers,geometry,targets}` + `active.json` via `python/eyelab_paths.py`, per-camera calibration profiles; validation: instrument-aware GT uncertainty with TUR vs the 1 mm goal, setup assistant in the capture preview, optional Surface IMU `python/surface_imu.py`, plate L fold angle from images). Previously 2026-10-05 (validation campaign runner `validation.bat` + guide `docs/validation-campaign.md`; fixed-focus option for calibration and capture). Previously 2026-09-25 (validation package: `python/validation/`, protocol `docs/validation.md`, printables `docs/validation_printables/`; `FrameResult.measured_pose` added. Earlier the same day: XREAL native route: S3b's call chain is complete and clean — perception group → perception → head tracking → 1925 poses — but every pose is the identity with tracking reason `Initializing`, so the tracker never starts; run 3 is done and **refutes the device-clock candidate** — real advancing clock, same 1930 identity poses; only perception id 1 is creatable and the SDK names it `nr_perception_id_remote`; Nebula has since updated the glasses' firmware, so §§11-14 are all pre-update. Continuation: `docs/S3-HANDOVER-2026-09-25.md`, audit §§11-14. XREAL's own FAQ confirms raw camera/IMU need the Enterprise licence). Previously 2026-09-24 (XREAL native route: S2 passed; S3 blocked - the NRSDK licensed-feature list is empty, so raw IMU *and* the grayscale camera are refused with 22, audit §§11-12; S3b takes the ungated fused head pose and now goes through the perception layer recovered on 2026-09-24). Previously 2026-09-23 (XREAL native route: S2 passed, S3 blocked by an NRSDK licensed-feature gate - `NRImuCreate` returns 22 `LicenseFeatureUnsupported`, see audit §11 - and S3b built to take the ungated fused head pose instead; see ADR-003's stage ladder). Previously 2026-09-22 (XREAL native route: S0 done, S1 passed, S2 built - see "Native route status" under "XREAL Eye access routes"; the Status section remains the 2026-05-14 snapshot, audit D1). Previously 2026-09-19 (tier-1 AR loop latency work + XREAL Eye access survey - see "XREAL Eye access routes" and open issue 6; `Start AR` / calibration GUI freeze traced with `python/ar_watchdog.py` to blocking camera opens and a spinning capture reader, both fixed - see open issue 5 and `docs/ar-freeze-diagnostics.md`; 2026-07-10: single-marker sustain via IPPE planar-ambiguity resolution against the Kalman prior — acquire still needs 3; per-marker axis overlay; Geometry Editor tab + `unv_writer.py` for synthetic .unv authoring; `docs/aruco_marker_best_practices.pdf`; Status section below still reflects the 2026-05-14 snapshot and needs a proper revision).

## Status

**Current status lives in two places: `docs/HARDWARE.md` (machines, what each can do, what is ordered) and `docs/CAPABILITIES.md` (dated capability matrix: host, firmware, transport, camera mode, tracking, display, evidence).** Beam Pro has **not arrived** and no compatible Android phone is available (confirmed 2026-10-06 and 2026-10-08). The Python webcam pipeline is an implemented research prototype, not a validated metrological system (`docs/AUDIT-2026-10-08.md`). The Unity project is a scaffold and plan, not an operational XREAL application.

*Historical snapshot (2026-05-14), kept for provenance; where it disagrees with the paragraph above, the paragraph wins:*

- **Hardware in hand (2026-05-14):** XREAL One Pro + XREAL Eye (planned for Aug 2026 — arrived ~3 months early). Beam Pro 8 GB / 256 GB ordered on 2026-05-14; expected arrival then estimated mid-to-late June 2026 (**superseded: not arrived**).
- **Webcam phase (Python, in `python/`):** ChArUco calibration, ArUco detection, pose estimation, registration, wireframe overlay all implemented. Recent activity in `python/.logs/` through Apr 29 2026.
- **Unity project:** `eyelab_xreal/` — fresh URP project for the XREAL One Pro + Eye port, opened successfully in Unity after the repo move. Empty Android APK build passed on 2026-05-11 with Unity 6000.4.5f1. Unity-generated folders (`Library/`, `Logs/`, `UserSettings/`, project files, builds) are ignored. The local `Packages/com.xreal.xr/` SDK folder is also ignored because the SDK package has no license file granting redistribution of modified vendor binaries; install XREAL XR Plugin 3.1.0 locally into that path, then run `tools/patch_xreal_sdk.ps1`.
- **A second Unity project** (`unity_projects/My project`) exists in MR Template (OpenXR / AR Foundation / XR Hands). Not the XREAL target — pending decision (archive / delete / repurpose).

## Verified XREAL XR Plugin 3.1.0 surface

Source: local `eyelab_xreal/Packages/com.xreal.xr/Runtime/Scripts/` SDK install. Namespace `Unity.XR.XREAL`. Static facade `XREALPlugin`. **None of the legacy `NR*` names exist in this version** — see open issue 1.

- **Raw RGB camera (XREAL Eye)** — `XREALPlugin.StartRGBCameraDataCapture(callback)` → `RGBCameraDataFrame { ulong timeStamp, Vector2Int resolution, ulong rawDataSize, IntPtr rawData }`. Poll path: `TryAcquireLatestImage` → `TryGetRGBCameraDataPlane(handle, planeIdx, out IntPtr, out size)`.
- **High-level texture path** — `XREALRGBCameraTexture.CreateSingleton().GetYUVFormatTextures()` → 3× `Texture2D` (Y/U/V, YUV_420_888) ready to bind to materials. Sample: `RGBCameraExample.cs`.
- **Capture pixel formats** — `BGRA32 | NV12 | JPEG | PNG`. Stereo: `CaptureSide { Single, Both, Left, Right }`.
- **AR Foundation extensions** — anchors with persistence, planes, image tracking via `XRReferenceImageLibrary`, mesh subsystem with semantic face classification, session/tracking states.
- **Capture pipeline** — `XREALPhotoCapture`, `XREALVideoCapture`, `XREALAudioCapture`, hardware-encoder bindings, frame blender.
- **Editor mock** — `EditorFrameProvider` provides mock frames so camera scenes can run in Editor without device.
- **Bundled `MarkerTracking` sample** — XREAL-branded interactive cards. **Not ArUco.** Different image targets, different protocol.
- **Local Android build compatibility patch** — in the local, gitignored SDK copy, `Runtime/Plugins/Android/nr_common.aar` was patched so its manifest package is `nrsdk.pack.common`; the original duplicated `nr_loader.aar`'s `nrsdk.pack`, which fails manifest merging under Unity 6000.4.5f1 / Android Gradle tooling. Reapply on other machines with `tools/patch_xreal_sdk.ps1`; the script keeps `nr_common.aar.original` locally as the restore point. The successful empty APK build still warns that `libXREALXRPlugin.so`, `libmedia_codec.so`, and `libVulkanSupport.so` are not 16 KB page-size aligned for some Android 15+ ARM64 devices.

## XREAL Eye access routes (surveyed 2026-09-19)

**Update 2026-10-09:** the Android-only/no-PC-camera statements in the historical
survey below describe the established 3.1.0 route, not all available packages.
The XREAL-hosted 3.1.2 preview archives are accessible and contain a Standalone
RGB camera sample, Windows native libraries and active Standalone camera
bindings. Desktop NCM/TCP 52999 connectivity was observed on the current unit;
actual beta frames/poses were not. Keep the pinned project SDK unchanged and
test the beta in isolation. Four core DLLs match the September runtime exactly;
the Unity wrapper differs. The sample/session lifecycle is the new testable
evidence, not proof that earlier license/initialization blockers are resolved.
The revised [audit](docs/AUDIT-2026-10-08.md) explains the conditional Windows
client/local Python bridge and missing implementation. See
[investigation and Unity test procedure](docs/XREAL-PC-BETA-2026-10-09.md).

The question "can we feed the pipeline from the glasses' own POV?" has three
answers, with different costs and different time horizons.

- **Official, Android only.** XREAL SDK **3.1.0** is still the latest release
  (3.0.0 is the only predecessor). RGB camera access exists solely for the Eye
  on One-series glasses, through the Unity XR plugin
  (`StartRGBCameraDataCapture`, `XREALRGBCameraTexture.GetYUVFormatTextures()`,
  YUV_420_888) - the surface already recorded above. **No PC/Windows camera
  API.** *First Person View* does stream to a PC ("StreamingReceiver", SDK
  >= 3.0.0) but the source must be the Android app and it is a casting path, not
  a sensor feed. Eye hardware: 12 MP f/2.25, video 1600x1200 @30 fps (60
  optional), photo 2016x1512, no battery, 2 GB onboard, firmware tied to One
  system updates.
- **Unofficial, Windows: NCM + TCP.** Over USB-C the glasses expose an NCM
  virtual Ethernet adapter; the glasses answer on `169.254.2.1`, the PC gets
  `169.254.2.10`. TCP **52997 = video**, 52998 = IMU, 52996 = metadata,
  52999 = control. The video is **4-bit grayscale, 512x378 per camera**.
  Reference implementation (Python, numpy/OpenCV/Pillow):
  `Aloim/Grayscale-Feed-Xreal-One-Pro-Eye-Windows-Nebula-Beta-needed-`.
  Requires the **Windows Nebula beta** (access via the XREAL Discord), the beta
  firmware on the glasses, and **Spatial Anchor enabled** in the glasses menu.
  This is the only route that keeps the whole pipeline in Python on the
  Surface.
- **Unofficial, Android: UVC over USB host.** `Aloim/Xreal-One-Pro-Eye-RGB-
  Camera-feed-on-Android` drives VID `0x3318` / PID `0x0436` directly:
  `NRBSPWaitPilotReady()` -> `NRBSPGetUsbConfigAll()` -> `config.uvc0 = 1` ->
  `NRBSPSetUsbConfigAll()` forces a USB re-enumeration (13 -> 17 interfaces)
  that exposes a UVC **bulk** endpoint streaming **H.265 at 2048x1512, ~30-60
  fps**. Needs `libnr_glasses_api.so` / `libota-lib.so` from the official
  Glasses Control APK on the phone. Full RGB resolution, but Android-side.
- **`0xcaff/xr-tools`** (Rust) collects low-level One / One Pro drivers - HID,
  IMU, camera, display control, key input, and calibration-data retrieval. The
  calibration path is worth reading: factory intrinsics would remove a
  calibration step from whichever route we take.

Consequences for the pipeline if we take the Windows NCM route:

- **Resolution and working range - no estimate yet.** *(Rewritten 2026-10-08, audit F14; the earlier text predicted that the working distance "falls by roughly 2.5x".)* That figure compared image widths (512 vs 1280 px) without establishing that the two cameras have the same field of view, and it treated `perspectiveRemovePixelPerCell` as a minimum image size. That parameter is the size of the canonical grid a candidate quadrilateral is resampled to *inside* the detector; it is not a limit on how many pixels a marker needs in the image. What is true: pixels on a marker scale with calibrated focal length (in pixels) times marker size over distance, so a lower-resolution sensor lowers the focal length in pixels unless its field of view is narrower, and a 4-bit quantised image lowers contrast and corner accuracy. The working range is therefore **determined empirically** from the calibrated focal length of the delivered mode, exposure, contrast, and detection-rate and accuracy sweeps over distance and angle, for the required point spacing (>= 50 mm) and the 1 mm accuracy target (statistic still to be agreed). Print a larger marker set for the first sweeps; do not size anything from the 2.5x figure.
- **Calibration.** New intrinsics at 512x378 through that feed, stored as its
  own YAML; `describe_resolution_mismatch` already refuses a mismatch. 4-bit
  quantisation will degrade sub-pixel corner accuracy, so expect more pose
  noise than the webcam baseline.
- **Architecture.** The Eye arrives as a *frame source*, not as a camera index:
  `ArucoPipeline` should take an object with `read()` / `stats()` / `stop()` /
  `source_id`, which `ThreadedCapture` already almost satisfies. That mirrors
  `IArucoPoseBridge` on the Unity side and keeps the webcam path untouched.
- **Nebula in the display path.** Nebula is a spatial-desktop compositor. It
  may be needed to obtain the beta firmware, but whether it must *run* for the
  NCM services to answer is unverified - and if the AR overlay is presented
  inside Nebula's virtual desktop it adds a compositing stage in front of the
  panel. Determine this before trusting any latency measurement taken with
  Nebula running.

### Native route status (2026-09-22)

Tracked stage by stage in ADR-003 (`docs/adr/0003-eye-pov-rigid-coupling.draft.md`).

- **S0 done** — `libnr_glasses_api.dll` (Tier A) loads standalone on Windows,
  glasses enumerate as `3318:0436`, the same identity the Android UVC route
  targets. The "Android only" label on that route above is therefore about
  where it has been *demonstrated*, not where the library exists; the unlock
  itself is still untested on Windows. Firmware version unread (signature).
- **S1 passed** — glasses as a plain DP monitor: all lines and colours legible;
  the electrochromic tint has a floor at level 1 of 3 (ADR-003).
- **S2 built, not run** — Tier B staged in `vendor/xreal/win-x64/`; the
  vendor host's bring-up sequence and the six signatures S2 calls recovered
  statically (`docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md` §9);
  `python/xreal_native_probe.py`. Entry point is `libnr_loader.dll`, and
  `NRAPIInitSetStandalone` is not part of the sequence.

## Workspace / collaboration constraints

- I (Claude) have file-level Read/Write/Edit access to `C:\Users\gcmdn\GitHub_Uni\EyeLab\`, including `eyelab_xreal/`. I author C# scripts, edit `Packages/manifest.json`, `ProjectSettings`, and read Unity `Logs/`.
- I cannot launch the Unity Editor, hit Play, build APKs, or deploy to a device. Loop: I write code → Jack runs Play/build in Unity → I read `Logs/Editor.log` and (eventually) `adb logcat` output to iterate.
- Python: virtualenv preferred (`python/venv/` already exists).
- Surface Pro 7 pre-Beam testing has established the Windows display path; webcam and Editor/mock work are also available. *(Revised 2026-10-09.)* The Python WebSocket service is planned, not implemented. The pinned SDK 3.1.0 deployment targets Android; the separately downloaded PC beta contains a Windows camera sample. Whether that sample delivers Eye frames and usable tracking on this unit's current firmware remains untested. The next gate is the isolated vendor sample, followed by the conditional Windows client/local Python bridge described in `docs/AUDIT-2026-10-08.md`.

## Deployment / host-device decision (2026-05-08)

- **Build target:** generic Android APK, NOT Beam-Pro-specific. Long-term vision: distribute the APK so any student with a compatible Android phone + XREAL One Pro can run it.
- **Primary lab setup:** phone-as-host connected to XREAL One Pro via USB-C DP-Alt, paired (over local network) with a Windows PC running Simcenter Testlab + the .NET COM/WebSocket bridge.
- **Hardware power note (deployment, not software):** most phone DP-Alt ports can't simultaneously deliver enough power to drive the glasses while running the app. A DP+PD splitter plus a 30 W+ PD power bank is required for sustained sessions. User-manual concern, not architecture.
- **Beam Pro:** ordered 2026-05-14; **has not arrived** (2026-10-08). Keep the APK generic Android and do not write Beam-specific intents/permissions into the manifest.
- **iOS:** out of scope. XREAL One Pro works as a display from iPhone 15+ but XREAL has no public iOS SDK for camera/SLAM access.

## Glossary (deltas / clarifications beyond `.docs/PROJECT.md` and `.docs/THEORY.md`)

- **`IArucoPoseBridge`** — Unity-side C# interface introduced by ADR-001. Three implementations (`WebSocketPoseBridge`, `NativePluginPoseBridge`, `OpenCVForUnityBridge`) plus `MockPoseBridge` for Editor smoke tests. The rest of the Unity app only depends on the interface, so swapping bridges is a 1–2 day change rather than a rewrite.
- **Step C / Step A / Step B** — the three rungs of ADR-001's OpenCV bridging staircase. C: off-device Python service over WebSocket. A: DIY native .aar. B: OpenCV for Unity ($95).
- **EyeLab Service** (preliminary name) — the standalone Python WebSocket process running an extended `webcam_pipeline.py`. Receives frames, returns poses. Lives on the Testlab PC at lab and on Jack's Surface Pro 7 for remote dev.

## Resolved decisions (2026-05-08)

- **Build target:** generic Android APK; not Beam-Pro-specific.
- **OpenCV bridge:** staircase **C → A → B**, captured in [`docs/adr/0001-opencv-bridging-staircase.md`](docs/adr/0001-opencv-bridging-staircase.md). Pivot from C is quality-driven (latency, accuracy, Wi-Fi reliability), not calendar-driven; project continuation at Siemens removes the "never leave C" risk.
- **Editor iteration:** USB webcam in Editor → `WebCamTexture` → JPEG → same WebSocket service used on-device. `MockPoseBridge` for camera-free smoke tests. Recorded-clip regression harness (deterministic replay into both Python service and Unity, accuracy assertions) added in week 2 of bring-up. Cameras identify themselves to the Python service in the connection handshake (`unity-editor-webcam-<model>` vs `unity-android-xreal-eye`) so the service picks the right calibration YAML.
- **Frame transport protocol (v0):**
  - **Pixel data:** grayscale Y-plane, 8-bit single channel, JPEG quality 85, **1280×720** baseline (configurable up to 1080p).
  - **Frame rate:** 15 fps initial, configurable up to 30. Bandwidth ~5–7 Mbps.
  - **Wire format (corrected 2026-10-08, audit F08):** uplink = WS *binary* frame, **20-byte** header (`<4sB3xIQ`: magic/version/3 reserved/seq/`timestamp_ns`) + JPEG bytes. Downlink = WS *text* frame, snake_case JSON; a `pose` carries `{ type, schema, session_id, seq, timestamp_ns, clock_id, rvec, tvec, marker_ids, rms_reproj_px, registration_rms_mm, latency_ms, calibration_hash, geometry_hash }`, and an explicit `no_pose` replaces silence. The executable definition (schema + golden fixtures) is specified in `docs/tasks/T2.6-implementation-checklist.md` §1.2.
  - **Handshake:** client sends `hello` JSON; service replies with `ready` carrying calibration intrinsics + ArUco board config. Eliminates the "two copies of calibration drift apart" risk; Unity does not own a YAML.
  - **Heartbeat:** WS ping/pong every 5 s.
  - **Reconnect:** exponential backoff 250 ms → 5 s cap. Re-handshake on each reconnect.
  - **Concurrency:** sync from Unity's POV (`EstimatePoseAsync`); 200 ms timeout per frame. Service keeps depth-2 frame queue, drops oldest on overflow.
  - **Observability:** every pose response includes `latency_ms`; Unity logs end-to-end latency. Both flushed to CSV per run for the regression harness.

## Open issues still to resolve

1. **T2.6 obsolete SDK names — substantive rewrite needed.** Use the `Unity.XR.XREAL.*` surface (verified above) and the new staircase architecture. ADR-001's "Decisions this ADR forces in T2.6" lists the specific sections. Implementation checklist captured in [`.docs/tasks/T2.6-implementation-checklist.md`](.docs/tasks/T2.6-implementation-checklist.md).
2. **`My project` (MR Template) folder.** Recommended: rename to `My project (archived OpenXR experiment)` and leave alone — keeps the option open for the WP5 Sony port.
3. **Pose application timing in Unity — RESOLVED 2026-05-09.** Chosen: **option (iii)** — plain GameObject `StructureAnchor` whose pose is updated on each ArUco pose received, EWMA-filtered (α=0.4 warmup, 0.15 steady-state) with outlier rejection (>3σ delta). Head-pose ring buffer (~1 s) keyed by an explicitly mapped clock lets us compose `T_world_from_structure(t_capture) = T_world_from_head(t_capture) * T_head_from_camera * T_camera_from_structure(t_capture)` at the time the frame was actually captured; times the buffer does not bracket are rejected, not clamped (audit F05). The head-camera extrinsic is an identity *placeholder* for v0 (an unquantified bias); Phase 8 now means a verified extrinsic plus a separately specified per-eye display calibration, not a four-point "SPAAM nudge" (audit F13). **Future upgrade path:** option (i) AR Foundation `XRAnchor` is reserved as the backup if SLAM drift over long sessions becomes measurable; the `IStructureAnchor` interface in Phase 5 leaves room for swap. Documented in [`docs/tasks/T2.6-implementation-checklist.md`](docs/tasks/T2.6-implementation-checklist.md) Phase 5.
4. **Camera intrinsics for XREAL Eye — DESIGN RESOLVED.** Calibration scene `10_Calibration.unity` runs the service in `mode:calibration`; service runs `cv2.aruco.calibrateCameraCharuco` and stores YAML keyed by client ID. Documented in [`docs/tasks/T2.6-implementation-checklist.md`](docs/tasks/T2.6-implementation-checklist.md) Phase 6.

5. **`Start AR` / calibration froze the GUI - root cause found and fixed (2026-09-19, awaiting retest).** The AR watchdog log (`.logs/ar_debug_20260919_113447.jsonl`) showed two defects: `cv2.VideoCapture(index, CAP_DSHOW)` blocking the tk main thread for up to 32 s at every call site (startup scan, calibration window, `Start AR`), and a capture that reported `isOpened()` while never delivering a frame, read in a loop with no pause - 70 905 456 failed reads in 30 s, which starved the UI and left the webcam needing a replug. Camera opens and scans now run on worker threads with a delivery check and backend fallback (`camera_utils.CameraOpener`, `open_camera_result`), and the reader paces retries and gives up with a reportable error. Full evidence and the remaining freeze candidates: [`docs/ar-freeze-diagnostics.md`](docs/ar-freeze-diagnostics.md). **Open:** why this webcam delivers no frames through DSHOW at all - the next run's `camera_open_result` record names the backend that works.

6. **AR loop latency - tier 1 landed 2026-09-19, tier 2 open.** Measured from
   `.logs/ar_debug_20260919_115730.jsonl`: `ui_fps` 20-24 idle, 10.5-15 with
   two or three markers in view; `frame.process` 3-12 ms idle and 25-56 ms with
   markers; `frame.display` 4.5-5.5 ms. Tier 1 (done): `Loop ms` reinterpreted
   as a frame period rather than an idle gap, the occluded notebook canvas no
   longer painted, `refineDetectedMarkers` duty-cycled, CLAHE object cached,
   per-stage `timings_ms` added so the next measurement can attribute cost.
   **Open (tier 2):** decouple display from detection - display at capture rate
   on the main thread, detection on a worker, overlay drawn from the most
   recent pose, so the *video* stops waiting for ArUco and only the overlay
   lags one detection period (this is the async shape ADR-001 already specifies
   for Unity); and detect at reduced resolution with scaled intrinsics,
   refining corners at full resolution, which is also the rehearsal for a
   512x378 Eye feed. **Also open:** the confound - the X1 reprojects the
   DisplayPort image against head motion, which adds perceived latency that no
   OpenCV change can touch. Compare the GUI's own FPS readout on the host panel
   against the same run seen through the glasses before attributing lag to the
   pipeline.

## Doc visibility (2026-05-09)

- **Committed (`docs/`):** ADRs, T2.6 implementation checklist, and any future agent-facing implementation guides. Reach Surface Pro 7 / future contributors via `git pull`.
- **Private (`.docs/`, gitignored):** PROJECT.md, TASKS.md, THEORY.md, the original task specs (T1.2 / T1.3 / T2.1 / T2.2 / T2.3 / T3.1 / T4.1), and `.archive/`. Contain Siemens collaboration scope, dates, prices, internal references — not for the public repo. Mirror these to Surface Pro 7 manually (OneDrive / Dropbox / cp).
- The `.docs/` files that have committed `docs/` counterparts are now **redirect stubs** so there is one source of truth.

## Portable distribution (Phase 1A)

- The EyeLab Service ships as a download-and-run zip from GitHub Releases (USMA pattern). Embedded Python 3.11 + pre-installed wheels + .bat launchers + default calibration YAMLs. No host Python needed, no admin rights, no internet at runtime.
- Build script `tools/build_portable.py` (run on dev machine) downloads the embed Python, configures `python311._pth` to enable site-packages, bootstraps pip, installs `python/requirements.txt` into the bundle, copies project sources, writes launchers, zips to `dist/EyeLab-portable-vX.Y.Z.zip`.
- Coexists with the existing `run_eyelab.bat` (venv-from-system-Python, for dev workflow).
- Surface Pro 7 can use either: pull from git for source, OR download the latest portable bundle from Releases.

## How this file relates to the rest

- `.docs/PROJECT.md` — authoritative scope, architecture, milestones (19-month UNIVPM × Siemens DI collaboration).
- `.docs/THEORY.md` — theory reference (camera model, ArUco, PnP, Kabsch, SLAM, hammer tracking).
- `.docs/TASKS.md` — full task plan, WP1–WP6, T1.1–T6.3.
- `.docs/tasks/T2.6-xreal-port.md` — implementation guide for the XREAL port. **Pending substantive rewrite** (open issue 1).
- `.docs/adr/` — Architecture Decision Records.
- `python/` — webcam-phase code; will be extended to the EyeLab Service for Step C.
- `EyeLab - Siemens XR Project/` — reserved sub-folder for AR-app-specific docs.
