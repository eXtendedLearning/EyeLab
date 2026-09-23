# EyeLab — domain context (XREAL phase)

> Living glossary and shared-mental-model notes. Captures terms, decisions, and open issues surfaced during grilling. Authoritative scope/plan lives in `.docs/PROJECT.md`, theory in `.docs/THEORY.md`, task plan in `.docs/TASKS.md`, decisions in `.docs/adr/`. This file holds **deltas the project plan does not capture** plus the live "open question" board.
>
> Last updated: 2026-09-22 (XREAL native route: S0 done, S1 passed, S2 built - see "Native route status" under "XREAL Eye access routes"; the Status section remains the 2026-05-14 snapshot, audit D1). Previously 2026-09-19 (tier-1 AR loop latency work + XREAL Eye access survey - see "XREAL Eye access routes" and open issue 6; `Start AR` / calibration GUI freeze traced with `python/ar_watchdog.py` to blocking camera opens and a spinning capture reader, both fixed - see open issue 5 and `docs/ar-freeze-diagnostics.md`; 2026-07-10: single-marker sustain via IPPE planar-ambiguity resolution against the Kalman prior — acquire still needs 3; per-marker axis overlay; Geometry Editor tab + `unv_writer.py` for synthetic .unv authoring; `docs/aruco_marker_best_practices.pdf`; Status section below still reflects the 2026-05-14 snapshot and needs a proper revision).

## Status

- **Hardware in hand:** XREAL One Pro + XREAL Eye (planned for Aug 2026 — arrived ~3 months early). Beam Pro 8 GB / 256 GB ordered on 2026-05-14; expected arrival roughly mid-to-late June 2026.
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

- **Resolution.** 512x378 is 0.4x linear against the current 1280x720. A 20 mm
  marker spanning ~40 px becomes ~16 px, and a 4x4 ArUco with
  `perspectiveRemovePixelPerCell = 6` wants ~36 px of marker side to decode
  comfortably: expect the working distance to fall by roughly 2.5x. Print a
  40-50 mm marker set before the first session.
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
- Surface Pro 7 pre-Beam testing is limited to Windows display output through the glasses, Unity Editor/mock modes, normal webcam input, and the Python WebSocket service. It cannot exercise real XREAL Eye camera, SLAM/head tracking, anchors, planes, meshing, or hand tracking because those require the Android XREAL runtime on Beam Pro or another supported Android host.

## Deployment / host-device decision (2026-05-08)

- **Build target:** generic Android APK, NOT Beam-Pro-specific. Long-term vision: distribute the APK so any student with a compatible Android phone + XREAL One Pro can run it.
- **Primary lab setup:** phone-as-host connected to XREAL One Pro via USB-C DP-Alt, paired (over local network) with a Windows PC running Simcenter Testlab + the .NET COM/WebSocket bridge.
- **Hardware power note (deployment, not software):** most phone DP-Alt ports can't simultaneously deliver enough power to drive the glasses while running the app. A DP+PD splitter plus a 30 W+ PD power bank is required for sustained sessions. User-manual concern, not architecture.
- **Beam Pro:** ordered 2026-05-14 and expected mid-to-late June 2026. Keep the APK generic Android and do not write Beam-specific intents/permissions into the manifest.
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
  - **Wire format:** uplink = WS *binary* frame, 16-byte header (magic/version/seq/timestampNs) + JPEG bytes. Downlink = WS *text* frame, JSON `{ type, seq, timestampNs, rvec, tvec, marker_ids, rms_reproj_px, registration_rms_mm, latency_ms }`.
  - **Handshake:** client sends `hello` JSON; service replies with `ready` carrying calibration intrinsics + ArUco board config. Eliminates the "two copies of calibration drift apart" risk; Unity does not own a YAML.
  - **Heartbeat:** WS ping/pong every 5 s.
  - **Reconnect:** exponential backoff 250 ms → 5 s cap. Re-handshake on each reconnect.
  - **Concurrency:** sync from Unity's POV (`EstimatePoseAsync`); 200 ms timeout per frame. Service keeps depth-2 frame queue, drops oldest on overflow.
  - **Observability:** every pose response includes `latency_ms`; Unity logs end-to-end latency. Both flushed to CSV per run for the regression harness.

## Open issues still to resolve

1. **T2.6 obsolete SDK names — substantive rewrite needed.** Use the `Unity.XR.XREAL.*` surface (verified above) and the new staircase architecture. ADR-001's "Decisions this ADR forces in T2.6" lists the specific sections. Implementation checklist captured in [`.docs/tasks/T2.6-implementation-checklist.md`](.docs/tasks/T2.6-implementation-checklist.md).
2. **`My project` (MR Template) folder.** Recommended: rename to `My project (archived OpenXR experiment)` and leave alone — keeps the option open for the WP5 Sony port.
3. **Pose application timing in Unity — RESOLVED 2026-05-09.** Chosen: **option (iii)** — plain GameObject `StructureAnchor` whose pose is updated on each ArUco pose received, EWMA-filtered (α=0.4 warmup, 0.15 steady-state) with outlier rejection (>3σ delta). Head-pose ring buffer (~60 entries, ~1 s) keyed by frame timestamp lets us compose the world-frame anchor pose at the time the frame was actually captured. Head-camera offset is identity for v0; SPAAM nudge step adds the real offset in week 3 (Phase 8). **Future upgrade path:** option (i) AR Foundation `XRAnchor` is reserved as the backup if SLAM drift over long sessions becomes measurable; the `IStructureAnchor` interface in Phase 5 leaves room for swap. Documented in [`docs/tasks/T2.6-implementation-checklist.md`](docs/tasks/T2.6-implementation-checklist.md) Phase 5.
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
