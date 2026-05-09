# EyeLab — domain context (XREAL phase)

> Living glossary and shared-mental-model notes. Captures terms, decisions, and open issues surfaced during grilling. Authoritative scope/plan lives in `.docs/PROJECT.md`, theory in `.docs/THEORY.md`, task plan in `.docs/TASKS.md`, decisions in `.docs/adr/`. This file holds **deltas the project plan does not capture** plus the live "open question" board.
>
> Last updated: 2026-05-08 (XREAL One Pro + Eye received).

## Status

- **Hardware in hand:** XREAL One Pro + XREAL Eye (planned for Aug 2026 — arrived ~3 months early).
- **Webcam phase (Python, in `python/`):** ChArUco calibration, ArUco detection, pose estimation, registration, wireframe overlay all implemented. Recent activity in `python/.logs/` through Apr 29 2026.
- **Unity project:** `C:\Users\gcmdn\unity_projects\xreal_test` — fresh URP project with `com.xreal.xr@3.1.0` installed locally from `com.xreal.xr.tar.gz`. Default scene only, no app code yet. The empty repo sub-folder `EyeLab - Siemens XR Project/` is reserved for AR-app-specific documentation and screenshots; the Unity project itself lives outside this repo to avoid bloating it with `Library/` and `Logs/`.
- **A second Unity project** (`unity_projects/My project`) exists in MR Template (OpenXR / AR Foundation / XR Hands). Not the XREAL target — pending decision (archive / delete / repurpose).

## Verified XREAL XR Plugin 3.1.0 surface

Source: `unity_projects/xreal_test/Packages/com.xreal.xr/Runtime/Scripts/`. Namespace `Unity.XR.XREAL`. Static facade `XREALPlugin`. **None of the legacy `NR*` names exist in this version** — see open issue 1.

- **Raw RGB camera (XREAL Eye)** — `XREALPlugin.StartRGBCameraDataCapture(callback)` → `RGBCameraDataFrame { ulong timeStamp, Vector2Int resolution, ulong rawDataSize, IntPtr rawData }`. Poll path: `TryAcquireLatestImage` → `TryGetRGBCameraDataPlane(handle, planeIdx, out IntPtr, out size)`.
- **High-level texture path** — `XREALRGBCameraTexture.CreateSingleton().GetYUVFormatTextures()` → 3× `Texture2D` (Y/U/V, YUV_420_888) ready to bind to materials. Sample: `RGBCameraExample.cs`.
- **Capture pixel formats** — `BGRA32 | NV12 | JPEG | PNG`. Stereo: `CaptureSide { Single, Both, Left, Right }`.
- **AR Foundation extensions** — anchors with persistence, planes, image tracking via `XRReferenceImageLibrary`, mesh subsystem with semantic face classification, session/tracking states.
- **Capture pipeline** — `XREALPhotoCapture`, `XREALVideoCapture`, `XREALAudioCapture`, hardware-encoder bindings, frame blender.
- **Editor mock** — `EditorFrameProvider` provides mock frames so camera scenes can run in Editor without device.
- **Bundled `MarkerTracking` sample** — XREAL-branded interactive cards. **Not ArUco.** Different image targets, different protocol.

## Workspace / collaboration constraints

- I (Claude) have file-level Read/Write/Edit access to `C:\Users\gcmdn\unity_projects\xreal_test` and `C:\Users\gcmdn\GitHub_Uni\EyeLab\`. I author C# scripts, edit `Packages/manifest.json`, `ProjectSettings`, and read `Logs/`.
- I cannot launch the Unity Editor, hit Play, build APKs, or deploy to a device. Loop: I write code → Jack runs Play/build in Unity → I read `Logs/Editor.log` and (eventually) `adb logcat` output to iterate.
- Python: virtualenv preferred (`python/venv/` already exists).

## Deployment / host-device decision (2026-05-08)

- **Build target:** generic Android APK, NOT Beam-Pro-specific. Long-term vision: distribute the APK so any student with a compatible Android phone + XREAL One Pro can run it.
- **Primary lab setup:** phone-as-host connected to XREAL One Pro via USB-C DP-Alt, paired (over local network) with a Windows PC running Simcenter Testlab + the .NET COM/WebSocket bridge.
- **Hardware power note (deployment, not software):** most phone DP-Alt ports can't simultaneously deliver enough power to drive the glasses while running the app. A DP+PD splitter plus a 30 W+ PD power bank is required for sustained sessions. User-manual concern, not architecture.
- **Beam Pro:** budget-permitting upgrade path; do not write Beam-specific intents/permissions into the manifest.
- **iOS:** out of scope. XREAL One Pro works as a display from iPhone 15+ but XREAL has no public iOS SDK for camera/SLAM access.

## Glossary (deltas / clarifications beyond `.docs/PROJECT.md` and `.docs/THEORY.md`)

- **`IArucoPoseBridge`** — Unity-side C# interface introduced by ADR-001. Three implementations (`WebSocketPoseBridge`, `NativePluginPoseBridge`, `OpenCVForUnityBridge`) plus `MockPoseBridge` for Editor smoke tests. The rest of the Unity app only depends on the interface, so swapping bridges is a 1–2 day change rather than a rewrite.
- **Step C / Step A / Step B** — the three rungs of ADR-001's OpenCV bridging staircase. C: off-device Python service over WebSocket. A: DIY native .aar. B: OpenCV for Unity ($95).
- **EyeLab Service** (preliminary name) — the standalone Python WebSocket process running an extended `webcam_pipeline.py`. Receives frames, returns poses. Lives on the Testlab PC at lab and on Jack's Surface 6 for remote dev.

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

## Doc visibility (2026-05-09)

- **Committed (`docs/`):** ADRs, T2.6 implementation checklist, and any future agent-facing implementation guides. Reach Surface 6 / future contributors via `git pull`.
- **Private (`.docs/`, gitignored):** PROJECT.md, TASKS.md, THEORY.md, the original task specs (T1.2 / T1.3 / T2.1 / T2.2 / T2.3 / T3.1 / T4.1), and `.archive/`. Contain Siemens collaboration scope, dates, prices, internal references — not for the public repo. Mirror these to Surface 6 manually (OneDrive / Dropbox / cp).
- The `.docs/` files that have committed `docs/` counterparts are now **redirect stubs** so there is one source of truth.

## Portable distribution (Phase 1A)

- The EyeLab Service ships as a download-and-run zip from GitHub Releases (USMA pattern). Embedded Python 3.11 + pre-installed wheels + .bat launchers + default calibration YAMLs. No host Python needed, no admin rights, no internet at runtime.
- Build script `tools/build_portable.py` (run on dev machine) downloads the embed Python, configures `python311._pth` to enable site-packages, bootstraps pip, installs `python/requirements.txt` into the bundle, copies project sources, writes launchers, zips to `dist/EyeLab-portable-vX.Y.Z.zip`.
- Coexists with the existing `run_eyelab.bat` (venv-from-system-Python, for dev workflow).
- Surface 6 can use either: pull from git for source, OR download the latest portable bundle from Releases.

## How this file relates to the rest

- `.docs/PROJECT.md` — authoritative scope, architecture, milestones (19-month UNIVPM × Siemens DI collaboration).
- `.docs/THEORY.md` — theory reference (camera model, ArUco, PnP, Kabsch, SLAM, hammer tracking).
- `.docs/TASKS.md` — full task plan, WP1–WP6, T1.1–T6.3.
- `.docs/tasks/T2.6-xreal-port.md` — implementation guide for the XREAL port. **Pending substantive rewrite** (open issue 1).
- `.docs/adr/` — Architecture Decision Records.
- `python/` — webcam-phase code; will be extended to the EyeLab Service for Step C.
- `EyeLab - Siemens XR Project/` — reserved sub-folder for AR-app-specific docs.
