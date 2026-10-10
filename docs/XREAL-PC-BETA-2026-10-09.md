# XREAL PC beta investigation — 2026-10-09

## Result

The PC SDK lead is actionable. Both 3.1.2 preview packages download directly
from XREAL's public resource domain. Static inspection finds a Standalone RGB
camera sample, Windows native libraries, and camera bindings compiled for
`UNITY_STANDALONE`. Camera frames and head tracking have **not** been observed
with this beta on this unit. The main Android project remains on SDK 3.1.0.
Four core native DLLs are identical to those already probed in September;
the Unity wrapper differs. The new opportunity is to test the supplied vendor
scene/session lifecycle, not a demonstrated fix or a wholly new runtime.
The [current project audit](AUDIT-2026-10-08.md#xreal-reassessment---9-october-2026)
reconciles this evidence with all ADRs and EyeLab's optical-overlay objective.

## Observed on the connected desktop

- Host: LAGANN_0526. USB device `3318:0436` is present.
- `UsbNcm Host Device` is Up; local IPv4 is `169.254.2.10/24`.
- A TCP connection to `169.254.2.1:52999` succeeds. The connection was closed
  without sending application bytes. This repeats the beta session manager's
  connectivity preflight, not a camera or tracking test.
- HID and audio interfaces enumerate. The separate CDC ECM interface reports
  Error; the working NCM interface does not share that status. Cause untested.
- Windows lists two desktop monitors and the ordinary HD Web Camera. No XREAL
  display or standard Windows Eye camera was enumerated in this check.
- Firmware version remains unknown. No firmware, driver, display-mode or USB
  configuration change was performed. No beta DLL was executed.

## Downloaded artifacts and source evidence

Source: [PC SDK beta document](https://github.com/dengxian-xreal/xreal.github.io/blob/main/docs/14_PC%20SDK%20%28Beta%29.md).
Its template link returned 404 in the preceding investigation. The packages
the document links are available independently:

- [Core package](https://public-resource.xreal.com/download/XREALSDK_Preview_3.1.2.20260206/20260206_185349_3.1.2_Preview/com.xreal.xr.tar.gz)
  — 239525465 bytes; SHA-256
  `FC623BD0BA9AE06A62C8FE6D251F29859AFA42E1D0EFB23C561415D1281243C3`.
- [Experimental package](https://public-resource.xreal.com/download/XREALSDK_Preview_3.1.2.20260206/20260206_185349_3.1.2_Preview/com.xreal.xr.experimental.tar.gz)
  — 77103963 bytes; SHA-256
  `24AAC59379F77DDA689B361C2A97E0771923E1B598AEB7589090D0E2B07928CD`.

Local archives and extracted packages are in the gitignored directory
`vendor/xreal/pc-beta-3.1.2/`. Hashes identify downloaded content, not a vendor
signature. Both manifests report 3.1.2. No redistribution licence was located;
keep binaries local. General support status and firmware compatibility remain
unconfirmed.

Paths below are relative to the extracted package roots:

| Evidence | What it establishes |
|---|---|
| Experimental `Samples~/Standalone/RGBCamera/RGBCamera.unity` and `RGBCameraExample.cs` | An actual desktop camera sample exists. Play/Stop invoke capture; Update displays Y/U/V textures. |
| Core `Runtime/Scripts/XREALRGBCamera.cs`, lines 1–3, 69–75, 112–120 | Standalone builds call native camera capture/acquisition; these are not the unsupported-platform fallback branches. |
| Core `Runtime/Scripts/Android/Camera Features/XREALRGBCameraTexture.cs` | Despite the directory name, the class is not Android-preprocessor-only. It exposes resolution, timestamp and frame-update events. |
| Experimental `Runtime/Plugins/X64/` | Windows DLLs include XREALXRPlugin, loader, API, glasses API, 6DoF plugin and FFmpeg dependencies. Presence does not prove operational support. |
| Experimental `Runtime/Standalone/XREALSessionManager.cs`, lines 115–211 | One-series startup checks TCP 52999, creates the vendor session and waits for glasses startup completion. |
| Experimental `Samples~/Standalone/HelloMR/HelloMR.cs` | Offers tracking-mode changes, including 6DoF. Enter XR also changes display mode/topology, so it is a separate test from camera acquisition. |

`NativeOTA` includes firmware-update methods, but no C# call to `.StartOTA(`
was found in these packages. This is a source inspection, not a guarantee about
every native implementation. Do not initiate firmware updates for this test.

The web guide's broad statement that PC AR features are unsupported does not
justify concluding that camera capture is absent: the archive contains a
specific standalone capture sample. Runtime success remains the deciding gate.

## Comparison with the previously explored PC runtime

SHA-256 was computed for the downloaded experimental package's
`Runtime/Plugins/X64/` files and the previously staged files in
`vendor/xreal/win-x64/`. All four pairs below are byte-for-byte identical:

| DLL | Bytes | SHA-256 (both copies) |
|---|---:|---|
| `libnr_api.dll` | 46487040 | `3586C14119C75A17FE2853123CC2E8874692B7FA80111707496E526D32BB1D69` |
| `libnr_loader.dll` | 6345728 | `FAA64E1687F82399996736F6E89E522959953E72E08FBB1251F5A1B77B683A63` |
| `libnr_glasses_api.dll` | 3518811 | `FFD4AF6885CC3992D79C3DEB9A0F5FC1B22F0113A6DB29FA9C0F0FF52C97FA5F` |
| `libnr_plugin_6dof.dll` | 20788224 | `37CB22858D0B6855D2EED3E4DB8A1E5CF3646F7AE765D1ED422E70BCA8995861` |

The Unity wrapper `XREALXRPlugin.dll` differs:

- Beta experimental package: 1144832 bytes,
  `B21ADCCBD57EC01848F397D6517C36F542AF71CBB505BA5E9B18505C7B13C4EF`.
- `Nebula_Windows_20260205/Nebula for Windows_Data/Plugins/x86_64/`:
  803328 bytes,
  `AEFBBD8CEEC0DFA66B51308EDFC006932C887F44B438ED3A1583CC9CB8A53191`.

The September investigation already found managed 3.1.2 components. Its direct
native probes encountered raw IMU/grayscale refusal (22) and fused poses stuck
in `Initializing`, including after the device-clock hypothesis was tested.
That evidence is not a runtime test of this RGB scene. The old static export
analysis listed `NRRgbCameraCreate` as ungated; this does not prove that it
works. The intervening firmware update and different Unity wrapper prevent
treating these as identical end-to-end experiments, while identical core DLLs
prevent claiming that the beta supplies a new core implementation.

The scene plus session manager provides a concrete way to test vendor startup
orchestration, which the independent Python DLL probes did not reproduce in
full. Whether it resolves the earlier initialization failure is untested.

## Next experiment: isolated Unity camera sample

`RGBCamera.unity` is a scene asset opened inside a Unity project. Play mode is
an Editor experiment; building it for Windows later creates an executable and
supporting runtime files that can run on the Surface without the Editor. The
sample supplies image acquisition/display, not EyeLab's Python bridge, UNV
registration or calibrated optical overlay. Those require implementation after
the relevant hardware gates pass.

Jack runs Unity; an agent must not launch the Editor, enter Play mode or build.
Use a separate Unity 6 project, not an SDK migration of `eyelab_xreal`.

1. Select Windows Standalone as the active build target. Install the two local
   beta tarballs through Package Manager and UniTask via the URL in the beta
   guide. Import **Standalone** from the experimental package.
2. Enable XREAL XR Loader under the Standalone tab; leave Initialize XR on
   Startup unchecked. Resolve compilation/import errors before connecting the
   sample lifecycle. For URP rendering, follow the guide's embedding-renderer
   feature configuration; camera acquisition is the first test.
3. Open the imported `Standalone/RGBCamera/RGBCamera.unity`. Close competing
   XREAL applications first. Record firmware from an existing version UI without
   accepting an update. Enter Play and use the camera Play control. Do not use
   Enter XR for the initial camera-only attempt: its handler changes EDID,
   monitor extension and primary-monitor selection.
4. Capture the Unity Console/Editor log and verify a live image changes with
   scene motion, delivered resolution, advancing frame timestamps and frame
   update rate. A successful StartCapture return or non-null texture alone is
   insufficient. Stop capture and check a clean restart.
5. If camera-only operation needs the DP path, repeat on the Surface. A desktop
   failure without DP does not establish a general Windows failure. Next test
   tracking and display on the Surface with the HelloMR sample, recording real
   translation and rotation rather than only a mode label.
6. After Editor capture succeeds, build a Windows x86-64 player and repeat
   capture/restart on the Surface. This is a separate gate: Editor success is
   not proof that the packaged application includes and initializes every
   required plugin. Verify concurrent capture and tracking before integrating
   the local Python service behind `IArucoPoseBridge`.

Acceptance: actual Eye frames first, then valid changing head pose and its clock,
then concurrent capture/tracking. Camera calibration and per-eye optical overlay
validation follow; the sample image is diagnostic, not a registered overlay.

## WSL + Android emulator assessment

This is not a faithful substitute for a DP-capable Android phone by default.
The Surface's physical DP output supplies a Windows display. Rendering an
emulator window there does not hand Android the physical display controller,
display discovery, mode changes, timing or XREAL runtime integration.

- [Microsoft USB/IP documentation](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)
  describes USB-device forwarding into WSL. It does not provide DP Alt Mode
  display-controller passthrough. USB and DP are separate paths on the cable.
- [WSLg architecture](https://github.com/microsoft/wslg#wslg-architecture-overview)
  presents Linux application windows through Weston/RDP integration into the
  Windows desktop. Hardware-accelerated graphics is not physical display
  ownership by the Android guest. This distinction is our architectural
  inference from that documented pipeline.
- [Google emulator release notes](https://developer.android.com/studio/releases/emulator)
  document Windows USB passthrough in 30.0.26 and ARM translation in Android 11
  x86_64 images. Therefore the older categorical claims "USB passthrough is
  Linux-only" and "ARM64 libraries alone rule out an emulator" are withdrawn.
  Neither feature establishes compatibility with XREAL's native libraries,
  USB interfaces or external-display runtime.
- [Google acceleration requirements](https://developer.android.com/studio/run/emulator-acceleration)
  distinguish architecture-matched VM acceleration from CPU translation and
  document restrictions for running accelerated emulators inside another VM.
  WSL adds nested-virtualization and USB-forwarding dependencies. Their exact
  availability here was not tested; they are not a proof of impossibility.

Recommendation: prioritize the native Windows beta. If that fails, an emulator
could be a bounded camera-only experiment with separate gates for USB access,
ABI compatibility and sustained frames. Full optical XR would require additional
display/runtime integration. Bare-metal Linux removes a virtualization layer
but does not automatically solve those Android-guest interfaces either.
