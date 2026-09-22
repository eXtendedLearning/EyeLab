# Nebula for Windows (build 20260205) — static audit and extraction survey

Analyst: Claude (Opus 5) · Date: 2026-09-21, revised 2026-09-22 (§5) · Subject: `Nebula_Windows_20260205/`
Method: **static only**. The executable was never run. All findings come from PE
header parsing (`pefile`), export/import tables, string extraction, and the
IL2CPP metadata blob.

---

## 1. Verdict on safety

**Not malware.** Nothing in the build behaves like a dropper, stealer, injector
or persistence mechanism. What it *is*: an unsigned pre-release build of a
first-party vendor application that (a) installs a kernel-adjacent display
driver, (b) can reflash the glasses' firmware, and (c) ships a Chinese
analytics SDK. Those are three real risks, none of them malicious.

### Evidence for

| Check | Result |
|---|---|
| Network endpoints | All first-party XREAL or their CDNs (§4). No third-party C2, no IP literals, no DGA. |
| Native-DLL URLs | Only libusb/OpenCV/IEC boilerplate. Zero telemetry URLs in native code. |
| Process/persistence | No `CreateService`/`StartService`, no `schtasks`, no `CurrentVersion\\Run` key, no named pipes or global mutexes found. |
| Code injection | No `VirtualAllocEx`, `WriteProcessMemory` or `CreateRemoteThread` anywhere in the build. |
| Obfuscation | None. Symbols intact, debug PDB paths still present (`D:\unity\Demo\ProtoDemo\CMDRunner\...`, `C:\pxyin\Nreal_Project\ELLA\...`). Malware does not ship its own build tree. |
| Anti-analysis | None beyond stock MSVC `IsDebuggerPresent` in CRT startup. |
| Build coherence | PE timestamps 2026-01-22 … 2026-02-05 match the folder name. No backdated or mismatched modules. |
| Elevation | Manifest is `asInvoker`. It does not silently request admin. |

### Evidence against — the three things to weigh

**(a) Nothing XREAL wrote is code-signed.**

Only `vcruntime140*.dll` (Microsoft), `UnityPlayer.dll` (Unity Technologies) and
`NrealVirtualDisplay.dll` (necessarily, to load as a driver) carry Authenticode
signatures. `Nebula for Windows.exe`, `GameAssembly.dll` and all 9 `libnr_*` /
helper DLLs are **unsigned**. Consequence: you have no cryptographic evidence
that the Discord copy is the copy XREAL built. Integrity rests entirely on the
distribution channel. Record the hashes below and compare against anyone else's
copy from the same Discord thread before running:

```
8d7389a7f73fdf3d828c683f3b0995ec013c31a8dd3f0c03b163578c18c49011  Nebula for Windows.exe
0fca58066bf096e441c0433a933022c800d3f50e4bd970b914dc46b7aadc757d  GameAssembly.dll
3c2ffd731be29f4bfe8e194be475e7d2c4733a80d3a1355a866fe192e481e718  UnityPlayer.dll
3497bb63ecfaca1075242096521bacbe3a40de6edcf7e5a03e03cb1a6ec19cd0  baselib.dll
717271527c7a056507b87ee4989e1776993b09504352398991badce56b474a68  UnityCrashHandler64.exe
```

**(b) It installs a display driver, and it will want firmware-flashing rights
over the glasses.**

`CMDRunner.dll` exports a single function, `RunCMD`, implemented as
`CreateProcessA("cmd.exe /C ...")` with an output pipe. The managed layer feeds
it `pnputil /add-driver "<inf>" /install /force` and `pnputil /delete-driver`.
The target is `StreamingAssets/NrealVirtualDisplay/` — a barely-modified copy of
Microsoft's IddCx indirect-display sample (the sample's own `; TODO: edit hw-id`
comments survive verbatim, `ManufacturerName = "<Nreal>"`). It is a UMDF
user-mode driver, root-enumerated, WHQL/attestation-signed. Standard technique
for a virtual monitor; the point is that it needs elevation and it leaves a
driver in the DriverStore after uninstall unless removed.

`StreamingAssets/NROTA/` carries 454 MB of glasses firmware, and every
One-family profile is marked for update:

```json
// NROTA/{gs,gina,gf}/glasses.cfg
{ "mcu": "15.1.03.165_20260105.bin", "mcuNeedUpdate": true,
  "dsp": "15.A.01.097_20250430.bin", "dspNeedUpdate": true }
```

`libnr_glasses_api.dll` exports the full flashing toolchain
(`xreal_glasses_mcu_soc_upgrade`, `xreal_glasses_dsp_audio_upgrade`,
`rommode_upgrade`, `NROTASetGlassToBoot`, `set_clear_usrdata_part`).

**Revised 2026-09-22 — this is much less dangerous than it first looks.**
See §5. The short version: the bundled images are *older* than current public
firmware, the library refuses to flash backwards, and an official recovery
channel exists that speaks the same protocol.



**(b-bis) Two capabilities that are legitimate here but worth naming.**

`libnr_api.dll` and `ov_utils.dll` both import `SetWindowsHookExW` /
`CallNextHookEx` / `UnhookWindowsHookEx`, and `libnr_api.dll` additionally
imports `RegisterRawInputDevices` and 142 USER32 entry points. A spatial-desktop
compositor that forwards keyboard and mouse into a virtual display needs exactly
this, and `libnr_api` registers its own `XrealWindowClass` — so the reading is
benign. But the same APIs are what a keylogger would import, and you asked, so:
**this build has system-wide input-hook capability.** Both DLLs also carry full
registry read/write imports (`RegCreateKeyEx`/`RegSetValueEx`/`RegDeleteKey`),
used for settings persistence; no autostart key path appears anywhere in the
binaries. Nothing here is evidence of misuse — it is evidence that the app is
not sandboxable, which is a reason to run it on a machine you can reimage rather
than on the Surface you do fieldwork with.

**(c) Analytics.**

`SASDK.dll` (Sensors Analytics / 神策) exports `Init`, `TrackInt`,
`TrackString`, `TrackNode`, `RegisterSuperProperties`, `Flush`. It imports
`libcurl` plus OpenSSL (`EVP_aes_256_cbc`, `EVP_PKEY_encrypt`, `HMAC`,
base64 BIO) — payloads are encrypted client-side. Sink:
`https://data-api.xreal.com/api/data-collection/v1/isc/dc/external` (`.cn`
variant uses `.../dc/mac`). A consent gate exists in the UI
(`Agreement`, `AgreementUI`, `ButtonAgree`, `ButtonReject`), so the behaviour is
declared rather than covert. Treat it as ordinary vendor telemetry and decline
if offered the choice.

---

## 2. What the build actually contains

Unity 2022.3-era IL2CPP player, XREAL XR Plugin **3.1.2**
(`UnitySubsystems/XREALXRPlugin/UnitySubsystemsManifest.json`), 776 MB total.
Managed assemblies of interest:

```
Unity.XR.XREAL.dll · Unity.XR.XREAL.Standalone.Samples.dll
Unity.XR.XREAL.Enterprise.dll · Unity.XR.XREAL.Experimental.dll
```

`Standalone` is the significant one: it confirms a **desktop flavour of the
XREAL SDK exists**, which `developer.xreal.com/download` does not publish (the
site still offers only Unity-for-Android SDK 3.1.0/3.0.0 and NRSDK 2.4.x).

### Native layer — the part worth extracting

| DLL | Size | Role |
|---|---:|---|
| `libnr_api.dll` | 46.5 MB | Complete NRSDK **C API**, 657 exports. Also exports `xrNegotiateLoaderRuntimeInterface` + ~40 `xr*` symbols → it is simultaneously an **OpenXR runtime**. |
| `libnr_plugin_6dof.dll` | 20.8 MB | SLAM/6DoF backend. Talks to `libnr_api` over loopback `127.0.0.1:53004`. |
| `libnr_glasses_api.dll` | 3.5 MB | USB/HID transport to the glasses: OTA, USB-function config, SN/FW/property reads. Statically links hidapi + cJSON. |
| `libnr_loader.dll` | 6.3 MB | Module loader / path resolution shim. |
| `ov_utils.dll` | 2.5 MB | OmniVision **OV580** stereo-ISP helper (the grayscale SLAM camera bridge). |
| `XREALXRPlugin.dll` | 0.8 MB | Unity XR subsystem glue only. Not needed outside Unity. |
| ffmpeg 6.x (`avcodec-60` etc.) | 17.6 MB | Video encode/decode for the capture path. |

**Decisive structural finding: none of the `libnr_*` DLLs import `UnityPlayer.dll`
or `GameAssembly.dll`.** They are plain Win32 C libraries.

```
libnr_glasses_api.dll -> KERNEL32, msvcrt, WS2_32                      (3 imports, nothing else)
libnr_api.dll         -> d3d11, dxgi, vulkan-1, SETUPAPI, WS2_32, IPHLPAPI,
                         ffmpeg, ov_utils, CRT
```

Nebula is a *consumer* of this stack, not a prerequisite for it.

---

## 3. Answer to your question: yes, independence is achievable — in tiers

Your framing ("independent = without Nebula") is the right one and the answer is
**yes for the sensor/control path, no for the compositor path**, split three
ways:

### Tier A — glasses control only · 3.5 MB · 1 file
`libnr_glasses_api.dll`, driven from Python by `ctypes`. No Unity, no Nebula,
no D3D. Gives you:

- `GetCurrentGlass`, `NROTAGetSDKGlassType`, `NROTAGetSKUString`,
  `xreal_get_glass_sn`, `Get_Glasses_FW_Version`
- `NRBSPGetUsbConfigAll` / **`NRBSPSetUsbConfigAll`** — the USB-function switch.
  Function bits present in the binary: `FUNC_UVC0`, `FUNC_UVC1`, `FUNC_UAC`,
  `FUNC_NCM`, plus `mtp`/`ecm`/`hid`.
- `NRBSPGetProperty`, `NRBSPGetCameraStatus`, `NRBSPGetPartitionDataFromUboot`

This is **exactly the primitive the Android UVC hack uses**
(`Aloim/Xreal-One-Pro-Eye-RGB-Camera-feed-on-Android`:
`NRBSPGetUsbConfigAll()` → `config.uvc0 = 1` → `NRBSPSetUsbConfigAll()` →
re-enumeration exposes a UVC bulk endpoint). CONTEXT.md currently files that
route under "Android only". **It is not.** The same library exists as a Windows
x64 DLL with identical symbols and no Unity dependency — the managed layer even
exposes `get_Ncm`/`set_Ncm` over the same struct. If the unlock transfers, the
Eye's full-resolution H.265 feed becomes reachable on Windows, and the
512×378 4-bit grayscale NCM compromise recorded in CONTEXT.md stops being the
only option. **This is the single highest-value finding in the build.**

The link-local addresses are baked into this DLL (`169.254.1.1`, `169.254.2.1`),
confirming the NCM services are hosted **on the glasses**, not by Nebula on the
PC.

### Tier B — full native NRSDK · 97.5 MB · 12 files
Add `libnr_api` + `libnr_loader` + `libnr_plugin_6dof` + `ov_utils` + ffmpeg +
`vcruntime`. A C API, so `ctypes`/`cffi` from Python or a thin C++ shim. Opens:

- **IMU** — `NRImuSetCaptureCallback`, `NRImuDataGet{Accelerometer,Gyroscope,Magnetometer}`, `NRImuDataGetHMDTimeNanosOnDevice`, `NRGlassesControlSetIMUFrequencyDivider`
- **6DoF head pose** — `NRHeadTrackingAcquireHeadPose`, `NRHeadPoseGetPose/GetVelocity`, `NRHeadTrackingRecenter/DeepRecenter`, `NRHeadTrackingSetCoordinateMode`
- **Grayscale stereo cameras** — `NRGrayscaleCameraSetCaptureCallback`, `…ImageGetData`, `…GetExposureTime/Gain`, `…UndistortImage`, `…{Project,UnProject}Point`
- **RGB camera (Eye)** — `NRRgbCameraSetCaptureCallback`, `NRRgbCameraSetImageFormat`, `…ImageGetRawData`
- **Factory calibration** — `NRHMDGetComponentIntrinsic`, `…Extrinsic`, `…Distortion`, `…PoseFromHead`, `…ProjectionMatrix`, `NRHMDGetIMU{Accelerometer,Gyroscope}Bias`. *This removes a calibration step from the pipeline* — the concern CONTEXT.md raises about `0xcaff/xr-tools` is satisfied directly here.
- **Anchors / planes / meshing / image targets** — `NRPerception*`, `NRTrackableAnchor{Save,Load,Remap}`, `NRTrackableImageDatabase*`
- **Display control** — `NRGlassesSetDpInputMode` (`NR_DP_INPUT_MODE_{MONO,STEREO}`), `NRGlassesSetDpStereoMode`, `NRGlassesSetDpCurrentEdid` over `NR_EDID_{1920,3840}_1080_{60,72,90,120}`, `NRGlassesSetGlassesSceneMode` (`NR_GLASSES_SCENE_MODE_SDK_RENDER` vs `SPACE_SCREEN`)
- **Compositor** — `NRRendering*` on D3D11 **and** Vulkan, `NRSwapchain*`, `NRFrame*`

The `3840×1080` EDID modes + `DP_INPUT_MODE_STEREO` + `SCENE_MODE_SDK_RENDER`
are the combination that lets an application drive both panels as a true stereo
pair instead of living inside the glasses' on-board space-screen — i.e. it makes
the AR overlay a first-class render target rather than a window in Nebula's
virtual desktop. That directly addresses the latency concern raised in
CONTEXT.md ("if the AR overlay is presented inside Nebula's virtual desktop it
adds a compositing stage in front of the panel").

**`NRAPIInitSetLicenseData` exists but nothing in the binary enforces it** — no
license-failure strings, no `nrsdk_license.bin` shipped. It gates optional
enterprise features, not initialisation.

### Tier C — OpenXR
`libnr_api.dll` exports `xrNegotiateLoaderRuntimeInterface` and the standard
`xrCreateInstance`/`xrCreateSession`/`xrEndFrame` set. If a runtime manifest JSON
can be synthesised and pointed at by `XR_RUNTIME_JSON`, the glasses become a
generic OpenXR device on Windows and EyeLab's Unity project could target
Windows/OpenXR rather than Android. Highest payoff, highest uncertainty — the
extension list and required init path need to be recovered before this is more
than a hypothesis.

### What genuinely requires Nebula
Only the spatial-desktop product itself: the IddCx virtual monitor, window
management (`MultiMonitorTool`, `WinHelper`, `SetDpi`, `GraphicsCapture`), the
account/OTA UI. **EyeLab needs none of it.** EyeLab wants sensors, calibration
and a stereo render target, all of which live below Nebula.

---

## 4. Network endpoints (complete list)

```
https://api.xreal.com/v1/checkip                 region probe (.com vs .cn routing)
https://api.xreal.com/v1/data-web-domain?domain  endpoint discovery
https://app-api.xreal.{com,cn}                   app backend  (+ app-uat-api.* staging)
https://data-api.xreal.com/api/data-collection/v1/isc/dc/external   telemetry sink
https://data-api.xreal.cn/api/data-collection/v1/isc/dc/mac         telemetry sink (CN)
https://s.xreal.com/02V8Nfn0                     short link
https://oss-cn-beijing.aliyuncs.com              asset/OTA CDN (Alibaba)
https://s3.eu-central-1.amazonaws.com            asset/OTA CDN (AWS EU)
```

(The apparent URL fragments in `GameAssembly.dll` are the Brotli static
dictionary, not endpoints.)

---

## 5. Firmware risk, reassessed (2026-09-22)

The first pass of this audit treated the bundled firmware as a one-way door.
That was wrong, and the correction matters enough to state plainly.

### Version arithmetic

| | MCU | DSP |
|---|---|---|
| Public release, 2026-07-22 | `15.1.03.` **`442`** `_20260722` | `15.A.01.097_20250430` |
| Nebula beta bundle, this build | `15.1.03.` **`165`** `_20260105` | `15.A.01.097_20250430` |

Same `15.1.03` branch. The public build is **277 builds and ~6 months newer**.
The DSP image is **byte-identical in version** — that leg is a no-op.

### The library will not downgrade

`libnr_glasses_api.dll` contains an explicit anti-downgrade guard:

```
[libnr_glasses_api] : file %s is older than bin, skip upgrade
```

`mcuNeedUpdate: true` in `glasses.cfg` is *permission to check*, not an
instruction to flash. The actual decision is made at runtime by
`NROTAGetNumber`, which emits either `mcu/soc firmware need to upgrade` or the
skip path above. **If the glasses are on current public firmware, Nebula
cannot flash them.** If they are on something older than January 2026, it would
flash the beta — and the public OTA then moves them forward again.

### Recovery architecture (from the binary)

The One series is a two-stage-bootloader design with a mask-ROM fallback:

- `rommode_upgrade` export; log line `current boot mode is upgraded from rom mode`
- `prepare to update spl and uboot to ddr for glass` — SPL + U-Boot are pushed
  into DDR and executed from RAM. Mask ROM is not writable, so this path
  survives any flash corruption.
- A half-flashed device still **enumerates and is still diagnosable**:
  `[NROTAGetNumber] : boot mode now, can only know need to upgrade mcu/soc`,
  `boot partition connect`, `can't find glass boot device`.
- Images are CRC- and SHA-256-verified before commit
  (`ota/upgrade/crc.c`, `WjCryptLib_Sha256.c`,
  `crc/hash or version mismatch, reset streaming upgrade`) — a corrupt transfer
  is rejected, not written.
- The DP firmware has a **backup partition** (`xreal_glasses_dp_backup_upgrade`,
  `dp_backup_crc`, `read 7911 backup partition crc failed`). Not exercised on
  One-series anyway: `NROTA/{gs,gina,gf}/` contain only MCU + DSP, no DP image.

### Official recovery channel

`https://www.xreal.com/ota` is a **WebHID** flasher — Chrome or Edge on
Windows/Mac/ChromeOS, no install, "Connect → select device → Update". It pushes
the latest public release, i.e. strictly forward from anything Nebula carries.
It speaks the same USB HID transport this DLL uses. Backstop:
`support@xreal.com`, stated 24/7.

### Residual risk, honestly stated

Small, and none of it is "brick":

1. **Unverified:** whether the *web* OTA implements the boot-mode/ROM-mode
   recovery path, or only the normal path. XREAL's own stack clearly does;
   the web app's coverage is unknown. If it does not, the fallback is Nebula
   itself or XREAL support.
2. **Support posture.** Disclosing a Discord beta build may complicate a
   warranty conversation. Recovery is self-service, so this likely never comes up.
3. **Interruption still requires care** — `Never disconnect the cable` is the
   only warning XREAL gives, and it is the right one.

### The interesting consequence

The "beta firmware required" precondition in the Eye-streaming community tooling
dates from a January 2026 build. Public firmware has moved 277 builds past it.
**The NCM/Eye paths may simply work on current public firmware**, which would
make the whole Nebula firmware question moot. Reading the installed version
(Tier A, read-only, no flash) tests this in one call.

---

## 6. Fallback: trimming Nebula to portable

If you decide to keep Nebula rather than replace it:

| Component | Size | Removable? |
|---|---:|---|
| `StreamingAssets/NROTA/` | **454 MB (59%)** | Yes — firmware for `air`, `flora`, `p55` (Air family), `ella`, and the non-matching One variants. Keep only your own model's directory, or delete all of it to make OTA structurally impossible. |
| `sharedassets0.assets(.resS)` | 127 MB | UI textures/fonts. Not safely trimmable without breaking scenes. |
| ffmpeg | 17.6 MB | Only if the capture/recording path is unused. |
| `libnr_plugin_6dof.dll` | 20.8 MB | Only if 6DoF is unused. |

Deleting NROTA alone takes the build from **776 MB → 322 MB** and is the
cheapest single risk reduction available: no firmware images on disk, nothing to
flash. Model codenames present: `gs`, `gf`, `gina`, `glory`, `hylla`, `flora`,
`air`, `p55`, `ella`, `vidda`. Device-type enum includes
`XREAL_DEVICE_TYPE_ONE_PROL` and `…ONE_PROM` (large/medium frame). The root
`NROTA/glasses.cfg` currently selects `{"gf":"gf"}`. **Which codename is the One
Pro is not determined by this audit** — resolve it with `GetCurrentGlass()` /
`NROTAGetSDKGlassType()` (Tier A, read-only) before deleting anything.

---

## 7. Recommended sequence

Ordered so the irreversible step comes last and only if forced.

1. **Snapshot the folder.** Copy `Nebula_Windows_20260205/` somewhere outside the
   repo. It is a Discord beta; it may not be re-obtainable.
2. **Do not run the .exe yet.** Do Tier A first — it is read-only and 3.5 MB.
   Write a `ctypes` wrapper around `libnr_glasses_api.dll`, plug in the glasses,
   call `hid_init` → `hid_enumerate` → `GetCurrentGlass` →
   `Get_Glasses_FW_Version` → `xreal_get_glass_sn` → `NRBSPGetUsbConfigAll`.
   This answers three questions at once: which codename you are, what firmware
   you are on *now* (record it), and what the current USB function mask is.
   Confirm the glasses' VID/PID against `0x3318` / `0x0436` from the Android
   route while you are there.
3. **Attempt the UVC unlock without Nebula.** `config.uvc0 = 1` →
   `NRBSPSetUsbConfigAll` → check for USB re-enumeration and a new UVC device in
   Windows. If this works on stock firmware, the entire Nebula question becomes
   moot for EyeLab and the 512×378 grayscale compromise in CONTEXT.md can be
   dropped in favour of the full-resolution Eye feed.
4. **If step 3 needs the beta firmware**, this is far less fraught than §1(b)
   originally implied — see §5. Check the installed MCU version first: if it is
   at or past `15.1.03.442_20260722`, Nebula cannot downgrade it and the
   question is moot. The recovery path is `xreal.com/ota` in Chrome, which only
   moves forward. Still: record the current version string before anything, and
   do not disconnect the cable mid-flash.
5. **Then** Tier B, in a VM or on a machine you can reimage, with the NROTA
   directory deleted first.

### Legal note (relevant to the repo, not to you personally)
There is no public Windows XREAL SDK. These DLLs arrived through a private beta
channel with no redistribution licence — the same reasoning already applied in
`README.md` to `Packages/com.xreal.xr/`. Keep `Nebula_Windows_20260205/` and any
extracted DLLs **gitignored**. Lab-internal research use is one thing;
committing vendor binaries to a public repository is another.

---

## 7a. Stage 0 result — probe run 2026-09-22

Two runs, `python/.logs/xreal_probe.jsonl`. Glasses connected, no Nebula, no
Unity, nothing flashed.

```
library      C:\GitHub_Uni\EyeLab\vendor\xreal\win-x64\libnr_glasses_api.dll
loaded       yes          hidapi 0.14.0 confirmed at runtime
exports      25 / 25 resolved
HID devices  44 total on the machine, 2 attributable to the glasses
```

| | VID:PID | MI | usage page:usage | product | serial |
|---|---|---|---|---|---|
| vendor control | `3318:0436` | `00` | `0xff00:0x0001` | XREAL One Pro | *(none)* |
| consumer keys | `3318:0436` | `08` | `0x000c:0x0001` | XREAL One Pro | *(none)* |

**What this establishes**

1. **Tier A is proven.** The DLL loads standalone on Windows and every export
   resolves — no Nebula, no Unity, no C runtime redistributable, 3.5 MB. The
   independence premise of §3 holds at least this far.
2. **`3318:0436` matches the Android route exactly** (`CONTEXT.md`, XREAL Eye
   access routes). The Windows device presents the same USB identity the
   Android UVC unlock targets, which is the strongest evidence yet that the
   unlock transfers.
3. `MI_00` at usage page `0xff00` is the vendor-defined control interface — the
   OTA, property and USB-config channel. `MI_08` at usage page `0x000c` is
   Consumer Control: the physical buttons.
4. **The interface numbering is the interesting part.** `MI_00` and `MI_08`
   with nothing between them: interfaces 1–7 exist but are not HID-class, so
   `hid_enumerate` cannot see them. They are the DP, audio and other functions.

**Correction to this audit's earlier framing.** An earlier note invited
comparing the HID interface count against the Android route's "13 before / 17
after". That is not a valid comparison — those are *total USB* interface counts
and this is HID only. The note has been corrected in `xreal_glasses.py`, and
`xreal_probe.py --usb` now enumerates every PnP node under `VID_3318` so the
real before-state can be recorded. **The signal that an unlock worked is a UVC
device appearing, not a change in the HID count.**

**Still unknown after Stage 0**

- **Firmware version.** `Get_Glasses_FW_Version` resolves but its signature is
  undocumented, so Stage 0 did not call it. The question from §5 — whether the
  glasses are at or past the public `15.1.03.442_20260722`, which would make
  the whole beta-firmware question moot — remains open.
- **Serial number.** hidapi reported none; `xreal_get_glass_sn` would be needed.
- **Glass codename** (`gs` / `gf` / `gina`) — `GetCurrentGlass` /
  `NROTAGetSDKGlassType`, same signature problem.

All three are the same blocker: **recovering Tier A/B function signatures** is
now the gating task, not library access.

---

## 8. Open questions this audit could not close

1. **Does the NCM/TCP service answer with Nebula not running?** The addresses
   live in `libnr_glasses_api.dll` and the server is on the glasses, which makes
   "yes" likely, but it is unverified and it is the load-bearing assumption for
   the whole independence plan.
2. **Is the beta MCU firmware required for the UVC/NCM unlock,** or does stock
   firmware accept `NRBSPSetUsbConfigAll`?
3. **Does the USB-function change persist across power cycles?**
   (`NRBSPGetPartitionDataFromUboot` / `set_clear_usrdata_part` suggest it is
   written to a u-boot partition, i.e. persistent.)
4. **Which codename is the One Pro** — `gs`, `gf` or `gina`.
5. **Can the OpenXR runtime be driven standalone** without Nebula's
   initialisation sequence (Tier C).
6. **Can `NRRendering*` present to the glasses without the IddCx virtual
   display,** or does the DP path require a Windows display device to exist?

Questions 1–3 are all answered by one read-only Python session against Tier A.
