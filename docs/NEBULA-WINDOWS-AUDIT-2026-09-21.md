# Nebula for Windows (build 20260205) — static audit and extraction survey

Analyst: Claude (Opus 5) · Date: 2026-09-21, revised 2026-09-22 (§5) · Subject: `Nebula_Windows_20260205/`
Method: **static only**. The executable was never run. All findings come from PE
header parsing (`pefile`), export/import tables, string extraction, and the
IL2CPP metadata blob.

> **Scope note added 2026-10-08 (project audit F14).** Read the verdicts below as
> a *triage of a static survey*, not as certification. Static inspection of
> exports, imports and strings cannot establish the absence of malicious
> behaviour or zero firmware risk, and "an export exists" is not "the function
> works on this firmware". Capability statements here are hypotheses until
> observed; the current, dated status of each capability (observed / documented /
> hypothesized) is `docs/CAPABILITIES.md`. The binary is unsigned and its
> integrity rests on the distribution channel (see §1(a)).

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

---

## 9. Tier B — S2 signatures recovered statically (2026-09-22)

Method: disassembly (capstone) of the export thunks in `libnr_api.dll` and of
the vendor's own caller, `XREALXRPlugin.dll`, from the same build. The plugin
logs every wrapper call (`[NativeAPI] Start`, `[NativeAPI] GetVersion: %d.%d.%d`,
…), which labels its call sites. Nothing was executed. Consumer:
`python/xreal_native.py`.

### How the vendor host brings the API up

1. `SetDllDirectoryA(<Plugins/x86_64>)`, then `LoadLibrary("libnr_loader.dll")`
   and `GetProcAddress` for 15 `NRAPI*`/`NRGetVersion*` names into a function
   table. **The entry point is the loader, not `libnr_api.dll`.** The loader
   exports 631 of `libnr_api`'s 657 symbols (a strict subset) and loads
   `libnr_api` by bare name, which is why the host sets the DLL directory.
2. `NRAPICreate(&handle)`, then, from settings: `NRAPIInitSetRenderMode(h, 1)`
   (conditional), `NRAPIInitSetLicenseData(h, data, len)` (only if configured),
   `NRAPIInitSetGlassesControl(h, 1, config_str)` (only if configured),
   `NRAPIInitSetGraphicContextType(h, type)`.
3. `[NativeAPI] Start`: `NRAPIInitSetNetworkType(h, 0)` → `NRAPIStart(h)` →
   `NRGetVersion(h, &v)`. A failed `InitSetNetworkType` is logged and Start
   proceeds anyway.
4. Shutdown: `NRAPIStop(h)` → `NRAPIDestroy(h)`.

**`NRAPIInitSetStandalone` is never called by the host.** ADR-003 listed it as
part of the standalone bring-up; that was an inference from the name, and the
binary does not support it. S2 resolves it and reports it, but does not call it.

### Signatures used by S2

| Function | Signature | Evidence |
|---|---|---|
| `NRAPICreate` | `int32 (uint64* out_handle)` | thunk forwards `rcx` as the out-pointer; host passes `&wrapper->handle` |
| `NRGetVersion` | `int32 (uint64 handle, NRVersion* out)` | returns 1 if `out` is NULL; writes dwords at +0/+4/+8 |
| `NRAPIInitSetNetworkType` | `int32 (uint64 handle, int32 value)` | `edx` 32-bit; host passes 0 |
| `NRAPIStart` / `Stop` / `Destroy` | `int32 (uint64 handle)` | `rcx` only |

`NRVersion` is three `int32` (12 bytes), logged by the host as `%d.%d.%d`.
Return value 0 is success — the only value the host tests. Other codes are
reported raw; their names are not recovered.

### Recovered but not called

| Function | Shape | Why not called |
|---|---|---|
| `NRAPIInitSetGraphicContextType` | `(h, int32)` | enum values not recovered; S2 is headless |
| `NRAPIInitSetRenderMode` | `(h, int32)` | host passes 1 only conditionally |
| `NRAPIInitSetGlassesControl` | `(h, bool, const char*)` | config string format unknown |
| `NRAPIInitSetLicenseData` | `(h, const char*, int32)` | §3: not enforced |
| `NRGetVersionExt` | — | a stub in this build: `mov eax, 4; ret` |

### Load-time dependencies not in the staging list

`libnr_api.dll` statically imports **`vulkan-1.dll`** (installed by the GPU
driver, not Windows) and `D3DCOMPILER_47.dll` (Windows). A machine without a
Vulkan-capable driver fails the load with error 126 before any NR code runs.
`xreal_native.preflight()` checks for both.

### What S2 cannot tell from static analysis

Whether `NRAPIStart` succeeds without a graphics context type set, what it
does to the glasses' display mode, and whether it expects the 6DoF plugin
(`libnr_plugin_6dof.dll`, loopback `127.0.0.1:53004`) to come up. Those are
the questions the `start` level exists to answer.

---

## 10. Tier B — S3 IMU signatures recovered statically (2026-09-23)

Same method as §9. The vendor host's `[NativeImu]` wrapper resolves its
function table in this order (offsets into the table): `+0x00`
`NRImuDataDestroy`, `+0x08` `…GetHMDTimeNanos`, `+0x10` `…OnDevice`, `+0x18`
`…GetGyroscope`, `+0x20` `…GetAccelerometer`, `+0x28` `…GetMagnetometer`,
`+0x30` `NRImuCreate`, `+0x38` `NRImuSetCaptureCallback`, `+0x40`
`NRImuStart`, then Pause, Resume, Stop, Destroy.

| Function | Signature | Evidence |
|---|---|---|
| `NRImuCreate` | `int32 (uint64* out)` | thunk uses `rcx` only; host passes `&wrapper->handle` |
| `NRImuSetCaptureCallback` | `int32 (uint64 imu, cb, void* user)` | `rcx`/`rdx`/`r8`; host passes its callback and `user = NULL` |
| `NRImuStart` / `Stop` / `Destroy` | `int32 (uint64 imu)` | `rcx` only (Start reached through a vtable in the host) |
| callback | `void (uint64 imu, uint64 data, void* user)` | library trampoline: `if (cb) cb(rdx, r8, r9)` |
| `NRImuDataGet{Gyroscope,Accelerometer,Magnetometer}` | `int32 (uint64 imu, uint64 data, float[3]* out)` | three `movss` stores at +0/+4/+8; 1 if `out` is NULL |
| `NRImuDataGetHMDTimeNanos[OnDevice]` | `int32 (uint64 imu, uint64 data, uint64* out)` | one qword store |
| `NRImuDataDestroy` | `int32 (uint64 imu, uint64 data)` | two registers |

**Ownership.** The host's callback (`[NativeImu] GetImuData`) reads time,
gyro, accel and mag **inside the callback**, packs them into a 48-byte record
(`u64 t_ns`, three `float3`), then calls `NRImuDataDestroy(imu, data)`. The
data handle does not outlive the callback. `python/xreal_imu.py` follows the
same contract.

**Not known statically:** units (rad/s vs deg/s, m/s² vs g), axis
convention, sample rate, and which of the three IMUs (`imu type 0/1/2` in the
S2 log) the single `NRImuCreate` handle streams. S3's rest and turn
recordings measure the first three; the fourth is recorded as open.

---

## 11. Tier B — why S3's IMU run failed: a licence gate (2026-09-23)

Method as in §9 and §10 — `pefile` + `capstone`, nothing executed — plus one
source not used before: **Nebula is an IL2CPP build**, so its managed enums
survive verbatim in
`Nebula for Windows_Data/il2cpp_data/Metadata/global-metadata.dat`.

### What the run did

Three runs of `python/xreal_imu_probe.py` (two `rest`, one `--turn yaw`), all
identical: `NRAPICreate` → `InitSetNetworkType(0)` → `NRAPIStart` →
`NRGetVersion` all returned 0, `glasses_link: connected`, then

```
NRImuCreate  ->  22      in 0.004 ms      samples 0
```

Two things rule out the obvious suspects. The link is up (`DeviceMode=1`,
three IMU objects created by `NRAPIStart`). And the call produced **no SDK log
output at all**: the S3 logs contain exactly the same four
`ImuInterface::CreateDefinition end` and three `Create imu type:N id:M` lines
as the §9 `start`-level runs, which never call `NRImuCreate`. 4 µs, no device
traffic.

### The dispatch is fine; the call is refused on arrival

`libnr_loader!NRImuCreate` (rva `0x21110`) is a table thunk: a global pointer,
slot `+0x30`, `jmp`. All thirteen `NRImu*` exports dispatch through the same
table (`+0x00` `DataDestroy` … `+0x60` `Destroy`, matching §10's ordering
exactly). **If that table were unbound the thunk would return 1, not 22**, so
the call reached `libnr_api`.

`libnr_api!NRImuCreate` (rva `0x366f0`) builds the string `"imu"` on the stack
and calls a gate at `0x12eef0(registry, "imu", 1)`. If the gate returns
non-zero it returns that code and **skips the implementation entirely** — the
vtable call at `+0x52` is never reached, and the out-pointer is never
dereferenced. So the argument shape recovered in §10 is not what failed, and
retrying with other argument shapes cannot help.

Inside the gate (`0x12f6a9`–`0x12f71b`) the module name is looked up by linear
scan in a global vector of 24-byte entries, `begin = 0x2c5b8e0`,
`end = 0x2c5b8e8`. **Empty vector → `mov edi, 0x16` (22). Scanned without a
match → the same branch.** The only other reference to that vector in the
whole 46 MB DLL is at `0x11e3ac`, inside a function that first calls a
bool-returning parser (`0x11efd0`) and, on success, fills a set of globals:
two strings, a 16-byte blob, this vector, a qword and a second vector. Every
string constant on that path is obfuscated and the function has no direct
callers (it is reached through a pointer), so *what* populates the list is not
recovered.

### 22 has a name

`XREALErrorCode` is in Nebula's metadata in declaration order, its constants
stored as zigzag varints in the field-default blob. The first block decodes
cleanly and sequentially, which is its own corroboration:

| | | | |
|---|---|---|---|
| 0 Success | 7 SdcardPermissionDeny | 14 UnSupportedHandtrackingCalculation | 21 TimeOut |
| 1 Failure | 8 RGBCameraDeviceNotFind | 15 Busy | **22 LicenseFeatureUnsupported** |
| 2 InvalidArgument | 9 DPDeviceNotFind | 16 Processing | 23 LicenseDeviceUnsupported |
| 3 NotEnoughMemory | 10 TrackingNotRunning | 17 NumberLimited | 24 LicenseExpiration |
| 4 UnSupported | 11 GetDisplayFailure | 18 DisplayNoInStereoMode | |
| 5 GlassesDisconnect | 12 GetDisplayModeMismatch | 19 InvalidData | |
| 6 SdkVersionMismatch | 13 InTheCoolDown | 20 NotFindRuntime | |

The enum continues (channel errors, `PermissionDenyError`, `UnSupportDevice`)
but those constants did not decode unambiguously and are deliberately not
reproduced. Consumer: `python/xreal_result_codes.py`, which reports an
unrecovered code as `N (unnamed)` rather than inventing a name.

**22 = `LicenseFeatureUnsupported`.** Two independent lines agree: the native
gate is a licensed-feature-name lookup, and the managed enum puts 22 on
"licence does not cover the requested feature". Nebula's UI strings
corroborate a third time — `InitLicenseData`, `LicenseNotSupportRequestedFeature`,
`LicenseFeatureUnsupported`, `LicenseExpiredErrorTip`.

### What is gated and what is not

Verified by disassembling each export and looking for the `0x12eef0` call
before its first real work:

| gated (module tag) | ungated |
|---|---|
| all 13 `NRImu*` and 5 `NRImuCalibration*` (`"imu"`) | the whole `NRAPI*` bring-up, `NRGetVersion` |
| `NRHMDGetIMUAccelerometerBias`, `NRHMDGetIMUGyroscopeBias` (`"imu"`) | `NRHMDCreate`, `NRHMDGetComponentIntrinsic` / `Extrinsic` / `Fov` / `Resolution` |
| `NRGrayscaleCamera*` (`"grayscale_camera"`) | `NRRgbCameraCreate` |
| `NRVsync*`, `NRGlasses*`, `NRGlassesControl*` | `NRHeadTrackingCreate` / `Destroy` / `AcquireHeadPose` / `Recenter` |
| `NRHeadPoseGetBias`, `NRHeadPoseGetVelocity`, `NRHeadTrackingSetCoordinateMode` | `NRHeadPoseGetPose` / `GetHMDTimeNanos` / `GetTrackingReason` / `Destroy` |

S2 passed because **nothing S2 called was gated**; S3's `NRImuCreate` was the
first gated call EyeLab ever made. Two consequences: §3's "the licence is not
enforced" is true only of the ungated bring-up and must not be read more
widely, and **S4's extrinsics lookup is not blocked by this mechanism**.

Raw IMU is licensed. Fused head pose is not — which inverts ADR-003's
ordering for S3 and is why `python/xreal_headpose.py` exists.

### S3b signatures — head pose

Recovered the §9 way, from the vendor's own caller. `XREALXRPlugin.dll`'s
`NativePerception` resolves the `NRHead*` exports into a table at `[this+8]`
and calls them with handles the object holds; the export thunks in
`libnr_api.dll` confirm the arity by which registers they read. Note the
resolution loop stores the **previous** `GetProcAddress` result, so a naive
read of it shifts every slot by one — the slots below are the corrected ones.

| Function | Signature | Evidence |
|---|---|---|
| `NRHeadTrackingCreate` | `int32 (uint64 session, uint64* out_tracking)` | slot `+0xe0`; right after `[NativePerception] Start Finish`: `rcx = [this+0x98]`, `rdx = &[this+0xa0]` |
| `NRHeadTrackingAcquireHeadPose` | `int32 (uint64 session, uint64 tracking, uint64 hmd_time_nanos, uint64* out_pose)` | slot `+0xf0`; four registers in the thunk, four set in `[NativePerception] GetHeadPose: %llu` |
| `NRHeadPoseGetPose` | `int32 (uint64 session, uint64 pose, float out[16])` | slot `+0x110`; the caller zeroes **64 bytes** at the out-pointer, the thunk copies four xmmwords back and returns early if it is NULL |
| `NRHeadPoseGetHMDTimeNanos` | `int32 (uint64 session, uint64 pose, uint64* out)` | slot `+0x138`; one qword |
| `NRHeadPoseGetTrackingReason` | `int32 (uint64 session, uint64 pose, int32* out)` | slot `+0x118`; one dword |
| `NRHeadPoseDestroy` | `int32 (uint64 session, uint64 pose)` | slot `+0x120`; two registers |
| `NRHeadTrackingDestroy` | `int32 (uint64 session, uint64 tracking)` | slot `+0xe8`; two registers |

There is **no `NRHeadTrackingStart`**: create, then acquire. The vendor
acquires, reads and destroys a pose handle inside one iteration;
`python/xreal_headpose.py` does the same.

**Not known statically:** (1) which handle the vendor keeps at `[this+0x98]` —
it is assigned outside the disassembled region, and S3b passes the
`NRAPICreate` handle, which the run will confirm or refute with a return code.
*Refuted 2026-09-24: it is a perception handle, not the API handle — see §12;*
(2) what `hmd_time_nanos` wants, so the probe takes a list of candidates and
starts with 0; (3) whether the 64 bytes are a 4x4 transform and in which
convention — `python/headpose_analysis.py` tests the candidate layouts against
the recording instead of assuming one; (4) what populates the licensed-feature
list.

---

## 12. S3b run 1, and the perception layer it uncovered (2026-09-24)

Two runs on the desktop (`lagann-0526`), glasses on USB, no Nebula.

### The gate discriminator answered its question

```
NRImuCreate              -> 22  (LicenseFeatureUnsupported)   0.004 ms
NRGrayscaleCameraCreate  -> 22  (LicenseFeatureUnsupported)   0.003 ms
NRHMDCreate              ->  0  (Success)                     0.006 ms
```

Both gated families refused, the ungated control fine, glasses connected, NR
3.1.1. **The licensed-feature list is empty** — this is not "raw IMU
specifically is withheld", it is "no licence is in force at all", which also
rules out `grayscale_camera` (and so the S5 SLAM-camera route through this API,
had we planned to take it). §11's branch between "empty list" and "per-feature"
is closed on the first branch.

### The pose run failed, and the SDK said why

`NRHeadTrackingCreate` returned **2** `InvalidArgument` in 0.628 ms — real work,
not a gate refusal — and the vendor log carried the reason verbatim:

```
[ERROR] [NRSDK] NRHeadTracking Session Invalid Argument!
```

S3b had passed the `NRAPICreate` handle, on §11's stated assumption about
`[this+0x98]`. Wrong handle. What goes there is a **perception** handle, from a
family ADR-003 never mentioned and §§9–11 never looked at.

### The perception layer

`XREALXRPlugin.dll` resolves 76 exports into one table; the slots below are the
corrected map. **The resolver stores the previous `GetProcAddress` result**, so
pairing each name with the store that follows it shifts every slot by one; the
map here is anchored on four independent call sites (`+0xe0` used with an
out-pointer straight after `[NativePerception] Start Finish` = Create, `+0xf0`
in `GetHeadPose` with four arguments = AcquireHeadPose, `+0x110` with a 64-byte
out-buffer = GetPose, `+0x118` with a dword = GetTrackingReason).

| slot | export | slot | export |
|---|---|---|---|
| +0x18 | `NRPerceptionGroupCreate` | +0x68 | `NRPerceptionResume` |
| +0x20 | `NRPerceptionGroupGetPerceptionId` | +0x70 | `NRPerceptionGetHMDTimeNanos` |
| +0x28 | `NRPerceptionGroupGetDescription` | +0xe0 | `NRHeadTrackingCreate` |
| +0x30 | `NRPerceptionGroupGetSupportedFeatures` | +0xe8 | `NRHeadTrackingDestroy` |
| +0x38 | `NRPerceptionGroupDestroy` | +0xf0 | `NRHeadTrackingAcquireHeadPose` |
| +0x40 | `NRPerceptionCreate` | +0x110 | `NRHeadPoseGetPose` |
| +0x48 | `NRPerceptionStart` | +0x118 | `NRHeadPoseGetTrackingReason` |
| +0x50 | `NRPerceptionStop` | +0x120 | `NRHeadPoseDestroy` |
| +0x58 | `NRPerceptionDestroy` | +0x138 | `NRHeadPoseGetHMDTimeNanos` |
| +0x60 | `NRPerceptionPause` | +0x148 | `NRConfigCreate` |

The bring-up the vendor performs between `NRAPIStart` and `NRHeadTrackingCreate`:

1. `NRPerceptionGroupCreate(&group)`
2. `NRPerceptionGroupGetCount(group, &count)`
3. per index: `NRPerceptionGroupGetPerceptionId(group, i, &id)` and
   `NRPerceptionGroupGetSupportedFeatures(group, i, &mask)`
4. `NRPerceptionCreate(id, &perception)`
5. `NRPerceptionStart(perception)`

then `NRHeadTrackingCreate(perception, &tracking)`, and teardown in reverse.

| Function | Signature | Evidence |
|---|---|---|
| `NRPerceptionGroupCreate` | `int32 (uint64* out_group)` | thunk forwards `rcx` only |
| `NRPerceptionGroupGetCount` | `int32 (uint64 group, int32* out)` | `rcx`, `rdx` |
| `NRPerceptionGroupGetPerceptionId` | `int32 (uint64 group, int32 index, uint32* out)` | `rcx`, **`edx` 32-bit** (`mov dword [rsp+0x3c], edx`), `r8` |
| `NRPerceptionGroupGetSupportedFeatures` | `int32 (uint64 group, int32 index, uint64* out)` | `rcx`→`rbx`, `edx`→`edi` 32-bit, `r8`→`rsi` |
| `NRPerceptionGroupDestroy` | `int32 (uint64 group)` | `rcx` only |
| `NRPerceptionCreate` | `int32 (int32 perception_id, uint64* out)` | **`mov edx, ecx`** — the id is 32-bit — then `mov r8, rdx` |
| `NRPerceptionStart` / `Stop` / `Pause` / `Resume` / `Destroy` | `int32 (uint64 perception)` | `rcx` only |
| `NRPerceptionGetHMDTimeNanos` | `int32 (uint64 perception, uint64* out)` | `rcx`, `rdx` |

**The whole family is ungated** — no `0x12eef0` call in any of the 35
`NRPerception*` / `NRConfig*` exports — and all 35 are re-exported by
`libnr_loader.dll`. So is `NRConfigCreate` and the plane/anchor/mesh/hand
toggles hanging off it, which matters for later stages.

`NRPerceptionGroupGetDescription` would name each perception, but its
buffer-length argument is not pinned down, so `python/xreal_perception.py`
resolves and reports it without calling it: a vendor DLL writing into a buffer
on a guessed length is not a risk worth taking for a label.

**Still open:** what `hmd_time_nanos` wants in `AcquireHeadPose` (the probe
takes a candidate list, starting at 0); whether the 64-byte pose payload is a
4x4 transform and in which convention (`python/headpose_analysis.py` tests the
candidates against the recording); what populates the licensed-feature list;
and what the feature-mask bits from step 3 mean — the enumeration prints them,
which is the first look anyone has had at what these glasses actually offer.

---

## 13. S3b run 2: the plumbing works, the tracker does not (2026-09-25)

### Perception enumeration

`NRPerceptionGroupCreate` → `GetCount` → 20 entries, ids **1..20**, and
`NRPerceptionGroupGetSupportedFeatures` returned **0 for every one of them**.
Sequential ids and a uniformly zero mask mean this is very likely a fixed table
of perception *types* rather than a report of what these glasses can do; the
group-level feature call may also simply not populate without a licence.
`NRPerceptionGetSupportedFeatures` on a *started* perception is a different call
on a different handle and has not been tried — run 3 does.

### Head pose: every call succeeded, no data came out

20 s at a 96 Hz poll, perception id 1 (the first the group offered), every call
returning 0 with no failures of any kind:

```
samples 1925    acquire 0    getter 0    destroy 0
NRPerceptionStart  79.99 ms      NRPerceptionStop  49.04 ms
```

and yet, across all 1925 samples:

| field | value | distinct values |
|---|---|---|
| `hmd_ns` | `0` | 1 |
| `tracking_reason` | `1` | 1 |
| pose (16 floats) | exact 4x4 **identity** | 1 |

The subject moved his head through yaw, pitch and roll throughout. The host
clock advanced normally (10.39 ms median), so the loop really ran; the device
returned the same default pose every time.

### `tracking_reason = 1` decoded

`NotTrackingReason`, from Nebula's `Unity.XR.XREAL` metadata, same method as
`XREALErrorCode` in §11, and the block decodes sequentially:

```
0 None   1 Initializing   2 Relocalizing   3 InsufficientLight
4 InsufficientFeatures   5 ExcessiveMotion   6 Unsupported
7 CameraUnavailable      8 Scanning
```

So the head tracker sat in **`Initializing`** for 20 s and never left it.
Consumer: `xreal_result_codes.NOT_TRACKING_REASONS`. Note that
`CameraUnavailable` is a *distinct* reason that the SDK did **not** report — it
is not complaining about the cameras, it simply never finished starting.

### What this rules in and out

The call sequence, the signatures, the handle chain and the ownership contract
are all confirmed correct: nothing returns an error, the pose object is created
and destroyed cleanly, the feature is not licence-gated. What is missing is
data. Four candidates, in the order run 3 tests them:

1. **The timestamp.** `hmd_time_nanos = 0` was a documented guess (§11), and the
   returned pose was stamped 0. `NRPerceptionGetHMDTimeNanos(perception, &ns)`
   exists for exactly this value. If asking for "the pose at time 0" yields a
   default pose, this is the whole story.
2. **The perception.** Id 1 was taken because it was first, out of 20 with
   indistinguishable metadata. `--sweep` now tries all of them in one session.
3. **A missing start step** — the 6DoF plugin (`libnr_plugin_6dof.dll`,
   loopback `127.0.0.1:53004`, §9) may need to be up, or a render/display
   session may be a precondition.
4. **The licence.** 6DoF SLAM needs the grayscale cameras and 3DoF needs the
   IMU, and both are refused with 22 (§12). If tracking has no sensor it can
   legitimately initialise forever. This is the least testable of the four and
   the most expensive if true, which is why it is last.

### External confirmation of the licence mechanism

XREAL's own NRSDK FAQ states it plainly: *"For the **Enterprise** version, it is
possible to access glasses raw data(grayscale camera & IMU) through its APIs"*,
with access by request through their contact page. That is the same split the
gate implements — the two module names we saw refused are `imu` and
`grayscale_camera` — so §§11–12 are not an artefact of our bring-up. The current
docs (NRSDK 3.1.0, matching our 3.1.1 binaries) do not document the mechanics.

---

## 14. S3b run 3: the timestamp is not the problem (2026-09-25)

### The device clock works, and changes nothing

`NRPerceptionGetHMDTimeNanos(perception, &ns)` returns a real, advancing device
clock (`529777489910600` ns). Passing it as `hmd_time_nanos` — the §13 candidate
1 — produced 1930 samples, acquire 0 throughout, and **exactly the same result as
passing 0**: `hmd_ns` 0 in every sample, one distinct 16-float pose across the
whole recording, that pose the exact identity, reason 1 `Initializing` for
1929 of 1930. Candidate 1 is refuted. `--clock zero` still reproduces run 2 for
comparison.

Also read on the live perception: `NRPerceptionGetSupportedFeatures` → **0**,
matching the group-level mask. A started perception reports no features.

### Only one perception id is real

The sweep tried all 20 ids the group enumerated. `NRPerceptionCreate` succeeded
for **id 1 only**; the other 19 returned 2 `InvalidArgument`. So
`NRPerceptionGroupGetCount` = 20 with `GetPerceptionId(i)` = `i+1` and a zero
mask is not an enumeration of anything real — it looks like a fixed id space
rather than a list of what this device offers, and the group's own
`GetDescription` (not called, buffer length unknown) is probably the call that
would name them.

The SDK log names the one that exists:

```
get device config failed
PerceptionBase Init failed
Initialize failed, invalid handle = 2808774442768
Plugin Initialize failed: retcode=1, id=nr_perception_id_remote
NRSDK LeopardClient remove connection(0) status=0x30000, disconnect
```

So perception ids map to **named, plugin-backed perceptions**, and id 1 —
the only creatable one, the one every pose run has used — is
`nr_perception_id_remote`. A "remote" perception is unlikely to be the
head-tracking one, which reframes candidate 2 from "try the other ids" to "find
the id/name map, because the group is not telling us". `nr_perception_id_*` is
not a plain ASCII string in `libnr_api.dll`, `libnr_loader.dll`,
`libnr_plugin_6dof.dll`, `ov_utils.dll` or `libnr_glasses_api.dll`, so it is
assembled at runtime or obfuscated like the licence-path strings.
`nr_imu_tracking` and `nr_imu_calibration` *are* present in `libnr_api.dll` and
look like sibling plugin names — and the first of those would be IMU-backed,
hence licence-gated.

### Two caveats that limit this run

`NRPerceptionStart` **failed with 1 in the sweep** and **succeeded 35 s later**
in the pose run, on the same perception id. The SDK errors above (device config,
`LeopardClient ... disconnect`) point at a transient device-state problem rather
than an argument problem. Nebula was launched around this time and **updated the
glasses' firmware**, which would explain it and also means every measurement in
§§11–14 is pre-update. The sweep must be re-run on a clean machine state before
its result is used for anything, and S0 should be re-run to record the new
firmware version.

Handover with the ranked candidates and the next steps:
`docs/S3-HANDOVER-2026-09-25.md`.
