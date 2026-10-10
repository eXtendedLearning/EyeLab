# Capability matrix — what is known, how, and when

> Created 2026-10-08 (audit finding F14). One current entry per capability. This
> file is the place to record a capability the moment it is observed or
> refuted; `CONTEXT.md` and `HARDWARE.md` link here instead of repeating status.
>
> **Evidence labels** (one per row):
> **observed** — a run on this project's hardware produced the result, dated;
> **documented** — a vendor or third-party document says so, not run here;
> **hypothesized** — inferred from exports, names or analogy; no run, no document
> that states it. **none** — nothing recorded or demonstrated yet.
> A "negative" row says only what was tried. For example a refused call proves
> that call was refused, not that no licence exists.
>
> **Firmware.** The glasses' firmware was updated by Nebula after the 2026-09-25
> runs and the new version is **not recorded**. Every row dated 2026-09-25 or
> earlier is a *pre-update* result and must be repeated before it is used as a
> statement about the current unit.

| # | Capability | Host | Firmware | Transport | Camera mode | Tracking | Display | Label | Evidence and date |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Glasses as a plain DisplayPort monitor | Surface Pro 7 | pre-update | USB-C DP Alt Mode | – | – | legible lines/colours; electrochromic tint has a floor at level 1 of 3 | observed | S1, 2026-09-22 (ADR-003) |
| 2 | Glasses as a display | Desktop (RTX 5060, no iGPU) | – | – | – | – | not available: the board's Type-C DP Alt Mode is sourced by the iGPU the CPU lacks | hypothesized | reasoning in `HARDWARE.md`; not connected as a monitor |
| 3 | Vendor library loads standalone; USB identity `3318:0436`; HID topology | Desktop | pre-update | USB | – | – | – | observed | S0 2026-09-22; S2 2026-09-23 (NR 3.1.1, no graphics context) |
| 4 | Firmware version readable | any | – | USB | – | – | – | none | the export `Get_Glasses_FW_Version` is resolved but never called (signature unrecovered); read the version from Nebula's UI |
| 5 | Raw IMU through the native API | Desktop | pre-update | USB | – | – | – | observed (negative) | `NRImuCreate` returned 22 `LicenseFeatureUnsupported`, 2026-09-23/24. Says that call and feature family were refused; it does not show that no licence of any kind exists |
| 6 | Grayscale/SLAM camera through the native API | Desktop | pre-update | USB | – | – | – | observed (negative) | `NRGrayscaleCameraCreate` returned 22 in the same run, 2026-09-24 |
| 7 | Fused head pose through the native API | Desktop | pre-update | USB | – | chain returns 0 end to end, 1930 poses, every pose the 4x4 identity with tracking reason `Initializing` | – | observed (negative) | S3b runs 2-3, 2026-09-25. Device-clock candidate refuted. Tracker never starts; cause unknown. **Rerun on current firmware** (`docs/S3-HANDOVER-2026-09-25.md`) |
| 8 | Official Eye RGB camera access | Android host (Beam Pro or listed phone) | – | XREAL SDK 3.1.0 / Unity | YUV_420_888 via `XREALRGBCameraTexture` | – | – | documented | XREAL documentation; **no Android host exists yet**, so never run here |
| 9 | Eye frames over Windows NCM + TCP | Windows | beta firmware + Nebula beta (per third-party repo) | USB network adapter, TCP | 4-bit grayscale, 512x378 (per repo) | – | – | hypothesized | third-party repository description. **On hold** since 2026-10-06 pending XREAL's answer (`claude/xreal-access-routes-2026-10-06.md`); whether it works on the current firmware is unknown |
| 10 | Eye frames over UVC after USB re-enumeration | Android | – | USB host | H.265 2048x1512 (per repo) | – | – | documented (third party) | not run here; requires vendor libraries from the Glasses Control app |
| 11 | PC SDK beta 3.1.2 packages and camera implementation | Windows/macOS target; inspected on desktop | unknown | One-series session preflights NCM/TCP 52999 | Standalone RGB sample and native capture bindings present; frames untested | claimed One-series + Eye tracking; untested | claimed; untested | observed (static inspection) | XREAL-hosted archives/source inspected 2026-10-09. Four core DLLs are identical to the September runtime; the Unity wrapper differs. Sample/session lifecycle remains untested. Main SDK stays 3.1.0; runtime support/licensing unconfirmed. `XREAL-PC-BETA-2026-10-09.md`; ADR reconciliation in revised `AUDIT-2026-10-08.md` |
| 12 | Webcam ArUco/board pipeline: calibration, detection, PnP, pose lock, recording | Desktop and Surface | – | USB webcam | 1280x720 | – | desktop window | observed (implemented) | unit tests; audit 2026-10-08. A **research prototype, not a validated metrological system**: absolute accuracy is not established (audit F04) |
| 13 | Optical see-through wireframe registered to a specimen | – | – | – | – | – | – | none | not demonstrated; requires rows 7/9/10-equivalent camera and pose, camera-head extrinsic, per-eye projection and presentation-time compensation (ADR-003 amendments) |
| 14 | Android emulator / WSL as an XREAL host | Desktop/Surface | – | USB forwarding possible in principle; actual XREAL compatibility untested | untested | untested | physical DP ownership not supplied by showing an emulator window | documented (assessment) | revised 2026-10-09: Google documents Windows USB passthrough and ARM translation; prior categorical exclusion withdrawn. XREAL guest runtime/display integration remains unverified. Native PC beta prioritized; `XREAL-PC-BETA-2026-10-09.md` |
| 15 | Current desktop USB/NCM and beta control preflight | LAGANN_0526 | post-update; version unread | USB `3318:0436`, NCM `169.254.2.10/24`, TCP `169.254.2.1:52999` reachable | no frames requested | not tested | no XREAL monitor enumerated | observed | 2026-10-09 enumeration and connect/close only, no application bytes sent. Separate CDC ECM interface reports Error. No firmware/driver/display changes; `XREAL-PC-BETA-2026-10-09.md` |

## How to update

Add or change a row in the same commit that records the observation. Keep the
date, the firmware state, and the exact thing that was tried. If a result is
superseded, change its row; do not append a second row for the same capability.
