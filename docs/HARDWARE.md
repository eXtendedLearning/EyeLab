# Hardware in hand — host machines and XREAL

> **Read this before any XREAL work** (native probes, display tests, S-ladder in
> ADR-003, Unity port). It is the single place for what hardware exists and what
> each machine can and cannot do with the glasses. Update it whenever hardware is
> bought, lent, or a capability is verified or refuted.
>
> Last updated: 2026-10-06 (Beam Pro / Android status, emulator verdict).

## Glasses

- **XREAL One Pro + XREAL Eye** (camera accessory, installed). Access to the
  Eye's camera feed is the project's critical dependency — see ADR-003 and
  `docs/S3-HANDOVER-2026-09-25.md`.
- **Firmware:** Nebula updated it after the 2026-09-25 runs; the new version is
  not recorded yet (re-run S0: `python/xreal_probe.py`). Do not let Nebula
  update the firmware again until the camera/IMU access routes are confirmed —
  the unofficial routes depend on firmware behaviour.
- **XREAL Beam Pro:** ordered 2026-05-14, **not arrived** (confirmed
  2026-10-06). **No Android phone with USB-C video output** is available
  either. So the only *official* Eye camera API (XREAL SDK 3.1.0, Unity,
  Android-only) has no host today.
- **Android emulator on Windows: ruled out (2026-10-06).** The XREAL Android
  libraries are arm64-v8a only (checked in the local SDK copy); Android
  Emulator USB passthrough is documented for Linux hosts only; and inside an
  emulator the glasses would be a Windows monitor, not an Android display,
  so the Surface's DP port does not help. Details:
  Project doc `claude/xreal-access-routes-2026-10-06.md`.

## Host machines

Two machines, with complementary limits:

| | Desktop | Surface Pro 7 |
|---|---|---|
| GPU | NVIDIA RTX 5060 (discrete only — F-series CPU, **no iGPU**) | Intel integrated |
| Motherboard | ASUS TUF B860-PLUS WIFI (has a Thunderbolt/USB4 header) | — |
| USB data to the glasses | **yes** — S0, S2 and the S3/S3b runs were made here | not tested with the native probes |
| Glasses as a display (USB-C DP Alt Mode) | **no** — on a desktop board the Type-C DP Alt Mode is sourced by the iGPU, which this CPU lacks; the RTX 5060's outputs are not USB-C | **yes** — S1 display smoke test passed 2026-09-22 |
| Cameras | desktop USB webcam (model to record) | built-in cameras |
| Current role | native-SDK probing (S0–S5), compute-heavy work | display path (S1; S6 candidate), portable demos |

The glasses have a single USB-C connector, so the host must deliver video
**and** USB data on that one cable for S6 (eye-POV overlay): only the Surface
does today.

## Desktop display path — purchase decision (ADR-003, open question 6)

- **Thunderbolt/USB4 add-in card with DisplayPort IN**, fed by a short DP cable
  from the RTX 5060: the only adapter class that gives video *and* USB data on
  one USB-C connector. Needs the board's Thunderbolt header and a compatible
  card. **Not bought.**
- **Plain DP→USB-C converter:** video only, no USB data — not enough for S6.
- **Decision gate:** buy only if S6 has to run on the desktop, i.e. once S6 is
  close and the Surface's compute has been measured as insufficient. S2–S5 need
  USB data only, which the desktop already has.

## Optional dual-camera webcam setup

Surface built-in camera + the desktop USB webcam (moved to the Surface) can
form a two-camera system. Scope, per ADR-003 decision 1: the pose that drives
the overlay must come from the glasses' own camera, so a two-webcam rig is
**not** on the S-ladder critical path. It is useful for:

- webcam-phase validation and cross-checks (an independent second view of the
  same registration);
- the deferred multi-POV tier (ADR-003, "Deferred, not rejected").

Requires a stereo/extrinsic calibration between the two cameras before any
measurement is combined.

## To record

- Surface Pro 7 CPU/RAM variant (sets the S6 compute budget).
- Desktop webcam model and resolution.
- Post-update glasses firmware version (S0).
