# Hardware in hand — host machines and XREAL

> **Read this before any XREAL work** (native probes, display tests, S-ladder in
> ADR-003, Unity port). It is the single place for what hardware exists and what
> each machine can and cannot do with the glasses. Update it whenever hardware is
> bought, lent, or a capability is verified or refuted.
>
> Last updated: 2026-10-14 (Surface throttles under sustained XR load; Eye buffer is bottom-up. Earlier: 2026-10-13 Surface runs of the PC beta 3.1.2 gate harness: Smart App Control must be off; Eye frames and 6DoF each observed, concurrent use degrades. Earlier: 2026-10-09 desktop USB/NCM and TCP preflight; PC beta sample inspected; emulator assessment corrected).

## Glasses

- **XREAL One Pro + XREAL Eye** (camera accessory, installed). Access to the
  Eye's camera feed is the project's critical dependency — see ADR-003 and
  `docs/S3-HANDOVER-2026-09-25.md`.
- **Firmware:** Nebula updated it after the 2026-09-25 runs; the new version is
  not recorded yet. The PC beta 3.1.2 version calls are licence-refused
  (`CAPABILITIES.md` row 4, 2026-10-13). **`python/xreal_probe.py` (S0) cannot record it**: it
  resolves `Get_Glasses_FW_Version` but deliberately never calls it (signature
  not recovered, `python/xreal_glasses.py`). Read the version from Nebula's UI
  for now; a read-only call is possible only after its signature is recovered.
  Do not let Nebula
  update the firmware again until the camera/IMU access routes are confirmed —
  the unofficial routes depend on firmware behaviour.
- **XREAL Beam Pro:** ordered 2026-05-14, **not arrived** (confirmed
  2026-10-06). **No Android phone with USB-C video output** is available
  either. The established Android SDK 3.1.0 camera route has no host today.
  **PC beta update (2026-10-09):** XREAL-hosted 3.1.2 packages are downloadable
  and contain a Standalone RGB camera sample and Windows camera bindings.
  They are staged locally, not installed in the main project. Runtime camera
  and tracking success remain unverified; see `XREAL-PC-BETA-2026-10-09.md`.
  Four core native DLLs match those already probed in September; the Unity
  wrapper differs. The vendor scene/session lifecycle supplies a new experiment,
  not evidence that the earlier initialization/license blockers are solved.
- **Android emulator / WSL: lower-priority, unverified (revised 2026-10-09).**
  Google documents Windows USB passthrough and ARM translation in selected
  emulator images, so the previous Linux-only/ARM64 categorical exclusion was
  incorrect. USB forwarding does not give Android the Surface's physical DP
  display path or XREAL runtime integration. A camera-only experiment is not
  ruled out; full optical XR is not a demonstrated emulator capability.
  Sources and distinctions: `XREAL-PC-BETA-2026-10-09.md`.

## Host machines

Two machines, with complementary limits:

| | Desktop | Surface Pro 7 |
|---|---|---|
| GPU | NVIDIA RTX 5060 (discrete only — F-series CPU, **no iGPU**) | Intel integrated |
| Motherboard | ASUS TUF B860-PLUS WIFI (has a Thunderbolt/USB4 header) | — |
| USB data to the glasses | **yes** — S0, S2 and the S3/S3b runs were made here | **yes** — PC beta 3.1.2 session, Eye frames and 6DoF observed 2026-10-13 (not with the native probes) |
| Glasses as a display (USB-C DP Alt Mode) | **no** — on a desktop board the Type-C DP Alt Mode is sourced by the iGPU, which this CPU lacks; the RTX 5060's outputs are not USB-C | **yes** — S1 display smoke test passed 2026-09-22 |
| Cameras | desktop USB webcam (model to record) | built-in cameras |
| Windows security | – | **Smart App Control must be off** for the XREAL PC SDK: it blocked the unsigned `libnr_api.dll` (error 4551) on 2026-10-10; turned off by Jack 2026-10-13 |
| Sustained XR load | – | **throttles**: CPU clock falls from ~220 % to 45-60 % of nominal about 50 s after XR starts and stays low across runs (2026-10-14, `perf.csv`); the Eye stream then decodes at 7-9 fps. Thermal vs power limit not separated; CPU/RAM variant still to record |
| Current role | native-SDK probing (S0–S5), compute-heavy work | display path (S1; S6 candidate), PC beta gate runs, portable demos |

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
- Post-update glasses firmware version (from Nebula's UI; S0 does not read it).

## Desktop connection observation (2026-10-09)

Glasses connected to LAGANN_0526 enumerate as `3318:0436` with HID/audio and an
active `UsbNcm Host Device`, address `169.254.2.10/24`. TCP connection to the
beta's control preflight endpoint `169.254.2.1:52999` succeeds without sending
application bytes. The separate CDC ECM interface reports Error. No XREAL
monitor or standard Windows Eye camera is enumerated. These observations
establish connectivity, not delivered camera frames or head poses.

## Surface PC-beta gate runs (2026-10-10, 2026-10-13)

Gate harness `unity_projects/xreal_pc_gates/`, procedure and analysis in
`XREAL-PC-GATES-G1-G3.md` (§12), status rows 4, 11 and 16-18 in `CAPABILITIES.md`.
In short: with Smart App Control off, the Surface runs an XREAL PC-beta session
that presents on the glasses; the Eye delivers 1280x720 frames at 30 Hz without
XR, and 6DoF tracking works on its own. Camera and 6DoF together degrade on this
host (growing camera latency, tracking never initialises or fails). Whether the
Surface's CPU is the cause has not been measured.
