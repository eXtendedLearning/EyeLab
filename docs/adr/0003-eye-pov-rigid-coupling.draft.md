# ADR-003 — The pose camera must be rigidly coupled to the display; eye-POV only for now

- **Status:** Proposed — awaiting decision
- **Date:** 2026-09-22
- **Deciders:** Giacomo (Jack)
- **Relates to:** `docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md`; the "XREAL Eye access routes" section of `CONTEXT.md`
- **Supersedes:** the implicit assumption that any calibrated camera can drive the glasses overlay

---

## Context

EyeLab's Python pipeline recovers `T_cam←unv` — the rigid transform from the UNV
structure frame to the frame of whichever camera saw the markers. To draw the
wireframe correctly for a person wearing the glasses, what is actually needed is

```
T_eye←unv  =  T_eye←cam  ·  T_cam←unv
```

The pipeline supplies the right-hand factor. The question this ADR settles is
where the left-hand factor comes from, because the two available answers are not
of comparable difficulty.

**If the camera is rigidly mounted on the glasses** — the XREAL Eye, or the
on-board grayscale SLAM pair — then `T_eye←cam` is a *hardware constant*. It is
fixed at manufacture, identical every frame, and already measured: the XREAL
native API exposes it as `NRHMDGetComponentExtrinsic` and
`NRHMDGetComponentPoseFromHead`, alongside per-component intrinsics, FOV,
projection matrix and distortion data (see §3 of the Nebula audit). Interocular
separation is handled separately by `NRHMDUpdateIPD` and does not disturb the
camera-to-head extrinsic.

**If the camera is external** — a webcam on a tripod, the current Phase-1 setup —
then `T_eye←cam` is time-varying and unobserved. Recovering it means tracking the
wearer's head pose in the webcam's frame continuously: fiducials mounted on the
glasses, or a second 6DoF system, plus its own calibration and its own error
budget. That is a strictly harder problem than the one the overlay is trying to
solve, and it would have to be solved *first*, to a *better* accuracy, for the
overlay to be no worse than the tracking feeding it.

This is a structural asymmetry, not a matter of effort. An external camera makes
the unknown a per-frame quantity; a head-mounted camera makes it a constant that
someone else already measured.

### The latency consequence, which is easy to underestimate

Rigid coupling has a second effect that cuts the other way. Once the camera rides
on the head, every head rotation moves the camera, so the overlay's registration
now depends on pose updates keeping up with head motion — and on optical
see-through, the real world is seen with **zero** latency while the overlay is
late. The mismatch is directly visible; there is no video stream to delay into
agreement with it.

Angular error is `ε ≈ ω · Δt`, and lateral error at viewing distance `d` is
`d · tan(ε)`. At `d = 1 m`:

| head rate `ω` | `Δt = 100 ms` | lateral error at 1 m |
|---|---|---|
| 10 °/s (holding still, breathing) | 1.0° | 17 mm |
| 30 °/s (slow deliberate look) | 3.0° | 52 mm |
| 100 °/s (ordinary glance) | 10° | 176 mm |

The transport budget currently recorded in `CONTEXT.md` — 15 fps frames, a
WebSocket round trip, a 200 ms per-frame timeout — puts `Δt` comfortably in the
70–150 ms band. Excitation points on a modal test grid are spaced at the
centimetre scale. **Even the "holding still" row is of the same order as the
quantity the overlay is supposed to indicate**, and the middle row exceeds it.

The standard fix is late-stage reprojection driven by a high-rate IMU: the vision
pipeline sets the anchor at its own leisurely rate, and gyro integration carries
the pose forward to the instant the frame is actually presented. Rotation
dominates the visible error at these distances, and rotation is precisely what a
gyro measures well over the few milliseconds involved, without drift mattering.
The XREAL native API exposes the whole mechanism: `NRImuSetCaptureCallback` with
`NRGlassesControlSetIMUFrequencyDivider`, then `NRFrameSetRenderingPose`,
`NRRenderingSetPredictPresentTime` and `NRRenderingSetWarpDelayTime`.

## Decision

1. **The camera that supplies pose must be the camera on the glasses.** Overlay
   rendering from an external-camera pose is out of scope. The wearer's
   viewpoint is a requirement of the application, not a refinement of it.
2. **Single POV — the wearer's — for the foreseeable stages.** Multiple
   viewpoints are deferred (see below).
3. **The IMU is on the critical path, not an optimisation.** It is required
   before the overlay can be judged, because without it the registration error
   is dominated by latency rather than by anything the vision pipeline does.
   This reorders the plan: IMU before camera quality.
4. **Consequently `libnr_api.dll` (Tier B of the audit) is required**, not
   optional. Both the factory extrinsics and the IMU live there;
   `libnr_glasses_api.dll` alone cannot supply either.

### Deferred, not rejected: multiple POV

Several cameras and several viewers become tractable the moment there is a
shared spatial frame that all of them can localise against — which is what a
SLAM system with persistent anchors provides. The XREAL stack already exposes
one: `NRPerceptionAcquireNewAnchor`, `NRTrackableAnchorSave`,
`NRTrackableAnchorLoad`, `NRTrackableAnchorRemap`, with UUIDs that survive
sessions. The architecture to revisit this is therefore "register every camera
and every grid to a common anchor", and it is a later tier, not a different
project. Nothing in this ADR forecloses it.

## Consequences

**Ruled out for now**

- Driving the glasses overlay from `python/webcam_pipeline.py` poses.
- Any "validate the pipeline end-to-end with the webcam" milestone. The webcam
  path remains valid for developing and regression-testing *detection,
  registration and the pose math* — it simply cannot terminate in the glasses.

**Still valid, and worth 20 minutes**

Plugging the glasses in as an ordinary DP monitor and displaying a **static**
white-on-black test pattern answers three questions that need no registration at
all, and answers them before any SDK work:

- Is pure `#000` genuinely transmissive on these optics, and at what point does
  near-black start to glow?
- Is a thin bright wireframe legible against a real specimen under lab lighting?
- Is stereo convergence at 3840×1080 SBS comfortable at 0.5–1.5 m working
  distance?

If a wireframe turns out to be unreadable against a grey steel specimen, that is
worth knowing now. This is a display smoke test, not a pipeline milestone, and
this ADR does not dress it up as one.

**Put on the critical path**

- Staging Tier B (`libnr_api.dll` + 11 dependencies, 97.5 MB — see
  `vendor/README.md`).
- Establishing whether the Tier B C API can be initialised standalone
  (`NRAPICreate` / `NRAPIInitSetNetworkType` / `NRAPIStart`) without Nebula and
  without a Unity graphics context. This is the largest unknown in the plan.
  *(Corrected 2026-09-22: this line originally listed
  `NRAPIInitSetStandalone`. The vendor's own host never calls it — see audit
  §9 — so S2 does not either.)*
- Recovering function signatures for the Tier B calls. Unlike hidapi, these are
  undocumented; Stage 0 deliberately calls none of them.

**Accepted costs**

- Detection quality is now bounded by whatever the glasses' own cameras deliver.
  If the NCM route is what becomes available, that is 512×378 at 4-bit
  grayscale, and the marker-size and working-distance consequences recorded in
  `CONTEXT.md` apply in full. Grid-only rendering does not relieve this: it is a
  decoding constraint, not a display one.
- The Surface Pro 7 cannot exercise any of this. Pre-Beam development is limited
  to the webcam path for detection work, plus Editor/mock modes.

## Revised stage ladder

| Stage | Goal | Needs | Status |
|---|---|---|---|
| **S0** | Identity, firmware version, HID topology | Tier A, read-only | **done 2026-09-22** — DLL loads standalone, 25/25 exports, `3318:0436`, 2 HID interfaces (`MI_00` vendor, `MI_08` keys). See audit §7a |
| **S1** | Display smoke test — black transmissivity, wireframe legibility, stereo comfort | glasses as DP monitor, static pattern | **passed 2026-09-22** on the Surface — all pages, lines and colours legible. One finding: see "Electrochromic floor" below |
| **S2** | Tier B "hello world": initialise the C API standalone, read `NRGetVersion` | Tier B staged, signatures recovered | **passed 2026-09-23** on the desktop — all three levels of `python/xreal_native_probe.py`, NR **3.1.1**, standalone, no graphics context. See "S2 result" |
| **S3** | Head pose at high rate — IMU stream, then `NRHeadTracking` if it initialises | S2 | not started |
| **S4** | Factory calibration — per-component intrinsics, extrinsics, distortion | S2 | not started |
| **S5** | Eye/SLAM camera as a frame source (`read()` / `stats()` / `stop()` / `source_id`) | S2, plus the UVC or NCM route from the audit | not started |
| **S6** | Wireframe rendered from the eye's viewpoint, IMU-stabilised | S3 + S4 + S5 | the first full test |

S3 and S4 both precede S5 under this ADR, which is the reordering point: the
previous plan in `CONTEXT.md` treated the camera feed as the gate.

## S1 result (2026-09-22)

Run on the Surface Pro 7. All four pages rendered; every line width and every
colour was legible through the optics. The display path is sound, and
black-on-transparent works as the grid-only architecture assumes.

### Electrochromic floor — new constraint

The lenses keep a visible tint even at the lowest dimming setting: roughly
"smoke glasses", not clear. **This is the product working as designed, not a
fault.** XREAL's own user guide documents the electrochromic control as
*three levels* cycled with the brightness button, and documents **no fully
transparent setting**. Level 1 of 3 is the floor the UI offers.

Whether that is also the *hardware* floor is an open question with real
consequences, and the native API hints that it may not be. `libnr_api.dll`
exposes two distinct families:

```
discrete   NRGlassesControlSetElectrochromicLevel   NRGlassesSetEcLevel
           NRGlassesControlGetElectrochromicTotalLevel   NRGlassesGetEcLevelCount
raw        NRGlassesControlSetElectrochromicValue   NRGlassesSetEcValue
                                                    NR_GLASSES_CONTROL_PARAMS_EC_VALUE
```

A *value* setter alongside a *level* setter usually means the value API drives
the film directly at finer granularity than the levels expose — plausibly
below the UI's floor. Testable at S3/S4, since these are Tier B calls.

**Free test first, before writing any code.** Electrochromic films are clear
when unpowered (fail-safe). Compare looking through the glasses **powered off**
against **powered on at level 1**:

- *Identical* → level 1 already means EC off, and the residual tint is the
  X-Prism optics themselves. No software will improve it; treat the
  transmittance as a fixed property of the hardware.
- *Powered-off is clearer* → the film still carries drive at level 1, and
  `NRGlassesSetEcValue` has headroom worth chasing.

**Result (2026-09-23): powered-off is clearer** — slightly darker when on, by
eye, no measurement. The electrochromic film is the only powered attenuator
in the stack, so the likely reading is that level 1 still drives it and the
raw `NRGlassesSetEcValue` path may go lighter than the UI allows. That is an
inference, not a demonstration: it needs the setter's signature (Tier B,
not yet recovered) and a before/after comparison. Lux-meter numbers through
the lens (off / level 1 / level 3) would turn "slightly" into a figure the
lab protocol can use.

### Consequence either way

For overlay *legibility* the tint is a benefit — attenuating the real world
raises overlay contrast, which is why the electrochromic layer exists at all.
The concern is **operator ergonomics and safety**: an operator placing impact
hammer strikes on a specimen is doing precise, physical work while seeing the
scene through attenuated optics. That belongs in the lab protocol — task
lighting on the specimen, and a check that the operator can see the hammer,
the structure and their own hands comfortably — not only in the software.

---

## S2 result (2026-09-23)

Run on the desktop (`lagann-0526`), glasses on USB, no Nebula, no Unity, entry
`libnr_loader.dll`. All three levels passed; every call returned 0.

```
load      preflight ok, 15/15 exports resolved
version   NRAPICreate 2.3 s -> NRGetVersion -> NRAPIDestroy     NR 3.1.1
start     Create -> InitSetNetworkType(0) -> NRAPIStart 16 ms -> NRGetVersion -> Stop -> Destroy
```

The vendor log (kept in `python/.logs/xreal_native_probe.jsonl`) shows
`Loader init success!`, `Load from local`, and on `start` the device
definitions for glasses config, vsync, three IMUs, the grayscale and RGB
cameras, each `connect type 1`. **The C API initialises standalone without a
graphics context**, which was the largest unknown in the plan.

Two cautions for S3. `NRAPIStart` took 16 ms and `NRAPIStop` 1 µs: Start
registers components rather than streaming, so "started" does not yet mean
data flows — S3 has to create the IMU/head-tracking objects and observe
samples. And no video appears on the desktop by design: S2 renders nothing,
and this machine cannot drive the glasses as a display at all (open question 6).

---

## Open questions

1. Can the Tier B C API initialise without Nebula and without a D3D/Vulkan
   context? If it demands a rendering context, S2 grows considerably.
2. Are `NRImu*` and `NRHMDGetComponent*` usable before `NRRendering*` is
   started, or does the SDK require a full session?
3. Which component id corresponds to the Eye RGB camera versus each grayscale
   SLAM camera, for the extrinsics lookup?
4. Does the overlay need per-user eye calibration beyond `NRHMDUpdateIPD`?
   Optical see-through systems usually do (SPAAM or similar); how much residual
   error remains without it is unmeasured.
5. Is the electrochromic floor a UI limit or a hardware limit? *Partly
   answered 2026-09-23:* powered-off is clearer, so the film is still driven at
   level 1; whether `NRGlassesSetEcValue` can go lower is untested.
6. **Host machine for S6.** The desktop (`lagann-0526`, TUF B860-PLUS WIFI,
   RTX 5060, F-series CPU with no iGPU) enumerates the glasses over USB but
   cannot drive them as a display: on a desktop board the Type-C DP Alt Mode
   is sourced from the integrated GPU, and there isn't one. The board does have
   a Thunderbolt (USB4) header, so a Thunderbolt add-in card with
   DisplayPort IN looped from the RTX 5060 would supply video *and* USB data on
   one connector — the only adapter class that does both. A plain DP-to-USB-C
   converter gives video only, which is not enough for S6. Deferred: S2-S5 need
   USB data only, and the Surface already does display. Decide when S6 is close
   and the Surface's compute has been measured.
