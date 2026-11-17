# ADR-0004 (draft): on the Beam Pro, the pose runs on the device

- **Status:** Proposed, 2026-11-16. Not accepted until Jack says so. If accepted it amends ADR-0001's
  staircase for the Beam Pro host (ADR-0001 stays in force for its other hosts and as history).
- **Deciders:** Jack.

## Context

ADR-0001 orders the bridge as Step C (frames to a Python pose service over the network) → Step A (OpenCV on
the device) → Step B, and makes the C → A pivot quality-driven: median end-to-end latency over 150 ms,
accuracy worse than 10 mm RMS, or unreliable Wi-Fi. Step C has only run on the Surface over localhost.

On 2026-11-16, with the Beam Pro in hand, Jack set the direction: measure the on-device chain; the Surface is
no longer needed as a pose host.

## Decision (proposed)

On the Beam Pro, the live pose (marker detection, PnP, lock filter) runs on the device. Step C over Wi-Fi is
not built for the live path.

Implementation options for Step A, to choose before B3a:

| option | what | for | against |
|---|---|---|---|
| A1 | OpenCV for Unity (commercial Asset Store package) | no native build, C# API, used by the Quest ArUco sample | paid licence; large; vendor wrapper between us and OpenCV |
| A2 | OpenCV Android SDK (Apache-2.0, arm64 prebuilt) behind a thin C ABI in a native plugin: `ArucoDetector` + `solvePnP` (IPPE / SQPnP) on the Y plane | free; same algorithms and parameters as the Python pipeline; smallest surface | an NDK/CMake build step to maintain; C++ to review |
| A3 | detection re-implemented in C# | no native code | re-implements OpenCV; parity risk; slow |

Recommendation: **A2**, with a parity test: the same saved Eye frames through the Python pipeline and the
native plugin must give the same marker corners and poses within stated tolerances.

## Consequences

- B3 (Step C over Wi-Fi) is replaced by **B3a**: detection + PnP time on the Beam Pro's CPU (Snapdragon 6
  Gen 1) per frame, median and p90, on Eye frames saved by T6.1. The Surface figure (55 ms median) is not
  comparable: different CPU, and the decode stage may differ.
- The Python pipeline stays the reference for validation and offline analysis. Constants (marker sizes,
  dictionary, calibration resolution) must have one definition shared by both: the native plugin reads them
  from the same config files, it does not hard-code them (AGENTS.md invariant).
- `IArucoPoseBridge` stays the seam: the native plugin is one more implementation behind it, so the Step C
  bridge remains available for development on a PC.

## The case against (recorded so the decision is made knowingly)

- Step C already works end to end (pose service, bridge, lock filters, logs); over a dedicated travel router it
  needs only the protocol fixes listed in the hand-over (timeouts, reconnect, frame compression). It is the
  shortest path to a working B5 before the campaign.
- Step A puts a native build and a C++/Python parity problem on the critical path of the publication campaign.
- Keeping Step C as a development fallback costs little, because the code exists.

A middle way: build A2 for the live path, keep the Python service runnable on the desktop for debugging and
as the parity reference, and do not invest further in its Wi-Fi robustness.

## Open

- Whether the Eye stream's decode on Android leaves enough CPU for detection at the camera rate (T6.1 answers
  the decode side first).
- Thermal behaviour of the Beam Pro under XR + camera + detection for 3-minute protocols.
