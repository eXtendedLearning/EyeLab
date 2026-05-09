# ADR-001 — OpenCV-on-Unity bridging strategy: staircase C → A → B

- **Status:** Accepted
- **Date:** 2026-05-08
- **Deciders:** Giacomo (Jack)
- **Supersedes:** the deferred decision in `.docs/PROJECT.md` (Phase 2 Unity Integration: Two Options) and the OpenCvSharp-on-Android assumption in `.docs/tasks/T2.6-xreal-port.md`

## Context

EyeLab's measurement-grade ArUco/PnP/registration pipeline is fully implemented in Python+OpenCV (`../python/`). The XREAL phase needs to surface that same pipeline through a Unity Android app on the XREAL One Pro + Eye. The XREAL Plugin 3.1.0 (`com.xreal.xr@3.1.0`, namespace `Unity.XR.XREAL`) does not provide ArUco — it has its own `MarkerTracking` sample using XREAL-branded interactive cards, which is a different protocol. ArUco support has to be added by us.

Three viable paths to get OpenCV's ArUco running with Unity:

- **Option A — DIY native plugin (.aar wrapping OpenCV 4.11+ Android SDK).** $0. Java side does ArUco + solvePnP, returns pose to C# via `AndroidJavaObject`. ~1–2 weeks of build-system + JNI marshalling work. Risk: image-data marshalling across JNI is the painful part. Reference exists (Wu, June 2025).
- **Option B — OpenCV for Unity (EnoxSoftware), $95.** Near-1:1 Python OpenCV API. Proven XREAL compatibility (their `NrealLightWithOpenCVForUnityExample`, but against the legacy NRSDK — would need re-verification on `com.xreal.xr@3.1.0`). ~2–3 days integration. Procuring even small funds at UNIVPM has non-trivial latency.
- **Option C — Off-device ArUco service.** Unity app streams XREAL Eye frames to a Python service running on the Testlab PC (or the dev's Surface 6 when remote). Service runs the existing webcam pipeline and returns poses. Maximum reuse of completed Python work; introduces network round-trip latency; service dependency breaks the "any student with an APK" deployment story.

The previously assumed `OpenCvSharp` route in `T2.6-xreal-port.md` is removed from consideration: OpenCvSharp4 has no official Android runtime, and unofficial Android builds are historically painful. (OpenCvSharp on Windows is fine, and remains an option for the .NET Testlab Bridge — see Consequences.)

The XREAL One Pro hardware arrived 2026-05-08, ~3 months before T2.6 was scheduled. Continuation of the project at Siemens beyond Jack's PhD has been discussed (Post-Doc or successor).

## Decision

We adopt a **staircase: Option C → Option A → Option B**, with the explicit option to stop at C.

**Step C — first** (target: working demo before Leuven Visit B, July 2026):
- Extend `python/webcam_pipeline.py` to receive JPEG-compressed frames over a WebSocket server instead of `cv2.VideoCapture`.
- Service runs as a standalone Python process on Windows. Deployment targets: the Testlab PC at the UNIVPM lab; Jack's Microsoft Surface 6 for remote development from Modena.
- Unity side: capture frames from the XREAL Eye via `XREALRGBCameraTexture` (Y plane is sufficient for ArUco), JPEG-encode, send over WebSocket, receive pose JSON, apply to overlay.
- Frame transport: JPEG over WebSocket at ~15 fps initial target, re-evaluated.
- Pose response payload: `{ rvec, tvec, timestamp_in_ns, marker_ids, rms_reprojection_px, registration_rms_mm }`.

**Step A — second**, only if Step C proves unfit (see "Pivot criteria"):
- Build a DIY .aar wrapping OpenCV 4.11+ for Android.
- Java façade: `init(calibration)`, `detectAndEstimatePose(byte[] image, int w, int h, int format) → float[12]`.
- Reuse Wu (June 2025) prebuilt `opencv-4.11-jar` to avoid full OpenCV-Android compile.
- C# side: `AndroidJavaObject` calls. Same `IArucoPoseBridge` interface as Step C.

**Step B — last resort**, only if Step A is blocked (procurement only after a documented attempt at A):
- Buy OpenCV for Unity ($95). Submit funds request at the time the decision is made, not before.

### Pivot criteria

The pivot from C to A is **quality-driven, not calendar-driven**. We pivot from C → A when *any* of:

- Measured end-to-end latency (frame captured on glasses → pose applied in Unity overlay) exceeds **150 ms median** during normal lab Wi-Fi conditions, and SLAM bridging cannot mask the lag.
- Registration accuracy on the real XREAL Eye degrades below **10 mm RMS** (the WP3 acceptance threshold) due to network-related frame loss or stale-pose application.
- The lab Wi-Fi at UNIVPM or Siemens proves too unreliable for sustained operation (>5% packet-loss episodes during a 30-min session).
- A sufficiently strong reason to ship a no-PC APK appears (e.g. external party wants to evaluate EyeLab without setting up a Python service).

The pivot from A to B is taken only if A's marshalling proves unworkable in practice (e.g. >2 weeks blocked on JNI + no end in sight) **and** the Leuven decision is "we still need on-device ArUco." If A is "merely slow" we accept the slowness and stay on A unless B's $95 is approved.

### Architectural discipline that makes the staircase cheap

All three implementations sit behind a single Unity-side interface:

```csharp
public interface IArucoPoseBridge {
    bool IsAvailable { get; }
    Task<PoseResult> EstimatePoseAsync(YGrayImage frame, ulong frameTimestampNs, CancellationToken ct);
    event Action<float> OnLatencyMeasured;
}

public sealed class WebSocketPoseBridge   : IArucoPoseBridge { /* Step C */ }
public sealed class NativePluginPoseBridge: IArucoPoseBridge { /* Step A */ }
public sealed class OpenCVForUnityBridge  : IArucoPoseBridge { /* Step B */ }
public sealed class MockPoseBridge        : IArucoPoseBridge { /* Editor smoke tests */ }
```

The rest of the Unity app (geometry overlay, registration application, hammer logic) only ever depends on `IArucoPoseBridge`. Implementation is selected at startup by config. Pivot cost between implementations: ~1–2 days, not a rewrite.

### Service location

- v0 (C): standalone Python process on Windows. Same `requirements.txt` as `../python/`. Listens on a configurable port; default `ws://localhost:8765/aruco`. Runs on the Testlab PC at lab; runs on Surface 6 in Modena for remote dev.
- WP4: when the .NET Testlab Bridge is built, OpenCV moves into the bridge process (using OpenCvSharp/Emgu CV — both fine on Windows). Standalone Python service is retired. The Unity side still talks to the same `IArucoPoseBridge` interface — only the URL/protocol changes.

## Consequences

### Easier

- Step C reuses every line of working Python from the webcam phase. Time from "hardware in hand" to "demoable AR overlay" goes from weeks to days.
- The same Python pipeline serves Unity-in-Editor (mock or USB webcam frames) and Unity-on-glasses (XREAL Eye frames). Iteration is fast on Surface 6 in Modena, not blocked on lab access.
- The Testlab Bridge (T4.1) gets to absorb OpenCV when it's built anyway, so we don't end up with an orphan Python service in the long-term architecture.
- The `IArucoPoseBridge` abstraction makes A and B drop-in replacements; future maintainers (PostDoc / successor) can re-evaluate the trade-off without unwinding application code.
- Procurement risk is pushed to the very end: B is only approached if both C and A fail, and only then does anyone need to ask UNIVPM for $95.

### Harder

- Network is now in the critical path of pose updates. Lab Wi-Fi quality becomes a measurement we have to make and track. THEORY.md §10–11 (Kalman + SLAM bridging) becomes load-bearing instead of theoretical.
- Frame transport (JPEG @ ~15 fps over WebSocket) needs a small protocol of its own, including back-pressure and reconnection. About 1–2 days of code.
- Step C requires a host PC reachable from the phone — the "any-student-with-an-APK" use case is *not* available in Step C. That deployment story comes back only after pivoting to A (or B).
- Camera intrinsics for the XREAL Eye must still be calibrated — but the calibration UX moves to a "calibrate against the same Python service from the phone" flow. New code, but small.
- A Step A pivot is not painless. Even with the interface boundary, the .aar build + JNI bring-up is real work. We accept that cost as deferred, not eliminated.

### Decisions this ADR forces in T2.6

`T2.6-xreal-port.md` needs a substantive rewrite:

- §1 (NRSDK names) replaced by `Unity.XR.XREAL.*` API surface — `XREALRGBCameraTexture`, `XREALCallbackHandler`, `XREALPlugin`, etc.
- §2 (camera swap from WebCamTexture) replaced by `XREALRGBCameraTexture.GetYUVFormatTextures()[0]` (Y plane only) → CPU readback → JPEG → WebSocket.
- §4 (hybrid SLAM+ArUco fusion) stays, but the ArUco half is now "incoming poses from `WebSocketPoseBridge`" instead of in-Unity OpenCV calls.
- §6 (thermal/Beam Pro) reframed to "Snapdragon 8 Gen 3 budget as example, primary build target is generic Android."
- §7 (manifest) drops the Beam-Pro `com.nreal.intent.action.LAUNCH` permission.
