using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.XR.XREAL;
using Unity.XR.XREAL.Experimental;
using Unity.XR.XREAL.Experimental.Samples;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLabGates
{
    /// <summary>
    /// Gate state machine (G1/G2/G3/G4/G5/calib/manual/cool-down) and guided protocol: timed steps announced by voice and
    /// beep and shown large by <see cref="GateHud"/>. Keyboard control and lifecycle event log. XR is
    /// entered and exited only through the scene's <see cref="HelloMR.OnClickSwitchXRButton"/>, i.e.
    /// the vendor flow. In the Editor, XR is refused: entering XR on the desktop changes its EDID and
    /// monitor layout (docs/XREAL-PC-GATES-G1-G3.md §1).
    /// </summary>
    [RequireComponent(typeof(CameraProbe), typeof(PoseProbe), typeof(DeviceInfoProbe))]
    [RequireComponent(typeof(PerfProbe), typeof(VendorCameraSource), typeof(DrainingCameraSource))]
    [RequireComponent(typeof(GateHud), typeof(GateOverlay), typeof(DisplayProbe))]
    public sealed class GateController : MonoBehaviour
    {
        const float GlassesTimeout = 30f;
        const float EnterXRTimeout = 20f;
        const float ExitXRTimeout = 10f;
        const float TrackingReadyTimeout = 30f;      // was 60 s; 6DoF was ready in 1.4-24 s on the Surface
        const double CoolEnoughPct = 150.0;          // CPU clock under load, % of nominal (cool: ~220, throttled: 45-60)
        // 14 Oct: a 3.5 s probe always read 205-233 % (the short turbo budget refills within seconds),
        // even right after a run that ended at 60 %. The probe now outlasts that budget: 20 s of load,
        // judged on the mean clock of its last 5 s.
        const float CooldownProbeLoadSeconds = 20f;
        const float CooldownJudgeSeconds = 5f;
        const float CooldownRestSeconds = 20f;
        const double CooldownMaxSeconds = 600.0;
        const float ScanPromptSeconds = 8f;          // repeat the movement prompt while the tracker scans
        const float ScreenshotEverySeconds = 30f;
        const float OverlayDistance = 1.0f;
        const float BurstTimeoutSeconds = 3f;
        const float BridgeConnectTimeout = 15f;
        // G5: a frame is sent for a lock only while the head is nearly still, so that the head pose at
        // receipt stands in for the (unmeasured) pose at exposure.
        const float MaxLockDegPerSec = 10f;
        const float MaxLockMetresPerSec = 0.08f;
        const float HeadHistorySeconds = 0.5f;
        const float HeadSpeedWindow = 0.25f;
        const float TrackingReadyHold = 1f;
        const float NoReportFallbackSeconds = 15f;
        const float TickFromStepSeconds = 8f;   // 3-2-1 ticks only on steps at least this long
        const double FlushCooldown = 5.0;
        const double LostCueCooldown = 5.0;

        [SerializeField] HelloMR m_HelloMR;
        [SerializeField] XREALRGBCameraTexture m_CameraTexture;

        GateArgs m_Args;
        GateRunLog m_Log;
        CameraProbe m_Camera;
        PoseProbe m_Pose;
        DeviceInfoProbe m_Device;
        PerfProbe m_Perf;
        DisplayProbe m_Display;
        GateOverlay m_Overlay;
        CameraSource m_Source;
        Speaker m_Speaker;

        bool m_GlassesConnected;
        bool m_SessionFailed;
        bool m_SubsystemsStarted;
        bool m_InProtocol;
        int m_Marks;
        int m_CaptureSessions;
        int m_LastReason = NativeTracking.NoValue;
        double m_LastLostCue = double.NegativeInfinity;
        double m_LastFlush = double.NegativeInfinity;
        double m_LastScreenshot = double.NegativeInfinity;
        int m_Bursts;
        bool m_Bursting;

        // ---- G5 lock state
        WebSocketPoseBridge m_WsBridge;
        IArucoPoseBridge m_Bridge;
        readonly CancellationTokenSource m_BridgeCts = new CancellationTokenSource();
        Task<PoseResult> m_PoseTask;
        PendingLock m_Pending;
        long m_LockSeq;
        int m_LocksSent, m_LocksPosed, m_LocksAccepted;
        bool m_HasRgbFromHead;
        Pose m_RgbFromHead;
        readonly LockAnchor m_Anchor = new LockAnchor();
        readonly Queue<HeadSample> m_HeadHistory = new Queue<HeadSample>();

        struct HeadSample { public double T; public Vector3 Position; public Quaternion Rotation; }
        sealed class PendingLock { public long Seq; public double T; public Vector3 HeadPosition; public Quaternion HeadRotation; public ulong CamTs; }
        int m_LastTick = -1;
        float m_StepSeconds;
        bool m_Quitting;
        bool m_SummaryWritten;
        Coroutine m_Sequence;
        readonly Dictionary<string, int> m_Repeats = new Dictionary<string, int>();

        AudioSource m_Audio;
        AudioClip m_Beep, m_Tick, m_ReadyBeep;

        public HelloMR HelloMR { get => m_HelloMR; set => m_HelloMR = value; }
        public XREALRGBCameraTexture CameraTexture { get => m_CameraTexture; set => m_CameraTexture = value; }

        // ---- state read by GateHud
        public CameraSource Source => m_Source;
        public bool XRActive => XREALUtility.GetActiveLoader() != null;
        public bool GlassesConnected => m_GlassesConnected;
        public int Marks => m_Marks;
        public string GateTitle { get; private set; } = "";
        public string StepTitle { get; private set; } = "STARTING";
        public string StepDetail { get; private set; } = "";
        public string NextTitle { get; private set; } = "";
        public int StepIndex { get; private set; }
        public int StepCount { get; private set; }
        public double StepStart { get; private set; } = double.NegativeInfinity;
        public double StepEnd { get; private set; } = double.NaN;
        public bool IsCooldown => m_Args != null && m_Args.Gate == GateKind.Cooldown;
        /// <summary>G4: banner at the top, no camera preview, so the overlay in the middle stays clear.</summary>
        public bool IsLight => m_Args != null && (m_Args.Gate == GateKind.G4 || m_Args.Gate == GateKind.G5);
        public int Bursts => m_Bursts;
        public bool IsG5 => m_Args != null && m_Args.Gate == GateKind.G5;
        public bool IsCalib => m_Args != null && m_Args.Gate == GateKind.Calib;
        /// <summary>Calibration: where the board should be in the image (normalised, y down), or null.</summary>
        public Rect? CalibZone { get; private set; }
        /// <summary>G5: one line for the HUD, and its state (0 none, 1 locked, 2 searching, 3 frozen).</summary>
        public string LockStatus { get; private set; } = "";
        public int LockState { get; private set; }

        void Awake()
        {
            m_Args = GateArgs.FromEnvironment();
            m_Log = GateRunLog.Create(m_Args);
            m_Camera = GetComponent<CameraProbe>();
            m_Pose = GetComponent<PoseProbe>();
            m_Device = GetComponent<DeviceInfoProbe>();
            m_Perf = GetComponent<PerfProbe>();
            m_Display = GetComponent<DisplayProbe>();
            m_Overlay = GetComponent<GateOverlay>();
            if (m_CameraTexture == null) m_CameraTexture = FindAnyObjectByType<XREALRGBCameraTexture>();
            if (m_HelloMR == null) m_HelloMR = FindAnyObjectByType<HelloMR>();

            var vendor = GetComponent<VendorCameraSource>();
            if (vendor.Texture == null) vendor.Texture = m_CameraTexture;
            var drain = GetComponent<DrainingCameraSource>();
            m_Source = m_Args.Reader == "drain" ? (CameraSource)drain : vendor;
            (m_Source == drain ? (Behaviour)vendor : drain).enabled = false;

            // The sample's full-screen YUV RawImage renders solid green while no frames arrive (14 Oct
            // G2) and spans both eyes in XR; GateHud draws a small preview instead.
            if (drain.Preview != null)
            {
                drain.Preview.gameObject.SetActive(false);
                drain.Preview = null;
            }

            // G4 and G5 save exactly one snapshot per burst (its last frame) instead of one every 2 s.
            m_Camera.Attach(m_Log, m_Source, (m_Args.Gate == GateKind.G4 || m_Args.Gate == GateKind.G5) ? double.PositiveInfinity : m_Args.SnapshotEvery);
            m_Pose.Attach(m_Log);
            m_Device.Attach(m_Log);
            m_Display.Attach(m_Log);
            m_Perf.Attach(m_Log, m_Camera, m_Pose, () => m_StepId);
            GateTitle = MakeGateTitle();
            GetComponent<GateHud>().Attach(this, m_Camera, m_Pose, m_Perf, BuildStamp());

            if (FindAnyObjectByType<AudioListener>() == null)
                gameObject.AddComponent<AudioListener>();
            m_Audio = gameObject.AddComponent<AudioSource>();
            m_Audio.spatialBlend = 0f;
            m_Audio.volume = 1f;
            m_Beep = Tone("beep", 880f, 0.18f);
            m_Tick = Tone("tick", 660f, 0.07f);
            m_ReadyBeep = Tone("ready", 1320f, 0.5f);
            m_Speaker = Speaker.TryStart((result, detail) => m_Log.Event("speech", result, detail));

            NativeTracking.Install();
            m_Log.Event("native_log", NativeTracking.Installed ? "INSTALLED" : "UNAVAILABLE", NativeTracking.InstallError);
        }

        void OnDestroy() => NativeTracking.Uninstall();

        string m_StepId = "setup";

        void OnEnable()
        {
            XREALSessionManager.OnGlassesConnect += OnGlassesConnect;
            XREALSessionManager.OnGlassesDisconnect += OnGlassesDisconnect;
            XREALSessionManager.OnStartSubsystems += OnStartSubsystems;
            XREALSessionManager.OnStopSubsystems += OnStopSubsystems;
            XREALCallbackHandler.OnXREALGlassesPowerSave += OnPowerSave;
            XREALCallbackHandler.OnXREALGlassesDisconnect += OnNativeDisconnect;
            XREALCallbackHandler.OnXREALError += OnXrealError;
            XREALPlugin.OnTrackingTypeChanged += OnTrackingTypeChanged;
            Application.logMessageReceived += OnUnityLog;
        }

        void OnDisable()
        {
            XREALSessionManager.OnGlassesConnect -= OnGlassesConnect;
            XREALSessionManager.OnGlassesDisconnect -= OnGlassesDisconnect;
            XREALSessionManager.OnStartSubsystems -= OnStartSubsystems;
            XREALSessionManager.OnStopSubsystems -= OnStopSubsystems;
            XREALCallbackHandler.OnXREALGlassesPowerSave -= OnPowerSave;
            XREALCallbackHandler.OnXREALGlassesDisconnect -= OnNativeDisconnect;
            XREALCallbackHandler.OnXREALError -= OnXrealError;
            XREALPlugin.OnTrackingTypeChanged -= OnTrackingTypeChanged;
            Application.logMessageReceived -= OnUnityLog;
        }

        void Start()
        {
            m_Device.WriteStatic(m_Args);
            foreach (var w in m_Args.Warnings)
                m_Log.Event("arg_warning", "", w);
            m_Log.Event("gate_start", m_Args.GateName,
                $"order={m_Args.Order} reader={m_Source.SourceName} flushLag={m_Args.FlushLag.ToString(CultureInfo.InvariantCulture)} " +
                $"label={m_Args.Label} dir={m_Log.Directory}");
            if (m_HelloMR == null) m_Log.Event("setup_warning", "", "HelloMR not found in scene");
            if (m_CameraTexture == null) m_Log.Event("setup_warning", "", "XREALRGBCameraTexture not found in scene");
            if (m_Overlay.Material == null) m_Log.Event("setup_warning", "", "GateOverlay material not assigned");
            m_Display.Log("start");

            switch (m_Args.Gate)
            {
                case GateKind.G1: m_Sequence = StartCoroutine(RunG1()); break;
                case GateKind.G2: m_Sequence = StartCoroutine(RunG2()); break;
                case GateKind.G3: m_Sequence = StartCoroutine(RunG3()); break;
                case GateKind.G4: m_Sequence = StartCoroutine(RunG4()); break;
                case GateKind.G5: m_Sequence = StartCoroutine(RunG5()); break;
                case GateKind.Calib: m_Sequence = StartCoroutine(RunCalib()); break;
                case GateKind.Cooldown: m_Sequence = StartCoroutine(RunCooldown()); break;
                default:
                    Show("manual", "MANUAL MODE", "Keys: C camera, X XR, 3/6 DoF, R recenter, M mark, Esc quit", "");
                    StartCoroutine(ProbeWhenConnected());
                    break;
            }
        }

        void Update()
        {
            HandleKeys();
            double now = Time.realtimeSinceStartupAsDouble;

            int reason = m_Pose.LastTrackingReason;
            if (reason != m_LastReason)
            {
                m_Log.Event("tracking_reason", NativeTracking.Name(reason), reason.ToString(CultureInfo.InvariantCulture));
                if (m_InProtocol && m_LastReason == NativeTracking.Ready && reason != NativeTracking.NoValue
                    && now - m_LastLostCue > LostCueCooldown)
                {
                    Say("Tracking lost");
                    m_LastLostCue = now;
                }
                m_LastReason = reason;
            }

            // 3-2-1 ticks before the end of long steps.
            if (!double.IsNaN(StepEnd) && m_StepSeconds >= TickFromStepSeconds)
            {
                int left = (int)Math.Ceiling(StepEnd - now);
                if (left >= 1 && left <= 3 && left != m_LastTick)
                {
                    m_Audio.PlayOneShot(m_Tick);
                    m_LastTick = left;
                }
            }

            // Optional flush: restart the camera when its backlog exceeds the threshold.
            if (m_Args.FlushLag > 0 && m_Source.IsCapturing && !m_Quitting
                && m_Camera.LagGrowth > m_Args.FlushLag && now - m_LastFlush > FlushCooldown)
            {
                m_LastFlush = now;
                StartCoroutine(FlushCamera());
            }

            RecordHead(now);
            if (m_PoseTask != null && m_PoseTask.IsCompleted)
            {
                Task<PoseResult> done = m_PoseTask;
                PendingLock pending = m_Pending;
                m_PoseTask = null;
                m_Pending = null;
                ApplyLock(done, pending);
            }

            while (NativeTracking.TryDequeue(out string nativeLine))
                m_Log.Native(nativeLine);
            m_Log.FlushIfDue();
        }

        void OnApplicationQuit()
        {
            NativeTracking.Uninstall();
            while (NativeTracking.TryDequeue(out string nativeLine))
                m_Log.Native(nativeLine);
            m_Log.Event("native_log", "SUMMARY",
                $"lines={NativeTracking.LinesSeen} reason_lines={NativeTracking.ReasonLinesSeen} dropped={NativeTracking.DroppedLines}");
            WriteRepeatSummary();
            m_Log.Event("application_quit", "", "");
            m_Log.Flush();
            m_Speaker?.Dispose();
            m_Speaker = null;
        }

        // ---------------------------------------------------------------- sequences

        IEnumerator ProbeWhenConnected()
        {
            yield return WaitUntilOrTimeout(() => m_GlassesConnected, float.PositiveInfinity);
            m_Device.Probe("connect");
        }

        IEnumerator RunG1()
        {
            Say("Gate one. Camera only.");
            string connect = null;
            yield return Connect(r => connect = r);
            if (connect != "OK") { yield return Quit(connect); yield break; }

            StartCapture("main");
            yield return RunProtocol(GateProtocol.G1());
            StopCapture("final");
            yield return Quit("COMPLETED");
        }

        IEnumerator RunG2()
        {
            Say("Gate two. Six degrees of freedom.");
            string connect = null;
            yield return Connect(r => connect = r);
            if (connect != "OK") { yield return Quit(connect); yield break; }

            string xr = null;
            yield return StartXR(r => xr = r);
            if (xr != "OK") { yield return Quit(xr); yield break; }

            yield return SwitchTracking(TrackingType.MODE_6DOF);
            // G2 continues without READY: a pose that never becomes ready is itself the result.
            yield return WaitTrackingReady(_ => { });
            yield return RunProtocol(GateProtocol.G2());
            yield return Quit("COMPLETED");
        }

        IEnumerator RunG3()
        {
            Say("Gate three. Camera and tracking.");
            string connect = null;
            yield return Connect(r => connect = r);
            if (connect != "OK") { yield return Quit(connect); yield break; }

            string order = m_Args.Order;
            if (order == "camfirst")
                StartCapture("before_xr");

            string xr = null;
            yield return StartXR(r => xr = r);
            if (xr != "OK") { yield return Quit(xr); yield break; }

            yield return SwitchTracking(order == "3dof" ? TrackingType.MODE_3DOF : TrackingType.MODE_6DOF);

            if (order == "tracked")
            {
                bool ready = false;
                yield return WaitTrackingReady(r => ready = r);
                if (!ready) { yield return Quit("TRACKING_TIMEOUT"); yield break; }
            }
            if (order != "camfirst")
            {
                StartCapture(order == "tracked" ? "after_tracking_ready" : order == "3dof" ? "after_3dof" : "after_6dof_switch");
                Say("Camera on");
            }

            yield return RunProtocol(GateProtocol.G3());
            StopCapture("end");
            yield return Quit("COMPLETED");
        }

        /// <summary>
        /// G4 light overlay (guide §16): XR at 30 fps, 6DoF (<c>-order tracked</c>) or 3DoF
        /// (<c>-order 3dof</c>), a world-locked grid placed on the board, no video on the glasses, and the
        /// camera started only for short bursts (<c>-burstEvery</c>, <c>-burstFrames</c>; 0 = never).
        /// Restarting the camera empties the SDK's decode queue (14 Oct G3c-flush: lag back to 0-0.2 s
        /// after every restart), so each burst frame is fresh and the decoder idles in between.
        /// </summary>
        IEnumerator RunG4()
        {
            bool sixDof = m_Args.Order != "3dof";
            Say(sixDof ? "Gate four. Light overlay, six degrees of freedom." : "Gate four. Light overlay, three degrees of freedom.");
            string connect = null;
            yield return Connect(r => connect = r);
            if (connect != "OK") { yield return Quit(connect); yield break; }

            string xr = null;
            yield return StartXR(r => xr = r);
            if (xr != "OK") { yield return Quit(xr); yield break; }

            yield return SwitchTracking(sixDof ? TrackingType.MODE_6DOF : TrackingType.MODE_3DOF);
            if (sixDof)
            {
                bool ready = false;
                yield return WaitTrackingReady(r => ready = r);
                if (!ready) Say("Continuing without tracking ready");
            }

            Show("place", "LOOK AT THE BOARD", "The grid will appear on it", "");
            Say("Look at the board. The grid appears in three seconds.");
            StepEnd = StepStart + 3.0;
            yield return new WaitForSecondsRealtime(3f);
            PlaceOverlay("start");

            if (m_Args.BurstEvery > 0)
                StartCoroutine(BurstLoop());
            else
                m_Log.Event("burst", "DISABLED", "burstEvery=0: camera never started");

            yield return RunProtocol(GateProtocol.G4());
            m_Bursting = false;
            if (m_Source.IsCapturing) StopCapture("end");
            m_Log.Event("burst_summary", m_Bursts, $"every={m_Args.BurstEvery.ToString(CultureInfo.InvariantCulture)}s frames={m_Args.BurstFrames}");
            yield return Quit("COMPLETED");
        }

        /// <summary>
        /// Eye calibration capture (guide §17): camera only, no XR, a snapshot every 0.5 s while the
        /// operator moves the 25 mm ChArUco board through the zones the HUD shows on the large preview.
        /// The launcher then runs <c>python -m xreal_gates calibrate</c> on the run.
        /// </summary>
        IEnumerator RunCalib()
        {
            Say("Calibration capture. Hold the twenty five millimetre board and keep your head still.");
            string connect = null;
            yield return Connect(r => connect = r);
            if (connect != "OK") { yield return Quit(connect); yield break; }
            StartCapture("calib");
            yield return RunProtocol(GateProtocol.Calib());
            StopCapture("end");
            yield return Quit("COMPLETED");
        }

        /// <summary>
        /// G5 locked overlay (guide §17): G4's light runtime (30 fps, 6DoF, camera bursts) plus the
        /// ADR-001 pose bridge. The last frame of each burst goes to the pose service while the head is
        /// nearly still; each pose is chained with the head pose and the RGB-camera-from-head pose into
        /// a world pose of the target, which <see cref="LockAnchor"/> filters. The overlay is drawn there.
        /// </summary>
        IEnumerator RunG5()
        {
            Say($"Gate five. Locked overlay on {SpokenTarget(m_Args.Target)}.");
            string connect = null;
            yield return Connect(r => connect = r);
            if (connect != "OK") { yield return Quit(connect); yield break; }

            // Pose service first, so a missing service ends the run before XR starts.
            Show("bridge", "CONNECTING TO POSE SERVICE", $"ws://127.0.0.1:{m_Args.Port} ({m_Args.Target})", "");
            m_WsBridge = new WebSocketPoseBridge();
            float[] vendor = VendorIntrinsics();
            Task<TargetOverlayGeometry> hello = m_WsBridge.ConnectAsync(m_Args.Port, m_Args.Target, Guid.NewGuid().ToString("N").Substring(0, 12),
                                                                        m_Log.Directory, vendor, m_BridgeCts.Token);
            yield return WaitUntilOrTimeout(() => hello.IsCompleted, BridgeConnectTimeout);
            if (!hello.IsCompleted || hello.IsFaulted || hello.IsCanceled)
            {
                string why = hello.IsFaulted ? hello.Exception?.GetBaseException().Message : "timeout";
                m_Log.Event("bridge", "NO_SERVICE", why);
                Say("Pose service not running");
                yield return Quit("NO_SERVICE");
                yield break;
            }
            TargetOverlayGeometry geometry = hello.Result;
            m_Bridge = m_WsBridge;
            m_Log.Event("bridge", "CONNECTED",
                $"target={geometry.Target} label=\"{geometry.Label}\" segments={geometry.Segments.Length / 6} " +
                $"points={geometry.Points.Length / 3} camera={geometry.CameraSource} ({geometry.CameraDescription})");
            m_Overlay.BuildTarget(geometry);
            if (geometry.CameraSource != "calibration") Say("Warning. No Eye calibration, using the vendor values.");

            string xr = null;
            yield return StartXR(r => xr = r);
            if (xr != "OK") { yield return Quit(xr); yield break; }
            yield return SwitchTracking(TrackingType.MODE_6DOF);
            bool ready = false;
            yield return WaitTrackingReady(r => ready = r);
            if (!ready) Say("Continuing without tracking ready");
            ReadRgbFromHead();

            LockStatus = "SEARCHING: look at the target";
            LockState = 2;
            StartCoroutine(BurstLoop());
            yield return RunProtocol(GateProtocol.G5());
            m_Bursting = false;
            if (m_Source.IsCapturing) StopCapture("end");
            m_Log.Event("lock_summary", m_LocksAccepted,
                $"sent={m_LocksSent} posed={m_LocksPosed} accepted={m_LocksAccepted} bursts={m_Bursts} " +
                $"window={m_Anchor.Count}");
            yield return Quit("COMPLETED");
        }

        static string SpokenTarget(string t) =>
            t == "plateA" ? "plate A" : t == "plateL" ? "plate L" : t == "board25" ? "the calibration board" : t;

        static float[] VendorIntrinsics()
        {
            try
            {
                Vector2 f = default, p = default;
                if (XREALPlugin.GetCameraIntrinsic(XREALComponent.XREAL_COMPONENT_RGB_CAMERA, ref f, ref p) && f.x > 0)
                    return new[] { f.x, f.y, p.x, p.y };
            }
            catch (Exception) { }
            return null; // the service then uses its stored vendor values
        }

        void ReadRgbFromHead()
        {
            try
            {
                Pose pose = default;
                m_HasRgbFromHead = XREALPlugin.GetDevicePoseFromHead(XREALComponent.XREAL_COMPONENT_RGB_CAMERA, ref pose);
                m_RgbFromHead = pose;
            }
            catch (Exception e)
            {
                m_HasRgbFromHead = false;
                m_Log.Event("rgb_from_head", "EXCEPTION", e.GetType().Name + ": " + e.Message);
            }
            if (!m_HasRgbFromHead) m_RgbFromHead = new Pose(Vector3.zero, Quaternion.identity);
            m_Log.Event("rgb_from_head", m_HasRgbFromHead ? "OK" : "UNAVAILABLE",
                $"pos={V(m_RgbFromHead.position)} rot={m_RgbFromHead.rotation.x.ToString("F5", CultureInfo.InvariantCulture)}," +
                $"{m_RgbFromHead.rotation.y.ToString("F5", CultureInfo.InvariantCulture)},{m_RgbFromHead.rotation.z.ToString("F5", CultureInfo.InvariantCulture)}," +
                $"{m_RgbFromHead.rotation.w.ToString("F5", CultureInfo.InvariantCulture)}");
        }

        Transform Head()
        {
            if (Camera.main != null) return Camera.main.transform;
            var cam = FindAnyObjectByType<Camera>();
            return cam != null ? cam.transform : null;
        }

        void RecordHead(double now)
        {
            if (!IsG5 || !XRActive) return;
            Transform head = Head();
            if (head == null) return;
            m_HeadHistory.Enqueue(new HeadSample { T = now, Position = head.position, Rotation = head.rotation });
            while (m_HeadHistory.Count > 0 && now - m_HeadHistory.Peek().T > HeadHistorySeconds)
                m_HeadHistory.Dequeue();
        }

        /// <summary>Head angular and linear speed over the last <see cref="HeadSpeedWindow"/> s.</summary>
        bool HeadSpeed(out float degPerSec, out float metresPerSec)
        {
            degPerSec = metresPerSec = float.NaN;
            if (m_HeadHistory.Count < 2) return false;
            HeadSample last = default, past = default;
            bool found = false;
            foreach (var h in m_HeadHistory) last = h;
            foreach (var h in m_HeadHistory)
                if (last.T - h.T <= HeadSpeedWindow) { past = h; found = true; break; }
            double dt = last.T - past.T;
            if (!found || dt <= 1e-3) return false;
            degPerSec = (float)(Quaternion.Angle(past.Rotation, last.Rotation) / dt);
            metresPerSec = (float)(Vector3.Distance(past.Position, last.Position) / dt);
            return true;
        }

        void TrySendLockFrame()
        {
            if (m_Bridge == null || !m_Bridge.IsAvailable) return;
            if (m_PoseTask != null) { m_Log.Event("lock_frame", "SKIP_BUSY", ""); return; }
            Transform head = Head();
            Texture2D y = m_Source.LumaTexture;
            if (head == null || y == null) return;
            if (!HeadSpeed(out float dps, out float mps) || dps > MaxLockDegPerSec || mps > MaxLockMetresPerSec)
            {
                m_Log.Event("lock_frame", "SKIP_MOTION",
                    $"deg_s={dps.ToString("F1", CultureInfo.InvariantCulture)} m_s={mps.ToString("F3", CultureInfo.InvariantCulture)}");
                return;
            }
            NativeArray<byte> raw = y.GetRawTextureData<byte>();
            int n = y.width * y.height;
            if (raw.Length < n) return;
            var data = new byte[n];
            NativeArray<byte>.Copy(raw, data, n);
            var img = new YGrayImage { Width = y.width, Height = y.height, Data = data, BottomUp = true, Sequence = ++m_LockSeq };
            m_Pending = new PendingLock
            {
                Seq = img.Sequence, T = Time.realtimeSinceStartupAsDouble, HeadPosition = head.position,
                HeadRotation = head.rotation, CamTs = m_Source.Timestamp,
            };
            m_LocksSent++;
            m_PoseTask = m_Bridge.EstimatePoseAsync(img, m_Source.Timestamp, m_BridgeCts.Token);
        }

        void ApplyLock(Task<PoseResult> task, PendingLock pending)
        {
            if (pending == null) return;
            if (task.IsFaulted || task.IsCanceled)
            {
                string why = task.IsFaulted ? task.Exception?.GetBaseException().Message : "cancelled";
                m_Log.Event("lock", "BRIDGE_ERROR", $"seq={pending.Seq} {why}");
                LockStatus = "POSE SERVICE ERROR";
                LockState = 2;
                return;
            }
            PoseResult r = task.Result;
            string common = $"seq={pending.Seq} markers={r.Markers} points={r.Points} rms_px={r.RmsPx.ToString("F2", CultureInfo.InvariantCulture)} " +
                            $"service_ms={r.ServiceMs.ToString("F0", CultureInfo.InvariantCulture)} rt_ms={r.RoundTripMs.ToString("F0", CultureInfo.InvariantCulture)} " +
                            $"cam_ts={pending.CamTs}";
            if (!r.Ok)
            {
                m_Log.Event("lock", "NO_POSE", $"{common} reason=\"{r.Reason}\"");
                if (!m_Anchor.HasAnchor) { LockStatus = $"SEARCHING: {r.Reason}"; LockState = 2; }
                return;
            }
            m_LocksPosed++;
            Quaternion camRot = CvRotationToUnity(r.RotationRowMajor);
            Vector3 camPos = new Vector3(r.TranslationCv.x, -r.TranslationCv.y, r.TranslationCv.z);
            Vector3 worldPos = pending.HeadPosition + pending.HeadRotation * (m_RgbFromHead.rotation * camPos + m_RgbFromHead.position);
            Quaternion worldRot = pending.HeadRotation * m_RgbFromHead.rotation * camRot;
            LockAnchor.Result res = m_Anchor.Add(worldPos, worldRot);
            if (res.Kind == LockAnchor.Kind.Accepted || res.Kind == LockAnchor.Kind.First || res.Kind == LockAnchor.Kind.Relocked)
                m_LocksAccepted++;
            m_Log.Event("lock", res.Kind.ToString().ToUpperInvariant(),
                $"{common} jump_mm={(res.JumpMetres * 1000f).ToString("F1", CultureInfo.InvariantCulture)} " +
                $"jump_deg={res.JumpDegrees.ToString("F2", CultureInfo.InvariantCulture)} window={m_Anchor.Count} " +
                $"dist_m={r.TranslationCv.magnitude.ToString("F3", CultureInfo.InvariantCulture)} world={V(worldPos)} anchor={V(m_Anchor.Position)}");
            if (res.Moved) m_Overlay.SetTargetPose(m_Anchor.Position, m_Anchor.Rotation);
            if (res.Kind == LockAnchor.Kind.First)
            {
                m_Audio.PlayOneShot(m_ReadyBeep);
                Say("Locked");
                Screenshot("first_lock", force: true);
            }
            string jump = $"{(res.JumpMetres * 1000f).ToString("F0", CultureInfo.InvariantCulture)} mm";
            switch (res.Kind)
            {
                case LockAnchor.Kind.Frozen: LockStatus = $"FROZEN: camera says {jump} off"; LockState = 3; break;
                case LockAnchor.Kind.Rejected: LockStatus = $"LOCKED, last pose rejected ({jump} off)"; LockState = 1; break;
                default: LockStatus = $"LOCKED ({m_Anchor.Count} poses, last {jump})"; LockState = 1; break;
            }
        }

        /// <summary>OpenCV rotation (row-major) to Unity: F R F with F = diag(1, -1, 1) (THEORY §6.3).</summary>
        static Quaternion CvRotationToUnity(float[] r)
        {
            float S(int i) => i == 1 ? -1f : 1f;
            Vector3 col1 = new Vector3(S(0) * r[1] * S(1), S(1) * r[4] * S(1), S(2) * r[7] * S(1));
            Vector3 col2 = new Vector3(S(0) * r[2] * S(2), S(1) * r[5] * S(2), S(2) * r[8] * S(2));
            return Quaternion.LookRotation(col2, col1);
        }

        static string V(Vector3 v) =>
            $"{v.x.ToString("F4", CultureInfo.InvariantCulture)},{v.y.ToString("F4", CultureInfo.InvariantCulture)},{v.z.ToString("F4", CultureInfo.InvariantCulture)}";

        void PlaceOverlay(string why)
        {
            Transform head = Camera.main != null ? Camera.main.transform : null;
            if (head == null)
            {
                var cam = FindAnyObjectByType<Camera>();
                head = cam != null ? cam.transform : null;
            }
            if (head == null) { m_Log.Event("overlay", "NO_CAMERA", why); return; }
            m_Overlay.PlaceInFront(head, OverlayDistance);
            Vector3 p = m_Overlay.Position;
            m_Log.Event("overlay", "PLACED",
                $"{why} pos={p.x.ToString("F3", CultureInfo.InvariantCulture)},{p.y.ToString("F3", CultureInfo.InvariantCulture)},{p.z.ToString("F3", CultureInfo.InvariantCulture)} " +
                $"yaw={m_Overlay.YawDeg.ToString("F1", CultureInfo.InvariantCulture)} size={GateOverlay.Width.ToString(CultureInfo.InvariantCulture)}x" +
                $"{GateOverlay.Height.ToString(CultureInfo.InvariantCulture)}m cell={GateOverlay.Cell.ToString(CultureInfo.InvariantCulture)}m");
            Screenshot("overlay_" + why, force: true);
        }

        /// <summary>
        /// One burst every <c>BurstEvery</c> s: start the camera, wait for <c>BurstFrames</c> unique
        /// frames (or <see cref="BurstTimeoutSeconds"/>), save the last one, stop. Logged per burst:
        /// time to the first frame, time on, and the lag growth of the last frame.
        /// </summary>
        IEnumerator BurstLoop()
        {
            m_Bursting = true;
            while (m_Bursting && !m_Quitting)
            {
                double start = Time.realtimeSinceStartupAsDouble;
                m_Bursts++;
                if (!StartCapture($"burst_{m_Bursts}"))
                {
                    yield return new WaitForSecondsRealtime((float)m_Args.BurstEvery);
                    continue;
                }
                bool requested = false;
                if (m_Args.BurstFrames <= 1)
                {
                    m_Camera.SnapshotNextFrame();
                    requested = true;
                }
                double first = double.NaN;
                while (m_Camera.FramesThisCapture < m_Args.BurstFrames
                       && Time.realtimeSinceStartupAsDouble - start < BurstTimeoutSeconds && m_Bursting && !m_Quitting)
                {
                    if (double.IsNaN(first) && m_Camera.FramesThisCapture > 0) first = Time.realtimeSinceStartupAsDouble - start;
                    if (!requested && m_Camera.FramesThisCapture >= m_Args.BurstFrames - 1)
                    {
                        m_Camera.SnapshotNextFrame();
                        requested = true;
                    }
                    yield return null;
                }
                if (double.IsNaN(first) && m_Camera.FramesThisCapture > 0) first = Time.realtimeSinceStartupAsDouble - start;
                int frames = m_Camera.FramesThisCapture;
                double on = Time.realtimeSinceStartupAsDouble - start;
                if (IsG5 && frames >= m_Args.BurstFrames) TrySendLockFrame();
                if (m_Source.IsCapturing) StopCapture($"burst_{m_Bursts}");
                m_Log.Event("burst", frames >= m_Args.BurstFrames ? "OK" : frames > 0 ? "SHORT" : "NO_FRAMES",
                    $"n={m_Bursts} frames={frames} first_s={GateRunLog.F(first, "F3")} on_s={on.ToString("F3", CultureInfo.InvariantCulture)} " +
                    $"lag_growth_s={GateRunLog.F(m_Camera.LagGrowth, "F3")} cam_ts={m_Camera.LastTimestamp}");
                double wait = m_Args.BurstEvery - (Time.realtimeSinceStartupAsDouble - start);
                if (wait > 0) yield return new WaitForSecondsRealtime((float)wait);
                else yield return null;
            }
        }

        /// <summary>
        /// Between XR runs: is the Surface cool again? An idle CPU clocks down by itself, so the clock is
        /// read under load: two busy threads for 20 s, judged on the mean clock of the last 5 s, then
        /// 20 s of rest. One probe at or above <see cref="CoolEnoughPct"/> ends the check ("Cool enough").
        /// A short probe is not enough: the CPU's turbo budget refills within seconds of idling, so
        /// even a hot Surface reads full clock for the first ~10 s (14 Oct, build 0.2.006). Every
        /// probe also logs the hottest thermal zone, to compare cooling methods and recalibrate.
        /// </summary>
        IEnumerator RunCooldown()
        {
            Say("Cool-down check. Leave the Surface alone.");
            double t0 = Time.realtimeSinceStartupAsDouble;
            int probe = 0;
            while (true)
            {
                probe++;
                Show("cool_load", $"MEASURING (check {probe})", "20 s CPU load to read the sustained clock", "");
                StepEnd = StepStart + CooldownProbeLoadSeconds;
                m_StepSeconds = CooldownProbeLoadSeconds;
                var stop = new ManualResetEventSlim(false);
                var load = new[] { new Thread(() => Spin(stop)), new Thread(() => Spin(stop)) };
                foreach (var th in load) { th.IsBackground = true; th.Start(); }
                double loadStart = Time.realtimeSinceStartupAsDouble, sum = 0;
                int n = 0;
                while (Time.realtimeSinceStartupAsDouble - loadStart < CooldownProbeLoadSeconds)
                {
                    yield return new WaitForSecondsRealtime(1f);
                    double c = m_Perf.CpuPerformancePct;
                    if (!double.IsNaN(c) && Time.realtimeSinceStartupAsDouble - loadStart >= CooldownProbeLoadSeconds - CooldownJudgeSeconds)
                    {
                        sum += c;
                        n++;
                    }
                }
                stop.Set();
                double clock = n > 0 ? sum / n : double.NaN;
                string temp = double.IsNaN(m_Perf.TempMaxC) ? "n/a" : m_Perf.TempMaxC.ToString("F1", CultureInfo.InvariantCulture);
                m_Log.Event("cooldown_probe", double.IsNaN(clock) ? "n/a" : clock.ToString("F0", CultureInfo.InvariantCulture),
                            $"probe={probe} threshold={CoolEnoughPct.ToString(CultureInfo.InvariantCulture)} " +
                            $"load={CooldownProbeLoadSeconds.ToString(CultureInfo.InvariantCulture)}s judged_last={CooldownJudgeSeconds.ToString(CultureInfo.InvariantCulture)}s " +
                            $"temp_max_c={temp} temps={m_Perf.Temps}");

                if (double.IsNaN(clock))
                {
                    Show("cool_na", "NO CPU COUNTERS", "Cannot read the clock on this machine", "");
                    Say("Cannot read the CPU clock");
                    yield return new WaitForSecondsRealtime(3f);
                    yield return Quit("NO_COUNTERS");
                    yield break;
                }
                string pct = clock.ToString("F0", CultureInfo.InvariantCulture);
                double elapsed = Time.realtimeSinceStartupAsDouble - t0;
                if (clock >= CoolEnoughPct)
                {
                    Show("cool_ready", "COOL ENOUGH",
                         $"Sustained CPU clock {pct}% after {(elapsed / 60).ToString("F1", CultureInfo.InvariantCulture)} min. Start the next gate.", "");
                    Say("Cool enough. Start the next gate.");
                    yield return new WaitForSecondsRealtime(4f);
                    yield return Quit("COOL");
                    yield break;
                }
                if (elapsed > CooldownMaxSeconds)
                {
                    Say("Still hot after ten minutes");
                    yield return Quit("COOLDOWN_TIMEOUT");
                    yield break;
                }
                Show("cool_wait", $"COOLING: CLOCK {pct}%",
                     $"Ready at {CoolEnoughPct.ToString("F0", CultureInfo.InvariantCulture)}% sustained. " +
                     $"Waiting {(elapsed / 60).ToString("F1", CultureInfo.InvariantCulture)} min so far. Hottest zone {temp} C.", "");
                StepEnd = StepStart + CooldownRestSeconds;
                Say($"{pct} percent. Still warm.");
                yield return new WaitForSecondsRealtime(CooldownRestSeconds);
            }
        }

        static double s_SpinSink;

        static void Spin(ManualResetEventSlim stop)
        {
            double x = 1.0;
            while (!stop.IsSet)
                for (int i = 0; i < 100000; i++) x = Math.Sqrt(x + i);
            s_SpinSink = x; // keep the loop observable
        }

        /// <summary>Waits for the glasses and checks that the native session was created.</summary>
        IEnumerator Connect(Action<string> result)
        {
            Show("connect", "CONNECTING", "Keep the glasses plugged into the Surface", "");
            m_Log.Event("wait_glasses", "begin", $"timeout={GlassesTimeout}s");
            bool ok = false;
            yield return WaitUntilOrTimeout(() => m_GlassesConnected, GlassesTimeout, r => ok = r);
            m_Log.Event("wait_glasses", ok ? "CONNECTED" : "NO_GLASSES", "");
            if (!ok) { result("NO_GLASSES"); yield break; }

            // XREALSessionManager raises OnGlassesConnect even when CreateSession failed; a native
            // NativeAPI+ error at this point means there is no session (e.g. a blocked runtime DLL).
            yield return null;
            if (m_SessionFailed)
            {
                m_Log.Event("session", "NO_SESSION", "native NativeAPI error during session creation; see xreal_error rows");
                result("NO_SESSION");
                yield break;
            }
            m_Log.Event("session", "OK", "");
            m_Device.Probe("connect");
            result("OK");
        }

        IEnumerator StartXR(Action<string> result)
        {
            Show("enter_xr", "PUT THE GLASSES ON", "Entering XR, the screens will rearrange", "");
            Say("Put the glasses on");
            if (!EnterXR()) { result("XR_REFUSED"); yield break; }
            bool started = false;
            yield return WaitUntilOrTimeout(() => m_SubsystemsStarted, EnterXRTimeout, r => started = r);
            m_Log.Event("enter_xr_wait", started ? "STARTED" : "TIMEOUT", $"timeout={EnterXRTimeout}s");
            if (!started) { result("XR_TIMEOUT"); yield break; }
            m_Device.Probe("xr");
            m_Display.Log("xr");
            if (m_Args.RenderFps == 30)
            {
                // HelloMR.OnEnterXR sets vSyncCount = 1 after this callback; override a few frames later.
                for (int i = 0; i < 5; i++) yield return null;
                QualitySettings.vSyncCount = 2;
                m_Log.Event("render_fps", 30, "vSyncCount=2 (half of the 60 Hz display rate)");
            }
            result("OK");
        }

        /// <summary>
        /// Waits until the native tracking reason is None (0) for <see cref="TrackingReadyHold"/> s.
        /// The operator looks around slowly; Scanning (8) means the tracker wants to see the environment.
        /// </summary>
        IEnumerator WaitTrackingReady(Action<bool> result)
        {
            Show("wait_tracking", "LOOK AROUND SLOWLY", "Well-lit, textured part of the room, until TRACKING READY", "");
            Say("Look around slowly until tracking is ready");
            m_Log.Event("tracking_ready", "WAIT", $"timeout={TrackingReadyTimeout}s");
            double t0 = Time.realtimeSinceStartupAsDouble;
            double readySince = double.NaN;
            double lastPrompt = t0;
            StepStart = t0;
            StepEnd = t0 + TrackingReadyTimeout;
            while (true)
            {
                double now = Time.realtimeSinceStartupAsDouble;
                // 14 Oct: both timeouts sat in Scanning while the operator held still; G2 became ready
                // only during a side-step. The tracker needs parallax, so ask for movement.
                int r = m_Pose.LastTrackingReason;
                if (r != NativeTracking.Ready && r != NativeTracking.NoValue && now - lastPrompt >= ScanPromptSeconds)
                {
                    lastPrompt = now;
                    StepTitle = "STEP SIDE TO SIDE SLOWLY";
                    StepDetail = $"Tracker: {NativeTracking.Name(r)}. Small side-steps, look at the room, not at a blank wall";
                    m_Log.Event("tracking_ready", "PROMPT", NativeTracking.Name(r));
                    Say("Still scanning. Step slowly side to side.");
                }
                if (m_Pose.LastTrackingReason == NativeTracking.Ready)
                {
                    if (double.IsNaN(readySince)) readySince = now;
                    if (now - readySince >= TrackingReadyHold)
                    {
                        m_Log.Event("tracking_ready", "READY", $"after={(now - t0).ToString("F1", CultureInfo.InvariantCulture)}s");
                        Screenshot("tracking_ready", force: true);
                        m_Audio.PlayOneShot(m_ReadyBeep);
                        Say("Tracking ready");
                        yield return new WaitForSecondsRealtime(1.5f); // let the cue finish before step 1
                        result(true);
                        yield break;
                    }
                }
                else
                {
                    readySince = double.NaN;
                }
                // No report at all (native log route not delivering): do not block the run on a
                // missing signal; continue after a fixed scan time and say so in the log.
                if (m_Pose.LastTrackingReason == NativeTracking.NoValue && NativeTracking.ReasonLinesSeen == 0
                    && now - t0 > NoReportFallbackSeconds)
                {
                    m_Log.Event("tracking_ready", "ASSUMED",
                        $"no tracking report after {NoReportFallbackSeconds}s (native_log installed={NativeTracking.Installed})");
                    Say("Tracking state unknown. Continuing.");
                    yield return new WaitForSecondsRealtime(1.5f);
                    result(true);
                    yield break;
                }
                if (now - t0 > TrackingReadyTimeout)
                {
                    m_Log.Event("tracking_ready", "TIMEOUT", $"last={NativeTracking.Name(m_Pose.LastTrackingReason)}");
                    Say("Tracking not ready");
                    result(false);
                    yield break;
                }
                yield return null;
            }
        }

        IEnumerator RunProtocol(List<ProtocolStep> steps)
        {
            StepCount = steps.Count;
            m_InProtocol = true;
            m_Log.Event("protocol", "begin", $"steps={steps.Count} seconds={GateProtocol.TotalSeconds(steps).ToString(CultureInfo.InvariantCulture)}");
            for (int i = 0; i < steps.Count; i++)
            {
                var step = steps[i];
                StepIndex = i + 1;
                Show(step.Id, step.Title, step.Detail, i + 1 < steps.Count ? steps[i + 1].Title : "END");
                double now = Time.realtimeSinceStartupAsDouble;
                StepStart = now;
                StepEnd = now + step.Seconds;
                m_StepSeconds = step.Seconds;
                m_LastTick = -1;
                switch (step.Action)
                {
                    case StepAction.CameraStart: StartCapture(step.Id); break;
                    case StepAction.CameraStop: StopCapture(step.Id); break;
                    case StepAction.Track3DoF: StartCoroutine(SwitchTracking(TrackingType.MODE_3DOF)); break;
                    case StepAction.Track6DoF: StartCoroutine(SwitchTracking(TrackingType.MODE_6DOF)); break;
                    case StepAction.AnchorFreeze: m_Anchor.Frozen = true; m_Log.Event("anchor", "FROZEN", step.Id); break;
                    case StepAction.AnchorLive: m_Anchor.Frozen = false; m_Log.Event("anchor", "LIVE", step.Id); break;
                }
                CalibZone = step.Zone;
                m_Log.Event("step", step.Id, $"{step.Seconds.ToString(CultureInfo.InvariantCulture)}s {step.Title}: {step.Detail}");
                m_Audio.PlayOneShot(m_Beep);
                Say(step.Speech);
                Screenshot(step.Id, force: i == 0);
                yield return new WaitForSecondsRealtime(step.Seconds);
            }
            m_InProtocol = false;
            CalibZone = null;
            m_Log.Event("protocol", "end", "");
            Show("done", "DONE", "Protocol complete, finishing", "");
            Say("Done");
        }

        /// <summary>
        /// Saves what the app drew (eye views + HUD) to screens/. In XR only, at most every
        /// <see cref="ScreenshotEverySeconds"/> s unless forced. If the glasses show the view upside down
        /// while the screenshot is upright, the flip happens after the app (Windows display rotation).
        /// </summary>
        void Screenshot(string why, bool force = false)
        {
            if (!XRActive || m_Log == null) return;
            double now = Time.realtimeSinceStartupAsDouble;
            if (!force && now - m_LastScreenshot < ScreenshotEverySeconds) return;
            m_LastScreenshot = now;
            try
            {
                string dir = Path.Combine(m_Log.Directory, "screens");
                Directory.CreateDirectory(dir);
                string file = $"{now.ToString("F1", CultureInfo.InvariantCulture)}_{why}.png";
                ScreenCapture.CaptureScreenshot(Path.Combine(dir, file));
                m_Log.Event("screenshot", file, $"{Screen.width}x{Screen.height}");
            }
            catch (Exception e)
            {
                m_Log.Event("screenshot", "EXCEPTION", e.GetType().Name + ": " + e.Message);
            }
        }

        void Show(string id, string title, string detail, string next)
        {
            m_StepId = id;
            StepTitle = title;
            StepDetail = detail;
            NextTitle = next;
            StepStart = Time.realtimeSinceStartupAsDouble;
            StepEnd = double.NaN;
            m_StepSeconds = 0;
        }

        void Say(string text)
        {
            if (!string.IsNullOrEmpty(text)) m_Speaker?.Say(text);
        }

        static IEnumerator WaitUntilOrTimeout(Func<bool> condition, float timeout, Action<bool> result = null)
        {
            float t0 = Time.realtimeSinceStartup;
            while (!condition())
            {
                if (Time.realtimeSinceStartup - t0 > timeout)
                {
                    result?.Invoke(false);
                    yield break;
                }
                yield return null;
            }
            result?.Invoke(true);
        }

        // ---------------------------------------------------------------- actions

        bool StartCapture(string why)
        {
            m_Camera.NotifyCaptureStarted();
            bool ok;
            try { ok = m_Source.StartCapture(); }
            catch (Exception e)
            {
                m_Log.Event("start_capture", "EXCEPTION", $"{why}: {e.GetType().Name}: {e.Message}");
                return false;
            }
            m_CaptureSessions++;
            m_Log.Event("start_capture", ok, $"{why} session={m_CaptureSessions} reader={m_Source.SourceName}");
            return ok;
        }

        void StopCapture(string why)
        {
            bool ok;
            try { ok = m_Source.StopCapture(); }
            catch (Exception e)
            {
                m_Log.Event("stop_capture", "EXCEPTION", $"{why}: {e.GetType().Name}: {e.Message}");
                return;
            }
            m_Log.Event("stop_capture", ok, why);
        }

        IEnumerator FlushCamera()
        {
            m_Log.Event("camera_flush", "begin", $"lag={m_Camera.LagGrowth.ToString("F2", CultureInfo.InvariantCulture)}s");
            StopCapture("flush");
            yield return new WaitForSecondsRealtime(0.2f);
            if (!m_Quitting) StartCapture("flush");
        }

        bool EnterXR()
        {
            if (Application.isEditor)
            {
                m_Log.Event("enter_xr", "BLOCKED", "XR is disabled in the Editor (desktop safety, guide §1)");
                return false;
            }
            if (m_HelloMR == null) { m_Log.Event("enter_xr", "NO_HELLOMR", ""); return false; }
            if (!m_GlassesConnected) { m_Log.Event("enter_xr", "NO_GLASSES", ""); return false; }
            if (XRActive) { m_Log.Event("enter_xr", "ALREADY_ACTIVE", ""); return true; }
            m_SubsystemsStarted = false;
            m_HelloMR.OnClickSwitchXRButton();
            m_Log.Event("enter_xr", "REQUESTED", "HelloMR.OnClickSwitchXRButton");
            return true;
        }

        void ExitXR()
        {
            if (!XRActive || m_HelloMR == null) return;
            m_HelloMR.OnClickSwitchXRButton();
            m_Log.Event("exit_xr", "REQUESTED", "HelloMR.OnClickSwitchXRButton");
        }

        IEnumerator SwitchTracking(TrackingType target)
        {
            if (!XRActive)
            {
                m_Log.Event("switch_tracking", "IGNORED", $"{target}: XR not active");
                yield break;
            }
            m_Log.Event("switch_tracking", "REQUESTED", target.ToString());
            Task<bool> task = XREALPlugin.SwitchTrackingTypeAsync(target);
            while (!task.IsCompleted)
                yield return null;
            string after = XRActive ? XREALPlugin.GetTrackingType().ToString() : "";
            if (task.IsFaulted)
                m_Log.Event("switch_tracking", "EXCEPTION", $"{target}: {task.Exception?.GetBaseException().Message} after={after}");
            else
                m_Log.Event("switch_tracking", task.Result, $"{target} after={after}");
        }

        void Recenter()
        {
            var input = XREALUtility.GetLoadedSubsystem<XRInputSubsystem>();
            bool ok = input != null && input.TryRecenter();
            m_Log.Event("recenter", ok, input == null ? "no XRInputSubsystem" : "");
        }

        IEnumerator Quit(string result)
        {
            if (m_Quitting) yield break;
            m_Quitting = true;
            m_InProtocol = false;
            Show("quit", result == "COMPLETED" ? "FINISHED" : result, "Closing, you can take the glasses off", "");
            m_Log.Event("gate_result", result, m_Args.GateName);
            m_Log.Flush(); // 14 Oct G2: the process ended during Exit XR before the next periodic flush
            Say(result == "COMPLETED" ? "Finished. You can take the glasses off." : "Stopped");

            if (m_Source.IsCapturing)
                StopCapture("quit");
            if (m_WsBridge != null)
            {
                m_BridgeCts.Cancel();
                Task close = m_WsBridge.CloseAsync();
                yield return WaitUntilOrTimeout(() => close.IsCompleted, 3f);
                m_WsBridge.Dispose();
                m_WsBridge = null;
                m_Bridge = null;
                m_Log.Event("bridge", "CLOSED", $"sent={m_LocksSent} posed={m_LocksPosed} accepted={m_LocksAccepted}");
            }
            WriteRepeatSummary();
            m_Log.Flush();
            if (XRActive)
            {
                ExitXR();
                bool stopped = false;
                yield return WaitUntilOrTimeout(() => !XRActive, ExitXRTimeout, r => stopped = r);
                m_Log.Event("exit_xr_wait", stopped ? "STOPPED" : "TIMEOUT", "");
            }
            m_Log.Event("quit", "", "");
            m_Log.Flush();
            yield return new WaitForSecondsRealtime(1.5f); // let the last spoken cue finish
            m_Speaker?.Dispose();
            m_Speaker = null;
            NativeTracking.Uninstall();
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        // ---------------------------------------------------------------- keyboard

        void HandleKeys()
        {
            if (m_Quitting) return;

            if (Input.GetKeyDown(KeyCode.M))
            {
                m_Marks++;
                m_Log.Event("mark", m_Marks, m_StepId);
                m_Audio.PlayOneShot(m_Tick);
            }
            if (Input.GetKeyDown(KeyCode.C))
            {
                m_Log.Event("key", "C", "");
                if (m_Source.IsCapturing) StopCapture("key");
                else StartCapture("key");
            }
            if (Input.GetKeyDown(KeyCode.X))
            {
                m_Log.Event("key", "X", "");
                if (XRActive) ExitXR();
                else EnterXR();
            }
            if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3))
            {
                m_Log.Event("key", "3", "");
                StartCoroutine(SwitchTracking(TrackingType.MODE_3DOF));
            }
            if (Input.GetKeyDown(KeyCode.Alpha6) || Input.GetKeyDown(KeyCode.Keypad6))
            {
                m_Log.Event("key", "6", "");
                StartCoroutine(SwitchTracking(TrackingType.MODE_6DOF));
            }
            if (Input.GetKeyDown(KeyCode.R))
            {
                m_Log.Event("key", "R", "");
                Recenter();
            }
            if (Input.GetKeyDown(KeyCode.P) && XRActive)
            {
                m_Log.Event("key", "P", "");
                PlaceOverlay("key");
            }
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                m_Log.Event("key", "Esc", m_StepId);
                if (m_Sequence != null) StopCoroutine(m_Sequence);
                StartCoroutine(Quit("ABORTED_BY_USER"));
            }
        }

        // ---------------------------------------------------------------- lifecycle and native events

        void OnGlassesConnect()
        {
            m_GlassesConnected = true;
            string device = XREALSessionManager.Singleton != null ? XREALSessionManager.Singleton.ConnectedDevice.ToString() : "";
            m_Log.Event("glasses_connect", "", device);
        }

        void OnGlassesDisconnect()
        {
            m_GlassesConnected = false;
            m_Log.Event("glasses_disconnect", "", "");
        }

        void OnStartSubsystems()
        {
            m_SubsystemsStarted = true;
            m_Log.Event("start_subsystems", "", "");
        }

        void OnStopSubsystems()
        {
            m_SubsystemsStarted = false;
            m_Log.Event("stop_subsystems", "", "");
        }

        void OnPowerSave() => m_Log.Event("power_save", "", "");

        void OnNativeDisconnect(XREALGlassesDisconnectReason reason) =>
            m_Log.Event("native_disconnect", reason.ToString(), "");

        void OnXrealError(XREALErrorCode code, string message)
        {
            if (message != null && message.StartsWith("NativeAPI+", StringComparison.Ordinal))
                m_SessionFailed = true;
            if (FirstOccurrence($"xreal_error|{code}|{message}"))
                m_Log.Event("xreal_error", code.ToString(), message);
        }

        void OnTrackingTypeChanged(bool result, TrackingType target) =>
            m_Log.Event("tracking_type_changed", result, target.ToString());

        void OnUnityLog(string condition, string stackTrace, LogType type)
        {
            if (condition.StartsWith("[EyeLabGates]", StringComparison.Ordinal)) return;
            if (condition.Contains("OnNativeErrorCallback")) return; // duplicated by xreal_error
            bool interesting = type == LogType.Error || type == LogType.Exception || type == LogType.Assert
                               || condition.IndexOf("XREAL", StringComparison.OrdinalIgnoreCase) >= 0
                               || condition.StartsWith("[RGBCamera]", StringComparison.Ordinal);
            if (!interesting) return;
            string detail = condition.Length > 500 ? condition.Substring(0, 500) : condition;
            if (!FirstOccurrence($"unity_log|{type}|{detail}")) return;
            if (type == LogType.Exception && !string.IsNullOrEmpty(stackTrace))
                detail += " @ " + stackTrace.Split('\n')[0];
            m_Log.Event("unity_log", type.ToString(), detail);
        }

        /// <summary>True the first time a key is seen; later occurrences are only counted.</summary>
        bool FirstOccurrence(string key)
        {
            m_Repeats.TryGetValue(key, out int n);
            m_Repeats[key] = n + 1;
            return n == 0;
        }

        void WriteRepeatSummary()
        {
            if (m_SummaryWritten) return;
            m_SummaryWritten = true;
            foreach (var kv in m_Repeats)
                if (kv.Value > 1)
                    m_Log.Event("repeat_summary", kv.Value, kv.Key);
        }

        // ---------------------------------------------------------------- helpers

        string MakeGateTitle()
        {
            switch (m_Args.Gate)
            {
                case GateKind.G1: return "G1 camera only";
                case GateKind.G2: return "G2 6DoF";
                case GateKind.Cooldown: return "Cool-down check";
                case GateKind.Calib: return "Calibration capture (25 mm board)";
                case GateKind.G5:
                    return $"G5 locked overlay on {m_Args.Target}, camera {m_Args.BurstFrames} fr every " +
                           $"{m_Args.BurstEvery.ToString(CultureInfo.InvariantCulture)} s, {m_Args.RenderFps} fps";
                case GateKind.G4:
                    string burst = m_Args.BurstEvery > 0
                        ? $"camera {m_Args.BurstFrames} fr every {m_Args.BurstEvery.ToString(CultureInfo.InvariantCulture)} s"
                        : "no camera";
                    return $"G4 light overlay {(m_Args.Order == "3dof" ? "3DoF" : "6DoF")}, {burst}, {m_Args.RenderFps} fps";
                case GateKind.G3:
                    string reader = m_Source.SourceName == "drain" ? " +drain" : "";
                    string flush = (m_Args.FlushLag > 0 ? " +flush" : "") + (m_Args.RenderFps == 30 ? " @30fps" : "");
                    switch (m_Args.Order)
                    {
                        case "tracked": return $"G3c camera after READY{reader}{flush}";
                        case "3dof": return $"G3 camera + 3DoF{reader}{flush}";
                        case "camfirst": return $"G3b camera before XR{reader}{flush}";
                        default: return $"G3 camera after 6DoF switch{reader}{flush}";
                    }
                default: return "Manual";
            }
        }

        static string BuildStamp()
        {
            var info = Resources.Load<TextAsset>(DeviceInfoProbe.BuildInfoResource);
            if (info == null) return "editor";
            foreach (var line in info.text.Split('\n'))
            {
                int eq = line.IndexOf('=');
                if (eq > 0 && line.Substring(0, eq).Trim() == "utc_build")
                    return line.Substring(eq + 1).Trim();
            }
            return "?";
        }

        static AudioClip Tone(string name, float hz, float seconds)
        {
            const int rate = 44100;
            int n = (int)(rate * seconds);
            var data = new float[n];
            for (int i = 0; i < n; i++)
            {
                float env = Mathf.Min(1f, Mathf.Min(i, n - i) / (0.01f * rate)); // 10 ms fade in/out
                data[i] = 0.6f * env * Mathf.Sin(2f * Mathf.PI * hz * i / rate);
            }
            var clip = AudioClip.Create(name, n, 1, rate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
