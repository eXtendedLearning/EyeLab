using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using Unity.XR.XREAL;
using Unity.XR.XREAL.Experimental;
using Unity.XR.XREAL.Experimental.Samples;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLabGates
{
    /// <summary>
    /// Gate state machine (G1/G2/G3/manual) and guided protocol: timed steps announced by voice and
    /// beep and shown large by <see cref="GateHud"/>. Keyboard control and lifecycle event log. XR is
    /// entered and exited only through the scene's <see cref="HelloMR.OnClickSwitchXRButton"/>, i.e.
    /// the vendor flow. In the Editor, XR is refused: entering XR on the desktop changes its EDID and
    /// monitor layout (docs/XREAL-PC-GATES-G1-G3.md §1).
    /// </summary>
    [RequireComponent(typeof(CameraProbe), typeof(PoseProbe), typeof(DeviceInfoProbe))]
    [RequireComponent(typeof(PerfProbe), typeof(VendorCameraSource), typeof(DrainingCameraSource))]
    [RequireComponent(typeof(GateHud))]
    public sealed class GateController : MonoBehaviour
    {
        const float GlassesTimeout = 30f;
        const float EnterXRTimeout = 20f;
        const float ExitXRTimeout = 10f;
        const float TrackingReadyTimeout = 60f;
        const float TrackingReadyHold = 1f;
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

        void Awake()
        {
            m_Args = GateArgs.FromEnvironment();
            m_Log = GateRunLog.Create(m_Args);
            m_Camera = GetComponent<CameraProbe>();
            m_Pose = GetComponent<PoseProbe>();
            m_Device = GetComponent<DeviceInfoProbe>();
            m_Perf = GetComponent<PerfProbe>();
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

            m_Camera.Attach(m_Log, m_Source, m_Args.SnapshotEvery);
            m_Pose.Attach(m_Log);
            m_Device.Attach(m_Log);
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
        }

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

            switch (m_Args.Gate)
            {
                case GateKind.G1: m_Sequence = StartCoroutine(RunG1()); break;
                case GateKind.G2: m_Sequence = StartCoroutine(RunG2()); break;
                case GateKind.G3: m_Sequence = StartCoroutine(RunG3()); break;
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

            m_Log.FlushIfDue();
        }

        void OnApplicationQuit()
        {
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
            StepStart = t0;
            StepEnd = t0 + TrackingReadyTimeout;
            while (true)
            {
                double now = Time.realtimeSinceStartupAsDouble;
                if (m_Pose.LastTrackingReason == NativeTracking.Ready)
                {
                    if (double.IsNaN(readySince)) readySince = now;
                    if (now - readySince >= TrackingReadyHold)
                    {
                        m_Log.Event("tracking_ready", "READY", $"after={(now - t0).ToString("F1", CultureInfo.InvariantCulture)}s");
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
                }
                m_Log.Event("step", step.Id, $"{step.Seconds.ToString(CultureInfo.InvariantCulture)}s {step.Title}: {step.Detail}");
                m_Audio.PlayOneShot(m_Beep);
                Say(step.Speech);
                yield return new WaitForSecondsRealtime(step.Seconds);
            }
            m_InProtocol = false;
            m_Log.Event("protocol", "end", "");
            Show("done", "DONE", "Protocol complete, finishing", "");
            Say("Done");
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
                case GateKind.G3:
                    string reader = m_Source.SourceName == "drain" ? " +drain" : "";
                    string flush = m_Args.FlushLag > 0 ? " +flush" : "";
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
