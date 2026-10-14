using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Threading.Tasks;
using Unity.XR.XREAL;
using Unity.XR.XREAL.Experimental;
using Unity.XR.XREAL.Experimental.Samples;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLabGates
{
    /// <summary>
    /// Gate state machine (G1/G2/G3/manual), guided protocol (timed steps with HUD countdown and beeps),
    /// keyboard control, HUD and lifecycle event log. XR is entered and exited only through the scene's
    /// <see cref="HelloMR.OnClickSwitchXRButton"/>, i.e. the vendor flow. In the Editor, XR is refused:
    /// entering XR on the desktop changes its EDID and monitor layout (docs/XREAL-PC-GATES-G1-G3.md §1).
    /// </summary>
    [RequireComponent(typeof(CameraProbe), typeof(PoseProbe), typeof(DeviceInfoProbe))]
    [RequireComponent(typeof(PerfProbe), typeof(VendorCameraSource), typeof(DrainingCameraSource))]
    public sealed class GateController : MonoBehaviour
    {
        const float GlassesTimeout = 30f;
        const float EnterXRTimeout = 20f;
        const float ExitXRTimeout = 10f;
        const float TrackingReadyTimeout = 60f;
        const float TrackingReadyHold = 1f;

        [SerializeField] HelloMR m_HelloMR;
        [SerializeField] XREALRGBCameraTexture m_CameraTexture;

        GateArgs m_Args;
        GateRunLog m_Log;
        CameraProbe m_Camera;
        PoseProbe m_Pose;
        DeviceInfoProbe m_Device;
        PerfProbe m_Perf;
        CameraSource m_Source;

        bool m_GlassesConnected;
        bool m_SessionFailed;
        bool m_SubsystemsStarted;
        int m_Marks;
        int m_CaptureSessions;
        int m_LastReason = NativeTracking.NoValue;
        double m_StartTime;
        bool m_Quitting;
        bool m_SummaryWritten;
        Coroutine m_Sequence;
        readonly Dictionary<string, int> m_Repeats = new Dictionary<string, int>();

        // Guided-protocol state shown on the HUD.
        string m_StepId = "setup";
        string m_Instruction = "Starting...";
        string m_NextInstruction = "";
        int m_StepIndex, m_StepCount;
        double m_StepEnd = double.NaN;

        AudioSource m_Audio;
        AudioClip m_Beep, m_ReadyBeep;
        GUIStyle m_HudStyle;

        public HelloMR HelloMR { get => m_HelloMR; set => m_HelloMR = value; }
        public XREALRGBCameraTexture CameraTexture { get => m_CameraTexture; set => m_CameraTexture = value; }

        bool XRActive => XREALUtility.GetActiveLoader() != null;

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

            m_Camera.Attach(m_Log, m_Source, m_Args.SnapshotEvery);
            m_Pose.Attach(m_Log);
            m_Device.Attach(m_Log);
            m_Perf.Attach(m_Log, m_Camera, m_Pose, () => m_StepId);
            m_StartTime = Time.realtimeSinceStartupAsDouble;

            m_Audio = gameObject.AddComponent<AudioSource>();
            m_Audio.spatialBlend = 0f;
            m_Beep = Tone("beep", 880f, 0.15f);
            m_ReadyBeep = Tone("ready", 1320f, 0.5f);
        }

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
                $"order={m_Args.Order} reader={m_Source.SourceName} label={m_Args.Label} dir={m_Log.Directory}");
            if (m_HelloMR == null) m_Log.Event("setup_warning", "", "HelloMR not found in scene");
            if (m_CameraTexture == null) m_Log.Event("setup_warning", "", "XREALRGBCameraTexture not found in scene");

            switch (m_Args.Gate)
            {
                case GateKind.G1: m_Sequence = StartCoroutine(RunG1()); break;
                case GateKind.G2: m_Sequence = StartCoroutine(RunG2()); break;
                case GateKind.G3: m_Sequence = StartCoroutine(RunG3()); break;
                default:
                    SetInstruction("manual", "Manual mode: use the keys below. Esc quits.", "");
                    StartCoroutine(ProbeWhenConnected());
                    break;
            }
        }

        void Update()
        {
            HandleKeys();
            int reason = m_Pose.LastTrackingReason;
            if (reason != m_LastReason)
            {
                m_Log.Event("tracking_reason", NativeTracking.Name(reason), reason.ToString(CultureInfo.InvariantCulture));
                m_LastReason = reason;
            }
            m_Log.FlushIfDue();
        }

        void OnApplicationQuit()
        {
            WriteRepeatSummary();
            m_Log.Event("application_quit", "", "");
            m_Log.Flush();
        }

        // ---------------------------------------------------------------- sequences

        IEnumerator ProbeWhenConnected()
        {
            yield return WaitUntilOrTimeout(() => m_GlassesConnected, float.PositiveInfinity);
            m_Device.Probe("connect");
        }

        IEnumerator RunG1()
        {
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
                StartCapture(order == "tracked" ? "after_tracking_ready" : order == "3dof" ? "after_3dof" : "after_6dof_switch");

            yield return RunProtocol(GateProtocol.G3());
            StopCapture("end");
            yield return Quit("COMPLETED");
        }

        /// <summary>Waits for the glasses and checks that the native session was created.</summary>
        IEnumerator Connect(Action<string> result)
        {
            SetInstruction("connect", "Connecting to the glasses... (keep them plugged in)", "");
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
            SetInstruction("enter_xr", "Entering XR: put the glasses on now.", "");
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
            SetInstruction("wait_tracking",
                "LOOK AROUND SLOWLY at a well-lit, textured part of the room until TRACKING turns READY.", "");
            m_Log.Event("tracking_ready", "WAIT", $"timeout={TrackingReadyTimeout}s");
            double t0 = Time.realtimeSinceStartupAsDouble;
            double readySince = double.NaN;
            m_StepEnd = t0 + TrackingReadyTimeout;
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
                    result(false);
                    yield break;
                }
                yield return null;
            }
        }

        IEnumerator RunProtocol(List<ProtocolStep> steps)
        {
            m_StepCount = steps.Count;
            m_Log.Event("protocol", "begin", $"steps={steps.Count} seconds={GateProtocol.TotalSeconds(steps).ToString(CultureInfo.InvariantCulture)}");
            for (int i = 0; i < steps.Count; i++)
            {
                var step = steps[i];
                m_StepIndex = i + 1;
                SetInstruction(step.Id, step.Instruction, i + 1 < steps.Count ? steps[i + 1].Instruction : "(end)");
                m_StepEnd = Time.realtimeSinceStartupAsDouble + step.Seconds;
                switch (step.Action)
                {
                    case StepAction.CameraStart: StartCapture(step.Id); break;
                    case StepAction.CameraStop: StopCapture(step.Id); break;
                    case StepAction.Track3DoF: StartCoroutine(SwitchTracking(TrackingType.MODE_3DOF)); break;
                    case StepAction.Track6DoF: StartCoroutine(SwitchTracking(TrackingType.MODE_6DOF)); break;
                }
                m_Log.Event("step", step.Id, $"{step.Seconds.ToString(CultureInfo.InvariantCulture)}s {step.Instruction}");
                m_Audio.PlayOneShot(m_Beep);
                yield return new WaitForSecondsRealtime(step.Seconds);
            }
            m_Log.Event("protocol", "end", "");
            SetInstruction("done", "Protocol complete. Finishing...", "");
        }

        void SetInstruction(string id, string text, string next)
        {
            m_StepId = id;
            m_Instruction = text;
            m_NextInstruction = next;
            m_StepEnd = double.NaN;
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
            SetInstruction("quit", $"Finished: {result}. Closing...", "");
            m_Log.Event("gate_result", result, m_Args.GateName);

            if (m_Source.IsCapturing)
                StopCapture("quit");
            if (XRActive)
            {
                ExitXR();
                bool stopped = false;
                yield return WaitUntilOrTimeout(() => !XRActive, ExitXRTimeout, r => stopped = r);
                m_Log.Event("exit_xr_wait", stopped ? "STOPPED" : "TIMEOUT", "");
            }
            WriteRepeatSummary();
            m_Log.Event("quit", "", "");
            m_Log.Flush();
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

        // ---------------------------------------------------------------- HUD and sound

        static AudioClip Tone(string name, float hz, float seconds)
        {
            const int rate = 44100;
            int n = (int)(rate * seconds);
            var data = new float[n];
            for (int i = 0; i < n; i++)
            {
                float env = Mathf.Min(1f, Mathf.Min(i, n - i) / (0.01f * rate)); // 10 ms fade in/out
                data[i] = 0.4f * env * Mathf.Sin(2f * Mathf.PI * hz * i / rate);
            }
            var clip = AudioClip.Create(name, n, 1, rate, false);
            clip.SetData(data, 0);
            return clip;
        }

        void OnGUI()
        {
            int fontSize = Mathf.Max(16, Screen.height / 42);
            if (m_HudStyle == null || m_HudStyle.fontSize != fontSize)
            {
                m_HudStyle = new GUIStyle(GUI.skin.box)
                {
                    alignment = TextAnchor.UpperLeft,
                    fontSize = fontSize,
                    richText = true,
                    wordWrap = true,
                };
                m_HudStyle.normal.textColor = Color.white;
            }

            string text = HudText();
            // In XR the main window spans both eyes side by side (3840x1080): draw once per eye.
            bool sideBySide = XRActive && Screen.width >= 2 * Screen.height;
            float width = (sideBySide ? Screen.width / 2f : Screen.width) - 40f;
            width = Mathf.Min(width, fontSize * 44f);
            float height = m_HudStyle.CalcHeight(new GUIContent(text), width);
            GUI.Box(new Rect(20, 20, width, height), text, m_HudStyle);
            if (sideBySide)
                GUI.Box(new Rect(Screen.width / 2f + 20, 20, width, height), text, m_HudStyle);
        }

        string HudText()
        {
            var inv = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            double now = Time.realtimeSinceStartupAsDouble;
            sb.Append($"<b>{GateTitle()}</b>   t = {(now - m_StartTime).ToString("F0", inv)} s\n");

            string countdown = double.IsNaN(m_StepEnd) ? "" : $"  ({Math.Max(0, m_StepEnd - now).ToString("F0", inv)} s)";
            string stepNo = m_StepCount > 0 && m_StepIndex > 0 ? $"[{m_StepIndex}/{m_StepCount}] " : "";
            sb.Append($"<size={(int)(m_HudStyle.fontSize * 1.35f)}><color=#ffd84a>{stepNo}{m_Instruction}{countdown}</color></size>\n");
            if (!string.IsNullOrEmpty(m_NextInstruction))
                sb.Append($"<color=#aaaaaa>next: {m_NextInstruction}</color>\n");

            int reason = m_Pose.LastTrackingReason;
            string tracking = reason == NativeTracking.NoValue
                ? "<color=#aaaaaa>TRACKING: n/a (XR off)</color>"
                : reason == NativeTracking.Ready
                    ? $"<color=#4cff4c><b>TRACKING: READY</b></color> ({m_Pose.LastTrackingType})"
                    : $"<color=#ff5c5c><b>TRACKING: {NativeTracking.Name(reason).ToUpperInvariant()}</b></color> ({m_Pose.LastTrackingType})";
            sb.Append(tracking + "\n");

            double lag = m_Camera.LagGrowth;
            sb.Append($"camera {(m_Source.IsCapturing ? "ON" : "off")} [{m_Source.SourceName}]  {m_Camera.Fps.ToString("F1", inv)} fps  " +
                      $"lag {(double.IsNaN(lag) ? "-" : (lag >= 0 ? "+" : "") + lag.ToString("F2", inv) + " s")}  " +
                      $"frames {m_Camera.UniqueFrames.ToString(inv)}  drained {m_Camera.DrainedFrames.ToString(inv)}  " +
                      $"cpu {m_Perf.CpuCoresBusy.ToString("F1", inv)} cores\n");

            Vector3 p = m_Pose.LastPosition * 1000f;
            Vector3 e = m_Pose.LastRotation.eulerAngles;
            sb.Append($"head mm x {p.x.ToString("F0", inv)} y {p.y.ToString("F0", inv)} z {p.z.ToString("F0", inv)}   " +
                      $"yaw {Wrap(e.y).ToString("F0", inv)} pitch {Wrap(e.x).ToString("F0", inv)} roll {Wrap(e.z).ToString("F0", inv)}\n");
            sb.Append($"<color=#aaaaaa>glasses {(m_GlassesConnected ? "connected" : "-")}  XR {(XRActive ? "on" : "off")}  marks {m_Marks}  " +
                      $"last: {m_Log.LastEvent}</color>\n");
            sb.Append("<color=#aaaaaa>Do not click anything. Keys: Esc abort  M mark  (manual: C camera  X XR  3/6 DoF  R recenter)</color>");
            return sb.ToString();
        }

        string GateTitle()
        {
            switch (m_Args.Gate)
            {
                case GateKind.G1: return "G1  camera only";
                case GateKind.G2: return "G2  6DoF";
                case GateKind.G3:
                    switch (m_Args.Order)
                    {
                        case "tracked": return $"G3c  camera after TRACKING READY ({m_Source.SourceName} reader)";
                        case "3dof": return $"G3  camera + 3DoF ({m_Source.SourceName} reader)";
                        case "camfirst": return $"G3b  camera before XR ({m_Source.SourceName} reader)";
                        default: return $"G3  camera right after 6DoF switch ({m_Source.SourceName} reader)";
                    }
                default: return "Manual";
            }
        }

        static float Wrap(float deg) => deg > 180f ? deg - 360f : deg;
    }
}
