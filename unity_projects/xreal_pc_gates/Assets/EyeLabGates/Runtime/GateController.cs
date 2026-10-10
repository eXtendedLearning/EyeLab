using System;
using System.Collections;
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
    /// Gate state machine (G1/G2/G3/manual), keyboard control, HUD and lifecycle event log.
    /// XR is entered and exited only through the scene's <see cref="HelloMR.OnClickSwitchXRButton"/>,
    /// i.e. the vendor flow. In the Editor, XR is refused outright: entering XR on the desktop changes
    /// its EDID and monitor layout (docs/XREAL-PC-GATES-G1-G3.md §1).
    /// </summary>
    [RequireComponent(typeof(CameraProbe), typeof(PoseProbe), typeof(DeviceInfoProbe))]
    public sealed class GateController : MonoBehaviour
    {
        const float GlassesTimeout = 30f;
        const float EnterXRTimeout = 20f;
        const float ExitXRTimeout = 10f;

        [SerializeField] HelloMR m_HelloMR;
        [SerializeField] XREALRGBCameraTexture m_CameraTexture;

        GateArgs m_Args;
        GateRunLog m_Log;
        CameraProbe m_Camera;
        PoseProbe m_Pose;
        DeviceInfoProbe m_Device;

        bool m_GlassesConnected;
        bool m_SubsystemsStarted;
        int m_Marks;
        int m_CaptureSessions;
        double m_StartTime;
        bool m_Quitting;
        Coroutine m_Sequence;
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
            if (m_CameraTexture == null) m_CameraTexture = FindAnyObjectByType<XREALRGBCameraTexture>();
            if (m_HelloMR == null) m_HelloMR = FindAnyObjectByType<HelloMR>();
            if (m_Camera.Texture == null) m_Camera.Texture = m_CameraTexture;

            m_Camera.Attach(m_Log, m_Args.SnapshotEvery);
            m_Pose.Attach(m_Log);
            m_Device.Attach(m_Log);
            m_StartTime = Time.realtimeSinceStartupAsDouble;
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
                $"order={m_Args.Order} duration={m_Args.Duration.ToString(CultureInfo.InvariantCulture)} label={m_Args.Label} dir={m_Log.Directory}");
            if (m_HelloMR == null) m_Log.Event("setup_warning", "", "HelloMR not found in scene");
            if (m_CameraTexture == null) m_Log.Event("setup_warning", "", "XREALRGBCameraTexture not found in scene");

            if (XREALSessionManager.Singleton != null && XREALSessionManager.Singleton.GlassConnected)
                m_Log.Event("glasses_state_at_start", "device_seen", XREALSessionManager.Singleton.ConnectedDevice.ToString());

            switch (m_Args.Gate)
            {
                case GateKind.G1: m_Sequence = StartCoroutine(RunG1()); break;
                case GateKind.G2: m_Sequence = StartCoroutine(RunXRGate(withCamera: false)); break;
                case GateKind.G3: m_Sequence = StartCoroutine(RunXRGate(withCamera: true)); break;
                default: StartCoroutine(ProbeWhenConnected()); break;
            }
        }

        void Update()
        {
            HandleKeys();
            m_Log.FlushIfDue();
        }

        void OnApplicationQuit()
        {
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
            bool connected = false;
            yield return WaitForGlasses(r => connected = r);
            if (!connected) { yield return Quit("NO_GLASSES"); yield break; }
            m_Device.Probe("connect");

            StartCapture("main");
            yield return Step("run", m_Args.Duration);
            for (int i = 1; i <= 3; i++)
            {
                StopCapture($"restart_{i}");
                yield return Step($"restart_{i}_off", 2);
                StartCapture($"restart_{i}");
                yield return Step($"restart_{i}_on", 10);
            }
            StopCapture("final");
            yield return Quit("COMPLETED");
        }

        IEnumerator RunXRGate(bool withCamera)
        {
            bool connected = false;
            yield return WaitForGlasses(r => connected = r);
            if (!connected) { yield return Quit("NO_GLASSES"); yield break; }
            m_Device.Probe("connect");

            if (withCamera && m_Args.CameraFirst)
                StartCapture("before_xr");

            if (!EnterXR())
            {
                yield return Quit("XR_REFUSED");
                yield break;
            }
            bool started = false;
            yield return WaitUntilOrTimeout(() => m_SubsystemsStarted, EnterXRTimeout, r => started = r);
            m_Log.Event("enter_xr_wait", started ? "STARTED" : "TIMEOUT", $"timeout={EnterXRTimeout}s");
            if (!started)
            {
                yield return Quit("XR_TIMEOUT");
                yield break;
            }
            m_Device.Probe("xr");

            yield return SwitchTracking(TrackingType.MODE_6DOF);

            if (withCamera && !m_Args.CameraFirst)
                StartCapture("after_6dof");

            yield return Step("run", m_Args.Duration);

            if (withCamera)
                StopCapture("before_exit_xr");
            yield return Quit("COMPLETED");
        }

        IEnumerator WaitForGlasses(Action<bool> result)
        {
            m_Log.Event("wait_glasses", "begin", $"timeout={GlassesTimeout}s");
            bool ok = false;
            yield return WaitUntilOrTimeout(() => m_GlassesConnected, GlassesTimeout, r => ok = r);
            m_Log.Event("wait_glasses", ok ? "CONNECTED" : "NO_GLASSES", "");
            result(ok);
        }

        IEnumerator Step(string name, double seconds)
        {
            m_Log.Event("step", name, $"seconds={seconds.ToString(CultureInfo.InvariantCulture)}");
            yield return new WaitForSecondsRealtime((float)seconds);
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
            if (m_CameraTexture == null)
            {
                m_Log.Event("start_capture", "NO_TEXTURE", why);
                return false;
            }
            m_Camera.NotifyCaptureStarted();
            bool ok;
            try { ok = m_CameraTexture.StartCapture(); }
            catch (Exception e)
            {
                m_Log.Event("start_capture", "EXCEPTION", $"{why}: {e.GetType().Name}: {e.Message}");
                return false;
            }
            m_CaptureSessions++;
            m_Log.Event("start_capture", ok, $"{why} session={m_CaptureSessions}");
            return ok;
        }

        void StopCapture(string why)
        {
            if (m_CameraTexture == null) return;
            bool ok;
            try { ok = m_CameraTexture.StopCapture(); }
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
            if (m_HelloMR == null)
            {
                m_Log.Event("enter_xr", "NO_HELLOMR", "");
                return false;
            }
            if (!m_GlassesConnected)
            {
                m_Log.Event("enter_xr", "NO_GLASSES", "");
                return false;
            }
            if (XRActive)
            {
                m_Log.Event("enter_xr", "ALREADY_ACTIVE", "");
                return true;
            }
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
            m_Log.Event("gate_result", result, m_Args.GateName);

            if (m_CameraTexture != null && m_CameraTexture.IsCapturing)
                StopCapture("quit");
            if (XRActive)
            {
                ExitXR();
                bool stopped = false;
                yield return WaitUntilOrTimeout(() => !XRActive, ExitXRTimeout, r => stopped = r);
                m_Log.Event("exit_xr_wait", stopped ? "STOPPED" : "TIMEOUT", "");
            }
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
                m_Log.Event("mark", m_Marks, "");
            }
            if (Input.GetKeyDown(KeyCode.C))
            {
                m_Log.Event("key", "C", "");
                if (m_CameraTexture != null && m_CameraTexture.IsCapturing) StopCapture("key");
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
                m_Log.Event("key", "Esc", "");
                if (m_Sequence != null) StopCoroutine(m_Sequence);
                StartCoroutine(Quit("ABORTED_BY_USER"));
            }
        }

        // ---------------------------------------------------------------- lifecycle events

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

        void OnXrealError(XREALErrorCode code, string message) =>
            m_Log.Event("xreal_error", code.ToString(), message);

        void OnTrackingTypeChanged(bool result, TrackingType target) =>
            m_Log.Event("tracking_type_changed", result, target.ToString());

        void OnUnityLog(string condition, string stackTrace, LogType type)
        {
            if (condition.StartsWith("[EyeLabGates]", StringComparison.Ordinal)) return;
            bool interesting = type == LogType.Error || type == LogType.Exception || type == LogType.Assert
                               || condition.IndexOf("XREAL", StringComparison.OrdinalIgnoreCase) >= 0
                               || condition.StartsWith("[RGBCamera]", StringComparison.Ordinal);
            if (!interesting) return;
            string detail = condition;
            if (type == LogType.Exception && !string.IsNullOrEmpty(stackTrace))
                detail += " @ " + stackTrace.Split('\n')[0];
            m_Log.Event("unity_log", type.ToString(), detail.Length > 500 ? detail.Substring(0, 500) : detail);
        }

        // ---------------------------------------------------------------- HUD

        void OnGUI()
        {
            if (m_HudStyle == null)
            {
                m_HudStyle = new GUIStyle(GUI.skin.box)
                {
                    alignment = TextAnchor.UpperLeft,
                    fontSize = 16,
                    richText = false,
                    wordWrap = false,
                };
                m_HudStyle.normal.textColor = Color.white;
            }

            var inv = CultureInfo.InvariantCulture;
            double elapsed = Time.realtimeSinceStartupAsDouble - m_StartTime;
            Vector3 p = m_Pose.LastPosition * 1000f;
            Vector3 e = m_Pose.LastRotation.eulerAngles;
            string text =
                $"EyeLab XREAL PC gates   gate {m_Args.GateName} ({m_Args.Label})   t = {elapsed.ToString("F1", inv)} s\n" +
                $"glasses {(m_GlassesConnected ? "connected" : "not connected")}   XR {(XRActive ? "active" : "off")}   marks {m_Marks}\n" +
                $"camera {(m_CameraTexture != null && m_CameraTexture.IsCapturing ? "capturing" : "stopped")}   " +
                $"fps {m_Camera.Fps.ToString("F1", inv)}   cam_ts {m_Camera.LastTimestamp.ToString(inv)}   " +
                $"res {m_Camera.LastResolution.x}x{m_Camera.LastResolution.y}   frames {m_Camera.UniqueFrames.ToString(inv)}\n" +
                $"tracking {m_Pose.LastTrackingType}   head valid {m_Pose.LastValid}   state {m_Pose.LastTrackingState}\n" +
                $"head pos mm  x {p.x.ToString("F1", inv)}  y {p.y.ToString("F1", inv)}  z {p.z.ToString("F1", inv)}   " +
                $"yaw {Wrap(e.y).ToString("F1", inv)}  pitch {Wrap(e.x).ToString("F1", inv)}  roll {Wrap(e.z).ToString("F1", inv)} deg\n" +
                $"last event: {m_Log.LastEvent}\n" +
                $"log: {m_Log.Directory}\n" +
                "keys: C camera   X enter/exit XR   3/6 tracking   R recenter   M mark   Esc quit";
            var size = m_HudStyle.CalcSize(new GUIContent(text));
            GUI.Box(new Rect(10, 10, size.x + 20, size.y + 10), text, m_HudStyle);
        }

        static float Wrap(float deg) => deg > 180f ? deg - 360f : deg;
    }
}
