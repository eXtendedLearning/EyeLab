using System;
using System.Collections;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

namespace EyeLab.Probes
{
    /// <summary>Probe parameters; from intent extras, else <c>probe.json</c> in the app's files folder, else the buttons.</summary>
    [Serializable]
    public sealed class ProbeConfig
    {
        public string probe = "";          // inventory | chain | timecode | all
        public float seconds = 0f;         // 0 = the probe's default
        public float imu_seconds = 20f;
        public int save_every = 0;         // chain: 0; timecode: 1
        public int scale = 2;
        public float snapshot_every = 10f;
        public bool quit_when_done = false;
    }

    /// <summary>
    /// Entry point of the EyeLab Beam Pro probes (T6.0 gate BS, T6.1 gate B1). Put it in the Probes scene
    /// (<c>EyeLab/Probes/Create Probes Scene</c>). Configuration, in order of precedence:
    /// <list type="number">
    /// <item>intent extras: <c>adb shell am start -n it.univpm.eyelab/com.unity3d.player.UnityPlayerActivity --es probe timecode</c>;</item>
    /// <item><c>probe.json</c> pushed to <c>/sdcard/Android/data/it.univpm.eyelab/files/</c> (works when MyGlasses launches the app);</item>
    /// <item>otherwise buttons on the Beam Pro screen.</item>
    /// </list>
    /// Runs land in <c>files/BeamProRuns/</c>; pull them with <c>adb pull</c>.
    /// </summary>
    public sealed class ProbeRunner : MonoBehaviour
    {
        [SerializeField] float m_StartDelay = 3f;

        Text m_Status;
        GameObject m_Panel;
        bool m_Busy;

        IEnumerator Start()
        {
            Application.targetFrameRate = 120;   // the XR display paces the loop; this only lifts Unity's cap
            Debug.Log("[EyeLab probe] runner started; camera permission " + (HasCamera() ? "granted" : "not granted"));
            // No permission request here: on the Beam Pro the system dialog opens on the phone screen and the
            // first build (2026-11-17) ran at about 1 frame/s behind it, so the buttons never appeared. The camera
            // probes ask when they start (RequestCamera); `adb shell pm grant it.univpm.eyelab
            // android.permission.CAMERA` avoids the dialog altogether.
            yield return new WaitForSecondsRealtime(m_StartDelay);   // let the XREAL session and controller come up
            BuildUi();
            ProbeConfig cfg = FromIntent() ?? FromFile();
            if (cfg != null && !string.IsNullOrEmpty(cfg.probe))
            {
                SetStatus($"Config: {cfg.probe}");
                yield return RunProbe(cfg);
                if (cfg.quit_when_done) Application.Quit();
            }
            else
            {
                SetStatus("Choose a probe. Runs go to files/BeamProRuns.");
            }
        }

        public IEnumerator RunProbe(ProbeConfig cfg)
        {
            if (m_Busy) yield break;
            m_Busy = true;
            if (m_Panel != null) m_Panel.SetActive(false);
            string p = cfg.probe.ToLowerInvariant();
            if (p == "inventory" || p == "all") yield return Inventory(cfg);
            if (p == "chain" || p == "all") yield return Chain(cfg, timecode: false);
            if (p == "timecode" || p == "all") yield return Chain(cfg, timecode: true);
            if (m_Panel != null) m_Panel.SetActive(true);
            m_Busy = false;
        }

        IEnumerator Inventory(ProbeConfig cfg)
        {
            using (var run = new ProbeRun("inventory"))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                yield return BeamProInventoryProbe.Run(run, cfg.imu_seconds, SetStatus);
                SetStatus("Inventory done: " + Path.GetFileName(run.Dir));
            }
        }

        static bool HasCamera()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return Permission.HasUserAuthorizedPermission(Permission.Camera);
#else
            return true;
#endif
        }

        /// <summary>Ask for CAMERA once, wait for the answer (the dialog is on the Beam Pro screen), max 60 s.</summary>
        IEnumerator RequestCamera()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (HasCamera()) yield break;
            bool answered = false;
            var cb = new PermissionCallbacks();
            cb.PermissionGranted += _ => answered = true;
            cb.PermissionDenied += _ => answered = true;
            SetStatus("Allow the camera on the Beam Pro screen");
            Permission.RequestUserPermission(Permission.Camera, cb);
            float t0 = Time.realtimeSinceStartup;
            while (!answered && Time.realtimeSinceStartup - t0 < 60f) yield return null;
#endif
            yield break;
        }

        IEnumerator Chain(ProbeConfig cfg, bool timecode)
        {
            yield return RequestCamera();
            Debug.Log("[EyeLab probe] camera permission " + (HasCamera() ? "granted" : "NOT granted, the Eye may still work through the XREAL service"));
            float seconds = cfg.seconds > 0 ? cfg.seconds : timecode ? 60f : 300f;
            int saveEvery = timecode ? Mathf.Max(1, cfg.save_every) : cfg.save_every;
            using (var run = new ProbeRun(timecode ? "timecode" : "chain"))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                TimecodeScreen tc = timecode ? TimecodeScreen.Begin(run) : null;
                bool done = false;
                EyeChainProbe probe = EyeChainProbe.Begin(run, seconds, saveEvery, cfg.scale, cfg.snapshot_every, () => done = true);
                float t0 = Time.realtimeSinceStartup;
                while (!done)
                {
                    SetStatus($"{(timecode ? "Timecode: Eye on this screen, 30-40 cm" : "Eye chain: look around normally")} | " +
                              $"{seconds - (Time.realtimeSinceStartup - t0):F0} s | {probe.Status}");
                    yield return new WaitForSecondsRealtime(0.5f);
                }
                if (tc != null) tc.End();
                SetStatus((timecode ? "Timecode" : "Chain") + " done: " + Path.GetFileName(run.Dir));
            }
        }

        // ------------------------------------------------------------------ configuration
        static ProbeConfig FromIntent()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using (var up = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (AndroidJavaObject activity = up.GetStatic<AndroidJavaObject>("currentActivity"))
                using (AndroidJavaObject intent = activity.Call<AndroidJavaObject>("getIntent"))
                {
                    string probe = intent.Call<string>("getStringExtra", "probe");
                    if (string.IsNullOrEmpty(probe)) return null;
                    var cfg = new ProbeConfig { probe = probe };
                    cfg.seconds = Float(intent.Call<string>("getStringExtra", "seconds"), cfg.seconds);
                    cfg.imu_seconds = Float(intent.Call<string>("getStringExtra", "imu_seconds"), cfg.imu_seconds);
                    cfg.save_every = (int)Float(intent.Call<string>("getStringExtra", "save_every"), cfg.save_every);
                    cfg.scale = (int)Float(intent.Call<string>("getStringExtra", "scale"), cfg.scale);
                    cfg.quit_when_done = intent.Call<string>("getStringExtra", "quit") == "1";
                    return cfg;
                }
            }
            catch (Exception e) { Debug.LogWarning("[EyeLab probe] intent: " + e.Message); }
#endif
            return null;
        }

        static float Float(string s, float fallback) =>
            float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : fallback;

        static ProbeConfig FromFile()
        {
            string path = Path.Combine(Application.persistentDataPath, "probe.json");
            if (!File.Exists(path)) return null;
            try { return JsonUtility.FromJson<ProbeConfig>(File.ReadAllText(path)); }
            catch (Exception e) { Debug.LogWarning("[EyeLab probe] probe.json: " + e.Message); return null; }
        }

        // ------------------------------------------------------------------ phone-screen UI
        void BuildUi()
        {
            Canvas canvas = null;
            if (Unity.XR.XREAL.XREALVirtualController.Singleton != null)
                canvas = Unity.XR.XREAL.XREALVirtualController.Singleton.GetComponentInChildren<Canvas>(true);
            if (canvas == null)
            {
                var cgo = new GameObject("EyeLab Probe Canvas");
                canvas = cgo.AddComponent<Canvas>();
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                canvas.sortingOrder = 31000;
                cgo.AddComponent<GraphicRaycaster>();
            }
            EnsureEventSystem();
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            m_Status = MakeText(canvas.transform, font, new Vector2(0f, 0.9f), new Vector2(1f, 1f), 28);
            m_Panel = new GameObject("EyeLab Probe Buttons", typeof(RectTransform), typeof(VerticalLayoutGroup));
            m_Panel.transform.SetParent(canvas.transform, false);
            var prt = (RectTransform)m_Panel.transform;
            prt.anchorMin = new Vector2(0.02f, 0.35f);
            prt.anchorMax = new Vector2(0.35f, 0.88f);
            prt.offsetMin = prt.offsetMax = Vector2.zero;
            var layout = m_Panel.GetComponent<VerticalLayoutGroup>();
            layout.spacing = 12;
            layout.childForceExpandHeight = true;
            layout.childControlHeight = true;
            layout.childControlWidth = true;

            AddButton(font, "Inventory + IMU 20 s", new ProbeConfig { probe = "inventory" });
            AddButton(font, "Eye chain 5 min", new ProbeConfig { probe = "chain" });
            AddButton(font, "Timecode 60 s", new ProbeConfig { probe = "timecode" });
            AddAction(font, "Quit EyeLab", () => { Debug.Log("[EyeLab probe] quit button"); Application.Quit(); });
            Debug.Log("[EyeLab probe] buttons built on " + (canvas.gameObject.name));
        }

        /// <summary>
        /// uGUI buttons only receive taps through an EventSystem. The SDK's virtual controller prefab has none (its
        /// samples get one from the XR Interaction Setup prefab), and the Probes scene is a copy of SampleScene, so on
        /// the Beam Pro (2026-11-17) the buttons showed but did not react. The project uses the Input System only
        /// (activeInputHandler 1), so the module is InputSystemUIInputModule, which reads the touchscreen.
        /// </summary>
        static void EnsureEventSystem()
        {
            if (UnityEngine.EventSystems.EventSystem.current != null)
            {
                Debug.Log("[EyeLab probe] EventSystem present: " + UnityEngine.EventSystems.EventSystem.current.name);
                return;
            }
            var go = new GameObject("EyeLab EventSystem", typeof(UnityEngine.EventSystems.EventSystem));
#if ENABLE_INPUT_SYSTEM
            go.AddComponent<UnityEngine.InputSystem.UI.InputSystemUIInputModule>();
#else
            go.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();
#endif
            DontDestroyOnLoad(go);
            Debug.Log("[EyeLab probe] EventSystem created");
        }

        void AddAction(Font font, string label, UnityEngine.Events.UnityAction action)
        {
            var go = new GameObject(label, typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(m_Panel.transform, false);
            go.GetComponent<Image>().color = new Color(0.45f, 0.1f, 0.1f, 0.9f);
            go.GetComponent<Button>().onClick.AddListener(action);
            Text t = MakeText(go.transform, font, Vector2.zero, Vector2.one, 30);
            t.text = label;
            t.alignment = TextAnchor.MiddleCenter;
        }

        void AddButton(Font font, string label, ProbeConfig cfg)
        {
            var go = new GameObject(label, typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(m_Panel.transform, false);
            go.GetComponent<Image>().color = new Color(0.15f, 0.15f, 0.15f, 0.9f);
            go.GetComponent<Button>().onClick.AddListener(() => StartCoroutine(RunProbe(cfg)));
            Text t = MakeText(go.transform, font, Vector2.zero, Vector2.one, 30);
            t.text = label;
            t.alignment = TextAnchor.MiddleCenter;
        }

        static Text MakeText(Transform parent, Font font, Vector2 min, Vector2 max, int size)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = min;
            rt.anchorMax = max;
            rt.offsetMin = rt.offsetMax = Vector2.zero;
            Text t = go.GetComponent<Text>();
            t.font = font;
            t.fontSize = size;
            t.color = Color.white;
            t.raycastTarget = false;
            return t;
        }

        void SetStatus(string s)
        {
            if (m_Status != null) m_Status.text = s;
            Debug.Log("[EyeLab probe] " + s);
        }
    }
}
