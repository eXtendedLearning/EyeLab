using System;
using System.Collections;
using System.Collections.Generic;
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
        public string probe = "";          // inventory | chain | timecode | head | head_eye | head_eye_3dof | stereo | loop | sync | pan | render | detect | budget | lean | overlay | all
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
        GlassesView m_Glasses;
        GameObject m_Panel, m_Panel2, m_RatePanel, m_OverlayRatePanel;
        string m_GlassesSeen = "not answered";
        CuePlayer m_Cues;
        bool m_Busy;

        // ------------------------------------------------------------------ T6.3 cue scripts (Jack, 2026-11-18)
        static readonly CueStep[] LoopCues =
        {
            new CueStep("ready", "This test needs about one metre of free floor on any one side of you. Stand on your mark and face ahead.", 9f,
                        "Free floor: 1 m on any side. Stand on the mark", ticks: true),
            new CueStep("still_1", "Stay still.", 15f),
            new CueStep("look_left", "Look left.", 4f),
            new CueStep("look_right", "Look right.", 4f),
            new CueStep("look_up", "Look up.", 4f),
            new CueStep("look_down", "Look down.", 4f),
            new CueStep("look_centre", "Back to the centre.", 3f, ticks: true),
            new CueStep("turn_left", "Turn ninety degrees left.", 5f, "Turn 90 degrees LEFT"),
            new CueStep("front_1", "Back to the front.", 5f),
            new CueStep("turn_right", "Turn ninety degrees right.", 5f, "Turn 90 degrees RIGHT"),
            new CueStep("front_2", "Back to the front.", 5f, ticks: true),
            new CueStep("walk_out", "Take two steps towards whichever side is free.", 5f, "2 steps to a free side"),
            new CueStep("away", "Stop there.", 4f),
            new CueStep("walk_back", "Walk back onto your mark and face ahead again.", 6f, "Back onto the mark, face ahead", ticks: true),
            new CueStep("still_2", "Stay still.", 15f),
        };

        static readonly CueStep[] SyncCues =
        {
            new CueStep("ready", "Stand anywhere, facing a wall with texture, or the board, one to two metres away.", 7f, "Face a textured wall, 1-2 m", ticks: true),
            new CueStep("still_1", "Stay still.", 5f, ticks: true),
            new CueStep("shake", "Shake your head left and right, about once a second.", 12f, "Shake LEFT - RIGHT", metronome: 0.5f),
            new CueStep("still_2", "Stop. Stay still.", 3f, ticks: true),
            new CueStep("nod", "Nod up and down, about once a second.", 8f, "Nod UP - DOWN", metronome: 0.5f),
            new CueStep("still_3", "Stay still.", 4f),
        };

        static readonly CueStep[] PanCues =
        {
            new CueStep("ready", "Anywhere. Hold the Beam Pro, rear cameras towards a scene with texture.", 6f, "Beam Pro in hand, rear cameras to a scene", ticks: true),
            new CueStep("still_1", "Hold still.", 4f, ticks: true),
            new CueStep("pan_slow", "Pan slowly left and right.", 10f, "Pan SLOWLY left - right", metronome: 1f),
            new CueStep("pan_fast", "Now quickly, left and right.", 10f, "Pan QUICKLY left - right", metronome: 0.5f),
            new CueStep("still_2", "Hold still.", 3f),
        };

        IEnumerator Start()
        {
            Application.targetFrameRate = 120;   // the XR display paces the loop; this only lifts Unity's cap
            // With the Beam Pro screen asleep the activity pauses (seen 2026-11-17); a 5-min chain run is worn
            // with the glasses and nobody touches the screen. Keep it on while EyeLab is in front.
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            Debug.Log("[EyeLab probe] render default " + RenderProbe.ApplyDefaults());
            Debug.Log("[EyeLab probe] runner started; camera permission " + (HasCamera() ? "granted" : "not granted"));
            // No permission request here: on the Beam Pro the system dialog opens on the phone screen and the
            // first build (2026-11-17) ran at about 1 frame/s behind it, so the buttons never appeared. The camera
            // probes ask when they start (RequestCamera); `adb shell pm grant it.univpm.eyelab
            // android.permission.CAMERA` avoids the dialog altogether.
            m_Cues = CuePlayer.Ensure();   // the speech engine initialises while the XREAL session comes up
            yield return new WaitForSecondsRealtime(m_StartDelay);   // let the XREAL session and controller come up
            BuildUi();
            Debug.Log("[EyeLab probe] speech " + m_Cues.SpeechState);
            StartCoroutine(GlassesCheck());
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

        /// <summary>T6.4: ask once at start what the glasses show; the answer goes into the app log and every run.</summary>
        IEnumerator GlassesCheck()
        {
            float t0 = Time.realtimeSinceStartup;
            while (m_Cues.SpeechState.StartsWith("init") && Time.realtimeSinceStartup - t0 < 8f) yield return null;
            yield return new WaitForSecondsRealtime(1f);
            m_Cues.Say("EyeLab is ready. In the glasses you should see green text, and a blue square ahead. " +
                       "Tap what you see, at the top of the Beam Pro screen.");
        }

        void AnswerGlasses(string answer)
        {
            m_GlassesSeen = $"{answer} (t {Time.realtimeSinceStartup:F0} s)";
            Debug.Log("[EyeLab probe] glasses check: " + m_GlassesSeen);
            m_Cues?.Say(answer == "all" ? "Thank you." : "Noted.");
            SetStatus("Glasses: " + answer + ". Choose a test.");
        }

        void ShowPage(GameObject page)
        {
            if (m_Panel != null) m_Panel.SetActive(page == m_Panel);
            if (m_Panel2 != null) m_Panel2.SetActive(page == m_Panel2);
            if (m_RatePanel != null) m_RatePanel.SetActive(page == m_RatePanel);
            if (m_OverlayRatePanel != null) m_OverlayRatePanel.SetActive(page == m_OverlayRatePanel);
        }

        public IEnumerator RunProbe(ProbeConfig cfg)
        {
            if (m_Busy) yield break;
            m_Busy = true;
            ShowPage(null);
            string p = cfg.probe.ToLowerInvariant();
            if (p == "inventory" || p == "all") yield return Inventory(cfg);
            if (p == "chain" || p == "all") yield return Chain(cfg, timecode: false);
            if (p == "timecode" || p == "all") yield return Chain(cfg, timecode: true);
            if (p == "head") yield return Head(cfg, Unity.XR.XREAL.TrackingType.MODE_6DOF, false);
            if (p == "head_eye") yield return Head(cfg, Unity.XR.XREAL.TrackingType.MODE_6DOF, true);
            if (p == "head_eye_3dof") yield return Head(cfg, Unity.XR.XREAL.TrackingType.MODE_3DOF, true);
            if (p == "stereo") yield return Stereo(cfg, null);
            if (p == "loop") yield return Guided(cfg, "loop", LoopCues, Unity.XR.XREAL.TrackingType.MODE_6DOF, 0, 2, false);
            if (p == "sync") yield return Guided(cfg, "sync", SyncCues, Unity.XR.XREAL.TrackingType.MODE_6DOF, 1, 4, true);
            if (p == "pan") yield return Stereo(cfg, PanCues);
            if (p == "render") yield return Render(cfg);
            if (p == "detect") yield return Detect(cfg, "detect");
            if (p == "budget") yield return Detect(cfg, "budget");
            if (p == "lean") yield return Detect(cfg, "lean");
            if (p == "overlay") yield return Detect(cfg, "overlay");
            m_Glasses?.ClearCue();
            ShowPage(m_Panel);
            m_Busy = false;
        }

        void ShowCue(string text, float left, float frac, string counter) => m_Glasses?.SetCue(text, frac, counter);

        /// <summary>Glasses start-up times and the speech engine state, into every run (T6.3).</summary>
        void GlassesMeta(ProbeRun run)
        {
            if (m_Glasses != null)
            {
                run.Meta("glasses_first_frame_s", ProbeRun.F(m_Glasses.FirstFrameAt));
                run.Meta("glasses_first_pose_s", ProbeRun.F(m_Glasses.FirstPoseAt));
                run.Meta("glasses_square_placed_s", ProbeRun.F(m_Glasses.PlacedAt));
            }
            run.Meta("speech", m_Cues != null ? m_Cues.SpeechState : "none");
            run.Meta("app_log", AppLog.Path);
            run.Meta("glasses_seen", m_GlassesSeen);
        }

        /// <summary>T6.3 guided head run with the Eye: "loop" (B2 loop closure) or "sync" (B1t motion sync).</summary>
        IEnumerator Guided(ProbeConfig cfg, string name, CueStep[] cues, Unity.XR.XREAL.TrackingType mode, int saveEvery, int scale, bool poseGrid)
        {
            yield return RequestCamera();
            using (var run = new ProbeRun(name))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                GlassesMeta(run);
                yield return HeadPoseProbe.Run(run, 0f, mode, true, SetStatus, cues, ShowCue, saveEvery, scale, poseGrid);
                SetStatus(name + " done: " + Path.GetFileName(run.Dir));
            }
        }

        /// <summary>T6.5 render gate: rendering settings switched live, rated on the screen (no Eye, no camera).</summary>
        IEnumerator Render(ProbeConfig cfg)
        {
            using (var run = new ProbeRun("render"))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                GlassesMeta(run);
                ShowPage(m_RatePanel);
                yield return RenderProbe.Run(run, m_Glasses, SetStatus, ShowCue);
                SetStatus("render done: " + Path.GetFileName(run.Dir));
            }
        }

        /// <summary>
        /// ArUco detection and board pose with the native plugin, bench on pushed frames then live on the Eye:
        /// "detect" (T6.6, gate B3a), "budget" (T6.7, modes and targets) or "lean" (T6.8, region only).
        /// </summary>
        IEnumerator Detect(ProbeConfig cfg, string variant)
        {
            yield return RequestCamera();
            using (var run = new ProbeRun(variant))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                GlassesMeta(run);
                bool overlay = variant == "overlay";
                if (overlay) { ShowPage(m_OverlayRatePanel); m_Glasses?.SetOverlayLayout(true); }
                yield return DetectProbe.Run(run, SetStatus, ShowCue, variant);
                if (overlay) m_Glasses?.SetOverlayLayout(false);
                SetStatus(variant + " done: " + Path.GetFileName(run.Dir));
            }
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
            float seconds = cfg.seconds > 0 ? cfg.seconds : timecode ? 60f : 120f;   // runs ≤ 3 min (Jack, 2026-11-18)
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

        IEnumerator Head(ProbeConfig cfg, Unity.XR.XREAL.TrackingType mode, bool withEye)
        {
            if (withEye) yield return RequestCamera();
            float seconds = cfg.seconds > 0 ? cfg.seconds : mode == Unity.XR.XREAL.TrackingType.MODE_6DOF ? 120f : 60f;
            string name = withEye ? (mode == Unity.XR.XREAL.TrackingType.MODE_6DOF ? "head_eye" : "head_eye_3dof") : "head";
            using (var run = new ProbeRun(name))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                GlassesMeta(run);
                yield return HeadPoseProbe.Run(run, seconds, mode, withEye, SetStatus);
                SetStatus("Head done: " + Path.GetFileName(run.Dir));
            }
        }

        /// <summary>
        /// T6.2 BS-2: open the hidden logical rear camera "3" (physical 0 + 2) for 30 s, Beam Pro on the table.
        /// T6.3 with <paramref name="cues"/> ("pan"): guided pans with the Beam Pro in hand; every image of both
        /// sensors saved at 160x90 and the Beam Pro IMU recorded, to measure the exposure offset between the sensors.
        /// </summary>
        IEnumerator Stereo(ProbeConfig cfg, CueStep[] cues)
        {
            yield return RequestCamera();
            float seconds = cfg.seconds > 0 ? cfg.seconds : 30f;
            using (var run = new ProbeRun(cues != null ? "pan" : "stereo"))
            {
                run.Meta("config", JsonUtility.ToJson(cfg));
                GlassesMeta(run);
#if UNITY_ANDROID && !UNITY_EDITOR
                AndroidJavaClass helper = null;
                AndroidJavaObject activity = null;
                try
                {
                    using (var up = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                        activity = up.GetStatic<AndroidJavaObject>("currentActivity");
                    helper = new AndroidJavaClass("it.univpm.eyelab.probes.StereoProbe");
                    string started = helper.CallStatic<string>("start", activity, run.Dir, "3", cues != null ? 4 : 0);
                    run.Meta("stereo_start", started);
                    Debug.Log("[EyeLab probe] stereo start " + started);
                }
                catch (Exception e) { run.Meta("stereo_start", "error " + e.Message); helper = null; }
                AndroidJavaClass imu = null;
                if (helper != null && cues != null)
                {
                    try
                    {
                        imu = new AndroidJavaClass("it.univpm.eyelab.probes.BeamProInventory");
                        run.Meta("imu_start", imu.CallStatic<string>("imuStart", activity, Path.Combine(run.Dir, "imu.csv"), 0));
                    }
                    catch (Exception e) { run.Meta("imu_error", e.Message); imu = null; }
                }
                if (helper != null)
                {
                    if (cues != null)
                    {
                        float nextStatus = 0f;
                        yield return m_Cues.Play(run, cues, (text, left, frac, counter) =>
                        {
                            ShowCue(text, left, frac, counter);
                            if (Time.realtimeSinceStartup < nextStatus) return;
                            nextStatus = Time.realtimeSinceStartup + 0.5f;
                            SetStatus($"{counter}: {text} | {left:F0} s");
                        });
                    }
                    else
                    {
                        float t0 = Time.realtimeSinceStartup;
                        while (Time.realtimeSinceStartup - t0 < seconds)
                        {
                            SetStatus($"Stereo: Beam Pro on the table, rear cameras to the scene | {seconds - (Time.realtimeSinceStartup - t0):F0} s");
                            yield return new WaitForSecondsRealtime(0.5f);
                        }
                    }
                    string stopped;
                    try { stopped = helper.CallStatic<string>("stop"); } catch (Exception e) { stopped = "error " + e.Message; }
                    run.Meta("stereo_stop", stopped);
                    Debug.Log("[EyeLab probe] stereo stop " + stopped);
                    helper.Dispose();
                }
                if (imu != null)
                {
                    try { run.Meta("imu_stop", imu.CallStatic<string>("imuStop")); } catch (Exception e) { run.Meta("imu_error", e.Message); }
                    imu.Dispose();
                }
                activity?.Dispose();
#else
                run.Meta("stereo_start", "not on Android");
                yield return null;
#endif
                SetStatus((cues != null ? "Pan" : "Stereo") + " done: " + Path.GetFileName(run.Dir));
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

            // Jack, 2026-11-19: the XREAL controller's App, trigger pad and Home buttons sat under the probe buttons.
            // EyeLab draws an opaque cover over the whole controller canvas (it also takes the taps), then its own
            // single column. Jack, 2026-11-20: the main page holds only the gate being run now; every other run sits on
            // "Other runs" and comes back to the main page only when a session needs it.
            var cover = new GameObject("EyeLab Cover", typeof(RectTransform), typeof(Image));
            cover.transform.SetParent(canvas.transform, false);
            var crt = (RectTransform)cover.transform;
            crt.anchorMin = Vector2.zero;
            crt.anchorMax = Vector2.one;
            crt.offsetMin = crt.offsetMax = Vector2.zero;
            cover.GetComponent<Image>().color = new Color(0.04f, 0.04f, 0.05f, 1f);

            m_Status = MakeText(canvas.transform, font, new Vector2(0.04f, 0.88f), new Vector2(0.96f, 0.98f), 30);
            m_Status.alignment = TextAnchor.MiddleCenter;
            m_Glasses = GlassesView.Create(font);

            m_Panel = Column(canvas.transform);
            AddButton(font, "T6.9  Overlay in the glasses: plate L, board  (2 min)", new ProbeConfig { probe = "overlay" });
            AddAction(font, "Other runs", () => ShowPage(m_Panel2), new Color(0.22f, 0.22f, 0.25f, 0.9f));
            AddAction(font, "Quit EyeLab", () => { Debug.Log("[EyeLab probe] quit button"); Application.Quit(); });

            // Jack, 2026-11-22: the glasses always show the text and the square now, so the glasses check and the
            // square placement left the main page; they come back only when something about the display changes.
            m_Panel2 = Column(canvas.transform);
            var row = Row(m_Panel2.transform);
            AddAction(font, "I see text + square", () => AnswerGlasses("all"), new Color(0.12f, 0.3f, 0.15f, 0.9f), row);
            AddAction(font, "Text only", () => AnswerGlasses("text only"), new Color(0.35f, 0.3f, 0.1f, 0.9f), row);
            AddAction(font, "Nothing", () => AnswerGlasses("nothing"), new Color(0.45f, 0.1f, 0.1f, 0.9f), row);
            AddAction(font, "Place square ahead", () => m_Glasses?.PlaceReference(), new Color(0.1f, 0.25f, 0.45f, 0.9f), m_Panel2);
            AddButton(font, "T6.8  Lean detection: board, plate L tilted  (2 min)", new ProbeConfig { probe = "lean" }, m_Panel2);
            AddButton(font, "T6.7  Detection budget: plate A, plate L, board  (2.5 min)", new ProbeConfig { probe = "budget" }, m_Panel2);
            AddButton(font, "T6.6  Detection, plate A, threads  (100 s)", new ProbeConfig { probe = "detect" }, m_Panel2);
            AddButton(font, "T6.5  Rendering, rate each setting  (85 s)", new ProbeConfig { probe = "render" }, m_Panel2);
            AddButton(font, "T6.3  Head loop + Eye  (95 s)", new ProbeConfig { probe = "loop" }, m_Panel2);
            AddButton(font, "T6.3  Motion sync  (40 s)", new ProbeConfig { probe = "sync" }, m_Panel2);
            AddButton(font, "T6.3  Stereo pan  (35 s)", new ProbeConfig { probe = "pan" }, m_Panel2);
            AddButton(font, "Timecode 60 s", new ProbeConfig { probe = "timecode" }, m_Panel2);
            AddButton(font, "Head 6DoF 2 min", new ProbeConfig { probe = "head" }, m_Panel2);
            AddButton(font, "Head 6DoF + Eye 2 min", new ProbeConfig { probe = "head_eye" }, m_Panel2);
            AddButton(font, "Head 3DoF + Eye 1 min", new ProbeConfig { probe = "head_eye_3dof" }, m_Panel2);
            AddButton(font, "Stereo 30 s (on table)", new ProbeConfig { probe = "stereo" }, m_Panel2);
            AddButton(font, "Eye chain 2 min", new ProbeConfig { probe = "chain" }, m_Panel2);
            AddButton(font, "Inventory + IMU 20 s", new ProbeConfig { probe = "inventory" }, m_Panel2);
            AddAction(font, "Back", () => ShowPage(m_Panel), new Color(0.22f, 0.22f, 0.25f, 0.9f), m_Panel2);

            // T6.5: shown only during the render run; the last tap in each setting's window is its rating.
            m_RatePanel = Column(canvas.transform);
            Color[] rateColors =
            {
                new Color(0.12f, 0.3f, 0.15f, 0.9f), new Color(0.35f, 0.3f, 0.1f, 0.9f),
                new Color(0.45f, 0.1f, 0.1f, 0.9f), new Color(0.22f, 0.22f, 0.25f, 0.9f),
            };
            for (int i = 0; i < RenderProbe.Answers.Length; i++)
            {
                string answer = RenderProbe.Answers[i];
                AddAction(font, char.ToUpperInvariant(answer[0]) + answer.Substring(1), () => RenderProbe.Rate(answer), rateColors[i], m_RatePanel);
            }
            // T6.9: shown only during the overlay run; each tap is logged with the step it falls in.
            m_OverlayRatePanel = Column(canvas.transform);
            Color[] overlayColors =
            {
                new Color(0.12f, 0.3f, 0.15f, 0.9f), new Color(0.25f, 0.32f, 0.12f, 0.9f), new Color(0.35f, 0.3f, 0.1f, 0.9f),
                new Color(0.45f, 0.1f, 0.1f, 0.9f), new Color(0.22f, 0.22f, 0.25f, 0.9f),
            };
            for (int i = 0; i < DetectProbe.OverlayAnswers.Length; i++)
            {
                string answer = DetectProbe.OverlayAnswers[i];
                AddAction(font, char.ToUpperInvariant(answer[0]) + answer.Substring(1), () => DetectProbe.Rate(answer), overlayColors[i], m_OverlayRatePanel);
            }
            ShowPage(m_Panel);
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

        static GameObject Column(Transform parent)
        {
            var panel = new GameObject("EyeLab Probe Buttons", typeof(RectTransform), typeof(VerticalLayoutGroup));
            panel.transform.SetParent(parent, false);
            var prt = (RectTransform)panel.transform;
            prt.anchorMin = new Vector2(0.06f, 0.06f);
            prt.anchorMax = new Vector2(0.94f, 0.86f);
            prt.offsetMin = prt.offsetMax = Vector2.zero;
            var layout = panel.GetComponent<VerticalLayoutGroup>();
            layout.spacing = 12;
            layout.childForceExpandHeight = true;
            layout.childControlHeight = true;
            layout.childControlWidth = true;
            return panel;
        }

        static Transform Row(Transform parent)
        {
            var row = new GameObject("row", typeof(RectTransform), typeof(HorizontalLayoutGroup));
            row.transform.SetParent(parent, false);
            var h = row.GetComponent<HorizontalLayoutGroup>();
            h.spacing = 12;
            h.childForceExpandWidth = true;
            h.childForceExpandHeight = true;
            h.childControlWidth = true;
            h.childControlHeight = true;
            return row.transform;
        }

        void AddAction(Font font, string label, UnityEngine.Events.UnityAction action, Color? color = null, object parent = null)
        {
            var go = new GameObject(label, typeof(RectTransform), typeof(Image), typeof(Button));
            Transform where = parent is GameObject g ? g.transform : parent as Transform ?? m_Panel.transform;
            go.transform.SetParent(where, false);
            go.GetComponent<Image>().color = color ?? new Color(0.45f, 0.1f, 0.1f, 0.9f);
            go.GetComponent<Button>().onClick.AddListener(action);
            Text t = MakeText(go.transform, font, Vector2.zero, Vector2.one, 30);
            t.text = label;
            t.alignment = TextAnchor.MiddleCenter;
        }

        void AddButton(Font font, string label, ProbeConfig cfg, GameObject panel = null)
        {
            var go = new GameObject(label, typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent((panel ?? m_Panel).transform, false);
            go.GetComponent<Image>().color = panel == null ? new Color(0.12f, 0.3f, 0.15f, 0.9f) : new Color(0.15f, 0.15f, 0.15f, 0.9f);
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
            if (m_Glasses != null) m_Glasses.SetText(s);
            Debug.Log("[EyeLab probe] " + s);
        }
    }
}
