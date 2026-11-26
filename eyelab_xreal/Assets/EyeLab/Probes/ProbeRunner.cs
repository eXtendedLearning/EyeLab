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
        public string probe = "";          // inventory | chain | timecode | head | head_eye | head_eye_3dof | stereo | loop | sync | pan | render | detect | budget | lean | overlay | align | spaam | eyecal | eyecheck | camcal | all
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
        // T6.10 align run: one page, its part chosen by the step (rate, align controls, flat or solid)
        GameObject m_AlignPage, m_AlignRate, m_AlignControls, m_AlignStyle;
        Text m_AlignOffset, m_AlignMode, m_AlignStepSize, m_AlignMinus, m_AlignPlus;
        bool m_AlignRun, m_AlignTurn = true, m_AlignFine;
        // T6.11 spaam run: one big tap button, "last tap bad", and the eye-check answers in chk_L
        GameObject m_SpaamPage, m_SpaamTap, m_SpaamCheck, m_SpaamRate;
        Text[] m_ModeLabels;
        Text m_TapLabel;
        Text m_SpaamInfo;
        bool m_SpaamRun;
        // Campaign 0.5 (T7.0): the trial set-up page and the page shown while a trial runs
        GameObject m_CampaignPage, m_MarkPage;
        Text m_CampaignInfo, m_MarkInfo;
        Text[] m_ConditionLabels;
        bool m_MarkRun;
        // 0.5.002 (Jack, 2026-11-26): per-participant calibration pages, the T6 runs page, the square toggle, the warning
        GameObject m_PanelT6, m_CalPage, m_CalResultPage, m_ConfirmPage, m_WarnPage;
        Text m_CalInfo, m_CalResultInfo, m_ConfirmInfo, m_WarnInfo, m_SquareLabel, m_SaveLabel, m_ChoiceLabel;
        int m_SaveAs = -1;   // participant the pcal result is saved as (chosen on the result page)

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
            if (m_AlignPage != null) m_AlignPage.SetActive(page == m_AlignPage);
            if (m_SpaamPage != null) m_SpaamPage.SetActive(page == m_SpaamPage);
            if (m_CampaignPage != null) m_CampaignPage.SetActive(page == m_CampaignPage);
            if (m_MarkPage != null) m_MarkPage.SetActive(page == m_MarkPage);
            foreach (GameObject g in new[] { m_PanelT6, m_CalPage, m_CalResultPage, m_ConfirmPage, m_WarnPage })
                if (g != null) g.SetActive(page == g);
            if (page == m_CampaignPage) RefreshCampaign();
            if (page == m_CalPage) RefreshCal();
            if (page == m_CalResultPage) RefreshCalResult();
        }

        void Update()
        {
            if (m_MarkRun) UpdateMark();
            if (m_SpaamRun && m_SpaamPage != null)
            {
                string s = DetectProbe.ActiveStep;
                bool check = s == "chk_L";
                bool rate = s.StartsWith("l_") || (s.Length > 1 && s[0] == 'c' && char.IsDigit(s[1]));
                if (m_SpaamCheck.activeSelf != check) m_SpaamCheck.SetActive(check);
                if (m_SpaamRate.activeSelf != rate) m_SpaamRate.SetActive(rate);
                if (m_SpaamTap.activeSelf != (!check && !rate)) m_SpaamTap.SetActive(!check && !rate);
                double age = DetectProbe.BoardAgeMs;
                string mode = DetectProbe.ActiveMode;
                string[] names = DetectProbe.ActiveVariant == "pcal" || DetectProbe.ActiveVariant == "pcheck" ? PersonModeNames : ModeNames;
                for (int i = 0; i < m_ModeLabels.Length; i++)
                {
                    m_ModeLabels[i].fontStyle = DetectProbe.EyeModes[i] == mode ? FontStyle.Bold : FontStyle.Normal;
                    string label = names[i] + (DetectProbe.ModeAvailable(DetectProbe.EyeModes[i]) ? "" : " (none)");
                    if (m_ModeLabels[i].text != label) m_ModeLabels[i].text = label;
                }
                string tapText = DetectProbe.ActiveVariant == "camcal" ? "NEXT: tap when set" : "ALIGNED: tap anywhere here";
                if (m_TapLabel.text != tapText) m_TapLabel.text = tapText;
                m_SpaamInfo.text = (age < 0 ? "Board not seen yet" : age < 500 ? "Board seen" : $"Board last seen {age / 1000:F1} s ago") + "   |   " + s +
                                   (mode.Length > 0 ? "   |   drawn with: " + mode : "") +
                                   (DetectProbe.CalText.Length > 0 ? "\n" + DetectProbe.CalText : "");
            }
            if (!m_AlignRun || m_AlignPage == null) return;
            string step = DetectProbe.ActiveStep;
            bool align = step == DetectProbe.AlignStep, style = step == "l_style";
            if (m_AlignControls.activeSelf != align) m_AlignControls.SetActive(align);
            if (m_AlignStyle.activeSelf != style) m_AlignStyle.SetActive(style);
            if (m_AlignRate.activeSelf != (!align && !style)) m_AlignRate.SetActive(!align && !style);
            m_AlignOffset.text = "Offset: " + DetectProbe.OffsetText;
        }

        /// <summary>T6.10: one press of an arrow (dx, dy) or of minus/plus (dz) on the align page.</summary>
        void Nudge(int dx, int dy, int dz, string label)
        {
            if (m_AlignTurn)
            {
                float s = m_AlignFine ? 0.1f : 0.5f;   // degrees; Unity: +yaw moves the drawing right, +pitch down, +roll anticlockwise
                DetectProbe.TurnOffset(Quaternion.Euler(-dy * s, dx * s, dz * s), $"{label} {s} deg");
            }
            else
            {
                float s = m_AlignFine ? 1f : 5f;   // mm, head frame: x right, y up, z forward
                DetectProbe.MoveOffset(new Vector3(dx, dy, dz) * s, $"{label} {s} mm");
            }
        }

        void SetAlignLabels()
        {
            m_AlignMode.text = m_AlignTurn ? "Mode: TURN (tap for MOVE)" : "Mode: MOVE (tap for TURN)";
            m_AlignStepSize.text = m_AlignTurn ? (m_AlignFine ? "Step 0.1 deg (fine)" : "Step 0.5 deg (coarse)")
                                               : (m_AlignFine ? "Step 1 mm (fine)" : "Step 5 mm (coarse)");
            m_AlignMinus.text = m_AlignTurn ? "Roll right" : "Nearer";
            m_AlignPlus.text = m_AlignTurn ? "Roll left" : "Farther";
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
            if (p == "align") yield return Detect(cfg, "align");
            if (p == "spaam") yield return Detect(cfg, "spaam");
            if (p == "eyecal") yield return Detect(cfg, "eyecal");
            if (p == "camcal") yield return Detect(cfg, "camcal");
            if (p == "eyecheck") yield return Detect(cfg, "eyecheck");
            if (p == "mark") yield return Detect(cfg, "mark");
            if (p == "pcal") yield return Detect(cfg, "pcal");
            if (p == "pcheck") yield return Detect(cfg, "pcheck");
            m_Glasses?.ClearCue();
            m_SaveAs = Campaign.Current.participant;
            ShowPage(p == "mark" ? m_CampaignPage : p == "pcal" ? m_CalResultPage : p == "pcheck" ? m_CalPage : m_Panel);
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
                bool spaam = variant == "spaam" || variant == "eyecal" || variant == "eyecheck" || variant == "camcal" || variant == "pcal" || variant == "pcheck";
                bool overlay = variant == "overlay" || variant == "align" || spaam;
                if (variant == "overlay") ShowPage(m_OverlayRatePanel);
                if (variant == "align") { m_AlignRun = true; ShowPage(m_AlignPage); }
                if (spaam) { m_SpaamRun = true; ShowPage(m_SpaamPage); }
                bool mark = variant == "mark";
                if (mark && Campaign.ActiveSheet == null && !Campaign.Prepare(out string why))
                {
                    run.Meta("campaign_error", why);
                    SetStatus("Cannot start: " + why);
                    yield break;
                }
                if (mark) { Campaign.Begin(run); m_MarkRun = true; ShowPage(m_MarkPage); overlay = true; }
                if (overlay) m_Glasses?.SetOverlayLayout(true);
                yield return DetectProbe.Run(run, SetStatus, ShowCue, variant);
                if (mark) { Campaign.End(run); m_MarkRun = false; }
                m_AlignRun = false;
                m_SpaamRun = false;
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

            // 0.5.002 (Jack, 2026-11-26): three buttons; the T6 runs moved to Other runs > T6 runs
            m_Panel = Column(canvas.transform);
            AddAction(font, "CALIBRATION: per participant, on A40", () => ShowPage(m_CalPage), new Color(0.12f, 0.3f, 0.15f, 0.9f));
            AddAction(font, "CAMPAIGN 0.5: hammering trials", () => ShowPage(m_CampaignPage), new Color(0.1f, 0.25f, 0.45f, 0.9f));
            AddAction(font, "Other runs", () => ShowPage(m_Panel2), new Color(0.22f, 0.22f, 0.25f, 0.9f));
            AddAction(font, "Quit EyeLab", () => { Debug.Log("[EyeLab probe] quit button"); Application.Quit(); });

            // Jack, 2026-11-22: the glasses always show the text and the square now, so the glasses check and the
            // square placement left the main page; they come back only when something about the display changes.
            m_Panel2 = Column(canvas.transform);
            var row = Row(m_Panel2.transform);
            AddAction(font, "I see text + square", () => AnswerGlasses("all"), new Color(0.12f, 0.3f, 0.15f, 0.9f), row);
            AddAction(font, "Text only", () => AnswerGlasses("text only"), new Color(0.35f, 0.3f, 0.1f, 0.9f), row);
            AddAction(font, "Nothing", () => AnswerGlasses("nothing"), new Color(0.45f, 0.1f, 0.1f, 0.9f), row);
            Transform sq = Row(m_Panel2.transform);
            AddAction(font, "Place square ahead", () => { m_Glasses?.PlaceReference(); SetSquareLabel(); }, new Color(0.1f, 0.25f, 0.45f, 0.9f), sq);
            m_SquareLabel = AddAction(font, "", () => { m_Glasses?.SetReference(!(m_Glasses?.ReferenceOn ?? false)); SetSquareLabel(); },
                                      new Color(0.1f, 0.25f, 0.45f, 0.9f), sq);
            SetSquareLabel();
            AddAction(font, "T6 runs (Eye calibration, overlay, detection...)", () => ShowPage(m_PanelT6), new Color(0.22f, 0.22f, 0.25f, 0.9f), m_Panel2);
            m_PanelT6 = Column(canvas.transform);
            AddButton(font, "T6.13  Camera calibration + tape check  (4 min)", new ProbeConfig { probe = "camcal" }, m_PanelT6);
            AddButton(font, "T6.12  Eye calibration: taps, fit, check  (6 min)", new ProbeConfig { probe = "eyecal" }, m_PanelT6);
            AddButton(font, "T6.12  Check only, saved calibration  (3 min)", new ProbeConfig { probe = "eyecheck" }, m_PanelT6);
            AddButton(font, "T6.11  Eye calibration taps (SPAAM): board  (4 min)", new ProbeConfig { probe = "spaam" }, m_PanelT6);
            AddButton(font, "T6.10  Align the overlay by eye: board, plate L  (5 min)", new ProbeConfig { probe = "align" }, m_PanelT6);
            AddButton(font, "T6.9  Overlay in the glasses: plate L, board  (2 min)", new ProbeConfig { probe = "overlay" }, m_PanelT6);
            AddButton(font, "T6.8  Lean detection: board, plate L tilted  (2 min)", new ProbeConfig { probe = "lean" }, m_PanelT6);
            AddButton(font, "T6.7  Detection budget: plate A, plate L, board  (2.5 min)", new ProbeConfig { probe = "budget" }, m_PanelT6);
            AddButton(font, "T6.6  Detection, plate A, threads  (100 s)", new ProbeConfig { probe = "detect" }, m_PanelT6);
            AddButton(font, "T6.5  Rendering, rate each setting  (85 s)", new ProbeConfig { probe = "render" }, m_PanelT6);
            AddButton(font, "T6.3  Head loop + Eye  (95 s)", new ProbeConfig { probe = "loop" }, m_PanelT6);
            AddButton(font, "T6.3  Motion sync  (40 s)", new ProbeConfig { probe = "sync" }, m_PanelT6);
            AddButton(font, "T6.3  Stereo pan  (35 s)", new ProbeConfig { probe = "pan" }, m_PanelT6);
            AddAction(font, "Back", () => ShowPage(m_Panel2), new Color(0.22f, 0.22f, 0.25f, 0.9f), m_PanelT6);
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
            BuildAlignPage(canvas.transform, font, overlayColors);
            BuildSpaamPage(canvas.transform, font, overlayColors);
            BuildCampaignPages(canvas.transform, font);
            BuildCalibrationPages(canvas.transform, font);
            ShowPage(m_Panel);
            Debug.Log("[EyeLab probe] buttons built on " + (canvas.gameObject.name));
        }

        /// <summary>
        /// T6.10: the align run's page. Ratings in every step but two; in the align step, the offset controls (TURN about
        /// the head or MOVE, coarse or fine, the four directions, roll or nearer/farther, Zero, Save, Done); in l_style,
        /// flat against solid. Each tap is logged with its step.
        /// </summary>
        void BuildAlignPage(Transform canvas, Font font, Color[] rateColors)
        {
            Color grey = new Color(0.22f, 0.22f, 0.25f, 0.9f), blue = new Color(0.1f, 0.25f, 0.45f, 0.9f);
            m_AlignPage = Column(canvas);
            m_AlignRate = Sub(m_AlignPage.transform);
            for (int i = 0; i < DetectProbe.OverlayAnswers.Length; i++)
            {
                string answer = DetectProbe.OverlayAnswers[i];
                AddAction(font, char.ToUpperInvariant(answer[0]) + answer.Substring(1), () => DetectProbe.Rate(answer), rateColors[i], m_AlignRate);
            }
            m_AlignStyle = Sub(m_AlignPage.transform);
            for (int i = 0; i < DetectProbe.StyleAnswers.Length; i++)
            {
                string answer = DetectProbe.StyleAnswers[i];
                AddAction(font, char.ToUpperInvariant(answer[0]) + answer.Substring(1), () => DetectProbe.Rate(answer), rateColors[i], m_AlignStyle);
            }
            m_AlignControls = Sub(m_AlignPage.transform);
            Transform r1 = Row(m_AlignControls.transform);
            m_AlignMode = AddAction(font, "", () => { m_AlignTurn = !m_AlignTurn; SetAlignLabels(); }, grey, r1);
            m_AlignStepSize = AddAction(font, "", () => { m_AlignFine = !m_AlignFine; SetAlignLabels(); }, grey, r1);
            Transform r2 = Row(m_AlignControls.transform);
            AddAction(font, "Left", () => Nudge(-1, 0, 0, "left"), blue, r2);
            AddAction(font, "Up", () => Nudge(0, 1, 0, "up"), blue, r2);
            AddAction(font, "Down", () => Nudge(0, -1, 0, "down"), blue, r2);
            AddAction(font, "Right", () => Nudge(1, 0, 0, "right"), blue, r2);
            Transform r3 = Row(m_AlignControls.transform);
            m_AlignMinus = AddAction(font, "", () => Nudge(0, 0, -1, m_AlignTurn ? "roll right" : "nearer"), blue, r3);
            m_AlignPlus = AddAction(font, "", () => Nudge(0, 0, 1, m_AlignTurn ? "roll left" : "farther"), blue, r3);
            Transform r4 = Row(m_AlignControls.transform);
            AddAction(font, "Zero", DetectProbe.ZeroOffset, new Color(0.45f, 0.1f, 0.1f, 0.9f), r4);
            AddAction(font, "Save", DetectProbe.SaveOffset, new Color(0.12f, 0.3f, 0.15f, 0.9f), r4);
            AddAction(font, "Done", DetectProbe.AlignDone, new Color(0.35f, 0.3f, 0.1f, 0.9f), r4);
            m_AlignOffset = MakeText(m_AlignPage.transform, font, Vector2.zero, Vector2.one, 26);
            m_AlignOffset.alignment = TextAnchor.MiddleCenter;
            m_AlignOffset.gameObject.AddComponent<LayoutElement>().preferredHeight = 60;
            SetAlignLabels();
            m_AlignStyle.SetActive(false);
            m_AlignControls.SetActive(false);
        }

        /// <summary>
        /// T6.11: the SPAAM run's page. A tap anywhere on the big button records the alignment and goes to the next
        /// crosshair; "last tap was bad" marks the previous one for the fit to drop; in chk_L, which eye sees the crosshair.
        /// </summary>
        static readonly string[] ModeNames = { "SDK", "This round", "All rounds" };
        static readonly string[] PersonModeNames = { "SDK", "Personal", "Device pool" };

        void BuildSpaamPage(Transform canvas, Font font, Color[] colors)
        {
            m_SpaamPage = Column(canvas);
            m_SpaamInfo = MakeText(m_SpaamPage.transform, font, Vector2.zero, Vector2.one, 26);
            m_SpaamInfo.alignment = TextAnchor.MiddleCenter;
            m_SpaamInfo.gameObject.AddComponent<LayoutElement>().preferredHeight = 150;
            m_SpaamTap = Sub(m_SpaamPage.transform);
            m_SpaamTap.GetComponent<LayoutElement>().flexibleHeight = 6;
            m_TapLabel = AddAction(font, "ALIGNED: tap anywhere here", DetectProbe.Tap, new Color(0.12f, 0.3f, 0.15f, 0.9f), m_SpaamTap);
            m_SpaamCheck = Sub(m_SpaamPage.transform);
            m_SpaamCheck.GetComponent<LayoutElement>().flexibleHeight = 6;
            for (int i = 0; i < DetectProbe.EyeCheckAnswers.Length; i++)
            {
                string answer = DetectProbe.EyeCheckAnswers[i];
                AddAction(font, char.ToUpperInvariant(answer[0]) + answer.Substring(1), () => DetectProbe.Rate(answer), colors[i + 1], m_SpaamCheck);
            }
            // T6.12 check steps: which eye model draws the squares, and the ratings (logged with the model)
            m_SpaamRate = Sub(m_SpaamPage.transform);
            m_SpaamRate.GetComponent<LayoutElement>().flexibleHeight = 6;
            Transform modes = Row(m_SpaamRate.transform);
            m_ModeLabels = new Text[ModeNames.Length];
            for (int i = 0; i < ModeNames.Length; i++)
            {
                string m = DetectProbe.EyeModes[i];
                m_ModeLabels[i] = AddAction(font, ModeNames[i], () => DetectProbe.SetMode(m), new Color(0.1f, 0.25f, 0.45f, 0.9f), modes);
            }
            for (int i = 0; i < DetectProbe.OverlayAnswers.Length; i++)
            {
                string answer = DetectProbe.OverlayAnswers[i];
                AddAction(font, char.ToUpperInvariant(answer[0]) + answer.Substring(1), () => DetectProbe.Rate(answer), colors[i], m_SpaamRate);
            }
            Transform bad = Sub(m_SpaamPage.transform).transform;
            AddAction(font, "Last tap was bad", DetectProbe.BadTap, new Color(0.45f, 0.1f, 0.1f, 0.9f), bad);
            m_SpaamCheck.SetActive(false);
            m_SpaamRate.SetActive(false);
        }

        // ------------------------------------------------------------------ campaign 0.5 (T7.0 C6)
        /// <summary>
        /// The trial set-up page (participant, condition, sheet, start) and the page during a trial (NEXT, bad mark,
        /// abort). A Bluetooth presenter does the same as the buttons: PageDown / Right / Down / Space / Enter = NEXT,
        /// PageUp / Left / Up / B / . = the last hit was bad.
        /// </summary>
        void BuildCampaignPages(Transform canvas, Font font)
        {
            Color grey = new Color(0.22f, 0.22f, 0.25f, 0.9f), blue = new Color(0.1f, 0.25f, 0.45f, 0.9f);
            Color green = new Color(0.12f, 0.3f, 0.15f, 0.9f), red = new Color(0.45f, 0.1f, 0.1f, 0.9f);
            m_CampaignPage = Column(canvas);
            m_CampaignInfo = MakeText(m_CampaignPage.transform, font, Vector2.zero, Vector2.one, 30);
            m_CampaignInfo.alignment = TextAnchor.MiddleCenter;
            m_CampaignInfo.gameObject.AddComponent<LayoutElement>().preferredHeight = 170;
            Transform r1 = Row(m_CampaignPage.transform);
            AddAction(font, "Participant -", () => { Campaign.SetParticipant(-1); RefreshCampaign(); }, grey, r1);
            AddAction(font, "Participant +", () => { Campaign.SetParticipant(+1); RefreshCampaign(); }, grey, r1);
            Transform r2 = Row(m_CampaignPage.transform);
            m_ConditionLabels = new Text[Campaign.Conditions.Length];
            for (int i = 0; i < Campaign.Conditions.Length; i++)
            {
                string c = Campaign.Conditions[i];
                m_ConditionLabels[i] = AddAction(font, Campaign.ConditionNames[i], () => { Campaign.SetCondition(c); RefreshCampaign(); }, blue, r2);
            }
            Transform r3 = Row(m_CampaignPage.transform);
            AddAction(font, "Sheet -", () => { Campaign.StepSheet(-1); RefreshCampaign(); }, grey, r3);
            AddAction(font, "Sheet +", () => { Campaign.StepSheet(+1); RefreshCampaign(); }, grey, r3);
            AddAction(font, "START TRIAL", StartTrial, green, m_CampaignPage);
            AddAction(font, "Back", () => ShowPage(m_Panel), grey, m_CampaignPage);

            m_MarkPage = Column(canvas);
            m_MarkInfo = MakeText(m_MarkPage.transform, font, Vector2.zero, Vector2.one, 30);
            m_MarkInfo.alignment = TextAnchor.MiddleCenter;
            m_MarkInfo.gameObject.AddComponent<LayoutElement>().preferredHeight = 170;
            GameObject next = Sub(m_MarkPage.transform);
            next.GetComponent<LayoutElement>().flexibleHeight = 5;
            AddAction(font, "NEXT  (target hit / start)", () => Campaign.Next("button"), green, next);
            Transform r4 = Row(m_MarkPage.transform);
            AddAction(font, "Last hit was BAD", () => Campaign.Bad("button"), new Color(0.35f, 0.3f, 0.1f, 0.9f), r4);
            AddAction(font, "Abort trial", Campaign.Abort, red, r4);
        }

        void StartTrial() => StartTrial(false);

        void StartTrial(bool warned)
        {
            if (m_Busy) return;
            string c = Campaign.Current.condition;
            if (!warned && Campaign.ShowsCross(c) && Campaign.EffectiveDisplay(c) != Campaign.DisplayFor(c))
            {
                // 0.5.002: no personal calibration: say so before the trial, not after
                m_WarnInfo.text = $"{Campaign.ParticipantId} has NO calibration.\nWithout it the crosshair is drawn with the SDK display, " +
                                  "and the results will be bad (about 13 mm off for Jack).\nCalibrate first, or start anyway.";
                ShowPage(m_WarnPage);
                return;
            }
            if (!Campaign.Prepare(out string why)) { SetStatus("Cannot start: " + why); RefreshCampaign(); return; }
            SetStatus($"Trial {Campaign.ParticipantId} {Campaign.ConditionName(Campaign.Current.condition)} {Campaign.CurrentSheetName}: {why}");
            StartCoroutine(RunProbe(new ProbeConfig { probe = "mark" }));
        }

        void RefreshCampaign()
        {
            if (m_CampaignInfo == null) return;
            Campaign.Reload();
            bool sdkOk = Campaign.SdkAllowed(Campaign.Current.participant);
            if (!sdkOk && Campaign.Current.condition == "S") Campaign.SetCondition("A");
            for (int i = 0; i < Campaign.Conditions.Length; i++)
                if (Campaign.Conditions[i] == "S") m_ConditionLabels[i].transform.parent.gameObject.SetActive(sdkOk);
            Campaign.State st = Campaign.Current;
            string sheet = Campaign.CurrentSheetName;
            Campaign.Sheet sh = Campaign.LoadSheet(sheet);
            string found = sh == null ? " (NOT FOUND)" : " (" + (sh.targets?.Length ?? 0) + " targets)";
            string eff = Campaign.EffectiveDisplay(st.condition);
            string display = !Campaign.ShowsCross(st.condition) ? "none: printed crosses"
                : eff == "person" ? "personal calibration" : eff == "all" ? "device pool" : eff == "sdk" && Campaign.DisplayFor(st.condition) == "person" ? "SDK (NO CALIBRATION: calibrate first)" : eff;
            string cal = (PersonCal.Exists(st.participant) ? PersonCal.Describe(st.participant) : Campaign.ParticipantId + ": no calibration") +
                         (PersonCal.UsesPool(st.participant) ? "  |  AR set to the device pool (Calibration page)" : "");
            m_CampaignInfo.text = "Participant " + Campaign.ParticipantId + "   |   " + Campaign.ConditionName(st.condition) + "   |   sheet " + sheet + found +
                                  "\nDisplay: " + display + "   |   Put this sheet on the glass, then START\n" + cal;
            for (int i = 0; i < m_ConditionLabels.Length; i++)
                m_ConditionLabels[i].fontStyle = Campaign.Conditions[i] == st.condition ? FontStyle.Bold : FontStyle.Normal;
        }

        // ------------------------------------------------------------------ 0.5.002: calibration per participant
        /// <summary>
        /// Calibration page: the participant (shared with the campaign page), NEW calibration (pcal: taps on A40's top-left
        /// square pair, fit, check on the whole grid) or LOAD + CHECK of the saved one (pcheck). After a new calibration,
        /// the result page saves it as a participant (asking before it overwrites one), redoes it or discards it.
        /// </summary>
        void BuildCalibrationPages(Transform canvas, Font font)
        {
            Color grey = new Color(0.22f, 0.22f, 0.25f, 0.9f), green = new Color(0.12f, 0.3f, 0.15f, 0.9f);
            Color blue = new Color(0.1f, 0.25f, 0.45f, 0.9f), red = new Color(0.45f, 0.1f, 0.1f, 0.9f);
            Text Info(GameObject page, int h)
            {
                Text x = MakeText(page.transform, font, Vector2.zero, Vector2.one, 28);
                x.alignment = TextAnchor.MiddleCenter;
                x.gameObject.AddComponent<LayoutElement>().preferredHeight = h;
                return x;
            }

            m_CalPage = Column(canvas);
            m_CalInfo = Info(m_CalPage, 170);
            Transform r1 = Row(m_CalPage.transform);
            AddAction(font, "Participant -", () => { Campaign.SetParticipant(-1); RefreshCal(); }, grey, r1);
            AddAction(font, "Participant +", () => { Campaign.SetParticipant(+1); RefreshCal(); }, grey, r1);
            AddAction(font, "NEW CALIBRATION on A40  (about 4 min)", () => { if (!m_Busy) StartCoroutine(RunProbe(new ProbeConfig { probe = "pcal" })); }, green, m_CalPage);
            AddAction(font, "LOAD + CHECK saved calibration  (2 min)", () =>
            {
                if (m_Busy) return;
                if (!PersonCal.Exists(Campaign.Current.participant)) { SetStatus(Campaign.ParticipantId + " has no saved calibration: make a new one"); return; }
                StartCoroutine(RunProbe(new ProbeConfig { probe = "pcheck" }));
            }, blue, m_CalPage);
            m_ChoiceLabel = AddAction(font, "", () =>
            {
                int pt = Campaign.Current.participant;
                PersonCal.SetPool(pt, !PersonCal.UsesPool(pt));
                RefreshCal();
            }, new Color(0.3f, 0.22f, 0.08f, 0.9f), m_CalPage);
            AddAction(font, "Back", () => ShowPage(m_Panel), grey, m_CalPage);

            m_CalResultPage = Column(canvas);
            m_CalResultInfo = Info(m_CalResultPage, 220);
            Transform r2 = Row(m_CalResultPage.transform);
            AddAction(font, "Save as -", () => { m_SaveAs = Mathf.Max(0, m_SaveAs - 1); RefreshCalResult(); }, grey, r2);
            AddAction(font, "Save as +", () => { m_SaveAs = Mathf.Min(99, m_SaveAs + 1); RefreshCalResult(); }, grey, r2);
            m_SaveLabel = AddAction(font, "SAVE", () => SaveCal(false), green, m_CalResultPage);
            Transform r3 = Row(m_CalResultPage.transform);
            AddAction(font, "REDO calibration", () => { if (!m_Busy) StartCoroutine(RunProbe(new ProbeConfig { probe = "pcal" })); }, blue, r3);
            AddAction(font, "Discard", () => { SetStatus("Calibration discarded"); ShowPage(m_CalPage); }, red, r3);

            m_ConfirmPage = Column(canvas);
            m_ConfirmInfo = Info(m_ConfirmPage, 220);
            AddAction(font, "YES, overwrite", () => SaveCal(true), red, m_ConfirmPage);
            AddAction(font, "No, go back", () => ShowPage(m_CalResultPage), grey, m_ConfirmPage);

            m_WarnPage = Column(canvas);
            m_WarnInfo = Info(m_WarnPage, 220);
            AddAction(font, "Calibrate this participant first", () => ShowPage(m_CalPage), green, m_WarnPage);
            AddAction(font, "Start anyway (SDK display)", () => { ShowPage(m_CampaignPage); StartTrial(true); }, red, m_WarnPage);
            AddAction(font, "Cancel", () => ShowPage(m_CampaignPage), grey, m_WarnPage);
        }

        void RefreshCal()
        {
            if (m_CalInfo == null) return;
            int pt = Campaign.Current.participant;
            bool pool = PersonCal.UsesPool(pt);
            if (m_ChoiceLabel != null)
                m_ChoiceLabel.text = "AR display for " + Campaign.ParticipantId + ": " + (pool ? "DEVICE POOL (tap for personal)" : "PERSONAL calibration (tap for device pool)");
            m_CalInfo.text = "Participant " + Campaign.ParticipantId + "\n" + PersonCal.Describe(pt) + (pool ? "  |  AR uses the device pool" : "") +
                             "\nA40 flat on the glass. NEW: crosshair taps on the top-left square pair, then a check on the whole sheet.";
        }

        void RefreshCalResult()
        {
            if (m_CalResultInfo == null) return;
            if (m_SaveAs < 0) m_SaveAs = Campaign.Current.participant;
            string who = PersonCal.Name(m_SaveAs);
            m_CalResultInfo.text = (DetectProbe.PcalResult != null ? "New calibration\n" : "No calibration fitted\n") + DetectProbe.PcalText +
                                   "\nSave as " + who + (PersonCal.Exists(m_SaveAs) ? " (has one: you will be asked before it is replaced)" : "");
            m_SaveLabel.text = DetectProbe.PcalResult != null ? "SAVE as " + who : "Nothing to save: REDO";
        }

        void SaveCal(bool confirmed)
        {
            if (DetectProbe.PcalResult == null) { SetStatus("Nothing to save: redo the calibration"); return; }
            if (!confirmed && PersonCal.Exists(m_SaveAs))
            {
                m_ConfirmInfo.text = PersonCal.Name(m_SaveAs) + " already has a calibration:\n" + PersonCal.Describe(m_SaveAs) + "\nReplace it with the new one?";
                ShowPage(m_ConfirmPage);
                return;
            }
            try
            {
                PersonCal.Save(m_SaveAs, DetectProbe.PcalResult, DetectProbe.PcalTaps);
                Debug.Log("[EyeLab probe] calibration saved: " + PersonCal.Describe(m_SaveAs) + (confirmed ? " (overwrote)" : ""));
                if (Campaign.Current.participant != m_SaveAs) Campaign.SetParticipant(m_SaveAs - Campaign.Current.participant);
                SetStatus("Saved: " + PersonCal.Describe(m_SaveAs));
                ShowPage(m_CalPage);
            }
            catch (System.Exception e) { SetStatus("Not saved: " + e.Message); ShowPage(m_CalResultPage); }
        }

        void SetSquareLabel()
        {
            if (m_SquareLabel != null) m_SquareLabel.text = "Square: " + ((m_Glasses?.ReferenceOn ?? false) ? "ON (tap for off)" : "OFF (tap for on)");
        }

        void UpdateMark()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb != null)
            {
                if (kb.pageDownKey.wasPressedThisFrame || kb.rightArrowKey.wasPressedThisFrame || kb.downArrowKey.wasPressedThisFrame ||
                    kb.spaceKey.wasPressedThisFrame || kb.enterKey.wasPressedThisFrame || kb.numpadEnterKey.wasPressedThisFrame)
                    Campaign.Next("key");
                else if (kb.pageUpKey.wasPressedThisFrame || kb.leftArrowKey.wasPressedThisFrame || kb.upArrowKey.wasPressedThisFrame ||
                         kb.bKey.wasPressedThisFrame || kb.periodKey.wasPressedThisFrame)
                    Campaign.Bad("key");
            }
#endif
            if (m_MarkInfo == null) return;
            double age = DetectProbe.BoardAgeMs;
            string seen = age < 0 ? "Sheet not seen yet" : age < 500 ? "Sheet tracked" : "Sheet last seen " + (age / 1000).ToString("F1", CultureInfo.InvariantCulture) + " s ago";
            m_MarkInfo.text = Campaign.ParticipantId + "  " + Campaign.ConditionName(Campaign.Current.condition) + "  " + Campaign.ActiveSheet?.name +
                              "   |   " + DetectProbe.ActiveStep + "\n" + seen + "   |   next " + Campaign.NextPresses + ", bad " + Campaign.BadMarks +
                              (Campaign.Aborted ? "   |   ABORTING" : "");
        }

        static GameObject Sub(Transform parent)
        {
            var go = new GameObject("part", typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            var v = go.GetComponent<VerticalLayoutGroup>();
            v.spacing = 12;
            v.childForceExpandHeight = true;
            v.childControlHeight = true;
            v.childControlWidth = true;
            go.GetComponent<LayoutElement>().flexibleHeight = 1;
            return go;
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

        Text AddAction(Font font, string label, UnityEngine.Events.UnityAction action, Color? color = null, object parent = null)
        {
            var go = new GameObject(label, typeof(RectTransform), typeof(Image), typeof(Button));
            Transform where = parent is GameObject g ? g.transform : parent as Transform ?? m_Panel.transform;
            go.transform.SetParent(where, false);
            go.GetComponent<Image>().color = color ?? new Color(0.45f, 0.1f, 0.1f, 0.9f);
            go.GetComponent<Button>().onClick.AddListener(action);
            Text t = MakeText(go.transform, font, Vector2.zero, Vector2.one, 30);
            t.text = label;
            t.alignment = TextAnchor.MiddleCenter;
            return t;
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
