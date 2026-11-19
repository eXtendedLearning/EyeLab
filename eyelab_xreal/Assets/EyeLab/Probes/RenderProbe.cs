using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.XR;
using Unity.XR.XREAL;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.5 render gate (Jack, 2026-11-19: the square flickers in the glasses). Two causes are possible and the logs so
    /// far cannot separate them: frame pacing (Unity draws 41-49 fps against a 60 Hz display, so some frames are shown
    /// twice) and aliasing (render scale 0.8, no anti-aliasing). The run switches the rendering settings live, one
    /// setting per 15 s (hold still, turn slowly, rate), and logs every frame's XR display counters to
    /// <c>render.csv</c>. Jack rates each setting on the Beam Pro screen; the ratings go to <c>events.csv</c> as
    /// <c>rating,&lt;setting&gt;:&lt;answer&gt;</c>. Settings are announced by number only, so the rating is blind.
    /// </summary>
    public sealed class RenderProbe : MonoBehaviour
    {
        /// <summary>
        /// The rendering default from T6.5 (2026-11-19): "lean" at render scale 0.8 (HDR, post-processing, shadows and
        /// the global volume off, no MSAA, no SMAA). Unity reaches 59.9 fps against the 60 Hz display (45 fps with the
        /// URP template defaults) and Jack rated it steady. Applied once at start-up by the runner, before any probe.
        /// </summary>
        public static string ApplyDefaults()
        {
            var urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            if (urp != null) { urp.renderScale = 0.8f; urp.msaaSampleCount = 1; urp.supportsHDR = false; }
            Camera cam = Camera.main;
            if (cam != null)
            {
                cam.allowHDR = false;
                cam.allowMSAA = false;
                UniversalAdditionalCameraData data = cam.GetUniversalAdditionalCameraData();
                if (data != null)
                {
                    data.renderPostProcessing = false;
                    data.renderShadows = false;
                    data.antialiasing = AntialiasingMode.None;
                }
            }
            Volume v = FindAnyObjectByType<Volume>();
            if (v != null) v.enabled = false;
            return $"lean: urp {(urp != null ? urp.name : "none")} scale 0.8, msaa 1, hdr off; camera {(cam != null ? cam.name : "none")} " +
                   $"post off, shadows off, aa none; volume {(v != null ? "off" : "none")}";
        }

        public const string File = "render.csv";
        public const string Header = "setting,step,unity_frame,dt_ms,present,dropped,m2p_ms,extended,early,teared,scale,msaa," +
                                     ProbeClocks.Sample.Header;

        /// <summary>One rendering configuration under test.</summary>
        public readonly struct Setting
        {
            public readonly string Id;
            public readonly float Scale;
            public readonly int Msaa;
            public readonly bool Hdr, Post, Shadows, Smaa, HeadLocked, Volume;

            public Setting(string id, float scale, int msaa = 1, bool hdr = false, bool post = false, bool shadows = false,
                           bool smaa = false, bool headLocked = false, bool volume = false)
            {
                Id = id; Scale = scale; Msaa = msaa; Hdr = hdr; Post = post; Shadows = shadows; Smaa = smaa;
                HeadLocked = headLocked; Volume = volume;
            }

            public override string ToString() =>
                $"{Id}: scale {Scale:F1}, msaa {Msaa}, hdr {Hdr}, post {Post}, shadows {Shadows}, smaa {Smaa}, " +
                $"head-locked {HeadLocked}, volume {Volume}";
        }

        /// <summary>
        /// In this order: the 0.4.004 settings, the same head-locked (motion versus aliasing), lean (HDR, post-processing
        /// and shadows off), lean at 0.6 and at 1.0.
        /// <para>0.4.005 also had SMAA (31.7 fps, rated shimmer and judder) and MSAA 2x: with MSAA the glasses went blank
        /// and stayed blank after MSAA was off again and after an app restart (T6.5, 2026-11-19; 4x did the same in
        /// 0.4.003). Never switch MSAA on, live or in the asset: <see cref="Apply"/> refuses it.</para>
        /// </summary>
        public static readonly Setting[] Settings =
        {
            new Setting("now", 0.8f, hdr: true, post: true, shadows: true, volume: true),
            new Setting("head_locked", 0.8f, hdr: true, post: true, shadows: true, volume: true, headLocked: true),
            new Setting("lean", 0.8f),
            new Setting("lean_06", 0.6f),
            new Setting("lean_10", 1.0f),
        };

        public static readonly string[] Answers = { "steady", "edges shimmer", "flicker or judder", "nothing visible" };

        public static List<CueStep> Cues()
        {
            var cues = new List<CueStep>
            {
                new CueStep("ready", "Rendering test. Stand anywhere and look at the square and the grid ahead. " +
                                     "For each setting: hold still, turn slowly, then rate it on the Beam Pro screen.", 10f,
                            "Look at the square and grid. Rate each setting", ticks: true),
            };
            for (int i = 0; i < Settings.Length; i++)
            {
                string id = Settings[i].Id, n = (i + 1).ToString(CultureInfo.InvariantCulture);
                cues.Add(new CueStep(id + "_still", $"Setting {n}. Hold still and look at it.", 5f, $"Setting {n}: HOLD STILL"));
                cues.Add(new CueStep(id + "_turn", "Turn your head slowly left and right.", 6f, $"Setting {n}: turn SLOWLY", metronome: 1f));
                cues.Add(new CueStep(id + "_rate", "Rate it now.", 4f, $"Setting {n}: RATE on the screen", ticks: true));
            }
            return cues;
        }

        static RenderProbe s_Active;
        ProbeRun m_Run;
        Setting m_Current;
        string m_Step = "";
        Camera m_Cam;
        UniversalAdditionalCameraData m_CamData;
        UniversalRenderPipelineAsset m_Urp;
        Volume m_Volume;
        XRDisplaySubsystem m_Display;
        long m_Rows;

        // what was there before the run, restored at the end
        float m_Scale0;
        int m_Msaa0;
        bool m_Hdr0, m_CamHdr0, m_CamMsaa0, m_Post0, m_Shadows0, m_Volume0;
        AntialiasingMode m_Aa0;

        /// <summary>A rating tap on the Beam Pro screen, for the setting now shown.</summary>
        public static void Rate(string answer)
        {
            if (s_Active == null) return;
            CuePlayer.Log(s_Active.m_Run, "rating", s_Active.m_Current.Id + ":" + answer);
            Debug.Log($"[EyeLab probe] render rating: {s_Active.m_Current.Id} ({s_Active.m_Step}) = {answer}");
        }

        public static IEnumerator Run(ProbeRun run, GlassesView glasses, Action<string> status, Action<string, float, float, string> show)
        {
            var go = new GameObject("EyeLab RenderProbe");
            var p = go.AddComponent<RenderProbe>();
            p.m_Run = run;
            s_Active = p;
            p.Capture();
            try { if (p.m_Display != null) p.m_Display.EnableTearedFrameCount(true); }
            catch (Exception e) { Debug.LogWarning("[EyeLab probe] teared frame count: " + e.Message); }
            int xrealTarget = -1;
            try { xrealTarget = XREALPlugin.GetTargetFrameRate(); }
            catch (Exception e) { Debug.LogWarning("[EyeLab probe] xreal target frame rate: " + e.Message); }
            run.Meta("xreal_target_frame_rate", xrealTarget.ToString(CultureInfo.InvariantCulture));
            run.Meta("unity_target_frame_rate", Application.targetFrameRate.ToString(CultureInfo.InvariantCulture));
            run.Meta("render_settings", string.Join(" | ", Array.ConvertAll(Settings, s => s.ToString())));
            run.Meta("render_before", $"scale {p.m_Scale0}, msaa {p.m_Msaa0}, hdr {p.m_Hdr0}, post {p.m_Post0}, shadows {p.m_Shadows0}, " +
                                      $"aa {p.m_Aa0}, volume {p.m_Volume0}, urp {(p.m_Urp != null ? p.m_Urp.name : "none")}");
            if (glasses != null && glasses.PlacedAt < 0f) glasses.PlaceReference();

            var cues = Cues();
            float nextStatus = 0f;
            yield return CuePlayer.Ensure().Play(run, cues, (text, left, frac, counter) =>
            {
                show?.Invoke(text, left, frac, counter);
                if (Time.realtimeSinceStartup < nextStatus) return;
                nextStatus = Time.realtimeSinceStartup + 0.5f;
                status($"{counter}: {text} | {left:F0} s | fps {1f / Mathf.Max(Time.smoothDeltaTime, 1e-4f):F0}");
            }, null, step =>
            {
                p.m_Step = step.Id;
                foreach (Setting s in Settings)
                    if (step.Id == s.Id + "_still") { p.Apply(s, glasses); break; }
            });

            p.Restore(glasses);
            run.Meta("render_rows", p.m_Rows.ToString(CultureInfo.InvariantCulture));
            s_Active = null;
            Destroy(go);
        }

        void Capture()
        {
            m_Cam = Camera.main;
            m_CamData = m_Cam != null ? m_Cam.GetUniversalAdditionalCameraData() : null;
            m_Urp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            m_Volume = FindAnyObjectByType<Volume>();
            m_Display = XREALDisplay();
            if (m_Urp != null) { m_Scale0 = m_Urp.renderScale; m_Msaa0 = m_Urp.msaaSampleCount; m_Hdr0 = m_Urp.supportsHDR; }
            if (m_Cam != null) { m_CamHdr0 = m_Cam.allowHDR; m_CamMsaa0 = m_Cam.allowMSAA; }
            if (m_CamData != null) { m_Post0 = m_CamData.renderPostProcessing; m_Shadows0 = m_CamData.renderShadows; m_Aa0 = m_CamData.antialiasing; }
            m_Volume0 = m_Volume != null && m_Volume.enabled;
        }

        static XRDisplaySubsystem XREALDisplay()
        {
            var list = new List<XRDisplaySubsystem>();
            SubsystemManager.GetSubsystems(list);
            foreach (XRDisplaySubsystem d in list) if (d.running) return d;
            return list.Count > 0 ? list[0] : null;
        }

        void Apply(Setting s, GlassesView glasses)
        {
            m_Current = s;
            try
            {
                if (s.Msaa > 1) throw new InvalidOperationException("MSAA blanks the XREAL glasses until they are reconnected (T6.5)");
                if (m_Urp != null)
                {
                    m_Urp.renderScale = s.Scale;
                    m_Urp.msaaSampleCount = s.Msaa;
                    m_Urp.supportsHDR = s.Hdr;
                }
                if (m_Cam != null) { m_Cam.allowHDR = s.Hdr; m_Cam.allowMSAA = s.Msaa > 1; }
                if (m_CamData != null)
                {
                    m_CamData.renderPostProcessing = s.Post;
                    m_CamData.renderShadows = s.Shadows;
                    m_CamData.antialiasing = s.Smaa ? AntialiasingMode.SubpixelMorphologicalAntiAliasing : AntialiasingMode.None;
                    m_CamData.antialiasingQuality = AntialiasingQuality.High;
                }
                if (m_Volume != null) m_Volume.enabled = s.Volume;
                glasses?.SetReferenceHeadLocked(s.HeadLocked);
                CuePlayer.Log(m_Run, "setting", s.ToString());
                Debug.Log("[EyeLab probe] render setting " + s);
            }
            catch (Exception e)
            {
                CuePlayer.Log(m_Run, "setting_error", s.Id + ": " + e.GetType().Name + " " + e.Message);
                Debug.LogWarning("[EyeLab probe] render setting " + s.Id + ": " + e);
            }
        }

        void Restore(GlassesView glasses)
        {
            if (m_Urp != null) { m_Urp.renderScale = m_Scale0; m_Urp.msaaSampleCount = m_Msaa0; m_Urp.supportsHDR = m_Hdr0; }
            if (m_Cam != null) { m_Cam.allowHDR = m_CamHdr0; m_Cam.allowMSAA = m_CamMsaa0; }
            if (m_CamData != null) { m_CamData.renderPostProcessing = m_Post0; m_CamData.renderShadows = m_Shadows0; m_CamData.antialiasing = m_Aa0; }
            if (m_Volume != null) m_Volume.enabled = m_Volume0;
            glasses?.SetReferenceHeadLocked(false);
            CuePlayer.Log(m_Run, "setting", "restored");
        }

        void Update()
        {
            if (m_Run == null || string.IsNullOrEmpty(m_Step) || m_Current.Id == null) return;
            var ci = CultureInfo.InvariantCulture;
            int present = -1, dropped = -1;
            float m2p = float.NaN, ext = float.NaN, early = float.NaN, tear = float.NaN;
            if (m_Display != null)
            {
                if (!m_Display.TryGetFramePresentCount(out present)) present = -1;
                if (!m_Display.TryGetDroppedFrameCount(out dropped)) dropped = -1;
                if (m_Display.TryGetMotionToPhoton(out float m)) m2p = m * 1000f;
                if (!UnityEngine.XR.Provider.XRStats.TryGetStat(m_Display, "ExtendedFrameCount", out ext)) ext = float.NaN;
                if (!UnityEngine.XR.Provider.XRStats.TryGetStat(m_Display, "EarlyFrameCount", out early)) early = float.NaN;
                if (!UnityEngine.XR.Provider.XRStats.TryGetStat(m_Display, "TearedFrameCount", out tear)) tear = float.NaN;
            }
            string f(float v) => float.IsNaN(v) ? "" : v.ToString("R", ci);
            m_Run.Row(File, Header,
                $"{m_Current.Id},{m_Step},{Time.frameCount},{(Time.unscaledDeltaTime * 1000f).ToString("R", ci)},{present},{dropped}," +
                $"{f(m2p)},{f(ext)},{f(early)},{f(tear)},{(m_Display != null ? m_Display.scaleOfAllRenderTargets : 0f).ToString("R", ci)}," +
                $"{(m_Urp != null ? m_Urp.msaaSampleCount : 0)},{ProbeClocks.Now().Csv()}");
            m_Rows++;
        }
    }
}
