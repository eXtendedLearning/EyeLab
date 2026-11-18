using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>One step of a guided run: what is said, what is shown, how long it lasts.</summary>
    public readonly struct CueStep
    {
        public readonly string Id, Say, Show;
        public readonly float Seconds;
        public readonly bool Ticks;     // three ticks before the step ends (a move follows)
        public readonly float Metronome; // > 0: a soft tick every this many seconds (pace for shaking)

        public CueStep(string id, string say, float seconds, string show = null, bool ticks = false, float metronome = 0f)
        {
            Id = id; Say = say; Show = show ?? say; Seconds = seconds; Ticks = ticks; Metronome = metronome;
        }
    }

    /// <summary>
    /// T6.3 guided runs (Jack, 2026-11-18: audio cues and structured prompts, so he watches the glasses and not the
    /// clock). Plays a list of <see cref="CueStep"/>s: at each step a tone and a spoken prompt (<c>CueSpeech.java</c>,
    /// Android TextToSpeech; tones only if no engine), the instruction large in the glasses and on the Beam Pro
    /// screen with the step's time left, three ticks before a move. Every step start is logged to the run's
    /// <c>events.csv</c> as <c>cue,&lt;id&gt;</c> with all clocks, so the analyser can cut the run into phases.
    /// </summary>
    public sealed class CuePlayer : MonoBehaviour
    {
        public const string EventFile = "events.csv";
        public const string EventHeader = "event,detail," + ProbeClocks.Sample.Header;

        static CuePlayer s_Instance;
        AudioSource m_Audio;
        AudioClip m_Step, m_Tick, m_Soft, m_End;
#if UNITY_ANDROID && !UNITY_EDITOR
        AndroidJavaClass m_Speech;
#endif

        /// <summary>Called by the runner at start: audio clips, and the speech engine starts initialising.</summary>
        public static CuePlayer Ensure()
        {
            if (s_Instance != null) return s_Instance;
            var go = new GameObject("EyeLab Cues");
            DontDestroyOnLoad(go);
            s_Instance = go.AddComponent<CuePlayer>();
            s_Instance.Init();
            return s_Instance;
        }

        void Init()
        {
            if (FindAnyObjectByType<AudioListener>() == null) gameObject.AddComponent<AudioListener>();
            m_Audio = gameObject.AddComponent<AudioSource>();
            m_Audio.playOnAwake = false;
            m_Audio.spatialBlend = 0f;
            m_Step = Tone("step", 880f, 0.15f, 0.5f);
            m_Tick = Tone("tick", 1320f, 0.06f, 0.5f);
            m_Soft = Tone("soft", 1000f, 0.03f, 0.25f);
            m_End = Tone("end", 660f, 0.6f, 0.5f);
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                m_Speech = new AndroidJavaClass("it.univpm.eyelab.probes.CueSpeech");
                using (var up = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (AndroidJavaObject activity = up.GetStatic<AndroidJavaObject>("currentActivity"))
                    Debug.Log("[EyeLab probe] speech " + m_Speech.CallStatic<string>("init", activity));
            }
            catch (Exception e) { Debug.LogWarning("[EyeLab probe] speech unavailable: " + e.Message); m_Speech = null; }
#endif
        }

        public string SpeechState
        {
            get
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                try { return m_Speech != null ? m_Speech.CallStatic<string>("state") : "no helper"; }
                catch (Exception e) { return "error " + e.Message; }
#else
                return "not on Android";
#endif
            }
        }

        static AudioClip Tone(string name, float hz, float seconds, float gain)
        {
            const int rate = 44100;
            int n = Mathf.CeilToInt(rate * seconds);
            var data = new float[n];
            int ramp = Mathf.Min(n / 4, rate / 200);   // 5 ms in and out, no clicks
            for (int i = 0; i < n; i++)
            {
                float env = Mathf.Min(1f, Mathf.Min(i, n - 1 - i) / (float)Mathf.Max(1, ramp));
                data[i] = gain * env * Mathf.Sin(2f * Mathf.PI * hz * i / rate);
            }
            AudioClip c = AudioClip.Create(name, n, 1, rate, false);
            c.SetData(data, 0);
            return c;
        }

        public void Say(string text)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try { if (m_Speech != null && m_Speech.CallStatic<bool>("say", text)) return; }
            catch (Exception e) { Debug.LogWarning("[EyeLab probe] say: " + e.Message); }
#endif
            Debug.Log("[EyeLab probe] (not spoken) " + text);
        }

        public void PlayEnd() => m_Audio.PlayOneShot(m_End);

        public static void Log(ProbeRun run, string name, string detail) =>
            run.Row(EventFile, EventHeader, $"{name},{detail.Replace(',', ';')},{ProbeClocks.Now().Csv()}");

        /// <summary>
        /// Plays <paramref name="steps"/> in order. <paramref name="show"/> gets (instruction, seconds left in the
        /// step, fraction of the step left, "step i/n"). Stops early if <paramref name="abort"/> returns true.
        /// </summary>
        public IEnumerator Play(ProbeRun run, IReadOnlyList<CueStep> steps, Action<string, float, float, string> show,
                                Func<bool> abort = null)
        {
            Log(run, "cue_speech", SpeechState);
            float total = 0f;
            foreach (CueStep s in steps) total += s.Seconds;
            run.Meta("cue_steps", steps.Count.ToString());
            run.Meta("cue_seconds", ProbeRun.F(total));
            for (int i = 0; i < steps.Count; i++)
            {
                CueStep s = steps[i];
                Log(run, "cue", s.Id);
                m_Audio.PlayOneShot(m_Step);
                Say(s.Say);
                float t0 = Time.realtimeSinceStartup, nextBeat = s.Metronome;
                int ticksPlayed = 0;
                string counter = $"step {i + 1}/{steps.Count}";
                while (true)
                {
                    float el = Time.realtimeSinceStartup - t0, left = s.Seconds - el;
                    if (left <= 0f) break;
                    if (abort != null && abort()) { Log(run, "cue_abort", s.Id); yield break; }
                    if (s.Ticks && ticksPlayed < 3 && left <= 3 - ticksPlayed) { m_Audio.PlayOneShot(m_Tick); ticksPlayed++; }
                    if (s.Metronome > 0f && el >= nextBeat) { m_Audio.PlayOneShot(m_Soft); nextBeat += s.Metronome; }
                    show?.Invoke(s.Show, left, left / s.Seconds, counter);
                    yield return null;
                }
            }
            Log(run, "cue", "end");
            PlayEnd();
            Say("Done.");
        }
    }
}
