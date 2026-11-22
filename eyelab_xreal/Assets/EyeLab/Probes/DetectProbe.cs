using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using EyeLab.Aruco;
using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLab.Probes
{
    /// <summary>
    /// ArUco detection and the board pose on the Beam Pro's CPU with the native plugin, while the glasses draw.
    /// Two variants share this probe:
    /// <list type="bullet">
    /// <item><b>detect</b> (T6.6, gate B3a): full-frame detection at 1-8 OpenCV threads, plate A.</item>
    /// <item><b>budget</b> (T6.7): the same pipeline within a budget. Per cue step a mode (full frame every Eye frame,
    /// full frame capped at 15 Hz, or a region of interest around the target's last box), a thread count, the worker's
    /// nice value and a target (plate A, plate L, the ChArUco board's markers). Each pose is marked trusted or not by
    /// PoseLock's planar-ambiguity rule (thresholds from set.json).</item>
    /// <item><b>lean</b> (T6.8): region detection only, at every Eye frame or capped at 15 Hz, on the ChArUco board and
    /// plate L tilted so both faces show; trusted also needs PoseLock's RMS ceiling (a non-planar solve can diverge).</item>
    /// </list>
    /// Each run has two parts:
    /// <list type="number">
    /// <item><b>Bench</b> (no wearing needed): every frame of the set the agent pushed to <c>files/b3a/</c> (<c>set.json</c>,
    /// PGM frames made by <c>python -m xreal_gates b3a-set</c>) is processed 3 times per bench mode on a worker thread.
    /// <c>bench.csv</c> and <c>bench_corners.csv</c> go to the parity test against the Python pipeline.</item>
    /// <item><b>Live</b>: every new Eye frame is copied once on the main thread, rows reversed to upright (the Eye stores
    /// them bottom-up), with the native head pose at its <c>cam_ts</c>, and processed on a worker thread (newest wins; a
    /// frame that arrives while the worker is busy replaces the waiting one and is counted as skipped).
    /// <c>live.csv</c> has the times, the region and the pose per processed frame, <c>head.csv</c> the head pose, and
    /// <c>frames.csv</c> every Unity frame's time; every 10th processed frame is saved for the parity test.</item>
    /// </list>
    /// All worker timestamps are CLOCK_MONOTONIC, the HMD clock of <c>cam_ts</c> (T6.0 BS), so the age of a pose when
    /// it is ready is <c>end_mono_ns − cam_ts</c>.
    /// </summary>
    public sealed class DetectProbe : MonoBehaviour
    {
        [Serializable]
        public sealed class Target
        {
            public string name;
            public int[] ids;
            public float[] corners;
        }

        [Serializable]
        public sealed class PoseLockSettings
        {
            public float min_ambiguity_ratio = 3f;          // python/pose_lock.py PoseLockConfig, written into set.json
            public float ambiguity_min_separation_deg = 10f;
            public float max_acquire_reproj_rms_px = 15f;    // T6.8: PoseLock's ceiling for any candidate
        }

        [Serializable]
        public sealed class FrameSet
        {
            public string schema;
            public string board_name;
            public string calibration;
            public int dictionary;
            public float[] tuning;
            public double[] K;
            public double[] dist;
            public int[] ids;
            public float[] corners;
            public string[] frames;
            public Target[] targets;           // T6.7
            public PoseLockSettings pose_lock;  // T6.7
        }

        /// <summary>How a cue step processes frames.</summary>
        public readonly struct Plan
        {
            public readonly string Mode;    // full | cap | roi | roi15 (region, capped at CapHz)
            public readonly int Threads, Nice;
            public readonly string Target;  // a set.json target name; "" = the set's own board (plate A)
            public Plan(string mode, int threads, int nice, string target) { Mode = mode; Threads = threads; Nice = nice; Target = target; }
        }

        public static string SetDir => Path.Combine(Application.persistentDataPath, "b3a");

        const string LiveFile = "live.csv";
        const string ResultHeader = "markers,rejected,board_markers,pose_ok,planar,points,clahe_ms,detect_ms,pose_ms,total_ms," +
                                    "r0,r1,r2,t0,t1,t2,rms_px,amb_ratio,amb_deg,roi_x,roi_y,roi_w,roi_h,solution,prior_used";
        const string LiveHeader = "seq,step,threads,cam_ts,arrival_mono_ns,start_mono_ns,end_mono_ns,skipped," + ResultHeader +
                                  ",ids,saved,mode,target,nice,accepted,pr0,pr1,pr2,cap_skipped";
        const string CornerHeader = "seq,id,x0,y0,x1,y1,x2,y2,x3,y3";
        const string BenchHeader = "frame,threads_req,threads,rep," + ResultHeader + ",mode";
        const string FramesFile = "frames.csv";
        const string FramesHeader = "step,threads,unity_frame,dt_ms,present,mono_ns,eye_ms";
        const string HeadFile = "head.csv";
        const string HeadHeader = "seq,cam_ts,ok,m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15";
        const int SaveEvery = 10;
        const int BenchReps = 3;
        const double CapHz = 15.0;
        const double RoiMaxAgeNs = 0.5e9;     // a box or prior older than this (by cam_ts) is not used
        const float RoiMarginFrac = 0.5f;     // ROI = projected board box grown by this share of its larger side ...
        const float RoiMarginMinPx = 48f;     // ... and at least this many pixels on each side
        const int NiceRoi = 10;

        // ------------------------------------------------------------------ T6.6 (detect)
        static readonly int[] DetectBenchThreads = { 1, 2, 4, 8 };

        public static List<CueStep> DetectCues() => new List<CueStep>
        {
            new CueStep("ready", "Detection test. Stand about half a metre from plate A, so it is straight ahead of you.", 8f,
                        "Plate A straight ahead, about 50 cm", ticks: true),
            new CueStep("t2_still", "Look at the plate and hold still.", 15f, "Look at the plate: HOLD STILL"),
            new CueStep("t2_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("t2_away", "Look away from the plate.", 6f, "Look AWAY from the plate"),
            new CueStep("t4_still", "Back to the plate. Hold still.", 15f, "Back to the plate: HOLD STILL"),
            new CueStep("t4_move", "Move your head slowly again.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("t4_away", "Look away again.", 6f, "Look AWAY from the plate", ticks: true),
        };

        /// <summary>OpenCV threads for a T6.6 cue step: the number after "t" in "t2_still"; 2 (OpenCV's Android default) otherwise.</summary>
        public static int ThreadsFor(string stepId)
        {
            if (stepId != null && stepId.Length > 2 && stepId[0] == 't')
            {
                int end = stepId.IndexOf('_');
                if (end > 1 && int.TryParse(stepId.Substring(1, end - 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0)
                    return n;
            }
            return 2;
        }

        // ------------------------------------------------------------------ T6.7 (budget)
        // Steps share a prefix per target: a_ plate A, l_ plate L, b_ the ChArUco board. The still phase on plate A runs
        // full, then cap, then roi on the same scene, so the three modes are compared without moving.
        public static List<CueStep> BudgetCues() => new List<CueStep>
        {
            new CueStep("a_ready", "Detection budget test. Plate A straight ahead, about sixty centimetres.", 8f,
                        "Plate A straight ahead, about 60 cm", ticks: true),
            new CueStep("a_full_still", "Look at plate A and hold still.", 10f, "Plate A: HOLD STILL"),
            new CueStep("a_cap_still", "Keep holding still.", 10f, "Keep HOLDING STILL"),
            new CueStep("a_roi_still", "Keep holding still.", 12f, "Keep HOLDING STILL", ticks: true),
            new CueStep("a_roi_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("a_roi_away", "Look away from the plate.", 5f, "Look AWAY"),
            new CueStep("a_roi_back", "Back to the plate. Hold still.", 8f, "Back to plate A: HOLD STILL", ticks: true),
            new CueStep("a_full_move", "Move your head slowly again.", 10f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("l_ready", "Now plate L. Put it where plate A was, with both faces towards you.", 12f,
                        "Swap to plate L, both faces towards you", ticks: true),
            new CueStep("l_roi_still", "Look at plate L and hold still.", 12f, "Plate L: HOLD STILL", ticks: true),
            new CueStep("l_roi_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("b_ready", "Now the ChArUco board, upright, about sixty centimetres away.", 12f,
                        "Swap to the ChArUco board, upright, 60 cm", ticks: true),
            new CueStep("b_roi_still", "Look at the board and hold still.", 12f, "Board: HOLD STILL", ticks: true),
            new CueStep("b_roi_move", "Move your head slowly. Keep the board in view.", 12f, "Move SLOWLY, board in view", metronome: 1f),
        };

        // ------------------------------------------------------------------ T6.8 (lean)
        // T6.7 run 1: the region at 1 thread was the only mode near the render margin (5-7 % late against 12-18 %);
        // plate A at 75 cm was too small for a stable normal. Board first, then plate L tilted about 45 degrees so
        // both faces show (standing on the table, its upward face was edge-on). Each target: region at every Eye
        // frame, then capped at 15 Hz, still and moving.
        public static List<CueStep> LeanCues() => new List<CueStep>
        {
            new CueStep("b_ready", "Lean detection test. The ChArUco board upright, about sixty centimetres away.", 10f,
                        "ChArUco board upright, about 60 cm", ticks: true),
            new CueStep("b_roi_still", "Look at the board and hold still.", 12f, "Board: HOLD STILL", ticks: true),
            new CueStep("b_r15_still", "Keep holding still.", 12f, "Keep HOLDING STILL", ticks: true),
            new CueStep("b_roi_move", "Move your head slowly. Keep the board in view.", 12f, "Move SLOWLY, board in view", metronome: 1f),
            new CueStep("b_r15_move", "Keep moving slowly.", 12f, "Keep moving SLOWLY", metronome: 1f),
            new CueStep("l_ready", "Now plate L, where the board was. Tilt it about forty five degrees towards you, so you see both faces.", 14f,
                        "Plate L tilted 45 degrees, both faces visible", ticks: true),
            new CueStep("l_roi_still", "Look at plate L and hold still.", 12f, "Plate L: HOLD STILL", ticks: true),
            new CueStep("l_r15_still", "Keep holding still.", 12f, "Keep HOLDING STILL", ticks: true),
            new CueStep("l_roi_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("l_r15_move", "Keep moving slowly.", 12f, "Keep moving SLOWLY", metronome: 1f),
        };

        /// <summary>The plan of a step: T6.6 steps by their thread count; T6.7 and T6.8 steps by their prefix and mode.</summary>
        public static Plan PlanFor(string variant, string stepId)
        {
            stepId = stepId ?? "";
            if (variant == "lean")
            {
                string t = stepId.StartsWith("l_") ? "plate_L" : "board25";
                return new Plan(stepId.Contains("_r15_") ? "roi15" : "roi", 1, NiceRoi, t);
            }
            if (variant != "budget") return new Plan("full", ThreadsFor(stepId), 0, "");
            string target = stepId.StartsWith("l_") ? "plate_L" : stepId.StartsWith("b_") ? "board25" : "plate_A";
            if (stepId.Contains("_full_")) return new Plan("full", 2, 0, target);
            if (stepId.Contains("_cap_")) return new Plan("cap", 2, 0, target);
            if (stepId == "a_ready") return new Plan("full", 2, 0, target);
            return new Plan("roi", 1, NiceRoi, target);   // roi steps and the l_/b_ ready steps (the target is being placed)
        }

        /// <summary>The region processed next: the projected board box grown by the margin, clipped to the frame.</summary>
        public static int[] RoiAround(double x0, double y0, double x1, double y1, int w, int h)
        {
            if (double.IsNaN(x0) || double.IsNaN(y0) || double.IsNaN(x1) || double.IsNaN(y1)) return null;
            float m = Mathf.Max(RoiMarginMinPx, RoiMarginFrac * (float)Math.Max(x1 - x0, y1 - y0));
            int ax = Mathf.Clamp(Mathf.FloorToInt((float)x0 - m), 0, w);
            int ay = Mathf.Clamp(Mathf.FloorToInt((float)y0 - m), 0, h);
            int bx = Mathf.Clamp(Mathf.CeilToInt((float)x1 + m), 0, w);
            int by = Mathf.Clamp(Mathf.CeilToInt((float)y1 + m), 0, h);
            if (bx - ax < 32 || by - ay < 32) return null;
            return new[] { ax, ay, bx - ax, by - ay };
        }

        struct Done
        {
            public long Seq, Skipped, CapSkipped, Arrival, Start, End;
            public ulong CamTs;
            public int Threads, Nice;
            public bool Accepted;
            public string Step, Ids, Mode, Target;
            public ArucoNative.Result R;
            public double[] Prior;
            public string Corners;
            public byte[] SaveCopy;
            public int W, H, Stride;
            public bool HeadOk;
            public float[] Head;
        }

        string m_Variant = "detect";
        ProbeRun m_Run;
        FrameSet m_Set;
        XRDisplaySubsystem m_Display;
        string m_Step = "";
        volatile int m_ThreadsWanted = 2;
        volatile bool m_Live, m_Stop;
        Thread m_Worker;
        readonly object m_Lock = new object();
        readonly AutoResetEvent m_Signal = new AutoResetEvent(false);
        readonly ConcurrentQueue<Done> m_Done = new ConcurrentQueue<Done>();
        // frame hand-over: main writes m_Back, swaps it with m_Front under the lock; the worker swaps m_Front with m_Work
        byte[] m_Back, m_Front, m_Work;
        float[] m_HeadBack = new float[16], m_HeadFront = new float[16];
        bool m_HasNew, m_HeadBackOk, m_HeadFrontOk;
        long m_FrontSeq, m_FrontArrival, m_Skipped, m_CapSkipped, m_Seq, m_Processed, m_EyeFrames;
        ulong m_FrontTs, m_LastTs;
        int m_W, m_H, m_Stride;
        string m_FrontStep = "";
        bool m_Capturing;
        string m_BenchStatus = "", m_Error = "";
        double m_LastTotal;
        double m_EyeMs;
        string m_LastMode = "";

        public static IEnumerator Run(ProbeRun run, Action<string> status, Action<string, float, float, string> show, string variant = "detect")
        {
            var go = new GameObject("EyeLab DetectProbe");
            var p = go.AddComponent<DetectProbe>();
            p.m_Run = run;
            p.m_Variant = variant == "budget" || variant == "lean" ? variant : "detect";
            run.Meta("variant", p.m_Variant);
            string info;
            try { info = ArucoNative.Info(); }
            catch (Exception e)
            {
                run.Meta("plugin_error", e.GetType().Name + " " + e.Message);
                status("Detection: the native plugin did not load: " + e.Message);
                Debug.LogError("[EyeLab probe] eyelab_aruco: " + e);
                Destroy(go);
                yield break;
            }
            run.Meta("plugin", info);
            Debug.Log("[EyeLab probe] " + info);
            p.m_Set = LoadSet(run);
            p.m_Display = XREALDisplay();
            if (p.m_Variant != "detect")
                run.Meta("budget", $"cap {CapHz} Hz; roi margin {RoiMarginFrac} x box, min {RoiMarginMinPx} px; box/prior max age " +
                                   $"{RoiMaxAgeNs / 1e6} ms; roi nice {NiceRoi}; trusted: non-planar, or ratio >= " +
                                   $"{p.LockCfg.min_ambiguity_ratio}, or <= {p.LockCfg.ambiguity_min_separation_deg} deg apart (no prior); and rms <= {p.LockCfg.max_acquire_reproj_rms_px} px; next region = projected board joined with detected target markers");

            // Part 1: bench on the pushed frames, on a worker thread so the glasses keep drawing.
            if (p.m_Set != null && p.m_Set.frames != null && p.m_Set.frames.Length > 0)
            {
                var bench = new Thread(p.Bench) { IsBackground = true, Name = "eyelab-b3a-bench" };
                bench.Start();
                while (bench.IsAlive)
                {
                    status("Bench: " + p.m_BenchStatus);
                    show?.Invoke("Detection bench running, please wait", 0f, 0f, "bench");
                    yield return null;
                }
                run.Meta("bench_status", p.m_BenchStatus);
            }
            else
            {
                status("No frame set in files/b3a: bench skipped");
                run.Meta("bench_status", "no set");
            }

            // Part 2: live on the Eye.
            p.StartLive();
            float nextStatus = 0f;
            List<CueStep> cues = p.m_Variant == "lean" ? LeanCues() : p.m_Variant == "budget" ? BudgetCues() : DetectCues();
            yield return CuePlayer.Ensure().Play(run, cues, (text, left, frac, counter) =>
            {
                show?.Invoke(text, left, frac, counter);
                if (Time.realtimeSinceStartup < nextStatus) return;
                nextStatus = Time.realtimeSinceStartup + 0.5f;
                status($"{counter}: {text} | {left:F0} s | Eye {p.m_EyeFrames}, done {p.m_Processed}, skipped {p.m_Skipped} | " +
                       $"{p.m_LastMode} {p.m_LastTotal:F1} ms | fps {1f / Mathf.Max(Time.smoothDeltaTime, 1e-4f):F0}");
            }, null, step =>
            {
                p.m_Step = step.Id;
                p.m_ThreadsWanted = PlanFor(p.m_Variant, step.Id).Threads;
            });
            p.StopLive();
            while (p.m_Done.TryDequeue(out Done d)) p.Write(d);
            run.Meta("eye_frames", p.m_EyeFrames.ToString(CultureInfo.InvariantCulture));
            run.Meta("processed", p.m_Processed.ToString(CultureInfo.InvariantCulture));
            run.Meta("skipped", p.m_Skipped.ToString(CultureInfo.InvariantCulture));
            run.Meta("cap_skipped", p.m_CapSkipped.ToString(CultureInfo.InvariantCulture));
            if (p.m_Error.Length > 0) run.Meta("worker_error", p.m_Error);
            Destroy(go);
        }

        PoseLockSettings LockCfg => m_Set?.pose_lock ?? new PoseLockSettings();

        static FrameSet LoadSet(ProbeRun run)
        {
            string path = Path.Combine(SetDir, "set.json");
            if (!System.IO.File.Exists(path)) { run.Meta("set", "missing " + path); return null; }
            try
            {
                var s = JsonUtility.FromJson<FrameSet>(System.IO.File.ReadAllText(path));
                var names = new StringBuilder();
                if (s.targets != null) foreach (Target t in s.targets) names.Append(names.Length > 0 ? " " : "").Append(t.name).Append('(').Append(t.ids?.Length ?? 0).Append(')');
                run.Meta("set", $"{s.schema}; board {s.board_name}; calibration {s.calibration}; frames {s.frames?.Length ?? 0}; markers {s.ids?.Length ?? 0}; " +
                                $"targets {(names.Length > 0 ? names.ToString() : "none")}");
                return s;
            }
            catch (Exception e)
            {
                run.Meta("set", "unreadable: " + e.Message);
                return null;
            }
        }

        static XRDisplaySubsystem XREALDisplay()
        {
            var list = new List<XRDisplaySubsystem>();
            SubsystemManager.GetSubsystems(list);
            foreach (XRDisplaySubsystem d in list) if (d.running) return d;
            return null;
        }

        ArucoNative MakeDetector()
        {
            var det = new ArucoNative(m_Set?.tuning != null && m_Set.tuning.Length >= ArucoNative.TuningCount ? m_Set.tuning : null,
                                      m_Set?.dictionary ?? ArucoNative.DictionaryArUco4x4_50);
            if (m_Set?.ids != null && m_Set.ids.Length > 0) det.SetBoard(m_Set.ids, m_Set.corners);
            return det;
        }

        /// <summary>Sets the target's board on <paramref name="det"/>; false (board unchanged) when the set has no such target.</summary>
        bool SetTarget(ArucoNative det, string name)
        {
            if (string.IsNullOrEmpty(name) || m_Set == null) return false;
            if (m_Set.targets != null)
                foreach (Target t in m_Set.targets)
                    if (t.name == name && t.ids != null && t.ids.Length > 0) { det.SetBoard(t.ids, t.corners); return true; }
            if (name == "plate_A" && m_Set.ids != null && m_Set.ids.Length > 0) { det.SetBoard(m_Set.ids, m_Set.corners); return true; }
            return false;
        }

        double[] K => m_Set?.K != null && m_Set.K.Length == 9 ? m_Set.K : null;

        int[] TargetIds(string name)
        {
            if (m_Set?.targets != null)
                foreach (Target t in m_Set.targets)
                    if (t.name == name) return t.ids;
            return m_Set?.ids;
        }

        /// <summary>The box of the projected board (when posed) joined with the target's detected marker corners.</summary>
        bool BoardBox(ArucoNative det, int n, in ArucoNative.Result r, string target, out double x0, out double y0, out double x1, out double y1)
        {
            x0 = y0 = double.PositiveInfinity;
            x1 = y1 = double.NegativeInfinity;
            if (r.PoseOk == 1 && !double.IsNaN(r.ProjX0))
            {
                x0 = r.ProjX0; y0 = r.ProjY0; x1 = r.ProjX1; y1 = r.ProjY1;
            }
            int[] ids = TargetIds(target);
            for (int i = 0; i < Math.Min(Math.Max(n, 0), ArucoNative.MaxMarkers); i++)
            {
                if (ids == null || Array.IndexOf(ids, det.Ids[i]) < 0) continue;
                for (int c = 0; c < 4; c++)
                {
                    double x = det.Corners[i * 8 + c * 2], y = det.Corners[i * 8 + c * 2 + 1];
                    x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
                }
            }
            return !double.IsInfinity(x0) && x1 > x0 && y1 > y0;
        }

        /// <summary>
        /// PoseLock's acquisition rule for a planar solve: trusted when the two IPPE solutions are well apart in error
        /// or close in rotation; a non-planar solve (plate L with both faces) is always trusted. No prior: on run 2's
        /// corners (T6.7 replay) most wrong poses were 2-marker poses where neither IPPE solution was right, and a prior
        /// chain spread them (wrong poses 56 → 147 on t2_still); the plugin keeps the prior for PoseLock's motion gates later.
        /// </summary>
        bool Trusted(in ArucoNative.Result r)
        {
            if (r.PoseOk != 1) return false;
            if (!(r.RmsPx <= LockCfg.max_acquire_reproj_rms_px)) return false;   // T6.8: run 1's plate L solves up to 2198 px
            if (r.Planar != 1) return true;
            if (double.IsNaN(r.AmbiguityRatio)) return true;   // one IPPE solution only
            return r.AmbiguityRatio >= LockCfg.min_ambiguity_ratio || r.AmbiguityDeg <= LockCfg.ambiguity_min_separation_deg;
        }

        // ---- bench --------------------------------------------------------------------------------------------------

        void Bench()
        {
            var ci = CultureInfo.InvariantCulture;
            var rows = new StringBuilder();
            var cornerRows = new StringBuilder();
            try
            {
                using (ArucoNative det = MakeDetector())
                {
                    for (int f = 0; f < m_Set.frames.Length; f++)
                    {
                        string name = m_Set.frames[f];
                        byte[] y = ReadPgm(Path.Combine(SetDir, name), out int w, out int h);
                        if (y == null) { rows.Append(name).Append(",unreadable\n"); continue; }
                        int[] roi = null;
                        // T6.6: full frame at 1, 2, 4, 8 threads. T6.7: full at 1 and 2, then the ROI around the full
                        // frame's projected box at 1 thread (the live ROI mode), for its cost and its parity.
                        var modes = new List<(string mode, int threads)>();
                        if (m_Variant != "detect") { modes.Add(("full", 1)); modes.Add(("full", 2)); modes.Add(("roi", 1)); }
                        else foreach (int t in DetectBenchThreads) modes.Add(("full", t));
                        foreach ((string mode, int tm) in modes)
                        {
                            if (mode == "roi" && roi == null) continue;
                            int used = ArucoNative.SetThreads(tm);
                            int[] useRoi = mode == "roi" ? roi : null;
                            det.ProcessRoi(y, w, h, w, useRoi, K, m_Set.dist, null, out _);   // warm-up, not logged
                            for (int rep = 0; rep < BenchReps; rep++)
                            {
                                int n = det.ProcessRoi(y, w, h, w, useRoi, K, m_Set.dist, null, out ArucoNative.Result r);
                                rows.Append(name).Append(',').Append(tm.ToString(ci)).Append(',').Append(used.ToString(ci)).Append(',')
                                    .Append(rep.ToString(ci)).Append(',').Append(ResultCsv(r)).Append(',').Append(mode).Append('\n');
                                if (tm == 1 && rep == 0)
                                {
                                    string key = mode == "roi" ? name + "#roi" : name;
                                    for (int i = 0; i < Math.Min(n, ArucoNative.MaxMarkers); i++)
                                        cornerRows.Append(key).Append(',').Append(CornerCsv(det, i)).Append('\n');
                                    if (mode == "full" && r.PoseOk == 1) roi = RoiAround(r.ProjX0, r.ProjY0, r.ProjX1, r.ProjY1, w, h);
                                }
                            }
                        }
                        m_BenchStatus = $"frame {f + 1}/{m_Set.frames.Length}";
                    }
                }
                m_BenchStatus += " done";
            }
            catch (Exception e)
            {
                m_BenchStatus += " error " + e.GetType().Name + ": " + e.Message;
            }
            finally
            {
                ArucoNative.SetThreads(1);
            }
            m_Run.Text("bench.csv", BenchHeader + "\n" + rows);
            m_Run.Text("bench_corners.csv", "frame,id,x0,y0,x1,y1,x2,y2,x3,y3\n" + cornerRows);
        }

        static string ResultCsv(in ArucoNative.Result r)
        {
            var ci = CultureInfo.InvariantCulture;
            string d(double v) => double.IsNaN(v) ? "" : v.ToString("R", ci);
            return string.Join(",", r.Markers.ToString(ci), r.Rejected.ToString(ci), r.BoardMarkers.ToString(ci), r.PoseOk.ToString(ci),
                r.Planar.ToString(ci), r.Points.ToString(ci), d(r.ClaheMs), d(r.DetectMs), d(r.PoseMs), d(r.TotalMs),
                r.PoseOk == 1 ? d(r.R0) : "", r.PoseOk == 1 ? d(r.R1) : "", r.PoseOk == 1 ? d(r.R2) : "",
                r.PoseOk == 1 ? d(r.T0) : "", r.PoseOk == 1 ? d(r.T1) : "", r.PoseOk == 1 ? d(r.T2) : "",
                d(r.RmsPx), d(r.AmbiguityRatio), d(r.AmbiguityDeg),
                r.RoiX.ToString(ci), r.RoiY.ToString(ci), r.RoiW.ToString(ci), r.RoiH.ToString(ci), r.Solution.ToString(ci), r.PriorUsed.ToString(ci));
        }

        static string CornerCsv(ArucoNative det, int i)
        {
            var ci = CultureInfo.InvariantCulture;
            var b = new StringBuilder();
            b.Append(det.Ids[i].ToString(ci));
            for (int k = 0; k < 8; k++) b.Append(',').Append(det.Corners[i * 8 + k].ToString("R", ci));
            return b.ToString();
        }

        static byte[] ReadPgm(string path, out int w, out int h)
        {
            w = h = 0;
            if (!System.IO.File.Exists(path)) return null;
            byte[] all = System.IO.File.ReadAllBytes(path);
            // "P5" ws width ws height ws maxval single-ws data
            int pos = 0;
            string Token()
            {
                while (pos < all.Length && (char.IsWhiteSpace((char)all[pos]) || all[pos] == '#'))
                {
                    if (all[pos] == '#') while (pos < all.Length && all[pos] != '\n') pos++;
                    else pos++;
                }
                int s = pos;
                while (pos < all.Length && !char.IsWhiteSpace((char)all[pos])) pos++;
                return Encoding.ASCII.GetString(all, s, pos - s);
            }
            if (Token() != "P5") return null;
            w = int.Parse(Token(), CultureInfo.InvariantCulture);
            h = int.Parse(Token(), CultureInfo.InvariantCulture);
            if (int.Parse(Token(), CultureInfo.InvariantCulture) != 255) return null;
            pos++;
            if (all.Length - pos < w * h) return null;
            var y = new byte[w * h];
            Buffer.BlockCopy(all, pos, y, 0, y.Length);
            return y;
        }

        // ---- live ---------------------------------------------------------------------------------------------------

        void StartLive()
        {
            ulong id = XREALPlugin.StartRGBCameraDataCapture();
            m_Capturing = id != ulong.MaxValue;
            m_Run.Meta("eye_start_capture", id.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("calibration_resolution", "1280x720 (set.json)");
            m_Run.Meta("eye_rows", "bottom-up from the SDK, reversed to upright on copy");
            m_Run.Meta("head_pose", "GetHeadPoseAtTime(cam_ts) per Eye frame, 4x4 column-major (T6.2), head.csv");
            m_Live = true;
            m_Worker = new Thread(Worker) { IsBackground = true, Name = "eyelab-b3a-live" };
            m_Worker.Start();
        }

        void StopLive()
        {
            m_Live = false;
            m_Stop = true;
            m_Signal.Set();
            m_Worker?.Join(2000);
            if (m_Capturing) XREALPlugin.StopRGBCameraDataCapture();
            m_Capturing = false;
        }

        void Update()
        {
            if (!m_Live) return;
            var ci = CultureInfo.InvariantCulture;
            int present = -1;
            if (m_Display != null && !m_Display.TryGetFramePresentCount(out present)) present = -1;
            // the main thread's Eye work (acquire, upright copy, head pose, hand-over) is timed, so its share of the frame is known
            m_EyeMs = 0;
            if (m_Capturing)
            {
                long t0 = ProbeClocks.MonoNs;
                if (Acquire()) m_EyeMs = (ProbeClocks.MonoNs - t0) / 1e6;
            }
            m_Run.Row(FramesFile, FramesHeader, $"{m_Step},{m_ThreadsWanted},{Time.frameCount},{(Time.unscaledDeltaTime * 1000f).ToString("R", ci)}," +
                                                $"{present},{ProbeClocks.MonoNs},{m_EyeMs.ToString("F3", ci)}");
            while (m_Done.TryDequeue(out Done d)) Write(d);
        }

        /// <returns>true when a new frame was copied and handed to the worker.</returns>
        bool Acquire()
        {
            int newest = -1, acquired = 0;
            ulong newestTs = 0;
            for (int i = 0; i < 16; i++)
            {
                ulong probeTs = 0;
                if (!XREALPlugin.TryGetRGBCameraFrame(ref probeTs)) break;
                int handle = 0;
                Vector2Int res = default;
                ulong ts = 0;
                if (!XREALPlugin.TryAcquireLatestImage(ref handle, ref res, ref ts)) break;
                if (acquired > 0 && ts == newestTs) { XREALPlugin.DisposeRGBCameraDataHandle(handle); break; }
                if (newest >= 0) XREALPlugin.DisposeRGBCameraDataHandle(newest);
                newest = handle;
                newestTs = ts;
                acquired++;
            }
            if (acquired == 0) return false;
            long arrival = ProbeClocks.MonoNs;
            try
            {
                if (newestTs == m_LastTs) return false;
                m_LastTs = newestTs;
                if (!XREALPlugin.TryGetRGBCameraDataPlane(newest, 0, out IntPtr data, out Vector2Int size) || data == IntPtr.Zero) return false;
                int bytes = size.x * size.y;
                if (m_Back == null || m_Back.Length != bytes) m_Back = new byte[bytes];
                // rows are stored bottom-up: copy them in reverse so row 0 is the top of the image
                for (int row = 0; row < size.y; row++)
                    Marshal.Copy(IntPtr.Add(data, (size.y - 1 - row) * size.x), m_Back, row * size.x, size.x);
                m_HeadBackOk = ProbeClocks.HeadPoseAt(newestTs, m_HeadBack);
                m_EyeFrames++;
                lock (m_Lock)
                {
                    if (m_HasNew) m_Skipped++;
                    (m_Front, m_Back) = (m_Back, m_Front);
                    (m_HeadFront, m_HeadBack) = (m_HeadBack, m_HeadFront);
                    m_HeadFrontOk = m_HeadBackOk;
                    m_HasNew = true;
                    m_FrontSeq = m_Seq++;
                    m_FrontTs = newestTs;
                    m_FrontArrival = arrival;
                    m_FrontStep = m_Step;
                    m_W = size.x; m_H = size.y; m_Stride = size.x;
                }
                m_Signal.Set();
                return true;
            }
            finally
            {
                XREALPlugin.DisposeRGBCameraDataHandle(newest);
            }
        }

        void Worker()
        {
            ArucoNative det = null;
            int threadsSet = 0, niceSet = 0;
            string targetSet = null;
            long lastStart = 0;
            // ROI state, per target
            double bx0 = double.NaN, by0 = double.NaN, bx1 = double.NaN, by1 = double.NaN;
            ulong boxTs = 0;
            try
            {
                det = MakeDetector();
                while (!m_Stop)
                {
                    m_Signal.WaitOne(200);
                    Done d;
                    lock (m_Lock)
                    {
                        if (!m_HasNew) continue;
                        (m_Work, m_Front) = (m_Front, m_Work);
                        m_HasNew = false;
                        d = new Done { Seq = m_FrontSeq, CamTs = m_FrontTs, Arrival = m_FrontArrival, Step = m_FrontStep, Skipped = m_Skipped,
                                       W = m_W, H = m_H, Stride = m_Stride, HeadOk = m_HeadFrontOk, Head = (float[])m_HeadFront.Clone() };
                    }
                    Plan plan = PlanFor(m_Variant, d.Step);
                    if ((plan.Mode == "cap" || plan.Mode == "roi15") && lastStart != 0 && ProbeClocks.MonoNs - lastStart < 1e9 / CapHz)
                    {
                        Interlocked.Increment(ref m_CapSkipped);
                        continue;
                    }
                    if (plan.Threads != threadsSet) { ArucoNative.SetThreads(plan.Threads); threadsSet = plan.Threads; }
                    if (plan.Nice != niceSet) { int now = ArucoNative.SetNice(plan.Nice); niceSet = plan.Nice; if (now == -100) m_Error = "setpriority failed"; }
                    if (m_Variant != "detect" && plan.Target != targetSet)
                    {
                        SetTarget(det, plan.Target);
                        targetSet = plan.Target;
                        bx0 = by0 = bx1 = by1 = double.NaN;
                    }
                    int[] roi = null;
                    if (plan.Mode.StartsWith("roi") && d.CamTs - boxTs < RoiMaxAgeNs) roi = RoiAround(bx0, by0, bx1, by1, d.W, d.H);
                    d.Threads = plan.Threads;
                    d.Nice = plan.Nice;
                    d.Mode = plan.Mode;
                    d.Target = plan.Target;
                    d.Prior = null;
                    d.CapSkipped = Interlocked.Read(ref m_CapSkipped);
                    d.Start = ProbeClocks.MonoNs;
                    lastStart = d.Start;
                    int n = det.ProcessRoi(m_Work, d.W, d.H, d.Stride, roi, K, m_Set?.dist, null, out d.R);
                    d.End = ProbeClocks.MonoNs;
                    d.Accepted = Trusted(d.R);
                    // Next region: the board projected with the pose, joined with the target's detected markers (a
                    // wrong 2-marker pose can misplace the projection; the detected corners are where the markers are).
                    if (BoardBox(det, n, d.R, plan.Target, out double x0, out double y0, out double x1, out double y1))
                    {
                        bx0 = x0; by0 = y0; bx1 = x1; by1 = y1;
                        boxTs = d.CamTs;
                    }
                    else if (roi != null)
                    {
                        bx0 = by0 = bx1 = by1 = double.NaN;   // lost inside the region: the next frame searches the whole frame
                    }
                    var ids = new StringBuilder();
                    var corners = new StringBuilder();
                    for (int i = 0; i < Math.Min(Math.Max(n, 0), ArucoNative.MaxMarkers); i++)
                    {
                        if (i > 0) ids.Append(' ');
                        ids.Append(det.Ids[i].ToString(CultureInfo.InvariantCulture));
                        corners.Append(d.Seq.ToString(CultureInfo.InvariantCulture)).Append(',').Append(CornerCsv(det, i)).Append('\n');
                    }
                    d.Ids = ids.ToString();
                    d.Corners = corners.ToString();
                    if (m_Processed % SaveEvery == 0) d.SaveCopy = (byte[])m_Work.Clone();
                    m_LastTotal = d.R.TotalMs;
                    m_LastMode = plan.Mode;
                    Interlocked.Increment(ref m_Processed);
                    m_Done.Enqueue(d);
                }
            }
            catch (Exception e)
            {
                m_Error = e.GetType().Name + ": " + e.Message;
            }
            finally
            {
                det?.Dispose();
                ArucoNative.SetThreads(1);
                ArucoNative.SetNice(0);
            }
        }

        void Write(Done d)
        {
            var ci = CultureInfo.InvariantCulture;
            string saved = "";
            if (d.SaveCopy != null)
            {
                string name = $"frames/y_{d.Seq:D6}.pgm";
                byte[] copy = d.SaveCopy;
                int w = d.W, h = d.H;
                if (m_Run.Bytes(name, () => Pgm(copy, w, h))) saved = name;
            }
            string pr(int k) => d.Prior != null ? d.Prior[k].ToString("R", ci) : "";
            m_Run.Row(LiveFile, LiveHeader, string.Join(",", d.Seq.ToString(ci), d.Step, d.Threads.ToString(ci), d.CamTs.ToString(ci),
                d.Arrival.ToString(ci), d.Start.ToString(ci), d.End.ToString(ci), d.Skipped.ToString(ci), ResultCsv(d.R), d.Ids, saved,
                d.Mode, d.Target, d.Nice.ToString(ci), d.Accepted ? "1" : "0", pr(0), pr(1), pr(2), d.CapSkipped.ToString(ci)));
            if (d.Corners.Length > 0) m_Run.Row("live_corners.csv", CornerHeader, d.Corners.TrimEnd('\n'));
            var b = new StringBuilder();
            b.Append(d.Seq.ToString(ci)).Append(',').Append(d.CamTs.ToString(ci)).Append(',').Append(d.HeadOk ? '1' : '0');
            for (int i = 0; i < 16; i++) { b.Append(','); if (d.HeadOk) b.Append(d.Head[i].ToString("R", ci)); }
            m_Run.Row(HeadFile, HeadHeader, b.ToString());
        }

        static byte[] Pgm(byte[] y, int w, int h)
        {
            byte[] head = Encoding.ASCII.GetBytes($"P5\n{w} {h}\n255\n");
            var outp = new byte[head.Length + w * h];
            Buffer.BlockCopy(head, 0, outp, 0, head.Length);
            Buffer.BlockCopy(y, 0, outp, head.Length, w * h);
            return outp;
        }

        void OnDestroy()
        {
            m_Stop = true;
            m_Signal.Set();
            if (m_Capturing) XREALPlugin.StopRGBCameraDataCapture();
        }
    }
}
