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
    /// T6.6, gate B3a (ADR-0004 option A2): ArUco detection and the board pose on the Beam Pro's CPU with the native
    /// plugin, while the glasses draw. Two parts in one run:
    /// <list type="number">
    /// <item><b>Bench</b> (no wearing needed): every frame of the set the agent pushed to <c>files/b3a/</c> (<c>set.json</c>,
    /// PGM frames made by <c>python -m xreal_gates b3a-set</c>) is processed with 1 thread and with OpenCV's default
    /// threads, 5 times each, on a worker thread. <c>bench.csv</c> and <c>bench_corners.csv</c> go to the parity test
    /// against the Python pipeline on the same frames.</item>
    /// <item><b>Live</b>: every new Eye frame is copied once on the main thread and processed on a worker thread (newest
    /// wins; a frame that arrives while the worker is busy replaces the waiting one and is counted as skipped).
    /// <c>live.csv</c> has the times and the pose per processed frame, <c>frames.csv</c> every Unity frame's time, so
    /// the cost to the render loop is visible; every 10th processed frame is saved at full resolution for the parity
    /// test on real frames.</item>
    /// </list>
    /// All worker timestamps are CLOCK_MONOTONIC, which is the HMD clock of <c>cam_ts</c> (T6.0 BS), so the age of a
    /// pose at the moment it is ready is <c>end_mono_ns − cam_ts</c>.
    /// </summary>
    public sealed class DetectProbe : MonoBehaviour
    {
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
        }

        public static string SetDir => Path.Combine(Application.persistentDataPath, "b3a");

        const string LiveFile = "live.csv";
        const string LiveHeader = "seq,step,threads,cam_ts,arrival_mono_ns,start_mono_ns,end_mono_ns,skipped,markers,rejected,board_markers," +
                                  "pose_ok,planar,points,clahe_ms,detect_ms,pose_ms,total_ms,r0,r1,r2,t0,t1,t2,rms_px,amb_ratio,amb_deg,ids,saved";
        const string CornerHeader = "seq,id,x0,y0,x1,y1,x2,y2,x3,y3";
        const string BenchHeader = "frame,threads_req,threads,rep,markers,rejected,board_markers,pose_ok,planar,points,clahe_ms,detect_ms,pose_ms,total_ms," +
                                   "r0,r1,r2,t0,t1,t2,rms_px,amb_ratio,amb_deg";
        const string FramesFile = "frames.csv";
        const string FramesHeader = "step,threads,unity_frame,dt_ms,present,mono_ns";
        const int SaveEvery = 10;
        const int BenchReps = 5;

        public static List<CueStep> Cues() => new List<CueStep>
        {
            new CueStep("ready", "Detection test. Stand about half a metre from plate A, so it is straight ahead of you.", 8f,
                        "Plate A straight ahead, about 50 cm", ticks: true),
            new CueStep("t1_still", "Look at the plate and hold still.", 15f, "Look at the plate: HOLD STILL"),
            new CueStep("t1_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("t1_away", "Look away from the plate.", 6f, "Look AWAY from the plate"),
            new CueStep("tn_still", "Back to the plate. Hold still.", 15f, "Back to the plate: HOLD STILL"),
            new CueStep("tn_move", "Move your head slowly again.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("tn_away", "Look away again.", 6f, "Look AWAY from the plate", ticks: true),
        };

        struct Done
        {
            public long Seq, Skipped, Arrival, Start, End;
            public ulong CamTs;
            public int Threads;
            public string Step, Ids;
            public ArucoNative.Result R;
            public string Corners;
            public byte[] SaveCopy;
            public int W, H, Stride;
        }

        ProbeRun m_Run;
        FrameSet m_Set;
        XRDisplaySubsystem m_Display;
        string m_Step = "";
        volatile int m_ThreadsWanted = 1;
        volatile bool m_Live, m_Stop;
        Thread m_Worker;
        readonly object m_Lock = new object();
        readonly AutoResetEvent m_Signal = new AutoResetEvent(false);
        readonly ConcurrentQueue<Done> m_Done = new ConcurrentQueue<Done>();
        // frame hand-over: main writes m_Back, swaps it with m_Front under the lock; the worker swaps m_Front with m_Work
        byte[] m_Back, m_Front, m_Work;
        bool m_HasNew;
        long m_FrontSeq, m_FrontArrival, m_Skipped, m_Seq, m_Processed, m_EyeFrames;
        ulong m_FrontTs, m_LastTs;
        int m_W, m_H, m_Stride;
        string m_FrontStep = "";
        bool m_Capturing;
        string m_BenchStatus = "", m_Error = "";
        double m_LastTotal;

        public static IEnumerator Run(ProbeRun run, Action<string> status, Action<string, float, float, string> show)
        {
            var go = new GameObject("EyeLab DetectProbe");
            var p = go.AddComponent<DetectProbe>();
            p.m_Run = run;
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
            yield return CuePlayer.Ensure().Play(run, Cues(), (text, left, frac, counter) =>
            {
                show?.Invoke(text, left, frac, counter);
                if (Time.realtimeSinceStartup < nextStatus) return;
                nextStatus = Time.realtimeSinceStartup + 0.5f;
                status($"{counter}: {text} | {left:F0} s | Eye {p.m_EyeFrames}, done {p.m_Processed}, skipped {p.m_Skipped}, " +
                       $"last {p.m_LastTotal:F1} ms | fps {1f / Mathf.Max(Time.smoothDeltaTime, 1e-4f):F0}");
            }, null, step =>
            {
                p.m_Step = step.Id;
                p.m_ThreadsWanted = step.Id.StartsWith("tn_") ? -1 : 1;
            });
            p.StopLive();
            while (p.m_Done.TryDequeue(out Done d)) p.Write(d);
            run.Meta("eye_frames", p.m_EyeFrames.ToString(CultureInfo.InvariantCulture));
            run.Meta("processed", p.m_Processed.ToString(CultureInfo.InvariantCulture));
            run.Meta("skipped", p.m_Skipped.ToString(CultureInfo.InvariantCulture));
            if (p.m_Error.Length > 0) run.Meta("worker_error", p.m_Error);
            Destroy(go);
        }

        static FrameSet LoadSet(ProbeRun run)
        {
            string path = Path.Combine(SetDir, "set.json");
            if (!System.IO.File.Exists(path)) { run.Meta("set", "missing " + path); return null; }
            try
            {
                var s = JsonUtility.FromJson<FrameSet>(System.IO.File.ReadAllText(path));
                run.Meta("set", $"{s.schema}; board {s.board_name}; calibration {s.calibration}; frames {s.frames?.Length ?? 0}; markers {s.ids?.Length ?? 0}");
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

        double[] K => m_Set?.K != null && m_Set.K.Length == 9 ? m_Set.K : null;

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
                    int[] threadModes = { 1, -1 };
                    for (int f = 0; f < m_Set.frames.Length; f++)
                    {
                        string name = m_Set.frames[f];
                        byte[] y = ReadPgm(Path.Combine(SetDir, name), out int w, out int h);
                        if (y == null) { rows.Append(name).Append(",unreadable\n"); continue; }
                        foreach (int tm in threadModes)
                        {
                            int used = ArucoNative.SetThreads(tm);
                            det.Process(y, w, h, w, K, m_Set.dist, out _);   // warm-up, not logged
                            for (int rep = 0; rep < BenchReps; rep++)
                            {
                                int n = det.Process(y, w, h, w, K, m_Set.dist, out ArucoNative.Result r);
                                rows.Append(name).Append(',').Append(tm.ToString(ci)).Append(',').Append(used.ToString(ci)).Append(',')
                                    .Append(rep.ToString(ci)).Append(',').Append(ResultCsv(r)).Append('\n');
                                if (tm == 1 && rep == 0)
                                    for (int i = 0; i < Math.Min(n, ArucoNative.MaxMarkers); i++)
                                        cornerRows.Append(name).Append(',').Append(CornerCsv(det, i)).Append('\n');
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
            // header written by the probe run's writer before the rows
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
                d(r.RmsPx), d(r.AmbiguityRatio), d(r.AmbiguityDeg));
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
            m_Run.Row(FramesFile, FramesHeader, $"{m_Step},{m_ThreadsWanted},{Time.frameCount},{(Time.unscaledDeltaTime * 1000f).ToString("R", ci)}," +
                                                $"{present},{ProbeClocks.MonoNs}");
            while (m_Done.TryDequeue(out Done d)) Write(d);
            if (m_Capturing) Acquire();
        }

        void Acquire()
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
            if (acquired == 0) return;
            long arrival = ProbeClocks.MonoNs;
            try
            {
                if (newestTs == m_LastTs) return;
                m_LastTs = newestTs;
                if (!XREALPlugin.TryGetRGBCameraDataPlane(newest, 0, out IntPtr data, out Vector2Int size) || data == IntPtr.Zero) return;
                int bytes = size.x * size.y;
                if (m_Back == null || m_Back.Length != bytes) m_Back = new byte[bytes];
                Marshal.Copy(data, m_Back, 0, bytes);
                m_EyeFrames++;
                lock (m_Lock)
                {
                    if (m_HasNew) m_Skipped++;
                    (m_Front, m_Back) = (m_Back, m_Front);
                    m_HasNew = true;
                    m_FrontSeq = m_Seq++;
                    m_FrontTs = newestTs;
                    m_FrontArrival = arrival;
                    m_FrontStep = m_Step;
                    m_W = size.x; m_H = size.y; m_Stride = size.x;
                }
                m_Signal.Set();
            }
            finally
            {
                XREALPlugin.DisposeRGBCameraDataHandle(newest);
            }
        }

        void Worker()
        {
            ArucoNative det = null;
            int threadsSet = 0;
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
                                       W = m_W, H = m_H, Stride = m_Stride };
                    }
                    int want = m_ThreadsWanted;
                    if (want != threadsSet) { ArucoNative.SetThreads(want); threadsSet = want; }
                    d.Threads = want;
                    d.Start = ProbeClocks.MonoNs;
                    int n = det.Process(m_Work, d.W, d.H, d.Stride, K, m_Set?.dist, out d.R);
                    d.End = ProbeClocks.MonoNs;
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
            m_Run.Row(LiveFile, LiveHeader, string.Join(",", d.Seq.ToString(ci), d.Step, d.Threads.ToString(ci), d.CamTs.ToString(ci),
                d.Arrival.ToString(ci), d.Start.ToString(ci), d.End.ToString(ci), d.Skipped.ToString(ci), ResultCsv(d.R), d.Ids, saved));
            if (d.Corners.Length > 0) m_Run.Row("live_corners.csv", CornerHeader, d.Corners.TrimEnd('\n'));
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
