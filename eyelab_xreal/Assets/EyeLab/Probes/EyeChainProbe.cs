using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.1 (gate B1): the Eye frame chain on the Beam Pro. Acquires every Eye frame through the public
    /// <see cref="XREALPlugin"/> capture API (the same draining loop as the PC gates' DrainingCameraSource:
    /// acquire until nothing newer is queued, keep the newest, count the rest) and logs, per kept frame,
    /// <c>cam_ts</c> next to every clock read at arrival (<see cref="ProbeClocks"/>). It answers:
    /// <list type="bullet">
    /// <item>rate, resolution, plane layout and row order of the Android stream;</item>
    /// <item>the clock domain and unit of <c>cam_ts</c> (which clock gives a small, stable, positive age);</item>
    /// <item>whether a decoder queue exists on Android (age growing over minutes, drained frames > 0);</item>
    /// <item>with <see cref="TimecodeScreen"/>, exposure → arrival through the optical timecode.</item>
    /// </list>
    /// No detection runs here: the frames that are saved are decoded offline, so the measured chain is the
    /// camera path alone. The arrival time is quantised to Update (one display frame at most).
    /// </summary>
    public sealed class EyeChainProbe : MonoBehaviour
    {
        const string File = "eye_frames.csv";
        const string Header = "seq,cam_ts,res_w,res_h,plane_w,plane_h,drained,unity_frame," + ProbeClocks.Sample.Header + ",saved";
        const int MaxAcquirePerUpdate = 16;

        ProbeRun m_Run;
        float m_Seconds;
        int m_SaveEvery, m_Scale;
        float m_SnapshotEvery;
        float m_T0, m_NextSnapshot;
        bool m_Capturing;
        long m_Seq, m_Drained, m_Saved, m_SameTs;
        ulong m_LastTs;
        Action m_OnDone;
        byte[] m_Copy;
        bool m_PairPoses;
        readonly float[] m_PoseCam = new float[16], m_PoseArr = new float[16];
        const string PoseFile = "eye_pose.csv";
        static readonly string PoseHeader = "seq,cam_ts,arrival_hmd_ns,cam_ok," + Cols("c") + ",arr_ok," + Cols("a");

        static string Cols(string p)
        {
            var b = new StringBuilder();
            for (int i = 0; i < 16; i++) { if (i > 0) b.Append(','); b.Append(p).Append(i); }
            return b.ToString();
        }

        public bool Running { get; private set; }
        public long Frames => m_Seq;
        public string Status => Running ? $"Eye frames {m_Seq}, drained {m_Drained}, saved {m_Saved}, writer queue {m_Run?.PendingJobs}" : "idle";

        /// <param name="saveEvery">Save every n-th kept Y plane (0 = none) as PGM, downscaled by <paramref name="scale"/>.</param>
        /// <param name="snapshotEvery">Seconds between full-resolution Y snapshots (orientation, calibration check).</param>
        /// <param name="pairPoses">Also log the native head pose at <c>cam_ts</c> and at arrival (T6.2, gate B2): <c>eye_pose.csv</c>.</param>
        public static EyeChainProbe Begin(ProbeRun run, float seconds, int saveEvery, int scale, float snapshotEvery, Action onDone,
                                          bool pairPoses = false)
        {
            var go = new GameObject("EyeLab EyeChainProbe");
            var p = go.AddComponent<EyeChainProbe>();
            p.m_Run = run;
            p.m_Seconds = seconds;
            p.m_SaveEvery = Mathf.Max(0, saveEvery);
            p.m_Scale = Mathf.Clamp(scale, 1, 8);
            p.m_SnapshotEvery = snapshotEvery;
            p.m_OnDone = onDone;
            p.m_PairPoses = pairPoses;
            p.StartChain();
            return p;
        }

        void StartChain()
        {
            m_Run.Meta("eye_seconds", ProbeRun.F(m_Seconds));
            m_Run.Meta("eye_save_every", m_SaveEvery.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("eye_save_scale", m_Scale.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("display_refresh_hz", DisplayRefresh());
            ulong id = EyeCamera.Ensure();   // 0.5.004: never stopped (the SDK's stop can hang the main thread)
            m_Capturing = id != ulong.MaxValue;
            m_Run.Meta("eye_start_capture", id.ToString(CultureInfo.InvariantCulture));
            m_T0 = Time.realtimeSinceStartup;
            m_NextSnapshot = m_T0;
            Running = true;
            if (!m_Capturing) Debug.LogWarning("[EyeLab probe] StartRGBCameraDataCapture failed");
        }

        static string DisplayRefresh()
        {
            var parts = new List<string> { "phone=" + ProbeRun.F(Screen.currentResolution.refreshRateRatio.value) };
            var displays = new List<XRDisplaySubsystem>();
            SubsystemManager.GetSubsystems(displays);
            foreach (XRDisplaySubsystem d in displays)
                if (d.running && d.TryGetDisplayRefreshRate(out float hz)) parts.Add("xr=" + ProbeRun.F(hz));
            return string.Join(";", parts);
        }

        void Update()
        {
            if (!Running) return;
            if (Time.realtimeSinceStartup - m_T0 >= m_Seconds) { Finish(); return; }
            if (!m_Capturing) return;

            int newest = -1, acquired = 0;
            ulong newestTs = 0;
            Vector2Int newestRes = default;
            for (int i = 0; i < MaxAcquirePerUpdate; i++)
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
                newestRes = res;
                acquired++;
            }
            if (acquired == 0) return;
            ProbeClocks.Sample arrival = ProbeClocks.Now();   // read before any copy, so the copy is not in the age

            try
            {
                if (newestTs == m_LastTs) { m_SameTs++; return; }
                m_LastTs = newestTs;
                m_Drained += acquired - 1;
                bool havePlane = XREALPlugin.TryGetRGBCameraDataPlane(newest, 0, out IntPtr data, out Vector2Int size) && data != IntPtr.Zero;
                string saved = "";
                if (havePlane)
                {
                    bool snapshot = m_SnapshotEvery > 0 && Time.realtimeSinceStartup >= m_NextSnapshot;
                    bool sample = m_SaveEvery > 0 && m_Seq % m_SaveEvery == 0;
                    if (snapshot || sample)
                    {
                        int bytes = size.x * size.y;
                        if (m_Copy == null || m_Copy.Length != bytes) m_Copy = new byte[bytes];
                        Marshal.Copy(data, m_Copy, 0, bytes);
                        byte[] copy = (byte[])m_Copy.Clone();
                        int stride = size.x, rows = size.y, w = Mathf.Min(newestRes.x > 0 ? newestRes.x : stride, stride);
                        long seq = m_Seq;
                        if (snapshot)
                        {
                            m_NextSnapshot = Time.realtimeSinceStartup + m_SnapshotEvery;
                            string name = $"snapshots/y_{seq:D6}.pgm";
                            if (m_Run.Bytes(name, () => Pgm(copy, stride, w, rows, 1))) { saved = name; m_Saved++; }
                        }
                        if (sample)
                        {
                            string name = $"frames/y_{seq:D6}.pgm";
                            int sc = m_Scale;
                            if (m_Run.Bytes(name, () => Pgm(copy, stride, w, rows, sc))) { saved = saved.Length > 0 ? saved + "|" + name : name; m_Saved++; }
                        }
                    }
                }
                m_Run.Row(File, Header, string.Join(",",
                    m_Seq.ToString(CultureInfo.InvariantCulture), newestTs.ToString(CultureInfo.InvariantCulture),
                    newestRes.x.ToString(CultureInfo.InvariantCulture), newestRes.y.ToString(CultureInfo.InvariantCulture),
                    havePlane ? size.x.ToString(CultureInfo.InvariantCulture) : "", havePlane ? size.y.ToString(CultureInfo.InvariantCulture) : "",
                    (acquired - 1).ToString(CultureInfo.InvariantCulture), Time.frameCount.ToString(CultureInfo.InvariantCulture),
                    arrival.Csv(), saved));
                if (m_PairPoses)
                {
                    bool camOk = ProbeClocks.HeadPoseAt(newestTs, m_PoseCam);
                    bool arrOk = ProbeClocks.HeadPoseAt(arrival.Hmd, m_PoseArr);
                    m_Run.Row(PoseFile, PoseHeader, string.Join(",",
                        m_Seq.ToString(CultureInfo.InvariantCulture), newestTs.ToString(CultureInfo.InvariantCulture),
                        arrival.Hmd.ToString(CultureInfo.InvariantCulture), camOk ? "1" : "0", Floats(m_PoseCam, camOk),
                        arrOk ? "1" : "0", Floats(m_PoseArr, arrOk)));
                }
                m_Seq++;
            }
            finally
            {
                XREALPlugin.DisposeRGBCameraDataHandle(newest);
            }
        }

        static string Floats(float[] m, bool ok)
        {
            var b = new StringBuilder(16 * 12);
            for (int i = 0; i < 16; i++)
            {
                if (i > 0) b.Append(',');
                if (ok) b.Append(m[i].ToString("R", CultureInfo.InvariantCulture));
            }
            return b.ToString();
        }

        /// <summary>Binary PGM of the Y plane, box-downscaled by <paramref name="scale"/>, rows as delivered.</summary>
        static byte[] Pgm(byte[] y, int stride, int width, int rows, int scale)
        {
            int ow = width / scale, oh = rows / scale;
            byte[] head = Encoding.ASCII.GetBytes($"P5\n{ow} {oh}\n255\n");
            var outp = new byte[head.Length + ow * oh];
            Buffer.BlockCopy(head, 0, outp, 0, head.Length);
            int o = head.Length, area = scale * scale;
            for (int r = 0; r < oh; r++)
                for (int c = 0; c < ow; c++)
                {
                    int sum = 0;
                    for (int dy = 0; dy < scale; dy++)
                    {
                        int row = (r * scale + dy) * stride + c * scale;
                        for (int dx = 0; dx < scale; dx++) sum += y[row + dx];
                    }
                    outp[o++] = (byte)(sum / area);
                }
            return outp;
        }

        public void Finish()
        {
            if (!Running) return;
            Running = false;
            m_Capturing = false;
            m_Run.Meta("eye_frames", m_Seq.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("eye_drained", m_Drained.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("eye_same_ts", m_SameTs.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("eye_saved_files", m_Saved.ToString(CultureInfo.InvariantCulture));
            if (m_PairPoses) m_Run.Meta("head_pose_at_time", $"ok {ProbeClocks.PoseOk}/{ProbeClocks.PoseCalls}");
            Action done = m_OnDone;
            m_OnDone = null;
            Destroy(gameObject);
            done?.Invoke();
        }

        void OnDestroy() { }   // the Eye camera stays on (EyeCamera)
    }
}
