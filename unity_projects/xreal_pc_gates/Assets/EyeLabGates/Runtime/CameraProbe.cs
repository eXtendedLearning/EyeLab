using System;
using System.Collections.Generic;
using System.Globalization;
using Unity.Collections;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Logs every frame of the active <see cref="CameraSource"/> to camera.csv:
    /// <c>t_unity, frame, cam_ts, res_w, res_h, y_mean, y_mad_prev, dup_ts, drained, source</c>. Luma
    /// statistics are taken on a 64×48 subsample of the Y plane; every <c>snapshotEvery</c> seconds the
    /// full Y plane is written losslessly as <c>snapshots/&lt;cam_ts&gt;.pgm</c>.
    /// </summary>
    public sealed class CameraProbe : MonoBehaviour
    {
        const int SubW = 64, SubH = 48;
        const double FpsWindow = 2.0;
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        GateRunLog m_Log;
        CameraSource m_Source;
        double m_SnapshotEvery = 2.0;
        double m_LastSnapshot = double.NegativeInfinity;
        readonly byte[] m_Sub = new byte[SubW * SubH];
        readonly byte[] m_PrevSub = new byte[SubW * SubH];
        bool m_HasPrevSub;
        bool m_HasPrevTs;
        ulong m_PrevTs;
        readonly Queue<double> m_UniqueTimes = new Queue<double>();

        public long UniqueFrames { get; private set; }
        public long DrainedFrames { get; private set; }
        public ulong LastTimestamp { get; private set; }
        public Vector2Int LastResolution { get; private set; }

        /// <summary>
        /// How much further behind the newest frame is than at the start of this capture session, in s:
        /// (receipt time - cam_ts) now minus the same at the first frame. Assumes cam_ts in ns (G1 data).
        /// </summary>
        public double LagGrowth { get; private set; } = double.NaN;

        double m_FirstOffset = double.NaN;

        /// <summary>Unique-timestamp frames per second over the last 2 s.</summary>
        public double Fps
        {
            get
            {
                double now = Time.realtimeSinceStartupAsDouble;
                while (m_UniqueTimes.Count > 0 && now - m_UniqueTimes.Peek() > FpsWindow)
                    m_UniqueTimes.Dequeue();
                return m_UniqueTimes.Count / FpsWindow;
            }
        }

        public void Attach(GateRunLog log, CameraSource source, double snapshotEvery)
        {
            if (m_Source != null) m_Source.FrameArrived -= OnFrame;
            m_Log = log;
            m_Source = source;
            m_SnapshotEvery = snapshotEvery;
            if (m_Source != null) m_Source.FrameArrived += OnFrame;
        }

        void OnDestroy()
        {
            if (m_Source != null) m_Source.FrameArrived -= OnFrame;
        }

        /// <summary>Call on every StartCapture so y_mad_prev never spans two capture sessions.</summary>
        public void NotifyCaptureStarted()
        {
            m_HasPrevSub = false;
            m_HasPrevTs = false;
            m_FirstOffset = double.NaN;
            LagGrowth = double.NaN;
        }

        void OnFrame()
        {
            if (m_Log == null) return;
            double t = Time.realtimeSinceStartupAsDouble;
            ulong ts = m_Source.Timestamp;
            Vector2Int res = m_Source.Resolution;
            bool dup = m_HasPrevTs && ts == m_PrevTs;
            m_PrevTs = ts;
            m_HasPrevTs = true;

            LastTimestamp = ts;
            LastResolution = res;
            double offset = t - ts * 1e-9;
            if (double.IsNaN(m_FirstOffset)) m_FirstOffset = offset;
            LagGrowth = offset - m_FirstOffset;
            int drained = m_Source.LastDrained;
            if (drained > 0) DrainedFrames += drained;
            if (!dup)
            {
                UniqueFrames++;
                m_UniqueTimes.Enqueue(t);
            }

            double yMean = double.NaN, yMad = double.NaN;
            Texture2D y = m_Source.LumaTexture;
            if (y != null && y.width > 0 && y.height > 0)
            {
                NativeArray<byte> data = y.GetRawTextureData<byte>();
                int w = y.width, h = y.height;
                if (data.Length >= w * h)
                {
                    Subsample(data, w, h, out yMean);
                    if (m_HasPrevSub)
                    {
                        long sum = 0;
                        for (int i = 0; i < m_Sub.Length; i++)
                            sum += Math.Abs(m_Sub[i] - m_PrevSub[i]);
                        yMad = (double)sum / m_Sub.Length;
                    }
                    Buffer.BlockCopy(m_Sub, 0, m_PrevSub, 0, m_Sub.Length);
                    m_HasPrevSub = true;

                    if (!dup && t - m_LastSnapshot >= m_SnapshotEvery)
                    {
                        m_LastSnapshot = t;
                        var copy = new byte[w * h];
                        NativeArray<byte>.Copy(data, copy, w * h);
                        m_Log.SavePgmAsync(ts.ToString(Inv), w, h, copy);
                    }
                }
            }

            m_Log.Camera(
                $"{GateRunLog.F(t)},{Time.frameCount},{ts.ToString(Inv)},{res.x},{res.y}," +
                $"{GateRunLog.F(yMean, "F3")},{GateRunLog.F(yMad, "F3")},{(dup ? 1 : 0)}," +
                $"{(drained >= 0 ? drained.ToString(Inv) : "")},{m_Source.SourceName}");
        }

        void Subsample(NativeArray<byte> data, int w, int h, out double mean)
        {
            long sum = 0;
            for (int j = 0; j < SubH; j++)
            {
                int row = (int)((j + 0.5) * h / SubH) * w;
                for (int i = 0; i < SubW; i++)
                {
                    byte v = data[row + (int)((i + 0.5) * w / SubW)];
                    m_Sub[j * SubW + i] = v;
                    sum += v;
                }
            }
            mean = (double)sum / (SubW * SubH);
        }
    }
}
