using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.XR.XREAL;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Logs every <see cref="XREALRGBCameraTexture.OnRGBCameraUpdate"/> to camera.csv:
    /// <c>t_unity, frame, cam_ts, res_w, res_h, y_mean, y_mad_prev, dup_ts</c>. Luma statistics are
    /// taken on a 64×48 subsample of the Y plane; every <c>snapshotEvery</c> seconds the full Y plane
    /// is written losslessly as <c>snapshots/&lt;cam_ts&gt;.pgm</c>.
    /// </summary>
    public sealed class CameraProbe : MonoBehaviour
    {
        const int SubW = 64, SubH = 48;
        const double FpsWindow = 2.0;

        [SerializeField] XREALRGBCameraTexture m_Texture;

        GateRunLog m_Log;
        double m_SnapshotEvery = 2.0;
        double m_LastSnapshot = double.NegativeInfinity;
        readonly byte[] m_Sub = new byte[SubW * SubH];
        readonly byte[] m_PrevSub = new byte[SubW * SubH];
        bool m_HasPrevSub;
        bool m_HasPrevTs;
        ulong m_PrevTs;
        readonly Queue<double> m_UniqueTimes = new Queue<double>();

        public XREALRGBCameraTexture Texture
        {
            get => m_Texture;
            set => m_Texture = value;
        }

        public long FrameEvents { get; private set; }
        public long UniqueFrames { get; private set; }
        public ulong LastTimestamp { get; private set; }
        public Vector2Int LastResolution { get; private set; }

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

        public void Attach(GateRunLog log, double snapshotEvery)
        {
            m_Log = log;
            m_SnapshotEvery = snapshotEvery;
        }

        void OnEnable()
        {
            if (m_Texture == null)
                m_Texture = FindAnyObjectByType<XREALRGBCameraTexture>();
            if (m_Texture != null)
                m_Texture.OnRGBCameraUpdate += OnFrame;
        }

        void OnDisable()
        {
            if (m_Texture != null)
                m_Texture.OnRGBCameraUpdate -= OnFrame;
        }

        /// <summary>Call on every StartCapture so y_mad_prev never spans two capture sessions.</summary>
        public void NotifyCaptureStarted()
        {
            m_HasPrevSub = false;
            m_HasPrevTs = false;
        }

        void OnFrame()
        {
            if (m_Log == null) return;
            double t = Time.realtimeSinceStartupAsDouble;
            ulong ts = m_Texture.GetTimeStamp();
            Vector2Int res = m_Texture.GetResolution();
            bool dup = m_HasPrevTs && ts == m_PrevTs;
            m_PrevTs = ts;
            m_HasPrevTs = true;

            FrameEvents++;
            LastTimestamp = ts;
            LastResolution = res;
            if (!dup)
            {
                UniqueFrames++;
                m_UniqueTimes.Enqueue(t);
            }

            double yMean = double.NaN, yMad = double.NaN;
            Texture2D y = m_Texture.GetYUVFormatTextures()[0];
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
                        m_Log.SavePgmAsync(ts.ToString(System.Globalization.CultureInfo.InvariantCulture), w, h, copy);
                    }
                }
            }

            m_Log.Camera(
                $"{GateRunLog.F(t)},{Time.frameCount},{ts.ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"{res.x},{res.y},{GateRunLog.F(yMean, "F3")},{GateRunLog.F(yMad, "F3")},{(dup ? 1 : 0)}");
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
