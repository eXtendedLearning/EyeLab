using System;
using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.UI;

namespace EyeLabGates
{
    /// <summary>
    /// Latest-frame reader. Calls the public <see cref="XREALPlugin"/> capture API directly (the vendor
    /// texture is left idle) and, every Update, acquires frames until none is left, keeping only the
    /// newest. <see cref="LastDrained"/> counts the discarded ones: if latency grows while it stays 0,
    /// the backlog is upstream of the acquisition API (decoder/transport), not in the application.
    /// Also feeds the sample's preview RawImage so the Eye image stays visible.
    /// </summary>
    public sealed class DrainingCameraSource : CameraSource
    {
        const int MaxAcquirePerUpdate = 16;
        static readonly int k_UTex = Shader.PropertyToID("_UTex");
        static readonly int k_VTex = Shader.PropertyToID("_VTex");

        [SerializeField] RawImage m_Preview;

        Texture2D m_Y, m_U, m_V;
        bool m_Capturing;
        ulong m_Timestamp;
        Vector2Int m_Resolution;
        int m_LastDrained;

        public RawImage Preview { get => m_Preview; set => m_Preview = value; }

        public override string SourceName => "drain";
        public override bool IsCapturing => m_Capturing;
        public override ulong Timestamp => m_Timestamp;
        public override Vector2Int Resolution => m_Resolution;
        public override Texture2D LumaTexture => m_Y;
        public override int LastDrained => m_LastDrained;

        public override bool StartCapture()
        {
            ulong id = XREALPlugin.StartRGBCameraDataCapture();
            m_Capturing = id != ulong.MaxValue;
            return m_Capturing;
        }

        public override bool StopCapture()
        {
            m_Capturing = false;
            return XREALPlugin.StopRGBCameraDataCapture();
        }

        void OnDestroy()
        {
            if (m_Capturing) StopCapture();
        }

        void Update()
        {
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
                if (acquired > 0 && ts == newestTs)
                {
                    // Same frame handed out again: nothing newer is queued.
                    XREALPlugin.DisposeRGBCameraDataHandle(handle);
                    break;
                }
                if (newest >= 0) XREALPlugin.DisposeRGBCameraDataHandle(newest);
                newest = handle;
                newestTs = ts;
                newestRes = res;
                acquired++;
            }
            if (acquired == 0) return;

            try
            {
                m_Timestamp = newestTs;
                m_Resolution = newestRes;
                m_LastDrained = acquired - 1;
                LoadYuv(newest);
            }
            finally
            {
                XREALPlugin.DisposeRGBCameraDataHandle(newest);
            }
            RaiseFrameArrived();
        }

        void LoadYuv(int handle)
        {
            int w = m_Resolution.x, h = m_Resolution.y;
            if (m_Y == null || m_Y.width != w || m_Y.height != h)
            {
                m_Y = new Texture2D(w, h, TextureFormat.Alpha8, false);
                m_U = new Texture2D(w / 2, h / 2, TextureFormat.Alpha8, false);
                m_V = new Texture2D(w / 2, h / 2, TextureFormat.Alpha8, false);
            }
            Load(m_Y, handle, 0);
            Load(m_U, handle, 1);
            Load(m_V, handle, 2);

            if (m_Preview != null)
            {
                m_Preview.texture = m_Y;
                if (m_Preview.material != null)
                {
                    m_Preview.material.SetTexture(k_UTex, m_U);
                    m_Preview.material.SetTexture(k_VTex, m_V);
                }
            }
        }

        static void Load(Texture2D tex, int handle, int plane)
        {
            if (!XREALPlugin.TryGetRGBCameraDataPlane(handle, plane, out IntPtr data, out Vector2Int size) || data == IntPtr.Zero)
                return;
            int available = size.x * size.y;
            if (available < tex.width * tex.height)
                return; // plane smaller than the texture: skip rather than throw
            tex.LoadRawTextureData(data, available);
            tex.Apply();
        }
    }
}
