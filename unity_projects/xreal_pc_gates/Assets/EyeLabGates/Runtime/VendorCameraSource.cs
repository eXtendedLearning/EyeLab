using Unity.XR.XREAL;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>Frames from the scene's vendor <see cref="XREALRGBCameraTexture"/> (the G1/G3 baseline).</summary>
    public sealed class VendorCameraSource : CameraSource
    {
        [SerializeField] XREALRGBCameraTexture m_Texture;

        public XREALRGBCameraTexture Texture { get => m_Texture; set => m_Texture = value; }

        public override string SourceName => "vendor";
        public override bool IsCapturing => m_Texture != null && m_Texture.IsCapturing;
        public override ulong Timestamp => m_Texture != null ? m_Texture.GetTimeStamp() : 0;
        public override Vector2Int Resolution => m_Texture != null ? m_Texture.GetResolution() : Vector2Int.zero;
        public override Texture2D LumaTexture => m_Texture != null ? m_Texture.GetYUVFormatTextures()[0] : null;
        public override Texture2D ChromaU => m_Texture != null ? m_Texture.GetYUVFormatTextures()[1] : null;
        public override Texture2D ChromaV => m_Texture != null ? m_Texture.GetYUVFormatTextures()[2] : null;

        void OnEnable()
        {
            if (m_Texture == null) m_Texture = FindAnyObjectByType<XREALRGBCameraTexture>();
            if (m_Texture != null) m_Texture.OnRGBCameraUpdate += RaiseFrameArrived;
        }

        void OnDisable()
        {
            if (m_Texture != null) m_Texture.OnRGBCameraUpdate -= RaiseFrameArrived;
        }

        public override bool StartCapture() => m_Texture != null && m_Texture.StartCapture();

        public override bool StopCapture() => m_Texture != null && m_Texture.StopCapture();
    }
}
