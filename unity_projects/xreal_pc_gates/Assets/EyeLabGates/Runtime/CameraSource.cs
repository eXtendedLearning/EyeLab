using System;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Where the harness gets Eye frames from. <see cref="VendorCameraSource"/> wraps the sample's
    /// <c>XREALRGBCameraTexture</c> (one acquisition per Unity Update); <see cref="DrainingCameraSource"/>
    /// acquires until the newest frame and counts the frames it discarded.
    /// </summary>
    public abstract class CameraSource : MonoBehaviour
    {
        /// <summary>Raised once per delivered frame, on the main thread.</summary>
        public event Action FrameArrived;

        public abstract string SourceName { get; }
        public abstract bool IsCapturing { get; }
        public abstract ulong Timestamp { get; }
        public abstract Vector2Int Resolution { get; }

        /// <summary>The Y (luma) plane of the latest frame, Alpha8, readable; null before the first frame.</summary>
        public abstract Texture2D LumaTexture { get; }

        /// <summary>U (Cb) and V (Cr) planes, half resolution, Alpha8; null when unavailable.</summary>
        public virtual Texture2D ChromaU => null;
        public virtual Texture2D ChromaV => null;

        /// <summary>Frames acquired and discarded in the same Update before this one; -1 when unknown.</summary>
        public virtual int LastDrained => -1;

        public abstract bool StartCapture();
        public abstract bool StopCapture();

        protected void RaiseFrameArrived() => FrameArrived?.Invoke();
    }
}
