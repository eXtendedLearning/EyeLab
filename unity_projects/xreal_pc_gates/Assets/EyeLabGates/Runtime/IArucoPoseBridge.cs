using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// ADR-001's pose-bridge contract (docs/adr/0001-opencv-bridging-staircase.md): the rest of the
    /// app asks for target poses in a camera frame and never knows how they were computed. The gates
    /// use <see cref="WebSocketPoseBridge"/> (Step C, Python on the same machine); a native or
    /// OpenCV-for-Unity implementation can replace it without touching the lock logic.
    /// </summary>
    public interface IArucoPoseBridge
    {
        bool IsAvailable { get; }
        Task<PoseResult> EstimatePoseAsync(YGrayImage frame, ulong frameTimestampNs, CancellationToken ct);
        event Action<float> OnLatencyMeasured;
    }

    /// <summary>One luma plane as the camera delivered it.</summary>
    public sealed class YGrayImage
    {
        public int Width;
        public int Height;
        /// <summary>Width*Height bytes, row-major.</summary>
        public byte[] Data;
        /// <summary>True when row 0 is the bottom of the image (the Eye on Windows, guide §15).</summary>
        public bool BottomUp;
        /// <summary>Strictly increasing per bridge session; each frame is processed once.</summary>
        public long Sequence;
    }

    /// <summary>The poses of every served target in one frame.</summary>
    public sealed class PoseResult
    {
        public long Sequence;
        public float ServiceMs;
        public float RoundTripMs;
        /// <summary>Set when the frame failed as a whole (transport, service error).</summary>
        public string Error = "";
        /// <summary>
        /// T5.1 Phase 1 stage times, QPC seconds (<see cref="Clocks.Qpc"/>; NaN when the bridge cannot tell):
        /// send started and done, the bridge's receipt of the reply, and the estimator's own times (frame
        /// received, detection start and end, reply sent). An in-process bridge sets the estimator times too.
        /// </summary>
        public double QSend0 = double.NaN, QSend1 = double.NaN, QReplyRecv = double.NaN;
        public double QRecv = double.NaN, QDet0 = double.NaN, QDet1 = double.NaN, QReply = double.NaN;
        public readonly List<TargetPose> Targets = new List<TargetPose>();
    }

    /// <summary>
    /// One target's pose in the OpenCV camera frame of the upright image (x right, y down, z forward),
    /// metres. Unity converts with F = diag(1, -1, 1).
    /// </summary>
    public sealed class TargetPose
    {
        public string Target = "";
        public bool Ok;
        public string Reason = "";
        /// <summary>Row-major 3x3 rotation, target frame to camera frame.</summary>
        public float[] RotationRowMajor;
        public Vector3 TranslationCv;
        public float RmsPx = -1f;
        public float Ambiguity = -1f;
        public int Markers;
        public int Points;
        /// <summary>Where the target's detected points sit in the image (px) and their extent, for the static-error log.</summary>
        public float CentreXPx = -1f, CentreYPx = -1f, SpanPx = -1f;
    }

    /// <summary>What the glasses draw on a target, in the target frame (OpenCV-handed, metres).</summary>
    public sealed class TargetOverlayGeometry
    {
        public string Target = "";
        public string Label = "";
        /// <summary>6 floats per segment: x0 y0 z0 x1 y1 z1.</summary>
        public float[] Segments = Array.Empty<float>();
        /// <summary>3 floats per point.</summary>
        public float[] Points = Array.Empty<float>();
    }
}
