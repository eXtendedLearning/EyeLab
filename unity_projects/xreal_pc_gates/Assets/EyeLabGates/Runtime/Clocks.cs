using System;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// T5.1 Phase 1: the clocks every stage of the Eye correction path is measured on.
    /// <para><see cref="Qpc"/>: QueryPerformanceCounter in seconds (<see cref="Stopwatch.GetTimestamp"/>). The pose
    /// service logs <c>time.perf_counter()</c>, which on Windows is the same counter in the same unit, so
    /// Unity and service times subtract directly (verified on the desktop, 2026-11-09: equal to 1 µs).</para>
    /// </summary>
    public static class Clocks
    {
        static readonly double s_Tick = 1.0 / Stopwatch.Frequency;

        /// <summary>QPC now, seconds.</summary>
        public static double Qpc => Stopwatch.GetTimestamp() * s_Tick;

        /// <summary>QPC at the start of this Unity frame (<see cref="FrameStartProbe"/>), seconds; NaN before the first.</summary>
        public static double FrameStart { get; internal set; } = double.NaN;
        public static double PrevFrameStart { get; internal set; } = double.NaN;

        public static string F(double v) => double.IsNaN(v) ? "" : v.ToString("F6", CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Records <see cref="Clocks.FrameStart"/> before any other script's Update. The vendor RGB texture polls the
    /// SDK once per Update, so a decoded frame became available between the previous frame start and this one.
    /// </summary>
    [DefaultExecutionOrder(-10000)]
    public sealed class FrameStartProbe : MonoBehaviour
    {
        void Update()
        {
            Clocks.PrevFrameStart = Clocks.FrameStart;
            Clocks.FrameStart = Clocks.Qpc;
        }
    }

    /// <summary>
    /// Read-only calls into two exports of <c>XREALXRPlugin.dll</c> that the C# wrapper does not bind (guide §15).
    /// Signatures recovered from the x64 export code (2026-11-09, read only; nothing modified):
    /// <list type="bullet">
    /// <item><c>bool GetHMDTimeNanos(uint64_t* out)</c>: writes the HMD clock and returns true while the
    /// perception session runs; false (out untouched) otherwise.</item>
    /// <item><c>bool GetHeadPoseAtTime(uint64_t time_ns, float out[16])</c>: copies a 64-byte pose (4x4 float
    /// matrix, convention to be identified offline by <c>python -m xreal_gates stages</c>) for the given HMD time.</item>
    /// </list>
    /// Every call is wrapped; the first failure disables the call for the rest of the run and is logged once.
    /// </summary>
    public static class NativeClock
    {
        const string Lib = "XREALXRPlugin";

        [DllImport(Lib, EntryPoint = "GetHMDTimeNanos")]
        [return: MarshalAs(UnmanagedType.U1)]
        static extern bool GetHMDTimeNanosNative(ref ulong timeNs);

        [DllImport(Lib, EntryPoint = "GetHeadPoseAtTime")]
        [return: MarshalAs(UnmanagedType.U1)]
        static extern bool GetHeadPoseAtTimeNative(ulong timeNs, [Out] float[] pose16);

        static bool s_TimeBroken, s_PoseBroken;
        public static string TimeError { get; private set; } = "";
        public static string PoseError { get; private set; } = "";
        public static long TimeCalls { get; private set; }
        public static long TimeOk { get; private set; }
        public static long PoseCalls { get; private set; }
        public static long PoseOk { get; private set; }

        /// <summary>HMD time in ns, or 0 when unavailable.</summary>
        public static ulong HmdNow()
        {
            if (s_TimeBroken) return 0;
            try
            {
                ulong t = 0;
                TimeCalls++;
                if (GetHMDTimeNanosNative(ref t) && t != 0) { TimeOk++; return t; }
            }
            catch (Exception e)
            {
                s_TimeBroken = true;
                TimeError = e.GetType().Name + ": " + e.Message;
            }
            return 0;
        }

        /// <summary>The native head pose at HMD time <paramref name="timeNs"/> as 16 floats in memory order, or null.</summary>
        public static float[] HeadPoseAt(ulong timeNs)
        {
            if (s_PoseBroken || timeNs == 0) return null;
            try
            {
                var m = new float[16];
                PoseCalls++;
                if (GetHeadPoseAtTimeNative(timeNs, m)) { PoseOk++; return m; }
            }
            catch (Exception e)
            {
                s_PoseBroken = true;
                PoseError = e.GetType().Name + ": " + e.Message;
            }
            return null;
        }

        public static string Describe() =>
            $"GetHMDTimeNanos ok {TimeOk}/{TimeCalls}{(TimeError.Length > 0 ? " error " + TimeError : "")}; " +
            $"GetHeadPoseAtTime ok {PoseOk}/{PoseCalls}{(PoseError.Length > 0 ? " error " + PoseError : "")}";

        /// <summary>16 floats as space-separated text for a CSV cell ("" for null).</summary>
        public static string Text(float[] m)
        {
            if (m == null) return "";
            var sb = new StringBuilder(16 * 10);
            for (int i = 0; i < m.Length; i++)
            {
                if (i > 0) sb.Append(' ');
                sb.Append(m[i].ToString("G7", CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }
    }
}
