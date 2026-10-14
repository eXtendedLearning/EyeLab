using System;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Once per second to perf.csv: Unity frame rate, this process's CPU use (percent of all logical
    /// cores, and cores busy), whole-system CPU utility and CPU performance (clock as percent of nominal;
    /// a falling value under load indicates thermal or power throttling), managed heap, camera rate and
    /// lag growth, native tracking reason and the current protocol step. Separates host load from SDK
    /// behaviour in G3.
    /// </summary>
    public sealed class PerfProbe : MonoBehaviour
    {
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        GateRunLog m_Log;
        CameraProbe m_Camera;
        PoseProbe m_Pose;
        Func<string> m_Step;
        Process m_Process;
        double m_LastWall;
        double m_LastCpu;
        int m_LastFrame;
        PdhCounters m_Pdh;

        public double CpuCoresBusy { get; private set; }
        public double SystemCpuPct { get; private set; } = double.NaN;
        public double CpuPerformancePct { get; private set; } = double.NaN;

        public void Attach(GateRunLog log, CameraProbe camera, PoseProbe pose, Func<string> step)
        {
            m_Log = log;
            m_Camera = camera;
            m_Pose = pose;
            m_Step = step;
            m_Process = Process.GetCurrentProcess();
            m_LastWall = Time.realtimeSinceStartupAsDouble;
            m_LastCpu = m_Process.TotalProcessorTime.TotalSeconds;
            m_LastFrame = Time.frameCount;
            m_Pdh = PdhCounters.TryOpen(out string error);
            m_Log.Event("perf_counters", m_Pdh != null ? "OK" : "UNAVAILABLE", error ?? "");
        }

        void OnDestroy() => m_Pdh?.Dispose();

        void Update()
        {
            if (m_Log == null) return;
            double now = Time.realtimeSinceStartupAsDouble;
            double dt = now - m_LastWall;
            if (dt < 1.0) return;

            double cpu;
            try
            {
                m_Process.Refresh();
                cpu = m_Process.TotalProcessorTime.TotalSeconds;
            }
            catch (Exception)
            {
                cpu = double.NaN;
            }
            CpuCoresBusy = (cpu - m_LastCpu) / dt;
            double pct = 100.0 * CpuCoresBusy / Math.Max(1, SystemInfo.processorCount);
            double fps = (Time.frameCount - m_LastFrame) / dt;
            double gcMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            int reason = m_Pose != null ? m_Pose.LastTrackingReason : NativeTracking.NoValue;
            if (m_Pdh != null && m_Pdh.Sample(out double util, out double perf))
            {
                SystemCpuPct = util;
                CpuPerformancePct = perf;
            }

            m_Log.Perf($"{GateRunLog.F(now)},{Time.frameCount},{GateRunLog.F(fps, "F1")},{GateRunLog.F(pct, "F1")}," +
                       $"{GateRunLog.F(CpuCoresBusy, "F2")},{GateRunLog.F(SystemCpuPct, "F1")},{GateRunLog.F(CpuPerformancePct, "F1")}," +
                       $"{GateRunLog.F(gcMb, "F1")},{GateRunLog.F(m_Camera != null ? m_Camera.Fps : double.NaN, "F1")}," +
                       $"{GateRunLog.F(m_Camera != null ? m_Camera.LagGrowth : double.NaN, "F2")}," +
                       $"{(reason == NativeTracking.NoValue ? "" : reason.ToString(Inv))},{GateRunLog.Csv(m_Step?.Invoke() ?? "")}");

            m_LastWall = now;
            m_LastCpu = cpu;
            m_LastFrame = Time.frameCount;
        }

        /// <summary>Windows PDH with English counter paths (works on localized Windows).</summary>
        sealed class PdhCounters : IDisposable
        {
            const uint PDH_FMT_DOUBLE = 0x00000200;
            const uint PDH_FMT_NOCAP100 = 0x00008000;

            [StructLayout(LayoutKind.Explicit)]
            struct PdhFmtCounterValue
            {
                [FieldOffset(0)] public uint CStatus;
                [FieldOffset(8)] public double DoubleValue;
            }

            [DllImport("pdh.dll", CharSet = CharSet.Unicode)]
            static extern uint PdhOpenQuery(string dataSource, IntPtr userData, out IntPtr query);

            [DllImport("pdh.dll", CharSet = CharSet.Unicode, EntryPoint = "PdhAddEnglishCounterW")]
            static extern uint PdhAddEnglishCounter(IntPtr query, string path, IntPtr userData, out IntPtr counter);

            [DllImport("pdh.dll")]
            static extern uint PdhCollectQueryData(IntPtr query);

            [DllImport("pdh.dll")]
            static extern uint PdhGetFormattedCounterValue(IntPtr counter, uint format, IntPtr type, out PdhFmtCounterValue value);

            [DllImport("pdh.dll")]
            static extern uint PdhCloseQuery(IntPtr query);

            IntPtr m_Query, m_Util, m_Perf;

            public static PdhCounters TryOpen(out string error)
            {
                error = null;
                try
                {
                    var c = new PdhCounters();
                    uint s = PdhOpenQuery(null, IntPtr.Zero, out c.m_Query);
                    if (s != 0) { error = $"PdhOpenQuery 0x{s:X}"; return null; }
                    s = PdhAddEnglishCounter(c.m_Query, @"\Processor Information(_Total)\% Processor Utility", IntPtr.Zero, out c.m_Util);
                    if (s != 0) { error = $"utility counter 0x{s:X}"; c.Dispose(); return null; }
                    s = PdhAddEnglishCounter(c.m_Query, @"\Processor Information(_Total)\% Processor Performance", IntPtr.Zero, out c.m_Perf);
                    if (s != 0) { error = $"performance counter 0x{s:X}"; c.Dispose(); return null; }
                    PdhCollectQueryData(c.m_Query); // rate counters need two samples
                    return c;
                }
                catch (Exception e)
                {
                    error = e.GetType().Name + ": " + e.Message;
                    return null;
                }
            }

            public bool Sample(out double utilityPct, out double performancePct)
            {
                utilityPct = performancePct = double.NaN;
                if (PdhCollectQueryData(m_Query) != 0) return false;
                if (PdhGetFormattedCounterValue(m_Util, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, IntPtr.Zero, out var u) == 0)
                    utilityPct = u.DoubleValue;
                if (PdhGetFormattedCounterValue(m_Perf, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, IntPtr.Zero, out var p) == 0)
                    performancePct = p.DoubleValue;
                return true;
            }

            public void Dispose()
            {
                if (m_Query != IntPtr.Zero) PdhCloseQuery(m_Query);
                m_Query = IntPtr.Zero;
            }
        }
    }
}
