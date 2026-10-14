using System;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Once per second to perf.csv: Unity frame rate, this process's CPU use (percent of all logical
    /// cores, and cores busy), managed heap, camera rate, native tracking reason and the current
    /// protocol step. Separates host load from SDK behaviour in G3.
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

        public double CpuCoresBusy { get; private set; }

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
        }

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

            m_Log.Perf($"{GateRunLog.F(now)},{Time.frameCount},{GateRunLog.F(fps, "F1")},{GateRunLog.F(pct, "F1")}," +
                       $"{GateRunLog.F(CpuCoresBusy, "F2")},{GateRunLog.F(gcMb, "F1")}," +
                       $"{GateRunLog.F(m_Camera != null ? m_Camera.Fps : double.NaN, "F1")}," +
                       $"{(reason == NativeTracking.NoValue ? "" : reason.ToString(Inv))},{GateRunLog.Csv(m_Step?.Invoke() ?? "")}");

            m_LastWall = now;
            m_LastCpu = cpu;
            m_LastFrame = Time.frameCount;
        }
    }
}
