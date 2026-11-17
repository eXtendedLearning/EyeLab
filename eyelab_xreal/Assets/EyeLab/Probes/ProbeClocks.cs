using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.1 (gate B1): every clock an Eye frame can be compared against, read as close together as possible.
    /// <para>On Android the kernel keeps CLOCK_MONOTONIC (Java <c>System.nanoTime</c>), CLOCK_BOOTTIME
    /// (<c>SystemClock.elapsedRealtimeNanos</c>, the clock of SensorEvent and of Camera2 when its timestamp source
    /// is REALTIME) and CLOCK_REALTIME (wall time). The XREAL HMD clock (<c>GetHMDTimeNanos</c>) was QPC on Windows;
    /// on Android its domain is unknown. The B1 analysis finds the domain of <c>cam_ts</c> by which clock gives a
    /// small, stable, positive frame age. Nothing here assumes the answer.</para>
    /// </summary>
    public static class ProbeClocks
    {
        public const int ClockRealtime = 0, ClockMonotonic = 1, ClockBoottime = 7;

        [StructLayout(LayoutKind.Sequential)]
        struct Timespec { public long sec; public long nsec; }

        [DllImport("c", EntryPoint = "clock_gettime")]
        static extern int ClockGettime(int clockId, ref Timespec ts);

        [DllImport("XREALXRPlugin", EntryPoint = "GetHMDTimeNanos")]
        [return: MarshalAs(UnmanagedType.U1)]
        static extern bool GetHMDTimeNanosNative(ref ulong timeNs);

        static bool s_LibcBroken, s_HmdBroken;
        public static string Errors { get; private set; } = "";

        /// <summary>The kernel clock in ns, or -1 when libc cannot be called (Editor on Windows).</summary>
        public static long Read(int clockId)
        {
            if (s_LibcBroken) return -1;
            try
            {
                var ts = new Timespec();
                if (ClockGettime(clockId, ref ts) == 0) return ts.sec * 1_000_000_000L + ts.nsec;
            }
            catch (Exception e)
            {
                s_LibcBroken = true;
                Errors += "libc: " + e.GetType().Name + "; ";
            }
            return -1;
        }

        public static long MonoNs => Read(ClockMonotonic);
        public static long BootNs => Read(ClockBoottime);
        public static long RealNs => Read(ClockRealtime);

        /// <summary>XREAL HMD clock in ns, 0 when unavailable (export missing or XR not started).</summary>
        public static ulong HmdNs
        {
            get
            {
                if (s_HmdBroken) return 0;
                try
                {
                    ulong t = 0;
                    if (GetHMDTimeNanosNative(ref t)) return t;
                }
                catch (Exception e)
                {
                    s_HmdBroken = true;
                    Errors += "hmd: " + e.GetType().Name + ": " + e.Message + "; ";
                }
                return 0;
            }
        }

        /// <summary>One sample of every clock, mono read twice so the bracket width bounds the read skew.</summary>
        public struct Sample
        {
            public long Mono0, Boot, Real, Mono1;
            public ulong Hmd;
            public double Unity;

            public string Csv() =>
                $"{Mono0},{Boot},{Real},{Hmd},{Mono1},{Unity.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)}";

            public const string Header = "mono_ns,boot_ns,real_ns,hmd_ns,mono_after_ns,unity_s";
        }

        public static Sample Now()
        {
            var s = new Sample { Mono0 = MonoNs, Boot = BootNs, Real = RealNs, Hmd = HmdNs };
            s.Mono1 = MonoNs;
            s.Unity = Time.realtimeSinceStartupAsDouble;
            return s;
        }
    }
}
