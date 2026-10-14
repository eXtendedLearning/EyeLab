using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Unity.XR.XREAL;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Native head-tracking reason and native log capture.
    ///
    /// The Windows XREALXRPlugin.dll does not export <c>GetTrackingReason</c> (export table checked
    /// 2026-10-14; the 14 Oct G3c runs timed out on "n/a" because of it). It does export
    /// <c>SetNativeLogCallback</c> and formats "[NativePerception] TrackingReason changed: %d -> %d"
    /// itself, so the reason is parsed from the native log. The vendor installs this callback only in
    /// the Editor; the harness installs it in the player and keeps every native line for native.log.
    /// Values follow the vendor <c>NotTrackingReason</c> enum; -1 means no value.
    /// </summary>
    public static class NativeTracking
    {
        public const int NoValue = -1;
        public const int Ready = 0; // NotTrackingReason.None: tracking is working normally
        const int MaxQueuedLines = 20000;

        static readonly string[] k_Names =
        {
            "None", "Initializing", "Relocalizing", "InsufficientLight", "InsufficientFeatures",
            "ExcessiveMotion", "Unsupported", "CameraUnavailable", "Scanning",
        };

        static readonly Regex k_Changed = new Regex(@"TrackingReason changed:\s*(-?\d+)\s*->\s*(-?\d+)", RegexOptions.Compiled);

        delegate void NativeLogCallback(LogType logType, IntPtr message);

        [DllImport("XREALXRPlugin", EntryPoint = "SetNativeLogCallback")]
        static extern void SetNativeLogCallbackNative(NativeLogCallback callback);

        static NativeLogCallback s_Callback; // kept alive for the native side
        static volatile int s_Reason = NoValue;
        static readonly ConcurrentQueue<string> s_Lines = new ConcurrentQueue<string>();
        static readonly Stopwatch s_Clock = new Stopwatch();
        static double s_ClockOffset;
        static int s_Dropped;

        public static bool Installed { get; private set; }
        public static string InstallError { get; private set; } = "";
        public static int LinesSeen { get; private set; }
        public static int ReasonLinesSeen { get; private set; }

        /// <summary>Install the native log callback. Call on the main thread before XR starts.</summary>
        public static void Install()
        {
            if (Installed) return;
            s_ClockOffset = Time.realtimeSinceStartupAsDouble;
            s_Clock.Restart();
            s_Callback = OnNativeLog;
            try
            {
                SetNativeLogCallbackNative(s_Callback);
                Installed = true;
            }
            catch (Exception e)
            {
                InstallError = e.GetType().Name + ": " + e.Message;
            }
        }

        /// <summary>Remove the callback before the managed side shuts down.</summary>
        public static void Uninstall()
        {
            if (!Installed) return;
            try { SetNativeLogCallbackNative(null); }
            catch (Exception) { /* best effort */ }
            Installed = false;
        }

        /// <summary>Tracking reason, or <see cref="NoValue"/> when XR is not active or nothing was reported yet.</summary>
        public static int Reason() => XREALUtility.GetActiveLoader() == null ? NoValue : s_Reason;

        /// <summary>Native log lines captured since the last call, prefixed with t_unity; main thread.</summary>
        public static bool TryDequeue(out string line) => s_Lines.TryDequeue(out line);

        public static int DroppedLines => s_Dropped;

        [AOT.MonoPInvokeCallback(typeof(NativeLogCallback))]
        static void OnNativeLog(LogType logType, IntPtr messagePtr)
        {
            try
            {
                string message = Marshal.PtrToStringUTF8(messagePtr) ?? "";
                LinesSeen++;
                var m = k_Changed.Match(message);
                if (m.Success && int.TryParse(m.Groups[2].Value, out int reason))
                {
                    s_Reason = reason;
                    ReasonLinesSeen++;
                }
                if (s_Lines.Count < MaxQueuedLines)
                {
                    double t = s_ClockOffset + s_Clock.Elapsed.TotalSeconds;
                    s_Lines.Enqueue(t.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + " " + logType + " " + message);
                }
                else
                {
                    s_Dropped++;
                }
            }
            catch (Exception)
            {
                // never throw into native code
            }
        }

        public static string Name(int reason) =>
            reason >= 0 && reason < k_Names.Length ? k_Names[reason] : reason == NoValue ? "n/a" : $"#{reason}";
    }
}
