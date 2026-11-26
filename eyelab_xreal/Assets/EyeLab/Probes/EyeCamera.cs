using System.Globalization;
using Unity.XR.XREAL;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// The Eye (RGB) camera, started once and left running until the app ends (0.5.004).
    /// On 2026-11-26 (run 20261126_140803_pcheck) <c>XREALPlugin.StopRGBCameraDataCapture()</c> never returned at the
    /// end of a run: the SDK logged "Rgb Stop Camera, end, ret=0" and then "SyncClose Wait Timeout!", the Unity main
    /// thread stayed inside the call (watchdog: STUCK in Update.ScriptRunDelayedDynamicFrameRate), and the app had to
    /// be killed. Earlier stops in the same session had returned. Nothing in EyeLab calls the stop any more: every
    /// probe asks <see cref="Ensure"/> and reads the newest frame, so frames that arrive between runs are dropped.
    /// </summary>
    public static class EyeCamera
    {
        static bool s_On;
        static ulong s_Id = ulong.MaxValue;

        public static bool On => s_On;

        /// <summary>Starts the capture if it is not running; returns the capture id (ulong.MaxValue = failed).</summary>
        public static ulong Ensure()
        {
            if (s_On) return s_Id;
            s_Id = XREALPlugin.StartRGBCameraDataCapture();
            s_On = s_Id != ulong.MaxValue;
            Debug.Log("[EyeLab probe] Eye camera " + (s_On ? "started, id " + s_Id.ToString(CultureInfo.InvariantCulture) : "start FAILED") +
                      " (left running until the app ends)");
            return s_Id;
        }
    }
}
