using System;
using System.Runtime.InteropServices;
using Unity.XR.XREAL;

namespace EyeLabGates
{
    /// <summary>
    /// Reads the native head-tracking reason. The vendor wrapper <c>XREALPlugin.GetTrackingReason</c>
    /// is <c>internal</c>, so the same export of XREALXRPlugin.dll is bound here. Values follow the
    /// vendor <c>NotTrackingReason</c> enum; -1 means no value (no XR session, or the call failed).
    /// </summary>
    public static class NativeTracking
    {
        public const int NoValue = -1;
        public const int Ready = 0; // NotTrackingReason.None: tracking is working normally

        static readonly string[] k_Names =
        {
            "None", "Initializing", "Relocalizing", "InsufficientLight", "InsufficientFeatures",
            "ExcessiveMotion", "Unsupported", "CameraUnavailable", "Scanning",
        };

        static bool s_Broken;

        [DllImport("XREALXRPlugin", EntryPoint = "GetTrackingReason")]
        static extern int GetTrackingReasonNative();

        /// <summary>Tracking reason, or <see cref="NoValue"/> when XR is not active.</summary>
        public static int Reason()
        {
            if (s_Broken || XREALUtility.GetActiveLoader() == null)
                return NoValue;
            try
            {
                return GetTrackingReasonNative();
            }
            catch (Exception)
            {
                s_Broken = true; // missing export: stop asking
                return NoValue;
            }
        }

        public static string Name(int reason) =>
            reason >= 0 && reason < k_Names.Length ? k_Names[reason] : reason == NoValue ? "n/a" : $"#{reason}";
    }
}
