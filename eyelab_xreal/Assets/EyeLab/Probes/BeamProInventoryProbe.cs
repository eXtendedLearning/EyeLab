using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.0 (gate BS): what this app can see of the Beam Pro's own sensors, through
    /// <c>Assets/Plugins/Android/BeamProInventory.java</c>: Camera2 characteristics of every camera id and every
    /// physical member of a logical camera (intrinsics, lens pose, timestamp source, sensor sync type, YUV sizes,
    /// concurrent-camera sets), the sensor list, the displays (the glasses are one), and a timed IMU recording
    /// (<c>imu.csv</c>: event time next to BOOTTIME and MONOTONIC at delivery). Nothing is opened: no permission
    /// is needed and no camera is started.
    /// </summary>
    public static class BeamProInventoryProbe
    {
        const string Helper = "it.univpm.eyelab.probes.BeamProInventory";

        public static IEnumerator Run(ProbeRun run, float imuSeconds, Action<string> status)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            AndroidJavaObject activity = null;
            AndroidJavaClass helper = null;
            try
            {
                using (var up = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                    activity = up.GetStatic<AndroidJavaObject>("currentActivity");
                helper = new AndroidJavaClass(Helper);
                status?.Invoke("Inventory: device, cameras, sensors, displays");
                run.Meta("device", helper.CallStatic<string>("device", activity));
                run.Meta("cameras", helper.CallStatic<string>("cameras", activity));
                run.Meta("sensors", helper.CallStatic<string>("sensors", activity));
                run.Meta("displays", helper.CallStatic<string>("displays", activity));
            }
            catch (Exception e)
            {
                run.Meta("inventory_error", e.GetType().Name + ": " + e.Message);
            }
            if (helper == null || imuSeconds <= 0) yield break;

            string csv = Path.Combine(run.Dir, "imu.csv");
            string started = null;
            try { started = helper.CallStatic<string>("imuStart", activity, csv, 0); }
            catch (Exception e) { run.Meta("imu_error", e.GetType().Name + ": " + e.Message); }
            if (started == null) yield break;
            run.Meta("imu_start", started);
            float t0 = Time.realtimeSinceStartup;
            while (Time.realtimeSinceStartup - t0 < imuSeconds)
            {
                status?.Invoke($"IMU recording, keep the Beam Pro still: {imuSeconds - (Time.realtimeSinceStartup - t0):F0} s");
                yield return new WaitForSecondsRealtime(0.5f);
            }
            try { run.Meta("imu_stop", helper.CallStatic<string>("imuStop")); }
            catch (Exception e) { run.Meta("imu_error", e.GetType().Name + ": " + e.Message); }
            helper.Dispose();
            activity?.Dispose();
#else
            run.Meta("inventory_error", "Android player only");
            status?.Invoke("Inventory runs on the Beam Pro only");
            yield break;
#endif
        }
    }
}
