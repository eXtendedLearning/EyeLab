using System;
using Unity.XR.XREAL;
using Unity.XR.XREAL.Experimental;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Writes host/build/args information and the glasses' self-reported device data to run.json.
    /// The calibration values are vendor claims recorded for G4, not trusted. Every native call's
    /// bool return is stored next to its value; an exception is stored instead of a value.
    /// </summary>
    public sealed class DeviceInfoProbe : MonoBehaviour
    {
        public const string BuildInfoResource = "EyeLabGatesBuildInfo";

        GateRunLog m_Log;

        public void Attach(GateRunLog log) => m_Log = log;

        /// <summary>Sections that do not need the glasses: args, build, host.</summary>
        public void WriteStatic(GateArgs args)
        {
            var a = new JsonObject()
                .Set("gate", args.GateName)
                .Set("order", args.Order)
                .Set("reader", args.Reader)
                .Set("flush_lag_s", args.FlushLag)
                .Set("duration_s", args.Duration)
                .Set("snapshot_every_s", args.SnapshotEvery)
                .Set("label", args.Label)
                .Set("log_root", args.LogRoot)
                .Set("argv", args.Raw)
                .Set("warnings", args.Warnings);
            m_Log.SetSection("args", a);

            var build = new JsonObject()
                .Set("unity_version", Application.unityVersion)
                .Set("product", Application.productName)
                .Set("is_editor", Application.isEditor);
            var info = Resources.Load<TextAsset>(BuildInfoResource);
            if (info != null)
            {
                foreach (var line in info.text.Split('\n'))
                {
                    int eq = line.IndexOf('=');
                    if (eq > 0)
                        build.Set(line.Substring(0, eq).Trim(), line.Substring(eq + 1).Trim());
                }
            }
            else
            {
                build.Set("build_info", "missing (Editor play mode or build without BuildGates)");
            }
            m_Log.SetSection("build", build);

            m_Log.SetSection("host", new JsonObject()
                .Set("host", Application.isEditor ? "editor" : "player")
                .Set("os", SystemInfo.operatingSystem)
                .Set("gpu", SystemInfo.graphicsDeviceName)
                .Set("graphics_api", SystemInfo.graphicsDeviceType.ToString())
                .Set("utc_start", DateTime.UtcNow.ToString("o"))
                .Set("persistent_data_path", Application.persistentDataPath)
                .Set("run_dir", m_Log.Directory));
            m_Log.Flush();
        }

        /// <summary>Queries the glasses; <paramref name="stage"/> names the section (e.g. "connect", "xr").</summary>
        public void Probe(string stage)
        {
            var d = new JsonObject();
            d.Set("utc", DateTime.UtcNow.ToString("o"));
            d.Set("t_unity", Time.realtimeSinceStartupAsDouble);

            Try(d, "device_type", () => XREALPlugin.GetDeviceType().ToString());
            Try(d, "device_category", () => XREALPlugin.GetDeviceCategory().ToString());
            Try(d, "session_manager_device", () => XREALSessionManager.Singleton != null
                ? XREALSessionManager.Singleton.ConnectedDevice.ToString() : "no session manager");
            Try(d, "glasses_system_version", XREALExperimentalPlugin.GetGlassesSystemVersion);
            Try(d, "glasses_sw_version", XREALExperimentalPlugin.GetGlassesSWVersion);
            Try(d, "glasses_hw_version", XREALExperimentalPlugin.GetGlassesHWVersion);
            Try(d, "glasses_dsp_version", XREALExperimentalPlugin.GetGlassesDspVersion);
            Try(d, "rgb_camera_feature_supported", () => XREALPlugin.IsHMDFeatureSupported(XREALSupportedFeature.XREAL_FEATURE_RGB_CAMERA));
            Try(d, "head_position_feature_supported", () => XREALPlugin.IsHMDFeatureSupported(XREALSupportedFeature.XREAL_FEATURE_PERCEPTION_HEAD_TRACKING_POSITION));

            d.Set("rgb_camera_intrinsic", Intrinsic(XREALComponent.XREAL_COMPONENT_RGB_CAMERA));
            d.Set("rgb_camera_pose_from_head", PoseFromHead(XREALComponent.XREAL_COMPONENT_RGB_CAMERA));
            d.Set("rgb_camera_resolution", Resolution(XREALComponent.XREAL_COMPONENT_RGB_CAMERA));
            d.Set("display_left_pose_from_head", PoseFromHead(XREALComponent.XREAL_COMPONENT_DISPLAY_LEFT));
            d.Set("display_right_pose_from_head", PoseFromHead(XREALComponent.XREAL_COMPONENT_DISPLAY_RIGHT));
            d.Set("display_left_resolution", Resolution(XREALComponent.XREAL_COMPONENT_DISPLAY_LEFT));

            var cur = Screen.currentResolution;
            d.Set("screen", new JsonObject()
                .Set("width", Screen.width)
                .Set("height", Screen.height)
                .Set("current_resolution", $"{cur.width}x{cur.height}@" +
                     cur.refreshRateRatio.value.ToString("F2", System.Globalization.CultureInfo.InvariantCulture))
                .Set("dpi", Screen.dpi)
                .Set("fullscreen_mode", Screen.fullScreenMode.ToString()));
            d.Set("unity_display_count", Display.displays.Length);
#if UNITY_STANDALONE
            Try(d, "active_monitor_count", () => XREALExperimentalPlugin.GetActiveMonitorCount());
#endif
            m_Log.SetSection("device_" + stage, d);
            m_Log.Flush();
            m_Log.Event("device_info", stage,
                $"type={Get(d, "device_type")} sys={Get(d, "glasses_system_version")} sw={Get(d, "glasses_sw_version")}");
        }

        static JsonObject Intrinsic(XREALComponent c)
        {
            var o = new JsonObject();
            try
            {
                Vector2 f = default, p = default;
                bool ok = XREALPlugin.GetCameraIntrinsic(c, ref f, ref p);
                o.Set("ok", ok).Set("fx", f.x).Set("fy", f.y).Set("cx", p.x).Set("cy", p.y);
            }
            catch (Exception e) { o.Set("error", e.GetType().Name + ": " + e.Message); }
            return o;
        }

        static JsonObject PoseFromHead(XREALComponent c)
        {
            var o = new JsonObject();
            try
            {
                Pose pose = default;
                bool ok = XREALPlugin.GetDevicePoseFromHead(c, ref pose);
                o.Set("ok", ok)
                 .Set("position_m", new[] { pose.position.x, pose.position.y, pose.position.z })
                 .Set("rotation_xyzw", new[] { pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w });
            }
            catch (Exception e) { o.Set("error", e.GetType().Name + ": " + e.Message); }
            return o;
        }

        static JsonObject Resolution(XREALComponent c)
        {
            var o = new JsonObject();
            try
            {
                Vector2Int size = default;
                bool ok = XREALPlugin.GetDeviceResolution(c, ref size);
                o.Set("ok", ok).Set("width", size.x).Set("height", size.y);
            }
            catch (Exception e) { o.Set("error", e.GetType().Name + ": " + e.Message); }
            return o;
        }

        static void Try(JsonObject d, string key, Func<object> f)
        {
            try { d.Set(key, f()); }
            catch (Exception e) { d.Set(key, "error " + e.GetType().Name + ": " + e.Message); }
        }

        static string Get(JsonObject d, string key) =>
            d.TryGet(key, out var v) ? Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture) : "";
    }
}
