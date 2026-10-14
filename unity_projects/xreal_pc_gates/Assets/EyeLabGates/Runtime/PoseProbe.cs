using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLabGates
{
    /// <summary>
    /// Samples the XR head pose every <c>Update</c> (phase <c>update</c>) and in
    /// <see cref="Application.onBeforeRender"/> (phase <c>render</c>) into pose.csv: head device validity,
    /// <see cref="CommonUsages.trackingState"/> flags, position (m) and rotation, the main camera's world
    /// pose, <see cref="XREALPlugin.GetTrackingType"/>, whether the XREAL loader is active and the native
    /// tracking reason (sampled in the update phase; 0 = tracking ready).
    /// </summary>
    public sealed class PoseProbe : MonoBehaviour
    {
        GateRunLog m_Log;
        Transform m_Camera;

        public bool LastValid { get; private set; }
        public InputTrackingState LastTrackingState { get; private set; }
        public Vector3 LastPosition { get; private set; }
        public Quaternion LastRotation { get; private set; } = Quaternion.identity;
        public string LastTrackingType { get; private set; } = "";
        public bool LoaderActive { get; private set; }
        public int LastTrackingReason { get; private set; } = NativeTracking.NoValue;

        public void Attach(GateRunLog log) => m_Log = log;

        void OnEnable() => Application.onBeforeRender += OnBeforeRender;

        void OnDisable() => Application.onBeforeRender -= OnBeforeRender;

        void Update() => Sample("update");

        void OnBeforeRender() => Sample("render");

        void Sample(string phase)
        {
            if (m_Log == null) return;

            LoaderActive = XREALUtility.GetActiveLoader() != null;
            if (phase == "update")
            {
                // Native tracking-type query only with an active session; without one there is nothing to ask.
                LastTrackingType = LoaderActive ? XREALPlugin.GetTrackingType().ToString() : "";
                LastTrackingReason = NativeTracking.Reason();
            }

            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            bool valid = head.isValid;
            InputTrackingState state = InputTrackingState.None;
            Vector3 p = Vector3.zero;
            Quaternion q = Quaternion.identity;
            if (valid)
            {
                head.TryGetFeatureValue(CommonUsages.trackingState, out state);
                head.TryGetFeatureValue(CommonUsages.devicePosition, out p);
                head.TryGetFeatureValue(CommonUsages.deviceRotation, out q);
            }
            LastValid = valid;
            LastTrackingState = state;
            LastPosition = p;
            LastRotation = q;

            if (m_Camera == null)
            {
                // The rig's CenterCamera carries the MainCamera tag but its Camera component is disabled in
                // DirectPresent mode, so Camera.main is null; the transform still follows CameraRoot's driver.
                var go = GameObject.FindWithTag("MainCamera");
                if (go != null) m_Camera = go.transform;
            }
            Vector3 cp = m_Camera != null ? m_Camera.position : Vector3.zero;
            Quaternion cq = m_Camera != null ? m_Camera.rotation : Quaternion.identity;

            m_Log.Pose(
                $"{GateRunLog.F(Time.realtimeSinceStartupAsDouble)},{Time.frameCount},{phase},{(valid ? 1 : 0)},{(int)state}," +
                $"{F(p.x)},{F(p.y)},{F(p.z)},{F(q.x)},{F(q.y)},{F(q.z)},{F(q.w)}," +
                $"{F(cp.x)},{F(cp.y)},{F(cp.z)},{F(cq.x)},{F(cq.y)},{F(cq.z)},{F(cq.w)}," +
                $"{LastTrackingType},{(LoaderActive ? 1 : 0)}," +
                $"{(LastTrackingReason == NativeTracking.NoValue ? "" : LastTrackingReason.ToString(System.Globalization.CultureInfo.InvariantCulture))}");
        }

        static string F(float v) => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
    }
}
