using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.2 (gate B2): the head pose on the Beam Pro, alone or with the Eye stream. Logs, every Update, the XR head
    /// device's pose and tracking state next to the probe clocks and the XREAL tracking type (<c>head.csv</c>), and
    /// tracking events (<c>events.csv</c>): mode switch requested and done, tracked/untracked transitions, pose jumps.
    /// With the Eye on it runs <see cref="EyeChainProbe"/> with pose pairing, so every kept frame also gets the native
    /// head pose at its <c>cam_ts</c> and at its arrival (<c>eye_pose.csv</c>).
    /// <para>The requested tracking type is set first (switch time logged); a 3DoF run switches back to 6DoF at the
    /// end so the next run starts from the default.</para>
    /// </summary>
    public sealed class HeadPoseProbe : MonoBehaviour
    {
        const string HeadFile = "head.csv";
        const string HeadHeader = "unity_frame," + ProbeClocks.Sample.Header +
                                  ",tracking_type,valid,tracked,state,px,py,pz,qx,qy,qz,qw";
        const string EventFile = CuePlayer.EventFile;
        const string EventHeader = CuePlayer.EventHeader;
        const string GridFile = "pose_grid.csv";
        static readonly string GridHeader = "hmd_ns,ok," + string.Join(",", Cols16());
        const ulong GridStepNs = 5_000_000, GridLagNs = 10_000_000;

        ProbeRun m_Run;
        bool m_Logging, m_LastTracked, m_HaveLast;
        Vector3 m_LastPos;
        long m_Rows, m_Untracked, m_Jumps, m_GridRows;
        bool m_PoseGrid;
        ulong m_NextGrid;
        readonly float[] m_Grid = new float[16];

        static IEnumerable<string> Cols16() { for (int i = 0; i < 16; i++) yield return "m" + i; }

        public string Status { get; private set; } = "";

        /// <param name="cues">T6.3: a guided run; its length is the cues' total and <paramref name="seconds"/> is ignored.</param>
        /// <param name="show">Shows the current cue (instruction, seconds left, fraction left, "step i/n").</param>
        /// <param name="saveEvery">Eye frames: save every n-th Y plane (T6.3 motion sync: every frame).</param>
        /// <param name="poseGrid">T6.3 motion sync: also log the native head pose every 5 ms of HMD time
        /// (<c>pose_grid.csv</c>), the reference the image motion is correlated with.</param>
        public static IEnumerator Run(ProbeRun run, float seconds, TrackingType mode, bool withEye, Action<string> status,
                                      IReadOnlyList<CueStep> cues = null, Action<string, float, float, string> show = null,
                                      int saveEvery = 0, int scale = 2, bool poseGrid = false)
        {
            var go = new GameObject("EyeLab HeadPoseProbe");
            var p = go.AddComponent<HeadPoseProbe>();
            p.m_Run = run;
            p.m_PoseGrid = poseGrid;
            if (cues != null)
            {
                seconds = 0f;
                foreach (CueStep c in cues) seconds += c.Seconds;
            }
            run.Meta("head_seconds", ProbeRun.F(seconds));
            run.Meta("head_mode_requested", mode.ToString());
            run.Meta("head_with_eye", withEye ? "1" : "0");
            run.Meta("head_mode_at_start", SafeType());
            p.m_Logging = true;   // log from the start, so the switch itself is in head.csv

            yield return p.Switch(mode, status);

            EyeChainProbe eye = null;
            bool eyeDone = false;
            if (withEye)
                eye = EyeChainProbe.Begin(run, seconds + 1f, saveEvery, scale, 10f, () => eyeDone = true, pairPoses: true);
            if (cues != null)
            {
                float nextStatus = 0f;
                yield return CuePlayer.Ensure().Play(run, cues, (text, left, frac, counter) =>
                {
                    show?.Invoke(text, left, frac, counter);
                    if (Time.realtimeSinceStartup < nextStatus) return;
                    nextStatus = Time.realtimeSinceStartup + 0.5f;
                    status($"{counter}: {text} | {left:F0} s | rows {p.m_Rows}, jumps {p.m_Jumps}" + (eye != null ? " | " + eye.Status : ""));
                });
            }
            else
            {
                float t0 = Time.realtimeSinceStartup;
                while (Time.realtimeSinceStartup - t0 < seconds)
                {
                    float left = seconds - (Time.realtimeSinceStartup - t0);
                    status($"{(withEye ? "Head + Eye" : "Head")} {mode.ToString().Replace("MODE_", "")} | {left:F0} s | " +
                           $"rows {p.m_Rows}, untracked {p.m_Untracked}, jumps {p.m_Jumps}" + (eye != null ? " | " + eye.Status : ""));
                    yield return new WaitForSecondsRealtime(0.5f);
                }
            }
            if (eye != null && !eyeDone) eye.Finish();
            if (mode != TrackingType.MODE_6DOF) yield return p.Switch(TrackingType.MODE_6DOF, status);
            p.m_Logging = false;
            run.Meta("head_rows", p.m_Rows.ToString(CultureInfo.InvariantCulture));
            run.Meta("head_untracked_rows", p.m_Untracked.ToString(CultureInfo.InvariantCulture));
            run.Meta("head_jumps", p.m_Jumps.ToString(CultureInfo.InvariantCulture));
            if (poseGrid) run.Meta("pose_grid_rows", p.m_GridRows.ToString(CultureInfo.InvariantCulture));
            run.Meta("head_mode_at_end", SafeType());
            Destroy(go);
        }

        /// <summary>Native head poses on a 5 ms HMD-time grid, up to 10 ms before now (at most 60 per Update).</summary>
        void LogGrid(ulong hmdNow)
        {
            if (hmdNow <= GridLagNs + 200_000_000UL) return;
            ulong until = hmdNow - GridLagNs;
            if (m_NextGrid == 0) m_NextGrid = until - 200_000_000UL;   // start 200 ms back: covers the first Eye frames
            var ci = CultureInfo.InvariantCulture;
            for (int n = 0; m_NextGrid <= until && n < 60; n++, m_NextGrid += GridStepNs)
            {
                bool ok = ProbeClocks.HeadPoseAt(m_NextGrid, m_Grid);
                var b = new System.Text.StringBuilder(220);
                b.Append(m_NextGrid.ToString(ci)).Append(ok ? ",1" : ",0");
                for (int i = 0; i < 16; i++) { b.Append(','); if (ok) b.Append(m_Grid[i].ToString("R", ci)); }
                m_Run.Row(GridFile, GridHeader, b.ToString());
                m_GridRows++;
            }
        }

        static string SafeType()
        {
            try { return XREALPlugin.GetTrackingType().ToString(); }
            catch (Exception e) { return "error " + e.GetType().Name; }
        }

        IEnumerator Switch(TrackingType mode, Action<string> status)
        {
            if (SafeType() == mode.ToString()) { Event("switch_skipped", mode.ToString()); yield break; }
            status("Switching tracking to " + mode.ToString().Replace("MODE_", "") + "...");
            Event("switch_request", mode.ToString());
            Task<bool> task;
            try { task = XREALPlugin.SwitchTrackingTypeAsync(mode); }
            catch (Exception e) { Event("switch_error", e.GetType().Name + ": " + e.Message); yield break; }
            float t0 = Time.realtimeSinceStartup;
            while (!task.IsCompleted && Time.realtimeSinceStartup - t0 < 30f) yield return null;
            string result = !task.IsCompleted ? "timeout 30 s" : task.IsFaulted ? "faulted " + task.Exception?.GetBaseException().Message
                          : task.Result ? "ok" : "false";
            Event("switch_done", $"{mode} {result} {(Time.realtimeSinceStartup - t0) * 1000f:F0} ms, now {SafeType()}");
        }

        void Event(string name, string detail)
        {
            m_Run.Row(EventFile, EventHeader, $"{name},{detail.Replace(',', ';')},{ProbeClocks.Now().Csv()}");
            Debug.Log($"[EyeLab probe] head event {name}: {detail}");
        }

        void Update()
        {
            if (!m_Logging) return;
            ProbeClocks.Sample now = ProbeClocks.Now();
            if (m_PoseGrid) LogGrid(now.Hmd);
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            bool valid = head.isValid;
            Vector3 pos = default;
            Quaternion rot = Quaternion.identity;
            bool tracked = false;
            InputTrackingState state = InputTrackingState.None;
            if (valid)
            {
                head.TryGetFeatureValue(CommonUsages.devicePosition, out pos);
                head.TryGetFeatureValue(CommonUsages.deviceRotation, out rot);
                head.TryGetFeatureValue(CommonUsages.isTracked, out tracked);
                head.TryGetFeatureValue(CommonUsages.trackingState, out state);
            }
            if (!tracked) m_Untracked++;
            if (m_HaveLast && tracked != m_LastTracked) Event(tracked ? "tracked" : "untracked", state.ToString());
            if (m_HaveLast && tracked && (pos - m_LastPos).magnitude > 0.10f)
            {
                m_Jumps++;   // > 10 cm in one frame: a relocalisation, not head motion
                Event("jump", $"{(pos - m_LastPos).magnitude * 1000f:F0} mm");
            }
            m_HaveLast = true;
            m_LastTracked = tracked;
            m_LastPos = pos;
            var ci = CultureInfo.InvariantCulture;
            m_Run.Row(HeadFile, HeadHeader, string.Join(",",
                Time.frameCount.ToString(ci), now.Csv(), SafeType(), valid ? "1" : "0", tracked ? "1" : "0", ((int)state).ToString(ci),
                pos.x.ToString("R", ci), pos.y.ToString("R", ci), pos.z.ToString("R", ci),
                rot.x.ToString("R", ci), rot.y.ToString("R", ci), rot.z.ToString("R", ci), rot.w.ToString("R", ci)));
            m_Rows++;
        }
    }
}
