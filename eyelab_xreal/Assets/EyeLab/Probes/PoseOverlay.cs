using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.9 (gate B3b): the detected target drawn in the glasses, room-fixed. Each target of the frame set is drawn as
    /// the outlines of its markers (thin strips on a world-space canvas, the same UI path the probes' square uses), at
    /// the room pose of the last trusted detection:
    /// <code>
    /// world = head(cam_ts) · rgb_from_head · F·cam·F
    /// </code>
    /// <list type="bullet">
    /// <item><c>cam</c>: the plugin's pose of the target in the Eye camera (OpenCV, metres);</item>
    /// <item><c>F = diag(1, −1, 1)</c>: OpenCV to Unity (THEORY §6.3; PC gate <c>CvRotationToUnity</c>);</item>
    /// <item><c>rgb_from_head</c>: the vendor's Eye-camera-from-head pose, <c>XREALPlugin.GetDevicePoseFromHead</c>;</item>
    /// <item><c>head(cam_ts)</c>: the native head pose at the frame's exposure, which is Unity's tracking pose with z
    /// flipped: position (x, y, −z), rotation S·R·S with S = diag(1, 1, −1). Fitted on the T6.3 loop run (2 m walk):
    /// 0.0 mm and 0.01° median against Unity's head pose; the other flips are 178-262 mm off.</item>
    /// </list>
    /// No filter: the drawing jumps to each trusted pose, so what the wearer sees is the raw chain.
    /// </summary>
    public sealed class PoseOverlay : MonoBehaviour
    {
        const float LineMm = 2f;
        static readonly Color Live = new Color(0.35f, 1f, 0.45f);
        static readonly Color Held = new Color(1f, 0.6f, 0.2f);   // detection off: the drawing holds by head tracking only

        readonly Dictionary<string, Transform> m_Roots = new Dictionary<string, Transform>();
        readonly Dictionary<string, List<Image>> m_Strips = new Dictionary<string, List<Image>>();
        Transform m_Space;   // the tracking space: the XR camera's parent, or the world

        public string Shown { get; private set; } = "";

        public static PoseOverlay Create(DetectProbe.FrameSet set)
        {
            var go = new GameObject("EyeLab Pose Overlay");
            var o = go.AddComponent<PoseOverlay>();
            Camera cam = Camera.main;
            o.m_Space = cam != null ? cam.transform.parent : null;
            if (set?.targets != null)
                foreach (DetectProbe.Target t in set.targets)
                    if (t.ids != null && t.corners != null && t.corners.Length >= t.ids.Length * 12) o.Build(t);
            return o;
        }

        void Build(DetectProbe.Target t)
        {
            var root = new GameObject("overlay " + t.name, typeof(RectTransform), typeof(Canvas));
            root.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            root.transform.SetParent(transform, false);
            var strips = new List<Image>();
            for (int m = 0; m < t.ids.Length; m++)
            {
                // the marker's corners in the target frame (OpenCV-handed, metres), into Unity-handed with F
                var c = new Vector3[4];
                for (int k = 0; k < 4; k++)
                {
                    int b = m * 12 + k * 3;
                    c[k] = new Vector3(t.corners[b], -t.corners[b + 1], t.corners[b + 2]);
                }
                Vector3 centre = (c[0] + c[1] + c[2] + c[3]) / 4f;
                Vector3 right = (c[1] - c[0]).normalized;
                Vector3 up = (c[0] - c[3]).normalized;
                Vector3 normal = Vector3.Cross(right, up);
                if (normal.sqrMagnitude < 1e-8f) continue;
                float side = (c[1] - c[0]).magnitude * 1000f;   // mm
                var marker = new GameObject("marker " + t.ids[m], typeof(RectTransform));
                marker.transform.SetParent(root.transform, false);
                marker.transform.localPosition = centre;
                marker.transform.localRotation = Quaternion.LookRotation(normal, up);
                marker.transform.localScale = Vector3.one * 0.001f;   // children in mm
                float h = side / 2f;
                strips.Add(Strip(marker.transform, new Vector2(0f, h), new Vector2(side + LineMm, LineMm)));
                strips.Add(Strip(marker.transform, new Vector2(0f, -h), new Vector2(side + LineMm, LineMm)));
                strips.Add(Strip(marker.transform, new Vector2(-h, 0f), new Vector2(LineMm, side + LineMm)));
                strips.Add(Strip(marker.transform, new Vector2(h, 0f), new Vector2(LineMm, side + LineMm)));
            }
            root.SetActive(false);
            m_Roots[t.name] = root.transform;
            m_Strips[t.name] = strips;
        }

        static Image Strip(Transform parent, Vector2 at, Vector2 size)
        {
            var go = new GameObject("strip", typeof(RectTransform), typeof(Image));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.sizeDelta = size;
            rt.anchoredPosition = at;
            var img = go.GetComponent<Image>();
            img.color = Live;
            img.raycastTarget = false;
            return img;
        }

        /// <summary>Shows <paramref name="target"/> alone at a tracking-space pose (Unity-handed).</summary>
        public void Show(string target, Vector3 position, Quaternion rotation)
        {
            if (!m_Roots.TryGetValue(target, out Transform root)) return;
            if (Shown != target) Hide();
            Vector3 p = m_Space != null ? m_Space.TransformPoint(position) : position;
            Quaternion q = m_Space != null ? m_Space.rotation * rotation : rotation;
            root.SetPositionAndRotation(p, q);
            root.gameObject.SetActive(true);
            SetHeld(target, false);
            Shown = target;
        }

        /// <summary>Detection off: the drawing stays where it was, in orange.</summary>
        public void Hold()
        {
            if (Shown.Length > 0) SetHeld(Shown, true);
        }

        public void Hide()
        {
            foreach (Transform r in m_Roots.Values) r.gameObject.SetActive(false);
            Shown = "";
        }

        void SetHeld(string target, bool held)
        {
            if (!m_Strips.TryGetValue(target, out List<Image> strips)) return;
            foreach (Image s in strips) s.color = held ? Held : Live;
        }

        // ---- the chain ----------------------------------------------------------------------------------------------

        /// <summary>
        /// The target's pose in the Eye camera, Unity-handed: rotation F·R·F (R from the Rodrigues vector), position
        /// (x, −y, z).
        /// </summary>
        public static void CvToUnity(double r0, double r1, double r2, double t0, double t1, double t2,
                                     out Vector3 position, out Quaternion rotation)
        {
            double th = System.Math.Sqrt(r0 * r0 + r1 * r1 + r2 * r2);
            var R = new double[9];
            if (th < 1e-12) { R[0] = R[4] = R[8] = 1; }
            else
            {
                double kx = r0 / th, ky = r1 / th, kz = r2 / th, c = System.Math.Cos(th), s = System.Math.Sin(th), v = 1 - c;
                R[0] = c + kx * kx * v;      R[1] = kx * ky * v - kz * s; R[2] = kx * kz * v + ky * s;
                R[3] = ky * kx * v + kz * s; R[4] = c + ky * ky * v;      R[5] = ky * kz * v - kx * s;
                R[6] = kz * kx * v - ky * s; R[7] = kz * ky * v + kx * s; R[8] = c + kz * kz * v;
            }
            float S(int i) => i == 1 ? -1f : 1f;   // F R F, entry (i, j) times s_i s_j
            var col1 = new Vector3((float)R[1] * S(0) * S(1), (float)R[4] * S(1) * S(1), (float)R[7] * S(2) * S(1));
            var col2 = new Vector3((float)R[2] * S(0) * S(2), (float)R[5] * S(1) * S(2), (float)R[8] * S(2) * S(2));
            rotation = Quaternion.LookRotation(col2, col1);
            position = new Vector3((float)t0, -(float)t1, (float)t2);
        }

        /// <summary>The native head pose (4x4 column-major, right-handed) as Unity's tracking pose: z flipped.</summary>
        public static void NativeHeadToUnity(float[] m, out Vector3 position, out Quaternion rotation)
        {
            // R (row i, column j) = m[j * 4 + i]; Unity's R = S R S with S = diag(1, 1, -1)
            float s(int i) => i == 2 ? -1f : 1f;
            var col1 = new Vector3(m[4] * s(0) * s(1), m[5] * s(1) * s(1), m[6] * s(2) * s(1));
            var col2 = new Vector3(m[8] * s(0) * s(2), m[9] * s(1) * s(2), m[10] * s(2) * s(2));
            rotation = Quaternion.LookRotation(col2, col1);
            position = new Vector3(m[12], m[13], -m[14]);
        }

        /// <summary>world = head · rgb_from_head · cam.</summary>
        public static void Chain(Vector3 headP, Quaternion headQ, Pose rgbFromHead, Vector3 camP, Quaternion camQ,
                                 out Vector3 position, out Quaternion rotation)
        {
            Vector3 inHead = rgbFromHead.rotation * camP + rgbFromHead.position;
            position = headP + headQ * inHead;
            rotation = headQ * rgbFromHead.rotation * camQ;
        }
    }
}
