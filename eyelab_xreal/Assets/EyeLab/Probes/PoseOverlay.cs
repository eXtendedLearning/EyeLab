using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.9 (gate B3b): the detected target drawn in the glasses, room-fixed, at the room pose of the last trusted
    /// detection:
    /// <code>
    /// world = head(cam_ts) · view_offset · rgb_from_head · F·cam·F
    /// </code>
    /// <list type="bullet">
    /// <item><c>cam</c>: the plugin's pose of the target in the Eye camera (OpenCV, metres);</item>
    /// <item><c>F = diag(1, −1, 1)</c>: OpenCV to Unity (THEORY §6.3; PC gate <c>CvRotationToUnity</c>);</item>
    /// <item><c>rgb_from_head</c>: the vendor's Eye-camera-from-head pose, <c>XREALPlugin.GetDevicePoseFromHead</c>;</item>
    /// <item><c>view_offset</c> (T6.10): the wearer's head-frame correction, set by eye (<see cref="ViewOffset"/>);</item>
    /// <item><c>head(cam_ts)</c>: the native head pose at the frame's exposure, which is Unity's tracking pose with z
    /// flipped: position (x, y, −z), rotation S·R·S with S = diag(1, 1, −1). Fitted on the T6.3 loop run (2 m walk):
    /// 0.0 mm and 0.01° median against Unity's head pose; the other flips are 178-262 mm off.</item>
    /// </list>
    /// Two styles (T6.10; Jack on T6.9: "our artefacts are clearly 2D even if in a 3D environment"):
    /// <list type="bullet">
    /// <item><b>flat</b>: the marker outlines as 2 mm strips on a world-space canvas (T6.9);</item>
    /// <item><b>solid</b>: the outlines as 2.5 mm square bars, and a 25 mm post with a cap out of each marker along its
    /// outward normal (the way an impact direction will be drawn), shaded by a fixed light so faces differ in
    /// brightness. Motion parallax and stereo are the same in both; the solid style adds thickness, shading and
    /// depth extent.</item>
    /// </list>
    /// No filter: the drawing jumps to each trusted pose.
    /// </summary>
    public sealed class PoseOverlay : MonoBehaviour
    {
        const float LineMm = 2f;
        const float BarM = 0.0025f, PostM = 0.025f, CapM = 0.005f;
        static readonly Color Live = new Color(0.35f, 1f, 0.45f);
        static readonly Color Held = new Color(1f, 0.6f, 0.2f);   // detection off: the drawing holds by head tracking only

        sealed class Drawn
        {
            public Transform Root, Flat, Solid;
            public List<Image> Strips = new List<Image>();
            public Material Mat;
            public List<Transform> Posts = new List<Transform>();
            public List<Vector3> PostAt = new List<Vector3>();   // marker centre, target frame
            public bool PostsOriented;
        }

        readonly Dictionary<string, Drawn> m_Drawn = new Dictionary<string, Drawn>();
        Transform m_Space;   // the tracking space: the XR camera's parent, or the world
        bool m_Solid;

        public string Shown { get; private set; } = "";
        public bool Solid => m_Solid;
        public string SolidShader { get; private set; } = "none";

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

        /// <summary>The marker's corners in the target frame (OpenCV-handed, metres), into Unity-handed with F.</summary>
        static Vector3[] Corners(DetectProbe.Target t, int m)
        {
            var c = new Vector3[4];
            for (int k = 0; k < 4; k++)
            {
                int b = m * 12 + k * 3;
                c[k] = new Vector3(t.corners[b], -t.corners[b + 1], t.corners[b + 2]);
            }
            return c;
        }

        void Build(DetectProbe.Target t)
        {
            var d = new Drawn();
            d.Root = new GameObject("overlay " + t.name).transform;
            d.Root.SetParent(transform, false);

            // flat: strips on a world-space canvas, one marker frame each
            var flat = new GameObject("flat", typeof(RectTransform), typeof(Canvas));
            flat.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            flat.transform.SetParent(d.Root, false);
            d.Flat = flat.transform;
            // solid: one mesh for the bars and the caps' bases, posts as children so each can face out
            var solid = new GameObject("solid", typeof(MeshFilter), typeof(MeshRenderer));
            solid.transform.SetParent(d.Root, false);
            d.Solid = solid.transform;
            // depth-tested unlit vertex colour (Resources/EyeLabVertexColor.shader); UI/Default has no depth test
            Shader sh = Resources.Load<Shader>("EyeLabVertexColor");
            d.Mat = sh != null && sh.isSupported ? new Material(sh) : new Material(Canvas.GetDefaultCanvasMaterial());
            d.Mat.color = Live;
            SolidShader = d.Mat.shader.name + (sh == null ? " (EyeLabVertexColor not found)" : !sh.isSupported ? " (EyeLabVertexColor not supported)" : "");
            var mb = new MeshBuilder();
            for (int m = 0; m < t.ids.Length; m++)
            {
                Vector3[] c = Corners(t, m);
                Vector3 centre = (c[0] + c[1] + c[2] + c[3]) / 4f;
                Vector3 right = (c[1] - c[0]).normalized;
                Vector3 up = (c[0] - c[3]).normalized;
                Vector3 normal = Vector3.Cross(right, up);
                if (normal.sqrMagnitude < 1e-8f) continue;
                float side = (c[1] - c[0]).magnitude * 1000f;   // mm

                var marker = new GameObject("marker " + t.ids[m], typeof(RectTransform));
                marker.transform.SetParent(flat.transform, false);
                marker.transform.localPosition = centre;
                marker.transform.localRotation = Quaternion.LookRotation(normal, up);
                marker.transform.localScale = Vector3.one * 0.001f;   // children in mm
                float h = side / 2f;
                d.Strips.Add(Strip(marker.transform, new Vector2(0f, h), new Vector2(side + LineMm, LineMm)));
                d.Strips.Add(Strip(marker.transform, new Vector2(0f, -h), new Vector2(side + LineMm, LineMm)));
                d.Strips.Add(Strip(marker.transform, new Vector2(-h, 0f), new Vector2(LineMm, side + LineMm)));
                d.Strips.Add(Strip(marker.transform, new Vector2(h, 0f), new Vector2(LineMm, side + LineMm)));

                for (int k = 0; k < 4; k++)
                {
                    Vector3 a = c[k], b = c[(k + 1) % 4], e = (b - a).normalized;
                    mb.Box((a + b) / 2f, Quaternion.LookRotation(e, normal), new Vector3(BarM, BarM, (b - a).magnitude + BarM));
                }
                var post = new GameObject("post " + t.ids[m], typeof(MeshFilter), typeof(MeshRenderer));
                post.transform.SetParent(solid.transform, false);
                post.transform.localPosition = centre;
                post.transform.localRotation = Quaternion.LookRotation(normal, up);
                var pb = new MeshBuilder();
                pb.Box(new Vector3(0f, 0f, PostM / 2f), Quaternion.identity, new Vector3(BarM, BarM, PostM));
                pb.Box(new Vector3(0f, 0f, PostM), Quaternion.identity, new Vector3(CapM, CapM, CapM));
                post.GetComponent<MeshFilter>().sharedMesh = pb.Mesh("post");
                post.GetComponent<MeshRenderer>().sharedMaterial = d.Mat;
                d.Posts.Add(post.transform);
                d.PostAt.Add(centre);
            }
            solid.GetComponent<MeshFilter>().sharedMesh = mb.Mesh("bars " + t.name);
            solid.GetComponent<MeshRenderer>().sharedMaterial = d.Mat;
            d.Root.gameObject.SetActive(false);
            m_Drawn[t.name] = d;
            ApplyStyle(d);
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

        /// <summary>T6.12: the solid style drawn per eye with the eye model (<see cref="EyeWarp"/>); the flat canvas is not.</summary>
        public void SetEyeWarp(bool on)
        {
            foreach (Drawn d in m_Drawn.Values) d.Mat.SetFloat("_EyeWarp", on ? 1f : 0f);
        }

        /// <summary>Flat strips (T6.9) or solid bars and posts (T6.10).</summary>
        public void SetStyle(bool solid)
        {
            m_Solid = solid;
            foreach (Drawn d in m_Drawn.Values) ApplyStyle(d);
        }

        void ApplyStyle(Drawn d)
        {
            d.Flat.gameObject.SetActive(!m_Solid);
            d.Solid.gameObject.SetActive(m_Solid);
        }

        /// <summary>Shows <paramref name="target"/> alone at a tracking-space pose (Unity-handed).</summary>
        public void Show(string target, Vector3 position, Quaternion rotation, Vector3 headPosition)
        {
            if (!m_Drawn.TryGetValue(target, out Drawn d)) return;
            if (Shown != target) Hide();
            Vector3 p = m_Space != null ? m_Space.TransformPoint(position) : position;
            Quaternion q = m_Space != null ? m_Space.rotation * rotation : rotation;
            d.Root.SetPositionAndRotation(p, q);
            if (!d.PostsOriented)
            {
                // posts out of the face: towards the viewer at the first sighting (a face seen at all faces the camera)
                Vector3 eye = m_Space != null ? m_Space.TransformPoint(headPosition) : headPosition;
                for (int i = 0; i < d.Posts.Count; i++)
                {
                    Vector3 at = d.Root.TransformPoint(d.PostAt[i]);
                    if (Vector3.Dot(d.Posts[i].forward, eye - at) < 0f)
                        d.Posts[i].localRotation = d.Posts[i].localRotation * Quaternion.AngleAxis(180f, Vector3.up);
                }
                d.PostsOriented = true;
            }
            d.Root.gameObject.SetActive(true);
            SetHeld(d, false);
            Shown = target;
        }

        /// <summary>Detection off: the drawing stays where it was, in orange.</summary>
        public void Hold()
        {
            if (Shown.Length > 0 && m_Drawn.TryGetValue(Shown, out Drawn d)) SetHeld(d, true);
        }

        public void Hide()
        {
            foreach (Drawn d in m_Drawn.Values) d.Root.gameObject.SetActive(false);
            Shown = "";
        }

        static void SetHeld(Drawn d, bool held)
        {
            Color c = held ? Held : Live;
            foreach (Image s in d.Strips) s.color = c;
            d.Mat.color = c;
        }

        void OnDestroy()
        {
            foreach (Drawn d in m_Drawn.Values) if (d.Mat != null) Destroy(d.Mat);
        }

        /// <summary>Boxes with flat-shaded faces: vertex grey = ambient + diffuse from a fixed light (target frame).</summary>
        sealed class MeshBuilder
        {
            static readonly Vector3 Light = new Vector3(-0.4f, 0.8f, -0.45f).normalized;
            readonly List<Vector3> m_V = new List<Vector3>();
            readonly List<Color> m_C = new List<Color>();
            readonly List<int> m_T = new List<int>();

            public void Box(Vector3 centre, Quaternion rot, Vector3 size)
            {
                Vector3 h = size / 2f;
                Vector3[] axes = { Vector3.right, Vector3.up, Vector3.forward };
                for (int a = 0; a < 3; a++)
                    for (int sgn = -1; sgn <= 1; sgn += 2)
                    {
                        Vector3 n = axes[a] * sgn, u = axes[(a + 1) % 3], v = axes[(a + 2) % 3];
                        Vector3 fc = Vector3.Scale(n, h);
                        Vector3 du = Vector3.Scale(u, h), dv = Vector3.Scale(v, h);
                        float g = 0.45f + 0.55f * Mathf.Max(0f, Vector3.Dot(rot * n, Light));
                        var col = new Color(g, g, g, 1f);
                        int i0 = m_V.Count;
                        m_V.Add(centre + rot * (fc - du - dv));
                        m_V.Add(centre + rot * (fc + du - dv));
                        m_V.Add(centre + rot * (fc + du + dv));
                        m_V.Add(centre + rot * (fc - du + dv));
                        for (int k = 0; k < 4; k++) m_C.Add(col);
                        // clockwise seen from outside (Unity's front face): (u, v, n) keeps its handedness for +n only
                        if (sgn > 0) { m_T.Add(i0); m_T.Add(i0 + 1); m_T.Add(i0 + 2); m_T.Add(i0); m_T.Add(i0 + 2); m_T.Add(i0 + 3); }
                        else { m_T.Add(i0); m_T.Add(i0 + 2); m_T.Add(i0 + 1); m_T.Add(i0); m_T.Add(i0 + 3); m_T.Add(i0 + 2); }
                    }
            }

            public Mesh Mesh(string name)
            {
                var m = new Mesh { name = name };
                m.SetVertices(m_V);
                m.SetColors(m_C);
                m.SetTriangles(m_T, 0);
                m.RecalculateBounds();
                return m;
            }
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

        /// <summary>The camera pose in the head frame before the view offset: rgb_from_head · cam.</summary>
        public static void InHead(Pose rgbFromHead, Vector3 camP, Quaternion camQ, out Vector3 position, out Quaternion rotation)
        {
            position = rgbFromHead.rotation * camP + rgbFromHead.position;
            rotation = rgbFromHead.rotation * camQ;
        }

        /// <summary>world = head · view_offset · in_head.</summary>
        public static void Chain(Vector3 headP, Quaternion headQ, ViewOffset off, Vector3 inHeadP, Quaternion inHeadQ,
                                 out Vector3 position, out Quaternion rotation)
        {
            Quaternion r = off != null ? off.Rotation : Quaternion.identity;
            Vector3 t = off != null ? off.Translation : Vector3.zero;
            position = headP + headQ * (r * inHeadP + t);
            rotation = headQ * r * inHeadQ;
        }
    }
}
