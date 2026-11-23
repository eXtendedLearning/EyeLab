using System.Collections.Generic;
using System.Globalization;
using Unity.XR.XREAL;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.11 (SPAAM): a crosshair fixed in ONE eye's display. The wearer moves the head until its centre sits on a real
    /// point (the centre of the ChArUco board's middle square), then taps. The crosshair is placed along a fixed direction
    /// from the SDK's own eye (<c>GetDevicePoseFromHead(DISPLAY_LEFT/RIGHT)</c>), so it stays at one display position
    /// whatever the head does. Each tap pairs that direction with the real point's position in the head (from the Eye
    /// camera and head tracking), which is what SPAAM fits the eye-display model from (Tuceryan &amp; Navab 2000;
    /// Grubert et al. 2017).
    /// <para>The direction is (tan x, tan y, 1) in the SDK eye's frame (x right, y up, z forward), drawn 1.5 m out. Only
    /// that eye draws it (<c>_EyeMask</c> in <c>EyeLabVertexColor.shader</c>); a gap in the centre leaves the target visible.</para>
    /// </summary>
    public sealed class SpaamCrosshair : MonoBehaviour
    {
        public const float DistanceM = 1.5f;
        const float GapDeg = 0.25f, ArmDeg = 1.2f, LineDeg = 0.08f;
        static readonly Color Green = new Color(0.35f, 1f, 0.45f);

        /// <summary>Crosshair directions in the eye's display, degrees (x right, y up): four corners of a box.</summary>
        public static readonly Vector2[] Positions = { new Vector2(-8f, 4f), new Vector2(8f, 4f), new Vector2(8f, -4f), new Vector2(-8f, -4f) };

        Pose m_Left = new Pose(new Vector3(-0.0315f, 0f, 0f), Quaternion.identity);
        Pose m_Right = new Pose(new Vector3(0.0315f, 0f, 0f), Quaternion.identity);
        Material m_Mat;
        Transform m_Cross;

        public string EyeSource { get; private set; } = "default ±31.5 mm";
        public string Eye { get; private set; } = "";
        public Vector2 Deg { get; private set; }
        public bool Visible => m_Cross != null && m_Cross.gameObject.activeSelf;

        public static SpaamCrosshair Create(ProbeRun run)
        {
            Camera cam = Camera.main;
            var go = new GameObject("EyeLab SPAAM crosshair");
            var c = go.AddComponent<SpaamCrosshair>();
            c.ReadEyes(run);
            c.Build(cam != null ? cam.transform : null);
            return c;
        }

        void ReadEyes(ProbeRun run)
        {
            var ci = CultureInfo.InvariantCulture;
            string P(Pose p) => $"p {p.position.x.ToString("R", ci)} {p.position.y.ToString("R", ci)} {p.position.z.ToString("R", ci)}; " +
                                $"q {p.rotation.x.ToString("R", ci)} {p.rotation.y.ToString("R", ci)} {p.rotation.z.ToString("R", ci)} {p.rotation.w.ToString("R", ci)}";
            bool okL = false, okR = false;
            try
            {
                Pose l = default, r = default;
                okL = XREALPlugin.GetDevicePoseFromHead(XREALComponent.XREAL_COMPONENT_DISPLAY_LEFT, ref l);
                okR = XREALPlugin.GetDevicePoseFromHead(XREALComponent.XREAL_COMPONENT_DISPLAY_RIGHT, ref r);
                if (okL && okR) { m_Left = l; m_Right = r; EyeSource = "GetDevicePoseFromHead(DISPLAY_LEFT/RIGHT)"; }
            }
            catch (System.Exception e)
            {
                run.Meta("display_pose_error", e.GetType().Name + ": " + e.Message);
            }
            run.Meta("display_left_from_head", (okL ? "ok; " : "unavailable; ") + P(m_Left));
            run.Meta("display_right_from_head", (okR ? "ok; " : "unavailable; ") + P(m_Right));
            run.Meta("spaam_eye_source", EyeSource);
            // the SDK's per-eye projections, for its own field of view and principal point
            foreach (var (name, comp) in new[] { ("left", XREALComponent.XREAL_COMPONENT_DISPLAY_LEFT), ("right", XREALComponent.XREAL_COMPONENT_DISPLAY_RIGHT) })
            {
                try
                {
                    Matrix4x4 m = default;
                    bool ok = XREALPlugin.GetCameraProjectionMatrix(comp, 0.1f, 100f, ref m);
                    var vals = new List<string>();
                    for (int i = 0; i < 16; i++) vals.Add(m[i].ToString("R", ci));   // column-major
                    run.Meta($"display_{name}_projection", (ok ? "ok; " : "unavailable; ") + string.Join(" ", vals));
                    Vector2Int res = default;
                    if (XREALPlugin.GetDeviceResolution(comp, ref res)) run.Meta($"display_{name}_resolution", $"{res.x}x{res.y}");
                }
                catch (System.Exception e)
                {
                    run.Meta($"display_{name}_projection", "error " + e.GetType().Name + ": " + e.Message);
                }
            }
            // what Unity's XR input reports for the eyes, against the head (centre eye)
            var dev = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.Head);
            if (dev.isValid && dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.leftEyePosition, out Vector3 le) &&
                dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.rightEyePosition, out Vector3 re) &&
                dev.TryGetFeatureValue(UnityEngine.XR.CommonUsages.centerEyePosition, out Vector3 ce))
                run.Meta("xr_eye_positions", $"left {(le - ce).ToString("F4")} right {(re - ce).ToString("F4")} (tracking axes, minus centre eye)");
            else run.Meta("xr_eye_positions", "unavailable");
            Camera cam = Camera.main;
            if (cam != null) run.Meta("camera_stereo_separation", cam.stereoSeparation.ToString("R", ci));
        }

        void Build(Transform head)
        {
            Shader sh = Resources.Load<Shader>("EyeLabVertexColor");
            m_Mat = sh != null ? new Material(sh) : new Material(Canvas.GetDefaultCanvasMaterial());
            m_Mat.color = Green;
            var go = new GameObject("crosshair", typeof(MeshFilter), typeof(MeshRenderer));
            m_Cross = go.transform;
            if (head != null) m_Cross.SetParent(head, false);
            float R = DistanceM;
            float g = R * Mathf.Tan(GapDeg * Mathf.Deg2Rad), a = R * Mathf.Tan(ArmDeg * Mathf.Deg2Rad), w = R * Mathf.Tan(LineDeg * Mathf.Deg2Rad) / 2f;
            var v = new List<Vector3>();
            var tri = new List<int>();
            void Quad(float x0, float y0, float x1, float y1)
            {
                int i = v.Count;
                v.Add(new Vector3(x0, y0, 0f)); v.Add(new Vector3(x0, y1, 0f)); v.Add(new Vector3(x1, y1, 0f)); v.Add(new Vector3(x1, y0, 0f));
                tri.AddRange(new[] { i, i + 1, i + 2, i, i + 2, i + 3 });
            }
            Quad(g, -w, a, w);
            Quad(-a, -w, -g, w);
            Quad(-w, g, w, a);
            Quad(-w, -a, w, -g);
            var mesh = new Mesh { name = "spaam crosshair" };
            mesh.SetVertices(v);
            var col = new List<Color>();
            for (int i = 0; i < v.Count; i++) col.Add(Color.white);
            mesh.SetColors(col);
            mesh.SetTriangles(tri, 0);
            mesh.RecalculateBounds();
            mesh.bounds = new Bounds(Vector3.zero, Vector3.one);   // never culled by a frustum test on the centre eye
            go.GetComponent<MeshFilter>().sharedMesh = mesh;
            go.GetComponent<MeshRenderer>().sharedMaterial = m_Mat;
            go.SetActive(false);
        }

        /// <summary>Shows the crosshair to <paramref name="eye"/> ("L" or "R") at <paramref name="deg"/> in that eye's display.</summary>
        public void Show(string eye, Vector2 deg)
        {
            if (m_Cross == null) return;
            Pose e = eye == "L" ? m_Left : m_Right;
            var dir = new Vector3(Mathf.Tan(deg.x * Mathf.Deg2Rad), Mathf.Tan(deg.y * Mathf.Deg2Rad), 1f);
            Vector3 inEye = dir.normalized * DistanceM;
            m_Cross.localPosition = e.position + e.rotation * inEye;
            m_Cross.localRotation = e.rotation * Quaternion.LookRotation(dir, Vector3.up);
            m_Mat.SetFloat("_EyeMask", eye == "L" ? 1f : 2f);
            m_Cross.gameObject.SetActive(true);
            Eye = eye;
            Deg = deg;
        }

        public void Hide()
        {
            if (m_Cross != null) m_Cross.gameObject.SetActive(false);
            Eye = "";
        }

        void OnDestroy()
        {
            if (m_Cross != null) Destroy(m_Cross.gameObject);
            if (m_Mat != null) Destroy(m_Mat);
        }
    }
}
