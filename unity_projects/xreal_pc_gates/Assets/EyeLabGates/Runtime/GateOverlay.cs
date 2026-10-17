using System.Collections.Generic;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// G4 overlay: a world-locked 60 x 40 cm grid (10 cm cells) with square points at the corners and
    /// the centre, the stand-in for the measurement wireframe. One static mesh of thin quads (4 mm
    /// lines, double-sided, unlit); nothing is updated per frame, so the cost is one draw call per eye.
    /// <see cref="PlaceInFront"/> puts it on the gaze line at a fixed distance, upright and facing the
    /// viewer. The material is assigned by GateSceneBuilder so its shader is included in the build.
    /// </summary>
    public sealed class GateOverlay : MonoBehaviour
    {
        public const float Width = 0.60f, Height = 0.40f, Cell = 0.10f;
        const float Line = 0.004f, Point = 0.02f;

        [SerializeField] Material m_Material;

        GameObject m_Grid;

        public Material Material { get => m_Material; set => m_Material = value; }
        public bool Placed => m_Grid != null && m_Grid.activeSelf;
        public Vector3 Position => m_Grid != null ? m_Grid.transform.position : Vector3.zero;
        public float YawDeg => m_Grid != null ? m_Grid.transform.eulerAngles.y : 0f;

        /// <summary>Places the grid <paramref name="distance"/> m along the horizontal gaze of <paramref name="head"/>.</summary>
        public void PlaceInFront(Transform head, float distance)
        {
            if (head == null) return;
            if (m_Grid == null) m_Grid = Build();
            Vector3 forward = head.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;
            forward.Normalize();
            m_Grid.transform.SetPositionAndRotation(head.position + forward * distance, Quaternion.LookRotation(forward, Vector3.up));
            m_Grid.SetActive(true);
        }

        public void Hide()
        {
            if (m_Grid != null) m_Grid.SetActive(false);
        }

        // ---------------------------------------------------------------- G5 target overlay

        const float TargetLine = 0.0025f, TargetPoint = 0.008f;
        GameObject m_Target;

        public bool TargetBuilt => m_Target != null;
        public bool TargetVisible => m_Target != null && m_Target.activeSelf;

        /// <summary>
        /// Builds the G5 overlay from the pose service's geometry (target frame, OpenCV-handed, metres):
        /// each segment as two crossed 2.5 mm strips and each point as three crossed 8 mm squares, so
        /// they read from any angle. Vertices are converted with F = diag(1, -1, 1); the object's
        /// rotation is then F R F (see <see cref="SetTargetPose"/>). Hidden until the first lock.
        /// </summary>
        public void BuildTarget(TargetOverlayGeometry geometry)
        {
            if (m_Target != null) Destroy(m_Target);
            var verts = new List<Vector3>();
            var tris = new List<int>();
            float[] s = geometry.Segments;
            for (int i = 0; i + 5 < s.Length; i += 6)
                Strip(verts, tris, Flip(s[i], s[i + 1], s[i + 2]), Flip(s[i + 3], s[i + 4], s[i + 5]), TargetLine);
            float[] p = geometry.Points;
            for (int i = 0; i + 2 < p.Length; i += 3)
            {
                Vector3 c = Flip(p[i], p[i + 1], p[i + 2]);
                float h = TargetPoint / 2f;
                Strip(verts, tris, c - Vector3.right * h, c + Vector3.right * h, TargetPoint);
                Strip(verts, tris, c - Vector3.up * h, c + Vector3.up * h, TargetPoint);
            }
            var mesh = new Mesh { name = "GateTargetOverlay", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateBounds();
            m_Target = new GameObject("GateTargetOverlay");
            m_Target.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = m_Target.AddComponent<MeshRenderer>();
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.sharedMaterial = m_Material != null ? m_Material : new Material(Shader.Find("Unlit/Color"));
            m_Target.SetActive(false);
        }

        public void SetTargetPose(Vector3 position, Quaternion rotation)
        {
            if (m_Target == null) return;
            m_Target.transform.SetPositionAndRotation(position, rotation);
            m_Target.SetActive(true);
        }

        static Vector3 Flip(float x, float y, float z) => new Vector3(x, -y, z);

        /// <summary>Segment a-b as two crossed quads of the given width (double-sided).</summary>
        static void Strip(List<Vector3> v, List<int> t, Vector3 a, Vector3 b, float width)
        {
            Vector3 d = b - a;
            if (d.sqrMagnitude < 1e-12f) return;
            d.Normalize();
            Vector3 u = Vector3.Cross(d, Mathf.Abs(d.y) < 0.9f ? Vector3.up : Vector3.right).normalized * (width / 2f);
            Vector3 w = Vector3.Cross(d, u).normalized * (width / 2f);
            foreach (Vector3 off in new[] { u, w })
            {
                int i = v.Count;
                v.Add(a - off);
                v.Add(a + off);
                v.Add(b + off);
                v.Add(b - off);
                t.AddRange(new[] { i, i + 1, i + 2, i, i + 2, i + 3, i, i + 2, i + 1, i, i + 3, i + 2 });
            }
        }

        GameObject Build()
        {
            var verts = new List<Vector3>();
            var tris = new List<int>();
            float hw = Width / 2f, hh = Height / 2f;
            for (float x = -hw; x <= hw + 1e-4f; x += Cell)
                Quad(verts, tris, x - Line / 2, -hh, x + Line / 2, hh);
            for (float y = -hh; y <= hh + 1e-4f; y += Cell)
                Quad(verts, tris, -hw, y - Line / 2, hw, y + Line / 2);
            foreach (var p in new[] { new Vector2(-hw, -hh), new Vector2(hw, -hh), new Vector2(-hw, hh), new Vector2(hw, hh), Vector2.zero })
                Quad(verts, tris, p.x - Point / 2, p.y - Point / 2, p.x + Point / 2, p.y + Point / 2);

            var mesh = new Mesh { name = "GateOverlayGrid" };
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateBounds();

            var go = new GameObject("GateOverlayGrid");
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            if (m_Material != null)
            {
                renderer.sharedMaterial = m_Material;
            }
            else
            {
                var fallback = Shader.Find("Unlit/Color");
                if (fallback != null) renderer.sharedMaterial = new Material(fallback) { color = new Color(0.2f, 1f, 0.35f) };
                Debug.LogWarning("[EyeLabGates] GateOverlay material not assigned; using Shader.Find fallback");
            }
            return go;
        }

        /// <summary>Axis-aligned rectangle in the grid plane, both windings so it shows from either side.</summary>
        static void Quad(List<Vector3> v, List<int> t, float x0, float y0, float x1, float y1)
        {
            int i = v.Count;
            v.Add(new Vector3(x0, y0, 0));
            v.Add(new Vector3(x0, y1, 0));
            v.Add(new Vector3(x1, y1, 0));
            v.Add(new Vector3(x1, y0, 0));
            t.AddRange(new[] { i, i + 1, i + 2, i, i + 2, i + 3, i, i + 2, i + 1, i, i + 3, i + 2 });
        }
    }
}
