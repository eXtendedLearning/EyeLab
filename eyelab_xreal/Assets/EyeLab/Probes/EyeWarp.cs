using Unity.XR.XREAL;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.12: feeds the overlay shader's per-eye correction (<c>_EyeWarp</c> in <c>EyeLabVertexColor.shader</c>). Every
    /// frame before rendering it sets each SDK eye's world-to-eye matrix (head pose × <c>GetDevicePoseFromHead(DISPLAY_*)</c>,
    /// the frame the SPAAM taps were fitted in) and the eye model (centre C in metres, w = 1/D). Which model is drawn
    /// (none = the SDK's, this round's, or all rounds') is chosen with <see cref="Use"/>; materials opt in with
    /// <c>_EyeWarp = 1</c>.
    /// </summary>
    public sealed class EyeWarp : MonoBehaviour
    {
        static readonly int W2E0 = Shader.PropertyToID("_EyeW2E0"), W2E1 = Shader.PropertyToID("_EyeW2E1");
        static readonly int E2W0 = Shader.PropertyToID("_EyeE2W0"), E2W1 = Shader.PropertyToID("_EyeE2W1");
        static readonly int M0 = Shader.PropertyToID("_EyeModel0"), M1 = Shader.PropertyToID("_EyeModel1");
        static readonly int D0 = Shader.PropertyToID("_EyeDisplay0"), D1 = Shader.PropertyToID("_EyeDisplay1");

        Pose m_Left = new Pose(new Vector3(-0.0315f, 0f, 0f), Quaternion.identity);
        Pose m_Right = new Pose(new Vector3(0.0315f, 0f, 0f), Quaternion.identity);
        Camera m_Cam;

        /// <summary>The model now drawn: null = the SDK's (no correction).</summary>
        public double[] Left { get; private set; }
        public double[] Right { get; private set; }
        public bool On => Left != null && Right != null;

        public static EyeWarp Create()
        {
            var w = new GameObject("EyeLab eye warp").AddComponent<EyeWarp>();
            try
            {
                Pose l = default, r = default;
                if (XREALPlugin.GetDevicePoseFromHead(XREALComponent.XREAL_COMPONENT_DISPLAY_LEFT, ref l) &&
                    XREALPlugin.GetDevicePoseFromHead(XREALComponent.XREAL_COMPONENT_DISPLAY_RIGHT, ref r))
                {
                    w.m_Left = l;
                    w.m_Right = r;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[EyeLab probe] eye warp: display poses unavailable: " + e.Message);
            }
            return w;
        }

        public void Use(double[] left, double[] right)
        {
            Left = left;
            Right = right;
            Push();
        }

        void OnEnable() => Application.onBeforeRender += Push;
        void OnDisable() => Application.onBeforeRender -= Push;

        void Push()
        {
            if (m_Cam == null) m_Cam = Camera.main;
            if (m_Cam == null) return;
            Transform h = m_Cam.transform;
            Matrix4x4 head = Matrix4x4.TRS(h.position, h.rotation, Vector3.one);
            Matrix4x4 e2w0 = head * Matrix4x4.TRS(m_Left.position, m_Left.rotation, Vector3.one);
            Matrix4x4 e2w1 = head * Matrix4x4.TRS(m_Right.position, m_Right.rotation, Vector3.one);
            Shader.SetGlobalMatrix(E2W0, e2w0);
            Shader.SetGlobalMatrix(E2W1, e2w1);
            Shader.SetGlobalMatrix(W2E0, e2w0.inverse);
            Shader.SetGlobalMatrix(W2E1, e2w1.inverse);
            Shader.SetGlobalVector(M0, V(Left));
            Shader.SetGlobalVector(M1, V(Right));
            Shader.SetGlobalVector(D0, Display(Left));
            Shader.SetGlobalVector(D1, Display(Right));
        }

        static Vector4 V(double[] c) => c == null ? new Vector4(0f, 0f, 0f, (float)EyeModel.W)
                                                  : new Vector4((float)c[0], (float)c[1], (float)c[2], (float)EyeModel.W);

        /// <summary>The display's centre p (tan units) and scale s, as (p_x, p_y, s, 0); the SDK's is (0, 0, 1, 0).</summary>
        static Vector4 Display(double[] c) => c == null ? new Vector4(0f, 0f, 1f, 0f) : new Vector4((float)c[4], (float)c[5], (float)c[3], 0f);
    }
}
