using System.Collections;
using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.UI;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.1 (gate B1) optical timecode on the Beam Pro's own screen. A DICT_6X6_1000 marker whose id is a draw
    /// counter mod 1000 is redrawn every Unity frame on the phone-screen canvas (the XREAL virtual controller's
    /// canvas, rendered by a non-XR camera). Each draw is logged on the same clocks as the Eye frames
    /// (<c>draws.csv</c>), so frame arrival minus draw time needs no clock synchronisation. The Eye is pointed at
    /// the phone screen; the saved Eye frames are decoded offline (<c>python -m xreal_gates bp RUN</c>).
    /// <para>What it measures is an upper bound on exposure → arrival: the excess is the phone panel's own
    /// presentation delay plus the wait for the next exposure (0 .. 1 frame period). Same method as the Surface
    /// timecode (guide §24-25); the photodiode or slow-motion check calibrates the panel term later.</para>
    /// </summary>
    public sealed class TimecodeScreen : MonoBehaviour
    {
        const string File = "draws.csv";
        const string Header = "counter,id,unity_frame,set_mono_ns,set_boot_ns,set_real_ns,set_hmd_ns,set_mono_after_ns,set_unity_s,eof_mono_ns,eof_boot_ns";

        ProbeRun m_Run;
        Texture2D m_Tex;
        Color32[] m_Pixels;
        GameObject m_Root;
        int m_Counter;
        bool m_Running;
        public string Host { get; private set; } = "";

        public static TimecodeScreen Begin(ProbeRun run, float sideFraction = 0.8f)
        {
            var go = new GameObject("EyeLab Timecode");
            var tc = go.AddComponent<TimecodeScreen>();
            tc.m_Run = run;
            tc.Build(sideFraction);
            tc.m_Running = true;
            tc.StartCoroutine(tc.EndOfFrameLoop());
            return tc;
        }

        void Build(float sideFraction)
        {
            Canvas canvas = null;
            if (XREALVirtualController.Singleton != null)
                canvas = XREALVirtualController.Singleton.GetComponentInChildren<Canvas>(true);
            if (canvas != null)
            {
                Host = "virtual_controller_canvas";
            }
            else
            {
                // Fallback: an overlay canvas. Whether it reaches the phone screen while XR runs is what B1 checks.
                var cgo = new GameObject("EyeLab Timecode Canvas");
                canvas = cgo.AddComponent<Canvas>();
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                canvas.sortingOrder = 32000;
                Host = "own_overlay_canvas";
            }
            m_Run.Meta("timecode_host", Host);

            m_Root = new GameObject("Timecode quiet zone", typeof(RectTransform), typeof(Image));
            m_Root.transform.SetParent(canvas.transform, false);
            m_Root.transform.SetAsLastSibling();
            var quiet = m_Root.GetComponent<Image>();
            quiet.color = Color.white;
            quiet.raycastTarget = false;
            var qrt = (RectTransform)m_Root.transform;
            qrt.anchorMin = qrt.anchorMax = qrt.pivot = new Vector2(0.5f, 0.5f);
            Rect cr = ((RectTransform)canvas.transform).rect;
            float side = Mathf.Min(cr.width, cr.height) * sideFraction;
            if (side <= 1f) side = Mathf.Min(Screen.width, Screen.height) * sideFraction;
            qrt.sizeDelta = new Vector2(side, side);

            var mgo = new GameObject("Timecode marker", typeof(RectTransform), typeof(RawImage));
            mgo.transform.SetParent(m_Root.transform, false);
            var mrt = (RectTransform)mgo.transform;
            mrt.anchorMin = mrt.anchorMax = mrt.pivot = new Vector2(0.5f, 0.5f);
            mrt.sizeDelta = new Vector2(side * 0.8f, side * 0.8f);   // 10 % quiet zone each side
            m_Tex = new Texture2D(8, 8, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
            m_Pixels = new Color32[64];
            var raw = mgo.GetComponent<RawImage>();
            raw.texture = m_Tex;
            raw.raycastTarget = false;
            m_Run.Meta("timecode_side_px", ProbeRun.F(side * 0.8f));
        }

        void Update()
        {
            if (!m_Running) return;
            int id = m_Counter % TimecodeBits.Count;
            for (int row = 0; row < 8; row++)
                for (int col = 0; col < 8; col++)
                {
                    byte v = TimecodeBits.White(id, row, col) ? (byte)255 : (byte)0;
                    m_Pixels[(7 - row) * 8 + col] = new Color32(v, v, v, 255);   // texture row 0 is the bottom
                }
            m_Tex.SetPixels32(m_Pixels);
            m_Tex.Apply(false);
            m_SetClocks = ProbeClocks.Now();
            m_SetFrame = Time.frameCount;
            m_SetId = id;
            m_SetCounter = m_Counter;
            m_Pending = true;
            m_Counter++;
        }

        ProbeClocks.Sample m_SetClocks;
        int m_SetFrame, m_SetId, m_SetCounter;
        bool m_Pending;

        IEnumerator EndOfFrameLoop()
        {
            var eof = new WaitForEndOfFrame();
            while (true)
            {
                yield return eof;
                if (!m_Pending) continue;
                m_Pending = false;
                long mono = ProbeClocks.MonoNs, boot = ProbeClocks.BootNs;
                m_Run.Row(File, Header, $"{m_SetCounter},{m_SetId},{m_SetFrame},{m_SetClocks.Csv()},{mono},{boot}");
            }
        }

        public void End()
        {
            m_Running = false;
            m_Run.Meta("timecode_draws", m_Counter.ToString(System.Globalization.CultureInfo.InvariantCulture));
            if (m_Root != null) Destroy(m_Root);
            Destroy(gameObject);
        }
    }
}
