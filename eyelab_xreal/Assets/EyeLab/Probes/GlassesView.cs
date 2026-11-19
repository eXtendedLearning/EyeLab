using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

namespace EyeLab.Probes
{
    /// <summary>
    /// What the probes show in the glasses (T6.2). Jack, 2026-11-18, on 0.4.001: the glasses showed Unity's sky,
    /// out of focus and moving with the head, and not the status. Causes: the Probes scene (a SampleScene copy) has
    /// no tracked camera rig, so the camera never moved, and its clear was the skybox, which an optical see-through
    /// display shows as a grey veil (black is transparent). This component:
    /// <list type="bullet">
    /// <item>clears the XR camera to black (see-through);</item>
    /// <item>drives the camera from the XR head device just before rendering, so world content stays put;</item>
    /// <item>shows the status as a short head-locked line 2 m ahead, a little low;</item>
    /// <item>puts a 40 cm square and cross 1.5 m ahead in the world when tracking first comes up: if 6DoF holds,
    /// it stays where it is while walking around it (B2's visual check).</item>
    /// </list>
    /// </summary>
    public sealed class GlassesView : MonoBehaviour
    {
        Camera m_Cam;
        Text m_Text, m_Cue, m_Counter;
        RectTransform m_Bar;
        Transform m_Reference;
        bool m_Placed, m_Rendered;
        int m_PoseFrames;

        /// <summary>Seconds since app start of the first glasses frame and the first head pose (T6.3: Jack saw nothing for a minute).</summary>
        public float FirstFrameAt { get; private set; } = -1f;
        public float FirstPoseAt { get; private set; } = -1f;
        public float PlacedAt { get; private set; } = -1f;

        public static GlassesView Create(Font font)
        {
            Camera cam = Camera.main;
            if (cam == null)
            {
                Debug.LogWarning("[EyeLab probe] no main camera: nothing in the glasses");
                return null;
            }
            var v = cam.gameObject.AddComponent<GlassesView>();
            v.m_Cam = cam;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0f, 0f, 0f, 0f);
            cam.nearClipPlane = 0.05f;
            v.m_Text = v.HeadLockedText(font);
            v.HeadLockedCue(font);
            v.m_Reference = WorldReference(font);
            v.m_Reference.gameObject.SetActive(false);
            Application.onBeforeRender += v.FollowHead;
            Debug.Log($"[EyeLab probe] glasses view on '{cam.name}', stereo {cam.stereoEnabled}, target eye {cam.stereoTargetEye}");
            return v;
        }

        void OnDestroy() => Application.onBeforeRender -= FollowHead;

        /// <summary>T6.3 guided runs: the instruction large in the middle, a bar for the step's time left, "step i/n".</summary>
        public void SetCue(string instruction, float fractionLeft, string counter)
        {
            if (m_Cue == null) return;
            m_Cue.text = instruction ?? "";
            m_Counter.text = counter ?? "";
            m_Bar.gameObject.SetActive(fractionLeft > 0f);
            m_Bar.localScale = new Vector3(Mathf.Clamp01(fractionLeft), 1f, 1f);
        }

        public void ClearCue() => SetCue("", 0f, "");

        /// <summary>Put the room-fixed square 1.5 m ahead of where the head now looks (button "Place square").</summary>
        public void PlaceReference()
        {
            Vector3 fwd = Vector3.ProjectOnPlane(m_Cam.transform.forward, Vector3.up);
            if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
            fwd.Normalize();
            m_Reference.position = m_Cam.transform.position + fwd * 1.5f;
            m_Reference.rotation = Quaternion.LookRotation(fwd, Vector3.up);
            m_Reference.gameObject.SetActive(true);
            m_Placed = true;
            PlacedAt = Time.realtimeSinceStartup;
            Debug.Log($"[EyeLab probe] world reference placed at {m_Reference.position} (t {PlacedAt:F1} s)");
        }

        public void SetText(string s)
        {
            if (m_Text == null) return;
            int cut = s.IndexOf(" | ");
            string first = cut > 0 ? s.Substring(0, cut) : s;
            string rest = cut > 0 ? s.Substring(cut + 3) : "";
            int cut2 = rest.IndexOf(" | ");
            if (cut2 > 0) rest = rest.Substring(0, cut2);
            m_Text.text = rest.Length > 0 ? first + "\n" + rest : first;   // two short lines: what runs, time left
        }

        void Update()
        {
            FollowHead();
            if (Time.realtimeSinceStartup >= m_NextStats) { m_NextStats = Time.realtimeSinceStartup + 5f; LogDisplay(); }
        }

        float m_NextStats;
        int m_LastPresent = -1;
        static readonly List<XRDisplaySubsystem> s_Displays = new List<XRDisplaySubsystem>();

        /// <summary>
        /// T6.4: what the XR display itself reports every 5 s. In 0.4.003 Unity drew (first frame 7.9 s, 59 fps) and
        /// the glasses showed nothing; the present count says whether frames reach the glasses' compositor.
        /// </summary>
        void LogDisplay()
        {
            SubsystemManager.GetSubsystems(s_Displays);
            if (s_Displays.Count == 0) { Debug.Log("[EyeLab probe] xr display: no display subsystem"); return; }
            foreach (XRDisplaySubsystem d in s_Displays)
            {
                string v(bool ok, float x, string f) => ok ? x.ToString(f) : "-";
                bool okR = d.TryGetDisplayRefreshRate(out float hz);
                bool okP = d.TryGetFramePresentCount(out int present);
                bool okD = d.TryGetDroppedFrameCount(out int dropped);
                bool okA = d.TryGetAppGPUTimeLastFrame(out float app);
                bool okC = d.TryGetCompositorGPUTimeLastFrame(out float comp);
                bool okM = d.TryGetMotionToPhoton(out float m2p);
                string rate = okP && m_LastPresent >= 0 ? ((present - m_LastPresent) / 5f).ToString("F1") : "-";
                if (okP) m_LastPresent = present;
                Debug.Log($"[EyeLab probe] xr display: running {d.running}, passes {d.GetRenderPassCount()}, " +
                          $"refresh {v(okR, hz, "F1")} Hz, presented {(okP ? present.ToString() : "-")} ({rate}/s), " +
                          $"dropped {(okD ? dropped.ToString() : "-")}, app gpu {v(okA, app * 1000f, "F1")} ms, compositor gpu " +
                          $"{v(okC, comp * 1000f, "F1")} ms, motion-to-photon {v(okM, m2p * 1000f, "F1")} ms, eye texture " +
                          $"{XRSettings.eyeTextureWidth}x{XRSettings.eyeTextureHeight}, render scale {d.scaleOfAllRenderTargets:F2}, " +
                          $"unity fps {1f / Mathf.Max(Time.smoothDeltaTime, 1e-4f):F0}");
            }
        }

        void FollowHead()
        {
            if (!m_Rendered)
            {
                m_Rendered = true;
                FirstFrameAt = Time.realtimeSinceStartup;
                Debug.Log($"[EyeLab probe] glasses view: first frame at t {FirstFrameAt:F1} s");
            }
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            if (!head.isValid) return;
            bool okP = head.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 p);
            bool okR = head.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion r);
            if (!okR) return;
            m_Cam.transform.localRotation = r;
            if (okP) m_Cam.transform.localPosition = p;
            if (m_PoseFrames++ == 0)
            {
                FirstPoseAt = Time.realtimeSinceStartup;
                Debug.Log($"[EyeLab probe] glasses view: head pose drives the camera from t {FirstPoseAt:F1} s");
            }
            if (!m_Placed && m_PoseFrames > 30) PlaceReference();   // let tracking settle for about half a second
        }

        Text HeadLockedText(Font font)
        {
            var go = new GameObject("EyeLab Glasses Status", typeof(RectTransform), typeof(Canvas));
            go.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            var rt = (RectTransform)go.transform;
            rt.SetParent(m_Cam.transform, false);
            rt.sizeDelta = new Vector2(900f, 200f);          // canvas units of 1 mm
            rt.localScale = Vector3.one * 0.001f;
            rt.localPosition = new Vector3(0f, -0.26f, 2f);  // about 7 degrees below the centre
            var t = Label(go.transform, font, 44, new Color(0.35f, 1f, 0.45f));
            t.text = "EyeLab starting";
            return t;
        }

        void HeadLockedCue(Font font)
        {
            var go = new GameObject("EyeLab Glasses Cue", typeof(RectTransform), typeof(Canvas));
            go.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            var rt = (RectTransform)go.transform;
            rt.SetParent(m_Cam.transform, false);
            rt.sizeDelta = new Vector2(1000f, 220f);
            rt.localScale = Vector3.one * 0.001f;
            rt.localPosition = new Vector3(0f, -0.06f, 2f);
            m_Cue = Label(go.transform, font, 84, new Color(1f, 0.92f, 0.35f));
            var bar = new GameObject("bar", typeof(RectTransform), typeof(Image));
            bar.transform.SetParent(go.transform, false);
            m_Bar = (RectTransform)bar.transform;
            m_Bar.anchorMin = m_Bar.anchorMax = new Vector2(0.5f, 0f);
            m_Bar.pivot = new Vector2(0f, 0.5f);
            m_Bar.sizeDelta = new Vector2(600f, 14f);
            m_Bar.anchoredPosition = new Vector2(-300f, 0f);
            var img = bar.GetComponent<Image>();
            img.color = new Color(1f, 0.92f, 0.35f, 0.8f);
            img.raycastTarget = false;
            var c = new GameObject("counter", typeof(RectTransform));
            c.transform.SetParent(go.transform, false);
            var crt = (RectTransform)c.transform;
            crt.anchoredPosition = new Vector2(0f, 140f);
            crt.sizeDelta = new Vector2(600f, 50f);
            m_Counter = Label(c.transform, font, 36, new Color(1f, 0.92f, 0.35f, 0.8f));
            ClearCue();
        }

        static Transform WorldReference(Font font)
        {
            var go = new GameObject("EyeLab World Reference", typeof(RectTransform), typeof(Canvas));
            go.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            var rt = (RectTransform)go.transform;
            rt.sizeDelta = new Vector2(900f, 400f);
            rt.localScale = Vector3.one * 0.001f;
            // T6.2: thin hard-edged quads shimmered in the glasses (render scale 0.8, no MSAA). Draw the outline
            // and cross from one texture with soft edges and mipmaps, trilinear, so they stay steady while the
            // head moves even if MSAA is off.
            var sq = new GameObject("square", typeof(RectTransform), typeof(RawImage));
            sq.transform.SetParent(go.transform, false);
            var srt = (RectTransform)sq.transform;
            srt.sizeDelta = new Vector2(400f, 400f);
            srt.anchoredPosition = new Vector2(-250f, 0f);
            var raw = sq.GetComponent<RawImage>();
            raw.texture = SquareTexture(512, 10f);
            raw.color = new Color(0.3f, 0.8f, 1f);
            raw.raycastTarget = false;
            var label = new GameObject("label", typeof(RectTransform));
            label.transform.SetParent(go.transform, false);
            var lrt = (RectTransform)label.transform;
            lrt.anchoredPosition = new Vector2(0, -250);
            lrt.sizeDelta = new Vector2(600, 60);
            var t = Label(label.transform, font, 36, new Color(0.3f, 0.8f, 1f));
            t.text = "fixed in the room: soft square, hard grid (40 cm)";
            HardGrid(go.transform, new Vector2(250f, 0f));
            return go.transform;
        }

        /// <summary>
        /// T6.5: a 40 cm grid of hard-edged 3 mm strips with two diagonals, right of the soft square. The measurement
        /// overlay will be thin geometry like this, which aliases in a way the soft texture does not (MSAA and SMAA
        /// act on geometry edges, not on a texture's alpha).
        /// </summary>
        static void HardGrid(Transform parent, Vector2 centre)
        {
            var grid = new GameObject("hard grid", typeof(RectTransform));
            grid.transform.SetParent(parent, false);
            ((RectTransform)grid.transform).anchoredPosition = centre;
            var col = new Color(1f, 0.55f, 0.85f);
            void Strip(Vector2 at, Vector2 size, float angle)
            {
                var go = new GameObject("strip", typeof(RectTransform), typeof(Image));
                go.transform.SetParent(grid.transform, false);
                var rt = (RectTransform)go.transform;
                rt.sizeDelta = size;
                rt.anchoredPosition = at;
                rt.localRotation = Quaternion.Euler(0f, 0f, angle);
                var img = go.GetComponent<Image>();
                img.color = col;
                img.raycastTarget = false;
            }
            for (int i = 0; i < 5; i++)
            {
                float x = -200f + 100f * i;
                Strip(new Vector2(x, 0f), new Vector2(3f, 400f), 0f);
                Strip(new Vector2(0f, x), new Vector2(400f, 3f), 0f);
            }
            Strip(Vector2.zero, new Vector2(566f, 3f), 45f);
            Strip(Vector2.zero, new Vector2(566f, 3f), -45f);
        }

        Transform m_RefParent;
        Vector3 m_RefPos;
        Quaternion m_RefRot;

        /// <summary>
        /// T6.5: hold the square and grid 1.5 m ahead of the head instead of fixed in the room. If the flicker goes
        /// away head-locked, it comes from motion (frame pacing, pose prediction), not from aliasing.
        /// </summary>
        public void SetReferenceHeadLocked(bool headLocked)
        {
            if (m_Reference == null) return;
            if (headLocked && m_Reference.parent != m_Cam.transform)
            {
                m_RefParent = m_Reference.parent;
                m_RefPos = m_Reference.position;
                m_RefRot = m_Reference.rotation;
                m_Reference.SetParent(m_Cam.transform, false);
                m_Reference.localPosition = new Vector3(0f, 0f, 1.5f);
                m_Reference.localRotation = Quaternion.identity;
                m_Reference.gameObject.SetActive(true);
            }
            else if (!headLocked && m_Reference.parent == m_Cam.transform)
            {
                m_Reference.SetParent(m_RefParent, false);
                m_Reference.SetPositionAndRotation(m_RefPos, m_RefRot);
            }
        }

        /// <summary>White outline square and centre cross on transparent; line width in texels, about 1.5 texel soft edge.</summary>
        static Texture2D SquareTexture(int n, float line)
        {
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, true) { filterMode = FilterMode.Trilinear, anisoLevel = 4, wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[n * n];
            float half = line / 2f, c = (n - 1) / 2f, arm = n * 0.1f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float dEdge = Mathf.Min(Mathf.Min(x, n - 1 - x), Mathf.Min(y, n - 1 - y)) - half - 1f;   // outline, inset by half a line
                    float dx = Mathf.Abs(x - c), dy = Mathf.Abs(y - c);
                    float dCross = Mathf.Min(dx <= arm ? dy : float.MaxValue, dy <= arm ? dx : float.MaxValue);
                    float d = Mathf.Min(Mathf.Abs(dEdge), dCross);
                    float a = Mathf.Clamp01(half + 0.75f - d);
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            tex.SetPixels32(px);
            tex.Apply(true, true);
            return tex;
        }

        static Text Label(Transform parent, Font font, int size, Color color)
        {
            var go = new GameObject("Text", typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = rt.offsetMax = Vector2.zero;
            var t = go.GetComponent<Text>();
            t.font = font;
            t.fontSize = size;
            t.color = color;
            t.alignment = TextAnchor.MiddleCenter;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.raycastTarget = false;
            return t;
        }
    }
}
