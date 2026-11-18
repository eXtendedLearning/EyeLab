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
        Text m_Text;
        Transform m_Reference;
        bool m_Placed;
        int m_PoseFrames;

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
            v.m_Reference = WorldReference(font);
            v.m_Reference.gameObject.SetActive(false);
            Application.onBeforeRender += v.FollowHead;
            Debug.Log($"[EyeLab probe] glasses view on '{cam.name}', stereo {cam.stereoEnabled}, target eye {cam.stereoTargetEye}");
            return v;
        }

        void OnDestroy() => Application.onBeforeRender -= FollowHead;

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

        void Update() => FollowHead();

        void FollowHead()
        {
            InputDevice head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            if (!head.isValid) return;
            bool okP = head.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 p);
            bool okR = head.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion r);
            if (!okR) return;
            m_Cam.transform.localRotation = r;
            if (okP) m_Cam.transform.localPosition = p;
            if (m_PoseFrames++ == 0) Debug.Log("[EyeLab probe] glasses view: head pose drives the camera");
            if (!m_Placed && m_PoseFrames > 30)   // let tracking settle for about half a second
            {
                Vector3 fwd = Vector3.ProjectOnPlane(m_Cam.transform.forward, Vector3.up);
                if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
                fwd.Normalize();
                m_Reference.position = m_Cam.transform.position + fwd * 1.5f;
                m_Reference.rotation = Quaternion.LookRotation(fwd, Vector3.up);
                m_Reference.gameObject.SetActive(true);
                m_Placed = true;
                Debug.Log($"[EyeLab probe] world reference placed at {m_Reference.position}");
            }
        }

        Text HeadLockedText(Font font)
        {
            var go = new GameObject("EyeLab Glasses Status", typeof(RectTransform), typeof(Canvas));
            go.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            var rt = (RectTransform)go.transform;
            rt.SetParent(m_Cam.transform, false);
            rt.sizeDelta = new Vector2(900f, 200f);          // canvas units of 1 mm
            rt.localScale = Vector3.one * 0.001f;
            rt.localPosition = new Vector3(0f, -0.18f, 2f);  // about 5 degrees below the centre
            var t = Label(go.transform, font, 64, new Color(0.35f, 1f, 0.45f));
            t.text = "EyeLab";
            return t;
        }

        static Transform WorldReference(Font font)
        {
            var go = new GameObject("EyeLab World Reference", typeof(RectTransform), typeof(Canvas));
            go.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
            var rt = (RectTransform)go.transform;
            rt.sizeDelta = new Vector2(400f, 400f);
            rt.localScale = Vector3.one * 0.001f;
            const float w = 6f;   // 6 mm lines
            var bars = new List<(Vector2 pos, Vector2 size)>
            {
                (new Vector2(0, 200 - w / 2), new Vector2(400, w)), (new Vector2(0, -200 + w / 2), new Vector2(400, w)),
                (new Vector2(-200 + w / 2, 0), new Vector2(w, 400)), (new Vector2(200 - w / 2, 0), new Vector2(w, 400)),
                (Vector2.zero, new Vector2(80, w)), (Vector2.zero, new Vector2(w, 80)),
            };
            foreach (var (pos, size) in bars)
            {
                var b = new GameObject("bar", typeof(RectTransform), typeof(Image));
                b.transform.SetParent(go.transform, false);
                var brt = (RectTransform)b.transform;
                brt.anchoredPosition = pos;
                brt.sizeDelta = size;
                var img = b.GetComponent<Image>();
                img.color = new Color(0.3f, 0.8f, 1f);
                img.raycastTarget = false;
            }
            var label = new GameObject("label", typeof(RectTransform));
            label.transform.SetParent(go.transform, false);
            var lrt = (RectTransform)label.transform;
            lrt.anchoredPosition = new Vector2(0, -250);
            lrt.sizeDelta = new Vector2(600, 60);
            var t = Label(label.transform, font, 36, new Color(0.3f, 0.8f, 1f));
            t.text = "fixed in the room (40 cm)";
            return go.transform;
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
