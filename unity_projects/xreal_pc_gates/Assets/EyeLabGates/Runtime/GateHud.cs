using System;
using System.Collections.Generic;
using System.Globalization;
using Unity.Collections;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Operator display. Drawn once per view: the whole window, or each eye's half when the XR window
    /// is side by side (3840x1080). Per view: a large centred banner with the step title, a countdown
    /// and a progress bar (flashes bright on every step change); a large tracking line; a small grey
    /// camera preview at the bottom (only while frames arrive); diagnostics in small print.
    /// On the glasses black is transparent, so dark panels read as see-through.
    /// </summary>
    public sealed class GateHud : MonoBehaviour
    {
        const float FlashSeconds = 0.8f;
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        static readonly Color Panel = new Color(0f, 0f, 0f, 0.65f);
        static readonly Color Flash = new Color(1f, 0.85f, 0.1f, 0.95f);
        static readonly Color Amber = new Color(1f, 0.85f, 0.3f);
        static readonly Color Green = new Color(0.3f, 1f, 0.3f);
        static readonly Color Red = new Color(1f, 0.35f, 0.35f);
        static readonly Color Grey = new Color(0.75f, 0.75f, 0.75f);

        GateController m_Gate;
        CameraProbe m_Camera;
        PoseProbe m_Pose;
        PerfProbe m_Perf;
        string m_Build = "";
        readonly Dictionary<int, GUIStyle> m_Styles = new Dictionary<int, GUIStyle>();

        // Small RGB copy of the Y plane for display (Alpha8 does not draw reliably through IMGUI).
        const int PreviewWidth = 320;
        const double PreviewInterval = 1.0 / 15.0;
        Texture2D m_Preview;
        byte[] m_PreviewRgb;
        ulong m_PreviewTs;
        double m_LastPreview;

        public void Attach(GateController gate, CameraProbe camera, PoseProbe pose, PerfProbe perf, string build)
        {
            m_Gate = gate;
            m_Camera = camera;
            m_Pose = pose;
            m_Perf = perf;
            m_Build = build ?? "";
        }

        void Update()
        {
            CameraSource src = m_Gate != null ? m_Gate.Source : null;
            if (src == null || !src.IsCapturing) return;
            Texture2D luma = src.LumaTexture;
            double now = Time.realtimeSinceStartupAsDouble;
            if (luma == null || src.Timestamp == m_PreviewTs || now - m_LastPreview < PreviewInterval) return;

            int w = luma.width, h = luma.height;
            NativeArray<byte> data = luma.GetRawTextureData<byte>();
            if (w <= 0 || h <= 0 || data.Length < w * h) return;
            int pw = PreviewWidth, ph = Math.Max(1, PreviewWidth * h / w);
            if (m_Preview == null || m_Preview.width != pw || m_Preview.height != ph)
            {
                m_Preview = new Texture2D(pw, ph, TextureFormat.RGB24, false) { filterMode = FilterMode.Bilinear };
                m_PreviewRgb = new byte[pw * ph * 3];
            }
            for (int y = 0; y < ph; y++)
            {
                // Texture row 0 is drawn at the bottom; native row 0 is the top of the image.
                int srcRow = (h - 1 - (int)((y + 0.5) * h / ph)) * w;
                int dst = y * pw * 3;
                for (int x = 0; x < pw; x++)
                {
                    byte v = data[srcRow + (int)((x + 0.5) * w / pw)];
                    m_PreviewRgb[dst++] = v;
                    m_PreviewRgb[dst++] = v;
                    m_PreviewRgb[dst++] = v;
                }
            }
            m_Preview.LoadRawTextureData(m_PreviewRgb);
            m_Preview.Apply(false);
            m_PreviewTs = src.Timestamp;
            m_LastPreview = now;
        }

        void OnGUI()
        {
            if (m_Gate == null || Event.current.type != EventType.Repaint) return;
            bool sideBySide = m_Gate.XRActive && Screen.width >= 2 * Screen.height;
            if (sideBySide)
            {
                float half = Screen.width / 2f;
                DrawView(new Rect(0, 0, half, Screen.height));
                DrawView(new Rect(half, 0, half, Screen.height));
            }
            else
            {
                DrawView(new Rect(0, 0, Screen.width, Screen.height));
            }
        }

        void DrawView(Rect v)
        {
            double now = Time.realtimeSinceStartupAsDouble;
            float h = v.height;
            float pad = h * 0.015f;

            // ---- banner
            bool flash = now - m_Gate.StepStart < FlashSeconds;
            var header = Style(Mathf.RoundToInt(h / 36f), TextAnchor.UpperCenter, false);
            var title = Style(Mathf.RoundToInt(h / 12f), TextAnchor.MiddleCenter, true);
            var detail = Style(Mathf.RoundToInt(h / 30f), TextAnchor.UpperCenter, false);
            var count = Style(Mathf.RoundToInt(h / 10f), TextAnchor.MiddleRight, true);

            float bw = v.width * 0.84f;
            float bx = v.x + (v.width - bw) / 2f;
            float by = v.y + h * 0.05f;
            string stepNo = m_Gate.StepCount > 0 && m_Gate.StepIndex > 0 ? $"step {m_Gate.StepIndex}/{m_Gate.StepCount}" : "";
            string headerText = $"{m_Gate.GateTitle}   {stepNo}";
            string next = string.IsNullOrEmpty(m_Gate.NextTitle) ? "" : $"   next: {m_Gate.NextTitle}";
            string detailText = m_Gate.StepDetail + next;

            float headerH = header.CalcHeight(new GUIContent(headerText), bw);
            float titleH = Mathf.Max(title.CalcHeight(new GUIContent(m_Gate.StepTitle), bw * 0.8f), count.fontSize * 1.2f);
            float detailH = detail.CalcHeight(new GUIContent(detailText), bw);
            float barH = h * 0.012f;
            float bh = pad + headerH + titleH + detailH + barH + 2 * pad;

            Fill(new Rect(bx, by, bw, bh), flash ? Flash : Panel);
            float y = by + pad;
            Label(new Rect(bx, y, bw, headerH), headerText, header, flash ? Color.black : Grey);
            y += headerH;
            Label(new Rect(bx + bw * 0.1f, y, bw * 0.8f, titleH), m_Gate.StepTitle, title, flash ? Color.black : Amber);
            if (!double.IsNaN(m_Gate.StepEnd))
            {
                double left = Math.Max(0, m_Gate.StepEnd - now);
                Color c = flash ? Color.black : left <= 3.0 ? Red : Color.white;
                Label(new Rect(bx, y, bw - pad * 2, titleH), Math.Ceiling(left).ToString("F0", Inv), count, c);
            }
            y += titleH;
            Label(new Rect(bx, y, bw, detailH), detailText, detail, flash ? Color.black : Color.white);
            y += detailH + pad;
            if (!double.IsNaN(m_Gate.StepEnd) && m_Gate.StepEnd > m_Gate.StepStart)
            {
                float frac = Mathf.Clamp01((float)((now - m_Gate.StepStart) / (m_Gate.StepEnd - m_Gate.StepStart)));
                Fill(new Rect(bx + pad, y, bw - 2 * pad, barH), new Color(1f, 1f, 1f, 0.25f));
                Fill(new Rect(bx + pad, y, (bw - 2 * pad) * frac, barH), flash ? Color.black : Amber);
            }

            // ---- tracking line
            var big = Style(Mathf.RoundToInt(h / 20f), TextAnchor.MiddleCenter, true);
            int reason = m_Pose.LastTrackingReason;
            string trackText;
            Color trackColor;
            if (reason == NativeTracking.NoValue)
            {
                trackText = m_Gate.XRActive ? "TRACKING: n/a" : "";
                trackColor = Grey;
            }
            else if (reason == NativeTracking.Ready)
            {
                trackText = $"TRACKING READY ({m_Pose.LastTrackingType})";
                trackColor = Green;
            }
            else
            {
                trackText = $"TRACKING: {NativeTracking.Name(reason).ToUpperInvariant()} - look around slowly";
                trackColor = Red;
            }
            if (trackText.Length > 0)
            {
                float th = big.fontSize * 1.5f;
                float ty = by + bh + pad;
                Fill(new Rect(bx, ty, bw, th), Panel);
                Label(new Rect(bx, ty, bw, th), trackText, big, trackColor);
            }

            // ---- camera preview (grey, only while frames arrive)
            if (m_Gate.Source.IsCapturing && m_Preview != null)
            {
                float pw = v.width * 0.30f;
                float ph = pw * m_Preview.height / Mathf.Max(1, m_Preview.width);
                var pr = new Rect(v.x + (v.width - pw) / 2f, v.yMax - ph - h * 0.04f, pw, ph);
                Fill(new Rect(pr.x - 2, pr.y - 2, pr.width + 4, pr.height + 4), Color.black);
                GUI.DrawTexture(pr, m_Preview);
                double lag = m_Camera.LagGrowth;
                var small = Style(Mathf.RoundToInt(h / 40f), TextAnchor.LowerCenter, false);
                string lagText = double.IsNaN(lag) ? "" : $"   lag {(lag >= 0 ? "+" : "")}{lag.ToString("F1", Inv)} s";
                Label(new Rect(pr.x, pr.y - small.fontSize * 1.4f, pr.width, small.fontSize * 1.4f),
                      $"Eye camera {m_Camera.Fps.ToString("F0", Inv)} fps{lagText}", small, lag > 1.0 ? Red : Grey);
            }

            // ---- diagnostics (small, bottom left)
            var diag = Style(Mathf.RoundToInt(h / 46f), TextAnchor.LowerLeft, false);
            Vector3 p = m_Pose.LastPosition * 1000f;
            Vector3 e = m_Pose.LastRotation.eulerAngles;
            string diagText =
                $"head mm {p.x.ToString("F0", Inv)} {p.y.ToString("F0", Inv)} {p.z.ToString("F0", Inv)}  yaw {Wrap(e.y).ToString("F0", Inv)}\n" +
                $"cpu {m_Perf.CpuCoresBusy.ToString("F1", Inv)} cores  sys {Pct(m_Perf.SystemCpuPct)}  clock {Pct(m_Perf.CpuPerformancePct)}\n" +
                $"glasses {(m_Gate.GlassesConnected ? "ok" : "-")}  XR {(m_Gate.XRActive ? "on" : "off")}  marks {m_Gate.Marks}  build {m_Build}\n" +
                "Esc abort   M mark";
            float dw = v.width * 0.32f;
            float dh = diag.CalcHeight(new GUIContent(diagText), dw);
            Label(new Rect(v.x + pad * 2, v.yMax - dh - pad * 2, dw, dh), diagText, diag, Grey);
        }

        static string Pct(double v) => double.IsNaN(v) ? "-" : v.ToString("F0", Inv) + "%";

        static float Wrap(float deg) => deg > 180f ? deg - 360f : deg;

        GUIStyle Style(int size, TextAnchor anchor, bool bold)
        {
            int key = size * 64 + (int)anchor * 2 + (bold ? 1 : 0);
            if (!m_Styles.TryGetValue(key, out var s))
            {
                s = new GUIStyle(GUI.skin.label)
                {
                    fontSize = Math.Max(10, size),
                    alignment = anchor,
                    wordWrap = true,
                    richText = false,
                    fontStyle = bold ? FontStyle.Bold : FontStyle.Normal,
                };
                m_Styles[key] = s;
            }
            return s;
        }

        static void Fill(Rect r, Color c)
        {
            var old = GUI.color;
            GUI.color = c;
            GUI.DrawTexture(r, Texture2D.whiteTexture);
            GUI.color = old;
        }

        static void Label(Rect r, string text, GUIStyle style, Color c)
        {
            style.normal.textColor = c;
            GUI.Label(r, text, style);
        }
    }
}
