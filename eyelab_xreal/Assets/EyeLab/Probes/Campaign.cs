using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// Campaign 0.5 (docs/tasks/T7.0-campaign-0.5.md §4, C6): the hammering trial. Everything that may change during the
    /// campaign is data, read at the start of each trial, so one APK serves the whole campaign:
    /// <list type="bullet">
    /// <item><c>files/campaign/protocol.json</c> (pushed), else <c>Resources/campaign/protocol</c>, else the defaults below;</item>
    /// <item><c>files/campaign/sheets/&lt;name&gt;.json</c> (pushed), else <c>Resources/campaign/sheets/&lt;name&gt;</c>
    /// (written by <c>python -m campaign sheets</c>, the same files the print and the analysis use).</item>
    /// </list>
    /// The trial (participant, condition, sheet) is chosen on the Beam Pro screen and kept in <c>files/campaign/state.json</c>.
    /// Each finished trial appends a row to <c>files/campaign/sheets_used.csv</c>, which maps sheet to participant and
    /// condition for <c>python -m campaign summary</c>.
    /// </summary>
    public static class Campaign
    {
        [Serializable]
        public sealed class Protocol
        {
            public string id = "0.5-default";
            public float ready_s = 120f;          // ends early on NEXT
            public float target_max_s = 120f;     // ends early on NEXT
            public float review_s = 6f;           // review look: every processed Eye frame saved
            public float end_s = 3f;
            public float cross_arm_mm = 10f;      // arm end, from the centre
            public float cross_gap_mm = 2f;       // arm start, from the centre (open centre)
            public float cross_line_mm = 1.2f;
            public float cross_depth_mm = 0.5f;
            public float[] cross_rgb = { 0.35f, 1f, 0.45f };
            public string display_ar = "all";     // eye model for condition A and T: "all" (pooled) or "sdk"
            public bool speak_numbers = true;
        }

        [Serializable]
        public sealed class SheetTarget
        {
            public int n;
            public float[] page_mm;
            public float[] model_m;
        }

        [Serializable]
        public sealed class Sheet
        {
            public string schema;
            public string name;
            public string kind;
            public int number;
            public string digest;
            public SheetTarget[] targets;
        }

        [Serializable]
        public sealed class State
        {
            public int participant = 0;     // 0 = Jack (P00)
            public string condition = "A";  // A (AR, pooled display) | S (AR, SDK display) | P (printed) | T (practice)
            public int sheet = 41;
        }

        public static readonly string[] Conditions = { "A", "S", "P", "T" };
        public static readonly string[] ConditionNames = { "AR", "AR-SDK", "Printed", "Practice" };

        public static string Dir => Path.Combine(Application.persistentDataPath, "campaign");
        static string StatePath => Path.Combine(Dir, "state.json");
        public const string UsedFile = "sheets_used.csv";
        const string UsedHeader = "sheet,participant,condition,display,protocol,run,sheet_digest,targets_shown,next_presses,bad_marks,aborted,local_time";

        public static State Current { get; private set; } = LoadState();
        public static Protocol Proto { get; private set; } = new Protocol();
        public static Sheet ActiveSheet { get; private set; }
        public static bool Running { get; private set; }
        public static int TargetsShown, NextPresses, BadMarks;
        public static string LastTargetStep = "";
        public static bool Aborted;
        static ProbeRun s_Run;

        // ------------------------------------------------------------------ state
        static State LoadState()
        {
            try { if (File.Exists(StatePath)) return JsonUtility.FromJson<State>(File.ReadAllText(StatePath)) ?? new State(); }
            catch (Exception e) { Debug.LogWarning("[EyeLab campaign] state: " + e.Message); }
            return new State();
        }

        public static void SaveState()
        {
            try { Directory.CreateDirectory(Dir); File.WriteAllText(StatePath, JsonUtility.ToJson(Current, true)); }
            catch (Exception e) { Debug.LogWarning("[EyeLab campaign] state not saved: " + e.Message); }
        }

        public static string Kind(string condition) => condition == "P" ? "P" : condition == "T" ? "T" : "M";
        public static string ParticipantId => "P" + Current.participant.ToString("D2", CultureInfo.InvariantCulture);
        public static string SheetName(int number, string kind) => kind + number.ToString("D3", CultureInfo.InvariantCulture);
        public static string CurrentSheetName => SheetName(Current.sheet, Kind(Current.condition));
        public static string ConditionName(string c) { int i = Array.IndexOf(Conditions, c); return i >= 0 ? ConditionNames[i] : c; }

        /// <summary>The eye model the overlay uses in this condition: SDK for S, else the protocol's (pooled by default).</summary>
        public static string DisplayFor(string condition) => condition == "S" ? "sdk" : Proto.display_ar;
        public static bool ShowsCross(string condition) => condition != "P";

        public static void SetParticipant(int delta) { Current.participant = Mathf.Clamp(Current.participant + delta, 0, 99); SaveState(); }

        public static void SetCondition(string c)
        {
            string before = Kind(Current.condition);
            Current.condition = c;
            if (Kind(c) != before) Current.sheet = FirstNumber(Kind(c));
            SaveState();
        }

        public static void StepSheet(int delta)
        {
            string k = Kind(Current.condition);
            int n = Current.sheet;
            for (int i = 0; i < 200; i++)
            {
                n += delta;
                if (n < 1 || n > 249) break;
                if (Exists(SheetName(n, k))) { Current.sheet = n; SaveState(); return; }
            }
        }

        static int FirstNumber(string kind) => kind == "P" ? 11 : kind == "T" ? 1 : 41;

        // ------------------------------------------------------------------ files
        static string ReadText(string relative, string resource)
        {
            string path = Path.Combine(Dir, relative);
            try { if (File.Exists(path)) return File.ReadAllText(path); }
            catch (Exception e) { Debug.LogWarning("[EyeLab campaign] " + path + ": " + e.Message); }
            var ta = Resources.Load<TextAsset>(resource);
            return ta != null ? ta.text : null;
        }

        static bool Exists(string name) =>
            File.Exists(Path.Combine(Dir, "sheets", name + ".json")) || Resources.Load<TextAsset>("campaign/sheets/" + name) != null;

        public static Protocol LoadProtocol(out string source)
        {
            string text = ReadText("protocol.json", "campaign/protocol");
            source = text == null ? "defaults" : File.Exists(Path.Combine(Dir, "protocol.json")) ? "files/campaign/protocol.json" : "Resources";
            var p = new Protocol();
            if (text != null)
            {
                try { JsonUtility.FromJsonOverwrite(text, p); }
                catch (Exception e) { source = "defaults (protocol unreadable: " + e.Message + ")"; p = new Protocol(); }
            }
            return p;
        }

        public static Sheet LoadSheet(string name)
        {
            string text = ReadText(Path.Combine("sheets", name + ".json"), "campaign/sheets/" + name);
            if (text == null) return null;
            try { return JsonUtility.FromJson<Sheet>(text); }
            catch (Exception e) { Debug.LogWarning("[EyeLab campaign] sheet " + name + ": " + e.Message); return null; }
        }

        // ------------------------------------------------------------------ trial
        /// <summary>Loads the protocol and the current sheet; false (with the reason) when the sheet is missing.</summary>
        public static bool Prepare(out string why)
        {
            Proto = LoadProtocol(out string src);
            ActiveSheet = LoadSheet(CurrentSheetName);
            TargetsShown = NextPresses = BadMarks = 0;
            LastTargetStep = "";
            Aborted = false;
            why = ActiveSheet == null ? "sheet " + CurrentSheetName + " not found (push python/config/campaign/sheets or rebuild)"
                : ActiveSheet.targets == null || ActiveSheet.targets.Length == 0 ? "sheet " + CurrentSheetName + " has no targets" : "protocol " + Proto.id + " from " + src;
            return ActiveSheet != null && ActiveSheet.targets != null && ActiveSheet.targets.Length > 0;
        }

        public static void Begin(ProbeRun run)
        {
            s_Run = run;
            Running = true;
            var ci = CultureInfo.InvariantCulture;
            run.Meta("campaign_participant", ParticipantId);
            run.Meta("campaign_condition", Current.condition + " (" + ConditionName(Current.condition) + ")");
            run.Meta("campaign_sheet", ActiveSheet.name + " digest " + ActiveSheet.digest);
            run.Meta("campaign_display", ShowsCross(Current.condition) ? DisplayFor(Current.condition) : "none (printed)");
            run.Meta("campaign_protocol", JsonUtility.ToJson(Proto));
            var b = new System.Text.StringBuilder();
            foreach (SheetTarget t in ActiveSheet.targets)
                b.Append(t.n.ToString(ci)).Append(':').Append(t.model_m[0].ToString("R", ci)).Append(' ').Append(t.model_m[1].ToString("R", ci)).Append(';');
            run.Meta("campaign_targets_model_m", b.ToString());
        }

        /// <summary>The trial's cue steps: ready, one step per target (each ends on NEXT), review look, end.</summary>
        public static List<CueStep> Cues()
        {
            var steps = new List<CueStep>();
            Sheet s = ActiveSheet;
            string c = Current.condition;
            int n = s.targets.Length;
            string spoken = s.kind + " " + s.number.ToString(CultureInfo.InvariantCulture);
            steps.Add(new CueStep("ready", $"Sheet {spoken}. Look at the sheet. The trial starts at next.", Proto.ready_s,
                                  $"Sheet {s.name}: look at the sheet. NEXT to start"));
            for (int k = 1; k <= n; k++)
            {
                string say = !Proto.speak_numbers ? "Next point." : c == "P" ? $"Point {k}: the printed cross number {k}." : $"Point {k}.";
                string show = c == "P" ? $"Hit the printed cross {k}  ({k}/{n})" : $"Point {k} of {n}";
                steps.Add(new CueStep("tgt_" + k.ToString("D2", CultureInfo.InvariantCulture), say, Proto.target_max_s, show));
            }
            steps.Add(new CueStep("review", "Done. Now look straight down at the sheet, from about thirty centimetres, and hold still.",
                                  Proto.review_s, "Look DOWN at the sheet, 30 cm: HOLD STILL", ticks: true));
            steps.Add(new CueStep("end", "Thank you. The sheet can go.", Proto.end_s, "Done: remove the sheet"));
            return steps;
        }

        /// <summary>The target of a "tgt_k" step, in the sheet's model frame (metres, OpenCV-handed as the set's targets).</summary>
        public static bool TargetOfStep(string stepId, out int k, out Vector3 model)
        {
            k = 0;
            model = Vector3.zero;
            if (ActiveSheet?.targets == null || stepId == null || !stepId.StartsWith("tgt_")) return false;
            if (!int.TryParse(stepId.Substring(4), NumberStyles.Integer, CultureInfo.InvariantCulture, out k)) return false;
            foreach (SheetTarget t in ActiveSheet.targets)
                if (t.n == k && t.model_m != null && t.model_m.Length >= 3) { model = new Vector3(t.model_m[0], t.model_m[1], t.model_m[2]); return true; }
            return false;
        }

        /// <summary>NEXT (button or presenter): the current target is done; the step ends.</summary>
        public static void Next(string source)
        {
            if (!Running) return;
            NextPresses++;
            string step = DetectProbe.ActiveStep;
            if (step.StartsWith("tgt_")) LastTargetStep = step;
            if (s_Run != null) CuePlayer.Log(s_Run, "mark_next", step + ":" + source);
            CuePlayer.Ensure().SkipStep();
        }

        /// <summary>The last finished target was a bad hit (tool slipped, two dots...): flagged for the analysis.</summary>
        public static void Bad(string source)
        {
            if (!Running) return;
            BadMarks++;
            string step = DetectProbe.ActiveStep;
            string which = step.StartsWith("tgt_") && LastTargetStep.Length == 0 ? step : LastTargetStep.Length > 0 ? LastTargetStep : step;
            if (s_Run != null) CuePlayer.Log(s_Run, "mark_bad", which + ":" + source + "@" + step);
            CuePlayer.Ensure().Say("Marked bad.");
        }

        public static void Abort()
        {
            if (!Running) return;
            Aborted = true;
            if (s_Run != null) CuePlayer.Log(s_Run, "mark_abort", DetectProbe.ActiveStep);
        }

        /// <summary>After the run: the sheets_used row, and the next sheet of the same kind selected.</summary>
        public static void End(ProbeRun run)
        {
            Running = false;
            s_Run = null;
            if (ActiveSheet == null) return;
            var ci = CultureInfo.InvariantCulture;
            string row = string.Join(",", ActiveSheet.name, ParticipantId, Current.condition,
                ShowsCross(Current.condition) ? DisplayFor(Current.condition) : "none", Proto.id, Path.GetFileName(run.Dir),
                ActiveSheet.digest, TargetsShown.ToString(ci), NextPresses.ToString(ci), BadMarks.ToString(ci), Aborted ? "1" : "0",
                DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", ci));
            run.Meta("campaign_result", row);
            try
            {
                Directory.CreateDirectory(Dir);
                string path = Path.Combine(Dir, UsedFile);
                bool head = !File.Exists(path);
                File.AppendAllText(path, (head ? UsedHeader + "\n" : "") + row + "\n");
            }
            catch (Exception e) { Debug.LogWarning("[EyeLab campaign] sheets_used not written: " + e.Message); }
            if (!Aborted) StepSheet(+1);
        }
    }

    /// <summary>
    /// Campaign 0.5: the crosshair of the current target, drawn on the sheet's plane at the sheet's pose and per eye with
    /// the eye model (EyeLabVertexColor with _EyeWarp, as the T6.12 solid overlay). An open centre so the dot shows.
    /// </summary>
    public sealed class MarkCross : MonoBehaviour
    {
        Transform m_Space, m_Cross;
        Material m_Mat;
        bool m_Placed, m_Wanted;

        public string ShaderName { get; private set; } = "none";

        public static MarkCross Create(Campaign.Protocol p)
        {
            var go = new GameObject("EyeLab Mark Cross");
            var m = go.AddComponent<MarkCross>();
            Camera cam = Camera.main;
            m.m_Space = cam != null ? cam.transform.parent : null;
            var cross = new GameObject("cross", typeof(MeshFilter), typeof(MeshRenderer));
            cross.transform.SetParent(go.transform, false);
            m.m_Cross = cross.transform;
            Shader sh = Resources.Load<Shader>("EyeLabVertexColor");
            m.m_Mat = sh != null && sh.isSupported ? new Material(sh) : new Material(Shader.Find("Sprites/Default"));
            m.ShaderName = m.m_Mat.shader.name;
            Color c = p.cross_rgb != null && p.cross_rgb.Length >= 3 ? new Color(p.cross_rgb[0], p.cross_rgb[1], p.cross_rgb[2]) : Color.green;
            m.m_Mat.color = c;
            m.m_Mat.SetFloat("_EyeWarp", 1f);
            cross.GetComponent<MeshFilter>().sharedMesh = Build(p.cross_gap_mm / 1000f, p.cross_arm_mm / 1000f, p.cross_line_mm / 1000f, p.cross_depth_mm / 1000f);
            cross.GetComponent<MeshRenderer>().sharedMaterial = m.m_Mat;
            go.SetActive(false);
            return m;
        }

        /// <summary>Four boxes in the target plane (local x, y), from <paramref name="gap"/> to <paramref name="arm"/>.</summary>
        static Mesh Build(float gap, float arm, float line, float depth)
        {
            var v = new List<Vector3>();
            var t = new List<int>();
            float len = arm - gap, mid = (arm + gap) / 2f;
            Box(v, t, new Vector3(mid, 0f, 0f), new Vector3(len, line, depth));
            Box(v, t, new Vector3(-mid, 0f, 0f), new Vector3(len, line, depth));
            Box(v, t, new Vector3(0f, mid, 0f), new Vector3(line, len, depth));
            Box(v, t, new Vector3(0f, -mid, 0f), new Vector3(line, len, depth));
            var mesh = new Mesh { name = "mark cross" };
            mesh.SetVertices(v);
            mesh.SetTriangles(t, 0);
            var white = new Color[v.Count];
            for (int i = 0; i < white.Length; i++) white[i] = Color.white;
            mesh.colors = white;
            mesh.RecalculateBounds();
            return mesh;
        }

        static readonly int[] BoxFaces = { 0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5 };

        static void Box(List<Vector3> v, List<int> t, Vector3 centre, Vector3 size)
        {
            int b = v.Count;
            for (int i = 0; i < 8; i++)
                v.Add(centre + new Vector3(((i & 1) == 0 ? -0.5f : 0.5f) * size.x, ((i & 2) == 0 ? -0.5f : 0.5f) * size.y,
                                           ((i & 4) == 0 ? -0.5f : 0.5f) * size.z));
            foreach (int i in BoxFaces) t.Add(b + i);
        }

        /// <summary>The target point in the sheet's model frame (OpenCV-handed metres); null hides the cross.</summary>
        public void SetTarget(Vector3? model)
        {
            m_Wanted = model.HasValue;
            if (model.HasValue) m_Cross.localPosition = new Vector3(model.Value.x, -model.Value.y, model.Value.z);   // F, as PoseOverlay
            gameObject.SetActive(m_Wanted && m_Placed);
        }

        /// <summary>The sheet's pose in tracking space (Unity-handed), from the latest trusted Eye pose.</summary>
        public void Place(Vector3 position, Quaternion rotation)
        {
            Vector3 p = m_Space != null ? m_Space.TransformPoint(position) : position;
            Quaternion q = m_Space != null ? m_Space.rotation * rotation : rotation;
            transform.SetPositionAndRotation(p, q);
            m_Placed = true;
            if (m_Wanted && !gameObject.activeSelf) gameObject.SetActive(true);
        }

        void OnDestroy()
        {
            if (m_Mat != null) Destroy(m_Mat);
        }
    }
}
