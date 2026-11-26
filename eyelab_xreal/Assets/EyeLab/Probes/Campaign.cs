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
            public string display_ar = "person";  // eye model for A and T: "person" (the participant's calibration), "all" (device pool) or "sdk"
            public bool speak_numbers = true;
            public int[] sdk_participants = { 13 };   // who may run AR-SDK (Jack, 2026-11-26: only P13)
        }

        [Serializable]
        public sealed class SheetTarget
        {
            public int n;
            public float[] page_mm;
            public float[] model_m;
        }

        [Serializable]
        public sealed class SheetSaddle
        {
            public int id;
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
            public SheetSaddle[] saddles;
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

        /// <summary>The kind of sheet number <paramref name="n"/>: T 1-10, P 11-40, M 41-100 (python/campaign/sheets.py PLAN).</summary>
        public static string KindOf(int n) => n <= 10 ? "T" : n <= 40 ? "P" : "M";

        /// <summary>Printed needs a P sheet; every AR condition takes a hidden-target sheet, T or M (0.5.002: T02 under AR-SDK).</summary>
        static bool Fits(string condition, int n) => condition == "P" ? KindOf(n) == "P" : KindOf(n) != "P";
        public static string ParticipantId => "P" + Current.participant.ToString("D2", CultureInfo.InvariantCulture);
        public static string SheetName(int number, string kind) => kind + number.ToString("D3", CultureInfo.InvariantCulture);
        public static string CurrentSheetName => SheetName(Current.sheet, KindOf(Current.sheet));
        public static string ConditionName(string c) { int i = Array.IndexOf(Conditions, c); return i >= 0 ? ConditionNames[i] : c; }

        /// <summary>The eye model the overlay uses in this condition: SDK for S, else the protocol's (pooled by default).</summary>
        public static string DisplayFor(string condition) =>
            condition == "S" ? "sdk" : Proto.display_ar == "person" && PersonCal.UsesPool(Current.participant) ? "all" : Proto.display_ar;

        /// <summary>AR-SDK is only for the participants the protocol names (P13 = Jack).</summary>
        public static bool SdkAllowed(int participant) => Proto.sdk_participants != null && Array.IndexOf(Proto.sdk_participants, participant) >= 0;

        /// <summary>The display actually used: "person" falls back to "sdk" without a calibration for this participant.</summary>
        public static string EffectiveDisplay(string condition)
        {
            if (!ShowsCross(condition)) return "none";
            string d = DisplayFor(condition);
            return d == "person" && !PersonCal.Exists(Current.participant) ? "sdk" : d;
        }

        /// <summary>The protocol re-read for the set-up page (sdk_participants, display).</summary>
        public static void Reload() => Proto = LoadProtocol(out _);
        public static bool ShowsCross(string condition) => condition != "P";

        public static void SetParticipant(int delta) { Current.participant = Mathf.Clamp(Current.participant + delta, 0, 99); SaveState(); }

        public static void SetCondition(string c)
        {
            Current.condition = c;
            if (!Fits(c, Current.sheet) || (c == "T" && KindOf(Current.sheet) != "T")) Current.sheet = FirstNumber(Kind(c));
            SaveState();
        }

        public static void StepSheet(int delta)
        {
            int n = Current.sheet;
            for (int i = 0; i < 250; i++)
            {
                n += delta;
                if (n < 1 || n > 249) break;
                if (Fits(Current.condition, n) && Exists(SheetName(n, KindOf(n)))) { Current.sheet = n; SaveState(); return; }
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
            run.Meta("campaign_display", ShowsCross(Current.condition) ? EffectiveDisplay(Current.condition) +
                     (EffectiveDisplay(Current.condition) != DisplayFor(Current.condition) ? " (wanted " + DisplayFor(Current.condition) + ": no calibration for " + ParticipantId + ")" : "")
                     : "none (printed)");
            if (PersonCal.Exists(Current.participant)) run.Meta("campaign_person_cal", PersonCal.Describe(Current.participant));
            run.Meta("campaign_display_choice", PersonCal.UsesPool(Current.participant) ? "device pool (" + PersonCal.ChoicePath(Current.participant) + ")" : "personal when calibrated");
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
            steps.Add(new CueStep("review", "Done. Now look straight down at the sheet from about twenty centimetres, so it fills your view, and hold still.",
                                  Proto.review_s, "Look DOWN at the sheet, 20 cm: HOLD STILL", ticks: true));
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
                EffectiveDisplay(Current.condition), Proto.id, Path.GetFileName(run.Dir),
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
    /// 0.5.002 (Jack, 2026-11-26): each participant's own display calibration, made on the A40 sheet (probe <c>pcal</c>),
    /// in <c>files/calibration/participants/Pnn.json</c> (the model, <see cref="EyeModel.Saved"/>) and <c>Pnn_taps.csv</c>
    /// (its taps). The campaign's AR conditions draw with it; without one they fall back to the SDK's display.
    /// </summary>
    public static class PersonCal
    {
        public static string Dir => Path.Combine(EyeModel.Dir, "participants");
        public static string Name(int participant) => "P" + participant.ToString("D2", CultureInfo.InvariantCulture);
        public static string JsonPath(int participant) => Path.Combine(Dir, Name(participant) + ".json");
        public static string TapsPath(int participant) => Path.Combine(Dir, Name(participant) + "_taps.csv");
        public static bool Exists(int participant) => File.Exists(JsonPath(participant));

        /// <summary>0.5.004 (Jack, 2026-11-26): a participant can be set to draw AR with the device pool ("all", the
        /// T6.12 store) instead of a personal calibration: <c>Pnn_display.txt</c> = "pool". Jack's first personal
        /// calibration looked worse on A40 than the pool.</summary>
        public static string ChoicePath(int participant) => Path.Combine(Dir, Name(participant) + "_display.txt");

        public static bool UsesPool(int participant)
        {
            try { return File.Exists(ChoicePath(participant)) && File.ReadAllText(ChoicePath(participant)).Trim() == "pool"; }
            catch (Exception) { return false; }
        }

        public static void SetPool(int participant, bool pool)
        {
            Directory.CreateDirectory(Dir);
            string path = ChoicePath(participant);
            if (File.Exists(path)) File.Delete(path);
            File.WriteAllText(path, pool ? "pool" : "person");
            Debug.Log("[EyeLab campaign] " + Name(participant) + " AR display: " + (pool ? "device pool" : "personal calibration"));
        }

        public static EyeModel.Saved Load(int participant)
        {
            try { return Exists(participant) ? JsonUtility.FromJson<EyeModel.Saved>(File.ReadAllText(JsonPath(participant))) : null; }
            catch (Exception e) { Debug.LogWarning("[EyeLab campaign] calibration " + Name(participant) + ": " + e.Message); return null; }
        }

        public static bool Load(int participant, out double[] left, out double[] right)
        {
            EyeModel.Saved s = Load(participant);
            left = s != null ? EyeModel.FromSaved(s, "L") : null;
            right = s != null ? EyeModel.FromSaved(s, "R") : null;
            return s != null;
        }

        public static void Save(int participant, EyeModel.Saved s, List<EyeModel.Tap> taps)
        {
            Directory.CreateDirectory(Dir);
            s.saved_utc = DateTime.UtcNow.ToString("o");
            string path = JsonPath(participant);
            if (File.Exists(path)) File.Delete(path);
            File.WriteAllText(path, JsonUtility.ToJson(s, true));
            EyeModel.WriteTaps(TapsPath(participant), taps);
        }

        public static string Describe(int participant)
        {
            EyeModel.Saved s = Load(participant);
            if (s == null) return Name(participant) + ": no calibration";
            string when = DateTime.TryParse(s.saved_utc, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out DateTime d)
                ? d.ToLocalTime().ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture) : s.saved_utc;
            return string.Format(CultureInfo.InvariantCulture, "{0}: saved {1}, {2}+{3} taps, fresh-tap error L {4:F1} mm, R {5:F1} mm",
                                 Name(participant), when, s.left_taps, s.right_taps, s.left_loo_mm, s.right_loo_mm);
        }
    }

    /// <summary>
    /// Campaign 0.5: the crosshair of the current target, drawn on the sheet's plane at the sheet's pose and per eye with
    /// the eye model (EyeLabVertexColor with _EyeWarp, as the T6.12 solid overlay). An open centre so the dot shows.
    /// </summary>
    public sealed class MarkCross : MonoBehaviour
    {
        Transform m_Space, m_Cross, m_Grid;
        Material m_Mat;
        bool m_Placed, m_Wanted, m_GridWanted;
        float m_Gap, m_Arm, m_Line, m_Depth;

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
            m.m_Gap = p.cross_gap_mm / 1000f; m.m_Arm = p.cross_arm_mm / 1000f; m.m_Line = p.cross_line_mm / 1000f; m.m_Depth = p.cross_depth_mm / 1000f;
            cross.GetComponent<MeshFilter>().sharedMesh = Build(m.m_Gap, m.m_Arm, m.m_Line, m.m_Depth, new[] { Vector3.zero });
            cross.GetComponent<MeshRenderer>().sharedMaterial = m.m_Mat;
            var grid = new GameObject("grid", typeof(MeshFilter), typeof(MeshRenderer));
            grid.transform.SetParent(go.transform, false);
            grid.GetComponent<MeshRenderer>().sharedMaterial = m.m_Mat;
            grid.SetActive(false);
            m.m_Grid = grid.transform;
            go.SetActive(false);
            return m;
        }

        /// <summary>Four boxes in the target plane (local x, y), from <paramref name="gap"/> to <paramref name="arm"/>.</summary>
        static Mesh Build(float gap, float arm, float line, float depth, IList<Vector3> at)
        {
            var v = new List<Vector3>();
            var t = new List<int>();
            float len = arm - gap, mid = (arm + gap) / 2f;
            foreach (Vector3 c in at)
            {
                Box(v, t, c + new Vector3(mid, 0f, 0f), new Vector3(len, line, depth));
                Box(v, t, c + new Vector3(-mid, 0f, 0f), new Vector3(len, line, depth));
                Box(v, t, c + new Vector3(0f, mid, 0f), new Vector3(line, len, depth));
                Box(v, t, c + new Vector3(0f, -mid, 0f), new Vector3(line, len, depth));
            }
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
            m_Cross.gameObject.SetActive(m_Wanted);
            gameObject.SetActive((m_Wanted || m_GridWanted) && m_Placed);
        }

        /// <summary>0.5.002: a crosshair on every point (model frame, OpenCV-handed metres) at once, for the calibration
        /// check on A40's grid; null or empty hides them.</summary>
        public void SetPoints(IList<Vector3> models)
        {
            m_GridWanted = models != null && models.Count > 0;
            if (m_GridWanted)
            {
                var at = new List<Vector3>();
                foreach (Vector3 m in models) at.Add(new Vector3(m.x, -m.y, m.z));   // F
                MeshFilter mf = m_Grid.GetComponent<MeshFilter>();
                if (mf.sharedMesh != null) Destroy(mf.sharedMesh);
                mf.sharedMesh = Build(m_Gap, m_Arm, m_Line, m_Depth, at);
            }
            m_Grid.gameObject.SetActive(m_GridWanted);
            gameObject.SetActive((m_Wanted || m_GridWanted) && m_Placed);
        }

        /// <summary>The sheet's pose in tracking space (Unity-handed), from the latest trusted Eye pose.</summary>
        public void Place(Vector3 position, Quaternion rotation)
        {
            Vector3 p = m_Space != null ? m_Space.TransformPoint(position) : position;
            Quaternion q = m_Space != null ? m_Space.rotation * rotation : rotation;
            transform.SetPositionAndRotation(p, q);
            m_Placed = true;
            if ((m_Wanted || m_GridWanted) && !gameObject.activeSelf) gameObject.SetActive(true);
        }

        void OnDestroy()
        {
            if (m_Mat != null) Destroy(m_Mat);
        }
    }
}
