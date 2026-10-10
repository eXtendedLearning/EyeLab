using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// One gate run on disk: <c>&lt;logRoot&gt;/&lt;yyyymmdd_HHmmss&gt;_&lt;gate&gt;_&lt;label&gt;/</c> with
    /// <c>run.json</c>, <c>camera.csv</c>, <c>pose.csv</c>, <c>events.csv</c> and <c>snapshots/</c>.
    /// CSV rows are buffered in memory and appended by <see cref="FlushIfDue"/> (every 1 s) and
    /// <see cref="Flush"/> (on quit). All numbers use the invariant culture.
    /// </summary>
    public sealed class GateRunLog
    {
        public const string CameraHeader = "t_unity,frame,cam_ts,res_w,res_h,y_mean,y_mad_prev,dup_ts";
        public const string PoseHeader = "t_unity,frame,phase,valid,tracking_state,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w," +
                                         "cam_pos_x,cam_pos_y,cam_pos_z,cam_rot_x,cam_rot_y,cam_rot_z,cam_rot_w,tracking_type,loader_active";
        public const string EventsHeader = "t_unity,frame,event,result,detail";

        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public string Directory { get; }
        public string SnapshotDirectory { get; }
        public JsonObject RunInfo { get; } = new JsonObject();
        public string LastEvent { get; private set; } = "";

        readonly object m_Lock = new object();
        readonly StringBuilder m_Camera = new StringBuilder();
        readonly StringBuilder m_Pose = new StringBuilder();
        readonly StringBuilder m_Events = new StringBuilder();
        readonly int m_MainThreadId;
        double m_LastFlush;
        bool m_RunInfoDirty;

        GateRunLog(string dir)
        {
            Directory = dir;
            SnapshotDirectory = Path.Combine(dir, "snapshots");
            System.IO.Directory.CreateDirectory(SnapshotDirectory);
            m_MainThreadId = Thread.CurrentThread.ManagedThreadId;
            File.WriteAllText(Path.Combine(dir, "camera.csv"), CameraHeader + "\n");
            File.WriteAllText(Path.Combine(dir, "pose.csv"), PoseHeader + "\n");
            File.WriteAllText(Path.Combine(dir, "events.csv"), EventsHeader + "\n");
        }

        public static GateRunLog Create(GateArgs args)
        {
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", Inv);
            string dir = Path.Combine(args.LogRoot, $"{stamp}_{args.GateName}_{args.Label}");
            int n = 1;
            string unique = dir;
            while (System.IO.Directory.Exists(unique))
                unique = $"{dir}_{++n}";
            return new GateRunLog(unique);
        }

        public static string F(double v, string format = "R") =>
            double.IsNaN(v) ? "" : v.ToString(format, Inv);

        bool OnMainThread => Thread.CurrentThread.ManagedThreadId == m_MainThreadId;

        string Stamp()
        {
            // Time.* throws off the main thread; such rows get an empty time and frame -1.
            return OnMainThread
                ? $"{F(Time.realtimeSinceStartupAsDouble)},{Time.frameCount}"
                : ",-1";
        }

        public void Camera(string row) { lock (m_Lock) m_Camera.Append(row).Append('\n'); }

        public void Pose(string row) { lock (m_Lock) m_Pose.Append(row).Append('\n'); }

        public void Event(string evt, object result = null, string detail = null)
        {
            string r = Csv(Convert.ToString(result, Inv) ?? "");
            string row = $"{Stamp()},{Csv(evt)},{r},{Csv(detail ?? "")}";
            lock (m_Lock)
            {
                m_Events.Append(row).Append('\n');
                LastEvent = detail == null ? $"{evt} {r}" : $"{evt} {r} {detail}";
            }
            Debug.Log($"[EyeLabGates] {evt} {r} {detail}");
        }

        public void SetInfo(string section, string key, object value)
        {
            lock (m_Lock)
            {
                RunInfo.Section(section).Set(key, value);
                m_RunInfoDirty = true;
            }
        }

        public void SetSection(string section, JsonObject value)
        {
            lock (m_Lock)
            {
                RunInfo.Set(section, value);
                m_RunInfoDirty = true;
            }
        }

        public void FlushIfDue()
        {
            double now = Time.realtimeSinceStartupAsDouble;
            if (now - m_LastFlush >= 1.0)
            {
                m_LastFlush = now;
                Flush();
            }
        }

        public void Flush()
        {
            string cam, pose, ev, json = null;
            lock (m_Lock)
            {
                cam = Take(m_Camera);
                pose = Take(m_Pose);
                ev = Take(m_Events);
                if (m_RunInfoDirty)
                {
                    json = RunInfo.ToJson();
                    m_RunInfoDirty = false;
                }
            }
            try
            {
                if (cam.Length > 0) File.AppendAllText(Path.Combine(Directory, "camera.csv"), cam);
                if (pose.Length > 0) File.AppendAllText(Path.Combine(Directory, "pose.csv"), pose);
                if (ev.Length > 0) File.AppendAllText(Path.Combine(Directory, "events.csv"), ev);
                if (json != null) File.WriteAllText(Path.Combine(Directory, "run.json"), json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[EyeLabGates] log flush failed: {e.Message}");
            }
        }

        /// <summary>Writes a binary PGM (P5) on a worker thread; <paramref name="pixels"/> must be a private copy.</summary>
        public void SavePgmAsync(string name, int width, int height, byte[] pixels)
        {
            string path = Path.Combine(SnapshotDirectory, name + ".pgm");
            Task.Run(() =>
            {
                try
                {
                    using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
                    {
                        byte[] header = Encoding.ASCII.GetBytes($"P5\n{width} {height}\n255\n");
                        fs.Write(header, 0, header.Length);
                        fs.Write(pixels, 0, width * height);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"[EyeLabGates] snapshot write failed: {e.Message}");
                }
            });
        }

        static string Take(StringBuilder sb)
        {
            string s = sb.ToString();
            sb.Clear();
            return s;
        }

        public static string Csv(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0) return s;
            return "\"" + s.Replace("\"", "\"\"").Replace("\r", " ").Replace("\n", " | ") + "\"";
        }
    }

    /// <summary>Insertion-ordered JSON object with a minimal serializer (no dependency on JsonUtility).</summary>
    public sealed class JsonObject
    {
        readonly List<KeyValuePair<string, object>> m_Items = new List<KeyValuePair<string, object>>();

        public JsonObject Set(string key, object value)
        {
            for (int i = 0; i < m_Items.Count; i++)
            {
                if (m_Items[i].Key == key)
                {
                    m_Items[i] = new KeyValuePair<string, object>(key, value);
                    return this;
                }
            }
            m_Items.Add(new KeyValuePair<string, object>(key, value));
            return this;
        }

        public bool TryGet(string key, out object value)
        {
            foreach (var kv in m_Items)
            {
                if (kv.Key == key)
                {
                    value = kv.Value;
                    return true;
                }
            }
            value = null;
            return false;
        }

        public JsonObject Section(string key)
        {
            foreach (var kv in m_Items)
                if (kv.Key == key && kv.Value is JsonObject o)
                    return o;
            var created = new JsonObject();
            Set(key, created);
            return created;
        }

        public string ToJson()
        {
            var sb = new StringBuilder();
            Write(sb, this, 0);
            sb.Append('\n');
            return sb.ToString();
        }

        static void Write(StringBuilder sb, object v, int indent)
        {
            switch (v)
            {
                case null:
                    sb.Append("null");
                    break;
                case JsonObject o:
                    if (o.m_Items.Count == 0) { sb.Append("{}"); break; }
                    sb.Append("{\n");
                    for (int i = 0; i < o.m_Items.Count; i++)
                    {
                        sb.Append(' ', (indent + 1) * 2);
                        WriteString(sb, o.m_Items[i].Key);
                        sb.Append(": ");
                        Write(sb, o.m_Items[i].Value, indent + 1);
                        sb.Append(i + 1 < o.m_Items.Count ? ",\n" : "\n");
                    }
                    sb.Append(' ', indent * 2).Append('}');
                    break;
                case string s:
                    WriteString(sb, s);
                    break;
                case bool b:
                    sb.Append(b ? "true" : "false");
                    break;
                case float f:
                    if (float.IsNaN(f) || float.IsInfinity(f)) sb.Append("null");
                    else sb.Append(f.ToString("R", CultureInfo.InvariantCulture));
                    break;
                case double d:
                    WriteNumber(sb, d);
                    break;
                case Enum e:
                    WriteString(sb, e.ToString());
                    break;
                case IEnumerable list:
                    sb.Append('[');
                    bool first = true;
                    foreach (var item in list)
                    {
                        if (!first) sb.Append(", ");
                        first = false;
                        Write(sb, item, indent + 1);
                    }
                    sb.Append(']');
                    break;
                case IFormattable n:
                    sb.Append(n.ToString(null, CultureInfo.InvariantCulture));
                    break;
                default:
                    WriteString(sb, v.ToString());
                    break;
            }
        }

        static void WriteNumber(StringBuilder sb, double d)
        {
            if (double.IsNaN(d) || double.IsInfinity(d)) sb.Append("null");
            else sb.Append(d.ToString("R", CultureInfo.InvariantCulture));
        }

        static void WriteString(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }
    }
}
