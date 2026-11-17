using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// One probe run: a folder under <c>persistentDataPath/BeamProRuns/&lt;stamp&gt;_&lt;probe&gt;/</c>
    /// (<c>/sdcard/Android/data/it.univpm.eyelab/files/BeamProRuns</c>, pulled with <c>adb pull</c>) holding
    /// <c>run.json</c> and CSVs. Rows are queued and written on a background thread so logging never stalls
    /// Update; binary files (frame dumps) go through the same thread.
    /// </summary>
    public sealed class ProbeRun : IDisposable
    {
        public static string Root => Path.Combine(Application.persistentDataPath, "BeamProRuns");

        public string Dir { get; }
        public string Probe { get; }

        readonly Dictionary<string, StreamWriter> m_Csv = new Dictionary<string, StreamWriter>();
        readonly ConcurrentQueue<Action> m_Jobs = new ConcurrentQueue<Action>();
        readonly AutoResetEvent m_Signal = new AutoResetEvent(false);
        readonly Thread m_Writer;
        volatile bool m_Stop;
        readonly Dictionary<string, string> m_Meta = new Dictionary<string, string>();
        public int PendingJobs => m_Jobs.Count;
        public long DroppedJobs { get; private set; }
        public int MaxPendingJobs = 256;

        public ProbeRun(string probe)
        {
            Probe = probe;
            Dir = Path.Combine(Root, DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) + "_" + probe);
            Directory.CreateDirectory(Dir);
            m_Writer = new Thread(WriterLoop) { IsBackground = true, Name = "eyelab-probe-writer" };
            m_Writer.Start();
            ProbeClocks.Sample c = ProbeClocks.Now();
            Meta("probe", probe);
            Meta("app_version", Application.version);
            Meta("unity", Application.unityVersion);
            Meta("device_model", SystemInfo.deviceModel);
            Meta("os", SystemInfo.operatingSystem);
            Meta("xreal_sdk", "3.1.0 (eyelab_xreal/Packages/com.xreal.xr)");
            Meta("start_local", DateTime.Now.ToString("o", CultureInfo.InvariantCulture));
            Meta("start_clocks", ProbeClocks.Sample.Header + "=" + c.Csv());
        }

        public void Meta(string key, string value) { lock (m_Meta) m_Meta[key] = value ?? ""; }

        /// <summary>Queue a CSV row; the header is written once, before the first row of that file.</summary>
        public void Row(string file, string header, string row)
        {
            Enqueue(() =>
            {
                if (!m_Csv.TryGetValue(file, out StreamWriter w))
                {
                    w = new StreamWriter(Path.Combine(Dir, file), false, new UTF8Encoding(false), 1 << 16);
                    w.Write(header);
                    w.Write('\n');
                    m_Csv[file] = w;
                }
                w.Write(row);
                w.Write('\n');
            }, critical: true);
        }

        /// <summary>Queue a binary file write; dropped (and counted) when the writer is MaxPendingJobs behind.</summary>
        public bool Bytes(string relativePath, Func<byte[]> produce)
        {
            return Enqueue(() =>
            {
                string p = Path.Combine(Dir, relativePath);
                Directory.CreateDirectory(Path.GetDirectoryName(p));
                File.WriteAllBytes(p, produce());
            }, critical: false);
        }

        public void Text(string relativePath, string text) =>
            Enqueue(() => File.WriteAllText(Path.Combine(Dir, relativePath), text, new UTF8Encoding(false)), critical: true);

        bool Enqueue(Action job, bool critical)
        {
            if (!critical && m_Jobs.Count >= MaxPendingJobs) { DroppedJobs++; return false; }
            m_Jobs.Enqueue(job);
            m_Signal.Set();
            return true;
        }

        void WriterLoop()
        {
            while (true)
            {
                while (m_Jobs.TryDequeue(out Action job))
                {
                    try { job(); }
                    catch (Exception e) { Debug.LogWarning("[EyeLab probe] write failed: " + e.Message); }
                }
                if (m_Stop && m_Jobs.IsEmpty) break;
                m_Signal.WaitOne(50);
            }
            foreach (StreamWriter w in m_Csv.Values) { try { w.Flush(); w.Dispose(); } catch (Exception) { } }
        }

        public void Dispose()
        {
            ProbeClocks.Sample c = ProbeClocks.Now();
            Meta("end_local", DateTime.Now.ToString("o", CultureInfo.InvariantCulture));
            Meta("end_clocks", ProbeClocks.Sample.Header + "=" + c.Csv());
            Meta("dropped_binary_jobs", DroppedJobs.ToString(CultureInfo.InvariantCulture));
            Meta("clock_errors", ProbeClocks.Errors);
            Text("run.json", MetaJson());
            m_Stop = true;
            m_Signal.Set();
            m_Writer.Join(10000);
            Debug.Log("[EyeLab probe] run written: " + Dir);
        }

        string MetaJson()
        {
            var sb = new StringBuilder("{\n");
            lock (m_Meta)
            {
                int i = 0;
                foreach (KeyValuePair<string, string> kv in m_Meta)
                {
                    string v = kv.Value;
                    bool raw = v.Length > 0 && (v[0] == '{' || v[0] == '[');
                    sb.Append("  \"").Append(Escape(kv.Key)).Append("\": ")
                      .Append(raw ? v : "\"" + Escape(v) + "\"")
                      .Append(++i < m_Meta.Count ? ",\n" : "\n");
                }
            }
            return sb.Append("}\n").ToString();
        }

        static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "");

        public static string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);
    }
}
