using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.3: every Unity log line also goes to <c>files/BeamProRuns/app_log_&lt;stamp&gt;.txt</c>, pulled with the runs.
    /// The logcat main buffer rolled over within 15 minutes in the T6.2 session, so the start-up of the glasses
    /// view (first frame, first head pose) was lost. Each line: local time, type (I/W/E), message.
    /// </summary>
    public static class AppLog
    {
        static readonly object s_Lock = new object();
        static StreamWriter s_Writer;

        public static string Path { get; private set; } = "";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Boot() => Start();

        public static void Start()
        {
            lock (s_Lock)
            {
                if (s_Writer != null) return;
                try
                {
                    Directory.CreateDirectory(ProbeRun.Root);
                    Path = System.IO.Path.Combine(ProbeRun.Root,
                        "app_log_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) + ".txt");
                    s_Writer = new StreamWriter(Path, false, new UTF8Encoding(false)) { AutoFlush = true };
                }
                catch (Exception e) { Debug.LogWarning("[EyeLab probe] app log not opened: " + e.Message); return; }
            }
            Application.logMessageReceivedThreaded += OnLog;
            Debug.Log($"[EyeLab probe] app log {Path}; app {Application.version}");
        }

        static void OnLog(string message, string stack, LogType type)
        {
            lock (s_Lock)
            {
                if (s_Writer == null) return;
                try
                {
                    s_Writer.Write(DateTime.Now.ToString("HH:mm:ss.fff", CultureInfo.InvariantCulture));
                    s_Writer.Write(' ');
                    s_Writer.Write(type == LogType.Log ? "I" : type == LogType.Warning ? "W" : "E");
                    s_Writer.Write(' ');
                    s_Writer.Write(message);
                    s_Writer.Write('\n');
                    if (type == LogType.Exception || type == LogType.Error) { s_Writer.Write(stack); s_Writer.Write('\n'); }
                }
                catch (Exception) { }
            }
        }
    }
}
