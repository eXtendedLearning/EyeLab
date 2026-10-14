using System;
using System.Diagnostics;
using System.Text;

namespace EyeLabGates
{
    /// <summary>
    /// Spoken step cues through Windows' built-in speech synthesiser (System.Speech). One hidden
    /// PowerShell process is started per run and fed one line per cue on stdin, so a cue costs no
    /// process start. Prefers an English voice; a new cue interrupts the previous one. Audio goes to
    /// the Windows default output device (the glasses' speakers if Windows selected them). If the
    /// process cannot start, the run continues silently and the failure is logged.
    /// </summary>
    public sealed class Speaker : IDisposable
    {
        const string Script =
            "Add-Type -AssemblyName System.Speech\n" +
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n" +
            "try { $v = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en*' } | Select-Object -First 1\n" +
            "      if ($v) { $s.SelectVoice($v.VoiceInfo.Name) } } catch {}\n" +
            "$s.Rate = 1\n" +
            "$s.SetOutputToDefaultAudioDevice()\n" +
            "[Console]::Out.WriteLine('VOICE ' + $s.Voice.Name)\n" +
            "while ($true) { $l = [Console]::In.ReadLine(); if ($l -eq $null) { break }\n" +
            "  $s.SpeakAsyncCancelAll(); if ($l.Length -gt 0) { [void]$s.SpeakAsync($l) } }\n";

        readonly Process m_Process;
        readonly Action<string, string> m_Log;
        bool m_Dead;

        public string Voice { get; private set; } = "";

        Speaker(Process process, Action<string, string> log)
        {
            m_Process = process;
            m_Log = log;
        }

        /// <param name="log">(result, detail) sink for speech events; called on worker threads too.</param>
        public static Speaker TryStart(Action<string, string> log)
        {
            try
            {
                string encoded = Convert.ToBase64String(Encoding.Unicode.GetBytes(Script));
                var psi = new ProcessStartInfo("powershell.exe",
                    "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + encoded)
                {
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };
                var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
                var speaker = new Speaker(p, log);
                p.OutputDataReceived += (_, e) =>
                {
                    if (e.Data != null && e.Data.StartsWith("VOICE ", StringComparison.Ordinal))
                    {
                        speaker.Voice = e.Data.Substring(6);
                        log("READY", speaker.Voice);
                    }
                };
                p.ErrorDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) log("STDERR", e.Data); };
                p.Exited += (_, __) => { speaker.m_Dead = true; };
                p.Start();
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.StandardInput.AutoFlush = true;
                return speaker;
            }
            catch (Exception e)
            {
                log("UNAVAILABLE", e.GetType().Name + ": " + e.Message);
                return null;
            }
        }

        public void Say(string text)
        {
            if (m_Dead || text == null) return;
            try
            {
                m_Process.StandardInput.WriteLine(text.Replace('\n', ' '));
            }
            catch (Exception)
            {
                m_Dead = true;
            }
        }

        public void Dispose()
        {
            try
            {
                if (!m_Process.HasExited)
                {
                    m_Process.StandardInput.Close();
                    if (!m_Process.WaitForExit(500)) m_Process.Kill();
                }
            }
            catch (Exception)
            {
                // best effort
            }
            m_Process.Dispose();
        }
    }
}
