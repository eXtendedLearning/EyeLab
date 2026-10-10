using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace EyeLabGates
{
    public enum GateKind { Manual, G1, G2, G3 }

    /// <summary>
    /// Command-line arguments of the gate player (docs/XREAL-PC-GATES-G1-G3.md §5):
    /// <c>-gate G1|G2|G3|manual -order xrfirst|camfirst -duration s -snapshotEvery s -label text -logRoot dir</c>.
    /// Unknown arguments (e.g. Unity's own <c>-logFile</c>) are ignored.
    /// </summary>
    public sealed class GateArgs
    {
        public GateKind Gate = GateKind.Manual;
        public string Order = "xrfirst";
        public double Duration;
        public double SnapshotEvery = 2.0;
        public string Label = "run";
        public string LogRoot;
        public string[] Raw = Array.Empty<string>();
        public readonly List<string> Warnings = new List<string>();

        public bool CameraFirst => Order == "camfirst";

        public static double DefaultDuration(GateKind gate)
        {
            switch (gate)
            {
                case GateKind.G1: return 60;
                case GateKind.G2: return 180;
                case GateKind.G3: return 180;
                default: return 0;
            }
        }

        public static GateArgs FromEnvironment()
        {
            // The Editor's command line belongs to the Editor, not to the gate: Play mode is always manual.
            return Application.isEditor ? Parse(Array.Empty<string>()) : Parse(Environment.GetCommandLineArgs());
        }

        public static GateArgs Parse(string[] argv)
        {
            var a = new GateArgs { Raw = argv ?? Array.Empty<string>() };
            double? duration = null;

            for (int i = 0; i < a.Raw.Length; i++)
            {
                string key = a.Raw[i];
                if (!key.StartsWith("-", StringComparison.Ordinal))
                    continue;
                string value = i + 1 < a.Raw.Length ? a.Raw[i + 1] : null;

                switch (key.ToLowerInvariant())
                {
                    case "-gate":
                        i++;
                        switch ((value ?? "").ToLowerInvariant())
                        {
                            case "g1": a.Gate = GateKind.G1; break;
                            case "g2": a.Gate = GateKind.G2; break;
                            case "g3": a.Gate = GateKind.G3; break;
                            case "manual": a.Gate = GateKind.Manual; break;
                            default: a.Warnings.Add($"unknown -gate '{value}', using manual"); break;
                        }
                        break;
                    case "-order":
                        i++;
                        string order = (value ?? "").ToLowerInvariant();
                        if (order == "xrfirst" || order == "camfirst")
                            a.Order = order;
                        else
                            a.Warnings.Add($"unknown -order '{value}', using xrfirst");
                        break;
                    case "-duration":
                        i++;
                        if (TryPositive(value, out double d)) duration = d;
                        else a.Warnings.Add($"bad -duration '{value}', using default");
                        break;
                    case "-snapshotevery":
                        i++;
                        if (TryPositive(value, out double s)) a.SnapshotEvery = s;
                        else a.Warnings.Add($"bad -snapshotEvery '{value}', using 2");
                        break;
                    case "-label":
                        i++;
                        if (!string.IsNullOrWhiteSpace(value)) a.Label = Sanitize(value);
                        break;
                    case "-logroot":
                        i++;
                        if (!string.IsNullOrWhiteSpace(value)) a.LogRoot = value;
                        break;
                }
            }

            a.Duration = duration ?? DefaultDuration(a.Gate);
            if (string.IsNullOrEmpty(a.LogRoot))
                a.LogRoot = Path.Combine(Application.persistentDataPath, "GateRuns");
            return a;
        }

        static bool TryPositive(string s, out double v)
        {
            return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v) && v > 0;
        }

        static string Sanitize(string s)
        {
            var chars = s.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
                if (!char.IsLetterOrDigit(chars[i]) && chars[i] != '-' && chars[i] != '_')
                    chars[i] = '_';
            return new string(chars);
        }

        public string GateName => Gate == GateKind.Manual ? "manual" : Gate.ToString();
    }
}
