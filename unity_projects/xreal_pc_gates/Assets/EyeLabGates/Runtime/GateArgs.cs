using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace EyeLabGates
{
    public enum GateKind { Manual, G1, G2, G3, G4, Cooldown }

    /// <summary>
    /// Command-line arguments of the gate player (docs/XREAL-PC-GATES-G1-G3.md §5):
    /// <c>-gate G1|G2|G3|G4|manual|cooldown -order tracked|xrfirst|3dof|camfirst -reader vendor|drain -flushLag s -renderFps 30|60
    /// -burstEvery s -burstFrames n -duration s -snapshotEvery s -label text -logRoot dir</c>. G1-G4 follow the timed protocols in
    /// <see cref="GateProtocol"/>; <c>-duration</c> only applies to manual (it is recorded otherwise). G4 (light overlay)
    /// renders at 30 fps unless <c>-renderFps 60</c> is given, and reads <c>-order tracked</c> as 6DoF, <c>3dof</c> as 3DoF.
    /// Unknown arguments (e.g. Unity's own <c>-logFile</c>) are ignored.
    /// </summary>
    public sealed class GateArgs
    {
        public GateKind Gate = GateKind.Manual;
        public string Order = "tracked";
        public string Reader = "vendor";
        /// <summary>Restart the camera when its lag has grown by this many seconds; 0 = never.</summary>
        public double FlushLag;
        /// <summary>XR render rate: 60 (default, display rate) or 30 (vSyncCount 2, lighter host load).</summary>
        public int RenderFps = 60;
        /// <summary>G4: start the camera every this many seconds for one short burst; 0 = camera never started.</summary>
        public double BurstEvery = 2.0;
        /// <summary>G4: frames per burst; the last one is saved as a snapshot.</summary>
        public int BurstFrames = 3;
        public double Duration;
        public double SnapshotEvery = 2.0;
        public string Label = "run";
        public string LogRoot;
        public string[] Raw = Array.Empty<string>();
        public readonly List<string> Warnings = new List<string>();

        public static readonly string[] Orders = { "tracked", "xrfirst", "3dof", "camfirst" };

        public static double DefaultDuration(GateKind gate)
        {
            switch (gate)
            {
                case GateKind.G1: return 60;
                case GateKind.G2: return 180;
                case GateKind.G3: return 180;
                case GateKind.G4: return 300;
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
            bool renderFpsGiven = false;

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
                            case "g4": a.Gate = GateKind.G4; break;
                            case "manual": a.Gate = GateKind.Manual; break;
                            case "cooldown": a.Gate = GateKind.Cooldown; break;
                            default: a.Warnings.Add($"unknown -gate '{value}', using manual"); break;
                        }
                        break;
                    case "-order":
                        i++;
                        string order = (value ?? "").ToLowerInvariant();
                        if (Array.IndexOf(Orders, order) >= 0)
                            a.Order = order;
                        else
                            a.Warnings.Add($"unknown -order '{value}', using tracked");
                        break;
                    case "-reader":
                        i++;
                        string reader = (value ?? "").ToLowerInvariant();
                        if (reader == "vendor" || reader == "drain")
                            a.Reader = reader;
                        else
                            a.Warnings.Add($"unknown -reader '{value}', using vendor");
                        break;
                    case "-duration":
                        i++;
                        if (TryPositive(value, out double d)) duration = d;
                        else a.Warnings.Add($"bad -duration '{value}', using default");
                        break;
                    case "-renderfps":
                        i++;
                        if (value == "30" || value == "60") { a.RenderFps = value == "30" ? 30 : 60; renderFpsGiven = true; }
                        else a.Warnings.Add($"bad -renderFps '{value}', using 60");
                        break;
                    case "-burstevery":
                        i++;
                        if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double be) && be >= 0) a.BurstEvery = be;
                        else a.Warnings.Add($"bad -burstEvery '{value}', using 2");
                        break;
                    case "-burstframes":
                        i++;
                        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int bf) && bf >= 1 && bf <= 30) a.BurstFrames = bf;
                        else a.Warnings.Add($"bad -burstFrames '{value}', using 3");
                        break;
                    case "-flushlag":
                        i++;
                        if (TryPositive(value, out double fl)) a.FlushLag = fl;
                        else a.Warnings.Add($"bad -flushLag '{value}', flushing disabled");
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
            if (a.Gate == GateKind.G4 && !renderFpsGiven) a.RenderFps = 30;
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
