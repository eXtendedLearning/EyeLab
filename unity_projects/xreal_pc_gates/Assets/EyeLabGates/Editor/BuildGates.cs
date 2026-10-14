// D8 of docs/XREAL-PC-GATES-G1-G3.md: builds the gate harness and the unmodified vendor control,
// checks the native plugin set, writes BUILD_INFO.txt, copies run_gates.bat and zips Builds/Win64.
//   unity run <project> -- -executeMethod EyeLabGates.EditorTools.BuildGates.BuildAll
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEditor.Build.Reporting;
using Debug = UnityEngine.Debug;

namespace EyeLabGates.EditorTools
{
    public static class BuildGates
    {
        const string k_Tag = "[BuildGates]";
        const string k_BuildInfoAsset = "Assets/EyeLabGates/Resources/" + DeviceInfoProbe.BuildInfoResource + ".txt";

        // §2 of the guide: the experimental package's Runtime/Plugins/X64 set.
        static readonly string[] k_RequiredDlls =
        {
            "XREALXRPlugin.dll", "libnr_api.dll", "libnr_loader.dll", "libnr_glasses_api.dll", "libnr_plugin_6dof.dll",
            "ov_utils.dll", "avcodec-60.dll", "avformat-60.dll", "avutil-58.dll", "swresample-4.dll", "swscale-7.dll",
        };

        static readonly string[] k_Tarballs = { "com.xreal.xr.tar.gz", "com.xreal.xr.experimental.tar.gz" };

        static string ProjectDir => Path.GetDirectoryName(UnityEngine.Application.dataPath);
        static string RepoDir => Path.GetFullPath(Path.Combine(ProjectDir, "..", ".."));
        static string VendorDir => Path.Combine(RepoDir, "vendor", "xreal", "pc-beta-3.1.2");
        static string Win64Dir => Path.Combine(ProjectDir, "Builds", "Win64");

        static void Log(string msg) => Debug.Log($"{k_Tag} {msg}");

        public static void BuildAll()
        {
            ProjectSetup.Apply();
            GateSceneBuilder.Build();

            var info = CollectBuildInfo();
            WriteBuildInfoResource(info);

            RescueRunZips();
            if (Directory.Exists(Win64Dir))
            {
                Directory.Delete(Win64Dir, true);
                Log($"cleaned {Win64Dir}");
            }

            var builds = new[]
            {
                (name: "GateHarness", exe: "XrealPcGates.exe", scene: GateSceneBuilder.HarnessScene),
                (name: "VendorRGBCamera", exe: "VendorRGBCamera.exe", scene: GateSceneBuilder.VendorScene),
            };

            var summary = new StringBuilder();
            foreach (var b in builds)
            {
                string exePath = Path.Combine(Win64Dir, b.name, b.exe);
                var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { b.scene },
                    locationPathName = exePath,
                    target = BuildTarget.StandaloneWindows64,
                    targetGroup = BuildTargetGroup.Standalone,
                    options = BuildOptions.None,
                });
                var s = report.summary;
                Log($"build {b.name}: {s.result}, {s.totalErrors} errors, {s.totalWarnings} warnings, {s.totalSize} bytes, {s.totalTime}");
                if (s.result != BuildResult.Succeeded)
                    throw new Exception($"{k_Tag} build {b.name} failed: {s.result}");

                string pluginDir = Path.Combine(Win64Dir, b.name, Path.GetFileNameWithoutExtension(b.exe) + "_Data", "Plugins", "x86_64");
                var present = Directory.Exists(pluginDir)
                    ? Directory.GetFiles(pluginDir, "*.dll").Select(Path.GetFileName).ToList()
                    : new List<string>();
                var missing = k_RequiredDlls.Where(d => !present.Contains(d, StringComparer.OrdinalIgnoreCase)).ToList();
                Log($"DLL check {b.name}: {k_RequiredDlls.Length - missing.Count}/{k_RequiredDlls.Length} present; " +
                    $"plugin dir has [{string.Join(", ", present)}]");
                if (missing.Count > 0)
                    throw new Exception($"{k_Tag} {b.name} missing DLLs: {string.Join(", ", missing)}");

                summary.AppendLine($"build.{b.name} = {b.name}/{b.exe} scene={b.scene} result={s.result} " +
                                   $"warnings={s.totalWarnings} dll_check=OK({k_RequiredDlls.Length}/{k_RequiredDlls.Length})");
            }

            File.WriteAllText(Path.Combine(Win64Dir, "BUILD_INFO.txt"), FormatInfo(info) + summary);
            Log($"wrote {Path.Combine(Win64Dir, "BUILD_INFO.txt")}");

            // cmd.exe needs CRLF; the tracked template is LF (repo rule: never commit CRLF).
            string bat = File.ReadAllText(Path.Combine(ProjectDir, "Tools", "run_gates.bat"));
            bat = bat.Replace("\r\n", "\n").Replace("\n", "\r\n");
            File.WriteAllText(Path.Combine(Win64Dir, "run_gates.bat"), bat, new UTF8Encoding(false));
            Log("copied run_gates.bat (CRLF)");

            string zip = Path.Combine(ProjectDir, "Builds", $"XrealPcGates_{DateTime.Now.ToString("yyyyMMdd_HHmm", CultureInfo.InvariantCulture)}.zip");
            if (File.Exists(zip)) File.Delete(zip);
            ZipFile.CreateFromDirectory(Win64Dir, zip, CompressionLevel.Optimal, false);
            Log($"ZIP {zip} ({new FileInfo(zip).Length} bytes)");

            ArchiveOldBuilds(zip);
            WriteLatestBuildPointer(info, zip);
        }

        /// <summary>
        /// Builds/ keeps only the newest build zip; older ones go to Builds/archive/, which keeps the
        /// <see cref="ArchiveKeep"/> most recent and deletes the rest (every build is reproducible from
        /// its git commit, recorded in BUILD_INFO.txt).
        /// </summary>
        const int ArchiveKeep = 2;

        static void ArchiveOldBuilds(string newest)
        {
            string builds = Path.Combine(ProjectDir, "Builds");
            string archive = Path.Combine(builds, "archive");
            Directory.CreateDirectory(archive);
            foreach (var z in Directory.GetFiles(builds, "XrealPcGates_*.zip"))
            {
                if (string.Equals(Path.GetFullPath(z), Path.GetFullPath(newest), StringComparison.OrdinalIgnoreCase)) continue;
                string dest = Path.Combine(archive, Path.GetFileName(z));
                if (File.Exists(dest)) File.Delete(dest);
                File.Move(z, dest);
                Log($"archived {Path.GetFileName(z)}");
            }
            var old = Directory.GetFiles(archive, "XrealPcGates_*.zip").OrderByDescending(Path.GetFileName).Skip(ArchiveKeep).ToList();
            foreach (var z in old)
            {
                File.Delete(z);
                Log($"deleted old build {Path.GetFileName(z)} (keeping {ArchiveKeep} in archive)");
            }
        }

        /// <summary>Run zips returned from the Surface are sometimes dropped into Builds/; move them to GateRuns/_inbox.</summary>
        static void RescueRunZips()
        {
            string builds = Path.Combine(ProjectDir, "Builds");
            if (!Directory.Exists(builds)) return;
            string inbox = Path.Combine(ProjectDir, "GateRuns", "_inbox");
            foreach (var z in Directory.GetFiles(builds, "GateRuns_*.zip", SearchOption.AllDirectories))
            {
                Directory.CreateDirectory(inbox);
                string dest = Path.Combine(inbox, Path.GetFileName(z));
                if (File.Exists(dest)) { Log($"run zip {Path.GetFileName(z)} already in GateRuns/_inbox; left in place"); continue; }
                File.Move(z, dest);
                Log($"moved run zip {Path.GetFileName(z)} -> GateRuns/_inbox/");
            }
        }

        /// <summary>
        /// Tracked pointer to the newest build. The Surface launcher compares it with its own
        /// BUILD_INFO.txt (after a git pull) and warns when the copied build is older.
        /// </summary>
        static void WriteLatestBuildPointer(List<KeyValuePair<string, string>> info, string zip)
        {
            string utc = info.First(kv => kv.Key == "utc_build").Value;
            string head = info.First(kv => kv.Key == "git_head").Value;
            string text =
                "# Newest XREAL gate build (written by BuildGates.BuildAll; commit it with the build's source).\n" +
                "# The Surface launcher warns when its BUILD_INFO.txt utc_build differs from this one.\n" +
                $"utc_build = {utc}\n" +
                $"zip = Builds/{Path.GetFileName(zip)}\n" +
                $"source_git_head = {head}\n";
            File.WriteAllText(Path.Combine(ProjectDir, "LATEST_BUILD.txt"), text);
            Log($"wrote LATEST_BUILD.txt (utc_build {utc})");
        }

        static List<KeyValuePair<string, string>> CollectBuildInfo()
        {
            var info = new List<KeyValuePair<string, string>>
            {
                Kv("utc_build", DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture)),
                Kv("git_head", Git("rev-parse HEAD")),
                Kv("git_dirty", string.IsNullOrWhiteSpace(Git("status --porcelain")) ? "false" : "true"),
                Kv("editor_version", UnityEngine.Application.unityVersion),
            };
            foreach (var p in UnityEditor.PackageManager.PackageInfo.GetAllRegisteredPackages()
                         .Where(p => p.name.StartsWith("com.xreal", StringComparison.Ordinal)
                                     || p.name == "com.cysharp.unitask"
                                     || p.name.StartsWith("com.unity.xr", StringComparison.Ordinal))
                         .OrderBy(p => p.name))
            {
                string extra = p.git != null && !string.IsNullOrEmpty(p.git.hash) ? $" git={p.git.hash}" : "";
                info.Add(Kv("package." + p.name, p.version + extra));
            }
            foreach (var t in k_Tarballs)
                info.Add(Kv("sha256." + t, Sha256(Path.Combine(VendorDir, t))));
            return info;
        }

        static void WriteBuildInfoResource(List<KeyValuePair<string, string>> info)
        {
            if (!AssetDatabase.IsValidFolder("Assets/EyeLabGates/Resources"))
                AssetDatabase.CreateFolder("Assets/EyeLabGates", "Resources");
            File.WriteAllText(Path.Combine(ProjectDir, k_BuildInfoAsset), FormatInfo(info));
            AssetDatabase.ImportAsset(k_BuildInfoAsset, ImportAssetOptions.ForceUpdate);
            Log($"wrote {k_BuildInfoAsset}");
        }

        static string FormatInfo(IEnumerable<KeyValuePair<string, string>> info) =>
            string.Concat(info.Select(kv => $"{kv.Key} = {kv.Value}\n"));

        static KeyValuePair<string, string> Kv(string k, string v) => new KeyValuePair<string, string>(k, v);

        static string Sha256(string path)
        {
            if (!File.Exists(path)) return "missing: " + path;
            using (var sha = SHA256.Create())
            using (var fs = File.OpenRead(path))
                return BitConverter.ToString(sha.ComputeHash(fs)).Replace("-", "");
        }

        static string Git(string args)
        {
            try
            {
                var psi = new ProcessStartInfo("git", $"-C \"{RepoDir}\" {args}")
                {
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };
                using (var p = Process.Start(psi))
                {
                    string output = p.StandardOutput.ReadToEnd();
                    p.WaitForExit();
                    return p.ExitCode == 0 ? output.Trim() : $"git exit {p.ExitCode}";
                }
            }
            catch (Exception e)
            {
                return "git unavailable: " + e.Message;
            }
        }
    }
}
