using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace EyeLab.Probes.Editor
{
    /// <summary>
    /// Builds <c>Assets/EyeLab/Scenes/Probes.unity</c> (a copy of SampleScene with a ProbeRunner) and puts it
    /// first in the build list, so the APK starts in the probes. Run once from the menu, or in batch mode with
    /// <c>-executeMethod EyeLab.Probes.Editor.ProbeSceneBuilder.CreateBatch</c>.
    /// <para><see cref="BuildApkBatch"/> does that and builds the probe APK in one call (T6.0 B0 steps 2-3):</para>
    /// <code>Unity.exe -batchmode -nographics -quit -projectPath eyelab_xreal -buildTarget Android
    ///   -executeMethod EyeLab.Probes.Editor.ProbeSceneBuilder.BuildApkBatch [-apkPath build/eyelab_probes.apk]
    ///   -logFile build/probes_build.log</code>
    /// Signed with the Unity debug key (no keystore in the project). Exit code 0 = APK written.
    /// </summary>
    public static class ProbeSceneBuilder
    {
        const string Source = "Assets/Scenes/SampleScene.unity";
        const string Target = "Assets/EyeLab/Scenes/Probes.unity";
        const string DefaultApk = "build/eyelab_probes.apk";
        // B3a native plugin, built by tools/build_aruco_plugin.sh (gitignored build artefact).
        const string ArucoPlugin = "Assets/Plugins/Android/libs/arm64-v8a/libeyelab_aruco.so";

        [MenuItem("EyeLab/Probes/Create Probes Scene")]
        public static void Create()
        {
            if (!AssetDatabase.IsValidFolder("Assets/EyeLab/Scenes")) AssetDatabase.CreateFolder("Assets/EyeLab", "Scenes");
            // Copy only once: copying over an existing scene gives it a new GUID, and Build Settings then pointed
            // at the old one (seen 2026-11-17).
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(Target) == null && !AssetDatabase.CopyAsset(Source, Target))
            {
                Debug.LogError($"[EyeLab] could not copy {Source} to {Target}");
                return;
            }
            var scene = EditorSceneManager.OpenScene(Target, OpenSceneMode.Single);
            if (UnityEngine.Object.FindAnyObjectByType<ProbeRunner>() == null)
                new GameObject("EyeLab Probes").AddComponent<ProbeRunner>();
            EditorSceneManager.SaveScene(scene);

            var scenes = EditorBuildSettings.scenes.Where(s => s.path != Target).ToList();
            scenes.Insert(0, new EditorBuildSettingsScene(Target, true));
            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log($"[EyeLab] {Target} created and placed first in Build Settings");
        }

        public static void CreateBatch() => Create();

        /// <summary>Batch mode: create the Probes scene, build an Android APK with only that scene, exit 0 on success.</summary>
        public static void BuildApkBatch()
        {
            int code = 1;
            try
            {
                Create();
                if (AssetDatabase.LoadAssetAtPath<SceneAsset>(Target) == null) throw new InvalidOperationException("Probes scene missing");
                if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android &&
                    !EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android))
                    throw new InvalidOperationException("cannot switch to Android (is Android Build Support installed?)");

                ConfigureArucoPlugin();
                string apk = Path.GetFullPath(Arg("-apkPath") ?? DefaultApk);
                Directory.CreateDirectory(Path.GetDirectoryName(apk));
                // 2026-11-17: the Gradle project kept a data.unity3d from a May 2026 build; it was packed into every
                // APK next to the fresh loose files and the player loaded it, so the probes APK ran SampleScene.
                // Unity rewrites this folder on every build, so removing it first is safe.
                const string StaleAssets = "Library/Bee/Android/Prj/IL2CPP/Gradle/unityLibrary/src/main/assets";
                if (Directory.Exists(StaleAssets))
                {
                    Directory.Delete(StaleAssets, true);
                    Debug.Log("[EyeLab] removed " + StaleAssets + " before the build");
                }
                EditorUserBuildSettings.buildAppBundle = false;
                var options = new BuildPlayerOptions
                {
                    scenes = new[] { Target },
                    locationPathName = apk,
                    target = BuildTarget.Android,
                    targetGroup = BuildTargetGroup.Android,
                    options = BuildOptions.None,
                };
                BuildReport report = BuildPipeline.BuildPlayer(options);
                BuildSummary s = report.summary;
                Debug.Log($"[EyeLab] probe APK {s.result}: {apk}, {s.totalSize / 1048576.0:F1} MB, " +
                          $"{s.totalErrors} errors, {s.totalWarnings} warnings, {s.totalTime.TotalSeconds:F0} s");
                if (s.result == BuildResult.Succeeded) code = CheckApk(apk) ? 0 : 2;
            }
            catch (Exception e)
            {
                Debug.LogError("[EyeLab] probe APK build failed: " + e);
            }
            EditorApplication.Exit(code);
        }

        /// <summary>The ArUco plugin is Android arm64 only; fail early when it was not built.</summary>
        static void ConfigureArucoPlugin()
        {
            AssetDatabase.ImportAsset(ArucoPlugin, ImportAssetOptions.ForceSynchronousImport);
            if (!(AssetImporter.GetAtPath(ArucoPlugin) is PluginImporter imp))
                throw new InvalidOperationException(ArucoPlugin + " missing: run bash tools/build_aruco_plugin.sh first");
            imp.SetCompatibleWithAnyPlatform(false);
            imp.SetCompatibleWithEditor(false);
            imp.SetCompatibleWithPlatform(BuildTarget.StandaloneWindows64, false);
            imp.SetCompatibleWithPlatform(BuildTarget.Android, true);
            imp.SetPlatformData(BuildTarget.Android, "CPU", "ARM64");
            imp.SaveAndReimport();
            Debug.Log("[EyeLab] " + ArucoPlugin + " set to Android ARM64");
        }

        /// <summary>An APK must carry its scenes either as loose files (level0, globalgamemanagers) or as data.unity3d, never both.</summary>
        static bool CheckApk(string apk)
        {
            using (var zip = System.IO.Compression.ZipFile.OpenRead(apk))
            {
                bool bundle = zip.GetEntry("assets/bin/Data/data.unity3d") != null;
                bool loose = zip.GetEntry("assets/bin/Data/level0") != null;
                if (bundle && loose)
                {
                    Debug.LogError("[EyeLab] APK holds both data.unity3d and loose level0: a stale bundle would win at run time");
                    return false;
                }
                Debug.Log($"[EyeLab] APK scene data: {(bundle ? "data.unity3d" : "loose files")}");
                if (zip.GetEntry("lib/arm64-v8a/libeyelab_aruco.so") == null)
                {
                    Debug.LogError("[EyeLab] APK has no lib/arm64-v8a/libeyelab_aruco.so");
                    return false;
                }
                return true;
            }
        }

        static string Arg(string name)
        {
            string[] a = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(a, name);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
        }
    }
}
