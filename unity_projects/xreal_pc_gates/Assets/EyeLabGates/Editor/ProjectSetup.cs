// D4 of docs/XREAL-PC-GATES-G1-G3.md: idempotent project configuration for the
// XREAL PC-beta gate project. Run in batch mode:
//   unity run <project> -- -executeMethod EyeLabGates.EditorTools.ProjectSetup.Apply
// Every value is logged with the "[ProjectSetup]" prefix; a failing check throws,
// which makes the batch Editor exit non-zero.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.XR.XREAL;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;

namespace EyeLabGates.EditorTools
{
    public static class ProjectSetup
    {
        const string k_Tag = "[ProjectSetup]";
        const string k_XRFolder = "Assets/XR";
        const string k_GeneralSettingsPath = "Assets/XR/XRGeneralSettingsPerBuildTarget.asset";
        const string k_XrealSettingsPath = "Assets/XR/Settings/XREALSettings.asset";
        const string k_LoaderType = "Unity.XR.XREAL.XREALXRLoader";

        public const string CompanyName = "EyeLab";
        public const string ProductName = "XrealPcGates";

        static void Log(string msg) => Debug.Log($"{k_Tag} {msg}");

        public static void Apply()
        {
            Log("begin");
            SetBuildTarget();
            SetPlayer();
            SetInputHandling();
            SetXRManagement();
            AssetDatabase.SaveAssets();
            Verify();
            Log("end: OK");
        }

        static void SetBuildTarget()
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneWindows64)
            {
                bool ok = EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneWindows64);
                Log($"SwitchActiveBuildTarget(StandaloneWindows64) -> {ok}");
                if (!ok)
                    throw new Exception($"{k_Tag} could not switch active build target");
            }
            Log($"activeBuildTarget = {EditorUserBuildSettings.activeBuildTarget} (x86_64)");
        }

        static void SetPlayer()
        {
            var standalone = NamedBuildTarget.Standalone;

            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, new[] { GraphicsDeviceType.Direct3D11 });
            Log($"Win64 UseDefaultGraphicsAPIs = {PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64)}");
            Log($"Win64 GraphicsAPIs = [{string.Join(",", PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64))}]");

            PlayerSettings.SetScriptingBackend(standalone, ScriptingImplementation.Mono2x);
            Log($"Standalone scripting backend = {PlayerSettings.GetScriptingBackend(standalone)}");

            PlayerSettings.companyName = CompanyName;
            PlayerSettings.productName = ProductName;
            Log($"companyName = {PlayerSettings.companyName}, productName = {PlayerSettings.productName}");

            PlayerSettings.runInBackground = true;
            PlayerSettings.visibleInBackground = true;
            Log($"runInBackground = {PlayerSettings.runInBackground}, visibleInBackground = {PlayerSettings.visibleInBackground}");

            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.resizableWindow = true;
            Log($"fullScreenMode = {PlayerSettings.fullScreenMode}, default = {PlayerSettings.defaultScreenWidth}x{PlayerSettings.defaultScreenHeight}, resizable = {PlayerSettings.resizableWindow}");
        }

        static void SetInputHandling()
        {
            // 0 = Input Manager (legacy), 1 = Input System, 2 = Both.
            var playerSettings = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/ProjectSettings.asset")
                .FirstOrDefault(o => o is PlayerSettings);
            if (playerSettings == null)
                throw new Exception($"{k_Tag} PlayerSettings object not found");
            var so = new SerializedObject(playerSettings);
            var prop = so.FindProperty("activeInputHandler");
            if (prop == null)
                throw new Exception($"{k_Tag} property activeInputHandler not found");
            if (prop.intValue != 2)
            {
                prop.intValue = 2;
                so.ApplyModifiedPropertiesWithoutUndo();
            }
            Log($"activeInputHandler = {prop.intValue} (Both)");
        }

        static void SetXRManagement()
        {
            if (!AssetDatabase.IsValidFolder(k_XRFolder))
                AssetDatabase.CreateFolder("Assets", "XR");
            if (!AssetDatabase.IsValidFolder(k_XRFolder + "/Settings"))
                AssetDatabase.CreateFolder(k_XRFolder, "Settings");

            // Per-build-target general settings (what Project Settings > XR Plug-in Management creates).
            if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.settingsKey, out XRGeneralSettingsPerBuildTarget perTarget) || perTarget == null)
            {
                perTarget = AssetDatabase.LoadAssetAtPath<XRGeneralSettingsPerBuildTarget>(k_GeneralSettingsPath);
                if (perTarget == null)
                {
                    perTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                    AssetDatabase.CreateAsset(perTarget, k_GeneralSettingsPath);
                    Log($"created {k_GeneralSettingsPath}");
                }
                EditorBuildSettings.AddConfigObject(XRGeneralSettings.settingsKey, perTarget, true);
            }
            Log($"XRGeneralSettingsPerBuildTarget = {AssetDatabase.GetAssetPath(perTarget)}");

            if (!perTarget.HasManagerSettingsForBuildTarget(BuildTargetGroup.Standalone))
            {
                perTarget.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Standalone);
                Log("created Standalone XRGeneralSettings + XRManagerSettings");
            }
            var general = perTarget.SettingsForBuildTarget(BuildTargetGroup.Standalone);
            general.InitManagerOnStart = false;
            EditorUtility.SetDirty(general);

            var manager = general.Manager;
            if (!manager.activeLoaders.Any(l => l != null && l.GetType().FullName == k_LoaderType))
            {
                bool assigned = XRPackageMetadataStore.AssignLoader(manager, k_LoaderType, BuildTargetGroup.Standalone);
                Log($"AssignLoader({k_LoaderType}, Standalone) -> {assigned}");
            }
            EditorUtility.SetDirty(manager);
            EditorUtility.SetDirty(perTarget);

            // XREALXRLoader dereferences XREALSettings.GetSettings() without a null check,
            // so the settings object the XR UI would create must exist. Defaults are kept.
            if (!EditorBuildSettings.TryGetConfigObject(XREALSettings.k_SettingsKey, out XREALSettings xreal) || xreal == null)
            {
                xreal = AssetDatabase.LoadAssetAtPath<XREALSettings>(k_XrealSettingsPath);
                if (xreal == null)
                {
                    xreal = ScriptableObject.CreateInstance<XREALSettings>();
                    AssetDatabase.CreateAsset(xreal, k_XrealSettingsPath);
                    Log($"created {k_XrealSettingsPath}");
                }
                EditorBuildSettings.AddConfigObject(XREALSettings.k_SettingsKey, xreal, true);
            }
            Log($"XREALSettings = {AssetDatabase.GetAssetPath(xreal)}: StereoRendering = {xreal.StereoRendering}, " +
                $"InitialTrackingType = {xreal.InitialTrackingType}, SupportDevices = [{string.Join(",", xreal.SupportDevices)}]");
        }

        static void Verify()
        {
            var failures = new List<string>();

            var general = XRGeneralSettingsPerBuildTarget.XRGeneralSettingsForBuildTarget(BuildTargetGroup.Standalone);
            var loaders = general?.Manager?.activeLoaders?.Where(l => l != null).Select(l => l.GetType().FullName).ToList()
                          ?? new List<string>();
            bool loaderOk = loaders.Contains(k_LoaderType);
            Log($"VERIFY Standalone loaders = [{string.Join(",", loaders)}] -> loader assigned: {loaderOk}");
            if (!loaderOk) failures.Add("XREAL loader not assigned for Standalone");

            bool initOnStart = general != null && general.InitManagerOnStart;
            Log($"VERIFY InitManagerOnStart = {initOnStart}");
            if (initOnStart) failures.Add("InitManagerOnStart is true");

            var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
            bool gfxOk = !PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64)
                         && apis.Length == 1 && apis[0] == GraphicsDeviceType.Direct3D11;
            Log($"VERIFY graphics API = [{string.Join(",", apis)}] -> Direct3D11 only: {gfxOk}");
            if (!gfxOk) failures.Add("graphics API is not Direct3D11 only");

            Log($"VERIFY XREALUtility.IsLoaderActive() = {XREALUtility.IsLoaderActive()}");

            // Evaluate the rules XREALProjectValidator registered with the core-utils BuildValidator.
            // BuildValidator.PlatformRules is internal; read it by reflection.
            var rules = ReadValidatorRules(BuildTargetGroup.Standalone);
            if (rules == null)
            {
                Log("VERIFY BuildValidator rules not readable (reflection); relying on the explicit checks above");
            }
            else
            {
                foreach (var rule in rules)
                {
                    string category = Prop<string>(rule, "Category");
                    if (category != "XREAL") continue;
                    string message = Prop<string>(rule, "Message");
                    bool enabled = Prop<Func<bool>>(rule, "IsRuleEnabled")?.Invoke() ?? true;
                    bool passes = Prop<Func<bool>>(rule, "CheckPredicate")?.Invoke() ?? true;
                    bool error = Prop<bool>(rule, "Error");
                    Log($"VERIFY rule '{message}': enabled={enabled} passes={passes} error={error}");
                    if (enabled && !passes && error)
                        failures.Add($"validator rule failing: {message}");
                }
            }

            if (failures.Count > 0)
                throw new Exception($"{k_Tag} FAILED: {string.Join("; ", failures)}");
        }

        static IEnumerable<object> ReadValidatorRules(BuildTargetGroup group)
        {
            var type = Type.GetType("Unity.XR.CoreUtils.Editor.BuildValidator, Unity.XR.CoreUtils.Editor");
            var prop = type?.GetProperty("PlatformRules", BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            if (!(prop?.GetValue(null) is IDictionary dict) || !dict.Contains(group))
                return null;
            return (dict[group] as IEnumerable)?.Cast<object>();
        }

        static T Prop<T>(object o, string name)
        {
            var p = o.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public);
            return p == null ? default : (T)p.GetValue(o);
        }
    }
}
