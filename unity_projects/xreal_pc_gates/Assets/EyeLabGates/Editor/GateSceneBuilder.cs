// D5 of docs/XREAL-PC-GATES-G1-G3.md: builds Assets/EyeLabGates/Scenes/GateHarness.unity from the
// vendor RGBCamera sample scene. The vendor scene itself is left untouched (control build).
//   unity run <project> -- -executeMethod EyeLabGates.EditorTools.GateSceneBuilder.Build
using System;
using System.Linq;
using Unity.XR.XREAL;
using Unity.XR.XREAL.Experimental.Samples;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

namespace EyeLabGates.EditorTools
{
    public static class GateSceneBuilder
    {
        const string k_Tag = "[GateSceneBuilder]";
        public const string SampleRoot = "Assets/Samples/XREAL XR Plugin Experimental/3.1.2/Standalone";
        public const string VendorScene = SampleRoot + "/RGBCamera/RGBCamera.unity";
        public const string HarnessScene = "Assets/EyeLabGates/Scenes/GateHarness.unity";

        static void Log(string msg) => Debug.Log($"{k_Tag} {msg}");

        public static void Build()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(VendorScene) == null)
                throw new Exception($"{k_Tag} vendor scene missing: {VendorScene} (D3 not done?)");

            if (!AssetDatabase.IsValidFolder("Assets/EyeLabGates/Scenes"))
                AssetDatabase.CreateFolder("Assets/EyeLabGates", "Scenes");
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(HarnessScene) != null)
            {
                AssetDatabase.DeleteAsset(HarnessScene);
                Log($"deleted previous {HarnessScene}");
            }
            if (!AssetDatabase.CopyAsset(VendorScene, HarnessScene))
                throw new Exception($"{k_Tag} CopyAsset failed");
            Log($"copied {VendorScene} -> {HarnessScene}");

            var scene = EditorSceneManager.OpenScene(HarnessScene, OpenSceneMode.Single);

            var textures = UnityEngine.Object.FindObjectsByType<XREALRGBCameraTexture>(FindObjectsInactive.Include);
            var hellos = UnityEngine.Object.FindObjectsByType<HelloMR>(FindObjectsInactive.Include);
            Log($"found XREALRGBCameraTexture x{textures.Length} ({string.Join(",", textures.Select(t => t.gameObject.name))}), " +
                $"HelloMR x{hellos.Length} ({string.Join(",", hellos.Select(h => h.gameObject.name))})");
            if (textures.Length != 1 || hellos.Length != 1)
                throw new Exception($"{k_Tag} expected exactly one XREALRGBCameraTexture and one HelloMR");
            var texture = textures[0];
            var hello = hellos[0];

            // The prefab ships m_UseSecondWindow = true. In a player, HelloMR.StartXRAsync then calls
            // XREALExperimentalPlugin.Restart() when the 3840x1080 display was not enumerated at startup,
            // which relaunches the exe WITHOUT arguments and TerminateProcess()es the running gate.
            // The harness uses the main window instead (the HUD is then shown on the glasses).
            var helloSo = new SerializedObject(hello);
            var secondWindow = helloSo.FindProperty("m_UseSecondWindow");
            Log($"HelloMR.m_UseSecondWindow was {secondWindow.boolValue}; setting false in the harness scene only");
            secondWindow.boolValue = false;
            helloSo.ApplyModifiedPropertiesWithoutUndo();

            // The sample's on-screen Play/Stop buttons call the vendor texture directly; a click stopped
            // the 2026-10-13 G1 capture. Hide them; the harness controls the camera.
            var example = UnityEngine.Object.FindAnyObjectByType<RGBCameraExample>(FindObjectsInactive.Include);
            if (example == null)
                throw new Exception($"{k_Tag} RGBCameraExample not found");
            var exampleSo = new SerializedObject(example);
            var preview = exampleSo.FindProperty("m_YUVImage").objectReferenceValue as RawImage;
            foreach (var name in new[] { "m_PlayButton", "m_StopButton" })
            {
                var button = exampleSo.FindProperty(name).objectReferenceValue as Button;
                if (button == null)
                    throw new Exception($"{k_Tag} RGBCameraExample.{name} not wired in the sample");
                button.gameObject.SetActive(false);
                Log($"hid sample button {button.gameObject.name}");
            }

            var go = new GameObject("GateHarness");
            var controller = go.AddComponent<GateController>(); // RequireComponent adds probes and sources
            var controllerSo = new SerializedObject(controller);
            controllerSo.FindProperty("m_HelloMR").objectReferenceValue = hello;
            controllerSo.FindProperty("m_CameraTexture").objectReferenceValue = texture;
            controllerSo.ApplyModifiedPropertiesWithoutUndo();
            var vendorSo = new SerializedObject(go.GetComponent<VendorCameraSource>());
            vendorSo.FindProperty("m_Texture").objectReferenceValue = texture;
            vendorSo.ApplyModifiedPropertiesWithoutUndo();
            var drainSo = new SerializedObject(go.GetComponent<DrainingCameraSource>());
            drainSo.FindProperty("m_Preview").objectReferenceValue = preview;
            drainSo.ApplyModifiedPropertiesWithoutUndo();
            Log($"wired vendor source -> {texture.gameObject.name}, drain preview -> {(preview != null ? preview.gameObject.name : "null")}");
            Log($"added GateHarness: components [{string.Join(",", go.GetComponents<Component>().Select(c => c.GetType().Name))}]");

            if (!EditorSceneManager.SaveScene(scene))
                throw new Exception($"{k_Tag} SaveScene failed");
            Log($"saved {HarnessScene}");

            Verify();

            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(HarnessScene, true) };
            Log("EditorBuildSettings.scenes = [GateHarness] (Play-mode convenience; BuildGates passes scenes explicitly)");
        }

        static void Verify()
        {
            EditorSceneManager.OpenScene(HarnessScene, OpenSceneMode.Single);
            var controller = UnityEngine.Object.FindAnyObjectByType<GateController>();
            if (controller == null || controller.HelloMR == null || controller.CameraTexture == null)
                throw new Exception($"{k_Tag} VERIFY failed: GateController references not saved");
            var vendor = controller.GetComponent<VendorCameraSource>();
            var drain = controller.GetComponent<DrainingCameraSource>();
            if (vendor == null || vendor.Texture != controller.CameraTexture)
                throw new Exception($"{k_Tag} VERIFY failed: VendorCameraSource texture not wired");
            if (drain == null || drain.Preview == null)
                throw new Exception($"{k_Tag} VERIFY failed: DrainingCameraSource preview not wired");
            if (controller.GetComponent<PerfProbe>() == null || controller.GetComponent<CameraProbe>() == null)
                throw new Exception($"{k_Tag} VERIFY failed: probes missing");
            var example = UnityEngine.Object.FindAnyObjectByType<RGBCameraExample>(FindObjectsInactive.Include);
            var exampleSo = new SerializedObject(example);
            foreach (var name in new[] { "m_PlayButton", "m_StopButton" })
                if (((Button)exampleSo.FindProperty(name).objectReferenceValue).gameObject.activeSelf)
                    throw new Exception($"{k_Tag} VERIFY failed: {name} still visible");
            bool second = new SerializedObject(controller.HelloMR).FindProperty("m_UseSecondWindow").boolValue;
            if (second)
                throw new Exception($"{k_Tag} VERIFY failed: m_UseSecondWindow still true");
            Log($"VERIFY OK: HelloMR={controller.HelloMR.gameObject.name}, texture on {controller.CameraTexture.gameObject.name}, " +
                $"m_UseSecondWindow={second}");

            EditorSceneManager.OpenScene(VendorScene, OpenSceneMode.Single);
            var vendorHello = UnityEngine.Object.FindAnyObjectByType<HelloMR>();
            bool vendorSecond = new SerializedObject(vendorHello).FindProperty("m_UseSecondWindow").boolValue;
            bool vendorHasHarness = UnityEngine.Object.FindAnyObjectByType<GateController>() != null;
            Log($"VERIFY vendor scene untouched: m_UseSecondWindow={vendorSecond}, has GateController={vendorHasHarness}");
            if (vendorHasHarness)
                throw new Exception($"{k_Tag} VERIFY failed: vendor scene was modified");
        }
    }
}
