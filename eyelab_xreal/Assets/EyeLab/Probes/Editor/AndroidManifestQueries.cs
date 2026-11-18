using System.IO;
using UnityEditor.Android;
using UnityEngine;

namespace EyeLab.Probes.Editor
{
    /// <summary>
    /// T6.3: the spoken cues (<c>CueSpeech.java</c>) bind to the system text-to-speech engine. From Android 11 an app
    /// targeting API 30+ (EyeLab targets 34) sees no TTS engine unless its manifest declares the query for
    /// <c>android.intent.action.TTS_SERVICE</c>. Unity generates the manifest, so the query is added to the generated
    /// unityLibrary manifest after each Gradle export; Gradle merges it into the APK.
    /// </summary>
    public sealed class AndroidManifestQueries : IPostGenerateGradleAndroidProject
    {
        const string Query =
            "    <queries>\n        <intent>\n            <action android:name=\"android.intent.action.TTS_SERVICE\" />\n        </intent>\n    </queries>\n";

        public int callbackOrder => 10;

        public void OnPostGenerateGradleAndroidProject(string path)
        {
            string manifest = Path.Combine(path, "src", "main", "AndroidManifest.xml");
            if (!File.Exists(manifest)) { Debug.LogWarning("[EyeLab] no manifest at " + manifest); return; }
            string xml = File.ReadAllText(manifest);
            if (xml.Contains("android.intent.action.TTS_SERVICE")) return;
            int app = xml.IndexOf("<application");
            if (app < 0) { Debug.LogWarning("[EyeLab] manifest has no <application>: TTS query not added"); return; }
            File.WriteAllText(manifest, xml.Insert(app, Query.TrimStart() + "  "));
            Debug.Log("[EyeLab] TTS_SERVICE query added to " + manifest);
        }
    }
}
