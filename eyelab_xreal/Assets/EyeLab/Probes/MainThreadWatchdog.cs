using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using UnityEngine;
using UnityEngine.LowLevel;
using Debug = UnityEngine.Debug;

namespace EyeLab.Probes
{
    /// <summary>
    /// Diagnostic for the Beam Pro bring-up (T6.0 B0, 2026-11-17): the Unity main thread stopped right after
    /// <c>[XREALVirtualController] Start</c> and no script after it ran. Without root there are no native stacks,
    /// so this marks every player-loop subsystem and a background thread logs, once a second while the loop is
    /// stuck (and every 5 s otherwise), the frame number and the subsystem the main thread entered last.
    /// Log tag: <c>[EyeLab watchdog]</c>. Costs one delegate call per subsystem per frame.
    /// </summary>
    public static class MainThreadWatchdog
    {
        static string[] s_Names;
        static volatile int s_Phase = -1;
        static volatile int s_Frame;
        static long s_LastChangeTicks;
        static Thread s_Thread;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void Install()
        {
            if (s_Thread != null) return;
            Debug.Log("[EyeLab watchdog] installing (assembly loaded)");
            var names = new List<string>();
            PlayerLoopSystem root = PlayerLoop.GetCurrentPlayerLoop();
            PlayerLoopSystem[] top = root.subSystemList;
            for (int i = 0; i < top.Length; i++)
            {
                PlayerLoopSystem[] subs = top[i].subSystemList ?? Array.Empty<PlayerLoopSystem>();
                var marked = new List<PlayerLoopSystem>(subs.Length * 2 + 1);
                string topName = top[i].type != null ? top[i].type.Name : "?";
                marked.Add(Marker(names, topName + ".begin"));
                foreach (PlayerLoopSystem s in subs)
                {
                    marked.Add(Marker(names, topName + "." + (s.type != null ? s.type.Name : "?")));
                    marked.Add(s);
                }
                if (topName == "PostLateUpdate") marked.Add(Marker(names, "FrameEnd", frameEnd: true));
                top[i].subSystemList = marked.ToArray();
            }
            root.subSystemList = top;
            PlayerLoop.SetPlayerLoop(root);
            s_Names = names.ToArray();
            Interlocked.Exchange(ref s_LastChangeTicks, Stopwatch.GetTimestamp());
            s_Thread = new Thread(Watch) { IsBackground = true, Name = "eyelab-watchdog" };
            s_Thread.Start();
        }

        /// <summary>After the first scene loads: every root object and its components (null = missing script).</summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void ReportScene()
        {
            UnityEngine.SceneManagement.Scene scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            Debug.Log($"[EyeLab watchdog] scene '{scene.name}' ({scene.path}), build index {scene.buildIndex}, {scene.rootCount} roots");
            foreach (GameObject go in scene.GetRootGameObjects())
            {
                var parts = new List<string>();
                foreach (Component c in go.GetComponents<Component>())
                    parts.Add(c == null ? "MISSING" : c.GetType().FullName + (c is Behaviour b && !b.enabled ? "(disabled)" : ""));
                Debug.Log($"[EyeLab watchdog] root '{go.name}' active={go.activeInHierarchy}: {string.Join(", ", parts)}");
            }
            Debug.Log("[EyeLab watchdog] ProbeRunner instances: " + UnityEngine.Object.FindObjectsByType<ProbeRunner>(FindObjectsInactive.Include, FindObjectsSortMode.None).Length);
        }

        static PlayerLoopSystem Marker(List<string> names, string name, bool frameEnd = false)
        {
            int id = names.Count;
            names.Add(name);
            return new PlayerLoopSystem
            {
                type = typeof(MainThreadWatchdog),
                updateDelegate = () =>
                {
                    s_Phase = id;
                    if (frameEnd) s_Frame++;
                    Interlocked.Exchange(ref s_LastChangeTicks, Stopwatch.GetTimestamp());
                },
            };
        }

        static void Watch()
        {
            int lastFrame = 0;
            double lastBeat = 0;
            var clock = Stopwatch.StartNew();
            while (true)
            {
                Thread.Sleep(1000);
                double stuck = (Stopwatch.GetTimestamp() - Interlocked.Read(ref s_LastChangeTicks)) / (double)Stopwatch.Frequency;
                int frame = s_Frame, phase = s_Phase;
                string name = phase >= 0 && s_Names != null && phase < s_Names.Length ? s_Names[phase] : "none yet";
                double t = clock.Elapsed.TotalSeconds;
                if (stuck > 1.0)
                    Debug.LogWarning($"[EyeLab watchdog] STUCK {stuck:F1} s in {name} (after it), frame {frame}, t={t:F0} s");
                else if (t - lastBeat >= 5.0)
                {
                    Debug.Log($"[EyeLab watchdog] alive: frame {frame} ({(frame - lastFrame) / (t - lastBeat):F1} fps), last {name}, t={t:F0} s");
                    lastBeat = t;
                    lastFrame = frame;
                }
            }
        }
    }
}
