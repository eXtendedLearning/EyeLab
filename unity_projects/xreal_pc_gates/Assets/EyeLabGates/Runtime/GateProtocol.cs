using System.Collections.Generic;

namespace EyeLabGates
{
    public enum StepAction { None, CameraStart, CameraStop, Track3DoF, Track6DoF }

    /// <summary>One guided step: shown on the HUD with a countdown, announced by a beep, logged as a
    /// <c>step</c> event (result = id). The action runs when the step begins.</summary>
    public sealed class ProtocolStep
    {
        public readonly string Id;
        public readonly string Instruction;
        public readonly float Seconds;
        public readonly StepAction Action;

        public ProtocolStep(string id, string instruction, float seconds, StepAction action = StepAction.None)
        {
            Id = id;
            Instruction = instruction;
            Seconds = seconds;
            Action = action;
        }
    }

    /// <summary>
    /// The operator protocols of docs/XREAL-PC-GATES-G1-G3.md §6 as timed steps, so the run needs no
    /// manual marks. Step ids are the phase names the analysis segments on.
    /// </summary>
    public static class GateProtocol
    {
        public static List<ProtocolStep> G1()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "Look at the plate. Keep your head STILL.", 20),
                new ProtocolStep("motion", "Move your head SLOWLY (small nods and turns); keep the plate in view.", 20),
                new ProtocolStep("covered", "COVER the Eye camera with your hand. Keep still.", 5),
                new ProtocolStep("uncovered", "UNCOVER. Look at the plate, keep still.", 10),
            };
            for (int i = 1; i <= 3; i++)
            {
                s.Add(new ProtocolStep($"restart_{i}_off", $"Keep still: automatic camera restart {i}/3 (camera off).", 2, StepAction.CameraStop));
                s.Add(new ProtocolStep($"restart_{i}_on", $"Keep still: automatic camera restart {i}/3 (camera on).", 10, StepAction.CameraStart));
            }
            return s;
        }

        public static List<ProtocolStep> G2()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "Stand on A, face forward. Keep your head STILL.", 30),
            };
            for (int i = 1; i <= 3; i++)
                AddTranslation(s, $"trans_{i}", $"{i}/3");
            for (int i = 1; i <= 3; i++)
            {
                s.Add(new ProtocolStep($"rot_{i}_ref1", $"Rotation {i}/3: face REFERENCE 1, keep still.", 5));
                s.Add(new ProtocolStep($"rot_{i}_turn", $"Rotation {i}/3: turn to REFERENCE 2 (90 deg), body and head together.", 4));
                s.Add(new ProtocolStep($"rot_{i}_ref2", $"Rotation {i}/3: hold on REFERENCE 2, keep still.", 5));
                s.Add(new ProtocolStep($"rot_{i}_back", $"Rotation {i}/3: turn back to REFERENCE 1.", 4));
                s.Add(new ProtocolStep($"rot_{i}_ref1_end", $"Rotation {i}/3: hold on REFERENCE 1, keep still.", 5));
            }
            s.Add(new ProtocolStep("covered", "On A, still: COVER the Eye camera with your hand.", 5));
            s.Add(new ProtocolStep("uncovered", "UNCOVER. Keep still.", 5));
            s.Add(new ProtocolStep("dof3_switch", "Switching to 3DoF. Stand on A, keep still.", 5, StepAction.Track3DoF));
            AddTranslation(s, "dof3_trans", "3DoF");
            s.Add(new ProtocolStep("dof6_switch", "Switching back to 6DoF. Keep still.", 5, StepAction.Track6DoF));
            return s;
        }

        /// <summary>S3: the S1 phases, one translation and free looking; at least 120 s with the camera on.</summary>
        public static List<ProtocolStep> G3()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "Stand on A, look at the plate. Keep your head STILL.", 20),
                new ProtocolStep("motion", "Move your head SLOWLY (small nods and turns); keep the plate in view.", 20),
                new ProtocolStep("covered", "COVER the Eye camera with your hand. Keep still.", 5),
                new ProtocolStep("uncovered", "UNCOVER. Look at the plate, keep still.", 10),
            };
            AddTranslation(s, "trans_1", "1/1");
            s.Add(new ProtocolStep("look_around", "Look around the room SLOWLY.", 40));
            s.Add(new ProtocolStep("static_end", "Look at the plate. Keep your head STILL.", 20));
            return s;
        }

        static void AddTranslation(List<ProtocolStep> s, string id, string label)
        {
            s.Add(new ProtocolStep($"{id}_A", $"Translation {label}: stand on A facing forward, keep still.", 5));
            s.Add(new ProtocolStep($"{id}_toB", $"Translation {label}: SIDE-STEP RIGHT to B; keep height and facing.", 4));
            s.Add(new ProtocolStep($"{id}_B", $"Translation {label}: hold on B, keep still.", 5));
            s.Add(new ProtocolStep($"{id}_toA", $"Translation {label}: SIDE-STEP back to A.", 4));
            s.Add(new ProtocolStep($"{id}_A_end", $"Translation {label}: hold on A, keep still.", 5));
        }

        public static float TotalSeconds(List<ProtocolStep> steps)
        {
            float t = 0;
            foreach (var s in steps) t += s.Seconds;
            return t;
        }
    }
}
