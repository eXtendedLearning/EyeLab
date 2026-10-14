using System.Collections.Generic;

namespace EyeLabGates
{
    public enum StepAction { None, CameraStart, CameraStop, Track3DoF, Track6DoF }

    /// <summary>
    /// One guided step. <see cref="Title"/> is shown large in the middle of each eye and spoken;
    /// <see cref="Detail"/> is the small line under it. The step is logged as a <c>step</c> event
    /// (result = <see cref="Id"/>); <see cref="Action"/> runs when it begins.
    /// </summary>
    public sealed class ProtocolStep
    {
        public readonly string Id;
        public readonly string Title;
        public readonly string Detail;
        public readonly string Speech;
        public readonly float Seconds;
        public readonly StepAction Action;

        public ProtocolStep(string id, string title, string detail, float seconds,
                            StepAction action = StepAction.None, string speech = null)
        {
            Id = id;
            Title = title;
            Detail = detail;
            Seconds = seconds;
            Action = action;
            Speech = speech ?? title;
        }
    }

    /// <summary>
    /// The operator protocols of docs/XREAL-PC-GATES-G1-G3.md §6 as timed steps, so the run needs no
    /// manual marks. Step ids are the phase names the analysis segments on. "A"/"B" are the two floor
    /// tape marks 500 mm apart (B to the right of A), not specimen plates.
    /// </summary>
    public static class GateProtocol
    {
        public static List<ProtocolStep> G1()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "HOLD STILL", "Look at the board", 20, speech: "Hold still, look at the board"),
                new ProtocolStep("motion", "MOVE HEAD SLOWLY", "Small nods and turns, keep the board in view", 20, speech: "Move your head slowly"),
                new ProtocolStep("covered", "COVER THE EYE", "Hand over the camera, keep still", 5, speech: "Cover the camera"),
                new ProtocolStep("uncovered", "UNCOVER, HOLD STILL", "Look at the board", 10, speech: "Uncover, hold still"),
            };
            for (int i = 1; i <= 3; i++)
            {
                s.Add(new ProtocolStep($"restart_{i}_off", "HOLD STILL", $"Automatic camera restart {i}/3 (off)", 2, StepAction.CameraStop,
                                       i == 1 ? "Hold still for the camera restarts" : ""));
                s.Add(new ProtocolStep($"restart_{i}_on", "HOLD STILL", $"Automatic camera restart {i}/3 (on)", 10, StepAction.CameraStart, ""));
            }
            return s;
        }

        public static List<ProtocolStep> G2()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "HOLD STILL ON A", "Stand on floor mark A, face forward", 30, speech: "Stand on A and hold still"),
            };
            for (int i = 1; i <= 3; i++)
                AddTranslation(s, $"trans_{i}", $"Translation {i}/3");
            for (int i = 1; i <= 3; i++)
            {
                string r = $"Rotation {i}/3";
                s.Add(new ProtocolStep($"rot_{i}_ref1", "FACE REFERENCE 1", $"{r}: hold still", 5, speech: "Face reference one"));
                s.Add(new ProtocolStep($"rot_{i}_turn", "TURN TO REFERENCE 2", $"{r}: 90 degrees, body and head together", 4, speech: "Turn to reference two"));
                s.Add(new ProtocolStep($"rot_{i}_ref2", "HOLD ON REFERENCE 2", $"{r}: hold still", 5, speech: "Hold"));
                s.Add(new ProtocolStep($"rot_{i}_back", "TURN BACK TO REFERENCE 1", r, 4, speech: "Turn back to reference one"));
                s.Add(new ProtocolStep($"rot_{i}_ref1_end", "HOLD ON REFERENCE 1", $"{r}: hold still", 5, speech: "Hold"));
            }
            s.Add(new ProtocolStep("covered", "COVER THE EYE", "On A, hand over the camera, keep still", 5, speech: "Cover the camera"));
            s.Add(new ProtocolStep("uncovered", "UNCOVER, HOLD STILL", "On A", 5, speech: "Uncover, hold still"));
            s.Add(new ProtocolStep("dof3_switch", "3DOF NOW: HOLD STILL ON A", "Tracking switched to 3DoF", 5, StepAction.Track3DoF,
                                   "Three degrees of freedom now. Hold still on A"));
            AddTranslation(s, "dof3_trans", "3DoF translation");
            s.Add(new ProtocolStep("dof6_switch", "6DOF AGAIN: HOLD STILL", "Tracking switched back to 6DoF", 5, StepAction.Track6DoF,
                                   "Six degrees of freedom again. Hold still"));
            return s;
        }

        /// <summary>S3: the S1 phases, one translation and free looking; at least 120 s with the camera on.</summary>
        public static List<ProtocolStep> G3()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "HOLD STILL", "On A, look at the board", 20, speech: "Look at the board and hold still"),
                new ProtocolStep("motion", "MOVE HEAD SLOWLY", "Small nods and turns, keep the board in view", 20, speech: "Move your head slowly"),
                new ProtocolStep("covered", "COVER THE EYE", "Hand over the camera, keep still", 5, speech: "Cover the camera"),
                new ProtocolStep("uncovered", "UNCOVER, HOLD STILL", "Look at the board", 10, speech: "Uncover, hold still"),
            };
            AddTranslation(s, "trans_1", "Translation");
            s.Add(new ProtocolStep("look_around", "LOOK AROUND SLOWLY", "Slowly scan the room", 40, speech: "Look around the room slowly"));
            s.Add(new ProtocolStep("static_end", "HOLD STILL", "Look at the board", 20, speech: "Look at the board and hold still"));
            return s;
        }

        static void AddTranslation(List<ProtocolStep> s, string id, string label)
        {
            s.Add(new ProtocolStep($"{id}_A", "ON A, HOLD STILL", $"{label}: face forward", 5, speech: "On A, hold still"));
            s.Add(new ProtocolStep($"{id}_toB", "STEP RIGHT TO B", $"{label}: keep height and facing", 4, speech: "Step right to B"));
            s.Add(new ProtocolStep($"{id}_B", "HOLD ON B", $"{label}: keep still", 5, speech: "Hold"));
            s.Add(new ProtocolStep($"{id}_toA", "STEP BACK TO A", $"{label}: keep height and facing", 4, speech: "Step back to A"));
            s.Add(new ProtocolStep($"{id}_A_end", "HOLD ON A", $"{label}: keep still", 5, speech: "Hold"));
        }

        public static float TotalSeconds(List<ProtocolStep> steps)
        {
            float t = 0;
            foreach (var s in steps) t += s.Seconds;
            return t;
        }
    }
}
