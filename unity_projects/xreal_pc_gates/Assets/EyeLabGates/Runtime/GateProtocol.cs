using System.Collections.Generic;
using UnityEngine;

namespace EyeLabGates
{
    public enum StepAction { None, CameraStart, CameraStop, Track3DoF, Track6DoF, AnchorFreeze, AnchorLive }

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
        /// <summary>Calibration: where the board should be in the camera image (normalised, y down); null = anywhere.</summary>
        public readonly Rect? Zone;

        public ProtocolStep(string id, string title, string detail, float seconds,
                            StepAction action = StepAction.None, string speech = null, Rect? zone = null)
        {
            Zone = zone;
            Id = id;
            Title = title;
            Detail = detail;
            Seconds = seconds;
            Action = action;
            Speech = speech ?? title;
        }
    }

    /// <summary>
    /// The operator protocols of docs/XREAL-PC-GATES-G1-G3.md §6 (G4: §16) as timed steps, so the run needs no
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

        /// <summary>
        /// G4 light overlay, about 5 min: the world-locked grid is placed on the board at the start and
        /// stays in view; the camera only takes short bursts. The long run tests whether the light load
        /// is thermally sustainable; the steps test whether the grid stays put (6DoF) or drifts (3DoF).
        /// </summary>
        public static List<ProtocolStep> G4()
        {
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("static", "HOLD STILL", "On A, grid on the board: does it stay put?", 30, speech: "Look at the board and hold still"),
                new ProtocolStep("motion", "MOVE HEAD SLOWLY", "Small nods and turns, keep the board in view", 30, speech: "Move your head slowly"),
            };
            AddTranslation(s, "trans_1", "Translation");
            s.Add(new ProtocolStep("look_away", "LOOK AWAY, THEN BACK", "Turn 90 degrees away, then back to the board", 20, speech: "Look away, then back to the board"));
            s.Add(new ProtocolStep("walk", "WALK AROUND SLOWLY", "A slow loop around the room, end on A facing the board", 40, speech: "Walk around slowly, then come back to A"));
            s.Add(new ProtocolStep("static_2", "HOLD STILL ON A", "Look at the board", 30, speech: "On A, hold still"));
            s.Add(new ProtocolStep("motion_2", "MOVE HEAD SLOWLY", "Small nods and turns, keep the board in view", 30, speech: "Move your head slowly"));
            s.Add(new ProtocolStep("sustain", "HOLD STILL, RELAX", "Long hold: is the Surface keeping up?", 60, speech: "Hold still and relax"));
            s.Add(new ProtocolStep("look_around", "LOOK AROUND SLOWLY", "Slowly scan the room, end on the board", 30, speech: "Look around slowly, end on the board"));
            return s;
        }

        /// <summary>
        /// Eye calibration capture, camera only: the operator wears the glasses, keeps the head still
        /// and moves the 25 mm ChArUco board so that it fills each zone of the camera image in turn
        /// (the large preview on the Surface screen shows the zone). Snapshots every 0.5 s.
        /// </summary>
        public static List<ProtocolStep> Calib()
        {
            const float t = 8f;
            const string hold = "Move the board there, then hold it still for a moment";
            var s = new List<ProtocolStep>
            {
                new ProtocolStep("centre_far", "BOARD IN THE CENTRE, ARM'S LENGTH", hold, t, speech: "Board in the centre, at arm's length", zone: Z(0.3f, 0.25f, 0.4f, 0.5f)),
                new ProtocolStep("centre_near", "BOARD CLOSE, ABOUT 25 CM", "Fill most of the picture", t, speech: "Board close, filling the picture", zone: Z(0.1f, 0.1f, 0.8f, 0.8f)),
                new ProtocolStep("top_left", "TOP LEFT", hold, t, speech: "Top left", zone: Z(0f, 0f, 0.4f, 0.45f)),
                new ProtocolStep("top_right", "TOP RIGHT", hold, t, speech: "Top right", zone: Z(0.6f, 0f, 0.4f, 0.45f)),
                new ProtocolStep("bottom_right", "BOTTOM RIGHT", hold, t, speech: "Bottom right", zone: Z(0.6f, 0.55f, 0.4f, 0.45f)),
                new ProtocolStep("bottom_left", "BOTTOM LEFT", hold, t, speech: "Bottom left", zone: Z(0f, 0.55f, 0.4f, 0.45f)),
                new ProtocolStep("left", "LEFT EDGE", hold, t, speech: "Left edge", zone: Z(0f, 0.25f, 0.35f, 0.5f)),
                new ProtocolStep("right", "RIGHT EDGE", hold, t, speech: "Right edge", zone: Z(0.65f, 0.25f, 0.35f, 0.5f)),
                new ProtocolStep("top", "TOP EDGE", hold, t, speech: "Top edge", zone: Z(0.3f, 0f, 0.4f, 0.4f)),
                new ProtocolStep("bottom", "BOTTOM EDGE", hold, t, speech: "Bottom edge", zone: Z(0.3f, 0.6f, 0.4f, 0.4f)),
                new ProtocolStep("tilt_lr", "CENTRE, TILT LEFT AND RIGHT", "Turn the board about 30 degrees each way, slowly", t, speech: "Centre, tilt left and right", zone: Z(0.25f, 0.2f, 0.5f, 0.6f)),
                new ProtocolStep("tilt_ud", "CENTRE, TILT UP AND DOWN", "Tip the board about 30 degrees each way, slowly", t, speech: "Centre, tilt up and down", zone: Z(0.25f, 0.2f, 0.5f, 0.6f)),
                new ProtocolStep("sweep", "SWEEP SLOWLY EVERYWHERE", "Slow loops through the whole picture, corners included", 20, speech: "Sweep slowly through the whole picture"),
            };
            return s;
        }

        static Rect Z(float x, float y, float w, float h) => new Rect(x, y, w, h);

        /// <summary>
        /// G5 locked grid, about 4 min: the target (plate A, plate L or the 25 mm board) stands about
        /// 70 cm in front of A. The overlay locks on the first good camera pose, then follows 6DoF and
        /// is corrected by every new pose. During the frozen steps corrections are logged, not applied,
        /// which shows how far 6DoF alone drifts from the camera.
        /// </summary>
        public static List<ProtocolStep> G5()
        {
            return new List<ProtocolStep>
            {
                new ProtocolStep("acquire", "LOOK AT THE TARGET, HOLD STILL", "From A, about 70 cm away, until LOCKED", 20, speech: "Look at the target and hold still"),
                new ProtocolStep("static", "HOLD STILL", "Do the points sit on the printed crosses?", 30, speech: "Hold still. Check the points on the crosses"),
                new ProtocolStep("move_head", "MOVE HEAD SLOWLY", "Small nods and turns, target in view", 20, speech: "Move your head slowly"),
                new ProtocolStep("side_left", "STEP LEFT, LOOK AT THE TARGET", "One step left, view it from the side", 15, speech: "Step left and look at the target"),
                new ProtocolStep("side_right", "STEP RIGHT, LOOK AT THE TARGET", "Two steps right, view it from the other side", 15, speech: "Step right and look at the target"),
                new ProtocolStep("closer", "COME CLOSER, ABOUT 40 CM", "Look at the target from close up", 15, speech: "Come closer"),
                new ProtocolStep("back_to_a", "BACK TO A, HOLD STILL", "Target in view", 15, speech: "Back to A and hold still"),
                new ProtocolStep("frozen_walk", "FROZEN: WALK AROUND, COME BACK", "Corrections paused; end on A facing the target", 40, StepAction.AnchorFreeze,
                                 "Lock frozen. Walk around and come back to A"),
                new ProtocolStep("frozen_check", "FROZEN: HOLD STILL ON A", "Has the overlay drifted off the target?", 20, speech: "Hold still. Has it drifted?"),
                new ProtocolStep("live", "LIVE AGAIN: HOLD STILL", "Corrections back on", 30, StepAction.AnchorLive, "Corrections back on. Hold still"),
            };
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
