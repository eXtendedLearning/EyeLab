using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using EyeLab.Aruco;
using Unity.XR.XREAL;
using UnityEngine;
using UnityEngine.XR;

namespace EyeLab.Probes
{
    /// <summary>
    /// ArUco detection and the board pose on the Beam Pro's CPU with the native plugin, while the glasses draw.
    /// Two variants share this probe:
    /// <list type="bullet">
    /// <item><b>detect</b> (T6.6, gate B3a): full-frame detection at 1-8 OpenCV threads, plate A.</item>
    /// <item><b>budget</b> (T6.7): the same pipeline within a budget. Per cue step a mode (full frame every Eye frame,
    /// full frame capped at 15 Hz, or a region of interest around the target's last box), a thread count, the worker's
    /// nice value and a target (plate A, plate L, the ChArUco board's markers). Each pose is marked trusted or not by
    /// PoseLock's planar-ambiguity rule (thresholds from set.json).</item>
    /// <item><b>lean</b> (T6.8): region detection only, at every Eye frame or capped at 15 Hz, on the ChArUco board and
    /// plate L tilted so both faces show; trusted also needs PoseLock's RMS ceiling (a non-planar solve can diverge).</item>
    /// <item><b>overlay</b> (T6.9, gate B3b): the region as in T6.8, and each trusted pose drawn in the glasses, room-fixed
    /// (<see cref="PoseOverlay"/>): plate L tilted, then the board. Steps with detection off hold the drawing by head
    /// tracking alone and measure the render floor without the worker. The wearer rates the drawing on the Beam Pro
    /// screen. A board pose with at least <see cref="TrustBoardMarkers"/> markers is trusted (T6.8: the ratio rule
    /// trusted 2 % of the board while 1 % was wrong).</item>
    /// <item><b>align</b> (T6.10): the overlay with the wearer's view offset (<see cref="ViewOffset"/>) between the head and
    /// the camera, set by eye with buttons on the Beam Pro and saved for the next run; a one-eye check before it, checks
    /// at two distances after it, and the flat drawing against the solid one.</item>
    /// <item><b>spaam</b> (T6.11): no drawing; a crosshair fixed in one eye's display (<see cref="SpaamCrosshair"/>) that
    /// the wearer lines up with the centre of the board's middle square and taps, at three distances, four display
    /// positions per eye. The taps go to <c>spaam.csv</c>; the per-eye model is fitted offline.</item>
    /// <item><b>eyecal</b> (T6.12): the T6.11 taps, the eye model fitted on the device right after them (this round, and
    /// pooled with every earlier round in <c>calibration/spaam_taps.csv</c>), its leave-one-out error on the screen, then
    /// the solid overlay drawn per eye with the model, switchable live between the SDK, this round and all rounds, rated
    /// at 60, 35 and 100 cm and on plate L. <b>eyecheck</b>: the check alone, with the saved taps.</item>
    /// </list>
    /// Each run has two parts:
    /// <list type="number">
    /// <item><b>Bench</b> (no wearing needed): every frame of the set the agent pushed to <c>files/b3a/</c> (<c>set.json</c>,
    /// PGM frames made by <c>python -m xreal_gates b3a-set</c>) is processed 3 times per bench mode on a worker thread.
    /// <c>bench.csv</c> and <c>bench_corners.csv</c> go to the parity test against the Python pipeline.</item>
    /// <item><b>Live</b>: every new Eye frame is copied once on the main thread, rows reversed to upright (the Eye stores
    /// them bottom-up), with the native head pose at its <c>cam_ts</c>, and processed on a worker thread (newest wins; a
    /// frame that arrives while the worker is busy replaces the waiting one and is counted as skipped).
    /// <c>live.csv</c> has the times, the region and the pose per processed frame, <c>head.csv</c> the head pose, and
    /// <c>frames.csv</c> every Unity frame's time; every 10th processed frame is saved for the parity test.</item>
    /// </list>
    /// All worker timestamps are CLOCK_MONOTONIC, the HMD clock of <c>cam_ts</c> (T6.0 BS), so the age of a pose when
    /// it is ready is <c>end_mono_ns − cam_ts</c>.
    /// </summary>
    public sealed class DetectProbe : MonoBehaviour
    {
        [Serializable]
        public sealed class Target
        {
            public string name;
            public int[] ids;
            public float[] corners;
        }

        [Serializable]
        public sealed class PoseLockSettings
        {
            public float min_ambiguity_ratio = 3f;          // python/pose_lock.py PoseLockConfig, written into set.json
            public float ambiguity_min_separation_deg = 10f;
            public float max_acquire_reproj_rms_px = 15f;    // T6.8: PoseLock's ceiling for any candidate
        }

        [Serializable]
        public sealed class FrameSet
        {
            public string schema;
            public string board_name;
            public string calibration;
            public int dictionary;
            public float[] tuning;
            public double[] K;
            public double[] dist;
            public int[] ids;
            public float[] corners;
            public string[] frames;
            public Target[] targets;           // T6.7
            public PoseLockSettings pose_lock;  // T6.7
        }

        /// <summary>How a cue step processes frames.</summary>
        public readonly struct Plan
        {
            public readonly string Mode;    // full | cap | roi | roi15 (region, capped at CapHz)
            public readonly int Threads, Nice;
            public readonly string Target;  // a set.json target name; "" = the set's own board (plate A)
            public Plan(string mode, int threads, int nice, string target) { Mode = mode; Threads = threads; Nice = nice; Target = target; }
        }

        public static string SetDir => Path.Combine(Application.persistentDataPath, "b3a");

        const string LiveFile = "live.csv";
        const string ResultHeader = "markers,rejected,board_markers,pose_ok,planar,points,clahe_ms,detect_ms,pose_ms,total_ms," +
                                    "r0,r1,r2,t0,t1,t2,rms_px,amb_ratio,amb_deg,roi_x,roi_y,roi_w,roi_h,solution,prior_used";
        const string LiveHeader = "seq,step,threads,cam_ts,arrival_mono_ns,start_mono_ns,end_mono_ns,skipped," + ResultHeader +
                                  ",ids,saved,mode,target,nice,accepted,pr0,pr1,pr2,cap_skipped";
        const string CornerHeader = "seq,id,x0,y0,x1,y1,x2,y2,x3,y3";
        const string BenchHeader = "frame,threads_req,threads,rep," + ResultHeader + ",mode";
        const string FramesFile = "frames.csv";
        const string FramesHeader = "step,threads,unity_frame,dt_ms,present,mono_ns,eye_ms";
        const string HeadFile = "head.csv";
        const string HeadHeader = "seq,cam_ts,ok,m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15";
        const int SaveEvery = 10;
        const int SaveEveryCalib = 4;   // T6.13: about 5 frames a second for the calibration
        const int BenchReps = 3;
        const double CapHz = 15.0;
        const double RoiMaxAgeNs = 0.5e9;     // a box or prior older than this (by cam_ts) is not used
        const float RoiMarginFrac = 0.5f;     // ROI = projected board box grown by this share of its larger side ...
        const float RoiMarginMinPx = 48f;     // ... and at least this many pixels on each side
        const int NiceRoi = 10;
        public const int TrustBoardMarkers = 12;   // T6.9: a planar pose on this many markers or more is trusted

        // ------------------------------------------------------------------ T6.6 (detect)
        static readonly int[] DetectBenchThreads = { 1, 2, 4, 8 };

        public static List<CueStep> DetectCues() => new List<CueStep>
        {
            new CueStep("ready", "Detection test. Stand about half a metre from plate A, so it is straight ahead of you.", 8f,
                        "Plate A straight ahead, about 50 cm", ticks: true),
            new CueStep("t2_still", "Look at the plate and hold still.", 15f, "Look at the plate: HOLD STILL"),
            new CueStep("t2_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("t2_away", "Look away from the plate.", 6f, "Look AWAY from the plate"),
            new CueStep("t4_still", "Back to the plate. Hold still.", 15f, "Back to the plate: HOLD STILL"),
            new CueStep("t4_move", "Move your head slowly again.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("t4_away", "Look away again.", 6f, "Look AWAY from the plate", ticks: true),
        };

        /// <summary>OpenCV threads for a T6.6 cue step: the number after "t" in "t2_still"; 2 (OpenCV's Android default) otherwise.</summary>
        public static int ThreadsFor(string stepId)
        {
            if (stepId != null && stepId.Length > 2 && stepId[0] == 't')
            {
                int end = stepId.IndexOf('_');
                if (end > 1 && int.TryParse(stepId.Substring(1, end - 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0)
                    return n;
            }
            return 2;
        }

        // ------------------------------------------------------------------ T6.7 (budget)
        // Steps share a prefix per target: a_ plate A, l_ plate L, b_ the ChArUco board. The still phase on plate A runs
        // full, then cap, then roi on the same scene, so the three modes are compared without moving.
        public static List<CueStep> BudgetCues() => new List<CueStep>
        {
            new CueStep("a_ready", "Detection budget test. Plate A straight ahead, about sixty centimetres.", 8f,
                        "Plate A straight ahead, about 60 cm", ticks: true),
            new CueStep("a_full_still", "Look at plate A and hold still.", 10f, "Plate A: HOLD STILL"),
            new CueStep("a_cap_still", "Keep holding still.", 10f, "Keep HOLDING STILL"),
            new CueStep("a_roi_still", "Keep holding still.", 12f, "Keep HOLDING STILL", ticks: true),
            new CueStep("a_roi_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("a_roi_away", "Look away from the plate.", 5f, "Look AWAY"),
            new CueStep("a_roi_back", "Back to the plate. Hold still.", 8f, "Back to plate A: HOLD STILL", ticks: true),
            new CueStep("a_full_move", "Move your head slowly again.", 10f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("l_ready", "Now plate L. Put it where plate A was, with both faces towards you.", 12f,
                        "Swap to plate L, both faces towards you", ticks: true),
            new CueStep("l_roi_still", "Look at plate L and hold still.", 12f, "Plate L: HOLD STILL", ticks: true),
            new CueStep("l_roi_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("b_ready", "Now the ChArUco board, upright, about sixty centimetres away.", 12f,
                        "Swap to the ChArUco board, upright, 60 cm", ticks: true),
            new CueStep("b_roi_still", "Look at the board and hold still.", 12f, "Board: HOLD STILL", ticks: true),
            new CueStep("b_roi_move", "Move your head slowly. Keep the board in view.", 12f, "Move SLOWLY, board in view", metronome: 1f),
        };

        // ------------------------------------------------------------------ T6.8 (lean)
        // T6.7 run 1: the region at 1 thread was the only mode near the render margin (5-7 % late against 12-18 %);
        // plate A at 75 cm was too small for a stable normal. Board first, then plate L tilted about 45 degrees so
        // both faces show (standing on the table, its upward face was edge-on). Each target: region at every Eye
        // frame, then capped at 15 Hz, still and moving.
        public static List<CueStep> LeanCues() => new List<CueStep>
        {
            new CueStep("b_ready", "Lean detection test. The ChArUco board upright, about sixty centimetres away.", 10f,
                        "ChArUco board upright, about 60 cm", ticks: true),
            new CueStep("b_roi_still", "Look at the board and hold still.", 12f, "Board: HOLD STILL", ticks: true),
            new CueStep("b_r15_still", "Keep holding still.", 12f, "Keep HOLDING STILL", ticks: true),
            new CueStep("b_roi_move", "Move your head slowly. Keep the board in view.", 12f, "Move SLOWLY, board in view", metronome: 1f),
            new CueStep("b_r15_move", "Keep moving slowly.", 12f, "Keep moving SLOWLY", metronome: 1f),
            new CueStep("l_ready", "Now plate L, where the board was. Tilt it about forty five degrees towards you, so you see both faces.", 14f,
                        "Plate L tilted 45 degrees, both faces visible", ticks: true),
            new CueStep("l_roi_still", "Look at plate L and hold still.", 12f, "Plate L: HOLD STILL", ticks: true),
            new CueStep("l_r15_still", "Keep holding still.", 12f, "Keep HOLDING STILL", ticks: true),
            new CueStep("l_roi_move", "Move your head slowly. Keep the plate in view.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("l_r15_move", "Keep moving slowly.", 12f, "Keep moving SLOWLY", metronome: 1f),
        };

        // ------------------------------------------------------------------ T6.9 (overlay, B3b)
        // T6.8 run 1: the region at every Eye frame, 1 thread, nice 10 (27.6 poses/s, late frames at the idle floor);
        // plate L tilted 45 degrees is the target with no wrong trusted pose. Each phase is rated on the screen.
        public static readonly string[] OverlayAnswers = { "on the markers", "a few mm off", "clearly off", "swims when I move", "not shown" };

        public static List<CueStep> OverlayCues() => new List<CueStep>
        {
            new CueStep("l_ready", "Overlay test. Plate L about sixty centimetres away, tilted about forty five degrees towards you. " +
                                   "Green squares should appear on its markers.", 14f, "Plate L tilted 45 degrees, about 60 cm", ticks: true),
            new CueStep("l_still", "Hold still. Look at where the green squares sit on the markers.", 12f, "Plate L: HOLD STILL, look at the squares"),
            new CueStep("l_still_rate", "Rate it on the screen.", 6f, "RATE on the screen", ticks: true),
            new CueStep("l_move", "Move your head slowly. Keep the plate in view, and watch the squares.", 12f, "Move SLOWLY, plate in view", metronome: 1f),
            new CueStep("l_move_rate", "Rate it on the screen.", 6f, "RATE on the screen", ticks: true),
            new CueStep("l_off_still", "Detection is off now. The squares turn orange and stay by head tracking only. Hold still.", 10f,
                        "Detection OFF: HOLD STILL"),
            new CueStep("l_off_move", "Move your head slowly. Do the orange squares stay on the markers?", 12f, "Detection OFF: move SLOWLY", metronome: 1f),
            new CueStep("l_off_rate", "Rate it on the screen.", 6f, "RATE on the screen", ticks: true),
            new CueStep("b_ready", "Now the ChArUco board, upright, where the plate was.", 12f, "ChArUco board upright, about 60 cm", ticks: true),
            new CueStep("b_still", "Hold still and look at the squares on the board.", 12f, "Board: HOLD STILL, look at the squares"),
            new CueStep("b_still_rate", "Rate it on the screen.", 6f, "RATE on the screen", ticks: true),
            new CueStep("b_move", "Move your head slowly. Keep the board in view.", 12f, "Move SLOWLY, board in view", metronome: 1f),
            new CueStep("b_move_rate", "Rate it on the screen.", 6f, "RATE on the screen", ticks: true),
        };

        // ------------------------------------------------------------------ T6.10 (align)
        // T6.9 run 1: room-fixed to a few mm, but "clearly off" the markers by a constant offset, with the camera side
        // verified (intrinsics, rgb_from_head). The offset is the display's. First the one-eye check (stereo against
        // boresight), then the alignment by eye, then two distances (an angle grows with distance, a shift does not),
        // then flat against solid on plate L.
        public static readonly string[] StyleAnswers = { "solid helps", "flat better", "about the same", "solid confusing" };
        public const string AlignStep = "b_align";

        public static List<CueStep> AlignCues() => new List<CueStep>
        {
            new CueStep("b_ready", "Align test. The ChArUco board upright, about fifty centimetres away. Green squares should appear.", 12f,
                        "ChArUco board upright, about 50 cm", ticks: true),
            new CueStep("b_both", "Both eyes open. Look at the squares and rate them on the screen.", 10f, "BOTH eyes: rate on the screen"),
            new CueStep("b_left_only", "Close your right eye, so only your left eye sees. Rate on the screen.", 10f, "LEFT eye only: rate"),
            new CueStep("b_right_only", "Now close your left eye, so only your right eye sees. Rate on the screen.", 10f, "RIGHT eye only: rate", ticks: true),
            new CueStep(AlignStep, "Both eyes open. Align the squares on the markers with the buttons on the screen. " +
                                   "Turn first; move only if turning cannot fix it. Press Save, then Done.", 150f,
                        "ALIGN with the buttons, then SAVE and DONE", ticks: true),
            new CueStep("b_check_still", "Hold still and rate the squares.", 10f, "Aligned: HOLD STILL, rate"),
            new CueStep("b_check_move", "Move your head slowly, then rate.", 12f, "Move SLOWLY, then rate", metronome: 1f),
            new CueStep("b_near_ready", "Bring the board closer, about thirty five centimetres.", 10f, "Board CLOSER, about 35 cm", ticks: true),
            new CueStep("b_near", "Hold still and rate.", 10f, "Near: HOLD STILL, rate"),
            new CueStep("b_far_ready", "Now about one metre away: move the board or step back.", 12f, "Board FAR, about 1 m", ticks: true),
            new CueStep("b_far", "Hold still and rate.", 10f, "Far: HOLD STILL, rate"),
            new CueStep("l_ready", "Now plate L, tilted forty five degrees, about sixty centimetres away.", 12f,
                        "Plate L tilted 45 degrees, about 60 cm", ticks: true),
            new CueStep("l_flat", "Flat squares. Look at them and move your head a little.", 10f, "FLAT: look, move a little", metronome: 1f),
            new CueStep("l_solid", "Now the solid style, with bars and posts. Look and move your head a little.", 12f,
                        "SOLID: look, move a little", metronome: 1f),
            new CueStep("l_style", "Which shows better where the plate is in space? Answer on the screen.", 8f, "Flat or solid? Answer on the screen"),
        };

        // ------------------------------------------------------------------ T6.11 (spaam)
        // T6.10 run 1: a rigid offset cannot align at arm's length; the error grows as the target comes nearer, and each
        // eye's drawing is off the opposite way (right eye: left of the marker; left eye: right of it). That is the eye
        // position against the SDK's, which SPAAM measures one eye at a time. The real point is the centre of the board's
        // middle square (marker 8, 62.5 / 87.5 mm), the board's pose being the most reliable one.
        public static readonly string[] EyeCheckAnswers = { "left eye only", "right eye only", "both eyes", "not shown" };
        public static readonly int[] SpaamDepthsCm = { 60, 35, 100 };
        public const int SpaamPerEye = 4;
        const float SpaamTapTimeout = 40f;

        public static List<CueStep> SpaamCues()
        {
            List<CueStep> c = TapCues();
            c.Add(new CueStep("s_done", "Calibration taps done.", 3f, "Done"));
            return c;
        }

        static List<CueStep> TapCues()
        {
            var c = new List<CueStep>
            {
                new CueStep("s_ready", "Calibration. The ChArUco board upright on the table, about sixty centimetres away. You will put a " +
                                       "green crosshair on the centre of the board's middle square, one eye at a time, and tap the screen.", 14f,
                            "ChArUco board upright, about 60 cm", ticks: true),
                new CueStep("chk_L", "A green crosshair should be in your left eye only. Close each eye in turn, then answer on the screen.", 12f,
                            "Which eye sees the crosshair? Answer on the screen"),
            };
            foreach (int cm in SpaamDepthsCm)
            {
                string words = cm == 60 ? "sixty centimetres" : cm == 35 ? "thirty five centimetres" : "one metre";
                c.Add(new CueStep($"d{cm}_place", $"Stand so the board is about {words} from your eyes.", 10f, $"Board at about {cm} cm", ticks: true));
                foreach (string eye in new[] { "L", "R" })
                {
                    string name = eye == "L" ? "Left" : "Right", other = eye == "L" ? "right" : "left";
                    for (int k = 1; k <= SpaamPerEye; k++)
                    {
                        string say = k == 1
                            ? $"{name} eye. Close your {other} eye. Move your head until the crosshair centre is on the centre of the middle square, then tap."
                            : "Next.";
                        c.Add(new CueStep($"d{cm}_{eye}_{k}", say, SpaamTapTimeout,
                                          $"{name.ToUpperInvariant()} eye: crosshair on the middle square, TAP ({k}/{SpaamPerEye})"));
                    }
                }
            }
            return c;
        }

        // ------------------------------------------------------------------ T6.13 (camcal)
        /// <summary>T6.13: the Eye's intrinsics on the Beam Pro. The glasses lie on the table facing the wearer, who moves
        /// the board through the camera's view (every part of the picture, tilted each way, near and far); then the tape
        /// check: the camera lens at 50.0 and 100.0 cm from the board, set with a tape, NEXT on the screen when set.</summary>
        public static List<CueStep> CamcalCues()
        {
            var c = new List<CueStep>
            {
                new CueStep("s_ready", "Camera calibration. Put the glasses on the table, lenses facing you, and keep them still. Hold the ChArUco board " +
                                       "in both hands about forty centimetres in front of them.", 14f, "Glasses on the table, board in your hands", ticks: true),
                new CueStep("m_centre", "Board in the middle of the camera's view, facing it. Move it slowly.", 8f, "Board in the MIDDLE, facing the camera"),
            };
            (string id, string say, string show)[] moves =
            {
                ("m_tl", "Move it to the top corner on your right, and tilt it a little.", "Top corner on YOUR RIGHT"),
                ("m_tr", "Now the top corner on your left.", "Top corner on YOUR LEFT"),
                ("m_bl", "Bottom corner on your left.", "Bottom corner on YOUR LEFT"),
                ("m_br", "Bottom corner on your right.", "Bottom corner on YOUR RIGHT"),
                ("m_edges", "Slowly along all four edges of the view.", "Along all four EDGES"),
                ("m_tilt_v", "Back to the middle. Tilt the top towards the camera, then away, about forty five degrees.", "TILT top towards, then away (45 deg)"),
                ("m_tilt_h", "Now turn it left and right, about forty five degrees.", "TURN left, then right (45 deg)"),
                ("m_near", "Closer, about twenty five centimetres, moving it around.", "CLOSER (25 cm), move around"),
                ("m_far", "Farther, about seventy centimetres, tilted, moving around.", "FARTHER (70 cm), tilted, move around"),
            };
            foreach (var m in moves) c.Add(new CueStep(m.id, m.say, 10f, m.show, ticks: true));
            c.Add(new CueStep("t50_place", "Distance check. Stand the board upright on the table. With a tape, put the glasses so the camera lens " +
                                           "is exactly fifty centimetres from the board, facing it. Then tap NEXT.", 90f, "Lens EXACTLY 50.0 cm from the board, then NEXT"));
            c.Add(new CueStep("t50", "Keep still.", 10f, "Keep still (50 cm)"));
            c.Add(new CueStep("t100_place", "Now exactly one metre, the same way. Then tap NEXT.", 90f, "Lens EXACTLY 100.0 cm from the board, then NEXT"));
            c.Add(new CueStep("t100", "Keep still.", 10f, "Keep still (100 cm)"));
            return c;
        }

        // ------------------------------------------------------------------ T6.12 (eyecal, eyecheck)
        public const string FitStep = "fit";
        public static readonly string[] EyeModes = { "sdk", "round", "all" };

        static List<CueStep> CheckCues() => new List<CueStep>
        {
            new CueStep("c60_place", "Board at about sixty centimetres. The squares now use your eye calibration. On the screen you can " +
                                     "switch between S D K, this round, and all rounds.", 12f, "Board at about 60 cm", ticks: true),
            new CueStep("c60", "Try each button, look at the squares with both eyes, and rate each one.", 25f, "60 cm: switch, look, RATE each"),
            new CueStep("c35_place", "Bring the board closer, about thirty five centimetres.", 10f, "Board CLOSER, about 35 cm", ticks: true),
            new CueStep("c35", "Try each button and rate each one.", 25f, "35 cm: switch, look, RATE each"),
            new CueStep("c100_place", "Now about one metre away.", 12f, "Board FAR, about 1 m", ticks: true),
            new CueStep("c100", "Try each button and rate each one.", 25f, "1 m: switch, look, RATE each"),
            new CueStep("l_ready", "Now plate L, tilted forty five degrees, about sixty centimetres away.", 12f,
                        "Plate L tilted 45 degrees, about 60 cm", ticks: true),
            new CueStep("l_check", "Try each button, move your head a little, and rate each one.", 25f, "Plate L: switch, look, RATE each", metronome: 1f),
        };

        public static List<CueStep> EyecalCues()
        {
            List<CueStep> c = TapCues();
            c.Add(new CueStep(FitStep, "Fitting your eyes. The result is on the screen.", 10f, "Fitting: see the Beam Pro screen"));
            c.AddRange(CheckCues());
            return c;
        }

        public static List<CueStep> EyecheckCues()
        {
            var c = new List<CueStep>
            {
                new CueStep("s_ready", "Calibration check. The ChArUco board upright on the table. The squares use the saved eye calibration.", 10f,
                            "ChArUco board upright", ticks: true),
            };
            c.AddRange(CheckCues());
            return c;
        }

        static bool CheckStep(string id) => id != null && (id.StartsWith("l_") || (id.Length > 1 && id[0] == 'c' && char.IsDigit(id[1])));

        /// <summary>A tap step's eye and crosshair position: "d60_L_3" gives ("L", position 3); chk_L the left eye's centre.</summary>
        public static bool SpaamStep(string stepId, out string eye, out Vector2 deg)
        {
            eye = "";
            deg = Vector2.zero;
            if (stepId == "chk_L") { eye = "L"; return true; }
            string[] parts = (stepId ?? "").Split('_');
            if (parts.Length != 3 || parts[0].Length < 2 || parts[0][0] != 'd' || (parts[1] != "L" && parts[1] != "R")) return false;
            if (!int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int k) || k < 1 || k > SpaamCrosshair.Positions.Length) return false;
            eye = parts[1];
            deg = SpaamCrosshair.Positions[k - 1];
            return true;
        }

        /// <summary>The plan of a step: T6.6 steps by their thread count; T6.7 to T6.11 steps by their prefix and mode.</summary>
        public static Plan PlanFor(string variant, string stepId)
        {
            stepId = stepId ?? "";
            if (variant == "spaam") return new Plan("roi", 1, NiceRoi, "board25");
            if (variant == "camcal") return new Plan("full", 2, 0, "board25");   // full frames: the board goes to the edges, tilted
            if (variant == "eyecal" || variant == "eyecheck")
                return new Plan("roi", 1, NiceRoi, stepId.StartsWith("l_") ? "plate_L" : "board25");
            if (variant == "align")
                return new Plan("roi", 1, NiceRoi, stepId.StartsWith("l_") ? "plate_L" : "board25");
            if (variant == "overlay")
            {
                string t = stepId.StartsWith("b_") ? "board25" : "plate_L";
                return new Plan(stepId.Contains("_off_") ? "off" : "roi", 1, NiceRoi, t);
            }
            if (variant == "lean")
            {
                string t = stepId.StartsWith("l_") ? "plate_L" : "board25";
                return new Plan(stepId.Contains("_r15_") ? "roi15" : "roi", 1, NiceRoi, t);
            }
            if (variant != "budget") return new Plan("full", ThreadsFor(stepId), 0, "");
            string target = stepId.StartsWith("l_") ? "plate_L" : stepId.StartsWith("b_") ? "board25" : "plate_A";
            if (stepId.Contains("_full_")) return new Plan("full", 2, 0, target);
            if (stepId.Contains("_cap_")) return new Plan("cap", 2, 0, target);
            if (stepId == "a_ready") return new Plan("full", 2, 0, target);
            return new Plan("roi", 1, NiceRoi, target);   // roi steps and the l_/b_ ready steps (the target is being placed)
        }

        /// <summary>The region processed next: the projected board box grown by the margin, clipped to the frame.</summary>
        public static int[] RoiAround(double x0, double y0, double x1, double y1, int w, int h)
        {
            if (double.IsNaN(x0) || double.IsNaN(y0) || double.IsNaN(x1) || double.IsNaN(y1)) return null;
            float m = Mathf.Max(RoiMarginMinPx, RoiMarginFrac * (float)Math.Max(x1 - x0, y1 - y0));
            int ax = Mathf.Clamp(Mathf.FloorToInt((float)x0 - m), 0, w);
            int ay = Mathf.Clamp(Mathf.FloorToInt((float)y0 - m), 0, h);
            int bx = Mathf.Clamp(Mathf.CeilToInt((float)x1 + m), 0, w);
            int by = Mathf.Clamp(Mathf.CeilToInt((float)y1 + m), 0, h);
            if (bx - ax < 32 || by - ay < 32) return null;
            return new[] { ax, ay, bx - ax, by - ay };
        }

        struct Done
        {
            public long Seq, Skipped, CapSkipped, Arrival, Start, End;
            public ulong CamTs;
            public int Threads, Nice;
            public bool Accepted;
            public string Step, Ids, Mode, Target;
            public ArucoNative.Result R;
            public double[] Prior;
            public string Corners;
            public byte[] SaveCopy;
            public int W, H, Stride;
            public bool HeadOk;
            public float[] Head;
        }

        string m_Variant = "detect";
        ProbeRun m_Run;
        FrameSet m_Set;
        XRDisplaySubsystem m_Display;
        string m_Step = "";
        volatile int m_ThreadsWanted = 2;
        volatile bool m_Live, m_Stop;
        Thread m_Worker;
        readonly object m_Lock = new object();
        readonly AutoResetEvent m_Signal = new AutoResetEvent(false);
        readonly ConcurrentQueue<Done> m_Done = new ConcurrentQueue<Done>();
        // frame hand-over: main writes m_Back, swaps it with m_Front under the lock; the worker swaps m_Front with m_Work
        byte[] m_Back, m_Front, m_Work;
        float[] m_HeadBack = new float[16], m_HeadFront = new float[16];
        bool m_HasNew, m_HeadBackOk, m_HeadFrontOk;
        long m_FrontSeq, m_FrontArrival, m_Skipped, m_CapSkipped, m_Seq, m_Processed, m_EyeFrames;
        ulong m_FrontTs, m_LastTs;
        int m_W, m_H, m_Stride;
        string m_FrontStep = "";
        bool m_Capturing;
        string m_BenchStatus = "", m_Error = "";
        double m_LastTotal;
        double m_EyeMs;
        string m_LastMode = "";
        // T6.9 overlay
        PoseOverlay m_Overlay;
        Pose m_RgbFromHead = new Pose(Vector3.zero, Quaternion.identity);
        bool m_HasRgbFromHead;
        long m_OffSkipped;
        static DetectProbe s_Active;
        const string OverlayFile = "overlay.csv";
        const string OverlayHeader = "seq,step,target,cam_ts,applied_mono_ns,unity_frame,px,py,pz,qx,qy,qz,qw," +
                                     "head_px,head_py,head_pz,head_qx,head_qy,head_qz,head_qw," +
                                     "off_tx_mm,off_ty_mm,off_tz_mm,off_qx,off_qy,off_qz,off_qw,style";
        // T6.10 view offset; null for T6.9 (no offset)
        ViewOffset m_Offset;
        bool m_OffsetSaved;
        bool m_HasLast;
        string m_LastTarget = "";
        Vector3 m_LastHeadP, m_LastInP;
        Quaternion m_LastHeadQ = Quaternion.identity, m_LastInQ = Quaternion.identity;

        /// <summary>The running probe's step and variant ("" when none runs), for the Beam Pro screen.</summary>
        public static string ActiveStep => s_Active != null ? s_Active.m_Step : "";
        public static string ActiveVariant => s_Active != null ? s_Active.m_Variant : "";
        public static string OffsetText => s_Active?.m_Offset != null
            ? s_Active.m_Offset.Describe() + (s_Active.m_OffsetSaved ? " (saved)" : "") : "";

        /// <summary>T6.10: turns the drawing about the head by <paramref name="dq"/> (head frame).</summary>
        public static void TurnOffset(Quaternion dq, string label)
        {
            if (s_Active?.m_Offset == null) return;
            s_Active.m_Offset.TurnAboutHead(dq);
            s_Active.OffsetChanged("turn " + label);
        }

        /// <summary>T6.10: shifts the drawing by <paramref name="mm"/> (head frame, millimetres).</summary>
        public static void MoveOffset(Vector3 mm, string label)
        {
            if (s_Active?.m_Offset == null) return;
            s_Active.m_Offset.Move(mm);
            s_Active.OffsetChanged("move " + label);
        }

        public static void ZeroOffset()
        {
            if (s_Active?.m_Offset == null) return;
            s_Active.m_Offset = new ViewOffset();
            s_Active.OffsetChanged("zero");
        }

        public static void SaveOffset()
        {
            DetectProbe p = s_Active;
            if (p?.m_Offset == null) return;
            try
            {
                p.m_Offset.Save(ViewOffset.DefaultPath);
                p.m_OffsetSaved = true;
                CuePlayer.Log(p.m_Run, "offset_saved", p.m_Step + ":" + p.m_Offset.Csv().Replace(',', ' '));
                p.m_Run.Meta("view_offset_saved", p.m_Offset.Csv() + "; " + p.m_Offset.Describe());
            }
            catch (Exception e)
            {
                CuePlayer.Log(p.m_Run, "offset_save_error", e.GetType().Name + " " + e.Message);
            }
        }

        // T6.11 spaam
        SpaamCrosshair m_Cross;
        int m_Taps;
        float m_StepStart;
        long m_LastTrustedNs;
        const string SpaamFile = "spaam.csv";
        const string SpaamHeader = "tap,step,eye,deg_x,deg_y,mono_ns,unity_frame,head_px,head_py,head_pz,head_qx,head_qy,head_qz,head_qw,board_age_ms," +
                                   "pe_x,pe_y,pe_z,n_board";
        // T6.12: board centres in the room (tracking space) from trusted board poses, for the device-side tap point
        readonly List<(ulong ts, Vector3 p)> m_BoardCentres = new List<(ulong, Vector3)>();
        readonly List<EyeModel.Tap> m_RoundTaps = new List<EyeModel.Tap>();
        List<EyeModel.Tap> m_Earlier = new List<EyeModel.Tap>();
        double[] m_RoundL, m_RoundR, m_AllL, m_AllR;
        EyeWarp m_Warp;
        string m_Mode = "";
        bool m_OverlayOn = true;
        Vector3 m_BoardCentreCv = new Vector3(0.0625f, 0.0875f, 0f);   // marker 8's centre, board frame (OpenCV), from the set
        Vector3 BoardCentreUnity => new Vector3(m_BoardCentreCv.x, -m_BoardCentreCv.y, m_BoardCentreCv.z);   // F applied

        /// <summary>The fit's result for the Beam Pro screen.</summary>
        public static string CalText { get; private set; } = "";
        public static string ActiveMode => s_Active != null ? s_Active.m_Mode : "";

        /// <summary>Whether <paramref name="mode"/> has a model in the running probe (the screen marks the others).</summary>
        public static bool ModeAvailable(string mode)
        {
            DetectProbe p = s_Active;
            if (p?.m_Warp == null) return false;
            return mode == "sdk" || (mode == "round" ? p.m_RoundL != null && p.m_RoundR != null : p.m_AllL != null && p.m_AllR != null);
        }

        /// <summary>T6.12: which eye model the overlay is drawn with: "sdk", "round" (this run's taps) or "all" (every round).</summary>
        public static void SetMode(string mode)
        {
            DetectProbe p = s_Active;
            if (p?.m_Warp == null) return;
            double[] l = mode == "round" ? p.m_RoundL : mode == "all" ? p.m_AllL : null;
            double[] r = mode == "round" ? p.m_RoundR : mode == "all" ? p.m_AllR : null;
            if (mode != "sdk" && (l == null || r == null)) { CuePlayer.Log(p.m_Run, "eyemode_unavailable", p.m_Step + ":" + mode); return; }
            p.m_Warp.Use(l, r);
            p.m_Mode = mode;
            CuePlayer.Log(p.m_Run, "eyemode", p.m_Step + ":" + mode);
        }

        /// <summary>The device-side tap point: the median board centre of the last 2 s, in the SDK eye's frame.</summary>
        bool TapPoint(string eye, Vector3 headP, Quaternion headQ, out Vector3 pe, out int n)
        {
            pe = Vector3.zero;
            ulong now = (ulong)ProbeClocks.MonoNs;
            var xs = new List<float>();
            var ys = new List<float>();
            var zs = new List<float>();
            foreach ((ulong ts, Vector3 p) in m_BoardCentres)
                if (ts <= now && now - ts <= 2_000_000_000UL) { xs.Add(p.x); ys.Add(p.y); zs.Add(p.z); }
            n = xs.Count;
            if (n < 3 || m_Cross == null) return false;
            xs.Sort(); ys.Sort(); zs.Sort();
            var world = new Vector3(xs[n / 2], ys[n / 2], zs[n / 2]);
            Vector3 inHead = Quaternion.Inverse(headQ) * (world - headP);
            Pose e = m_Cross.EyePose(eye);
            pe = Quaternion.Inverse(e.rotation) * (inHead - e.position);
            return true;
        }

        void Fit()
        {
            var ci = CultureInfo.InvariantCulture;
            List<EyeModel.Tap> Sel(IEnumerable<EyeModel.Tap> taps, string eye) { var l = new List<EyeModel.Tap>(); foreach (var t in taps) if (t.Eye == eye) l.Add(t); return l; }
            var all = new List<EyeModel.Tap>(m_Earlier);
            all.AddRange(m_RoundTaps);
            var sb = new StringBuilder();
            var wearings = new HashSet<string>();
            foreach (var t in all) wearings.Add(t.Wearing);
            if (m_RoundTaps.Count > 0)
            {
                try { EyeModel.AppendTaps(m_RoundTaps); }
                catch (Exception e) { m_Run.Meta("eye_taps_save_error", e.Message); }
            }
            foreach (string eye in new[] { "L", "R" })
            {
                List<EyeModel.Tap> round = Sel(m_RoundTaps, eye), pooled = Sel(all, eye);
                double[] cr = EyeModel.Fit(round), ca = EyeModel.Fit(pooled);
                double lr = EyeModel.LeaveOneOutMm(round), la = EyeModel.LeaveOneOutMm(pooled);
                double sdk = EyeModel.SdkMm(pooled);
                if (eye == "L") { m_RoundL = cr; m_AllL = ca; } else { m_RoundR = cr; m_AllR = ca; }
                string diff = cr != null && ca != null ? $"{EyeModel.DiffMm(cr, ca):F1} mm at 60 cm" : "-";
                sb.Append($"{(eye == "L" ? "Left" : "Right")}: this round {round.Count} taps, fresh-tap error {lr:F1} mm; " +
                          $"all rounds {pooled.Count} taps, {la:F1} mm (SDK {sdk:F1} mm); round vs all {diff}\n");
                m_Run.Meta($"eye_{eye}_round", $"{EyeModel.Mm(cr)}; {round.Count} taps; leave-one-out median {lr.ToString("F2", ci)} mm");
                m_Run.Meta($"eye_{eye}_all", $"{EyeModel.Mm(ca)}; {pooled.Count} taps; leave-one-out median {la.ToString("F2", ci)} mm; SDK median {sdk.ToString("F2", ci)} mm");
            }
            if (m_AllL != null && m_AllR != null)
            {
                var s = new EyeModel.Saved
                {
                    left_taps = Sel(all, "L").Count, right_taps = Sel(all, "R").Count, wearings = wearings.Count,
                    left_loo_mm = (float)EyeModel.LeaveOneOutMm(Sel(all, "L")), right_loo_mm = (float)EyeModel.LeaveOneOutMm(Sel(all, "R")),
                    source = $"T6.12 {m_Run.Dir}: all rounds in eye_taps.csv", camera = CameraName,
                };
                EyeModel.Fill(s, m_AllL, m_AllR);
                try { EyeModel.Save(s); m_Run.Meta("eye_model_saved", EyeModel.ModelPath); }
                catch (Exception e) { m_Run.Meta("eye_model_save_error", e.Message); }
            }
            sb.Append($"{wearings.Count} wearing(s) pooled. Saved for the next run.");
            CalText = sb.ToString();
            m_Run.Meta("eye_fit", CalText.Replace("\n", " | "));
            SetMode(m_AllL != null && m_AllR != null ? "all" : m_RoundL != null && m_RoundR != null ? "round" : "sdk");
        }

        /// <summary>The age of the latest trusted board pose (ms), for the Beam Pro screen; -1 when none.</summary>
        public static double BoardAgeMs => s_Active == null || s_Active.m_LastTrustedNs == 0 ? -1 : (ProbeClocks.MonoNs - s_Active.m_LastTrustedNs) / 1e6;

        /// <summary>T6.11: the crosshair is on the point. Logs the head pose now and goes to the next step.</summary>
        public static void Tap()
        {
            DetectProbe p = s_Active;
            if (p != null && p.m_Variant == "camcal")
            {
                // T6.13: NEXT ends a placement step (the glasses are set at the tape distance)
                if (!p.m_Step.EndsWith("_place") || Time.realtimeSinceStartup - p.m_StepStart < 0.4f) return;
                CuePlayer.Log(p.m_Run, "next", p.m_Step);
                CuePlayer.Ensure().SkipStep();
                return;
            }
            if (p == null || p.m_Cross == null || !p.m_Cross.Visible || p.m_Step == "chk_L") return;
            if (Time.realtimeSinceStartup - p.m_StepStart < 0.4f) return;   // a double tap would land on the next step
            var ci = CultureInfo.InvariantCulture;
            Camera cam = Camera.main;
            Vector3 hp = cam != null ? cam.transform.localPosition : Vector3.zero;
            Quaternion hq = cam != null ? cam.transform.localRotation : Quaternion.identity;
            string f(float v) => v.ToString("R", ci);
            p.m_Taps++;
            bool ok = p.TapPoint(p.m_Cross.Eye, hp, hq, out Vector3 pe, out int nb);
            if (ok && p.m_Variant == "eyecal")
                p.m_RoundTaps.Add(new EyeModel.Tap
                {
                    Wearing = Path.GetFileName(p.m_Run.Dir), Camera = p.CameraName, Eye = p.m_Cross.Eye, Step = p.m_Step, Px = pe.x, Py = pe.y, Pz = pe.z,
                    U = Math.Tan(p.m_Cross.Deg.x * Math.PI / 180), V = Math.Tan(p.m_Cross.Deg.y * Math.PI / 180),
                });
            p.m_Run.Row(SpaamFile, SpaamHeader, string.Join(",", p.m_Taps.ToString(ci), p.m_Step, p.m_Cross.Eye, f(p.m_Cross.Deg.x), f(p.m_Cross.Deg.y),
                ProbeClocks.MonoNs.ToString(ci), Time.frameCount.ToString(ci), f(hp.x), f(hp.y), f(hp.z), f(hq.x), f(hq.y), f(hq.z), f(hq.w),
                BoardAgeMs.ToString("F0", ci), ok ? f(pe.x) : "", ok ? f(pe.y) : "", ok ? f(pe.z) : "", nb.ToString(ci)));
            CuePlayer.Log(p.m_Run, "tap", p.m_Step + ":" + p.m_Taps.ToString(ci));
            CuePlayer.Ensure().SkipStep();
        }

        /// <summary>T6.11: the last tap was not aligned; the fit drops it.</summary>
        public static void BadTap()
        {
            DetectProbe p = s_Active;
            if (p == null || p.m_Taps == 0) return;
            CuePlayer.Log(p.m_Run, "tap_bad", p.m_Step + ":" + p.m_Taps.ToString(CultureInfo.InvariantCulture));
            if (p.m_RoundTaps.Count > 0) p.m_RoundTaps.RemoveAt(p.m_RoundTaps.Count - 1);   // the last tap's point, if it had one
        }

        void SpaamOnStep(string stepId)
        {
            m_StepStart = Time.realtimeSinceStartup;
            if (m_Variant == "eyecal" || m_Variant == "eyecheck")
            {
                m_OverlayOn = CheckStep(stepId);
                if (!m_OverlayOn) m_Overlay?.Hide();
                if (stepId == FitStep) Fit();
            }
            if (m_Cross == null) return;
            if (SpaamStep(stepId, out string eye, out Vector2 deg)) m_Cross.Show(eye, deg);
            else m_Cross.Hide();
        }

        /// <summary>T6.10: "done" ends the alignment step early.</summary>
        public static void AlignDone()
        {
            if (s_Active == null || s_Active.m_Step != AlignStep) return;
            CuePlayer.Ensure().SkipStep();
        }

        void OffsetChanged(string what)
        {
            m_OffsetSaved = false;
            CuePlayer.Log(m_Run, "offset", m_Step + ":" + what + ":" + m_Offset.Csv().Replace(',', ' '));
            if (!m_HasLast || m_Overlay == null) return;
            PoseOverlay.Chain(m_LastHeadP, m_LastHeadQ, m_Offset, m_LastInP, m_LastInQ, out Vector3 p, out Quaternion q);
            m_Overlay.Show(m_LastTarget, p, q, m_LastHeadP);
        }

        /// <summary>A rating tap on the Beam Pro screen (T6.9), for the step now running.</summary>
        public static void Rate(string answer)
        {
            if (s_Active == null) return;
            CuePlayer.Log(s_Active.m_Run, "rating", s_Active.m_Step + ":" + answer + (s_Active.m_Mode.Length > 0 ? "@" + s_Active.m_Mode : ""));
            Debug.Log($"[EyeLab probe] overlay rating: {s_Active.m_Step} = {answer}");
        }

        public static IEnumerator Run(ProbeRun run, Action<string> status, Action<string, float, float, string> show, string variant = "detect")
        {
            var go = new GameObject("EyeLab DetectProbe");
            var p = go.AddComponent<DetectProbe>();
            p.m_Run = run;
            p.m_Variant = Array.IndexOf(new[] { "budget", "lean", "overlay", "align", "spaam", "eyecal", "eyecheck", "camcal" }, variant) >= 0 ? variant : "detect";
            run.Meta("variant", p.m_Variant);
            string info;
            try { info = ArucoNative.Info(); }
            catch (Exception e)
            {
                run.Meta("plugin_error", e.GetType().Name + " " + e.Message);
                status("Detection: the native plugin did not load: " + e.Message);
                Debug.LogError("[EyeLab probe] eyelab_aruco: " + e);
                Destroy(go);
                yield break;
            }
            run.Meta("plugin", info);
            Debug.Log("[EyeLab probe] " + info);
            p.m_Set = LoadSet(run);
            p.m_Display = XREALDisplay();
            if (p.m_Variant != "detect")
                run.Meta("budget", $"cap {CapHz} Hz; roi margin {RoiMarginFrac} x box, min {RoiMarginMinPx} px; box/prior max age " +
                                   $"{RoiMaxAgeNs / 1e6} ms; roi nice {NiceRoi}; trusted: non-planar, or ratio >= " +
                                   $"{p.LockCfg.min_ambiguity_ratio}, or <= {p.LockCfg.ambiguity_min_separation_deg} deg apart (no prior); and rms <= {p.LockCfg.max_acquire_reproj_rms_px} px; next region = projected board joined with detected target markers" +
                                   (p.Drawn ? $"; or planar on >= {TrustBoardMarkers} markers" : ""));
            p.LogDeviceGeometry();
            p.FindBoardCentre();
            if (p.Drawn) p.StartOverlay();

            // Part 1: bench on the pushed frames, on a worker thread so the glasses keep drawing.
            if (p.m_Set != null && p.m_Set.frames != null && p.m_Set.frames.Length > 0)
            {
                var bench = new Thread(p.Bench) { IsBackground = true, Name = "eyelab-b3a-bench" };
                bench.Start();
                while (bench.IsAlive)
                {
                    status("Bench: " + p.m_BenchStatus);
                    show?.Invoke("Detection bench running, please wait", 0f, 0f, "bench");
                    yield return null;
                }
                run.Meta("bench_status", p.m_BenchStatus);
            }
            else
            {
                status("No frame set in files/b3a: bench skipped");
                run.Meta("bench_status", "no set");
            }

            // Part 2: live on the Eye.
            p.StartLive();
            if (p.m_Variant == "eyecal" || p.m_Variant == "eyecheck") p.StartEyeModel();
            if (p.m_Variant == "spaam" || p.m_Variant == "eyecal")
            {
                p.m_Cross = SpaamCrosshair.Create(run);
                run.Meta("spaam", $"crosshair at {SpaamCrosshair.DistanceM} m along (tan x, tan y, 1) in the SDK eye's frame; positions (deg) " +
                                  string.Join(" ", Array.ConvertAll(SpaamCrosshair.Positions, v => $"({v.x:F0},{v.y:F0})")) +
                                  $"; real point: board25 centre of the middle square ({p.BoardCentreText}) m in the board frame; head pose = Unity camera at the tap");
            }
            float nextStatus = 0f;
            List<CueStep> cues = p.m_Variant == "camcal" ? CamcalCues() : p.m_Variant == "eyecal" ? EyecalCues() : p.m_Variant == "eyecheck" ? EyecheckCues() : p.m_Variant == "spaam" ? SpaamCues() : p.m_Variant == "align" ? AlignCues() : p.m_Variant == "overlay" ? OverlayCues() : p.m_Variant == "lean" ? LeanCues()
                               : p.m_Variant == "budget" ? BudgetCues() : DetectCues();
            s_Active = p;
            yield return CuePlayer.Ensure().Play(run, cues, (text, left, frac, counter) =>
            {
                show?.Invoke(text, left, frac, counter);
                if (Time.realtimeSinceStartup < nextStatus) return;
                nextStatus = Time.realtimeSinceStartup + 0.5f;
                status($"{counter}: {text} | {left:F0} s | Eye {p.m_EyeFrames}, done {p.m_Processed}, skipped {p.m_Skipped} | " +
                       $"{p.m_LastMode} {p.m_LastTotal:F1} ms | fps {1f / Mathf.Max(Time.smoothDeltaTime, 1e-4f):F0}");
            }, null, step =>
            {
                p.m_Step = step.Id;
                p.m_ThreadsWanted = PlanFor(p.m_Variant, step.Id).Threads;
                if (p.m_Variant == "align" && p.m_Overlay != null) p.m_Overlay.SetStyle(step.Id.Contains("_solid"));
                if (p.m_Variant == "spaam" || p.m_Variant == "eyecal" || p.m_Variant == "eyecheck" || p.m_Variant == "camcal") p.SpaamOnStep(step.Id);
            });
            p.StopLive();
            s_Active = null;
            while (p.m_Done.TryDequeue(out Done d)) p.Write(d);
            if (p.m_Overlay != null) Destroy(p.m_Overlay.gameObject);
            if (p.m_Cross != null) Destroy(p.m_Cross.gameObject);
            if (p.m_Warp != null) Destroy(p.m_Warp.gameObject);
            CalText = "";
            run.Meta("spaam_taps", p.m_Taps.ToString(CultureInfo.InvariantCulture));
            if (p.m_Offset != null)
                run.Meta("view_offset_end", p.m_Offset.Csv() + "; " + p.m_Offset.Describe() + (p.m_OffsetSaved ? "; saved" : "; not saved"));
            run.Meta("off_skipped", p.m_OffSkipped.ToString(CultureInfo.InvariantCulture));
            run.Meta("eye_frames", p.m_EyeFrames.ToString(CultureInfo.InvariantCulture));
            run.Meta("processed", p.m_Processed.ToString(CultureInfo.InvariantCulture));
            run.Meta("skipped", p.m_Skipped.ToString(CultureInfo.InvariantCulture));
            run.Meta("cap_skipped", p.m_CapSkipped.ToString(CultureInfo.InvariantCulture));
            if (p.m_Error.Length > 0) run.Meta("worker_error", p.m_Error);
            Destroy(go);
        }

        PoseLockSettings LockCfg => m_Set?.pose_lock ?? new PoseLockSettings();

        static FrameSet LoadSet(ProbeRun run)
        {
            string path = Path.Combine(SetDir, "set.json");
            if (!System.IO.File.Exists(path)) { run.Meta("set", "missing " + path); return null; }
            try
            {
                var s = JsonUtility.FromJson<FrameSet>(System.IO.File.ReadAllText(path));
                var names = new StringBuilder();
                if (s.targets != null) foreach (Target t in s.targets) names.Append(names.Length > 0 ? " " : "").Append(t.name).Append('(').Append(t.ids?.Length ?? 0).Append(')');
                run.Meta("set", $"{s.schema}; board {s.board_name}; calibration {s.calibration}; frames {s.frames?.Length ?? 0}; markers {s.ids?.Length ?? 0}; " +
                                $"targets {(names.Length > 0 ? names.ToString() : "none")}");
                return s;
            }
            catch (Exception e)
            {
                run.Meta("set", "unreadable: " + e.Message);
                return null;
            }
        }

        static XRDisplaySubsystem XREALDisplay()
        {
            var list = new List<XRDisplaySubsystem>();
            SubsystemManager.GetSubsystems(list);
            foreach (XRDisplaySubsystem d in list) if (d.running) return d;
            return null;
        }

        ArucoNative MakeDetector()
        {
            var det = new ArucoNative(m_Set?.tuning != null && m_Set.tuning.Length >= ArucoNative.TuningCount ? m_Set.tuning : null,
                                      m_Set?.dictionary ?? ArucoNative.DictionaryArUco4x4_50);
            if (m_Set?.ids != null && m_Set.ids.Length > 0) det.SetBoard(m_Set.ids, m_Set.corners);
            return det;
        }

        /// <summary>Sets the target's board on <paramref name="det"/>; false (board unchanged) when the set has no such target.</summary>
        bool SetTarget(ArucoNative det, string name)
        {
            if (string.IsNullOrEmpty(name) || m_Set == null) return false;
            if (m_Set.targets != null)
                foreach (Target t in m_Set.targets)
                    if (t.name == name && t.ids != null && t.ids.Length > 0) { det.SetBoard(t.ids, t.corners); return true; }
            if (name == "plate_A" && m_Set.ids != null && m_Set.ids.Length > 0) { det.SetBoard(m_Set.ids, m_Set.corners); return true; }
            return false;
        }

        double[] K => m_Set?.K != null && m_Set.K.Length == 9 ? m_Set.K : null;

        int[] TargetIds(string name)
        {
            if (m_Set?.targets != null)
                foreach (Target t in m_Set.targets)
                    if (t.name == name) return t.ids;
            return m_Set?.ids;
        }

        /// <summary>The box of the projected board (when posed) joined with the target's detected marker corners.</summary>
        bool BoardBox(ArucoNative det, int n, in ArucoNative.Result r, string target, out double x0, out double y0, out double x1, out double y1)
        {
            x0 = y0 = double.PositiveInfinity;
            x1 = y1 = double.NegativeInfinity;
            if (r.PoseOk == 1 && !double.IsNaN(r.ProjX0))
            {
                x0 = r.ProjX0; y0 = r.ProjY0; x1 = r.ProjX1; y1 = r.ProjY1;
            }
            int[] ids = TargetIds(target);
            for (int i = 0; i < Math.Min(Math.Max(n, 0), ArucoNative.MaxMarkers); i++)
            {
                if (ids == null || Array.IndexOf(ids, det.Ids[i]) < 0) continue;
                for (int c = 0; c < 4; c++)
                {
                    double x = det.Corners[i * 8 + c * 2], y = det.Corners[i * 8 + c * 2 + 1];
                    x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
                }
            }
            return !double.IsInfinity(x0) && x1 > x0 && y1 > y0;
        }

        /// <summary>
        /// PoseLock's acquisition rule for a planar solve: trusted when the two IPPE solutions are well apart in error
        /// or close in rotation; a non-planar solve (plate L with both faces) is always trusted. No prior: on run 2's
        /// corners (T6.7 replay) most wrong poses were 2-marker poses where neither IPPE solution was right, and a prior
        /// chain spread them (wrong poses 56 → 147 on t2_still); the plugin keeps the prior for PoseLock's motion gates later.
        /// </summary>
        bool Trusted(in ArucoNative.Result r)
        {
            if (r.PoseOk != 1) return false;
            if (!(r.RmsPx <= LockCfg.max_acquire_reproj_rms_px)) return false;   // T6.8: run 1's plate L solves up to 2198 px
            if (!(r.T2 > 0)) return false;   // T6.9: T6.8 run 1 had 5 trusted non-planar plate L poses behind the camera (t_z ≈ −0.56 m, RMS 0.6-3 px)
            if (r.Planar != 1) return true;
            if (double.IsNaN(r.AmbiguityRatio)) return true;   // one IPPE solution only
            if (BoardRule && r.BoardMarkers >= TrustBoardMarkers) return true;   // T6.9: the board
            return r.AmbiguityRatio >= LockCfg.min_ambiguity_ratio || r.AmbiguityDeg <= LockCfg.ambiguity_min_separation_deg;
        }

        // ---- bench --------------------------------------------------------------------------------------------------

        void Bench()
        {
            var ci = CultureInfo.InvariantCulture;
            var rows = new StringBuilder();
            var cornerRows = new StringBuilder();
            try
            {
                using (ArucoNative det = MakeDetector())
                {
                    for (int f = 0; f < m_Set.frames.Length; f++)
                    {
                        string name = m_Set.frames[f];
                        byte[] y = ReadPgm(Path.Combine(SetDir, name), out int w, out int h);
                        if (y == null) { rows.Append(name).Append(",unreadable\n"); continue; }
                        int[] roi = null;
                        // T6.6: full frame at 1, 2, 4, 8 threads. T6.7: full at 1 and 2, then the ROI around the full
                        // frame's projected box at 1 thread (the live ROI mode), for its cost and its parity.
                        var modes = new List<(string mode, int threads)>();
                        if (m_Variant != "detect") { modes.Add(("full", 1)); modes.Add(("full", 2)); modes.Add(("roi", 1)); }
                        else foreach (int t in DetectBenchThreads) modes.Add(("full", t));
                        foreach ((string mode, int tm) in modes)
                        {
                            if (mode == "roi" && roi == null) continue;
                            int used = ArucoNative.SetThreads(tm);
                            int[] useRoi = mode == "roi" ? roi : null;
                            det.ProcessRoi(y, w, h, w, useRoi, K, m_Set.dist, null, out _);   // warm-up, not logged
                            for (int rep = 0; rep < BenchReps; rep++)
                            {
                                int n = det.ProcessRoi(y, w, h, w, useRoi, K, m_Set.dist, null, out ArucoNative.Result r);
                                rows.Append(name).Append(',').Append(tm.ToString(ci)).Append(',').Append(used.ToString(ci)).Append(',')
                                    .Append(rep.ToString(ci)).Append(',').Append(ResultCsv(r)).Append(',').Append(mode).Append('\n');
                                if (tm == 1 && rep == 0)
                                {
                                    string key = mode == "roi" ? name + "#roi" : name;
                                    for (int i = 0; i < Math.Min(n, ArucoNative.MaxMarkers); i++)
                                        cornerRows.Append(key).Append(',').Append(CornerCsv(det, i)).Append('\n');
                                    if (mode == "full" && r.PoseOk == 1) roi = RoiAround(r.ProjX0, r.ProjY0, r.ProjX1, r.ProjY1, w, h);
                                }
                            }
                        }
                        m_BenchStatus = $"frame {f + 1}/{m_Set.frames.Length}";
                    }
                }
                m_BenchStatus += " done";
            }
            catch (Exception e)
            {
                m_BenchStatus += " error " + e.GetType().Name + ": " + e.Message;
            }
            finally
            {
                ArucoNative.SetThreads(1);
            }
            m_Run.Text("bench.csv", BenchHeader + "\n" + rows);
            m_Run.Text("bench_corners.csv", "frame,id,x0,y0,x1,y1,x2,y2,x3,y3\n" + cornerRows);
        }

        static string ResultCsv(in ArucoNative.Result r)
        {
            var ci = CultureInfo.InvariantCulture;
            string d(double v) => double.IsNaN(v) ? "" : v.ToString("R", ci);
            return string.Join(",", r.Markers.ToString(ci), r.Rejected.ToString(ci), r.BoardMarkers.ToString(ci), r.PoseOk.ToString(ci),
                r.Planar.ToString(ci), r.Points.ToString(ci), d(r.ClaheMs), d(r.DetectMs), d(r.PoseMs), d(r.TotalMs),
                r.PoseOk == 1 ? d(r.R0) : "", r.PoseOk == 1 ? d(r.R1) : "", r.PoseOk == 1 ? d(r.R2) : "",
                r.PoseOk == 1 ? d(r.T0) : "", r.PoseOk == 1 ? d(r.T1) : "", r.PoseOk == 1 ? d(r.T2) : "",
                d(r.RmsPx), d(r.AmbiguityRatio), d(r.AmbiguityDeg),
                r.RoiX.ToString(ci), r.RoiY.ToString(ci), r.RoiW.ToString(ci), r.RoiH.ToString(ci), r.Solution.ToString(ci), r.PriorUsed.ToString(ci));
        }

        static string CornerCsv(ArucoNative det, int i)
        {
            var ci = CultureInfo.InvariantCulture;
            var b = new StringBuilder();
            b.Append(det.Ids[i].ToString(ci));
            for (int k = 0; k < 8; k++) b.Append(',').Append(det.Corners[i * 8 + k].ToString("R", ci));
            return b.ToString();
        }

        static byte[] ReadPgm(string path, out int w, out int h)
        {
            w = h = 0;
            if (!System.IO.File.Exists(path)) return null;
            byte[] all = System.IO.File.ReadAllBytes(path);
            // "P5" ws width ws height ws maxval single-ws data
            int pos = 0;
            string Token()
            {
                while (pos < all.Length && (char.IsWhiteSpace((char)all[pos]) || all[pos] == '#'))
                {
                    if (all[pos] == '#') while (pos < all.Length && all[pos] != '\n') pos++;
                    else pos++;
                }
                int s = pos;
                while (pos < all.Length && !char.IsWhiteSpace((char)all[pos])) pos++;
                return Encoding.ASCII.GetString(all, s, pos - s);
            }
            if (Token() != "P5") return null;
            w = int.Parse(Token(), CultureInfo.InvariantCulture);
            h = int.Parse(Token(), CultureInfo.InvariantCulture);
            if (int.Parse(Token(), CultureInfo.InvariantCulture) != 255) return null;
            pos++;
            if (all.Length - pos < w * h) return null;
            var y = new byte[w * h];
            Buffer.BlockCopy(all, pos, y, 0, y.Length);
            return y;
        }

        // ---- live ---------------------------------------------------------------------------------------------------

        void StartLive()
        {
            ulong id = XREALPlugin.StartRGBCameraDataCapture();
            m_Capturing = id != ulong.MaxValue;
            m_Run.Meta("eye_start_capture", id.ToString(CultureInfo.InvariantCulture));
            m_Run.Meta("calibration_resolution", "1280x720 (set.json)");
            m_Run.Meta("eye_rows", "bottom-up from the SDK, reversed to upright on copy");
            m_Run.Meta("head_pose", "GetHeadPoseAtTime(cam_ts) per Eye frame, 4x4 column-major (T6.2), head.csv");
            m_Live = true;
            m_Worker = new Thread(Worker) { IsBackground = true, Name = "eyelab-b3a-live" };
            m_Worker.Start();
        }

        void StopLive()
        {
            m_Live = false;
            m_Stop = true;
            m_Signal.Set();
            m_Worker?.Join(2000);
            if (m_Capturing) XREALPlugin.StopRGBCameraDataCapture();
            m_Capturing = false;
        }

        void Update()
        {
            if (!m_Live) return;
            var ci = CultureInfo.InvariantCulture;
            int present = -1;
            if (m_Display != null && !m_Display.TryGetFramePresentCount(out present)) present = -1;
            // the main thread's Eye work (acquire, upright copy, head pose, hand-over) is timed, so its share of the frame is known
            m_EyeMs = 0;
            if (m_Capturing)
            {
                long t0 = ProbeClocks.MonoNs;
                if (Acquire()) m_EyeMs = (ProbeClocks.MonoNs - t0) / 1e6;
            }
            m_Run.Row(FramesFile, FramesHeader, $"{m_Step},{m_ThreadsWanted},{Time.frameCount},{(Time.unscaledDeltaTime * 1000f).ToString("R", ci)}," +
                                                $"{present},{ProbeClocks.MonoNs},{m_EyeMs.ToString("F3", ci)}");
            while (m_Done.TryDequeue(out Done d)) { Apply(d); Write(d); }
            if (m_Overlay != null && PlanFor(m_Variant, m_Step).Mode == "off") m_Overlay.Hold();
        }

        // ---- overlay (T6.9, T6.10) ----------------------------------------------------------------------------------

        /// <summary>The variants that draw the pose in the glasses.</summary>
        bool Drawn => m_Variant == "overlay" || m_Variant == "align" || m_Variant == "eyecal" || m_Variant == "eyecheck";

        /// <summary>The board rule (12+ markers trusted) applies to every variant from T6.9 on; T6.11 run 1 lacked it.</summary>
        bool BoardRule => Drawn || m_Variant == "spaam" || m_Variant == "camcal";

        /// <summary>The camera calibration this run poses with (the set's name), the key of the eye-tap store.</summary>
        string CameraName => string.IsNullOrEmpty(m_Set?.calibration) ? "?" : m_Set.calibration;

        /// <summary>T6.12: the earlier rounds' taps, their model, and the per-eye correction for the overlay.</summary>
        void StartEyeModel()
        {
            m_Earlier = EyeModel.LoadTaps(CameraName);
            var wearings = new HashSet<string>();
            int nl = 0, nr = 0;
            var l = new List<EyeModel.Tap>();
            var r = new List<EyeModel.Tap>();
            foreach (var t in m_Earlier) { wearings.Add(t.Wearing); if (t.Eye == "L") { nl++; l.Add(t); } else { nr++; r.Add(t); } }
            m_AllL = EyeModel.Fit(l);
            m_AllR = EyeModel.Fit(r);
            m_Run.Meta("eye_taps_earlier", $"{m_Earlier.Count} taps (L {nl}, R {nr}) from {wearings.Count} wearing(s) in {EyeModel.TapsPath}, camera '{CameraName}'");
            m_Run.Meta("eye_all_at_start", $"L {EyeModel.Mm(m_AllL)}, R {EyeModel.Mm(m_AllR)}");
            m_Warp = EyeWarp.Create();
            if (m_Overlay != null) { m_Overlay.SetStyle(true); m_Overlay.SetEyeWarp(true); }
            CalText = m_Earlier.Count > 0
                ? $"Earlier rounds: {m_Earlier.Count} taps from {wearings.Count} wearing(s); left fresh-tap error {EyeModel.LeaveOneOutMm(l):F1} mm, right {EyeModel.LeaveOneOutMm(r):F1} mm"
                : "No earlier rounds";
            s_Active = this;
            SetMode(m_AllL != null && m_AllR != null ? "all" : "sdk");
        }

        void StartOverlay()
        {
            if (m_Variant == "align")
            {
                m_Offset = ViewOffset.LoadOrZero(ViewOffset.DefaultPath, out string offStatus);
                m_Run.Meta("view_offset_start", m_Offset.Csv() + "; " + offStatus);
                m_Run.Meta("view_offset_model", "in_head' = R_off * (rgb_from_head * cam) + t_off, head frame (Unity: x right, y up, z forward); " +
                                                "turn = about the head origin (R_off' = dq R_off, t_off' = dq t_off), move = t_off += d");
            }
            m_Run.Meta("overlay_chain", "world = head(cam_ts) native with z flipped (S R S, S = diag(1,1,-1)) * rgb_from_head * " +
                                        "F R F, t (x,-y,z); latest trusted pose, no filter");
            m_Overlay = PoseOverlay.Create(m_Set);
            m_Run.Meta("overlay_solid_shader", m_Overlay.SolidShader);
        }

        /// <summary>The Eye camera's pose in the head and the vendor's intrinsics, logged by every variant: T6.11 run 1
        /// (spaam) logged neither, and its offline tap points fell back to identity (2026-11-23).</summary>
        void LogDeviceGeometry()
        {
            try
            {
                Pose pose = default;
                m_HasRgbFromHead = XREALPlugin.GetDevicePoseFromHead(XREALComponent.XREAL_COMPONENT_RGB_CAMERA, ref pose);
                if (m_HasRgbFromHead) m_RgbFromHead = pose;
            }
            catch (Exception e)
            {
                m_Run.Meta("rgb_from_head_error", e.GetType().Name + ": " + e.Message);
            }
            var ci = CultureInfo.InvariantCulture;
            Vector3 rp = m_RgbFromHead.position;
            Quaternion rq = m_RgbFromHead.rotation;
            m_Run.Meta("rgb_from_head", (m_HasRgbFromHead ? "ok" : "unavailable, identity") +
                       $"; p {rp.x.ToString("R", ci)} {rp.y.ToString("R", ci)} {rp.z.ToString("R", ci)}" +
                       $"; q {rq.x.ToString("R", ci)} {rq.y.ToString("R", ci)} {rq.z.ToString("R", ci)} {rq.w.ToString("R", ci)}");
            try
            {
                // the vendor's Eye intrinsics on this host, against the set's calibration (Surface, 1280x720: vendor fx 2.7 % short)
                Vector2 f = default, c = default;
                Vector2Int res = default;
                bool okI = XREALPlugin.GetCameraIntrinsic(XREALComponent.XREAL_COMPONENT_RGB_CAMERA, ref f, ref c);
                bool okR = XREALPlugin.GetDeviceResolution(XREALComponent.XREAL_COMPONENT_RGB_CAMERA, ref res);
                m_Run.Meta("eye_vendor_intrinsics", (okI ? $"fx {f.x.ToString("R", ci)} fy {f.y.ToString("R", ci)} cx {c.x.ToString("R", ci)} cy {c.y.ToString("R", ci)}" : "unavailable") +
                           (okR ? $"; resolution {res.x}x{res.y}" : ""));
            }
            catch (Exception e)
            {
                m_Run.Meta("eye_vendor_intrinsics", "error " + e.GetType().Name + ": " + e.Message);
            }
        }

        /// <summary>The tap point: the centre of board25's middle square (marker 8) from the set's own corners, so it
        /// follows the board size the set poses with (the print is 3.9 % over nominal, T6.13). Logged as board_centre_m.</summary>
        void FindBoardCentre()
        {
            var ci = CultureInfo.InvariantCulture;
            if (m_Set?.targets != null)
                foreach (Target t in m_Set.targets)
                {
                    if (t.name != "board25" || t.ids == null || t.corners == null) continue;
                    int k = Array.IndexOf(t.ids, 8);
                    if (k < 0 || t.corners.Length < (k + 1) * 12) break;
                    double x = 0, y = 0, z = 0;
                    for (int c = 0; c < 4; c++) { x += t.corners[k * 12 + c * 3]; y += t.corners[k * 12 + c * 3 + 1]; z += t.corners[k * 12 + c * 3 + 2]; }
                    m_BoardCentreCv = new Vector3((float)(x / 4), (float)(y / 4), (float)(z / 4));
                    break;
                }
            m_Run.Meta("board_centre_m", $"{m_BoardCentreCv.x.ToString("R", ci)} {m_BoardCentreCv.y.ToString("R", ci)} {m_BoardCentreCv.z.ToString("R", ci)}");
        }

        string BoardCentreText => string.Format(CultureInfo.InvariantCulture, "{0:F5}, {1:F5}, {2:F5}", m_BoardCentreCv.x, m_BoardCentreCv.y, m_BoardCentreCv.z);

        /// <summary>Main thread: a trusted pose with its head pose moves the drawing; the shown pose goes to overlay.csv.</summary>
        void Apply(in Done d)
        {
            if (d.Accepted) m_LastTrustedNs = d.End;
            if (d.Accepted && d.HeadOk && (m_Variant == "eyecal" || m_Variant == "spaam") && d.Target == "board25")
            {
                PoseOverlay.CvToUnity(d.R.R0, d.R.R1, d.R.R2, d.R.T0, d.R.T1, d.R.T2, out Vector3 bP, out Quaternion bQ);
                PoseOverlay.NativeHeadToUnity(d.Head, out Vector3 bhP, out Quaternion bhQ);
                PoseOverlay.InHead(m_RgbFromHead, bP, bQ, out Vector3 biP, out Quaternion biQ);
                PoseOverlay.Chain(bhP, bhQ, null, biP, biQ, out Vector3 wP, out Quaternion wQ);
                m_BoardCentres.Add((d.CamTs, wP + wQ * BoardCentreUnity));
                if (m_BoardCentres.Count > 200) m_BoardCentres.RemoveRange(0, m_BoardCentres.Count - 200);
            }
            if (m_Overlay == null || !d.Accepted || !d.HeadOk || d.Mode == "off" || !m_OverlayOn) return;
            PoseOverlay.CvToUnity(d.R.R0, d.R.R1, d.R.R2, d.R.T0, d.R.T1, d.R.T2, out Vector3 camP, out Quaternion camQ);
            PoseOverlay.NativeHeadToUnity(d.Head, out Vector3 headP, out Quaternion headQ);
            PoseOverlay.InHead(m_RgbFromHead, camP, camQ, out Vector3 inP, out Quaternion inQ);
            PoseOverlay.Chain(headP, headQ, m_Offset, inP, inQ, out Vector3 p, out Quaternion q);
            if (PlanFor(m_Variant, m_Step).Mode == "off") return;   // the step changed to off while this frame ran
            m_Overlay.Show(d.Target, p, q, headP);
            m_HasLast = true;
            m_LastTarget = d.Target;
            m_LastHeadP = headP; m_LastHeadQ = headQ; m_LastInP = inP; m_LastInQ = inQ;
            var ci = CultureInfo.InvariantCulture;
            Camera cam = Camera.main;
            Vector3 hp = cam != null ? cam.transform.localPosition : Vector3.zero;
            Quaternion hq = cam != null ? cam.transform.localRotation : Quaternion.identity;
            string f(float v) => v.ToString("R", ci);
            m_Run.Row(OverlayFile, OverlayHeader, string.Join(",", d.Seq.ToString(ci), d.Step, d.Target, d.CamTs.ToString(ci),
                ProbeClocks.MonoNs.ToString(ci), Time.frameCount.ToString(ci), f(p.x), f(p.y), f(p.z), f(q.x), f(q.y), f(q.z), f(q.w),
                f(hp.x), f(hp.y), f(hp.z), f(hq.x), f(hq.y), f(hq.z), f(hq.w),
                OffsetCsv(m_Offset), m_Overlay.Solid ? "solid" : "flat"));
        }

        /// <summary>The view offset as overlay.csv columns: translation (mm), then the rotation quaternion; zero when none.</summary>
        static string OffsetCsv(ViewOffset o)
        {
            var ci = CultureInfo.InvariantCulture;
            Vector3 t = o != null ? o.Translation * 1000f : Vector3.zero;
            Quaternion q = o != null ? o.Rotation : Quaternion.identity;
            string f(float v) => v.ToString("R", ci);
            return string.Join(",", f(t.x), f(t.y), f(t.z), f(q.x), f(q.y), f(q.z), f(q.w));
        }

        /// <returns>true when a new frame was copied and handed to the worker.</returns>
        bool Acquire()
        {
            int newest = -1, acquired = 0;
            ulong newestTs = 0;
            for (int i = 0; i < 16; i++)
            {
                ulong probeTs = 0;
                if (!XREALPlugin.TryGetRGBCameraFrame(ref probeTs)) break;
                int handle = 0;
                Vector2Int res = default;
                ulong ts = 0;
                if (!XREALPlugin.TryAcquireLatestImage(ref handle, ref res, ref ts)) break;
                if (acquired > 0 && ts == newestTs) { XREALPlugin.DisposeRGBCameraDataHandle(handle); break; }
                if (newest >= 0) XREALPlugin.DisposeRGBCameraDataHandle(newest);
                newest = handle;
                newestTs = ts;
                acquired++;
            }
            if (acquired == 0) return false;
            long arrival = ProbeClocks.MonoNs;
            try
            {
                if (newestTs == m_LastTs) return false;
                m_LastTs = newestTs;
                if (!XREALPlugin.TryGetRGBCameraDataPlane(newest, 0, out IntPtr data, out Vector2Int size) || data == IntPtr.Zero) return false;
                int bytes = size.x * size.y;
                if (m_Back == null || m_Back.Length != bytes) m_Back = new byte[bytes];
                // rows are stored bottom-up: copy them in reverse so row 0 is the top of the image
                for (int row = 0; row < size.y; row++)
                    Marshal.Copy(IntPtr.Add(data, (size.y - 1 - row) * size.x), m_Back, row * size.x, size.x);
                m_HeadBackOk = ProbeClocks.HeadPoseAt(newestTs, m_HeadBack);
                m_EyeFrames++;
                lock (m_Lock)
                {
                    if (m_HasNew) m_Skipped++;
                    (m_Front, m_Back) = (m_Back, m_Front);
                    (m_HeadFront, m_HeadBack) = (m_HeadBack, m_HeadFront);
                    m_HeadFrontOk = m_HeadBackOk;
                    m_HasNew = true;
                    m_FrontSeq = m_Seq++;
                    m_FrontTs = newestTs;
                    m_FrontArrival = arrival;
                    m_FrontStep = m_Step;
                    m_W = size.x; m_H = size.y; m_Stride = size.x;
                }
                m_Signal.Set();
                return true;
            }
            finally
            {
                XREALPlugin.DisposeRGBCameraDataHandle(newest);
            }
        }

        void Worker()
        {
            ArucoNative det = null;
            int threadsSet = 0, niceSet = 0;
            string targetSet = null;
            long lastStart = 0;
            // ROI state, per target
            double bx0 = double.NaN, by0 = double.NaN, bx1 = double.NaN, by1 = double.NaN;
            ulong boxTs = 0;
            try
            {
                det = MakeDetector();
                while (!m_Stop)
                {
                    m_Signal.WaitOne(200);
                    Done d;
                    lock (m_Lock)
                    {
                        if (!m_HasNew) continue;
                        (m_Work, m_Front) = (m_Front, m_Work);
                        m_HasNew = false;
                        d = new Done { Seq = m_FrontSeq, CamTs = m_FrontTs, Arrival = m_FrontArrival, Step = m_FrontStep, Skipped = m_Skipped,
                                       W = m_W, H = m_H, Stride = m_Stride, HeadOk = m_HeadFrontOk, Head = (float[])m_HeadFront.Clone() };
                    }
                    Plan plan = PlanFor(m_Variant, d.Step);
                    if (plan.Mode == "off")
                    {
                        Interlocked.Increment(ref m_OffSkipped);   // T6.9: detection off, the worker stays idle
                        continue;
                    }
                    if ((plan.Mode == "cap" || plan.Mode == "roi15") && lastStart != 0 && ProbeClocks.MonoNs - lastStart < 1e9 / CapHz)
                    {
                        Interlocked.Increment(ref m_CapSkipped);
                        continue;
                    }
                    if (plan.Threads != threadsSet) { ArucoNative.SetThreads(plan.Threads); threadsSet = plan.Threads; }
                    if (plan.Nice != niceSet) { int now = ArucoNative.SetNice(plan.Nice); niceSet = plan.Nice; if (now == -100) m_Error = "setpriority failed"; }
                    if (m_Variant != "detect" && plan.Target != targetSet)
                    {
                        SetTarget(det, plan.Target);
                        targetSet = plan.Target;
                        bx0 = by0 = bx1 = by1 = double.NaN;
                    }
                    int[] roi = null;
                    if (plan.Mode.StartsWith("roi") && d.CamTs - boxTs < RoiMaxAgeNs) roi = RoiAround(bx0, by0, bx1, by1, d.W, d.H);
                    d.Threads = plan.Threads;
                    d.Nice = plan.Nice;
                    d.Mode = plan.Mode;
                    d.Target = plan.Target;
                    d.Prior = null;
                    d.CapSkipped = Interlocked.Read(ref m_CapSkipped);
                    d.Start = ProbeClocks.MonoNs;
                    lastStart = d.Start;
                    int n = det.ProcessRoi(m_Work, d.W, d.H, d.Stride, roi, K, m_Set?.dist, null, out d.R);
                    d.End = ProbeClocks.MonoNs;
                    d.Accepted = Trusted(d.R);
                    // Next region: the board projected with the pose, joined with the target's detected markers (a
                    // wrong 2-marker pose can misplace the projection; the detected corners are where the markers are).
                    if (BoardBox(det, n, d.R, plan.Target, out double x0, out double y0, out double x1, out double y1))
                    {
                        bx0 = x0; by0 = y0; bx1 = x1; by1 = y1;
                        boxTs = d.CamTs;
                    }
                    else if (roi != null)
                    {
                        bx0 = by0 = bx1 = by1 = double.NaN;   // lost inside the region: the next frame searches the whole frame
                    }
                    var ids = new StringBuilder();
                    var corners = new StringBuilder();
                    for (int i = 0; i < Math.Min(Math.Max(n, 0), ArucoNative.MaxMarkers); i++)
                    {
                        if (i > 0) ids.Append(' ');
                        ids.Append(det.Ids[i].ToString(CultureInfo.InvariantCulture));
                        corners.Append(d.Seq.ToString(CultureInfo.InvariantCulture)).Append(',').Append(CornerCsv(det, i)).Append('\n');
                    }
                    d.Ids = ids.ToString();
                    d.Corners = corners.ToString();
                    if (m_Processed % (m_Variant == "camcal" ? SaveEveryCalib : SaveEvery) == 0) d.SaveCopy = (byte[])m_Work.Clone();
                    m_LastTotal = d.R.TotalMs;
                    m_LastMode = plan.Mode;
                    Interlocked.Increment(ref m_Processed);
                    m_Done.Enqueue(d);
                }
            }
            catch (Exception e)
            {
                m_Error = e.GetType().Name + ": " + e.Message;
            }
            finally
            {
                det?.Dispose();
                ArucoNative.SetThreads(1);
                ArucoNative.SetNice(0);
            }
        }

        void Write(Done d)
        {
            var ci = CultureInfo.InvariantCulture;
            string saved = "";
            if (d.SaveCopy != null)
            {
                string name = $"frames/y_{d.Seq:D6}.pgm";
                byte[] copy = d.SaveCopy;
                int w = d.W, h = d.H;
                if (m_Run.Bytes(name, () => Pgm(copy, w, h))) saved = name;
            }
            string pr(int k) => d.Prior != null ? d.Prior[k].ToString("R", ci) : "";
            m_Run.Row(LiveFile, LiveHeader, string.Join(",", d.Seq.ToString(ci), d.Step, d.Threads.ToString(ci), d.CamTs.ToString(ci),
                d.Arrival.ToString(ci), d.Start.ToString(ci), d.End.ToString(ci), d.Skipped.ToString(ci), ResultCsv(d.R), d.Ids, saved,
                d.Mode, d.Target, d.Nice.ToString(ci), d.Accepted ? "1" : "0", pr(0), pr(1), pr(2), d.CapSkipped.ToString(ci)));
            if (d.Corners.Length > 0) m_Run.Row("live_corners.csv", CornerHeader, d.Corners.TrimEnd('\n'));
            var b = new StringBuilder();
            b.Append(d.Seq.ToString(ci)).Append(',').Append(d.CamTs.ToString(ci)).Append(',').Append(d.HeadOk ? '1' : '0');
            for (int i = 0; i < 16; i++) { b.Append(','); if (d.HeadOk) b.Append(d.Head[i].ToString("R", ci)); }
            m_Run.Row(HeadFile, HeadHeader, b.ToString());
        }

        static byte[] Pgm(byte[] y, int w, int h)
        {
            byte[] head = Encoding.ASCII.GetBytes($"P5\n{w} {h}\n255\n");
            var outp = new byte[head.Length + w * h];
            Buffer.BlockCopy(head, 0, outp, 0, head.Length);
            Buffer.BlockCopy(y, 0, outp, head.Length, w * h);
            return outp;
        }

        void OnDestroy()
        {
            m_Stop = true;
            m_Signal.Set();
            if (m_Capturing) XREALPlugin.StopRGBCameraDataCapture();
        }
    }
}
