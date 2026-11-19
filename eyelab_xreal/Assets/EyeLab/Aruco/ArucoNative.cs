using System;
using System.Runtime.InteropServices;
using System.Text;

namespace EyeLab.Aruco
{
    /// <summary>
    /// P/Invoke surface of <c>libeyelab_aruco.so</c> (native/eyelab_aruco, ADR-0004 option A2): CLAHE, ArUco detection
    /// and the board pose, mirroring the Python pipeline so the parity test can compare them on the same frames.
    /// Build the library with <c>bash tools/build_aruco_plugin.sh</c>; it is staged at
    /// <c>Assets/Plugins/Android/libs/arm64-v8a/</c> (gitignored). One <see cref="ArucoNative"/> per thread.
    /// </summary>
    public sealed class ArucoNative : IDisposable
    {
        const string Lib = "eyelab_aruco";
        public const int Abi = 1;
        public const int TuningCount = 14;
        public const int DictionaryArUco4x4_50 = 0;   // cv::aruco::DICT_4X4_50, the dictionary of every EyeLab marker

        /// <summary>The Python "balanced" preset (ArucoDetectorTuning defaults), in field order.</summary>
        public static readonly float[] BalancedTuning =
            { 2.5f, 37, 6, 8.0f, 0.022f, 0.055f, 0.70f, 5, 0.04f, 0.04f, 3, 6, 0.16f, 4.0f };

        /// <summary>Keep in step with <c>ElaResult</c> in eyelab_aruco.cpp.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Result
        {
            public int Markers, Rejected, BoardMarkers, PoseOk, Planar, Points;
            public double ClaheMs, DetectMs, PoseMs, TotalMs;
            public double R0, R1, R2, T0, T1, T2;
            public double RmsPx, AmbiguityRatio, AmbiguityDeg;
        }

        [DllImport(Lib)] static extern int ela_abi();
        [DllImport(Lib)] static extern int ela_info(StringBuilder buf, int n);
        [DllImport(Lib)] static extern int ela_set_threads(int n);
        [DllImport(Lib)] static extern IntPtr ela_create(int dictionaryId, float[] tuning, int nTuning);
        [DllImport(Lib)] static extern void ela_destroy(IntPtr h);
        [DllImport(Lib)] static extern int ela_set_board(IntPtr h, int[] ids, float[] obj, int n);
        [DllImport(Lib)] static extern int ela_last_error(IntPtr h, StringBuilder buf, int n);
        [DllImport(Lib)]
        static extern int ela_process(IntPtr h, byte[] y, int width, int height, int stride, double[] k, double[] dist, int nDist,
                                      int maxOut, int[] idsOut, float[] cornersOut, ref Result res);

        IntPtr m_Handle;
        public const int MaxMarkers = 64;
        public readonly int[] Ids = new int[MaxMarkers];
        public readonly float[] Corners = new float[MaxMarkers * 8];

        public static string Info()
        {
            var b = new StringBuilder(256);
            ela_info(b, b.Capacity);
            return b.ToString();
        }

        /// <summary>OpenCV's worker threads (process-wide). 1 = single thread, -1 = OpenCV's default. Returns the count now used.</summary>
        public static int SetThreads(int n) => ela_set_threads(n);

        public ArucoNative(float[] tuning = null, int dictionary = DictionaryArUco4x4_50)
        {
            int abi = ela_abi();
            if (abi != Abi) throw new InvalidOperationException($"libeyelab_aruco ABI {abi}, expected {Abi}: rebuild the plugin");
            tuning = tuning ?? BalancedTuning;
            m_Handle = ela_create(dictionary, tuning, tuning.Length);
            if (m_Handle == IntPtr.Zero) throw new InvalidOperationException("ela_create failed");
        }

        /// <summary>Board: ids[n] and corners[n*12] (4 corners x xyz in metres, marker_object_corners order).</summary>
        public void SetBoard(int[] ids, float[] corners)
        {
            if (ela_set_board(m_Handle, ids, corners, ids.Length) != ids.Length) throw new InvalidOperationException("ela_set_board: " + LastError());
        }

        /// <summary>One frame. <paramref name="k"/> null = detection only. Returns the markers found (all ids), -1 on error.</summary>
        public int Process(byte[] y, int width, int height, int stride, double[] k, double[] dist, out Result res)
        {
            res = default;
            return ela_process(m_Handle, y, width, height, stride, k, dist, dist?.Length ?? 0, MaxMarkers, Ids, Corners, ref res);
        }

        public string LastError()
        {
            var b = new StringBuilder(512);
            ela_last_error(m_Handle, b, b.Capacity);
            return b.ToString();
        }

        public void Dispose()
        {
            if (m_Handle != IntPtr.Zero) ela_destroy(m_Handle);
            m_Handle = IntPtr.Zero;
        }
    }
}
