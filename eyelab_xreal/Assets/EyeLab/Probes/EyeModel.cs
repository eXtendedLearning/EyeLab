using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.12: the per-eye eye-display model from SPAAM taps (T6.11; Python twin: <c>python/xreal_gates/spaam.py</c>,
    /// model "eye"). In the SDK eye's frame (x right, y up, z forward, from <c>GetDevicePoseFromHead(DISPLAY_*)</c>):
    /// the wearer's eye centre C, the display's virtual image at D = 4 m (w = 1/D). A real point P is seen at display
    /// coordinates (tan units)
    /// <code>
    /// uv = w·C_xy + (1 − w·C_z)·(P_xy − C_xy) / (P_z − C_z)
    /// </code>
    /// The SDK draws P at P_xy / P_z, so drawing the point Q = (uv·P_z, P_z) instead puts it where the wearer sees P
    /// (the overlay shader does this per eye). T6.11 run 1 (Jack): left eye C = (+9.4, −7.4, −14.8) mm, right eye
    /// (−12.1, −8.7, −12.4) mm; leave-one-out 1.9 / 2.9 mm against 12-14 mm for the SDK's model.
    /// <para>T6.13: those taps were measured with an Eye camera model 25-28 % short in focal length (tape check,
    /// 2026-11-23), which put every real point about a quarter too near. The store therefore keys each tap by the camera
    /// calibration its point came from (the frame set's <c>calibration</c>), and a fit uses only the current one's.</para>
    /// </summary>
    public static class EyeModel
    {
        public const double W = 0.25;   // display at 4 m (One series default screen distance); not identifiable from 35-100 cm

        public struct Tap
        {
            public string Wearing, Camera, Eye, Step;
            public double Px, Py, Pz, U, V;
        }

        [Serializable]
        public sealed class Saved
        {
            public float left_cx_mm, left_cy_mm, left_cz_mm, right_cx_mm, right_cy_mm, right_cz_mm;
            public int left_taps, right_taps, wearings;
            public float left_loo_mm, right_loo_mm;
            public float inv_d_per_m = (float)W;
            public string saved_utc = "", source = "", camera = "";
        }

        public static string Dir => Path.Combine(Application.persistentDataPath, "calibration");
        public static string TapsPath => Path.Combine(Dir, "eye_taps.csv");   // T6.13; spaam_taps.csv (T6.12) is left as it is
        public static string ModelPath => Path.Combine(Dir, "eye_model.json");
        const string TapsHeader = "wearing,camera,eye,step,px,py,pz,u,v";

        public static void Predict(double[] c, double px, double py, double pz, out double u, out double v)
        {
            double dz = pz - c[2];
            double k = 1 - W * c[2];
            u = W * c[0] + k * (px - c[0]) / dz;
            v = W * c[1] + k * (py - c[1]) / dz;
        }

        /// <summary>Gauss-Newton with damping on the 3 eye-centre parameters (metres). Null with fewer than 3 taps.</summary>
        public static double[] Fit(IList<Tap> taps)
        {
            if (taps == null || taps.Count < 3) return null;
            var c = new double[3];
            double lambda = 1e-3;
            double cost = Cost(c, taps);
            for (int it = 0; it < 50; it++)
            {
                var JtJ = new double[3, 3];
                var Jtr = new double[3];
                foreach (Tap t in taps)
                {
                    Predict(c, t.Px, t.Py, t.Pz, out double u0, out double v0);
                    double ru = u0 - t.U, rv = v0 - t.V;
                    var ju = new double[3];
                    var jv = new double[3];
                    for (int k = 0; k < 3; k++)
                    {
                        var c2 = (double[])c.Clone();
                        c2[k] += 1e-6;
                        Predict(c2, t.Px, t.Py, t.Pz, out double u1, out double v1);
                        ju[k] = (u1 - u0) / 1e-6;
                        jv[k] = (v1 - v0) / 1e-6;
                    }
                    for (int a = 0; a < 3; a++)
                    {
                        Jtr[a] += ju[a] * ru + jv[a] * rv;
                        for (int b = 0; b < 3; b++) JtJ[a, b] += ju[a] * ju[b] + jv[a] * jv[b];
                    }
                }
                for (int a = 0; a < 3; a++) JtJ[a, a] *= 1 + lambda;
                double[] step = Solve3(JtJ, Jtr);
                if (step == null) break;
                var next = new double[3];
                for (int k = 0; k < 3; k++) next[k] = c[k] - step[k];
                double nc = Cost(next, taps);
                if (nc < cost)
                {
                    double gain = cost - nc;
                    c = next;
                    cost = nc;
                    lambda *= 0.3;
                    if (gain < 1e-16) break;
                }
                else lambda *= 10;
            }
            return c;
        }

        static double Cost(double[] c, IList<Tap> taps)
        {
            double s = 0;
            foreach (Tap t in taps)
            {
                Predict(c, t.Px, t.Py, t.Pz, out double u, out double v);
                s += (u - t.U) * (u - t.U) + (v - t.V) * (v - t.V);
            }
            return s;
        }

        static double[] Solve3(double[,] A, double[] b)
        {
            double det = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
                       + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]);
            if (Math.Abs(det) < 1e-30) return null;
            var x = new double[3];
            for (int k = 0; k < 3; k++)
            {
                var M = (double[,])A.Clone();
                for (int r = 0; r < 3; r++) M[r, k] = b[r];
                x[k] = (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
                      + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])) / det;
            }
            return x;
        }

        /// <summary>Each tap predicted by the model fitted on the others: the median error (mm at the target) a fresh tap
        /// would have. NaN with fewer than 4 taps.</summary>
        public static double LeaveOneOutMm(IList<Tap> taps)
        {
            if (taps == null || taps.Count < 4) return double.NaN;
            var errs = new List<double>();
            for (int i = 0; i < taps.Count; i++)
            {
                var rest = new List<Tap>(taps);
                rest.RemoveAt(i);
                double[] c = Fit(rest);
                if (c == null) continue;
                Tap t = taps[i];
                Predict(c, t.Px, t.Py, t.Pz, out double u, out double v);
                errs.Add(Math.Sqrt((u - t.U) * (u - t.U) + (v - t.V) * (v - t.V)) * t.Pz * 1000);
            }
            if (errs.Count == 0) return double.NaN;
            errs.Sort();
            return errs.Count % 2 == 1 ? errs[errs.Count / 2] : (errs[errs.Count / 2 - 1] + errs[errs.Count / 2]) / 2;
        }

        /// <summary>The SDK model's median error (mm at the target) on these taps: the point drawn at P_xy / P_z.</summary>
        public static double SdkMm(IList<Tap> taps)
        {
            if (taps == null || taps.Count == 0) return double.NaN;
            var errs = new List<double>();
            foreach (Tap t in taps)
            {
                double du = t.Px / t.Pz - t.U, dv = t.Py / t.Pz - t.V;
                errs.Add(Math.Sqrt(du * du + dv * dv) * t.Pz * 1000);
            }
            errs.Sort();
            return errs[errs.Count / 2];
        }

        /// <summary>The stored taps measured under the camera calibration <paramref name="camera"/>.</summary>
        public static List<Tap> LoadTaps(string camera)
        {
            var list = new List<Tap>();
            try
            {
                if (!File.Exists(TapsPath)) return list;
                var ci = CultureInfo.InvariantCulture;
                foreach (string line in File.ReadAllLines(TapsPath))
                {
                    string[] f = line.Split(',');
                    if (f.Length < 9 || f[0] == "wearing" || f[1] != camera) continue;
                    list.Add(new Tap
                    {
                        Wearing = f[0], Camera = f[1], Eye = f[2], Step = f[3],
                        Px = double.Parse(f[4], ci), Py = double.Parse(f[5], ci), Pz = double.Parse(f[6], ci),
                        U = double.Parse(f[7], ci), V = double.Parse(f[8], ci),
                    });
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[EyeLab probe] eye_taps.csv unreadable: " + e.Message);
            }
            return list;
        }

        public static void AppendTaps(IEnumerable<Tap> taps)
        {
            Directory.CreateDirectory(Dir);
            var ci = CultureInfo.InvariantCulture;
            var b = new StringBuilder();
            if (!File.Exists(TapsPath)) b.Append(TapsHeader).Append('\n');
            foreach (Tap t in taps)
                b.Append(string.Join(",", t.Wearing, t.Camera, t.Eye, t.Step, t.Px.ToString("R", ci), t.Py.ToString("R", ci), t.Pz.ToString("R", ci),
                                     t.U.ToString("R", ci), t.V.ToString("R", ci))).Append('\n');
            try { File.AppendAllText(TapsPath, b.ToString()); }
            catch (UnauthorizedAccessException)
            {
                // a store pushed over adb belongs to the shell user, and the app may read it but not write it (T6.12 run 1):
                // the folder is the app's, so replace the file with an app-owned copy that has the new taps too
                string old = File.ReadAllText(TapsPath);
                File.Delete(TapsPath);
                File.WriteAllText(TapsPath, old + (old.EndsWith("\n") || old.Length == 0 ? "" : "\n") + b.ToString());
            }
        }

        public static Saved LoadSaved()
        {
            try { return File.Exists(ModelPath) ? JsonUtility.FromJson<Saved>(File.ReadAllText(ModelPath)) : null; }
            catch (Exception) { return null; }
        }

        public static void Save(Saved s)
        {
            Directory.CreateDirectory(Dir);
            s.saved_utc = DateTime.UtcNow.ToString("o");
            File.WriteAllText(ModelPath, JsonUtility.ToJson(s, true));
        }

        public static string Mm(double[] c) =>
            c == null ? "none" : string.Format(CultureInfo.InvariantCulture, "({0:+0.0;-0.0}, {1:+0.0;-0.0}, {2:+0.0;-0.0}) mm", c[0] * 1000, c[1] * 1000, c[2] * 1000);
    }
}
