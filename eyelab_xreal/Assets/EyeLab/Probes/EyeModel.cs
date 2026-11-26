using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.12: the per-eye eye-display model from SPAAM taps (T6.11; Python twin: <c>python/xreal_gates/spaam.py</c>
    /// <c>predict</c>). In the SDK eye's frame (x right, y up, z forward, from <c>GetDevicePoseFromHead(DISPLAY_*)</c>):
    /// the wearer's eye centre C, the display's virtual image at D = 4 m (w = 1/D), the display's scale s and centre p
    /// (tan units). A real point P is seen at display coordinates
    /// <code>
    /// uv = (w·C_xy + (1 − w·C_z)·(P_xy − C_xy) / (P_z − C_z) − p) / s
    /// </code>
    /// The SDK draws P at P_xy / P_z, so drawing the point Q = (uv·P_z, P_z) instead puts it where the wearer sees P
    /// (the overlay shader does this per eye). Parameters: <c>[C_x, C_y, C_z, s, p_x, p_y]</c>; the SDK's model is
    /// <c>[0, 0, 0, 1, 0, 0]</c>.
    /// <para>T6.13 (2026-11-24): with the ChArUco board posed at its true printed size (34.17 mm squares, not 25 mm), the
    /// SDK's eye positions are right (the fitted spacing is 59 mm) and what is left is a fixed angle per eye, about
    /// 0.5-0.8 deg left and 0.65-0.8 deg up, and a 2.5 % scale: the same in three wearings on two days. So the fit frees
    /// only s and p (<see cref="Free"/>): 3 mm for a fresh tap, in the same wearing or another, against 13 mm for the SDK.
    /// The eye-centre-only model of T6.12 had absorbed the board's scale error.</para>
    /// <para>The store keys each tap by the camera calibration its point came from (the frame set's
    /// <c>calibration</c>), and a fit uses only the current one's.</para>
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
            public float left_scale = 1, left_px_deg, left_py_deg, right_scale = 1, right_px_deg, right_py_deg;
            public int left_taps, right_taps, wearings;
            public float left_loo_mm, right_loo_mm;
            public float inv_d_per_m = (float)W;
            public string saved_utc = "", source = "", camera = "", model = "scale+centre";
        }

        public static string Dir => Path.Combine(Application.persistentDataPath, "calibration");
        public static string TapsPath => Path.Combine(Dir, "eye_taps.csv");   // T6.13; spaam_taps.csv (T6.12) is left as it is
        public static string ModelPath => Path.Combine(Dir, "eye_model.json");
        const string TapsHeader = "wearing,camera,eye,step,px,py,pz,u,v";

        /// <summary>The parameters fitted: s, p_x, p_y (T6.13); the eye centre stays the SDK's.</summary>
        public static readonly int[] Free = { 3, 4, 5 };
        public static double[] Sdk() => new double[] { 0, 0, 0, 1, 0, 0 };

        public static void Predict(double[] c, double px, double py, double pz, out double u, out double v)
        {
            double dz = pz - c[2];
            double k = 1 - W * c[2];
            u = (W * c[0] + k * (px - c[0]) / dz - c[4]) / c[3];
            v = (W * c[1] + k * (py - c[1]) / dz - c[5]) / c[3];
        }

        /// <summary>Damped Gauss-Newton on the <see cref="Free"/> parameters. Null with fewer than 3 taps.</summary>
        public static double[] Fit(IList<Tap> taps)
        {
            if (taps == null || taps.Count < 3) return null;
            int n = Free.Length;
            double[] c = Sdk();
            double lambda = 1e-3;
            double cost = Cost(c, taps);
            for (int it = 0; it < 50; it++)
            {
                var JtJ = new double[n, n];
                var Jtr = new double[n];
                foreach (Tap t in taps)
                {
                    Predict(c, t.Px, t.Py, t.Pz, out double u0, out double v0);
                    double ru = u0 - t.U, rv = v0 - t.V;
                    var ju = new double[n];
                    var jv = new double[n];
                    for (int k = 0; k < n; k++)
                    {
                        var c2 = (double[])c.Clone();
                        c2[Free[k]] += 1e-6;
                        Predict(c2, t.Px, t.Py, t.Pz, out double u1, out double v1);
                        ju[k] = (u1 - u0) / 1e-6;
                        jv[k] = (v1 - v0) / 1e-6;
                    }
                    for (int a = 0; a < n; a++)
                    {
                        Jtr[a] += ju[a] * ru + jv[a] * rv;
                        for (int b = 0; b < n; b++) JtJ[a, b] += ju[a] * ju[b] + jv[a] * jv[b];
                    }
                }
                for (int a = 0; a < n; a++) JtJ[a, a] *= 1 + lambda;
                double[] step = Solve(JtJ, Jtr);
                if (step == null) break;
                var next = (double[])c.Clone();
                for (int k = 0; k < n; k++) next[Free[k]] -= step[k];
                double nc = next[3] > 0.1 ? Cost(next, taps) : double.MaxValue;
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

        /// <summary>A x = b by Gaussian elimination with partial pivoting; null if singular.</summary>
        static double[] Solve(double[,] A, double[] b)
        {
            int n = b.Length;
            var M = (double[,])A.Clone();
            var x = (double[])b.Clone();
            for (int col = 0; col < n; col++)
            {
                int piv = col;
                for (int r = col + 1; r < n; r++) if (Math.Abs(M[r, col]) > Math.Abs(M[piv, col])) piv = r;
                if (Math.Abs(M[piv, col]) < 1e-30) return null;
                if (piv != col)
                {
                    for (int k = 0; k < n; k++) { double tmp = M[col, k]; M[col, k] = M[piv, k]; M[piv, k] = tmp; }
                    double tb = x[col]; x[col] = x[piv]; x[piv] = tb;
                }
                for (int r = col + 1; r < n; r++)
                {
                    double f = M[r, col] / M[col, col];
                    for (int k = col; k < n; k++) M[r, k] -= f * M[col, k];
                    x[r] -= f * x[col];
                }
            }
            for (int r = n - 1; r >= 0; r--)
            {
                for (int k = r + 1; k < n; k++) x[r] -= M[r, k] * x[k];
                x[r] /= M[r, r];
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

        static double Deg(double tan) => Math.Atan(tan) * 180 / Math.PI;

        public static string Mm(double[] c) =>
            c == null ? "none" : string.Format(CultureInfo.InvariantCulture,
                "centre ({0:+0.00;-0.00}, {1:+0.00;-0.00}) deg, scale {2:0.000}, eye ({3:+0.0;-0.0}, {4:+0.0;-0.0}, {5:+0.0;-0.0}) mm",
                Deg(c[4]), Deg(c[5]), c[3], c[0] * 1000, c[1] * 1000, c[2] * 1000);

        /// <summary>How far apart two models draw a point at 60 cm (mm, the largest over the middle and four points
        /// 10 cm off it).</summary>
        public static double DiffMm(double[] a, double[] b, double z = 0.6)
        {
            double worst = 0;
            foreach (var (x, y) in new[] { (0.0, 0.0), (-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1) })
            {
                Predict(a, x, y, z, out double ua, out double va);
                Predict(b, x, y, z, out double ub, out double vb);
                worst = Math.Max(worst, Math.Sqrt((ua - ub) * (ua - ub) + (va - vb) * (va - vb)) * z * 1000);
            }
            return worst;
        }

        /// <summary>The parameters of one eye ("L" or "R") back from a saved model (inverse of <see cref="Fill"/>).</summary>
        public static double[] FromSaved(Saved s, string eye)
        {
            double T(float deg) => Math.Tan(deg * Math.PI / 180);
            return eye == "L"
                ? new double[] { s.left_cx_mm / 1000.0, s.left_cy_mm / 1000.0, s.left_cz_mm / 1000.0, s.left_scale, T(s.left_px_deg), T(s.left_py_deg) }
                : new double[] { s.right_cx_mm / 1000.0, s.right_cy_mm / 1000.0, s.right_cz_mm / 1000.0, s.right_scale, T(s.right_px_deg), T(s.right_py_deg) };
        }

        /// <summary>Taps written to <paramref name="path"/> (replacing it) in the store's format.</summary>
        public static void WriteTaps(string path, IEnumerable<Tap> taps)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            var ci = CultureInfo.InvariantCulture;
            var b = new StringBuilder(TapsHeader).Append('\n');
            foreach (Tap t in taps)
                b.Append(string.Join(",", t.Wearing, t.Camera, t.Eye, t.Step, t.Px.ToString("R", ci), t.Py.ToString("R", ci), t.Pz.ToString("R", ci),
                                     t.U.ToString("R", ci), t.V.ToString("R", ci))).Append('\n');
            if (File.Exists(path)) File.Delete(path);   // a pushed file may be the shell's; the folder is the app's
            File.WriteAllText(path, b.ToString());
        }

        public static void Fill(Saved s, double[] l, double[] r)
        {
            s.left_cx_mm = (float)(l[0] * 1000); s.left_cy_mm = (float)(l[1] * 1000); s.left_cz_mm = (float)(l[2] * 1000);
            s.right_cx_mm = (float)(r[0] * 1000); s.right_cy_mm = (float)(r[1] * 1000); s.right_cz_mm = (float)(r[2] * 1000);
            s.left_scale = (float)l[3]; s.left_px_deg = (float)Deg(l[4]); s.left_py_deg = (float)Deg(l[5]);
            s.right_scale = (float)r[3]; s.right_px_deg = (float)Deg(r[4]); s.right_py_deg = (float)Deg(r[5]);
        }
    }
}
