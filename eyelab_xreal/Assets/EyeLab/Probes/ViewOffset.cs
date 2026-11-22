using System;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace EyeLab.Probes
{
    /// <summary>
    /// T6.10: the wearer's eye-to-display correction, set by eye on the Beam Pro. Ported from the PC gate
    /// (<c>unity_projects/xreal_pc_gates/.../ViewOffset.cs</c>, §19, §23-24). The camera finds the target; the glasses
    /// draw it through XREAL's display model and the vendor's Eye-camera-from-head pose. What is left when the two
    /// disagree is a small rigid error that stays fixed relative to the head (the Surface's was about 1.3°). This is that
    /// correction: a translation (mm) and a rotation (deg) in the HEAD frame, applied between the head and the camera:
    /// <code>
    /// in_head' = R_off · in_head + t_off
    /// </code>
    /// Because it lives in the head frame, it holds when the wearer walks around, unlike a world offset.
    /// </summary>
    [Serializable]
    public sealed class ViewOffset
    {
        public float dx_mm, dy_mm, dz_mm;          // head frame: x right, y up, z forward
        public float yaw_deg, pitch_deg, roll_deg;
        public string saved_utc = "";
        public string note = "";

        public const string FileName = "view_offset.json";

        public static string DefaultPath => Path.Combine(Application.persistentDataPath, "calibration", FileName);

        public Vector3 Translation => new Vector3(dx_mm, dy_mm, dz_mm) * 0.001f;
        public Quaternion Rotation => Quaternion.Euler(pitch_deg, yaw_deg, roll_deg);

        public static ViewOffset LoadOrZero(string path, out string status)
        {
            try
            {
                if (File.Exists(path))
                {
                    var o = JsonUtility.FromJson<ViewOffset>(File.ReadAllText(path));
                    if (o != null) { status = "loaded " + path + ": " + o.Describe(); return o; }
                }
                status = "none saved (zero)";
            }
            catch (Exception e)
            {
                status = $"unreadable ({e.GetType().Name}), using zero";
            }
            return new ViewOffset();
        }

        public void Save(string path)
        {
            saved_utc = DateTime.UtcNow.ToString("o");
            note = "EyeLab T6.10: head-frame correction between the vendor Eye-camera-from-head pose and the display, set by eye";
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(this, true));
        }

        /// <summary>
        /// Turns the drawing by <paramref name="dq"/> (head frame) about <paramref name="centre"/> (head frame, before the
        /// offset): with x' = R x + t and c' = R c + t, the new offset is R' = dq R, t' = dq (t − c') + c', so c' stays put
        /// (PC gate <c>RotateAboutTarget</c>).
        /// </summary>
        public void RotateAbout(Quaternion dq, Vector3 centre)
        {
            Quaternion r = Rotation;
            Vector3 t = Translation;
            Vector3 c = r * centre + t;
            Quaternion r2 = dq * r;
            Vector3 t2 = dq * (t - c) + c;
            Vector3 e = r2.eulerAngles;
            pitch_deg = Wrap(e.x);
            yaw_deg = Wrap(e.y);
            roll_deg = Wrap(e.z);
            dx_mm = t2.x * 1000f;
            dy_mm = t2.y * 1000f;
            dz_mm = t2.z * 1000f;
        }

        /// <summary>
        /// Turns the drawing about the head origin by <paramref name="dq"/> (head frame): R' = dq R, t' = dq t. A display
        /// boresight error is an angle, so the drawing moves by the angle times its distance.
        /// </summary>
        public void TurnAboutHead(Quaternion dq) => Set(dq * Rotation, dq * Translation);

        /// <summary>Shifts the drawing by <paramref name="mm"/> in the head frame, whatever its distance.</summary>
        public void Move(Vector3 mm) => Set(Rotation, Translation + mm * 0.001f);

        void Set(Quaternion r, Vector3 t)
        {
            Vector3 e = r.eulerAngles;
            pitch_deg = Wrap(e.x);
            yaw_deg = Wrap(e.y);
            roll_deg = Wrap(e.z);
            dx_mm = t.x * 1000f;
            dy_mm = t.y * 1000f;
            dz_mm = t.z * 1000f;
        }

        static float Wrap(float d) => d > 180f ? d - 360f : d;

        public string Csv()
        {
            var ci = CultureInfo.InvariantCulture;
            return string.Join(",", dx_mm.ToString("R", ci), dy_mm.ToString("R", ci), dz_mm.ToString("R", ci),
                               yaw_deg.ToString("R", ci), pitch_deg.ToString("R", ci), roll_deg.ToString("R", ci));
        }

        public string Describe()
        {
            string F(float v) => (v >= 0 ? "+" : "") + v.ToString("F1", CultureInfo.InvariantCulture);
            return $"x {F(dx_mm)} y {F(dy_mm)} z {F(dz_mm)} mm, yaw {F(yaw_deg)} pitch {F(pitch_deg)} roll {F(roll_deg)} deg";
        }
    }
}
