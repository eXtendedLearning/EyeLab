using System;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// Manual eye-to-display alignment (guide §19). The camera finds the target in the Eye camera's
    /// frame; the glasses draw it for the wearer's eyes through XREAL's display model and the vendor's
    /// RGB-camera-from-head pose. What is left over when the two disagree (vendor extrinsic, display
    /// model vs. this wearer's eyes) is a small rigid error that stays fixed relative to the head. This
    /// is that correction: a translation (mm) and rotation (deg) in the HEAD frame, applied between
    /// the head and the camera, set by eye with the keyboard and saved next to the Eye calibration.
    /// Because it lives in the head frame it holds when the wearer walks around, unlike a world offset.
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
        public bool IsZero => dx_mm == 0 && dy_mm == 0 && dz_mm == 0 && yaw_deg == 0 && pitch_deg == 0 && roll_deg == 0;

        public static ViewOffset LoadOrZero(string path, out string status)
        {
            try
            {
                if (File.Exists(path))
                {
                    var o = JsonUtility.FromJson<ViewOffset>(File.ReadAllText(path));
                    if (o != null)
                    {
                        status = $"loaded {path}";
                        return o;
                    }
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
            note = "EyeLab gates: head-frame correction between the vendor RGB-camera-from-head pose and the display, set by eye";
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(this, true));
        }

        public string Describe()
        {
            string F(float v) => (v >= 0 ? "+" : "") + v.ToString("F1", CultureInfo.InvariantCulture);
            return $"x {F(dx_mm)} y {F(dy_mm)} z {F(dz_mm)} mm, yaw {F(yaw_deg)} pitch {F(pitch_deg)} roll {F(roll_deg)} deg";
        }
    }
}
