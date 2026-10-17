using System.Collections.Generic;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// World pose of a fixed target from successive camera locks (G5). The first lock places it;
    /// a lock within <see cref="AcceptMetres"/> and <see cref="AcceptDegrees"/> of the anchor joins a
    /// sliding window whose mean is the anchor; anything further is an outlier. Three outliers in a
    /// row that agree with each other mean the target (or the tracking origin) really moved, and the
    /// window restarts from them. While frozen, locks are compared and logged but change nothing.
    /// </summary>
    public sealed class LockAnchor
    {
        public const float AcceptMetres = 0.03f;
        public const float AcceptDegrees = 3f;
        public const int Window = 8;
        public const int RelockAfter = 3;

        public enum Kind { First, Accepted, Rejected, Relocked, Frozen }

        public struct Result
        {
            public Kind Kind;
            public float JumpMetres;     // new lock vs the anchor before this lock
            public float JumpDegrees;
            public bool Moved;           // the anchor changed
        }

        readonly List<Pose> m_Window = new List<Pose>();
        readonly List<Pose> m_Outliers = new List<Pose>();

        public bool HasAnchor { get; private set; }
        public Vector3 Position { get; private set; }
        public Quaternion Rotation { get; private set; } = Quaternion.identity;
        public bool Frozen { get; set; }
        public int Count => m_Window.Count;

        public Result Add(Vector3 position, Quaternion rotation)
        {
            if (!HasAnchor)
            {
                if (Frozen) return new Result { Kind = Kind.Frozen };
                m_Window.Add(new Pose(position, rotation));
                Recompute();
                HasAnchor = true;
                return new Result { Kind = Kind.First, Moved = true };
            }
            var r = new Result
            {
                JumpMetres = Vector3.Distance(position, Position),
                JumpDegrees = Quaternion.Angle(rotation, Rotation),
            };
            if (Frozen)
            {
                r.Kind = Kind.Frozen;
                return r;
            }
            if (r.JumpMetres <= AcceptMetres && r.JumpDegrees <= AcceptDegrees)
            {
                m_Outliers.Clear();
                m_Window.Add(new Pose(position, rotation));
                if (m_Window.Count > Window) m_Window.RemoveAt(0);
                Recompute();
                r.Kind = Kind.Accepted;
                r.Moved = true;
                return r;
            }
            m_Outliers.Add(new Pose(position, rotation));
            if (m_Outliers.Count >= RelockAfter && Agree(m_Outliers))
            {
                m_Window.Clear();
                m_Window.AddRange(m_Outliers);
                m_Outliers.Clear();
                Recompute();
                r.Kind = Kind.Relocked;
                r.Moved = true;
                return r;
            }
            if (m_Outliers.Count >= RelockAfter) m_Outliers.RemoveAt(0);
            r.Kind = Kind.Rejected;
            return r;
        }

        static bool Agree(List<Pose> poses)
        {
            for (int i = 1; i < poses.Count; i++)
                if (Vector3.Distance(poses[i].position, poses[0].position) > AcceptMetres
                    || Quaternion.Angle(poses[i].rotation, poses[0].rotation) > AcceptDegrees)
                    return false;
            return true;
        }

        void Recompute()
        {
            Vector3 p = Vector3.zero;
            Vector4 q = Vector4.zero;
            Quaternion first = m_Window[0].rotation;
            foreach (var pose in m_Window)
            {
                p += pose.position;
                Quaternion r = pose.rotation;
                // same hemisphere as the first, then a normalised sum (fine for rotations a few degrees apart)
                float sign = Quaternion.Dot(first, r) < 0f ? -1f : 1f;
                q += sign * new Vector4(r.x, r.y, r.z, r.w);
            }
            Position = p / m_Window.Count;
            q.Normalize();
            Rotation = new Quaternion(q.x, q.y, q.z, q.w);
        }
    }
}
