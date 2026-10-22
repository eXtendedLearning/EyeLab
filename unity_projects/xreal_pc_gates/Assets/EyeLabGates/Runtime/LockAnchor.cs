using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// World pose of one target from successive camera locks (G5, demo). Policy since 22 Oct:
    /// <b>follow the latest lock</b>. At the 21 Oct conference runs the 6DoF *position* drifted
    /// (10 cm in 30 s standing still, metres while walking) while rotation stayed good, so a world
    /// anchor averaged over many locks and guarded by a 3 cm gate kept the overlay at a stale,
    /// drifted place (median 50 mm, p90 230-315 mm off). Taking each new lock instead leaves only the
    /// drift since the last lock (median about 18 mm at 2 s spacing, from the same logs).
    ///
    /// A lock within <see cref="SmoothMetres"/>/<see cref="SmoothDegrees"/> of the anchor is blended
    /// in by half (hides per-frame PnP noise); a larger one up to <see cref="ConfirmMetres"/> replaces
    /// it; a jump beyond that is applied only when the next lock agrees with it (one wrong pose cannot
    /// throw the overlay away). While frozen, locks are compared and logged but change nothing.
    /// </summary>
    public sealed class LockAnchor
    {
        public const float SmoothMetres = 0.015f;
        public const float SmoothDegrees = 1.5f;
        public const float ConfirmMetres = 0.10f;
        public const float AgreeMetres = 0.03f;
        public const float AgreeDegrees = 3f;

        public enum Kind { First, Smoothed, Replaced, Pending, Confirmed, Frozen }

        public struct Result
        {
            public Kind Kind;
            public float JumpMetres;     // new lock vs the anchor before this lock
            public float JumpDegrees;
            public bool Moved;           // the anchor changed
        }

        bool m_HasPending;
        Pose m_Pending;

        public bool HasAnchor { get; private set; }
        public Vector3 Position { get; private set; }
        public Quaternion Rotation { get; private set; } = Quaternion.identity;
        public bool Frozen { get; set; }
        public int Applied { get; private set; }

        public Result Add(Vector3 position, Quaternion rotation)
        {
            if (!HasAnchor)
            {
                if (Frozen) return new Result { Kind = Kind.Frozen };
                Set(position, rotation);
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
            if (r.JumpMetres <= SmoothMetres && r.JumpDegrees <= SmoothDegrees)
            {
                m_HasPending = false;
                Set(Vector3.Lerp(Position, position, 0.5f), Quaternion.Slerp(Rotation, rotation, 0.5f));
                r.Kind = Kind.Smoothed;
            }
            else if (r.JumpMetres <= ConfirmMetres)
            {
                m_HasPending = false;
                Set(position, rotation);
                r.Kind = Kind.Replaced;
            }
            else if (m_HasPending && Vector3.Distance(position, m_Pending.position) <= AgreeMetres
                     && Quaternion.Angle(rotation, m_Pending.rotation) <= AgreeDegrees)
            {
                m_HasPending = false;
                Set(position, rotation);
                r.Kind = Kind.Confirmed;
            }
            else
            {
                m_HasPending = true;
                m_Pending = new Pose(position, rotation);
                r.Kind = Kind.Pending;
                return r;
            }
            r.Moved = true;
            return r;
        }

        /// <summary>Places the anchor directly (view-offset change); clears a pending jump.</summary>
        public void Reset(Vector3 position, Quaternion rotation)
        {
            m_HasPending = false;
            HasAnchor = true;
            Set(position, rotation);
        }

        void Set(Vector3 p, Quaternion q)
        {
            Position = p;
            Rotation = q;
            Applied++;
        }
    }
}
