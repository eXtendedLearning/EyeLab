using System;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// World pose of one fixed target from successive camera locks (G5, demo, align). Three policies
    /// (guide §19-20), run side by side on the same measurements so they can be compared fairly; one is
    /// shown. Common to all: a measurement more than <see cref="ConfirmMetres"/> from the current pose is
    /// applied only when the next one agrees with it (one wrong pose cannot throw the overlay away; a real
    /// jump, e.g. a step in 3DoF, snaps after two locks). While frozen, nothing changes.
    /// </summary>
    public abstract class LockFilter
    {
        public const float ConfirmMetres = 0.10f;
        public const float AgreeMetres = 0.03f;
        public const float AgreeDegrees = 3f;

        public enum Kind { First, Smoothed, Replaced, Pending, Confirmed, Frozen, Filtered, Held, Corrected }

        public struct Result
        {
            public Kind Kind;
            public float JumpMetres;     // measurement vs the pose before it
            public float JumpDegrees;
            public bool Moved;
        }

        bool m_HasPending;
        Pose m_Pending;

        public abstract string Name { get; }
        /// <summary>Time constant (s) of the per-frame glide of the shown overlay towards this filter's pose.</summary>
        public abstract float RenderTau { get; }
        public bool HasAnchor { get; private set; }
        public Vector3 Position { get; protected set; }
        public Quaternion Rotation { get; protected set; } = Quaternion.identity;
        public bool Frozen { get; set; }
        public int Applied { get; protected set; }

        /// <param name="weight">Measurement quality in (0, 1]; 1 = full trust.</param>
        public Result Add(Vector3 position, Quaternion rotation, double t, float weight)
        {
            if (!HasAnchor)
            {
                if (Frozen) return new Result { Kind = Kind.Frozen };
                Reset(position, rotation, t);
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
            if (r.JumpMetres > ConfirmMetres)
            {
                if (m_HasPending && Vector3.Distance(position, m_Pending.position) <= AgreeMetres
                    && Quaternion.Angle(rotation, m_Pending.rotation) <= AgreeDegrees)
                {
                    Reset(position, rotation, t);
                    r.Kind = Kind.Confirmed;
                    r.Moved = true;
                    return r;
                }
                m_HasPending = true;
                m_Pending = new Pose(position, rotation);
                r.Kind = Kind.Pending;
                return r;
            }
            m_HasPending = false;
            return Update(position, rotation, t, Mathf.Clamp(weight, 0.05f, 1f), r);
        }

        /// <summary>Places the pose directly (first lock, confirmed jump, view-offset change).</summary>
        public virtual void Reset(Vector3 position, Quaternion rotation, double t)
        {
            m_HasPending = false;
            HasAnchor = true;
            Position = position;
            Rotation = rotation;
            Applied++;
        }

        protected abstract Result Update(Vector3 p, Quaternion q, double t, float weight, Result r);

        /// <summary>+1 = smoother/stiffer, -1 = more responsive (keys ] and [).</summary>
        public abstract void Adjust(int direction);
        public abstract string Describe();

        protected static float Alpha(double dt, double cutoffHz)
        {
            double tau = 1.0 / (2.0 * Math.PI * Math.Max(cutoffHz, 1e-3));
            return (float)(1.0 / (1.0 + tau / Math.Max(dt, 1e-3)));
        }
    }

    /// <summary>
    /// Rev. 2 policy ("latest"): follow every lock; within 15 mm and 1.5 deg blend half, otherwise replace.
    /// Kept as the baseline: it removed the stale-anchor error of 21 Oct but shows every measurement's
    /// noise (22 Oct: 3-4 mm and 0.6 deg per correction at 7 Hz, seen as jitter).
    /// </summary>
    public sealed class LockAnchor : LockFilter
    {
        public const float SmoothMetres = 0.015f;
        public const float SmoothDegrees = 1.5f;

        public override string Name => "latest";
        public override float RenderTau => 0.06f;

        protected override Result Update(Vector3 p, Quaternion q, double t, float weight, Result r)
        {
            if (r.JumpMetres <= SmoothMetres && r.JumpDegrees <= SmoothDegrees)
            {
                Position = Vector3.Lerp(Position, p, 0.5f);
                Rotation = Quaternion.Slerp(Rotation, q, 0.5f);
                r.Kind = Kind.Smoothed;
            }
            else
            {
                Position = p;
                Rotation = q;
                r.Kind = Kind.Replaced;
            }
            Applied++;
            r.Moved = true;
            return r;
        }

        public override void Adjust(int direction) { }
        public override string Describe() => "latest (blend under 15 mm / 1.5 deg)";
    }

    /// <summary>
    /// One Euro filter (Casiez, Roussel and Vogel, CHI 2012) on position and rotation: a low-pass whose
    /// cutoff rises with the estimated speed, so a still target is smoothed hard (no jitter) and a moving
    /// one (6DoF drift, a step in 3DoF) is followed with little lag. The measurement weight scales the
    /// step. Defaults tuned offline on the 22 Oct runs (<c>python -m xreal_gates filters</c>): 3DoF still,
    /// the shown corners travel 4 mm/s instead of 13 (0.3.002) at 3 mm error; matched in jitter to the
    /// hysteresis defaults so the blind comparison is decided by accuracy and feel.
    /// </summary>
    public sealed class EuroLock : LockFilter
    {
        public float MinCutoffHz = 0.1f;
        public float BetaPerMps = 2f;          // cutoff increase per m/s of estimated speed
        public float BetaPerDegps = 0.015f;    // cutoff increase per deg/s of estimated angular speed
        public float DerivativeCutoffHz = 1f;

        double m_LastT;
        Vector3 m_Velocity;
        float m_AngularSpeed;

        public override string Name => "euro";
        public override float RenderTau => 0.05f;

        public override void Reset(Vector3 position, Quaternion rotation, double t)
        {
            base.Reset(position, rotation, t);
            m_LastT = t;
            m_Velocity = Vector3.zero;
            m_AngularSpeed = 0f;
        }

        protected override Result Update(Vector3 p, Quaternion q, double t, float weight, Result r)
        {
            double dt = Math.Max(t - m_LastT, 1e-3);
            m_LastT = t;
            float ad = Alpha(dt, DerivativeCutoffHz);
            m_Velocity = Vector3.Lerp(m_Velocity, (p - Position) / (float)dt, ad);
            m_AngularSpeed = Mathf.Lerp(m_AngularSpeed, r.JumpDegrees / (float)dt, ad);
            float a = Alpha(dt, MinCutoffHz + BetaPerMps * m_Velocity.magnitude) * weight;
            float ar = Alpha(dt, MinCutoffHz + BetaPerDegps * m_AngularSpeed) * weight;
            Position = Vector3.Lerp(Position, p, a);
            Rotation = Quaternion.Slerp(Rotation, q, ar);
            Applied++;
            r.Kind = Kind.Filtered;
            r.Moved = true;
            return r;
        }

        public override void Adjust(int direction)
        {
            MinCutoffHz = Mathf.Clamp(direction > 0 ? MinCutoffHz / 1.5f : MinCutoffHz * 1.5f, 0.03f, 5f);
        }

        public override string Describe() =>
            $"one euro (min cutoff {MinCutoffHz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)} Hz)";
    }

    /// <summary>
    /// Anchor with hysteresis (the ARKit/ARCore anchor idea): the shown pose stays put while a smoothed
    /// estimate of the measurements (exponential, time constant <see cref="EstimateTauSeconds"/>) stays
    /// within <see cref="ThresholdMetres"/>/<see cref="ThresholdDegrees"/> of it; when the estimate is
    /// outside for <see cref="OutsideCount"/> locks in a row the anchor moves to it, and the overlay
    /// glides there (<see cref="RenderTau"/>). No jitter while still; corrections arrive as occasional
    /// small glides.
    /// </summary>
    public sealed class HysteresisLock : LockFilter
    {
        public float ThresholdMetres = 0.005f;
        public float ThresholdDegrees = 0.6f;
        public const int OutsideCount = 3;
        public const float EstimateTauSeconds = 0.6f;

        Vector3 m_EstPos;
        Quaternion m_EstRot = Quaternion.identity;
        double m_LastT;
        int m_Outside;

        public override string Name => "hysteresis";
        public override float RenderTau => 0.3f;

        public override void Reset(Vector3 position, Quaternion rotation, double t)
        {
            base.Reset(position, rotation, t);
            m_EstPos = position;
            m_EstRot = rotation;
            m_LastT = t;
            m_Outside = 0;
        }

        protected override Result Update(Vector3 p, Quaternion q, double t, float weight, Result r)
        {
            double dt = Math.Max(t - m_LastT, 1e-3);
            m_LastT = t;
            float a = (float)(1.0 - Math.Exp(-dt / EstimateTauSeconds)) * weight;
            m_EstPos = Vector3.Lerp(m_EstPos, p, a);
            m_EstRot = Quaternion.Slerp(m_EstRot, q, a);
            bool outside = Vector3.Distance(m_EstPos, Position) > ThresholdMetres
                           || Quaternion.Angle(m_EstRot, Rotation) > ThresholdDegrees;
            m_Outside = outside ? m_Outside + 1 : 0;
            if (m_Outside >= OutsideCount)
            {
                Position = m_EstPos;
                Rotation = m_EstRot;
                Applied++;
                m_Outside = 0;
                r.Kind = Kind.Corrected;
                r.Moved = true;
                return r;
            }
            r.Kind = Kind.Held;
            return r;
        }

        public override void Adjust(int direction)
        {
            float f = direction > 0 ? 1.5f : 1f / 1.5f;
            ThresholdMetres = Mathf.Clamp(ThresholdMetres * f, 0.002f, 0.05f);
            ThresholdDegrees = Mathf.Clamp(ThresholdDegrees * f, 0.25f, 6f);
        }

        public override string Describe() =>
            $"hysteresis ({(ThresholdMetres * 1000f).ToString("F0", System.Globalization.CultureInfo.InvariantCulture)} mm / " +
            $"{ThresholdDegrees.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)} deg)";
    }

    /// <summary>
    /// One target's three filters (fed the same measurements) and the overlay pose actually shown, which
    /// glides every frame towards the active filter's pose instead of stepping at the camera rate.
    /// </summary>
    public sealed class TargetLock
    {
        public static readonly string[] Names = { "latest", "euro", "hysteresis" };

        public readonly LockFilter[] Filters = { new LockAnchor(), new EuroLock(), new HysteresisLock() };
        public int Active = 1;
        public bool HasShown { get; private set; }
        public Vector3 ShownPosition { get; private set; }
        public Quaternion ShownRotation { get; private set; } = Quaternion.identity;

        public LockFilter Current => Filters[Active];
        public bool HasAnchor => Current.HasAnchor;

        public bool Frozen
        {
            get => Current.Frozen;
            set { foreach (var f in Filters) f.Frozen = value; }
        }

        public static int IndexOf(string name) => Math.Max(0, Array.IndexOf(Names, name));

        public void Reset(Vector3 p, Quaternion q, double t)
        {
            foreach (var f in Filters) f.Reset(p, q, t);
        }

        /// <summary>Per frame: glide the shown pose (snap when more than 20 cm away, e.g. a confirmed jump).</summary>
        public void Tick(float dt)
        {
            LockFilter f = Current;
            if (!f.HasAnchor) return;
            if (!HasShown || Vector3.Distance(ShownPosition, f.Position) > 0.2f)
            {
                ShownPosition = f.Position;
                ShownRotation = f.Rotation;
                HasShown = true;
                return;
            }
            float k = 1f - Mathf.Exp(-dt / Mathf.Max(f.RenderTau, 1e-3f));
            ShownPosition = Vector3.Lerp(ShownPosition, f.Position, k);
            ShownRotation = Quaternion.Slerp(ShownRotation, f.Rotation, k);
        }
    }
}
