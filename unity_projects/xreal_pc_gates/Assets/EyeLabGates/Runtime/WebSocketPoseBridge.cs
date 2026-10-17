using System;
using System.Diagnostics;
using System.Globalization;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace EyeLabGates
{
    /// <summary>
    /// ADR-001 Step C on one machine: the Python pose service (<c>python -m xreal_gates serve</c>,
    /// started by the launcher) on ws://127.0.0.1:port. Frames go as raw Y planes (localhost needs no
    /// JPEG; about 0.9 MB every 2 s), one request in flight at a time. Protocol: python/xreal_gates/pose_service.py.
    /// </summary>
    public sealed class WebSocketPoseBridge : IArucoPoseBridge, IDisposable
    {
        const int ReceiveChunk = 64 * 1024;

        readonly ClientWebSocket m_Socket = new ClientWebSocket();
        readonly SemaphoreSlim m_Gate = new SemaphoreSlim(1, 1);
        bool m_HelloOk;

        public bool IsAvailable => m_HelloOk && m_Socket.State == WebSocketState.Open;
        public event Action<float> OnLatencyMeasured;
        public TargetOverlayGeometry Overlay { get; private set; }

        /// <summary>Connects and says hello; the reply carries the camera model and the overlay geometry.</summary>
        public async Task<TargetOverlayGeometry> ConnectAsync(int port, string target, string session, string runDir,
                                                              float[] vendorFxFyCxCy, CancellationToken ct)
        {
            await m_Socket.ConnectAsync(new Uri($"ws://127.0.0.1:{port}/"), ct).ConfigureAwait(false);
            var hello = new HelloMessage
            {
                type = "hello", target = target, session = session, run_dir = runDir,
                vendor_fx_fy_cx_cy = vendorFxFyCxCy, width = 1280, height = 720,
            };
            await SendTextAsync(JsonUtility.ToJson(hello), ct).ConfigureAwait(false);
            string text = await ReceiveTextAsync(ct).ConfigureAwait(false);
            var reply = JsonUtility.FromJson<HelloReply>(text);
            if (reply == null || reply.type != "hello_ok")
                throw new InvalidOperationException("pose service refused: " + (reply?.message ?? text));
            Overlay = new TargetOverlayGeometry
            {
                Target = reply.target, Label = reply.target_label,
                Segments = reply.overlay_segments ?? Array.Empty<float>(),
                Points = reply.overlay_points ?? Array.Empty<float>(),
                CameraSource = reply.camera?.source ?? "", CameraDescription = reply.camera?.description ?? "",
            };
            m_HelloOk = true;
            return Overlay;
        }

        public async Task<PoseResult> EstimatePoseAsync(YGrayImage frame, ulong frameTimestampNs, CancellationToken ct)
        {
            if (!IsAvailable) return new PoseResult { Ok = false, Reason = "bridge not connected", Sequence = frame.Sequence };
            await m_Gate.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                var sw = Stopwatch.StartNew();
                string header = "{\"seq\":" + frame.Sequence.ToString(CultureInfo.InvariantCulture) +
                                ",\"cam_ts\":" + frameTimestampNs.ToString(CultureInfo.InvariantCulture) +
                                ",\"w\":" + frame.Width.ToString(CultureInfo.InvariantCulture) +
                                ",\"h\":" + frame.Height.ToString(CultureInfo.InvariantCulture) +
                                ",\"rows\":\"" + (frame.BottomUp ? "bottom_up" : "top_down") + "\"}";
                byte[] h = Encoding.UTF8.GetBytes(header);
                int n = frame.Width * frame.Height;
                var msg = new byte[4 + h.Length + n];
                BitConverter.GetBytes(h.Length).CopyTo(msg, 0); // little-endian on Windows/Android
                Buffer.BlockCopy(h, 0, msg, 4, h.Length);
                Buffer.BlockCopy(frame.Data, 0, msg, 4 + h.Length, n);
                await m_Socket.SendAsync(new ArraySegment<byte>(msg), WebSocketMessageType.Binary, true, ct).ConfigureAwait(false);

                while (true)
                {
                    string text = await ReceiveTextAsync(ct).ConfigureAwait(false);
                    var r = JsonUtility.FromJson<PoseReply>(text);
                    if (r == null || r.type != "pose" || (r.seq != 0 && r.seq != frame.Sequence)) continue;
                    float rt = (float)sw.Elapsed.TotalMilliseconds;
                    OnLatencyMeasured?.Invoke(rt);
                    bool ok = r.ok && r.R != null && r.R.Length == 9 && r.tvec != null && r.tvec.Length == 3;
                    return new PoseResult
                    {
                        Ok = ok, Reason = r.reason ?? "", Sequence = frame.Sequence,
                        RotationRowMajor = r.R,
                        TranslationCv = ok ? new Vector3(r.tvec[0], r.tvec[1], r.tvec[2]) : Vector3.zero,
                        RmsPx = r.rms_px, Ambiguity = r.ambiguity, Markers = r.n_markers, Points = r.n_points,
                        ServiceMs = r.ms, RoundTripMs = rt,
                    };
                }
            }
            finally
            {
                m_Gate.Release();
            }
        }

        async Task SendTextAsync(string text, CancellationToken ct)
        {
            byte[] b = Encoding.UTF8.GetBytes(text);
            await m_Socket.SendAsync(new ArraySegment<byte>(b), WebSocketMessageType.Text, true, ct).ConfigureAwait(false);
        }

        async Task<string> ReceiveTextAsync(CancellationToken ct)
        {
            var buffer = new byte[ReceiveChunk];
            var sb = new StringBuilder();
            while (true)
            {
                WebSocketReceiveResult r = await m_Socket.ReceiveAsync(new ArraySegment<byte>(buffer), ct).ConfigureAwait(false);
                if (r.MessageType == WebSocketMessageType.Close)
                    throw new WebSocketException("pose service closed the connection");
                sb.Append(Encoding.UTF8.GetString(buffer, 0, r.Count));
                if (r.EndOfMessage) return sb.ToString();
            }
        }

        /// <summary>Says bye and closes; never throws.</summary>
        public async Task CloseAsync()
        {
            try
            {
                if (m_Socket.State == WebSocketState.Open)
                {
                    using (var cts = new CancellationTokenSource(2000))
                    {
                        await SendTextAsync("{\"type\":\"bye\"}", cts.Token).ConfigureAwait(false);
                        await m_Socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "gate done", cts.Token).ConfigureAwait(false);
                    }
                }
            }
            catch (Exception)
            {
                // the service may already be gone
            }
            m_HelloOk = false;
        }

        public void Dispose() => m_Socket.Dispose();

        // ---- JSON shapes (JsonUtility: public fields, no nested arrays)
#pragma warning disable 0649
        [Serializable] sealed class HelloMessage
        {
            public string type, target, session, run_dir;
            public float[] vendor_fx_fy_cx_cy;
            public int width, height;
        }

        [Serializable] sealed class CameraInfo
        {
            public string source, path, description;
            public float rms_px;
        }

        [Serializable] sealed class HelloReply
        {
            public string type, target, target_label, message;
            public CameraInfo camera;
            public float[] overlay_segments, overlay_points;
        }

        [Serializable] sealed class PoseReply
        {
            public string type, reason;
            public long seq;
            public bool ok;
            public float[] R, tvec;
            public float rms_px, ambiguity, ms;
            public int n_markers, n_points;
        }
#pragma warning restore 0649
    }
}
