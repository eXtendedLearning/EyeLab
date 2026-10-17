"""ADR-001 Step C pose service for the G5 locked-grid gate, on the same machine.

``python -m xreal_gates serve --target plateA|plateL|board25`` listens on
``ws://127.0.0.1:8765``. The Unity gate (``WebSocketPoseBridge`` behind
``IArucoPoseBridge``) connects and says hello; the service answers with the
camera model in use and the target's overlay geometry, then estimates the
target pose for every frame it is sent and answers with it. It serves one
client and exits when that client disconnects (or after the idle timeout).

Messages (JSON text unless stated):

- client ``{"type": "hello", "target", "session", "run_dir", "vendor_fx_fy_cx_cy", "width", "height"}``
- server ``{"type": "hello_ok", "target", "target_label", "camera", "overlay_segments", "overlay_points"}``
  (flat: 6 floats per segment, 3 per point, target frame, metres)
- client binary: 4-byte little-endian header length, JSON header
  ``{"seq", "cam_ts", "w", "h", "rows": "bottom_up"|"top_down"}``, then w*h bytes of Y
- server ``{"type": "pose", "seq", "cam_ts", "ok", "reason", "R", "tvec", ...}``
  (OpenCV camera frame of the upright image, metres)

Each (session, seq) is processed at most once, in increasing order (AGENTS.md:
process each camera frame once). Every frame is logged to ``pose_service.csv``
in the gate's run folder.
"""

from __future__ import annotations

import csv
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import ws
from .eyecam import EyeCamera, load_camera
from .targets import make_target, overlay_counts

DEFAULT_PORT = 8765
LOG_FIELDS = ["t_host", "session", "seq", "cam_ts", "ok", "reason", "n_markers", "n_points", "rms_px",
              "ambiguity", "tx", "ty", "tz", "rx", "ry", "rz", "dist_m", "ms", "marker_ids"]


def parse_frame(payload: bytes):
    """(header dict, upright uint8 image) from a binary frame message."""
    (n,) = struct.unpack_from("<I", payload, 0)
    header = json.loads(payload[4:4 + n].decode("utf-8"))
    w, h = int(header["w"]), int(header["h"])
    data = payload[4 + n:]
    if len(data) < w * h:
        raise ValueError(f"frame {header.get('seq')}: {len(data)} bytes for {w}x{h}")
    img = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
    if header.get("rows", "bottom_up") == "bottom_up":
        img = img[::-1]
    return header, np.ascontiguousarray(img)


class PoseService:
    def __init__(self, target_name: str, calib_path: Optional[Path] = None, log=print):
        self.target = make_target(target_name)
        self.calib_path = calib_path
        self.camera: Optional[EyeCamera] = None
        self.log = log
        self.session = ""
        self.last_seq = -1
        self.csv = None
        self.writer = None
        self.frames = self.accepted = 0

    # ---- messages
    def hello(self, msg: dict) -> dict:
        if msg.get("target") and msg["target"] != self.target.name:
            raise ValueError(f"client wants target {msg['target']}, service serves {self.target.name}")
        self.session = str(msg.get("session", ""))
        self.last_seq = -1
        self.camera = load_camera(self.calib_path, msg.get("vendor_fx_fy_cx_cy"))
        run_dir = msg.get("run_dir")
        if run_dir and Path(run_dir).is_dir():
            self.csv = open(Path(run_dir) / "pose_service.csv", "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.csv, LOG_FIELDS)
            self.writer.writeheader()
            (Path(run_dir) / "pose_service.json").write_text(json.dumps({
                "target": self.target.name, "target_label": self.target.label, "target_source": self.target.source,
                "camera": self.camera.describe(), "K": self.camera.K.tolist(),
                "dist": np.ravel(self.camera.dist).tolist()}, indent=2), encoding="utf-8")
        overlay = self.target.overlay()
        self.log(f"hello: session {self.session}, {self.target.label}, camera {self.camera.describe()}, "
                 f"overlay {overlay_counts(overlay)}")
        # Flat arrays and no nulls: Unity's JsonUtility reads neither nested arrays nor null numbers.
        return {"type": "hello_ok", "target": self.target.name, "target_label": self.target.label,
                "camera": {"source": self.camera.source, "path": self.camera.path,
                           "rms_px": -1.0 if self.camera.rms_px is None else float(self.camera.rms_px),
                           "description": self.camera.describe()},
                "overlay_segments": [c for seg in overlay["segments"] for c in seg],
                "overlay_points": [c for p in overlay["points"] for c in p]}

    def on_frame(self, payload: bytes) -> Optional[dict]:
        from frame_packet import FramePacket

        if self.camera is None:
            raise ValueError("frame before hello")
        header, img = parse_frame(payload)
        seq = int(header["seq"])
        if seq <= self.last_seq:
            self.log(f"frame {seq} ignored: not newer than {self.last_seq} (each frame is processed once)")
            return None
        self.last_seq = seq
        cam_ts = int(header.get("cam_ts", 0))
        packet = FramePacket(image=img, source_id="xreal_eye", session_id=self.session, sequence=seq,
                             received_at=time.perf_counter(), image_mode="gray8",
                             device_time_s=cam_ts * 1e-9, device_clock="xreal_eye_cam_ts")
        t0 = time.perf_counter()
        est = self.target.estimate(packet.image, self.camera.K, self.camera.dist)
        ms = (time.perf_counter() - t0) * 1000.0
        self.frames += 1
        self.accepted += int(est.ok)
        reply = {"type": "pose", "seq": seq, "cam_ts": cam_ts, "ms": ms, **est.to_json()}
        for k in ("rms_px", "ambiguity"):
            if reply[k] is None:
                reply[k] = -1.0
        if self.writer is not None:
            t = np.ravel(est.tvec) if est.tvec is not None else [float("nan")] * 3
            r = np.ravel(est.rvec) if est.rvec is not None else [float("nan")] * 3
            self.writer.writerow({
                "t_host": f"{time.time():.3f}", "session": self.session, "seq": seq, "cam_ts": cam_ts,
                "ok": int(est.ok), "reason": est.reason, "n_markers": est.n_markers, "n_points": est.n_points,
                "rms_px": f"{est.rms_px:.3f}", "ambiguity": f"{est.ambiguity:.3f}",
                "tx": f"{t[0]:.5f}", "ty": f"{t[1]:.5f}", "tz": f"{t[2]:.5f}",
                "rx": f"{r[0]:.5f}", "ry": f"{r[1]:.5f}", "rz": f"{r[2]:.5f}",
                "dist_m": f"{float(np.linalg.norm(t)):.4f}", "ms": f"{ms:.1f}",
                "marker_ids": " ".join(str(i) for i in est.marker_ids)})
            self.csv.flush()
        return reply

    def close(self) -> None:
        if self.csv is not None:
            self.csv.close()
            self.csv = self.writer = None


def serve(target_name: str, calib_path: Optional[Path] = None, port: int = DEFAULT_PORT,
          idle_timeout: float = 600.0, log=print) -> int:
    service = PoseService(target_name, calib_path, log)
    srv = ws.listen("127.0.0.1", port)
    log(f"EyeLab pose service: {service.target.label}; ws://127.0.0.1:{port}; waiting for the gate "
        f"(gives up after {idle_timeout:.0f} s)")
    try:
        conn = ws.accept(srv, idle_timeout)
        if conn is None:
            log("no client connected; exiting")
            return 1
        log("gate connected")
        try:
            while True:
                kind, payload = conn.receive()
                if kind == ws.OP_TEXT:
                    msg = json.loads(payload.decode("utf-8"))
                    if msg.get("type") == "hello":
                        try:
                            conn.send_text(json.dumps(service.hello(msg)))
                        except Exception as e:  # report to the gate instead of dying silently
                            conn.send_text(json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}))
                            log(f"hello failed: {e}")
                    elif msg.get("type") == "bye":
                        break
                else:
                    try:
                        reply = service.on_frame(payload)
                    except Exception as e:
                        reply = {"type": "pose", "ok": False, "reason": f"{type(e).__name__}: {e}"}
                    if reply is not None:
                        conn.send_text(json.dumps(reply))
                        if service.frames % 10 == 1:
                            log(f"frame {reply.get('seq')}: {'LOCK' if reply['ok'] else 'no pose'} "
                                f"({reply.get('reason')}), {service.accepted}/{service.frames} accepted")
        except ws.ConnectionClosed as e:
            log(f"gate disconnected ({e})")
        finally:
            conn.close()
    finally:
        service.close()
        srv.close()
    log(f"done: {service.accepted}/{service.frames} frames gave a pose")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(serve(sys.argv[1] if len(sys.argv) > 1 else "plateA"))
