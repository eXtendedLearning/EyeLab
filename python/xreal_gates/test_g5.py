"""G5 pieces: target poses on synthetic renders, the WebSocket transport, the pose service."""

import base64
import json
import os
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from xreal_gates import ws
from xreal_gates.anchor import F, qmat, qrot
from xreal_gates.eyecam import VENDOR_FX_FY_CX_CY, vendor_camera
from xreal_gates.pose_service import PoseService, parse_frame
from xreal_gates.targets import BoardTarget, PlateTarget, make_target

CAM = vendor_camera()


def _rot_err_deg(rvec_a, rvec_b) -> float:
    Ra, Rb = cv2.Rodrigues(np.asarray(rvec_a, float))[0], cv2.Rodrigues(np.asarray(rvec_b, float))[0]
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


class PlatePoseTest(unittest.TestCase):
    def _check(self, key, distance, tilt=0.0, pan=0.0):
        from validation.synth import render_scene, view_pose
        t = PlateTarget(key)
        rvec, tvec = view_pose(t.target, distance, tilt, pan)
        img = render_scene(t.layout, t.target, CAM.K, rvec, tvec)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        est = t.estimate(gray, CAM.K, CAM.dist)
        self.assertTrue(est.ok, est.reason)
        self.assertLess(np.linalg.norm(np.ravel(est.tvec) - np.ravel(tvec)) * 1000, 5.0)
        self.assertLess(_rot_err_deg(est.rvec, rvec), 1.0)
        return est

    def test_plate_a_front(self):
        est = self._check("A", 0.5)
        self.assertEqual(sorted(est.marker_ids), [30, 31, 32])
        self.assertGreaterEqual(est.n_points, 6)

    def test_plate_a_oblique(self):
        self._check("A", 0.6, tilt=20.0, pan=15.0)

    def test_plate_l(self):
        est = self._check("L", 0.5)
        self.assertGreaterEqual(est.n_markers, 3)

    def test_no_plate(self):
        est = PlateTarget("A").estimate(np.full((720, 1280), 128, np.uint8), CAM.K, CAM.dist)
        self.assertFalse(est.ok)

    def test_overlay_geometry_in_metres(self):
        ov = PlateTarget("A").overlay()
        pts = np.array(ov["points"])
        self.assertEqual(len(pts), 20)
        self.assertAlmostEqual(np.ptp(pts[:, 0]), 0.160, places=6)   # 5 saddles, 40 mm pitch
        segs = np.array(ov["segments"])
        self.assertAlmostEqual(np.ptp(segs[:, [0, 3]]), 0.297, places=6)  # A4 landscape outline
        ovl = PlateTarget("L").overlay()
        self.assertEqual(len(ovl["points"]), 20)
        self.assertGreater(np.ptp(np.array(ovl["points"])[:, 2]), 0.05)   # upright face leaves the plane


class BoardPoseTest(unittest.TestCase):
    def test_board_render(self):
        t = BoardTarget()
        ppm = 8.0                                        # board image px per mm
        W, H = t.cols * t.square * 1000, t.rows * t.square * 1000
        page = t.board.generateImage((int(W * ppm), int(H * ppm)), marginSize=0)
        rvec, tvec = np.array([0.15, -0.2, 0.05]), np.array([-0.05, 0.02, 0.45])
        R = cv2.Rodrigues(rvec)[0]
        A = np.array([[1 / (1000 * ppm), 0, 0], [0, 1 / (1000 * ppm), 0], [0, 0, 1]])
        Hm = CAM.K @ np.column_stack([R[:, 0], R[:, 1], tvec]) @ A
        out = np.full((720, 1280), 128, np.uint8)
        warped = cv2.warpPerspective(page, Hm, (1280, 720), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(page, 255), Hm, (1280, 720)) > 0
        out[mask] = warped[mask]
        est = t.estimate(out, CAM.K, CAM.dist)
        self.assertTrue(est.ok, est.reason)
        self.assertGreaterEqual(est.n_points, 20)
        self.assertLess(np.linalg.norm(np.ravel(est.tvec) - tvec) * 1000, 3.0)
        self.assertLess(_rot_err_deg(est.rvec, rvec), 1.0)

    def test_overlay(self):
        ov = BoardTarget().overlay()
        self.assertEqual(len(ov["points"]), 24)          # (5-1) x (7-1) inner corners
        self.assertEqual(len(ov["segments"]), 6 + 8)


class FrameChainTest(unittest.TestCase):
    def test_flip_matches_unity_rule(self):
        # F R F is a proper rotation; THEORY 6.3: quaternion (w, x, y, z) -> (w, -x, y, -z).
        rvec = np.array([0.3, -0.5, 0.2])
        R = cv2.Rodrigues(rvec)[0]
        Ru = F @ R @ F
        self.assertAlmostEqual(np.linalg.det(Ru), 1.0, places=9)
        angle = np.linalg.norm(rvec)
        axis = rvec / angle
        x, y, z = axis * np.sin(angle / 2)
        w = np.cos(angle / 2)
        np.testing.assert_allclose(qmat([-x, y, -z, w]), Ru, atol=1e-9)

    def test_qrot_identity_and_turn(self):
        np.testing.assert_allclose(qrot([0, 0, 0, 1], [1, 2, 3]), [1, 2, 3])
        s = np.sqrt(0.5)
        np.testing.assert_allclose(qrot([0, s, 0, s], [0, 0, 1]), [1, 0, 0], atol=1e-12)  # +90 deg yaw


class TransportTest(unittest.TestCase):
    def test_accept_key_rfc_example(self):
        self.assertEqual(ws.accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_parse_frame_flips_bottom_up(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        header = json.dumps({"seq": 1, "w": 4, "h": 3, "rows": "bottom_up"}).encode()
        _, out = parse_frame(struct.pack("<I", len(header)) + header + img[::-1].tobytes())
        np.testing.assert_array_equal(out, img)

    def test_round_trip(self):
        srv = ws.listen("127.0.0.1", 0)
        port = srv.getsockname()[1]
        got = {}

        def server():
            conn = ws.accept(srv, 5)
            kind, payload = conn.receive()
            got["kind"], got["len"] = kind, len(payload)
            conn.send_text(json.dumps({"echo": len(payload)}))
            try:
                conn.receive()
            except ws.ConnectionClosed:
                got["closed"] = True
            conn.sock.close()

        th = threading.Thread(target=server)
        th.start()
        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        c.sendall(f"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += c.recv(1024)
        self.assertIn(ws.accept_key(key).encode(), resp)
        payload = os.urandom(200000)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        c.sendall(struct.pack(">BBQ", 0x82, 0x80 | 127, len(payload)) + mask + masked)
        b0, b1 = c.recv(2)
        reply = json.loads(c.recv(b1 & 0x7F).decode())
        self.assertEqual(reply["echo"], len(payload))
        c.sendall(struct.pack(">BB", 0x88, 0x80 | 2) + mask + bytes(b ^ mask[i] for i, b in enumerate(b"\x03\xe8")))
        th.join(5)
        c.close()
        srv.close()
        self.assertEqual(got["kind"], ws.OP_BINARY)
        self.assertEqual(got["len"], 200000)
        self.assertTrue(got.get("closed"))


class ServiceTest(unittest.TestCase):
    def test_hello_frame_and_once(self):
        from validation.synth import render_scene, view_pose
        with tempfile.TemporaryDirectory() as tmp:
            svc = PoseService("plateA", None, log=lambda *_: None)
            hello = svc.hello({"type": "hello", "target": "plateA", "session": "s1", "run_dir": tmp,
                               "vendor_fx_fy_cx_cy": list(VENDOR_FX_FY_CX_CY)})
            self.assertEqual(hello["camera"]["source"] in ("vendor", "calibration"), True)
            t = make_target("plateA")
            rvec, tvec = view_pose(t.target, 0.5)
            img = np.clip(render_scene(t.layout, t.target, svc.camera.K, rvec, tvec), 0, 255).astype(np.uint8)
            header = json.dumps({"seq": 5, "cam_ts": 123, "w": 1280, "h": 720, "rows": "bottom_up"}).encode()
            msg = struct.pack("<I", len(header)) + header + img[::-1].tobytes()
            reply = svc.on_frame(msg)
            self.assertTrue(reply["ok"], reply["reason"])
            self.assertLess(abs(reply["tvec"][2] - tvec[2]) * 1000, 5.0)
            self.assertIsNone(svc.on_frame(msg))            # same seq: processed once only
            svc.close()
            rows = (Path(tmp) / "pose_service.csv").read_text().splitlines()
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
