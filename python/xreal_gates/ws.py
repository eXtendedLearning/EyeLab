"""Minimal blocking WebSocket server (RFC 6455), stdlib only.

Just enough for ADR-001 Step C on one machine: one client at a time, text and
binary messages, ping/pong and close. No extensions, no TLS. The Unity side is
``System.Net.WebSockets.ClientWebSocket``; client frames are masked, server
frames are not.
"""

from __future__ import annotations

import base64
import hashlib
import socket
import struct
from typing import Optional, Tuple

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA

#: Largest message accepted (a 1280x720 Y plane plus header is ~0.9 MB).
MAX_MESSAGE = 16 * 1024 * 1024


class ConnectionClosed(Exception):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _GUID).encode("ascii")).digest()).decode("ascii")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionClosed("peer closed the TCP connection")
        buf += chunk
    return bytes(buf)


def _unmask(payload: bytes, mask: bytes) -> bytes:
    if not payload:
        return payload
    try:
        import numpy as np
        data = np.frombuffer(payload, dtype=np.uint8)
        key = np.frombuffer(mask * ((len(payload) + 3) // 4), dtype=np.uint8)[:len(payload)]
        return (data ^ key).tobytes()
    except ImportError:  # pragma: no cover - numpy is a project dependency
        return bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


class WebSocket:
    """One accepted connection."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    # ---- handshake
    @classmethod
    def handshake(cls, sock: socket.socket) -> "WebSocket":
        request = bytearray()
        while b"\r\n\r\n" not in request:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionClosed("closed during handshake")
            request += chunk
            if len(request) > 65536:
                raise ConnectionClosed("handshake too large")
        headers = {}
        for line in request.decode("latin-1").split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        key = headers.get("sec-websocket-key")
        if not key:
            sock.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            raise ConnectionClosed("not a WebSocket request")
        sock.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n").encode("ascii"))
        return cls(sock)

    # ---- frames
    def _read_frame(self) -> Tuple[bool, int, bytes]:
        b0, b1 = _recv_exact(self.sock, 2)
        fin, opcode = bool(b0 & 0x80), b0 & 0x0F
        masked, length = bool(b1 & 0x80), b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", _recv_exact(self.sock, 2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _recv_exact(self.sock, 8))[0]
        if length > MAX_MESSAGE:
            raise ConnectionClosed(f"frame of {length} bytes exceeds the limit")
        mask = _recv_exact(self.sock, 4) if masked else b""
        payload = _recv_exact(self.sock, length)
        return fin, opcode, _unmask(payload, mask) if masked else payload

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            header = struct.pack(">BB", 0x80 | opcode, n)
        elif n < 1 << 16:
            header = struct.pack(">BBH", 0x80 | opcode, 126, n)
        else:
            header = struct.pack(">BBQ", 0x80 | opcode, 127, n)
        self.sock.sendall(header + payload)

    def receive(self) -> Tuple[int, bytes]:
        """Next complete message as (OP_TEXT or OP_BINARY, payload). Raises ConnectionClosed."""
        parts, kind = [], None
        while True:
            fin, opcode, payload = self._read_frame()
            if opcode == OP_PING:
                self._send_frame(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                try:
                    self._send_frame(OP_CLOSE, payload[:2])
                except OSError:
                    pass
                raise ConnectionClosed("close frame")
            if opcode in (OP_TEXT, OP_BINARY):
                kind, parts = opcode, [payload]
            elif opcode == OP_CONT and kind is not None:
                parts.append(payload)
            if fin and kind is not None:
                return kind, b"".join(parts)

    def send_text(self, text: str) -> None:
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    def close(self) -> None:
        try:
            self._send_frame(OP_CLOSE, struct.pack(">H", 1000))
        except OSError:
            pass
        self.sock.close()


def listen(host: str = "127.0.0.1", port: int = 8765) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv


def accept(srv: socket.socket, timeout: Optional[float] = None) -> Optional[WebSocket]:
    """Waits for one client; None on timeout."""
    srv.settimeout(timeout)
    try:
        sock, _ = srv.accept()
    except socket.timeout:
        return None
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return WebSocket.handshake(sock)
