#!/usr/bin/env python3
"""
EyeLab session logger.

Captures application events, stdout/stderr, and uncaught exceptions to a
JSON-Lines file inside the .logs/ directory. JSONL is chosen because each
line is an independent JSON record, which is easy for an AI agent (or any
tool) to parse line-by-line without loading the whole file.

Each record has this shape:

    {
      "ts": "2026-04-08T14:23:55.123456",
      "level": "INFO",          # DEBUG | INFO | WARNING | ERROR | EXCEPTION
      "source": "gui",          # gui | stdout | stderr | exception | system
      "msg": "Calibration loaded from ..."
    }

Use:

    from eyelab_logger import SessionLogger
    logger = SessionLogger.start(Path(__file__).parent / ".logs")
    logger.info("App started")
    logger.error("Camera not found")
    SessionLogger.shutdown()
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class _StreamTee:
    """Wrap a stream so writes are mirrored to the JSONL log."""

    def __init__(self, original: TextIO, logger: "SessionLogger", source: str):
        self._original = original
        self._logger = logger
        self._source = source
        self._buffer = ""

    def write(self, data: str) -> int:
        try:
            self._original.write(data)
        except Exception:
            pass
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                level = "ERROR" if self._source == "stderr" else "INFO"
                self._logger._write(level, self._source, line)
        return len(data)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        if self._buffer.strip():
            level = "ERROR" if self._source == "stderr" else "INFO"
            self._logger._write(level, self._source, self._buffer.strip())
            self._buffer = ""

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._original.fileno()


class SessionLogger:
    """Singleton-style logger for one EyeLab session."""

    _instance: Optional["SessionLogger"] = None

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"eyelab_{ts}.jsonl"
        self._fh = open(self.log_path, "a", encoding="utf-8", buffering=1)

        # Tee stdout / stderr
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _StreamTee(self._orig_stdout, self, "stdout")
        sys.stderr = _StreamTee(self._orig_stderr, self, "stderr")

        # Catch uncaught exceptions
        self._orig_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

        self._write("INFO", "system", f"Session log started: {self.log_path.name}")

    # ── Public ────────────────────────────────────────────────────────────
    @classmethod
    def start(cls, log_dir: Path) -> "SessionLogger":
        if cls._instance is None:
            cls._instance = SessionLogger(log_dir)
        return cls._instance

    @classmethod
    def get(cls) -> Optional["SessionLogger"]:
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst._write("INFO", "system", "Session log closed.")
        sys.stdout = inst._orig_stdout
        sys.stderr = inst._orig_stderr
        sys.excepthook = inst._orig_excepthook
        try:
            inst._fh.close()
        except Exception:
            pass
        cls._instance = None

    def debug(self, msg: str) -> None:
        self._write("DEBUG", "gui", msg)

    def info(self, msg: str) -> None:
        self._write("INFO", "gui", msg)

    def warning(self, msg: str) -> None:
        self._write("WARNING", "gui", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", "gui", msg)

    def exception(self, msg: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._write("EXCEPTION", "gui", f"{msg}\n{tb}")

    # ── Internal ──────────────────────────────────────────────────────────
    def _write(self, level: str, source: str, msg: str) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="microseconds"),
            "level": level,
            "source": source,
            "msg": msg,
        }
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _excepthook(self, exc_type, exc_value, exc_tb) -> None:
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        self._write("EXCEPTION", "exception", tb)
        if self._orig_excepthook is not None:
            try:
                self._orig_excepthook(exc_type, exc_value, exc_tb)
            except Exception:
                pass
