#!/usr/bin/env python3
"""Tests for Stage S2: the Tier B bindings, the isolation layer and the CLI.

No vendor DLL is needed: a fake library records every call, so the tests pin
the *contract* — which functions may be called, in what order, and that
cleanup always runs — on any platform.
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import native_isolation
import xreal_native
import xreal_native_probe
from xreal_native import (
    NR_SUCCESS,
    RESOLVE_ONLY_EXPORTS,
    SIGNATURES,
    TIER_B_FILES,
    NRVersion,
)


class _Fn:
    def __init__(self, name, behaviour, log):
        self.name, self.behaviour, self.log = name, behaviour, log
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.log.append(self.name)
        if isinstance(self.behaviour, BaseException):
            raise self.behaviour
        if callable(self.behaviour):
            return self.behaviour(*args)
        return self.behaviour


class _FakeNR:
    """A fake libnr_loader: every known export, configurable per function."""

    HANDLE = 0x5EED

    def __init__(self, overrides=None, missing=()):
        self.log: list[str] = []
        behaviours = {name: NR_SUCCESS for name in list(SIGNATURES) + list(RESOLVE_ONLY_EXPORTS)}
        behaviours["NRAPICreate"] = self._create
        behaviours["NRGetVersion"] = self._version
        behaviours.update(overrides or {})
        self._fns = {n: _Fn(n, b, self.log) for n, b in behaviours.items() if n not in missing}

    def _create(self, handle_ptr):
        handle_ptr.contents.value = self.HANDLE
        return NR_SUCCESS

    @staticmethod
    def _version(handle, out_ptr):
        out_ptr.contents.major, out_ptr.contents.minor, out_ptr.contents.patch = 3, 1, 2
        return NR_SUCCESS

    def __getattr__(self, name):
        try:
            return self.__dict__["_fns"][name]
        except KeyError:
            raise AttributeError(name) from None


class SignatureTableTests(unittest.TestCase):
    def test_called_and_resolve_only_sets_are_disjoint(self):
        self.assertFalse(set(SIGNATURES) & set(RESOLVE_ONLY_EXPORTS))

    def test_standalone_is_never_called(self):
        # The vendor host never calls it; ADR-003's assumption was wrong.
        self.assertIn("NRAPIInitSetStandalone", RESOLVE_ONLY_EXPORTS)

    def test_version_struct_is_three_dwords(self):
        self.assertEqual(ctypes.sizeof(NRVersion), 12)

    def test_handle_is_64_bit(self):
        self.assertEqual(ctypes.sizeof(xreal_native.NRHandle), 8)

    def test_bind_applies_signatures(self):
        lib = _FakeNR()
        functions, resolved = xreal_native.bind(lib)
        self.assertTrue(all(resolved.values()))
        for name, sig in SIGNATURES.items():
            self.assertEqual(functions[name].argtypes, list(sig.argtypes))
            self.assertIs(functions[name].restype, sig.restype)
        self.assertFalse(set(functions) & set(RESOLVE_ONLY_EXPORTS))


class SequenceTests(unittest.TestCase):
    def test_load_level_calls_nothing(self):
        lib = _FakeNR()
        report = xreal_native.run("load", lib=lib)
        self.assertEqual(lib.log, [])
        self.assertTrue(report.ok)

    def test_version_level_order(self):
        lib = _FakeNR()
        report = xreal_native.run("version", lib=lib)
        self.assertEqual(lib.log, ["NRAPICreate", "NRGetVersion", "NRAPIDestroy"])
        self.assertEqual(report.version, "3.1.2")
        self.assertEqual(report.handle, hex(_FakeNR.HANDLE))
        self.assertTrue(report.ok)

    def test_start_level_follows_host_order(self):
        lib = _FakeNR()
        report = xreal_native.run("start", lib=lib)
        self.assertEqual(lib.log, [
            "NRAPICreate", "NRAPIInitSetNetworkType", "NRAPIStart",
            "NRGetVersion", "NRAPIStop", "NRAPIDestroy",
        ])
        self.assertTrue(report.started)
        self.assertTrue(report.ok)

    def test_network_type_argument_is_zero(self):
        seen = []
        lib = _FakeNR({"NRAPIInitSetNetworkType": lambda h, v: seen.append(v) or 0})
        xreal_native.run("start", lib=lib)
        self.assertEqual(seen, [0])

    def test_no_resolve_only_export_is_ever_called(self):
        for level in xreal_native.LEVELS:
            lib = _FakeNR()
            xreal_native.run(level, lib=lib)
            self.assertFalse(set(lib.log) & set(RESOLVE_ONLY_EXPORTS), level)

    def test_create_failure_stops_everything(self):
        lib = _FakeNR({"NRAPICreate": 1})
        report = xreal_native.run("start", lib=lib)
        self.assertEqual(lib.log, ["NRAPICreate"])
        self.assertFalse(report.ok)
        self.assertIn("NRAPICreate failed", report.error)

    def test_start_failure_skips_stop_but_destroys(self):
        lib = _FakeNR({"NRAPIStart": 7})
        report = xreal_native.run("start", lib=lib)
        self.assertEqual(lib.log, ["NRAPICreate", "NRAPIInitSetNetworkType",
                                   "NRAPIStart", "NRAPIDestroy"])
        self.assertFalse(report.started)
        self.assertFalse(report.ok)
        self.assertEqual(report.calls[2].result, 7)

    def test_version_failure_is_reported_and_cleanup_runs(self):
        lib = _FakeNR({"NRGetVersion": 3})
        report = xreal_native.run("version", lib=lib)
        self.assertEqual(lib.log[-1], "NRAPIDestroy")
        self.assertIsNone(report.version)
        self.assertFalse(report.ok)

    def test_native_fault_is_recorded_not_raised(self):
        # What ctypes raises on Windows for an access violation in the callee.
        lib = _FakeNR({"NRGetVersion": OSError("exception: access violation reading 0x0")})
        report = xreal_native.run("version", lib=lib)
        self.assertIn("access violation", report.calls[1].error)
        self.assertEqual(lib.log[-1], "NRAPIDestroy")
        self.assertFalse(report.ok)

    def test_missing_required_export_calls_nothing(self):
        lib = _FakeNR(missing=("NRAPIStart",))
        report = xreal_native.run("version", lib=lib)
        self.assertEqual(lib.log, [])
        self.assertIn("NRAPIStart", report.missing_required)
        self.assertFalse(report.ok)

    def test_unknown_level_rejected(self):
        with self.assertRaises(ValueError):
            xreal_native.run("flash", lib=_FakeNR())

    def test_report_is_json_serialisable(self):
        payload = xreal_native.run("start", lib=_FakeNR()).to_dict()
        text = json.dumps(payload)
        self.assertTrue(json.loads(text)["ok"])
        self.assertTrue(all(c["succeeded"] for c in payload["calls"]))


class PreflightTests(unittest.TestCase):
    def test_reports_missing_files_and_system_dlls(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)
            for name in TIER_B_FILES[:3]:
                (staged / name).write_bytes(b"MZ")
            result = xreal_native.preflight(staged, dll_present=lambda n: n != "vulkan-1.dll")
        self.assertEqual(result.missing_files, list(TIER_B_FILES[3:]))
        self.assertEqual(result.missing_system, ["vulkan-1.dll"])
        self.assertFalse(result.ok)

    def test_run_stops_at_failed_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = xreal_native.run("version", directory=Path(tmp))
        self.assertFalse(report.loaded)
        self.assertIn("preflight", report.error)
        self.assertTrue(report.preflight.missing_files)

    def test_env_override(self):
        with patch.dict("os.environ", {xreal_native.ENV_DIR: "/somewhere"}):
            self.assertEqual(xreal_native.find_directory(), Path("/somewhere"))

    def test_default_is_the_shared_vendor_dir(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(xreal_native.find_directory().parts[-3:],
                             ("vendor", "xreal", "win-x64"))


class _Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class IsolationTests(unittest.TestCase):
    def test_payload_survives_vendor_noise_on_stdout(self):
        noisy = ("[libnr_api] init...\n\n" + native_isolation.SENTINEL
                 + json.dumps({"ok": True}) + "\n[libnr_api] shutdown\n")
        with patch.object(subprocess, "run", return_value=_Done(stdout=noisy)):
            payload = native_isolation.run_json_worker(["x"], 5)
        self.assertTrue(payload["ok"])
        self.assertIn("init...", payload["native_output"])
        self.assertIn("shutdown", payload["native_output"])

    def test_plain_json_still_accepted(self):
        with patch.object(subprocess, "run", return_value=_Done(stdout='{"ok": true}')):
            self.assertTrue(native_isolation.run_json_worker(["x"], 5)["ok"])

    def test_access_violation_exit_code_is_named(self):
        with patch.object(subprocess, "run", return_value=_Done(returncode=0xC0000005)):
            payload = native_isolation.run_json_worker(["x"], 5, label="S2")
        self.assertFalse(payload["ok"])
        self.assertIn("access violation", payload["fatal"])

    def test_timeout_is_reported(self):
        with patch.object(subprocess, "run",
                          side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            payload = native_isolation.run_json_worker(["x"], 1, label="S2")
        self.assertIn("hanging", payload["fatal"])

    def test_emit_round_trips(self):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            native_isolation.emit({"a": 1})
        payload, other = native_isolation.extract_payload("junk\n" + sink.getvalue())
        self.assertEqual(payload, {"a": 1})
        self.assertEqual(other, "junk")


class ProbeCliTests(unittest.TestCase):
    def _main(self, argv, payload):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), \
                patch.object(xreal_native_probe, "run_isolated", return_value=payload) as run, \
                patch.object(xreal_native_probe, "append_jsonl"):
            code = xreal_native_probe.main(argv)
        return code, sink.getvalue(), run

    def test_default_level_is_version_via_loader(self):
        code, _, run = self._main([], {"ok": True})
        self.assertEqual(code, 0)
        level, entry, _dir, timeout = run.call_args[0]
        self.assertEqual((level, entry), ("version", "loader"))
        self.assertEqual(timeout, xreal_native_probe.DEFAULT_TIMEOUT_S["version"])

    def test_start_prints_warning_and_fails_exit_code(self):
        code, out, _ = self._main(["--level", "start"], {"ok": False, "fatal": "boom"})
        self.assertEqual(code, 1)
        self.assertIn("NRAPIStart", out)
        self.assertIn("boom", out)

    def test_render_full_report(self):
        payload = xreal_native.run("start", lib=_FakeNR()).to_dict()
        payload["native_output"] = "[libnr] hello"
        text = xreal_native_probe._render(payload)
        for needle in ("NRAPIStart", "3.1.2", "started           yes", "[libnr] hello", "OK"):
            self.assertIn(needle, text)

    def test_render_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = xreal_native.run("version", directory=Path(tmp)).to_dict()
        text = xreal_native_probe._render(payload)
        self.assertIn("not staged", text)
        self.assertIn("incomplete", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
