#!/usr/bin/env python3
"""Tests for the read-only XREAL HID probe.

All of these run without the vendor DLL and without the glasses. The
hid_device_info linked list is built in memory with the same ctypes
Structure the real code walks, so the parsing is exercised for real on any
platform even though the DLL itself is Windows-only.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import unittest
from ctypes import POINTER, byref, c_char_p, c_int, c_ushort, pointer
from pathlib import Path
from unittest.mock import patch

import xreal_glasses
import xreal_probe
from xreal_glasses import (
    HIDAPI_EXPORTS,
    MUTATING_EXPORTS,
    UNDOCUMENTED_READ_EXPORTS,
    XREAL_VENDOR_ID,
    HidInterface,
    XrealLibraryError,
    _HidDeviceInfo,
    enumerate_hid,
    find_library,
    inspect_library,
    load_library,
    probe,
)


def _make_node(vendor_id, product_id, *, interface=0, serial="SN", product="Glasses",
               manufacturer="XREAL", usage_page=0xFF00, usage=0x0001, bus_type=1,
               path=b"\\\\?\\hid#vid_3318"):
    """Build one hid_device_info. Keeps its string buffers alive on the node."""
    node = _HidDeviceInfo()
    node.path = path
    node.vendor_id = vendor_id
    node.product_id = product_id
    node.serial_number = serial
    node.release_number = 0x0100
    node.manufacturer_string = manufacturer
    node.product_string = product
    node.usage_page = usage_page
    node.usage = usage
    node.interface_number = interface
    node.next = None
    node.bus_type = bus_type
    return node


def _link(nodes):
    """Chain nodes and return a pointer to the head, keeping refs alive."""
    for current, following in zip(nodes, nodes[1:]):
        current.next = pointer(following)
    return pointer(nodes[0]) if nodes else POINTER(_HidDeviceInfo)()


class _FakeFn:
    """Stand-in for a ctypes function pointer: records calls, returns a value."""

    def __init__(self, result=None, recorder=None, name=""):
        self.result = result
        self.restype = None
        self.argtypes = None
        self._recorder = recorder if recorder is not None else []
        self._name = name

    def __call__(self, *args):
        self._recorder.append((self._name, args))
        if callable(self.result):
            return self.result(*args)
        return self.result


class _FakeLib:
    """Stand-in for ctypes.CDLL exposing a chosen set of exports."""

    def __init__(self, exports):
        self.calls: list[tuple] = []
        self._exports = {}
        for name, result in exports.items():
            self._exports[name] = _FakeFn(result, self.calls, name)

    def __getattr__(self, name):
        try:
            return self.__dict__["_exports"][name]
        except KeyError:
            raise AttributeError(name) from None


def _hidapi_exports(nodes, *, init_result=0):
    head = _link(nodes)
    return {
        "hid_init": init_result,
        "hid_exit": 0,
        "hid_enumerate": head,
        "hid_free_enumeration": None,
        "hid_error": None,
        "hid_version_str": b"0.14.0",
    }


class LibraryDiscoveryTests(unittest.TestCase):
    def test_env_override_takes_priority(self):
        with patch.dict(os.environ, {xreal_glasses.ENV_OVERRIDE: "/tmp/custom.dll"}):
            first = xreal_glasses.candidate_library_paths()[0]
        self.assertEqual(str(first), str(Path("/tmp/custom.dll")))

    def test_vendor_dir_is_a_candidate(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = [str(p) for p in xreal_glasses.candidate_library_paths()]
        self.assertTrue(
            any(p.endswith(os.path.join("vendor", "xreal", "win-x64",
                                        "libnr_glasses_api.dll")) for p in paths),
            paths,
        )

    def test_missing_library_raises_with_instructions(self):
        with patch.object(xreal_glasses, "find_library", return_value=None):
            with self.assertRaises(XrealLibraryError) as ctx:
                load_library()
        message = str(ctx.exception)
        self.assertIn("vendor/README.md", message)
        self.assertIn(xreal_glasses.ENV_OVERRIDE, message)

    def test_non_file_path_rejected(self):
        with self.assertRaises(XrealLibraryError):
            load_library(Path("/definitely/not/here.dll"))


class EnumerationTests(unittest.TestCase):
    def test_walks_the_linked_list(self):
        nodes = [
            _make_node(XREAL_VENDOR_ID, 0x0436, interface=0),
            _make_node(XREAL_VENDOR_ID, 0x0436, interface=3, usage_page=0xFF01),
            _make_node(0x046D, 0xC534, manufacturer="Logitech", product="Receiver"),
        ]
        lib = _FakeLib(_hidapi_exports(nodes))
        found = enumerate_hid(lib)

        self.assertEqual(len(found), 3)
        self.assertEqual(found[0].vendor_id, XREAL_VENDOR_ID)
        self.assertEqual(found[1].interface_number, 3)
        self.assertEqual(found[1].usage_page, 0xFF01)
        self.assertEqual(found[2].manufacturer, "Logitech")
        self.assertEqual(found[0].bus_type, "usb")
        self.assertTrue(found[0].is_xreal)
        self.assertFalse(found[2].is_xreal)

    def test_enumerates_unfiltered(self):
        """hid_enumerate(0, 0) — a wrong vendor-id guess must not hide devices."""
        nodes = [_make_node(XREAL_VENDOR_ID, 0x0436)]
        lib = _FakeLib(_hidapi_exports(nodes))
        enumerate_hid(lib)
        enumerate_calls = [c for c in lib.calls if c[0] == "hid_enumerate"]
        self.assertEqual(enumerate_calls[0][1], (0, 0))

    def test_frees_enumeration_and_exits(self):
        nodes = [_make_node(XREAL_VENDOR_ID, 0x0436)]
        lib = _FakeLib(_hidapi_exports(nodes))
        enumerate_hid(lib)
        names = [c[0] for c in lib.calls]
        self.assertIn("hid_free_enumeration", names)
        self.assertIn("hid_exit", names)
        self.assertLess(names.index("hid_free_enumeration"), names.index("hid_exit"))

    def test_frees_enumeration_even_when_parsing_raises(self):
        nodes = [_make_node(XREAL_VENDOR_ID, 0x0436)]
        lib = _FakeLib(_hidapi_exports(nodes))
        with patch.object(xreal_glasses, "_to_interface", side_effect=ValueError("x")):
            with self.assertRaises(ValueError):
                enumerate_hid(lib)
        self.assertIn("hid_free_enumeration", [c[0] for c in lib.calls])
        self.assertIn("hid_exit", [c[0] for c in lib.calls])

    def test_empty_enumeration_is_not_an_error(self):
        lib = _FakeLib(_hidapi_exports([]))
        self.assertEqual(enumerate_hid(lib), [])

    def test_hid_init_failure_raises(self):
        lib = _FakeLib(_hidapi_exports([], init_result=-1))
        with self.assertRaises(XrealLibraryError):
            enumerate_hid(lib)

    def test_missing_hidapi_export_is_reported_clearly(self):
        exports = _hidapi_exports([])
        del exports["hid_enumerate"]
        lib = _FakeLib(exports)
        with self.assertRaises(XrealLibraryError) as ctx:
            enumerate_hid(lib)
        self.assertIn("hid_enumerate", str(ctx.exception))

    def test_cyclic_list_does_not_spin_forever(self):
        node = _make_node(XREAL_VENDOR_ID, 0x0436)
        node.next = pointer(node)
        lib = _FakeLib(
            {**_hidapi_exports([]), "hid_enumerate": pointer(node)}
        )
        with self.assertRaises(XrealLibraryError) as ctx:
            enumerate_hid(lib)
        self.assertIn("cyclic", str(ctx.exception))


class LibraryInspectionTests(unittest.TestCase):
    def test_reports_present_and_absent_exports(self):
        exports = _hidapi_exports([])
        exports["NRBSPGetUsbConfigAll"] = 0
        lib = _FakeLib(exports)
        report = inspect_library(lib, Path("/tmp/libnr_glasses_api.dll"))

        self.assertTrue(report.loaded)
        self.assertEqual(report.hidapi_version, "0.14.0")
        self.assertTrue(report.resolved["hid_enumerate"])
        self.assertTrue(report.resolved["NRBSPGetUsbConfigAll"])
        self.assertFalse(report.resolved["NRBSPSetUsbConfigAll"])
        self.assertEqual(report.missing_hidapi, [])

    def test_inspection_calls_only_hid_version_str(self):
        lib = _FakeLib(_hidapi_exports([]))
        inspect_library(lib, Path("/tmp/x.dll"))
        called = {c[0] for c in lib.calls}
        self.assertEqual(called, {"hid_version_str"})


class SafetyBoundaryTests(unittest.TestCase):
    """The point of Stage 0: nothing proprietary, nothing mutating, is called."""

    def test_no_mutating_export_is_ever_invoked(self):
        nodes = [_make_node(XREAL_VENDOR_ID, 0x0436)]
        exports = _hidapi_exports(nodes)
        for name in MUTATING_EXPORTS + UNDOCUMENTED_READ_EXPORTS:
            exports[name] = 0
        lib = _FakeLib(exports)

        inspect_library(lib, Path("/tmp/x.dll"))
        enumerate_hid(lib)

        invoked = {c[0] for c in lib.calls}
        forbidden = invoked.intersection(set(MUTATING_EXPORTS))
        self.assertEqual(forbidden, set(), f"mutating call made: {forbidden}")
        proprietary = invoked.intersection(set(UNDOCUMENTED_READ_EXPORTS))
        self.assertEqual(proprietary, set(), f"undocumented call made: {proprietary}")
        self.assertTrue(invoked.issubset(set(HIDAPI_EXPORTS)), invoked)

    def test_set_usb_config_is_classified_as_mutating(self):
        """Guards the classification itself, not just the call sites."""
        self.assertIn("NRBSPSetUsbConfigAll", MUTATING_EXPORTS)
        self.assertNotIn("NRBSPSetUsbConfigAll", UNDOCUMENTED_READ_EXPORTS)
        self.assertNotIn("NRBSPSetUsbConfigAll", HIDAPI_EXPORTS)

    def test_export_classifications_are_disjoint(self):
        sets = [set(HIDAPI_EXPORTS), set(UNDOCUMENTED_READ_EXPORTS),
                set(MUTATING_EXPORTS)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertEqual(sets[a] & sets[b], set())


class ProbeResultTests(unittest.TestCase):
    def test_probe_reports_missing_library_without_raising(self):
        with patch.object(xreal_glasses, "find_library", return_value=None):
            result = probe()
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.library.error)
        self.assertFalse(result.library.loaded)

    def test_probe_result_is_json_serialisable(self):
        with patch.object(xreal_glasses, "find_library", return_value=None):
            result = probe()
        json.dumps(result.to_dict())  # must not raise

    def test_note_when_no_xreal_device_present(self):
        nodes = [_make_node(0x046D, 0xC534, manufacturer="Logitech")]
        lib = _FakeLib(_hidapi_exports(nodes))
        with patch.object(xreal_glasses, "load_library",
                          return_value=(lib, Path("/tmp/x.dll"))):
            result = probe()
        self.assertFalse(result.ok)
        self.assertEqual(result.interfaces and len(result.interfaces), 1)
        self.assertTrue(any("vendor id" in n for n in result.notes))

    def test_ok_when_xreal_device_present(self):
        nodes = [_make_node(XREAL_VENDOR_ID, 0x0436, interface=0),
                 _make_node(XREAL_VENDOR_ID, 0x0436, interface=1)]
        lib = _FakeLib(_hidapi_exports(nodes))
        with patch.object(xreal_glasses, "load_library",
                          return_value=(lib, Path("/tmp/x.dll"))):
            result = probe()
        self.assertTrue(result.ok)
        self.assertEqual(len(result.xreal_interfaces), 2)
        self.assertTrue(any("HID interfaces only" in n for n in result.notes))
        self.assertTrue(any("--usb" in n for n in result.notes))


class InterfaceDescribeTests(unittest.TestCase):
    def test_describe_includes_known_product_label(self):
        item = HidInterface(
            path="p", vendor_id=XREAL_VENDOR_ID, product_id=0x0436,
            serial_number="SN", release_number=1, manufacturer="XREAL",
            product="One Pro", usage_page=0xFF00, usage=1,
            interface_number=0, bus_type="usb",
        )
        text = item.describe()
        self.assertIn("3318:0436", text)
        self.assertIn("One Pro + Eye", text)


class ProbeCliTests(unittest.TestCase):
    def test_timeout_is_reported_not_raised(self):
        import subprocess as sp

        with patch.object(xreal_probe.subprocess, "run",
                          side_effect=sp.TimeoutExpired(cmd="x", timeout=1)):
            payload = xreal_probe.run_probe(timeout_s=1)
        self.assertFalse(payload["ok"])
        self.assertIn("hanging", payload["fatal"])

    def test_subprocess_crash_is_reported_not_raised(self):
        class _Done:
            returncode = -11
            stdout = ""
            stderr = "Segmentation fault\n"

        with patch.object(xreal_probe.subprocess, "run", return_value=_Done()):
            payload = xreal_probe.run_probe()
        self.assertFalse(payload["ok"])
        self.assertIn("Segmentation fault", payload["fatal"])

    def test_unparseable_output_is_reported(self):
        class _Done:
            returncode = 0
            stdout = "not json"
            stderr = ""

        with patch.object(xreal_probe.subprocess, "run", return_value=_Done()):
            payload = xreal_probe.run_probe()
        self.assertFalse(payload["ok"])
        self.assertIn("unparseable", payload["fatal"])

    def test_render_handles_fatal_payload(self):
        text = xreal_probe._render({"ok": False, "fatal": "boom"}, show_all=False)
        self.assertIn("FAILED", text)
        self.assertIn("boom", text)

    def test_render_lists_xreal_interfaces(self):
        payload = {
            "ok": True,
            "platform": "win32",
            "library": {"path": "x.dll", "loaded": True,
                        "hidapi_version": "0.14.0", "resolved": {}, "error": None},
            "interfaces": [],
            "xreal_interfaces": [{
                "path": "p", "vendor_id": 0x3318, "product_id": 0x0436,
                "serial_number": "SN1", "release_number": 1,
                "manufacturer": "XREAL", "product": "One Pro",
                "usage_page": 0xFF00, "usage": 1, "interface_number": 2,
                "bus_type": "usb",
            }],
            "notes": ["a note"],
            "elapsed_s": 0.1,
        }
        text = xreal_probe._render(payload, show_all=False)
        self.assertIn("3318:0436", text)
        self.assertIn("SN1", text)
        self.assertIn("a note", text)
        self.assertIn("OK", text)

    def test_exit_code_reflects_outcome(self):
        import contextlib
        import io

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            with patch.object(xreal_probe, "run_probe", return_value={"ok": False}):
                with patch.object(xreal_probe, "_log"):
                    self.assertEqual(xreal_probe.main([]), 1)
            with patch.object(xreal_probe, "run_probe", return_value={"ok": True}):
                with patch.object(xreal_probe, "_log"):
                    self.assertEqual(xreal_probe.main(["--json"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
