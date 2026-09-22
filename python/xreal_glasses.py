#!/usr/bin/env python3
"""Read-only interrogation of XREAL glasses over USB HID.

Stage 0 of the "EyeLab without Nebula" work (see
``docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md``). It answers three questions and
deliberately nothing else:

1. Is ``libnr_glasses_api.dll`` present and loadable on this machine?
2. Which of its exports actually resolve?
3. What does the glasses' USB HID topology look like — vendor/product id,
   how many interfaces, serial number, usage pages?

**This module never calls an XREAL-proprietary function.** The only native
calls it makes are hidapi's, whose ABI is public, versioned and documented;
the DLL statically links hidapi 0.14.0 and re-exports it. Everything
XREAL-specific — reading firmware versions, glasses type, SKU, and above all
``NRBSPSetUsbConfigAll`` — has an undocumented signature, so calling it would
mean guessing at pointer arguments. That is how you crash a process, and in
the case of the USB-config setter it is how you write to a persistent u-boot
partition by accident. Those names are listed in :data:`MUTATING_EXPORTS` and
:data:`UNDOCUMENTED_READ_EXPORTS` so the boundary is explicit and testable,
not merely intended.

The DLL is a vendor binary with no redistribution licence and is not in this
repository. Stage it yourself; see ``vendor/README.md``. Search order:

1. ``$EYELAB_XREAL_DLL`` (full path to the file)
2. ``<repo>/vendor/xreal/win-x64/libnr_glasses_api.dll``
3. the OS loader's own search path

Windows-only in practice — the library is a PE — but everything here except
the load itself is platform-independent so the parsing is testable anywhere.

Nothing in this module prints or exits. ``xreal_probe.py`` owns the I/O, and
runs this in a child process so a hang in a native call cannot wedge its
caller. That matters because the same code is destined for the GUI, where a
blocking native call on the tkinter thread is the exact failure already
documented in ``docs/ar-freeze-diagnostics.md``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, Structure, c_char_p, c_int, c_ushort, c_wchar_p
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

LIBRARY_FILENAME = "libnr_glasses_api.dll"
ENV_OVERRIDE = "EYELAB_XREAL_DLL"

#: XREAL's USB vendor id. Recorded in ``CONTEXT.md`` from the Android
#: reverse-engineering work and corroborated by a literal in the DLL. Treated
#: as a hint for labelling only — enumeration is never filtered by it, because
#: a wrong guess here would silently hide the device we are looking for.
XREAL_VENDOR_ID = 0x3318

#: Product id reported for the One Pro + Eye on Android. Same caveat.
KNOWN_PRODUCT_IDS = {0x0436: "One Pro + Eye (Android UVC route)"}

#: hidapi functions this module calls. Public ABI, hidapi 0.14.0.
HIDAPI_EXPORTS = (
    "hid_init",
    "hid_exit",
    "hid_enumerate",
    "hid_free_enumeration",
    "hid_error",
    "hid_version_str",
)

#: Resolved and reported, never called: signatures are undocumented.
UNDOCUMENTED_READ_EXPORTS = (
    "GetCurrentGlass",
    "Get_Glasses_FW_Version",
    "Get_LIB_Version",
    "NRBSPGetCameraStatus",
    "NRBSPGetProperty",
    "NRBSPGetUsbConfigAll",
    "NROTAGetSDKGlassType",
    "NROTAGetSKUString",
    "xreal_get_glass_sn",
)

#: Resolved and reported, never called, and never to be called casually.
#: ``NRBSPSetUsbConfigAll`` is the UVC/NCM unlock; the rest write firmware.
MUTATING_EXPORTS = (
    "NRBSPSetUsbConfig",
    "NRBSPSetUsbConfigAll",
    "NROTARebootGlass",
    "NROTASetGlassToBoot",
    "NROTAStart",
    "rommode_upgrade",
    "set_clear_usrdata_part",
    "xreal_glasses_mcu_soc_upgrade",
    "xreal_glasses_upgrade",
    "xreal_start_ota",
)


class XrealLibraryError(RuntimeError):
    """The vendor DLL could not be found or loaded."""


class _HidDeviceInfo(Structure):
    """hidapi 0.14.0 ``struct hid_device_info``.

    Field order is load-bearing and matches ``hidapi.h`` at the 0.14.0 tag,
    where ``bus_type`` is last, after ``next``. ``ctypes`` computes the
    padding, so this is correct on both 64-bit Windows (2-byte ``wchar_t``)
    and Linux (4-byte), which is what lets the tests construct one in memory
    on either platform.
    """


_HidDeviceInfo._fields_ = [
    ("path", c_char_p),
    ("vendor_id", c_ushort),
    ("product_id", c_ushort),
    ("serial_number", c_wchar_p),
    ("release_number", c_ushort),
    ("manufacturer_string", c_wchar_p),
    ("product_string", c_wchar_p),
    ("usage_page", c_ushort),
    ("usage", c_ushort),
    ("interface_number", c_int),
    ("next", POINTER(_HidDeviceInfo)),
    ("bus_type", c_int),
]

_BUS_TYPES = {0: "unknown", 1: "usb", 2: "bluetooth", 3: "i2c", 4: "spi"}


@dataclass(frozen=True)
class HidInterface:
    """One HID interface as hidapi reports it."""

    path: str
    vendor_id: int
    product_id: int
    serial_number: Optional[str]
    release_number: int
    manufacturer: Optional[str]
    product: Optional[str]
    usage_page: int
    usage: int
    interface_number: int
    bus_type: str

    @property
    def is_xreal(self) -> bool:
        return self.vendor_id == XREAL_VENDOR_ID

    def describe(self) -> str:
        vid_pid = f"{self.vendor_id:04x}:{self.product_id:04x}"
        known = KNOWN_PRODUCT_IDS.get(self.product_id)
        name = " / ".join(p for p in (self.manufacturer, self.product) if p)
        bits = [vid_pid]
        if name:
            bits.append(name)
        bits.append(f"if={self.interface_number}")
        bits.append(f"usage={self.usage_page:#06x}:{self.usage:#06x}")
        bits.append(f"bus={self.bus_type}")
        if known:
            bits.append(f"[{known}]")
        return "  ".join(bits)


@dataclass
class LibraryReport:
    """What we learned about the DLL itself, without calling into XREAL code."""

    path: Optional[str] = None
    loaded: bool = False
    hidapi_version: Optional[str] = None
    resolved: dict[str, bool] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def missing_hidapi(self) -> list[str]:
        return [n for n in HIDAPI_EXPORTS if not self.resolved.get(n, False)]


@dataclass
class ProbeResult:
    """Everything a read-only probe can establish. Serialisable as-is."""

    platform: str
    library: LibraryReport
    interfaces: list[HidInterface] = field(default_factory=list)
    xreal_interfaces: list[HidInterface] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.library.loaded and bool(self.xreal_interfaces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "library": asdict(self.library),
            "interfaces": [asdict(i) for i in self.interfaces],
            "xreal_interfaces": [asdict(i) for i in self.xreal_interfaces],
            "notes": list(self.notes),
            "ok": self.ok,
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def candidate_library_paths() -> list[Path]:
    """Where to look for the DLL, in priority order."""
    paths: list[Path] = []
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        paths.append(Path(override))
    paths.append(repo_root() / "vendor" / "xreal" / "win-x64" / LIBRARY_FILENAME)
    return paths


def find_library() -> Optional[Path]:
    for candidate in candidate_library_paths():
        if candidate.is_file():
            return candidate
    return None


def load_library(path: Optional[Path] = None) -> tuple[Any, Path]:
    """Load the DLL. Raises :class:`XrealLibraryError` with a usable message."""
    resolved = path or find_library()
    if resolved is None:
        looked = "\n  ".join(str(p) for p in candidate_library_paths())
        raise XrealLibraryError(
            f"{LIBRARY_FILENAME} not found. Looked in:\n  {looked}\n"
            f"Stage it from an unpacked Nebula build (see vendor/README.md), "
            f"or set {ENV_OVERRIDE} to its full path."
        )
    if not resolved.is_file():
        raise XrealLibraryError(f"{resolved} is not a file")

    # The DLL resolves hid.dll and setupapi.dll from the system at runtime,
    # but adding its own directory keeps a co-located dependency findable if
    # a future tier stages one next to it.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(resolved.parent))
        except OSError:
            pass

    try:
        lib = ctypes.CDLL(str(resolved))
    except OSError as exc:
        hint = ""
        if sys.platform != "win32":
            hint = (
                " This is a Windows PE and cannot be loaded on "
                f"{sys.platform}; run the probe on the machine with the glasses."
            )
        raise XrealLibraryError(f"failed to load {resolved}: {exc}.{hint}") from exc
    return lib, resolved


def _resolve(lib: Any, name: str) -> Optional[Any]:
    try:
        return getattr(lib, name)
    except AttributeError:
        return None


def inspect_library(lib: Any, path: Path) -> LibraryReport:
    """Report which exports resolve. Calls only ``hid_version_str``."""
    report = LibraryReport(path=str(path), loaded=True)
    for name in HIDAPI_EXPORTS + UNDOCUMENTED_READ_EXPORTS + MUTATING_EXPORTS:
        report.resolved[name] = _resolve(lib, name) is not None

    version = _resolve(lib, "hid_version_str")
    if version is not None:
        try:
            version.restype = c_char_p
            version.argtypes = []
            raw = version()
            if raw:
                report.hidapi_version = raw.decode("ascii", "replace")
        except Exception as exc:  # pragma: no cover - defensive
            report.error = f"hid_version_str failed: {exc}"
    return report


def _bind_hidapi(lib: Any) -> dict[str, Any]:
    """Apply documented hidapi 0.14.0 signatures."""
    fns: dict[str, Any] = {}
    for name in HIDAPI_EXPORTS:
        fn = _resolve(lib, name)
        if fn is None:
            raise XrealLibraryError(
                f"{name} is not exported by this DLL — it may not be the "
                f"hidapi-linked build this module expects."
            )
        fns[name] = fn

    fns["hid_init"].restype = c_int
    fns["hid_init"].argtypes = []
    fns["hid_exit"].restype = c_int
    fns["hid_exit"].argtypes = []
    fns["hid_enumerate"].restype = POINTER(_HidDeviceInfo)
    fns["hid_enumerate"].argtypes = [c_ushort, c_ushort]
    fns["hid_free_enumeration"].restype = None
    fns["hid_free_enumeration"].argtypes = [POINTER(_HidDeviceInfo)]
    return fns


def _to_interface(node: _HidDeviceInfo) -> HidInterface:
    path = node.path.decode("utf-8", "replace") if node.path else ""
    return HidInterface(
        path=path,
        vendor_id=node.vendor_id,
        product_id=node.product_id,
        serial_number=node.serial_number or None,
        release_number=node.release_number,
        manufacturer=node.manufacturer_string or None,
        product=node.product_string or None,
        usage_page=node.usage_page,
        usage=node.usage,
        interface_number=node.interface_number,
        bus_type=_BUS_TYPES.get(node.bus_type, str(node.bus_type)),
    )


def enumerate_hid(lib: Any) -> list[HidInterface]:
    """Enumerate every HID device on the machine.

    Deliberately unfiltered: passing ``(0, 0)`` to ``hid_enumerate`` means a
    wrong assumption about the vendor id cannot hide the glasses. Filtering
    happens afterwards, in Python, where it is visible.

    Enumeration does not open any device, so it cannot take an interface away
    from another application that has one open.
    """
    fns = _bind_hidapi(lib)
    if fns["hid_init"]() != 0:
        raise XrealLibraryError("hid_init() failed")

    interfaces: list[HidInterface] = []
    head = None
    try:
        head = fns["hid_enumerate"](0, 0)
        node_ptr = head
        seen = 0
        while node_ptr:
            interfaces.append(_to_interface(node_ptr.contents))
            node_ptr = node_ptr.contents.next
            seen += 1
            if seen > 4096:  # a corrupt list must not spin forever
                raise XrealLibraryError("hid_enumerate returned a cyclic list")
    finally:
        if head:
            fns["hid_free_enumeration"](head)
        fns["hid_exit"]()
    return interfaces


def probe(path: Optional[Path] = None) -> ProbeResult:
    """Run the whole read-only probe. Never raises for expected failures."""
    result = ProbeResult(platform=sys.platform, library=LibraryReport())

    try:
        lib, resolved = load_library(path)
    except XrealLibraryError as exc:
        result.library.error = str(exc)
        result.library.path = str(find_library() or "")
        return result

    result.library = inspect_library(lib, resolved)

    try:
        result.interfaces = enumerate_hid(lib)
    except XrealLibraryError as exc:
        result.library.error = str(exc)
        return result

    result.xreal_interfaces = [i for i in result.interfaces if i.is_xreal]

    if not result.interfaces:
        result.notes.append("No HID devices at all — unusual; check permissions.")
    elif not result.xreal_interfaces:
        result.notes.append(
            f"No device with vendor id {XREAL_VENDOR_ID:#06x}. The glasses may "
            f"be unplugged, in DP-only mode, or use a different vendor id — "
            f"check the full interface list below before assuming the former."
        )
    else:
        count = len(result.xreal_interfaces)
        result.notes.append(
            f"{count} XREAL HID interface(s). Note this counts HID interfaces "
            f"only -- hid_enumerate cannot see UVC, audio or DP interfaces, so "
            f"it is not comparable to the 13-before / 17-after *total USB* "
            f"interface counts recorded for the Android route. Run with --usb "
            f"for the full picture, and treat 'a UVC device appears' as the "
            f"signal that an unlock worked."
        )

    missing = result.library.missing_hidapi
    if missing:
        result.notes.append(f"Missing expected hidapi exports: {', '.join(missing)}")
    return result
