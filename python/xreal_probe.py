#!/usr/bin/env python3
"""Read-only probe of the XREAL glasses. Stage 0 of the Nebula-independence work.

Run it with the glasses plugged in:

    python python/xreal_probe.py
    python python/xreal_probe.py --json
    python python/xreal_probe.py --all          # every HID device, not just XREAL
    python python/xreal_probe.py --usb          # + every USB interface, via Windows PnP

It loads ``libnr_glasses_api.dll``, reports which exports resolve, and
enumerates USB HID. It opens nothing, writes nothing to the glasses, and
calls no XREAL-proprietary function — see ``xreal_glasses`` for why.

The native work runs in a **child process**. A native call that hangs or
segfaults then costs one subprocess rather than the whole session, and the
timeout is enforceable. This is deliberate: the same code path is headed for
the GUI, and a blocking call on the tkinter thread is precisely the failure
recorded in ``docs/ar-freeze-diagnostics.md``.

Every run appends a JSON line to ``python/.logs/xreal_probe.jsonl`` so the
before/after states of an experiment can be diffed later.

Exit status: 0 if the library loaded and at least one XREAL interface was
found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from native_isolation import append_jsonl, emit, run_json_worker

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / ".logs" / "xreal_probe.jsonl"
DEFAULT_TIMEOUT_S = 30.0

_WORKER_FLAG = "--_worker"


def _run_worker() -> int:
    """Child-process entry point: probe, print JSON on stdout, exit."""
    sys.path.insert(0, str(HERE))
    from xreal_glasses import probe

    emit(probe().to_dict())
    return 0


def run_probe(timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Run the probe out-of-process. Always returns a dict, never raises."""
    return run_json_worker(
        [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
        timeout_s,
        label="probe",
    )


USB_QUERY = (
    "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_3318*' } | "
    "Select-Object Status,Class,FriendlyName,InstanceId | "
    "Sort-Object InstanceId | ConvertTo-Json -Compress"
)


def enumerate_usb_windows(timeout_s: float = 20.0) -> dict:
    """List every USB interface the OS attributes to the glasses.

    ``hid_enumerate`` sees HID interfaces only, so it cannot tell you whether
    a UVC (camera) interface exists -- which is the actual signal that a USB
    function unlock worked. This asks Windows PnP instead.

    Read-only: ``Get-PnpDevice`` queries; it does not enable, disable or
    reconfigure anything. Windows-only; returns a ``supported: False`` payload
    elsewhere.
    """
    if sys.platform != "win32":
        return {"supported": False, "reason": f"Windows only (platform={sys.platform})"}

    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", USB_QUERY],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"supported": False, "reason": f"PowerShell unavailable: {exc}"}

    raw = (completed.stdout or "").strip()
    if completed.returncode != 0 or not raw:
        tail = (completed.stderr or "").strip().splitlines()
        return {
            "supported": True,
            "error": tail[-1] if tail else f"exit code {completed.returncode}",
            "devices": [],
        }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"supported": True, "error": f"unparseable: {exc}", "devices": []}

    if isinstance(parsed, dict):  # ConvertTo-Json collapses a single result
        parsed = [parsed]

    devices = []
    for entry in parsed:
        instance = entry.get("InstanceId") or ""
        marker = "&MI_"
        interface = None
        if marker in instance.upper():
            idx = instance.upper().index(marker) + len(marker)
            digits = instance[idx : idx + 2]
            try:
                interface = int(digits, 16)
            except ValueError:
                interface = None
        devices.append(
            {
                "status": entry.get("Status"),
                "device_class": entry.get("Class"),
                "name": entry.get("FriendlyName"),
                "instance_id": instance,
                "interface_number": interface,
            }
        )

    classes = sorted({d["device_class"] for d in devices if d["device_class"]})
    interfaces = sorted({d["interface_number"] for d in devices
                         if d["interface_number"] is not None})
    return {
        "supported": True,
        "devices": devices,
        "classes": classes,
        "interface_numbers": interfaces,
        "has_camera_class": any(
            (d["device_class"] or "").lower() in {"camera", "image", "media"}
            for d in devices
        ),
    }


def _log(payload: dict) -> None:
    append_jsonl(LOG_PATH, payload)  # best-effort; never fails a probe


def _render(payload: dict, show_all: bool) -> str:
    lines: list[str] = ["XREAL glasses — read-only probe", "=" * 34, ""]

    if payload.get("fatal"):
        lines += [f"FAILED: {payload['fatal']}", ""]
        return "\n".join(lines)

    lib = payload.get("library") or {}
    lines.append(f"platform          {payload.get('platform', '?')}")
    lines.append(f"library           {lib.get('path') or '(not found)'}")
    lines.append(f"loaded            {'yes' if lib.get('loaded') else 'NO'}")
    if lib.get("hidapi_version"):
        lines.append(f"hidapi            {lib['hidapi_version']}")
    if lib.get("error"):
        lines.append(f"error             {lib['error']}")
    lines.append("")

    resolved = lib.get("resolved") or {}
    if resolved:
        present = sum(1 for v in resolved.values() if v)
        lines.append(f"exports resolved  {present}/{len(resolved)}")
        absent = sorted(n for n, v in resolved.items() if not v)
        if absent:
            lines.append(f"  not present:    {', '.join(absent)}")
        lines.append("")

    xreal = payload.get("xreal_interfaces") or []
    every = payload.get("interfaces") or []
    lines.append(f"HID devices seen  {len(every)} total, {len(xreal)} XREAL")
    lines.append("")

    shown = every if show_all else xreal
    if shown:
        label = "all HID interfaces" if show_all else "XREAL HID interfaces"
        lines.append(f"{label}:")
        for item in shown:
            vid_pid = f"{item['vendor_id']:04x}:{item['product_id']:04x}"
            name = " / ".join(
                p for p in (item.get("manufacturer"), item.get("product")) if p
            )
            lines.append(
                f"  {vid_pid}  if={item['interface_number']:<3} "
                f"usage={item['usage_page']:#06x}:{item['usage']:#06x}  "
                f"bus={item['bus_type']:<9} {name}"
            )
            if item.get("serial_number"):
                lines.append(f"      serial  {item['serial_number']}")
            if item.get("path"):
                lines.append(f"      path    {item['path']}")
        lines.append("")
    elif not show_all:
        lines.append("No XREAL interfaces. Re-run with --all to see every device.")
        lines.append("")

    for note in payload.get("notes") or []:
        lines.append(f"note: {note}")
    if payload.get("notes"):
        lines.append("")

    usb = payload.get("usb")
    if usb:
        if not usb.get("supported"):
            lines.append(f"usb enumeration   skipped: {usb.get('reason')}")
        elif usb.get("error"):
            lines.append(f"usb enumeration   failed: {usb['error']}")
        else:
            devices = usb.get("devices") or []
            nums = usb.get("interface_numbers") or []
            lines.append(
                f"USB interfaces    {len(devices)} PnP node(s); "
                f"MI_ numbers seen: "
                + (", ".join(f"{n:02x}" for n in nums) if nums else "none")
            )
            lines.append(f"  classes:        {', '.join(usb.get('classes') or []) or '-'}")
            lines.append(
                f"  camera present: {'YES' if usb.get('has_camera_class') else 'no'}"
            )
            for d in devices:
                mi = (f"MI_{d['interface_number']:02x}"
                      if d["interface_number"] is not None else "-    ")
                lines.append(
                    f"    {mi}  {str(d.get('device_class') or '?'):<12} "
                    f"{str(d.get('status') or '?'):<8} {d.get('name') or ''}"
                )
        lines.append("")

    lines.append(f"verdict           {'OK' if payload.get('ok') else 'incomplete'}")
    lines.append(f"elapsed           {payload.get('elapsed_s', '?')}s")
    lines.append(f"logged to         {LOG_PATH}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _WORKER_FLAG in argv:
        return _run_worker()

    parser = argparse.ArgumentParser(
        description="Read-only probe of XREAL glasses over USB HID."
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--all", action="store_true", help="list every HID device, not just XREAL"
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds"
    )
    parser.add_argument(
        "--usb",
        action="store_true",
        help="also enumerate every USB interface via Windows PnP (read-only)",
    )
    args = parser.parse_args(argv)

    payload = run_probe(args.timeout)
    if args.usb:
        payload["usb"] = enumerate_usb_windows()
    _log(payload)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render(payload, show_all=args.all))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
