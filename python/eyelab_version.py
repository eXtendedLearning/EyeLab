"""Single source of truth for the EyeLab version.

Scheme ``MAJOR.MINOR.BUILD`` (e.g. ``0.2.014``), BUILD zero-padded to 3 digits.
BUILD is a development counter: +1 for every commit, reset to ``000`` when MINOR
changes. Full policy: ``docs/VERSIONING.md``.

Bump with ``python tools/bump_version.py`` -- never edit ``__version__`` by hand
and never copy the number into another file. Docs, launchers and the Unity
project are stamped from here by that tool, and ``test_version_scheme.py`` fails
when any of them drifts.

Stdlib only: the ``.bat`` launchers run this file directly
(``python python/eyelab_version.py``) before the venv exists.
"""

import re
import sys

__version__ = "0.3.000"

APP_NAME = "EyeLab"

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d{3})$")


def parse_version(text):
    """``"0.2.014"`` -> ``(0, 2, 14)``. Raises ValueError on any other shape."""
    m = _VERSION_RE.match(text)
    if not m:
        raise ValueError(f"not a MAJOR.MINOR.BUILD version (BUILD = 3 digits): {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_version(major, minor, build):
    """``(0, 2, 14)`` -> ``"0.2.014"``. BUILD must fit in 3 digits."""
    if not 0 <= build <= 999:
        raise ValueError(f"BUILD {build} does not fit in 3 digits; bump MINOR instead")
    return f"{major}.{minor}.{build:03d}"


VERSION_TUPLE = parse_version(__version__)
VERSION_STRING = f"{APP_NAME} v{__version__}"

# Same number without the zero padding (``0.2.14``). Use this -- not
# ``__version__`` -- anywhere a packaging tool parses the version: PEP 440
# silently normalises ``0.2.014`` to ``0.2.14`` and SemVer forbids leading zeros.
PACKAGE_VERSION = ".".join(str(n) for n in VERSION_TUPLE)

if __name__ == "__main__":
    # Used by the .bat launchers: python eyelab_version.py [--string|--package]
    flag = sys.argv[1] if len(sys.argv) > 1 else ""
    if flag == "--string":
        print(VERSION_STRING)
    elif flag == "--package":
        print(PACKAGE_VERSION)
    elif flag == "":
        print(__version__)
    else:
        sys.exit("usage: eyelab_version.py [--string | --package]")
