"""Bump / stamp / verify the EyeLab version. Policy: docs/VERSIONING.md.

    python tools/bump_version.py            # +1 BUILD (every commit), then stamp
    python tools/bump_version.py minor      # MINOR+1, BUILD=000
    python tools/bump_version.py major      # MAJOR+1, MINOR=0, BUILD=000
    python tools/bump_version.py sync       # re-stamp targets from the source
    python tools/bump_version.py check      # exit 1 if any target drifted
    python tools/bump_version.py show       # print version + commit prefix

The single source is ``python/eyelab_version.py``. This tool is the only thing
that should ever write a version number. Stdlib only.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "python" / "eyelab_version.py"

# Markdown files carrying <!-- eyelab-version -->X.Y.ZZZ<!-- /eyelab-version -->.
MARKER_FILES = ["README.md", "CHANGELOG.md"]
MARKER_RE = re.compile(r"(<!-- eyelab-version -->)(.*?)(<!-- /eyelab-version -->)")

# Unity player setting mirrored from the source (the Android APK's versionName).
UNITY_SETTINGS = "eyelab_xreal/ProjectSettings/ProjectSettings.asset"
UNITY_RE = re.compile(r"^(  bundleVersion: )(.*?)(\r?)$", re.MULTILINE)

# CONTEXT.md is NOT auto-stamped: its "Last updated" line records the build at
# which someone last verified the facts, so it must be written by hand.
CONTEXT_FILE = "CONTEXT.md"
CONTEXT_BUILD_RE = re.compile(r"Last updated:[^\n]*?\bbuild (\d+\.\d+\.\d{3})")

_SRC_RE = re.compile(r'^__version__ = "([^"]+)"(\r?)$', re.MULTILINE)
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d{3})$")


def parse_version(text):
    m = _VERSION_RE.match(text)
    if not m:
        raise ValueError(f"not a MAJOR.MINOR.BUILD version (BUILD = 3 digits): {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_version(major, minor, build):
    if not 0 <= build <= 999:
        raise ValueError(f"BUILD {build} does not fit in 3 digits; run 'minor' instead")
    return f"{major}.{minor}.{build:03d}"


def next_version(current, part="build"):
    major, minor, build = parse_version(current)
    if part == "build":
        return format_version(major, minor, build + 1)
    if part == "minor":
        return format_version(major, minor + 1, 0)
    if part == "major":
        return format_version(major + 1, 0, 0)
    raise ValueError(f"unknown part {part!r}")


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:  # newline="": keep EOLs byte-exact
        return f.read()


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def read_source_version(root=ROOT):
    text = _read(Path(root) / "python" / "eyelab_version.py")
    m = _SRC_RE.search(text)
    if not m:
        raise SystemExit('python/eyelab_version.py: no line `__version__ = "X.Y.ZZZ"` found')
    parse_version(m.group(1))
    return m.group(1)


def write_source_version(version, root=ROOT):
    path = Path(root) / "python" / "eyelab_version.py"
    text, n = _SRC_RE.subn(lambda m: f'__version__ = "{version}"{m.group(2)}', _read(path))
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one __version__ line, found {n}")
    _write(path, text)


def stamp_text(rel, text, version):
    """Return ``text`` with the version stamp for target ``rel`` set to ``version``."""
    if rel in MARKER_FILES:
        return MARKER_RE.sub(lambda m: f"{m.group(1)}{version}{m.group(3)}", text)
    if rel == UNITY_SETTINGS:
        return UNITY_RE.sub(lambda m: f"{m.group(1)}{version}{m.group(3)}", text)
    raise ValueError(rel)


def _count_stamps(rel, text):
    return len((UNITY_RE if rel == UNITY_SETTINGS else MARKER_RE).findall(text))


def targets():
    return MARKER_FILES + [UNITY_SETTINGS]


def sync(version, root=ROOT, dry_run=False):
    """Stamp every target. Returns the list of files that changed."""
    changed = []
    for rel in targets():
        path = Path(root) / rel
        old = _read(path)
        if _count_stamps(rel, old) == 0:
            raise SystemExit(f"{rel}: no version stamp found (marker deleted?)")
        new = stamp_text(rel, old, version)
        if new != old:
            changed.append(rel)
            if not dry_run:
                _write(path, new)
    return changed


def check(root=ROOT):
    """Return ``(problems, warnings)``; problems make the exit code non-zero."""
    root = Path(root)
    problems, warnings = [], []
    try:
        version = read_source_version(root)
    except (SystemExit, ValueError) as e:
        return [str(e)], []
    for rel in targets():
        text = _read(root / rel)
        if _count_stamps(rel, text) == 0:
            problems.append(f"{rel}: no version stamp found")
        elif stamp_text(rel, text, version) != text:
            problems.append(f"{rel}: stamp differs from source {version} (run 'sync')")
    m = CONTEXT_BUILD_RE.search(_read(root / CONTEXT_FILE))
    if not m:
        warnings.append(f"{CONTEXT_FILE}: 'Last updated' line has no 'build X.Y.ZZZ'")
    elif parse_version(m.group(1)) > parse_version(version):
        problems.append(f"{CONTEXT_FILE}: stamped build {m.group(1)} is ahead of source {version}")
    return problems, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="build",
                    choices=["build", "minor", "major", "sync", "check", "show"])
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args(argv)

    current = read_source_version()
    if args.action == "show":
        print(current)
        return 0
    if args.action == "check":
        problems, warnings = check()
        for w in warnings:
            print(f"warning: {w}")
        for p in problems:
            print(f"DRIFT: {p}")
        print("version stamps consistent" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    version = current if args.action == "sync" else next_version(current, args.action)
    if version != current:
        print(f"{current} -> {version}")
        if not args.dry_run:
            write_source_version(version)
    for rel in sync(version, dry_run=args.dry_run):
        print(f"stamped {rel}")
    if args.action != "sync":
        print(f'\ncommit message prefix:  git commit -m "{version} <what changed>"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
