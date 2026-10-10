"""commit-msg guard: every commit subject starts with the build it bumped to.

    0.2.014 geometry bugfix

Enforces (policy: docs/VERSIONING.md):
  1. subject = "<MAJOR.MINOR.BUILD> <text>"
  2. the prefix equals the version in the *staged* python/eyelab_version.py
  3. that version differs from HEAD's (the commit really bumped)
  4. no earlier commit on this branch already used the prefix (two PCs / two
     agents bumping to the same number)

Exempt: merges, reverts, fixup!/squash! commits. Bypass for a one-off (amend,
history surgery): `git commit --no-verify`, or EYELAB_ALLOW_SAME_VERSION=1 to
waive rule 3 only. Install once per clone: tools/githooks/README.md.
"""

import os
import re
import subprocess
import sys

VERSION_FILE = "python/eyelab_version.py"
_SRC_RE = re.compile(r'^__version__ = "(\d+\.\d+\.\d{3})"', re.MULTILINE)
_EXEMPT = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _version_in(blob):
    m = _SRC_RE.search(blob or "")
    return m.group(1) if m else None


def validate(subject, staged, head, used_subjects, allow_same=False):
    """Return an error string, or None when the subject is acceptable."""
    if subject.startswith(_EXEMPT):
        return None
    m = re.match(r"^(\d+\.\d+\.\d{3}) \S", subject)
    if not m:
        return 'subject must start with the build: "0.2.014 what changed"'
    prefix = m.group(1)
    if staged is None:
        return f"cannot read the staged {VERSION_FILE}"
    if prefix != staged:
        return f"prefix {prefix} != staged version {staged} (run tools/bump_version.py)"
    if head is not None and staged == head and not allow_same:
        return f"version {staged} was not bumped since HEAD (run tools/bump_version.py)"
    if any(s.startswith(prefix + " ") for s in used_subjects):
        return f"build {prefix} was already used by an earlier commit; pull, then bump again"
    return None


def main(argv):
    with open(argv[1], encoding="utf-8") as f:
        subject = next((ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#")), "")
    staged = _version_in(_git("show", f":{VERSION_FILE}"))
    head = _version_in(_git("show", f"HEAD:{VERSION_FILE}"))
    used = (_git("log", "-n", "1000", "--format=%s", "HEAD") or "").splitlines()
    if head is not None and subject.startswith(head + " ") and head == staged:
        used = used[1:]  # amending: HEAD's own subject is not a duplicate
    err = validate(subject, staged, head, used, os.environ.get("EYELAB_ALLOW_SAME_VERSION") == "1")
    if err:
        sys.stderr.write(f"\nversion check failed: {err}\n  subject: {subject!r}\n\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
