"""Version scheme: one source, no drifting copies. Policy: docs/VERSIONING.md."""

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import eyelab_version

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bump = _load_tool("bump_version")
commit_msg = _load_tool("check_commit_msg")


class SourceTests(unittest.TestCase):
    def test_source_has_scheme_shape(self):
        self.assertRegex(eyelab_version.__version__, r"^\d+\.\d+\.\d{3}$")
        self.assertEqual(eyelab_version.format_version(*eyelab_version.VERSION_TUPLE),
                         eyelab_version.__version__)
        self.assertEqual(eyelab_version.VERSION_STRING, f"EyeLab v{eyelab_version.__version__}")

    def test_package_version_has_no_leading_zero(self):
        self.assertNotRegex(eyelab_version.PACKAGE_VERSION, r"(^|\.)0\d")

    def test_rejects_other_shapes(self):
        for bad in ("0.2.1", "0.2", "0.2.0001", "v0.2.001", "0.2.001-dev", ""):
            with self.assertRaises(ValueError, msg=bad):
                eyelab_version.parse_version(bad)
        with self.assertRaises(ValueError):
            eyelab_version.format_version(0, 2, 1000)

    def test_cli_prints_the_source(self):
        out = subprocess.run([sys.executable, str(ROOT / "python" / "eyelab_version.py"), "--string"],
                             capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, eyelab_version.VERSION_STRING)

    def test_tool_reads_same_version_as_module(self):
        self.assertEqual(bump.read_source_version(), eyelab_version.__version__)

    def test_tool_reads_source_with_lf_and_crlf(self):
        for newline in (b"\n", b"\r\n"):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "python" / "eyelab_version.py"
                source.parent.mkdir()
                source.write_bytes(b'__version__ = "0.2.009"' + newline)
                self.assertEqual(bump.read_source_version(root), "0.2.009")

    def test_tool_writes_source_preserving_lf_and_crlf(self):
        for newline in (b"\n", b"\r\n"):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "python" / "eyelab_version.py"
                source.parent.mkdir()
                original = newline.join((b"# version source", b'__version__ = "0.2.009"', b""))
                source.write_bytes(original)
                bump.write_source_version("0.2.010", root)
                self.assertEqual(source.read_bytes(), original.replace(b"0.2.009", b"0.2.010"))


class BumpTests(unittest.TestCase):
    def test_build_minor_major(self):
        self.assertEqual(bump.next_version("0.2.009"), "0.2.010")
        self.assertEqual(bump.next_version("0.2.099"), "0.2.100")
        self.assertEqual(bump.next_version("0.2.047", "minor"), "0.3.000")
        self.assertEqual(bump.next_version("0.9.999", "major"), "1.0.000")

    def test_build_overflow_forces_a_minor(self):
        with self.assertRaises(ValueError):
            bump.next_version("0.2.999")


class DriftTests(unittest.TestCase):
    def test_all_stamps_match_source(self):
        problems, _ = bump.check()
        self.assertEqual(problems, [], "run: python tools/bump_version.py sync")

    def test_launchers_read_the_source_and_hardcode_nothing(self):
        for name in ("run_eyelab.bat", "validation.bat"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("eyelab_version.py", text, name)
            self.assertNotRegex(text, r"[Vv]\d+\.\d+\.\d+", f"{name} hardcodes a version")

    def test_python_files_do_not_hardcode_the_version(self):
        needle = eyelab_version.__version__
        for path in (ROOT / "python").glob("*.py"):
            if path.name in ("eyelab_version.py", Path(__file__).name):
                continue
            self.assertNotIn(f'"{needle}"', path.read_text(encoding="utf-8"), path.name)

    def test_agents_md_does_not_state_a_version(self):
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        stated = set(re.findall(r"\bv?\d+\.\d+\.\d{3}\b", text))
        self.assertLessEqual(stated, {"0.2.014", "v0.3.000", "0.3.000"}, "AGENTS.md carries a live version")


class CommitMessageTests(unittest.TestCase):
    ok = staticmethod(lambda subject, staged="0.2.014", head="0.2.013", used=(), **kw:
                      commit_msg.validate(subject, staged, head, list(used), **kw))

    def test_accepts_well_formed(self):
        self.assertIsNone(self.ok("0.2.014 geometry bugfix"))

    def test_rejects_missing_or_wrong_prefix(self):
        self.assertIn("must start", self.ok("geometry bugfix"))
        self.assertIn("!=", self.ok("0.2.015 geometry bugfix"))

    def test_rejects_unbumped_and_duplicate(self):
        self.assertIn("not bumped", self.ok("0.2.013 x", staged="0.2.013"))
        self.assertIsNone(self.ok("0.2.013 x", staged="0.2.013", allow_same=True))
        self.assertIn("already used", self.ok("0.2.014 x", used=["0.2.014 earlier commit"]))

    def test_exempt_subjects(self):
        for s in ("Merge branch 'main'", "Revert \"0.2.014 x\"", "fixup! 0.2.014 x"):
            self.assertIsNone(self.ok(s), s)

    def test_first_commit_without_head(self):
        self.assertIsNone(self.ok("0.2.014 x", head=None))


if __name__ == "__main__":
    unittest.main()
