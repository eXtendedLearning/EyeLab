"""Shared config tree: profiles, active selection, legacy migration."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import eyelab_paths as P


def _patched(root: Path):
    cfg = root / "config"
    cams = cfg / "cameras"
    return mock.patch.multiple(
        P, CONFIG_DIR=cfg, CAMERAS_DIR=cams, CAMERA_HISTORY_DIR=cams / "history",
        MARKER_CONFIGS_DIR=cfg / "markers", GEOMETRY_DIR=cfg / "geometry",
        TARGETS_DIR=cfg / "targets", ACTIVE_FILE=cfg / "active.json",
        DEFAULT_MARKER_CONFIG=cfg / "markers" / "marker_config.json",
        VALIDATION_RUNS_DIR=root / "validation_runs")


class TestPaths(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(P.slug(" Surface Rear "), "surface_rear")
        self.assertEqual(P.slug("LAGANN-0526_cam0"), "lagann-0526_cam0")
        with self.assertRaises(ValueError):
            P.slug("  //  ")

    def test_active_selection_round_trip(self):
        with tempfile.TemporaryDirectory() as d, _patched(Path(d)):
            P.ensure_dirs()
            self.assertEqual(P.active_marker_config(), P.DEFAULT_MARKER_CONFIG)
            self.assertIsNone(P.active_geometry())
            m = P.MARKER_CONFIGS_DIR / "plate_A.markers.json"
            P.save_active(camera="Desktop C920", markers=m, geometry=P.GEOMETRY_DIR / "lastra.unv")
            raw = P.ACTIVE_FILE.read_text(encoding="utf-8")
            self.assertIn('"markers/plate_A.markers.json"', raw)      # relative, POSIX
            self.assertEqual(P.active_camera(), "desktop_c920")
            self.assertEqual(P.active_calibration_file(), P.CAMERAS_DIR / "desktop_c920.yaml")
            self.assertEqual(P.active_marker_config(), m)
            outside = Path(d) / "elsewhere" / "g.unv"
            P.save_active(geometry=outside)
            self.assertEqual(P.active_geometry().resolve(), outside.resolve())
            P.save_active(geometry=None)
            self.assertIsNone(P.active_geometry())

    def test_single_profile_is_used_without_selection(self):
        with tempfile.TemporaryDirectory() as d, _patched(Path(d)):
            P.ensure_dirs()
            (P.CAMERAS_DIR / "surface_rear.yaml").write_text("x")
            self.assertEqual(P.active_camera(), "surface_rear")
            (P.CAMERAS_DIR / "desktop.yaml").write_text("x")
            self.assertEqual(P.active_camera(), P.default_camera_name())

    def test_backup_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d, _patched(Path(d)):
            P.ensure_dirs()
            f = P.camera_file("cam")
            self.assertIsNone(P.backup_calibration(f))
            f.write_text("v1")
            b1 = P.backup_calibration(f)
            b2 = P.backup_calibration(f)
            self.assertNotEqual(b1, b2)
            self.assertEqual(b1.read_text(), "v1")

    def test_legacy_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d, _patched(Path(d)):
            P.CONFIG_DIR.mkdir(parents=True)
            (P.CONFIG_DIR / "camera_params.yaml").write_text("cal")
            (P.CONFIG_DIR / "marker_config.json").write_text("{}")
            old = P.VALIDATION_RUNS_DIR / "calibration_backups"
            old.mkdir(parents=True)
            (old / "camera_params_1.yaml").write_text("b")
            done = P.migrate_legacy_layout()
            self.assertEqual(len(done), 3)
            self.assertEqual(P.active_calibration_file().read_text(), "cal")
            self.assertTrue((P.MARKER_CONFIGS_DIR / "marker_config.json").exists())
            self.assertTrue((P.CAMERA_HISTORY_DIR / "camera_params_1.yaml").exists())
            self.assertEqual(P.migrate_legacy_layout(), [])


if __name__ == "__main__":
    unittest.main()
