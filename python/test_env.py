#!/usr/bin/env python3
"""
EyeLab Python environment verification script.

Checks that all required packages are installed and that the cv2.aruco
module is functional (ArUco detection requires OpenCV >=4.8).

Usage:
    python test_env.py

Exit code 0 if all checks pass, 1 if any fail.
"""

import sys


def _ok(msg: str) -> None:
    print(f"  \u2713  {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717  {msg}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ── Individual checks ─────────────────────────────────────────────────────────

def check_numpy() -> bool:
    try:
        import numpy as np
        arr = np.zeros((3, 3), dtype=np.float64)
        _ok(f"numpy {np.__version__}  (array shape: {arr.shape})")
        return True
    except Exception as e:
        _fail(f"numpy import failed: {e}")
        return False


def check_scipy() -> bool:
    try:
        import scipy
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler("z", 45, degrees=True)
        _ok(f"scipy {scipy.__version__}  (Rotation.from_euler OK)")
        return True
    except Exception as e:
        _fail(f"scipy import failed: {e}")
        return False


def check_opencv() -> bool:
    try:
        import cv2
        _ok(f"opencv-python {cv2.__version__}")
    except Exception as e:
        _fail(f"opencv-python import failed: {e}")
        return False

    # Verify ArUco sub-module
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        det_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, det_params)
        _ok(f"cv2.aruco  (ArucoDetector instantiated, DICT_4X4_50)")
    except AttributeError as e:
        _fail(
            f"cv2.aruco not available: {e}\n"
            f"         Install opencv-contrib-python>=4.8 for the aruco module."
        )
        return False
    except Exception as e:
        _fail(f"cv2.aruco check failed: {e}")
        return False

    # Verify CharucoBoard (API changed in 4.8)
    try:
        board = cv2.aruco.CharucoBoard(
            (5, 7),
            squareLength=0.025,
            markerLength=0.019,
            dictionary=aruco_dict,
        )
        _ok(f"cv2.aruco.CharucoBoard  (5×7, squareLength=25 mm)")
    except Exception as e:
        _fail(f"CharucoBoard constructor failed: {e}")
        return False

    return True


def check_pyuff() -> bool:
    try:
        import pyuff
        _ok(f"pyuff {getattr(pyuff, '__version__', '(version unknown)')}")
        return True
    except Exception as e:
        _fail(f"pyuff import failed: {e}")
        return False


def check_matplotlib() -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")        # headless — no display required for this test
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plt.close(fig)
        _ok(f"matplotlib {matplotlib.__version__}  (non-interactive backend OK)")
        return True
    except Exception as e:
        _fail(f"matplotlib check failed: {e}")
        return False


def check_pandas() -> bool:
    try:
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        _ok(f"pandas {pd.__version__}  (DataFrame OK)")
        return True
    except Exception as e:
        _fail(f"pandas import failed: {e}")
        return False


def check_yaml() -> bool:
    try:
        import yaml
        data = yaml.safe_load("key: value")
        assert data == {"key": "value"}
        _ok(f"pyyaml {yaml.__version__}  (safe_load OK)")
        return True
    except Exception as e:
        _fail(f"pyyaml check failed: {e}")
        return False


def check_open3d() -> bool:
    """open3d is optional — warn but do not count as failure."""
    try:
        import open3d as o3d
        _ok(f"open3d {o3d.__version__}  (optional — available)")
        return True
    except Exception:
        print(f"  -  open3d not installed (optional — needed for T2.3 marker registration)")
        return True   # not counted as failure


# ── Camera smoke test ─────────────────────────────────────────────────────────

def check_camera(camera_index: int = 0) -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            _fail(f"Camera {camera_index} could not be opened.")
            return False
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            _fail(f"Camera {camera_index} opened but returned no frame.")
            return False
        h, w = frame.shape[:2]
        _ok(f"Camera {camera_index}: {w}×{h} frame captured")
        return True
    except Exception as e:
        _fail(f"Camera check failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 52)
    print("  EyeLab Python Environment Verification")
    print(f"  Python {sys.version.split()[0]}")
    print("=" * 52)

    _section("Required packages")
    results = {
        "numpy":      check_numpy(),
        "scipy":      check_scipy(),
        "opencv":     check_opencv(),
        "pyuff":      check_pyuff(),
        "matplotlib": check_matplotlib(),
        "pandas":     check_pandas(),
        "pyyaml":     check_yaml(),
    }

    _section("Optional packages")
    check_open3d()

    _section("Hardware")
    cam_ok = check_camera(0)
    results["camera"] = cam_ok

    # Summary
    print(f"\n{'=' * 52}")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    ok_str = "ALL PASSED" if passed == total else f"{passed}/{total} PASSED"
    print(f"  Results: {ok_str}")
    print("=" * 52)

    if passed < total:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  Failed checks: {', '.join(failed)}", file=sys.stderr)
        print(f"  Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
