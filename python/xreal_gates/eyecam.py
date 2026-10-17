"""The XREAL Eye camera model used by the gates: calibrated, or the vendor claim.

The Eye delivers 1280x720 YUV with the rows stored bottom-up (§15); every gate
snapshot and every frame the pose service receives is flipped to upright before
use, and all intrinsics here are for the upright image. The vendor reports
fx, fy, cx, cy (inside XR, ``run.json`` ``device_xr.rgb_camera_intrinsic``) and
no distortion. Its cy is taken to refer to the stored, bottom-up rows, so the
upright cy is ``height - 1 - cy`` (an assumption that moves cy by 0.4 px).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

EYE_SIZE = (1280, 720)

#: Vendor intrinsics observed on the Surface (14-16 Oct, ``device_xr``), stored-row convention.
VENDOR_FX_FY_CX_CY = (607.2307, 607.396057, 638.263855, 359.813721)

PYTHON_DIR = Path(__file__).resolve().parents[1]
#: A calibration committed with the repo (copied here from a calibration run).
REPO_CALIBRATION = PYTHON_DIR / "config" / "cameras" / "xreal_eye_1280x720.yaml"


@dataclass
class EyeCamera:
    K: np.ndarray
    dist: np.ndarray
    size: Tuple[int, int]
    source: str            # "calibration" or "vendor"
    path: str = ""
    rms_px: Optional[float] = None

    def describe(self) -> str:
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        d = ", ".join(f"{v:.4f}" for v in np.ravel(self.dist)[:5])
        extra = f" rms {self.rms_px:.3f} px" if self.rms_px is not None else ""
        return f"{self.source} {self.path} fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f} dist [{d}]{extra}"


def vendor_camera(fx_fy_cx_cy: Optional[Sequence[float]] = None, size=EYE_SIZE) -> EyeCamera:
    fx, fy, cx, cy = fx_fy_cx_cy or VENDOR_FX_FY_CX_CY
    K = np.array([[fx, 0.0, cx], [0.0, fy, size[1] - 1 - cy], [0.0, 0.0, 1.0]])
    return EyeCamera(K, np.zeros(5), tuple(size), "vendor", "SDK intrinsics, no distortion")


def load_camera(path: Optional[Path], fallback_vendor: Optional[Sequence[float]] = None) -> EyeCamera:
    """The calibration at ``path`` (or the repo one), else the vendor claim.

    A calibration taken at another resolution is refused: intrinsics in the
    wrong pixel units scale every pose while reprojecting perfectly.
    """
    from calibrate import read_calibration

    for candidate in (path, REPO_CALIBRATION):
        if candidate is None or not Path(candidate).exists():
            continue
        data = read_calibration(str(candidate))
        if data.image_size is not None and tuple(data.image_size) != EYE_SIZE:
            raise ValueError(f"{candidate}: calibrated at {data.image_size}, the Eye delivers {EYE_SIZE}")
        return EyeCamera(np.asarray(data.camera_matrix, float), np.asarray(data.dist_coeffs, float).ravel(),
                         EYE_SIZE, "calibration", str(candidate), data.rms_error)
    return vendor_camera(fallback_vendor)


def read_pgm_upright(path: Path) -> np.ndarray:
    """Gate snapshots are already written upright (rows reversed at save time)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img
