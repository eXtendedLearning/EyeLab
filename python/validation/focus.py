"""Interactive focus finder: pick one fixed focus value for calibration and capture."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

HELP = "A autofocus on/off | L lock here | +/- step | ENTER accept | Q cancel"


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian over the central half of the image."""
    h, w = gray.shape[:2]
    roi = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def find_focus(camera_index: int, step: float = 5.0) -> Optional[dict]:
    """Returns ``{"focus", "readback"}``, or None if cancelled / camera unavailable."""
    from camera_utils import lock_focus, open_camera_result
    res = open_camera_result(camera_index)
    if res.cap is None:
        print(res.detail)
        return None
    cap = res.cap
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    focus = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            cur = cap.get(cv2.CAP_PROP_FOCUS)
            vis = frame.copy()
            h, w = gray.shape
            cv2.rectangle(vis, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 220, 255), 1)
            cv2.putText(vis, f"autofocus {af:g} | focus {cur:g} | sharpness {sharpness(gray):.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, HELP, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow("EyeLab focus", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                return None
            if k == ord("a"):
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if af else 1)
            elif k == ord("l"):
                focus = cur
                lock_focus(cap, focus)
            elif k in (ord("+"), ord("="), ord("-")):
                focus = (cur if focus is None else focus) + (step if k != ord("-") else -step)
                lock_focus(cap, focus)
            elif k in (13, 10):
                if focus is None:
                    focus = cur
                return {"focus": float(focus), "readback": lock_focus(cap, focus)}
    finally:
        cap.release()
        cv2.destroyAllWindows()


def focus_supported(readback: dict) -> bool:
    return readback.get("autofocus", 1) == 0 and readback.get("focus", -1) >= 0
