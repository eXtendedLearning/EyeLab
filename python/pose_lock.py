#!/usr/bin/env python3
"""
Pose-lock resilience layer for the ArUco board pipeline.

Classes:
    PoseLockConfig   — tunables for acquisition / sustain / coasting.
    PoseLock         — SEARCHING → LOCKED → COASTING state machine. Owns the
                       Kalman filter; validates measurements with reprojection
                       and pose-jump gates; coasts on prediction through brief
                       dropouts.
    MarkerCarryover  — LK optical-flow gap filler: markers detected recently
                       but missed this frame are carried forward for a few
                       frames so the board pose survives detection flicker.
    RoiRedetector    — pose-guided recovery: projects missing board markers
                       through the predicted pose, crops an upscaled ROI and
                       re-runs detection there (small/distant marker recall).

Kept separate from pose_estimator.py so the detection pipeline stays modular.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from pose_estimator import PoseKalmanFilter, PoseResult


# ── Lock states ───────────────────────────────────────────────────────────────

LOCK_SEARCHING = "searching"
LOCK_LOCKED = "locked"
LOCK_COASTING = "coasting"


@dataclass(frozen=True)
class PoseLockConfig:
    """Tunables for the pose-lock state machine."""
    acquire_min_markers: int = 3      # markers required to (re)acquire lock
    sustain_min_markers: int = 1      # markers that keep an existing lock alive
    coast_duration_s: float = 0.3     # predict-only coasting on full dropout
    # RMS gate applies ONLY to low-marker-count (sustain) poses. Full-count
    # poses are accepted unconditionally — real board geometry (hand-measured
    # marker positions) can have a large but stable baseline RMS, and gating
    # acquisition on it would prevent the lock from ever engaging.
    max_reproj_rms_px: float = 3.0        # floor of the sustain RMS gate
    rms_baseline_factor: float = 2.5      # gate = max(floor, factor * baseline)
    max_translation_jump_m: float = 0.05  # gate for low-marker-count poses
    max_rotation_jump_deg: float = 15.0   # gate for low-marker-count poses


def align_rvec(rvec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Pick the Rodrigues representation of `rvec` closest to `reference`.

    (θ, axis) and (2π−θ, −axis) encode the same rotation; raw solvePnP output
    can flip between them frame-to-frame, which breaks linear Kalman smoothing
    and jump gating. Returns a (3, 1) float array.
    """
    r = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    ref = np.asarray(reference, dtype=np.float64).reshape(3, 1)
    theta = float(np.linalg.norm(r))
    if theta < 1e-9:
        return r
    alt = r * ((theta - 2.0 * np.pi) / theta)
    if np.linalg.norm(alt - ref) < np.linalg.norm(r - ref):
        return alt
    return r


def rotation_angle_between(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    """Angle (degrees) of the relative rotation between two Rodrigues vectors."""
    Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    R_rel = Ra.T @ Rb
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


# ── Pose lock state machine ───────────────────────────────────────────────────

class PoseLock:
    """
    Keeps the board pose alive through partial and brief total detection loss.

    - SEARCHING: needs `acquire_min_markers` with acceptable reprojection RMS.
    - LOCKED: accepts poses down to `sustain_min_markers`; low-count poses are
      additionally gated against the Kalman prediction (translation/rotation
      jump limits) so a bad 2-marker solve cannot teleport the overlay.
    - COASTING: on dropout, outputs the Kalman prediction for up to
      `coast_duration_s`, then falls back to SEARCHING (full reacquisition).
    """

    def __init__(
        self,
        kalman: "PoseKalmanFilter",
        config: PoseLockConfig | None = None,
    ):
        self.kalman = kalman
        self.config = config or PoseLockConfig()
        self.state = LOCK_SEARCHING
        self._last_accept_time = 0.0
        self._rms_baseline: Optional[float] = None  # EWMA of accepted pose RMS
        self.last_reject_reason: Optional[str] = None  # "count"|"rms"|"jump"

    @property
    def is_locked(self) -> bool:
        return self.state in (LOCK_LOCKED, LOCK_COASTING)

    @property
    def min_markers(self) -> int:
        """Marker-count threshold to hand the detector for this frame."""
        return (
            self.config.sustain_min_markers
            if self.is_locked
            else self.config.acquire_min_markers
        )

    def predicted_pose(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """(rvec, tvec) one-step prediction, or None if the filter is cold."""
        if not self.kalman.initialized:
            return None
        return self.kalman.predict_measurement()

    def process(
        self,
        pose: Optional["PoseResult"],
        now: Optional[float] = None,
    ) -> Optional["PoseResult"]:
        """
        Feed one frame's measured pose (or None). Returns the smoothed pose,
        a coasted prediction, or None; updates `self.state`.
        """
        now = time.monotonic() if now is None else now

        if pose is not None and self._accept(pose):
            self.last_reject_reason = None
            if pose.rms_reproj_px is not None:
                self._rms_baseline = (
                    pose.rms_reproj_px
                    if self._rms_baseline is None
                    else 0.9 * self._rms_baseline + 0.1 * pose.rms_reproj_px
                )
            prediction = self.predicted_pose()
            rvec = pose.rvec
            if prediction is not None:
                rvec = align_rvec(rvec, prediction[0])
            smooth_rvec, smooth_tvec = self.kalman.update(rvec, pose.tvec)
            self.state = LOCK_LOCKED
            self._last_accept_time = now
            return replace(pose, rvec=smooth_rvec, tvec=smooth_tvec)

        if (
            self.is_locked
            and self.kalman.initialized
            and (now - self._last_accept_time) <= self.config.coast_duration_s
        ):
            rvec, tvec = self.kalman.coast()
            self.state = LOCK_COASTING
            from pose_estimator import PoseResult  # local import: avoids cycle
            return PoseResult(
                rvec=rvec,
                tvec=tvec,
                marker_ids=[],
                marker_count=0,
                timestamp=now,
                rms_reproj_px=None,
                coasted=True,
            )

        self.state = LOCK_SEARCHING
        self.kalman.reset()
        return None

    def reset(self) -> None:
        self.state = LOCK_SEARCHING
        self.kalman.reset()
        self._rms_baseline = None
        self.last_reject_reason = None

    # ── internal ──

    def _sustain_rms_gate_px(self) -> float:
        cfg = self.config
        if self._rms_baseline is None:
            return cfg.max_reproj_rms_px
        return max(cfg.max_reproj_rms_px, cfg.rms_baseline_factor * self._rms_baseline)

    def _accept(self, pose: "PoseResult") -> bool:
        cfg = self.config
        n = pose.marker_count

        # Full-count poses: accept unconditionally (pre-lock behaviour).
        if n >= cfg.acquire_min_markers:
            return True

        if not self.is_locked or not self.kalman.initialized:
            self.last_reject_reason = "count"
            return False

        if n < cfg.sustain_min_markers:
            self.last_reject_reason = "count"
            return False

        # Low-marker-count pose: RMS gate (adaptive) + jump gates vs prediction
        if (
            pose.rms_reproj_px is not None
            and pose.rms_reproj_px > self._sustain_rms_gate_px()
        ):
            self.last_reject_reason = "rms"
            return False

        prediction = self.predicted_pose()
        if prediction is None:
            self.last_reject_reason = "jump"
            return False
        pred_rvec, pred_tvec = prediction
        t_jump = float(
            np.linalg.norm(
                np.asarray(pose.tvec, dtype=np.float64).reshape(3)
                - np.asarray(pred_tvec, dtype=np.float64).reshape(3)
            )
        )
        if t_jump > cfg.max_translation_jump_m:
            self.last_reject_reason = "jump"
            return False
        r_jump = rotation_angle_between(pose.rvec, pred_rvec)
        if r_jump > cfg.max_rotation_jump_deg:
            self.last_reject_reason = "jump"
            return False
        return True


# ── Optical-flow marker carryover ─────────────────────────────────────────────

class MarkerCarryover:
    """
    Carries recently seen marker corners across frames with LK optical flow.

    Detection runs every frame; markers detected are stored fresh. Markers
    seen within the last `max_age_frames` frames but missed now are tracked
    from the previous frame and merged back in, so a marker flickering in and
    out of the detector keeps contributing to the board pose.
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, max_age_frames: int = 5, max_area_ratio_change: float = 0.5):
        self.max_age_frames = max(1, int(max_age_frames))
        self.max_area_ratio_change = float(max_area_ratio_change)
        self._prev_gray: Optional[np.ndarray] = None
        # marker_id -> {"corners": (4,2) float32, "age": int, "area": float}
        self._tracks: dict[int, dict] = {}
        self.last_carryover_count = 0

    def reset(self) -> None:
        self._prev_gray = None
        self._tracks = {}
        self.last_carryover_count = 0

    def process(
        self,
        gray: np.ndarray,
        corners: list[np.ndarray],
        ids: Optional[np.ndarray],
    ) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
        """
        Merge this frame's detections with tracked carryover markers.

        Returns (corners, ids) in the same format detectMarkers produces.
        """
        self.last_carryover_count = 0
        detected_ids = (
            [int(i) for i in ids.flatten()] if ids is not None and len(ids) else []
        )

        # Refresh tracks for markers detected this frame
        for i, marker_id in enumerate(detected_ids):
            quad = np.asarray(corners[i], dtype=np.float32).reshape(4, 2)
            self._tracks[marker_id] = {
                "corners": quad,
                "age": 0,
                "area": abs(float(cv2.contourArea(quad))),
            }

        # Track markers that were seen recently but missed now
        carry_corners: list[np.ndarray] = []
        carry_ids: list[int] = []
        stale: list[int] = []
        missing = [m for m in self._tracks if m not in detected_ids]

        if missing and self._prev_gray is not None:
            pts = np.vstack(
                [self._tracks[m]["corners"] for m in missing]
            ).astype(np.float32)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts.reshape(-1, 1, 2), None, **self.LK_PARAMS,
            )
            h, w = gray.shape[:2]
            for k, marker_id in enumerate(missing):
                track = self._tracks[marker_id]
                track["age"] += 1
                if track["age"] > self.max_age_frames:
                    stale.append(marker_id)
                    continue
                if new_pts is None or status is None:
                    stale.append(marker_id)
                    continue
                sl = slice(k * 4, k * 4 + 4)
                if not status[sl].flatten().astype(bool).all():
                    stale.append(marker_id)
                    continue
                quad = new_pts[sl].reshape(4, 2)
                if not self._quad_is_sane(quad, track["area"], w, h):
                    stale.append(marker_id)
                    continue
                track["corners"] = quad.astype(np.float32)
                carry_corners.append(quad.reshape(1, 4, 2).astype(np.float32))
                carry_ids.append(marker_id)
        elif missing:
            # No previous frame to track from: just age them out
            for marker_id in missing:
                self._tracks[marker_id]["age"] += 1
                if self._tracks[marker_id]["age"] > self.max_age_frames:
                    stale.append(marker_id)

        for marker_id in stale:
            self._tracks.pop(marker_id, None)

        self._prev_gray = gray.copy()
        self.last_carryover_count = len(carry_ids)

        merged_corners = list(corners) + carry_corners
        merged_ids = detected_ids + carry_ids
        if not merged_ids:
            return [], None
        return merged_corners, np.array(merged_ids, dtype=np.int32).reshape(-1, 1)

    def _quad_is_sane(
        self, quad: np.ndarray, ref_area: float, width: int, height: int,
    ) -> bool:
        if not np.isfinite(quad).all():
            return False
        if (quad[:, 0].min() < 0 or quad[:, 1].min() < 0
                or quad[:, 0].max() >= width or quad[:, 1].max() >= height):
            return False
        area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        if ref_area <= 0 or area <= 0:
            return False
        ratio = area / ref_area
        lo = 1.0 - self.max_area_ratio_change
        hi = 1.0 + self.max_area_ratio_change
        if not (lo <= ratio <= hi):
            return False
        return cv2.isContourConvex(quad.astype(np.float32))


# ── Pose-guided ROI re-detection ──────────────────────────────────────────────

class RoiRedetector:
    """
    Recovers missing board markers by re-detecting inside pose-predicted ROIs.

    Given the last (or predicted) board pose, each missing marker's object
    corners are projected into the image; a margin-expanded crop around the
    projection is upscaled and run through the ArUco detector again. Small
    markers that fail full-frame detection often decode fine at 2× in a crop.
    """

    def __init__(
        self,
        board: "cv2.aruco.Board",
        detector: "cv2.aruco.ArucoDetector",
        upscale: float = 2.0,
        margin: float = 1.6,
        max_rois_per_frame: int = 4,
        min_roi_px: int = 12,
        max_roi_frac: float = 0.25,
    ):
        self.detector = detector
        self.upscale = max(1.0, float(upscale))
        self.margin = max(1.0, float(margin))
        self.max_rois_per_frame = max(1, int(max_rois_per_frame))
        self.min_roi_px = int(min_roi_px)
        self.max_roi_frac = float(max_roi_frac)
        self.object_corners_by_id = self._extract_board_geometry(board)
        self.last_recovered_count = 0

    @staticmethod
    def _extract_board_geometry(
        board: "cv2.aruco.Board",
    ) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        try:
            ids = np.asarray(board.getIds()).flatten()
            obj_points = board.getObjPoints()
        except Exception:
            return out
        for marker_id, pts in zip(ids, obj_points):
            out[int(marker_id)] = np.asarray(pts, dtype=np.float32).reshape(4, 3)
        return out

    def recover(
        self,
        gray: np.ndarray,
        missing_ids: list[int],
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
    ) -> tuple[list[np.ndarray], list[int]]:
        """Return (corners, ids) for markers recovered from predicted ROIs."""
        self.last_recovered_count = 0
        found_corners: list[np.ndarray] = []
        found_ids: list[int] = []
        h, w = gray.shape[:2]
        max_side = max(1.0, self.max_roi_frac * max(w, h))

        for marker_id in missing_ids[: self.max_rois_per_frame]:
            obj_pts = self.object_corners_by_id.get(int(marker_id))
            if obj_pts is None:
                continue
            proj, _ = cv2.projectPoints(
                obj_pts, rvec, tvec, camera_matrix, dist_coeffs,
            )
            proj = proj.reshape(4, 2)
            if not np.isfinite(proj).all():
                continue

            cx, cy = float(proj[:, 0].mean()), float(proj[:, 1].mean())
            side = float(
                max(proj[:, 0].max() - proj[:, 0].min(),
                    proj[:, 1].max() - proj[:, 1].min())
            ) * self.margin
            if side < self.min_roi_px or side > max_side:
                continue

            x0 = int(max(0, cx - side / 2))
            y0 = int(max(0, cy - side / 2))
            x1 = int(min(w, cx + side / 2))
            y1 = int(min(h, cy + side / 2))
            if x1 - x0 < self.min_roi_px or y1 - y0 < self.min_roi_px:
                continue

            crop = gray[y0:y1, x0:x1]
            if self.upscale > 1.0:
                crop = cv2.resize(
                    crop, None, fx=self.upscale, fy=self.upscale,
                    interpolation=cv2.INTER_CUBIC,
                )

            corners, ids, _ = self.detector.detectMarkers(crop)
            if ids is None or len(ids) == 0:
                continue
            for i, found_id in enumerate(ids.flatten()):
                if int(found_id) != int(marker_id):
                    continue
                quad = np.asarray(corners[i], dtype=np.float32).reshape(4, 2)
                quad /= self.upscale
                quad[:, 0] += x0
                quad[:, 1] += y0
                found_corners.append(quad.reshape(1, 4, 2).astype(np.float32))
                found_ids.append(int(marker_id))
                break

        self.last_recovered_count = len(found_ids)
        return found_corners, found_ids
