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
    """
    Tunables for the pose-lock state machine.

    Three marker-count thresholds, ordered sustain <= acquire <= trusted:

    - ``trusted_min_markers``: a pose from at least this many markers locks at
      once from SEARCHING (no multi-frame confirmation). It is also the count
      below which ROI re-detection looks for the missing markers. The count is
      supporting evidence only: it never exempts a pose from the validity,
      reprojection, ambiguity and motion gates below (audit F02).
    - ``acquire_min_markers``: the fewest markers that can start a lock. Below
      the trusted count, acquisition is *confirmed*: the planar ambiguity must
      be resolvable (``min_ambiguity_ratio``, ``ambiguity_min_separation_deg``)
      and the pose must repeat over
      ``acquire_confirm_frames`` consecutive frames, because there is no
      Kalman prediction yet to pick the right one of the two planar solutions.
    - ``sustain_min_markers``: the fewest markers that keep a lock alive.

    The marker poses on the structure come from the geometry file, so a single
    marker does define the whole structure pose; the thresholds bound how much
    the lever arm and the planar ambiguity are allowed to cost.

    Phases (every candidate pose, at any marker count, first has to be finite
    with positive depth):

    - acquisition (SEARCHING): absolute reprojection-RMS ceiling, planar
      ambiguity check, then immediate lock (trusted count) or confirmation.
    - sustained tracking (LOCKED): adaptive RMS gate under the same ceiling and
      time-aware motion gates against the Kalman prediction.
    - reacquisition: a candidate that keeps failing the sustain gates (a
      specimen that was really moved, or a lost track) exhausts the coasting
      window; the lock then falls back to SEARCHING, the filter is reset and
      the new pose goes through acquisition again.
    """
    acquire_min_markers: int = 2      # fewest markers that can start a lock
    sustain_min_markers: int = 1      # fewest markers that keep a lock alive
    trusted_min_markers: int = 3      # lock at once / skip gates / no ROI search
    acquire_confirm_frames: int = 5   # consecutive consistent poses to confirm
    confirm_max_translation_m: float = 0.02   # frame-to-frame consistency
    confirm_max_rotation_deg: float = 8.0     # frame-to-frame consistency
    # Planar solves only. A candidate is "ambiguous" when the two mirror IPPE
    # poses explain the image about equally well (2nd-best / best
    # reprojection error below min_ambiguity_ratio) AND they are far apart
    # (more than ambiguity_min_separation_deg), i.e. picking the wrong one
    # would cost a large rotation. Ambiguous candidates cannot start a lock.
    # Engineering defaults from a synthetic sweep (20 mm markers, 0.3 px
    # corner noise); to be calibrated by validation test B1c.
    min_ambiguity_ratio: float = 3.0
    ambiguity_min_separation_deg: float = 10.0
    coast_duration_s: float = 0.3     # predict-only coasting on full dropout
    # RMS gate applies ONLY to low-marker-count (sustain) poses. Full-count
    # poses are accepted unconditionally — real board geometry (hand-measured
    # marker positions) can have a large but stable baseline RMS, and gating
    # acquisition on it would prevent the lock from ever engaging.
    max_reproj_rms_px: float = 3.0        # floor of the sustain RMS gate
    rms_baseline_factor: float = 2.5      # gate = max(floor, factor * baseline)
    # Hard ceiling on reprojection RMS for ANY candidate at ANY marker count,
    # acquisition included. Deliberately generous (hand-measured geometry can
    # have a large but stable baseline, see above); it only has to stop gross
    # solves. Engineering default, to be set from validation data.
    max_acquire_reproj_rms_px: float = 15.0
    # Motion gates against the Kalman prediction, for every marker count while
    # locked. The allowed jump is max(per-frame floor, speed * dt) where dt is
    # the time since the last accepted measurement, so a slow frame is not
    # penalised and a fast one is not waved through.
    max_translation_jump_m: float = 0.05  # per-frame floor
    max_rotation_jump_deg: float = 15.0   # per-frame floor
    max_translation_speed_m_s: float = 1.0
    max_rotation_speed_deg_s: float = 360.0

    def __post_init__(self) -> None:
        if not (
            1 <= self.sustain_min_markers
            <= self.acquire_min_markers
            <= self.trusted_min_markers
        ):
            raise ValueError(
                "PoseLockConfig needs 1 <= sustain_min_markers <= "
                "acquire_min_markers <= trusted_min_markers, got "
                f"{self.sustain_min_markers}/{self.acquire_min_markers}/"
                f"{self.trusted_min_markers}"
            )
        if self.acquire_confirm_frames < 1:
            raise ValueError("acquire_confirm_frames must be >= 1")
        if self.max_acquire_reproj_rms_px < self.max_reproj_rms_px:
            raise ValueError(
                "max_acquire_reproj_rms_px must be >= max_reproj_rms_px"
            )
        if self.max_translation_speed_m_s < 0 or self.max_rotation_speed_deg_s < 0:
            raise ValueError("motion speed limits must be >= 0")


# Named configurations. "standard" is the default; "single" allows a lock to
# start from one marker (confirmed, see PoseLockConfig); "legacy3" is the
# behaviour before 2026-10 (three markers to acquire), kept for A/B runs.
POSE_LOCK_PRESETS: dict[str, PoseLockConfig] = {
    "standard": PoseLockConfig(),
    "single": PoseLockConfig(acquire_min_markers=1),
    "legacy3": PoseLockConfig(acquire_min_markers=3),
}


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

    - SEARCHING: a valid, unambiguous pose with `trusted_min_markers` locks at
      once; one with at least `acquire_min_markers` locks after the
      confirmation described in PoseLockConfig (repeated over
      `acquire_confirm_frames` frames).
    - LOCKED: accepts poses down to `sustain_min_markers`; every accepted pose,
      whatever its marker count, passes the RMS gate and the time-aware
      translation/rotation jump limits against the Kalman prediction, so a bad
      solve cannot teleport the overlay.
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
        # "invalid" | "count" | "rms" | "jump" | "ambiguous" | "confirming"
        self.last_reject_reason: Optional[str] = None
        self._candidate: Optional["PoseResult"] = None  # acquisition candidate
        self._candidate_streak = 0

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
        if pose is None:
            self._clear_candidate()  # confirmation needs consecutive frames

        if pose is not None and self._accept(pose, now):
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
        self._clear_candidate()

    # ── internal ──

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_streak = 0

    @staticmethod
    def _is_valid(pose: "PoseResult") -> bool:
        """Finite pose with the structure in front of the camera (z > 0)."""
        try:
            rvec = np.asarray(pose.rvec, dtype=np.float64).reshape(3)
            tvec = np.asarray(pose.tvec, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return False
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
            return False
        if tvec[2] <= 0.0:
            return False
        rms = pose.rms_reproj_px
        return rms is None or bool(np.isfinite(rms))

    def _ambiguous(self, pose: "PoseResult") -> bool:
        """Planar two-fold ambiguity, judged from the solve's geometry."""
        cfg = self.config
        ratio = getattr(pose, "ambiguity_ratio", None)
        separation = getattr(pose, "ambiguity_separation_deg", None)
        return (
            ratio is not None
            and ratio < cfg.min_ambiguity_ratio
            and (separation is None or separation > cfg.ambiguity_min_separation_deg)
        )

    def _reject(self, reason: str, clear_candidate: bool = True) -> bool:
        if clear_candidate:
            self._clear_candidate()
        self.last_reject_reason = reason
        return False

    def _acquire(self, pose: "PoseResult") -> bool:
        """Acquisition: SEARCHING, or LOCKED with a cold filter."""
        cfg = self.config
        if pose.marker_count < cfg.acquire_min_markers:
            return self._reject("count")
        if (
            pose.rms_reproj_px is not None
            and pose.rms_reproj_px > cfg.max_acquire_reproj_rms_px
        ):
            return self._reject("rms")
        # No Kalman prediction exists yet to pick between the two planar
        # solutions, so an ambiguous solve cannot start a lock at any count.
        if self._ambiguous(pose):
            return self._reject("ambiguous")
        if pose.marker_count >= cfg.trusted_min_markers:
            self._clear_candidate()
            return True
        return self._confirm_acquisition(pose)

    def _confirm_acquisition(self, pose: "PoseResult") -> bool:
        """Confirmation for acquisition from fewer than the trusted count."""
        cfg = self.config
        if self._candidate is not None and self._consistent(self._candidate, pose):
            self._candidate_streak += 1
        else:
            self._candidate_streak = 1
        self._candidate = pose
        if self._candidate_streak >= cfg.acquire_confirm_frames:
            self._clear_candidate()
            return True
        self.last_reject_reason = "confirming"
        return False

    def _consistent(self, a: "PoseResult", b: "PoseResult") -> bool:
        cfg = self.config
        dt = float(np.linalg.norm(
            np.asarray(a.tvec, dtype=np.float64).reshape(3)
            - np.asarray(b.tvec, dtype=np.float64).reshape(3)
        ))
        if dt > cfg.confirm_max_translation_m:
            return False
        return rotation_angle_between(a.rvec, b.rvec) <= cfg.confirm_max_rotation_deg

    def _sustain_rms_gate_px(self) -> float:
        cfg = self.config
        if self._rms_baseline is None:
            gate = cfg.max_reproj_rms_px
        else:
            gate = max(cfg.max_reproj_rms_px, cfg.rms_baseline_factor * self._rms_baseline)
        return min(gate, cfg.max_acquire_reproj_rms_px)

    def _sustain(self, pose: "PoseResult", now: float) -> bool:
        """Sustained tracking: gates apply at every marker count."""
        cfg = self.config
        if pose.marker_count < cfg.sustain_min_markers:
            return self._reject("count", clear_candidate=False)

        if (
            pose.rms_reproj_px is not None
            and pose.rms_reproj_px > self._sustain_rms_gate_px()
        ):
            return self._reject("rms", clear_candidate=False)

        prediction = self.predicted_pose()
        if prediction is None:
            return self._reject("jump", clear_candidate=False)
        pred_rvec, pred_tvec = prediction
        dt = max(0.0, now - self._last_accept_time)
        max_t = max(cfg.max_translation_jump_m, cfg.max_translation_speed_m_s * dt)
        max_r = max(cfg.max_rotation_jump_deg, cfg.max_rotation_speed_deg_s * dt)
        t_jump = float(
            np.linalg.norm(
                np.asarray(pose.tvec, dtype=np.float64).reshape(3)
                - np.asarray(pred_tvec, dtype=np.float64).reshape(3)
            )
        )
        if t_jump > max_t:
            return self._reject("jump", clear_candidate=False)
        if rotation_angle_between(pose.rvec, pred_rvec) > max_r:
            return self._reject("jump", clear_candidate=False)
        return True

    def _accept(self, pose: "PoseResult", now: float) -> bool:
        # Validity first, at every marker count and in every state: a NaN or
        # behind-the-camera pose must never reach the Kalman filter.
        if not self._is_valid(pose):
            return self._reject("invalid")
        if not self.is_locked or not self.kalman.initialized:
            return self._acquire(pose)
        self._clear_candidate()
        return self._sustain(pose, now)


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
