#!/usr/bin/env python3
"""
T2.3 — Marker-Based Spatial Registration.

Aligns UNV geometry to the physical test structure by computing an optimal
rigid-body transformation (R, t) from ArUco marker correspondences using the
SVD-based Kabsch / Procrustes algorithm.

Core classes:
    MarkerCorrespondence — Single marker ↔ UNV-node binding.
    SpatialRegistration  — Kabsch solver + quality metrics + drift monitor.

Quality metrics:
    - RMS registration error (mm)
    - Per-marker residual errors
    - Condition number of the point configuration (warns if near-collinear)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MarkerCorrespondence:
    """Binding between an ArUco marker and a known 3D position in the UNV frame."""
    marker_id: int
    unv_position: np.ndarray       # (3,) in metres — position in UNV / model frame
    node_id: Optional[int] = None  # UNV node ID this marker is placed on (if any)
    description: str = ""


@dataclass
class RegistrationResult:
    R: np.ndarray                  # (3,3) rotation matrix
    t: np.ndarray                  # (3,1) translation vector
    rms_error_mm: float
    per_marker_errors_mm: dict[int, float]  # marker_id → residual in mm
    condition_number: float
    n_correspondences: int
    timestamp: float = 0.0

    @property
    def transform_4x4(self) -> np.ndarray:
        """Homogeneous 4×4 transformation matrix (UNV → world)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.flatten()
        return T


# ── Marker config I/O ─────────────────────────────────────────────────────────

def load_marker_config(json_path: str) -> list[MarkerCorrespondence]:
    """
    Load marker correspondences from a JSON file.

    Expected format:
    {
      "markers": [
        {"markerId": 0, "unvPosition": [x, y, z], "nodeId": null, "description": "..."},
        ...
      ]
    }
    """
    with open(json_path) as f:
        data = json.load(f)

    correspondences = []
    for entry in data["markers"]:
        correspondences.append(MarkerCorrespondence(
            marker_id=entry["markerId"],
            unv_position=np.array(entry["unvPosition"], dtype=np.float64),
            node_id=entry.get("nodeId"),
            description=entry.get("description", ""),
        ))
    return correspondences


def save_marker_config(json_path: str, correspondences: list[MarkerCorrespondence]) -> None:
    """Save marker correspondences to JSON."""
    data = {
        "markers": [
            {
                "markerId": c.marker_id,
                "unvPosition": c.unv_position.tolist(),
                "nodeId": c.node_id,
                "description": c.description,
            }
            for c in correspondences
        ]
    }
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# ── Kabsch / SVD registration ─────────────────────────────────────────────────

class SpatialRegistration:
    """
    Computes an optimal rigid transform mapping UNV coordinates → world (camera)
    coordinates using SVD (Kabsch algorithm).

    Usage:
        reg = SpatialRegistration(correspondences)
        # When markers are detected, update world positions:
        reg.update_detected_position(marker_id, world_xyz)
        # Then compute registration:
        result = reg.compute()
    """

    def __init__(self, correspondences: list[MarkerCorrespondence]):
        self._correspondences: dict[int, MarkerCorrespondence] = {
            c.marker_id: c for c in correspondences
        }
        self._detected_positions: dict[int, np.ndarray] = {}
        self._result: Optional[RegistrationResult] = None

        # Drift monitoring
        self._drift_threshold_mm = 10.0
        self._last_registration_time = 0.0
        self._needs_reregistration = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def correspondences(self) -> dict[int, MarkerCorrespondence]:
        return self._correspondences

    @property
    def result(self) -> Optional[RegistrationResult]:
        return self._result

    @property
    def is_registered(self) -> bool:
        return self._result is not None

    @property
    def needs_reregistration(self) -> bool:
        return self._needs_reregistration

    def set_correspondences(self, correspondences: list[MarkerCorrespondence]) -> None:
        self._correspondences = {c.marker_id: c for c in correspondences}
        self._result = None

    def update_detected_position(self, marker_id: int, world_position: np.ndarray) -> None:
        """Update the detected world-space position for a marker."""
        self._detected_positions[marker_id] = np.asarray(world_position, dtype=np.float64)

    def clear_detected_positions(self) -> None:
        self._detected_positions.clear()

    def compute(self) -> Optional[RegistrationResult]:
        """
        Compute the rigid registration from current correspondences.

        Requires ≥3 non-collinear correspondences with both a known UNV position
        and a detected world position.

        Returns RegistrationResult or None on failure.
        """
        # Collect matched pairs
        unv_pts = []
        world_pts = []
        matched_ids = []

        for mid, corr in self._correspondences.items():
            if mid in self._detected_positions:
                unv_pts.append(corr.unv_position)
                world_pts.append(self._detected_positions[mid])
                matched_ids.append(mid)

        n = len(matched_ids)
        if n < 3:
            return None

        P = np.array(unv_pts, dtype=np.float64)    # (N, 3) — UNV frame
        Q = np.array(world_pts, dtype=np.float64)   # (N, 3) — world frame

        # Check collinearity
        cond = self._condition_number(P)
        if cond > 1e6:
            return None   # near-collinear or degenerate

        # Kabsch algorithm
        R, t = self._kabsch(P, Q)

        # Per-marker residuals
        per_marker_errors: dict[int, float] = {}
        sum_sq = 0.0
        for i, mid in enumerate(matched_ids):
            transformed = R @ P[i] + t.flatten()
            err_m = np.linalg.norm(transformed - Q[i])
            err_mm = err_m * 1000.0
            per_marker_errors[mid] = err_mm
            sum_sq += err_m ** 2

        rms_mm = np.sqrt(sum_sq / n) * 1000.0

        self._result = RegistrationResult(
            R=R,
            t=t.reshape(3, 1),
            rms_error_mm=rms_mm,
            per_marker_errors_mm=per_marker_errors,
            condition_number=cond,
            n_correspondences=n,
            timestamp=time.time(),
        )
        self._last_registration_time = time.time()
        self._needs_reregistration = False
        return self._result

    def transform_point(self, unv_point: np.ndarray) -> Optional[np.ndarray]:
        """Transform a single point from UNV frame to world frame."""
        if self._result is None:
            return None
        p = np.asarray(unv_point, dtype=np.float64)
        return (self._result.R @ p + self._result.t.flatten())

    def transform_points(self, unv_points: np.ndarray) -> Optional[np.ndarray]:
        """Transform (N, 3) array from UNV frame to world frame."""
        if self._result is None:
            return None
        return (unv_points @ self._result.R.T) + self._result.t.flatten()

    # ── Drift monitoring ──────────────────────────────────────────────────────

    def check_drift(self, current_detected: dict[int, np.ndarray]) -> float:
        """
        Estimate drift by measuring residual errors against current detections.

        Returns estimated drift in mm.  If drift exceeds threshold, sets
        needs_reregistration flag.
        """
        if self._result is None:
            return float("inf")

        errors = []
        for mid, world_pos in current_detected.items():
            if mid in self._correspondences:
                expected = self.transform_point(self._correspondences[mid].unv_position)
                if expected is not None:
                    errors.append(np.linalg.norm(expected - world_pos) * 1000.0)

        if not errors:
            return float("inf")

        drift_mm = float(np.mean(errors))
        if drift_mm > self._drift_threshold_mm:
            self._needs_reregistration = True
        return drift_mm

    def set_drift_threshold(self, threshold_mm: float) -> None:
        self._drift_threshold_mm = threshold_mm

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Kabsch algorithm: find R, t minimising  ||Q - (R·P + t)||².

        Args:
            P: (N, 3) source points (UNV frame)
            Q: (N, 3) target points (world frame)

        Returns:
            R: (3, 3) rotation matrix
            t: (3, 1) translation vector
        """
        centroid_P = P.mean(axis=0)
        centroid_Q = Q.mean(axis=0)

        P_c = P - centroid_P
        Q_c = Q - centroid_Q

        H = P_c.T @ Q_c                 # (3, 3) cross-covariance

        U, S, Vt = np.linalg.svd(H)
        V = Vt.T

        # Correct for reflection
        d = np.linalg.det(V @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        R = V @ sign_matrix @ U.T

        t = (centroid_Q - R @ centroid_P).reshape(3, 1)
        return R, t

    @staticmethod
    def _condition_number(pts: np.ndarray) -> float:
        """
        Condition number of the point cloud's covariance matrix.

        Low (<10): well-distributed markers.
        High (>100): markers are nearly collinear — registration ill-conditioned.
        """
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)
        return float(eigenvalues[-1] / (eigenvalues[0] + 1e-12))
