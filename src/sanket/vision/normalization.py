"""Coordinate normalization subsystem for skeletal hand landmarks (SANKET-026).

Implements wrist-centered translation, palm-scale Euclidean distance normalization,
and optional in-plane 2D rotation alignment to achieve translation, scale, and
orientation invariance for sign language recognition.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sanket.vision.landmarks import (
    COORDINATE_DIM,
    NUM_HAND_LANDMARKS,
    FrameLandmarks,
    SequenceLandmarks,
)

logger = logging.getLogger(__name__)

# Keypoint indices
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def normalize_hand_landmarks(
    landmarks: np.ndarray,
    method: str = "palm",
    align_rotation: bool = False,
    min_scale: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize a single hand's 3D landmarks for translation and scale invariance.

    Steps applied:
      1. Translation normalization: Translates landmarks so that the wrist (L_0) is at (0, 0, 0).
      2. Scale normalization: Scales coordinates so that reference distance equals 1.0.
         - 'palm': Distance between wrist (L_0) and middle MCP (L_9) becomes 1.0.
         - 'max_span': Maximum distance from wrist to any joint becomes 1.0.
         - 'bbox': Maximum bounding box dimension becomes 1.0.
      3. Optional 2D rotation alignment: Rotates xy-coordinates so palm vector points along -y.

    Args:
        landmarks: Array of shape (21, 3) or (21, 2) and dtype float.
        method: Scale normalization reference metric ('palm', 'max_span', 'bbox').
        align_rotation: If True, applies 2D in-plane rotational alignment.
        min_scale: Minimum scale threshold to protect against division by zero.

    Returns:
        Tuple of (normalized_landmarks, normalization_metadata).
        normalized_landmarks has identical shape and float32 dtype.

    Raises:
        ValueError: If input shape is invalid or method is unsupported.
    """
    if landmarks is None or landmarks.size == 0:
        raise ValueError("Cannot normalize None or empty landmarks array")
    if landmarks.ndim != 2 or landmarks.shape[0] != NUM_HAND_LANDMARKS:
        raise ValueError(f"Expected shape ({NUM_HAND_LANDMARKS}, D), got {landmarks.shape}")

    coords = landmarks.astype(np.float32).copy()

    # If hand is completely zero (undetected), return as-is
    if np.all(coords == 0.0):
        metadata: dict[str, Any] = {
            "wrist_origin": np.zeros(coords.shape[1], dtype=np.float32),
            "scale_factor": 1.0,
            "rotation_angle": 0.0,
            "method": method,
            "is_valid": False,
        }
        return coords, metadata

    # 1. Translation Normalization (Wrist-centered: L_0 -> (0, 0, 0))
    wrist = coords[WRIST_IDX].copy()
    coords -= wrist

    # 2. Scale Normalization
    valid_methods = {"palm", "max_span", "bbox"}
    if method not in valid_methods:
        msg = f"Unsupported normalization method '{method}'. Choose from {valid_methods}"
        raise ValueError(msg)

    if method == "palm":
        # Distance between wrist and middle finger MCP joint
        scale = float(np.linalg.norm(coords[MIDDLE_MCP_IDX]))
    elif method == "max_span":
        # Maximum Euclidean distance from wrist to any landmark
        distances = np.linalg.norm(coords, axis=1)
        scale = float(np.max(distances))
    elif method == "bbox":
        # Max bounding box range across dimensions
        ranges = np.ptp(coords, axis=0)
        scale = float(np.max(ranges))

    # Protect against degenerate/collapsed hands
    if scale < min_scale:
        scale_factor = 1.0
    else:
        scale_factor = scale
        coords /= scale_factor

    # 3. Optional 2D In-Plane Rotation Alignment
    rot_angle = 0.0
    if align_rotation:
        # Vector from wrist to middle MCP in xy plane
        vec_x, vec_y = coords[MIDDLE_MCP_IDX, 0], coords[MIDDLE_MCP_IDX, 1]
        # Calculate angle relative to upward vector (0, -1)
        # Note: in screen space y points downwards, so upward is -y
        rot_angle = float(np.arctan2(vec_x, -vec_y))
        cos_t = np.cos(-rot_angle)
        sin_t = np.sin(-rot_angle)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        # Rotate x and y coordinates
        coords[:, :2] = np.dot(coords[:, :2], rot_mat.T)

    metadata = {
        "wrist_origin": wrist,
        "scale_factor": scale_factor,
        "rotation_angle": rot_angle,
        "method": method,
        "is_valid": True,
    }

    return coords, metadata


def denormalize_hand_landmarks(
    normalized_landmarks: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Invert normalization transformation to recover original coordinate values.

    Args:
        normalized_landmarks: Array of shape (21, 3) or (21, 2).
        metadata: Normalization metadata dictionary returned by `normalize_hand_landmarks`.

    Returns:
        Recovered coordinates in original space.
    """
    if not metadata.get("is_valid", False):
        return normalized_landmarks.copy()

    coords = normalized_landmarks.astype(np.float32).copy()

    # Invert rotation if applied
    rot_angle = metadata.get("rotation_angle", 0.0)
    if abs(rot_angle) > 1e-7:
        cos_t = np.cos(rot_angle)
        sin_t = np.sin(rot_angle)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        coords[:, :2] = np.dot(coords[:, :2], rot_mat.T)

    # Invert scale
    scale = metadata["scale_factor"]
    coords *= scale

    # Invert translation
    wrist = metadata["wrist_origin"]
    coords += wrist

    return coords


def normalize_frame_landmarks(
    frame: FrameLandmarks,
    method: str = "palm",
    align_rotation: bool = False,
) -> FrameLandmarks:
    """Normalize hand landmarks in a single FrameLandmarks object.

    Args:
        frame: The input FrameLandmarks object.
        method: Normalization method ('palm', 'max_span', 'bbox').
        align_rotation: Whether to apply 2D rotational alignment.

    Returns:
        A new FrameLandmarks object with normalized coordinates.
    """
    left_norm = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right_norm = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    if frame.left_detected:
        left_norm, _ = normalize_hand_landmarks(
            frame.left_hand, method=method, align_rotation=align_rotation
        )

    if frame.right_detected:
        right_norm, _ = normalize_hand_landmarks(
            frame.right_hand, method=method, align_rotation=align_rotation
        )

    return FrameLandmarks(
        left_hand=left_norm,
        right_hand=right_norm,
        left_detected=frame.left_detected,
        right_detected=frame.right_detected,
        left_confidence=frame.left_confidence,
        right_confidence=frame.right_confidence,
    )


def normalize_sequence_landmarks(
    sequence: SequenceLandmarks,
    method: str = "palm",
    align_rotation: bool = False,
) -> SequenceLandmarks:
    """Normalize hand landmarks across all frames in a SequenceLandmarks object.

    Args:
        sequence: The input SequenceLandmarks object.
        method: Normalization method ('palm', 'max_span', 'bbox').
        align_rotation: Whether to apply 2D rotational alignment.

    Returns:
        A new SequenceLandmarks object with normalized coordinates.
    """
    t = sequence.num_frames
    left_norm = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right_norm = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    for i in range(t):
        if sequence.left_detected[i]:
            left_norm[i], _ = normalize_hand_landmarks(
                sequence.left_hand[i], method=method, align_rotation=align_rotation
            )
        if sequence.right_detected[i]:
            right_norm[i], _ = normalize_hand_landmarks(
                sequence.right_hand[i], method=method, align_rotation=align_rotation
            )

    return SequenceLandmarks(
        left_hand=left_norm,
        right_hand=right_norm,
        left_detected=sequence.left_detected.copy(),
        right_detected=sequence.right_detected.copy(),
        left_confidence=sequence.left_confidence.copy(),
        right_confidence=sequence.right_confidence.copy(),
    )


class LandmarkNormalizer:
    """Configurable normalization pipeline engine for training and real-time inference."""

    def __init__(
        self,
        method: str = "palm",
        align_rotation: bool = False,
        min_scale: float = 1e-6,
    ) -> None:
        """Initialize LandmarkNormalizer.

        Args:
            method: Scale normalization metric ('palm', 'max_span', 'bbox').
            align_rotation: Whether to apply in-plane 2D rotational alignment.
            min_scale: Scale threshold to prevent numerical instability.
        """
        self.method = method
        self.align_rotation = align_rotation
        self.min_scale = min_scale

    def normalize_hand(self, landmarks: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Normalize a single hand array (21, 3)."""
        return normalize_hand_landmarks(
            landmarks,
            method=self.method,
            align_rotation=self.align_rotation,
            min_scale=self.min_scale,
        )

    def normalize_frame(self, frame: FrameLandmarks) -> FrameLandmarks:
        """Normalize a FrameLandmarks object."""
        return normalize_frame_landmarks(
            frame,
            method=self.method,
            align_rotation=self.align_rotation,
        )

    def normalize_sequence(self, sequence: SequenceLandmarks) -> SequenceLandmarks:
        """Normalize a SequenceLandmarks object."""
        return normalize_sequence_landmarks(
            sequence,
            method=self.method,
            align_rotation=self.align_rotation,
        )

    def denormalize_hand(
        self, normalized_landmarks: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """Denormalize a single hand array back to its original space."""
        return denormalize_hand_landmarks(normalized_landmarks, metadata)
