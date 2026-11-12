"""Temporal landmark trajectory resampling, interpolation, and velocity modeling (SANKET-062).

Provides high-fidelity temporal resampling algorithms to normalize variable-speed
and variable-frame-rate gesture signing trajectories into fixed-length sequences (e.g. T=30).

Supported Resampling Methods:
- 'linear': Piecewise linear interpolation across temporal trajectories.
- 'cubic': Cubic spline interpolation ensuring smooth velocity profiles.
- 'nearest': Stepwise nearest-neighbor sampling for discrete state transitions.

Also computes instantaneous velocity and acceleration derivatives to ensure
temporal invariance across signing speeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from sanket.vision.landmarks import SequenceLandmarks

logger = logging.getLogger(__name__)

ResampleMethod = Literal["linear", "cubic", "nearest"]


@dataclass(frozen=True)
class ResampledTrajectory:
    """Container holding temporally resampled landmark trajectory coordinates and kinematics.

    Attributes:
        sequence: Normalized SequenceLandmarks containing resampled (target_length, 21, 3) arrays.
        velocities: First temporal derivative (velocity) of shape (target_length, 2, 21, 3).
        accelerations: Second temporal derivative (acceleration) of shape (target_length, 2, 21, 3).
        original_frames: Original sequence length prior to temporal resampling.
        target_frames: Resampled temporal length.
        method: Interpolation algorithm employed ('linear', 'cubic', 'nearest').
    """

    sequence: SequenceLandmarks
    velocities: np.ndarray  # Shape: (T, 2, 21, 3), dtype: float32
    accelerations: np.ndarray  # Shape: (T, 2, 21, 3), dtype: float32
    original_frames: int
    target_frames: int
    method: str


def compute_temporal_derivatives(
    coords: np.ndarray,
    dt: float = 1.0 / 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1st (velocity) and 2nd (acceleration) temporal derivatives using central differences.

    Args:
        coords: Landmark array of arbitrary shape (T, ...), float32.
        dt: Time delta between consecutive frames in seconds (default: 1/30s).

    Returns:
        Tuple of (velocity, acceleration) arrays sharing the input shape.
    """
    t = coords.shape[0]
    if t <= 1:
        vel = np.zeros_like(coords, dtype=np.float32)
        acc = np.zeros_like(coords, dtype=np.float32)
        return vel, acc

    # Central difference with forward/backward boundaries
    vel = np.gradient(coords, dt, axis=0).astype(np.float32)
    acc = np.gradient(vel, dt, axis=0).astype(np.float32)
    return vel, acc


def interpolate_array_1d_temporal(
    arr: np.ndarray,
    target_length: int,
    method: ResampleMethod = "linear",
) -> np.ndarray:
    """Interpolate an arbitrary multidimensional array along its temporal axis (axis 0).

    Args:
        arr: Input array of shape (T_orig, ...).
        target_length: Desired output length T_target along axis 0.
        method: Interpolation mode ('linear', 'cubic', 'nearest').

    Returns:
        Resampled array of shape (target_length, ...) and dtype float32.

    Raises:
        ValueError: If arr has 0 frames or target_length <= 0.
    """
    t_orig = arr.shape[0]
    if t_orig == 0:
        raise ValueError("Cannot interpolate array with 0 temporal frames.")
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    if t_orig == target_length:
        return arr.astype(np.float32, copy=True)

    # For single-frame sequences, repeat across target frames
    if t_orig == 1:
        tile_reps = [target_length] + [1] * (arr.ndim - 1)
        return np.tile(arr, tile_reps).astype(np.float32)

    orig_times = np.linspace(0.0, 1.0, t_orig, endpoint=True)
    target_times = np.linspace(0.0, 1.0, target_length, endpoint=True)

    # Fall back to linear if cubic requested on fewer than 4 frames
    actual_method = method
    if method == "cubic" and t_orig < 4:
        logger.debug("Falling back from cubic to linear interpolation (T_orig=%d < 4)", t_orig)
        actual_method = "linear"

    # Flatten spatial/landmark dimensions for vectorized interpolation
    orig_shape = arr.shape
    spatial_shape = orig_shape[1:]
    flattened = arr.reshape(t_orig, -1)

    interp_fn = interp1d(
        orig_times,
        flattened,
        axis=0,
        kind=actual_method,
        bounds_error=False,
        fill_value="extrapolate",
    )
    resampled_flat = interp_fn(target_times).astype(np.float32)

    new_shape = (target_length,) + spatial_shape
    return resampled_flat.reshape(new_shape)


class TemporalTrajectoryInterpolator:
    """High-fidelity temporal resampler for Indian Sign Language landmark sequences.

    Normalizes gestures performed at varying physical velocities (fast vs. slow execution)
    to a standardized temporal length, preserving keyframe boundary poses, motion continuity,
    and velocity characteristics.
    """

    def __init__(
        self,
        target_length: int = 30,
        method: ResampleMethod = "linear",
        fps: float = 30.0,
    ) -> None:
        """Initialize the TemporalTrajectoryInterpolator.

        Args:
            target_length: Fixed temporal sequence length (frames), default 30.
            method: Default interpolation strategy ('linear', 'cubic', 'nearest').
            fps: Nominal playback frame rate in frames per second.

        Raises:
            ValueError: If target_length <= 0 or invalid method.
        """
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        if method not in {"linear", "cubic", "nearest"}:
            raise ValueError(f"Invalid method '{method}'. Choose 'linear', 'cubic', or 'nearest'.")

        self.target_length = target_length
        self.method: ResampleMethod = method
        self.fps = max(1.0, float(fps))

    def resample_sequence(
        self,
        sequence: SequenceLandmarks,
        method: ResampleMethod | None = None,
    ) -> ResampledTrajectory:
        """Resample a SequenceLandmarks object to target_length frames and compute kinematics.

        Args:
            sequence: Input SequenceLandmarks with T frames.
            method: Optional override for interpolation mode.

        Returns:
            ResampledTrajectory container holding resampled sequence, velocities, and accelerations.
        """
        chosen_method = method or self.method
        t_orig = sequence.num_frames
        if t_orig == 0:
            raise ValueError("Cannot resample empty SequenceLandmarks (0 frames).")

        # 1. Resample coordinates
        left_resampled = interpolate_array_1d_temporal(
            sequence.left_hand, self.target_length, method=chosen_method
        )
        right_resampled = interpolate_array_1d_temporal(
            sequence.right_hand, self.target_length, method=chosen_method
        )

        # 2. Resample detection flags and confidence scores
        left_det_f = sequence.left_detected.astype(np.float32)
        right_det_f = sequence.right_detected.astype(np.float32)

        left_det_resampled = (
            interpolate_array_1d_temporal(left_det_f, self.target_length, method="nearest") > 0.5
        )
        right_det_resampled = (
            interpolate_array_1d_temporal(right_det_f, self.target_length, method="nearest") > 0.5
        )

        left_conf_resampled = interpolate_array_1d_temporal(
            sequence.left_confidence, self.target_length, method="linear"
        )
        right_conf_resampled = interpolate_array_1d_temporal(
            sequence.right_confidence, self.target_length, method="linear"
        )

        resampled_seq = SequenceLandmarks(
            left_hand=left_resampled,
            right_hand=right_resampled,
            left_detected=left_det_resampled,
            right_detected=right_det_resampled,
            left_confidence=np.clip(left_conf_resampled, 0.0, 1.0),
            right_confidence=np.clip(right_conf_resampled, 0.0, 1.0),
        )

        # 3. Compute kinematic velocity and acceleration derivatives: shape (T, 2, 21, 3)
        dt = 1.0 / self.fps
        both_hands = np.stack([left_resampled, right_resampled], axis=1)  # (T, 2, 21, 3)
        vel, acc = compute_temporal_derivatives(both_hands, dt=dt)

        return ResampledTrajectory(
            sequence=resampled_seq,
            velocities=vel,
            accelerations=acc,
            original_frames=t_orig,
            target_frames=self.target_length,
            method=chosen_method,
        )


def uniform_temporal_subsample(
    sequence: SequenceLandmarks,
    target_length: int = 30,
) -> SequenceLandmarks:
    """Uniformly subsample or supersample keyframes by nearest index selection.

    Args:
        sequence: Input SequenceLandmarks.
        target_length: Desired number of frames.

    Returns:
        Subsampled SequenceLandmarks of exact length target_length.
    """
    t_orig = sequence.num_frames
    if t_orig == 0:
        raise ValueError("Cannot subsample empty SequenceLandmarks (0 frames).")

    if t_orig == target_length:
        return sequence

    indices = np.linspace(0, t_orig - 1, target_length).round().astype(int)

    return SequenceLandmarks(
        left_hand=sequence.left_hand[indices].copy(),
        right_hand=sequence.right_hand[indices].copy(),
        left_detected=sequence.left_detected[indices].copy(),
        right_detected=sequence.right_detected[indices].copy(),
        left_confidence=sequence.left_confidence[indices].copy(),
        right_confidence=sequence.right_confidence[indices].copy(),
    )
