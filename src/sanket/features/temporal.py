"""Temporal dynamics and summary statistics aggregation for landmark sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_frame_velocities(sequence_features: np.ndarray) -> np.ndarray:
    """Compute instantaneous frame-to-frame velocity vectors.

    Args:
        sequence_features: 2D array of shape (T, D) containing per-frame features.

    Returns:
        2D array of shape (T-1, D) containing differences between consecutive frames.
        If T <= 1, returns an empty array of shape (0, D).
    """
    if sequence_features.ndim != 2:
        msg = f"Expected 2D array (T, D), got {sequence_features.shape}"
        raise ValueError(msg)

    t, d = sequence_features.shape
    if t <= 1:
        return np.zeros((0, d), dtype=np.float32)

    return (sequence_features[1:] - sequence_features[:-1]).astype(np.float32)


def compute_temporal_statistics(
    sequence_features: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute temporal summary statistics across sequence frames.

    Args:
        sequence_features: 2D array of shape (T, D).
        mask: Optional boolean 1D array of shape (T,) indicating valid frames.
              If None, all frames are considered valid.

    Returns:
        Dictionary mapping statistic names ('mean', 'std', 'delta', 'max_velocity',
        'min', 'max') to 1D float32 arrays of shape (D,).
    """
    if sequence_features.ndim != 2:
        msg = f"Expected 2D array (T, D), got {sequence_features.shape}"
        raise ValueError(msg)

    t, d = sequence_features.shape
    if t == 0:
        return {
            "mean": np.zeros(d, dtype=np.float32),
            "std": np.zeros(d, dtype=np.float32),
            "delta": np.zeros(d, dtype=np.float32),
            "max_velocity": np.zeros(d, dtype=np.float32),
            "min": np.zeros(d, dtype=np.float32),
            "max": np.zeros(d, dtype=np.float32),
        }

    if mask is not None:
        if mask.shape != (t,):
            msg = f"Mask shape {mask.shape} does not match sequence length {t}"
            raise ValueError(msg)
        valid_features = sequence_features[mask]
        if len(valid_features) == 0:
            return {
                "mean": np.zeros(d, dtype=np.float32),
                "std": np.zeros(d, dtype=np.float32),
                "delta": np.zeros(d, dtype=np.float32),
                "max_velocity": np.zeros(d, dtype=np.float32),
                "min": np.zeros(d, dtype=np.float32),
                "max": np.zeros(d, dtype=np.float32),
            }
    else:
        valid_features = sequence_features

    # Summary statistics
    mean_val = np.mean(valid_features, axis=0).astype(np.float32)
    std_val = np.std(valid_features, axis=0).astype(np.float32)
    min_val = np.min(valid_features, axis=0).astype(np.float32)
    max_val = np.max(valid_features, axis=0).astype(np.float32)

    # Net displacement (last frame - first frame)
    delta_val = (valid_features[-1] - valid_features[0]).astype(np.float32)

    # Maximum absolute instantaneous velocity
    if len(valid_features) > 1:
        velocities = np.abs(valid_features[1:] - valid_features[:-1])
        max_vel = np.max(velocities, axis=0).astype(np.float32)
    else:
        max_vel = np.zeros(d, dtype=np.float32)

    return {
        "mean": mean_val,
        "std": std_val,
        "delta": delta_val,
        "max_velocity": max_vel,
        "min": min_val,
        "max": max_val,
    }


def aggregate_sequence_features(
    sequence_features: np.ndarray,
    stats: Sequence[str] = ("mean", "std", "delta"),
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate a sequence of frame features into a fixed-dimension vector.

    Args:
        sequence_features: 2D array of shape (T, D).
        stats: Sequence of statistical moments to concatenate.
               Available options: ('mean', 'std', 'delta', 'max_velocity', 'min', 'max').
        mask: Optional boolean 1D array of shape (T,) indicating valid frames.

    Returns:
        1D float32 array of shape (len(stats) * D,).
    """
    stat_dict = compute_temporal_statistics(sequence_features, mask=mask)
    vectors: list[np.ndarray] = []
    for s in stats:
        if s not in stat_dict:
            msg = f"Unknown statistic '{s}'. Available: {list(stat_dict.keys())}"
            raise ValueError(msg)
        vectors.append(stat_dict[s])

    if not vectors:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(vectors).astype(np.float32)
