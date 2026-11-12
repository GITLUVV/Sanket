"""Unit tests for temporal landmark trajectory resampling and interpolation (SANKET-062)."""

from __future__ import annotations

import numpy as np
import pytest

from sanket.vision.landmarks import SequenceLandmarks
from sanket.vision.temporal import (
    ResampledTrajectory,
    TemporalTrajectoryInterpolator,
    compute_temporal_derivatives,
    interpolate_array_1d_temporal,
    uniform_temporal_subsample,
)


def _create_mock_trajectory(num_frames: int = 15) -> SequenceLandmarks:
    """Helper generating deterministic sinusoidal moving hand trajectory."""
    t = np.linspace(0, 2 * np.pi, num_frames).astype(np.float32)
    left_hand = np.zeros((num_frames, 21, 3), dtype=np.float32)
    right_hand = np.zeros((num_frames, 21, 3), dtype=np.float32)

    # Wrist (idx 0) moves along x = sin(t), y = cos(t)
    left_hand[:, 0, 0] = np.sin(t)
    left_hand[:, 0, 1] = np.cos(t)
    right_hand[:, 0, 0] = -np.sin(t)
    right_hand[:, 0, 1] = np.cos(t)

    left_det = np.ones(num_frames, dtype=bool)
    right_det = np.ones(num_frames, dtype=bool)
    left_conf = np.full(num_frames, 0.9, dtype=np.float32)
    right_conf = np.full(num_frames, 0.85, dtype=np.float32)

    return SequenceLandmarks(
        left_hand=left_hand,
        right_hand=right_hand,
        left_detected=left_det,
        right_detected=right_det,
        left_confidence=left_conf,
        right_confidence=right_conf,
    )


class TestTemporalInterpolation:
    """Test suite validating 1D multidimensional interpolation."""

    def test_interpolate_array_1d_linear_upsampling(self) -> None:
        arr = np.array([0.0, 10.0, 20.0], dtype=np.float32).reshape(3, 1)
        resampled = interpolate_array_1d_temporal(arr, target_length=5, method="linear")

        assert resampled.shape == (5, 1)
        assert np.isclose(resampled[0, 0], 0.0)
        assert np.isclose(resampled[-1, 0], 20.0)
        assert np.isclose(resampled[2, 0], 10.0)

    def test_interpolate_array_1d_downsampling(self) -> None:
        arr = np.linspace(0, 100, 50, dtype=np.float32).reshape(50, 1)
        resampled = interpolate_array_1d_temporal(arr, target_length=25, method="linear")

        assert resampled.shape == (25, 1)
        assert np.isclose(resampled[0, 0], 0.0)
        assert np.isclose(resampled[-1, 0], 100.0)

    def test_interpolate_array_single_frame(self) -> None:
        arr = np.array([[5.0, 10.0]], dtype=np.float32)
        resampled = interpolate_array_1d_temporal(arr, target_length=10, method="linear")

        assert resampled.shape == (10, 2)
        assert np.all(resampled[:, 0] == 5.0)
        assert np.all(resampled[:, 1] == 10.0)

    def test_interpolate_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="0 temporal frames"):
            interpolate_array_1d_temporal(np.zeros((0, 3)), target_length=10)


class TestTemporalTrajectoryInterpolator:
    """Test suite validating SequenceLandmarks trajectory resampling."""

    def test_resample_sequence_linear_preserves_bounds(self) -> None:
        interpolator = TemporalTrajectoryInterpolator(target_length=30, method="linear")
        seq = _create_mock_trajectory(num_frames=15)
        res = interpolator.resample_sequence(seq)

        assert isinstance(res, ResampledTrajectory)
        assert res.target_frames == 30
        assert res.original_frames == 15
        assert res.sequence.num_frames == 30

        # Verify endpoint preservation for wrist coordinates
        assert np.isclose(res.sequence.left_hand[0, 0, 0], seq.left_hand[0, 0, 0], atol=1e-4)
        assert np.isclose(res.sequence.left_hand[-1, 0, 0], seq.left_hand[-1, 0, 0], atol=1e-4)

    def test_resample_sequence_cubic_smoothness(self) -> None:
        interpolator = TemporalTrajectoryInterpolator(target_length=40, method="cubic")
        seq = _create_mock_trajectory(num_frames=20)
        res = interpolator.resample_sequence(seq)

        assert res.sequence.num_frames == 40
        assert res.velocities.shape == (40, 2, 21, 3)
        assert res.accelerations.shape == (40, 2, 21, 3)

    def test_resample_sequence_nearest_mode(self) -> None:
        interpolator = TemporalTrajectoryInterpolator(target_length=20, method="nearest")
        seq = _create_mock_trajectory(num_frames=10)
        res = interpolator.resample_sequence(seq)

        assert res.sequence.num_frames == 20
        assert res.method == "nearest"

    def test_temporal_derivatives_calculation(self) -> None:
        # Array with linear progression: x = 10 * t -> velocity should be 10.0, acc = 0.0
        dt = 0.1
        t = np.arange(10, dtype=np.float32) * dt
        coords = (10.0 * t).reshape(10, 1)

        vel, acc = compute_temporal_derivatives(coords, dt=dt)
        assert np.allclose(vel[1:-1], 10.0, atol=1e-4)
        assert np.allclose(acc[1:-1], 0.0, atol=1e-4)

    def test_uniform_temporal_subsample(self) -> None:
        seq = _create_mock_trajectory(num_frames=50)
        sub = uniform_temporal_subsample(seq, target_length=25)

        assert sub.num_frames == 25
        assert np.allclose(sub.left_hand[0], seq.left_hand[0])
        assert np.allclose(sub.left_hand[-1], seq.left_hand[-1])

