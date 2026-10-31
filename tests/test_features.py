"""Comprehensive unit test suite for feature engineering pipeline (SANKET-027)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sanket.features import (
    ABDUCTION_ANGLE_PAIRS,
    FLEXION_ANGLE_TRIPLETS,
    INTER_HAND_DISTANCE_PAIRS,
    NUM_HAND_ANGLES,
    NUM_HAND_DISTANCES,
    PAIRWISE_DISTANCE_PAIRS,
    FeatureExtractor,
    FeatureVector,
    aggregate_sequence_features,
    compute_angle_between_vectors,
    compute_frame_velocities,
    compute_joint_angles,
    compute_pairwise_distances,
    compute_temporal_statistics,
    get_hand_angle_names,
    get_hand_distance_names,
)
from sanket.vision.landmarks import (
    COORDINATE_DIM,
    NUM_HAND_LANDMARKS,
    FrameLandmarks,
    SequenceLandmarks,
    save_landmarks_npz,
)


@pytest.fixture
def synthetic_hand_coords() -> np.ndarray:
    """Create a realistic synthetic 21-landmark hand coordinate array."""
    coords = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    # Wrist at origin
    coords[0] = [0.0, 0.0, 0.0]
    # Thumb (1-4) extending diagonally
    coords[1] = [0.05, -0.05, 0.0]
    coords[2] = [0.10, -0.10, 0.0]
    coords[3] = [0.15, -0.12, 0.0]
    coords[4] = [0.20, -0.15, 0.0]
    # Index finger (5-8) extending upwards along negative y
    coords[5] = [0.05, -0.20, 0.0]
    coords[6] = [0.05, -0.30, 0.0]
    coords[7] = [0.05, -0.38, 0.0]
    coords[8] = [0.05, -0.45, 0.0]
    # Middle finger (9-12) along negative y
    coords[9] = [0.0, -0.22, 0.0]
    coords[10] = [0.0, -0.34, 0.0]
    coords[11] = [0.0, -0.42, 0.0]
    coords[12] = [0.0, -0.50, 0.0]
    # Ring finger (13-16)
    coords[13] = [-0.05, -0.20, 0.0]
    coords[14] = [-0.05, -0.30, 0.0]
    coords[15] = [-0.05, -0.38, 0.0]
    coords[16] = [-0.05, -0.45, 0.0]
    # Pinky finger (17-20)
    coords[17] = [-0.09, -0.16, 0.0]
    coords[18] = [-0.10, -0.24, 0.0]
    coords[19] = [-0.11, -0.30, 0.0]
    coords[20] = [-0.12, -0.36, 0.0]
    return coords


@pytest.fixture
def synthetic_sequence_landmarks(synthetic_hand_coords: np.ndarray) -> SequenceLandmarks:
    """Create a temporal sequence of 10 landmark frames."""
    t = 10
    left_hand = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right_hand = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    for i in range(t):
        # Apply slight temporal drift
        drift = float(i) * 0.01
        right_hand[i] = synthetic_hand_coords + np.array([drift, -drift, 0.0], dtype=np.float32)
        # Left hand also present
        left_hand[i] = (synthetic_hand_coords * 0.8) + np.array(
            [-0.3 + drift, 0.0, 0.0], dtype=np.float32
        )

    return SequenceLandmarks(
        left_hand=left_hand,
        right_hand=right_hand,
        left_detected=np.ones(t, dtype=bool),
        right_detected=np.ones(t, dtype=bool),
        left_confidence=np.full(t, 0.90, dtype=np.float32),
        right_confidence=np.full(t, 0.92, dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# 1. Geometry Feature Tests (Angles & Pairwise Distances)
# -----------------------------------------------------------------------------


def test_compute_angle_between_vectors() -> None:
    """Verify angular calculations on orthogonal, parallel, and zero vectors."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 90 degrees = pi / 2
    angle_right = compute_angle_between_vectors(v1, v2)
    assert pytest.approx(angle_right, abs=1e-5) == np.pi / 2

    # Parallel = 0 radians
    angle_parallel = compute_angle_between_vectors(v1, v1)
    assert pytest.approx(angle_parallel, abs=1e-5) == 0.0

    # Opposite = pi radians
    angle_opposite = compute_angle_between_vectors(v1, -v1)
    assert pytest.approx(angle_opposite, abs=1e-5) == np.pi

    # Zero vector returns 0.0 without crashing or raising ZeroDivisionError
    angle_zero = compute_angle_between_vectors(v1, np.zeros(3, dtype=np.float32))
    assert angle_zero == 0.0


def test_compute_joint_angles_blank() -> None:
    """Verify joint angles on all-zero hand return an array of zeros."""
    blank = np.zeros((21, 3), dtype=np.float32)
    angles = compute_joint_angles(blank)
    assert angles.shape == (NUM_HAND_ANGLES,)
    assert np.all(angles == 0.0)


def test_compute_joint_angles_synthetic(synthetic_hand_coords: np.ndarray) -> None:
    """Verify joint angle calculations on articulated hand skeleton."""
    angles = compute_joint_angles(synthetic_hand_coords)
    assert angles.shape == (NUM_HAND_ANGLES,)
    assert np.all(np.isfinite(angles))
    # All angles in radians [0, pi]
    assert np.all((angles >= 0.0) & (angles <= np.pi + 1e-5))

    # Test invalid shape raises ValueError
    with pytest.raises(ValueError, match="Expected shape"):
        compute_joint_angles(np.zeros((20, 3), dtype=np.float32))


def test_compute_pairwise_distances_translation_invariance(
    synthetic_hand_coords: np.ndarray,
) -> None:
    """Prove that pairwise Euclidean distances are strictly translation invariant."""
    base_dist = compute_pairwise_distances(synthetic_hand_coords)
    assert base_dist.shape == (NUM_HAND_DISTANCES,)

    # Apply spatial 3D translation
    shift = np.array([150.0, -320.0, 85.0], dtype=np.float32)
    translated = synthetic_hand_coords + shift
    trans_dist = compute_pairwise_distances(translated)

    # Distances must be identical within float32 precision
    assert np.allclose(base_dist, trans_dist, atol=1e-4)

    # For localized camera shift, precision is even tighter
    local_shift = np.array([0.5, -0.8, 0.2], dtype=np.float32)
    assert np.allclose(
        base_dist,
        compute_pairwise_distances(synthetic_hand_coords + local_shift),
        atol=1e-5,
    )


def test_compute_pairwise_distances_blank() -> None:
    """Verify pairwise distances on blank hand return zeros."""
    blank = np.zeros((21, 3), dtype=np.float32)
    dist = compute_pairwise_distances(blank)
    assert dist.shape == (NUM_HAND_DISTANCES,)
    assert np.all(dist == 0.0)

    with pytest.raises(ValueError, match="Expected shape"):
        compute_pairwise_distances(np.zeros((21, 2), dtype=np.float32))


def test_angle_and_distance_names() -> None:
    """Verify name generation matches configured constants."""
    a_names = get_hand_angle_names(prefix="right")
    d_names = get_hand_distance_names(prefix="right")

    assert len(a_names) == NUM_HAND_ANGLES
    assert len(d_names) == NUM_HAND_DISTANCES
    assert all(name.startswith("right_") for name in a_names)
    assert all(name.startswith("right_") for name in d_names)
    assert len(FLEXION_ANGLE_TRIPLETS) + len(ABDUCTION_ANGLE_PAIRS) == NUM_HAND_ANGLES
    assert len(PAIRWISE_DISTANCE_PAIRS) == NUM_HAND_DISTANCES


# -----------------------------------------------------------------------------
# 2. Temporal Dynamics and Statistics Tests
# -----------------------------------------------------------------------------


def test_compute_frame_velocities() -> None:
    """Verify velocity extraction between consecutive frames."""
    features = np.array(
        [[1.0, 10.0], [2.0, 20.0], [4.0, 25.0]],
        dtype=np.float32,
    )
    vel = compute_frame_velocities(features)
    assert vel.shape == (2, 2)
    expected = np.array([[1.0, 10.0], [2.0, 5.0]], dtype=np.float32)
    assert np.allclose(vel, expected)

    # Single frame returns empty velocity
    single = np.array([[1.0, 2.0]], dtype=np.float32)
    vel_single = compute_frame_velocities(single)
    assert vel_single.shape == (0, 2)

    with pytest.raises(ValueError, match="Expected 2D array"):
        compute_frame_velocities(np.zeros((3, 2, 1), dtype=np.float32))


def test_compute_temporal_statistics() -> None:
    """Verify statistical moments calculation across time."""
    features = np.array(
        [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]],
        dtype=np.float32,
    )
    stats = compute_temporal_statistics(features)
    assert np.allclose(stats["mean"], [3.0, 20.0])
    assert np.allclose(stats["min"], [1.0, 10.0])
    assert np.allclose(stats["max"], [5.0, 30.0])
    assert np.allclose(stats["delta"], [4.0, 20.0])  # 5-1, 30-10
    assert np.allclose(stats["max_velocity"], [2.0, 10.0])

    # Test with boolean validity mask
    mask = np.array([True, False, True])
    stats_masked = compute_temporal_statistics(features, mask=mask)
    assert np.allclose(stats_masked["mean"], [3.0, 20.0])  # mean of [1, 10] and [5, 30]


def test_compute_temporal_statistics_empty() -> None:
    """Verify statistics on zero-length sequence."""
    empty = np.zeros((0, 4), dtype=np.float32)
    stats = compute_temporal_statistics(empty)
    assert stats["mean"].shape == (4,)
    assert np.all(stats["mean"] == 0.0)

    # All masked out
    features = np.ones((3, 2), dtype=np.float32)
    mask = np.array([False, False, False])
    stats_none = compute_temporal_statistics(features, mask=mask)
    assert np.all(stats_none["mean"] == 0.0)


def test_aggregate_sequence_features() -> None:
    """Verify concatenation of requested statistical moments."""
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vec = aggregate_sequence_features(features, stats=("mean", "std"))
    assert vec.shape == (4,)

    # Unknown statistic raises ValueError
    with pytest.raises(ValueError, match="Unknown statistic"):
        aggregate_sequence_features(features, stats=("unknown_stat",))


# -----------------------------------------------------------------------------
# 3. FeatureVector Data Container Tests
# -----------------------------------------------------------------------------


def test_feature_vector_container() -> None:
    """Verify FeatureVector instantiation, validation, and dictionary serialization."""
    f = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    names = ["feat_1", "feat_2", "feat_3"]

    fvec = FeatureVector(features=f, feature_names=names, pooling_strategy="mean")
    assert fvec.dimension == 3
    d = fvec.to_dict()
    assert d == {"feat_1": 1.5, "feat_2": 2.5, "feat_3": 3.5}

    # Mismatched name length raises ValueError
    with pytest.raises(ValueError, match="does not match"):
        FeatureVector(features=f, feature_names=["feat_1", "feat_2"])


# -----------------------------------------------------------------------------
# 4. FeatureExtractor Pipeline Tests
# -----------------------------------------------------------------------------


def test_feature_extractor_dimensions() -> None:
    """Verify frame and sequence feature vector dimensionality."""
    extractor = FeatureExtractor(
        include_coordinates=True,  # 126
        include_angles=True,  # 38
        include_distances=True,  # 48
        include_inter_hand=True,  # 3
        pooling_strategy="mean_std_delta",  # 3x
    )

    # Frame dim = 126 + 38 + 48 + 3 = 215
    assert extractor.frame_dimension == 215
    # Output dim = 215 * 3 = 645
    assert extractor.output_dimension == 645
    assert len(extractor.get_feature_names()) == 645
    assert len(INTER_HAND_DISTANCE_PAIRS) == 3


def test_feature_extractor_frame(synthetic_hand_coords: np.ndarray) -> None:
    """Verify feature extraction on a single FrameLandmarks object."""
    extractor = FeatureExtractor(normalize_input=True)

    frame = FrameLandmarks(
        left_hand=np.zeros((21, 3), dtype=np.float32),
        right_hand=synthetic_hand_coords,
        left_detected=False,
        right_detected=True,
        left_confidence=0.0,
        right_confidence=0.95,
    )

    feat = extractor.extract_frame_features(frame)
    assert feat.shape == (extractor.frame_dimension,)
    assert np.all(np.isfinite(feat))

    # Inter-hand distance should be zero when one hand is absent
    inter_hand_slice = feat[-3:]
    assert np.all(inter_hand_slice == 0.0)


def test_feature_extractor_sequence(synthetic_sequence_landmarks: SequenceLandmarks) -> None:
    """Verify feature extraction on temporal SequenceLandmarks."""
    extractor = FeatureExtractor(
        normalize_input=True,
        pooling_strategy="mean_std_delta",
    )

    fvec = extractor.extract_sequence_features(synthetic_sequence_landmarks)
    assert isinstance(fvec, FeatureVector)
    assert fvec.dimension == extractor.output_dimension
    assert np.all(np.isfinite(fvec.features))
    assert fvec.metadata["num_frames"] == synthetic_sequence_landmarks.num_frames
    assert fvec.metadata["valid_frames"] == synthetic_sequence_landmarks.num_frames

    # Determinism check
    fvec2 = extractor.extract_sequence_features(synthetic_sequence_landmarks)
    assert np.allclose(fvec.features, fvec2.features)


def test_feature_extractor_pooling_none(
    synthetic_sequence_landmarks: SequenceLandmarks,
) -> None:
    """Verify pooling_strategy='none' returns full 2D temporal feature matrix."""
    extractor = FeatureExtractor(pooling_strategy="none")
    fvec = extractor.extract_sequence_features(synthetic_sequence_landmarks)

    assert fvec.features.ndim == 2
    assert fvec.features.shape == (
        synthetic_sequence_landmarks.num_frames,
        extractor.frame_dimension,
    )
    assert fvec.dimension == extractor.frame_dimension
    assert fvec.features.size == synthetic_sequence_landmarks.num_frames * extractor.frame_dimension


def test_feature_extractor_empty_sequence() -> None:
    """Verify handling of zero-frame sequence returns zeros."""
    empty_seq = SequenceLandmarks(
        left_hand=np.zeros((0, 21, 3), dtype=np.float32),
        right_hand=np.zeros((0, 21, 3), dtype=np.float32),
        left_detected=np.zeros(0, dtype=bool),
        right_detected=np.zeros(0, dtype=bool),
        left_confidence=np.zeros(0, dtype=np.float32),
        right_confidence=np.zeros(0, dtype=np.float32),
    )

    extractor = FeatureExtractor()
    fvec = extractor.extract_sequence_features(empty_seq)
    assert fvec.dimension == extractor.output_dimension
    assert np.all(fvec.features == 0.0)


def test_temporal_edge_cases() -> None:
    """Verify edge conditions in temporal statistics calculations."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # Invalid 1D array raises ValueError
    with pytest.raises(ValueError, match="Expected 2D array"):
        compute_temporal_statistics(np.zeros(5, dtype=np.float32))

    # Invalid mask shape raises ValueError
    with pytest.raises(ValueError, match="Mask shape"):
        compute_temporal_statistics(data, mask=np.array([True]))

    # Single frame has 0.0 max velocity
    single = np.array([[5.0, 10.0]], dtype=np.float32)
    res = compute_temporal_statistics(single)
    assert np.all(res["max_velocity"] == 0.0)

    # Empty stats returns empty array
    empty_agg = aggregate_sequence_features(data, stats=())
    assert empty_agg.shape == (0,)


def test_feature_extractor_edge_cases() -> None:
    """Verify edge conditions in FeatureVector and FeatureExtractor."""
    # 2D to_dict() raises ValueError
    matrix_vec = FeatureVector(
        features=np.zeros((5, 2), dtype=np.float32),
        feature_names=["f1", "f2"],
        pooling_strategy="none",
    )
    with pytest.raises(ValueError, match="only supported for 1D"):
        matrix_vec.to_dict()

    # Mismatched 2D columns raises ValueError
    with pytest.raises(ValueError, match="Feature columns"):
        FeatureVector(
            features=np.zeros((5, 3), dtype=np.float32),
            feature_names=["f1", "f2"],
        )

    # FeatureExtractor with all flags False
    bare_extractor = FeatureExtractor(
        include_coordinates=False,
        include_angles=False,
        include_distances=False,
        include_inter_hand=False,
    )
    assert bare_extractor.frame_dimension == 0
    frame = FrameLandmarks(
        left_hand=np.zeros((21, 3), dtype=np.float32),
        right_hand=np.zeros((21, 3), dtype=np.float32),
        left_detected=False,
        right_detected=False,
        left_confidence=0.0,
        right_confidence=0.0,
    )
    assert bare_extractor.extract_frame_features(frame).shape == (0,)

    # get_feature_names with pooling_strategy='none'
    extractor_none = FeatureExtractor(pooling_strategy="none")
    assert extractor_none.get_feature_names() == extractor_none.get_frame_feature_names()


# -----------------------------------------------------------------------------
# 5. CLI Script Execution Tests
# -----------------------------------------------------------------------------


def test_cli_verify() -> None:
    """Verify scripts/generate_features.py --verify runs and passes."""
    cmd = [sys.executable, "scripts/generate_features.py", "--verify"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert "Verification successful" in res.stderr or "Verification successful" in res.stdout


def test_cli_batch_feature_generation(
    synthetic_sequence_landmarks: SequenceLandmarks,
    tmp_path: Path,
) -> None:
    """Verify batch feature generation across directory with split manifest."""
    # Setup landmark files
    landmarks_dir = tmp_path / "landmarks"
    class_dir = landmarks_dir / "HELLO"
    class_dir.mkdir(parents=True)

    seq_file1 = class_dir / "Signer01_HELLO_1.npz"
    seq_file2 = class_dir / "Signer02_HELLO_2.npz"
    save_landmarks_npz(synthetic_sequence_landmarks, seq_file1)
    save_landmarks_npz(synthetic_sequence_landmarks, seq_file2)

    # Setup dummy split manifest
    splits_file = tmp_path / "splits.json"
    manifest_data = {
        "partitions": {
            "train": {"samples": ["Signer02_HELLO_2"]},
            "val": {"samples": ["Signer01_HELLO_1"]},
            "test": {"samples": []},
        }
    }
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    out_dir = tmp_path / "features_out"

    cmd = [
        sys.executable,
        "scripts/generate_features.py",
        "--input-dir",
        str(landmarks_dir),
        "--output-dir",
        str(out_dir),
        "--splits-file",
        str(splits_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"

    # Verify generated artifacts
    assert (out_dir / "feature_names.json").is_file()
    assert (out_dir / "class_to_index.json").is_file()
    assert (out_dir / "dataset_summary.json").is_file()
    assert (out_dir / "X_train.npy").is_file()
    assert (out_dir / "y_train.npy").is_file()
    assert (out_dir / "X_val.npy").is_file()
    assert (out_dir / "y_val.npy").is_file()

    # Load and verify shapes
    x_train = np.load(out_dir / "X_train.npy")
    y_train = np.load(out_dir / "y_train.npy")
    assert x_train.shape == (1, 645)
    assert y_train.shape == (1,)
