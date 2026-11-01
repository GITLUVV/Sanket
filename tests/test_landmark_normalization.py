"""Unit test suite for skeletal landmark coordinate normalization (SANKET-026)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sanket.vision.landmarks import (
    COORDINATE_DIM,
    NUM_HAND_LANDMARKS,
    FrameLandmarks,
    SequenceLandmarks,
    load_landmarks_npz,
    save_landmarks_npz,
)
from sanket.vision.normalization import (
    MIDDLE_MCP_IDX,
    WRIST_IDX,
    LandmarkNormalizer,
    denormalize_hand_landmarks,
    normalize_frame_landmarks,
    normalize_hand_landmarks,
    normalize_sequence_landmarks,
)


@pytest.fixture
def sample_hand() -> np.ndarray:
    """Generate a realistic synthetic 21-point hand landmark array (21, 3)."""
    np.random.seed(123)
    # Wrist at (0.5, 0.7, 0.0)
    wrist = np.array([0.5, 0.7, 0.0], dtype=np.float32)
    # 21 points with relative offsets
    offsets = np.random.uniform(-0.15, 0.15, size=(NUM_HAND_LANDMARKS, COORDINATE_DIM)).astype(
        np.float32
    )
    offsets[WRIST_IDX] = 0.0
    # Ensure middle MCP (L_9) has clear upward offset from wrist
    offsets[MIDDLE_MCP_IDX] = np.array([0.02, -0.12, 0.01], dtype=np.float32)
    return wrist + offsets


# -----------------------------------------------------------------------------
# 1. Wrist Origin Centering & Palm Scale Normalization Tests
# -----------------------------------------------------------------------------


def test_wrist_origin_centering(sample_hand: np.ndarray) -> None:
    """Verify that normalized wrist landmark is anchored at the exact origin (0, 0, 0)."""
    norm_hand, meta = normalize_hand_landmarks(sample_hand, method="palm")
    assert norm_hand.shape == (NUM_HAND_LANDMARKS, COORDINATE_DIM)
    assert meta["is_valid"] is True
    assert np.allclose(norm_hand[WRIST_IDX], np.zeros(3, dtype=np.float32), atol=1e-6)


def test_palm_scale_normalization(sample_hand: np.ndarray) -> None:
    """Verify that palm vector (wrist to middle MCP) length equals exactly 1.0."""
    norm_hand, _ = normalize_hand_landmarks(sample_hand, method="palm")
    palm_len = float(np.linalg.norm(norm_hand[MIDDLE_MCP_IDX] - norm_hand[WRIST_IDX]))
    assert pytest.approx(palm_len, abs=1e-5) == 1.0


def test_max_span_scale_normalization(sample_hand: np.ndarray) -> None:
    """Verify 'max_span' normalization scales maximum joint distance to 1.0."""
    norm_hand, meta = normalize_hand_landmarks(sample_hand, method="max_span")
    assert meta["method"] == "max_span"
    max_dist = float(np.max(np.linalg.norm(norm_hand, axis=1)))
    assert pytest.approx(max_dist, abs=1e-5) == 1.0


def test_bbox_scale_normalization(sample_hand: np.ndarray) -> None:
    """Verify 'bbox' normalization scales max bounding box range to 1.0."""
    norm_hand, meta = normalize_hand_landmarks(sample_hand, method="bbox")
    assert meta["method"] == "bbox"
    max_range = float(np.max(np.ptp(norm_hand, axis=0)))
    assert pytest.approx(max_range, abs=1e-5) == 1.0


def test_invalid_normalization_method(sample_hand: np.ndarray) -> None:
    """Verify that unsupported method raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_hand_landmarks(sample_hand, method="unsupported_method")


# -----------------------------------------------------------------------------
# 2. Synthetic Perturbation Invariance Proof Tests
# -----------------------------------------------------------------------------


def test_translation_invariance(sample_hand: np.ndarray) -> None:
    """Prove numerical invariance under arbitrary 3D spatial translation."""
    base_norm, _ = normalize_hand_landmarks(sample_hand, method="palm")

    # Shift hand by arbitrary 3D translation vector
    shift = np.array([42.75, -88.33, 19.12], dtype=np.float32)
    translated_hand = sample_hand + shift

    trans_norm, _ = normalize_hand_landmarks(translated_hand, method="palm")

    # Assert invariant within float32 numerical precision (atol=1e-4)
    assert np.allclose(trans_norm, base_norm, atol=1e-4)

    # For localized camera translations, precision is even tighter (atol=1e-5)
    local_shift = np.array([0.45, -0.25, 0.10], dtype=np.float32)
    local_norm, _ = normalize_hand_landmarks(sample_hand + local_shift, method="palm")
    assert np.allclose(local_norm, base_norm, atol=1e-5)


def test_scaling_invariance(sample_hand: np.ndarray) -> None:
    """Prove numerical invariance under arbitrary uniform scaling (camera distance)."""
    base_norm, _ = normalize_hand_landmarks(sample_hand, method="palm")

    # Uniform scale factors: small (distant signer) and large (close signer)
    scaled_small = sample_hand * 0.25
    scaled_large = sample_hand * 5.80

    norm_small, _ = normalize_hand_landmarks(scaled_small, method="palm")
    norm_large, _ = normalize_hand_landmarks(scaled_large, method="palm")

    assert np.allclose(norm_small, base_norm, atol=1e-5)
    assert np.allclose(norm_large, base_norm, atol=1e-5)


def test_combined_translation_and_scale_invariance(sample_hand: np.ndarray) -> None:
    """Prove numerical invariance under combined translation and scaling."""
    base_norm, _ = normalize_hand_landmarks(sample_hand, method="palm")

    perturbed = (sample_hand * 3.14) + np.array([-15.0, 30.0, 7.5], dtype=np.float32)
    pert_norm, _ = normalize_hand_landmarks(perturbed, method="palm")

    assert np.allclose(pert_norm, base_norm, atol=1e-5)


def test_rotation_alignment_invariance(sample_hand: np.ndarray) -> None:
    """Prove in-plane 2D rotational alignment brings middle MCP to x=0."""
    norm_aligned, meta = normalize_hand_landmarks(
        sample_hand, method="palm", align_rotation=True
    )
    # Middle MCP x-coordinate should be 0.0, y-coordinate negative (pointing up)
    assert pytest.approx(norm_aligned[MIDDLE_MCP_IDX, 0], abs=1e-5) == 0.0
    assert norm_aligned[MIDDLE_MCP_IDX, 1] < 0.0

    # Test that pre-rotating the input yields the identical aligned output
    theta = np.deg2rad(45.0)
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    rotated_hand = np.dot(sample_hand, rot_matrix.T)

    rot_norm, _ = normalize_hand_landmarks(rotated_hand, method="palm", align_rotation=True)
    assert np.allclose(rot_norm, norm_aligned, atol=1e-5)


# -----------------------------------------------------------------------------
# 3. Inversion (Denormalization) Tests
# -----------------------------------------------------------------------------


def test_denormalization_exact_reconstruction(sample_hand: np.ndarray) -> None:
    """Verify that denormalize_hand_landmarks accurately recovers the original coordinates."""
    norm_hand, meta = normalize_hand_landmarks(
        sample_hand, method="palm", align_rotation=True
    )
    recovered = denormalize_hand_landmarks(norm_hand, meta)
    assert np.allclose(recovered, sample_hand, atol=1e-5)


def test_denormalization_undetected_noop() -> None:
    """Verify denormalization on invalid/undetected metadata is a clean no-op."""
    zeros = np.zeros((21, 3), dtype=np.float32)
    meta = {"is_valid": False}
    recovered = denormalize_hand_landmarks(zeros, meta)
    assert np.array_equal(recovered, zeros)


# -----------------------------------------------------------------------------
# 4. Degenerate & Edge Case Handling Tests
# -----------------------------------------------------------------------------


def test_normalize_undetected_all_zeros() -> None:
    """Verify that an undetected hand (all zeros) returns all zeros safely."""
    zeros = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    norm, meta = normalize_hand_landmarks(zeros)
    assert np.array_equal(norm, zeros)
    assert meta["is_valid"] is False


def test_normalize_zero_scale_degenerate() -> None:
    """Verify collapsed hand coordinates (all points at same position) do not divide by zero."""
    collapsed = np.full((NUM_HAND_LANDMARKS, COORDINATE_DIM), 0.5, dtype=np.float32)
    norm, meta = normalize_hand_landmarks(collapsed, min_scale=1e-4)
    # Wrist centered means all points become 0.0, scale fallback to 1.0
    assert np.all(norm == 0.0)
    assert meta["scale_factor"] == 1.0


def test_normalize_invalid_inputs() -> None:
    """Verify invalid shapes and empty arrays raise ValueError."""
    with pytest.raises(ValueError, match="Cannot normalize None or empty"):
        normalize_hand_landmarks(np.array([]))

    with pytest.raises(ValueError, match="Expected shape"):
        normalize_hand_landmarks(np.zeros((15, 3), dtype=np.float32))


# -----------------------------------------------------------------------------
# 5. Frame & Sequence Normalization Tests
# -----------------------------------------------------------------------------


def test_normalize_frame_landmarks(sample_hand: np.ndarray) -> None:
    """Test normalizing a FrameLandmarks object with active right hand and inactive left hand."""
    frame = FrameLandmarks(
        left_hand=np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32),
        right_hand=sample_hand,
        left_detected=False,
        right_detected=True,
        left_confidence=0.0,
        right_confidence=0.92,
    )
    norm_frame = normalize_frame_landmarks(frame, method="palm")
    assert np.all(norm_frame.left_hand == 0.0)
    assert np.allclose(norm_frame.right_hand[WRIST_IDX], 0.0, atol=1e-6)
    palm_len = float(
        np.linalg.norm(norm_frame.right_hand[MIDDLE_MCP_IDX] - norm_frame.right_hand[WRIST_IDX])
    )
    assert pytest.approx(palm_len, abs=1e-5) == 1.0


def test_normalize_sequence_landmarks(sample_hand: np.ndarray) -> None:
    """Test normalizing a SequenceLandmarks object across multiple frames."""
    t = 6
    left_seq = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right_seq = np.array([sample_hand * (1.0 + i * 0.1) for i in range(t)])

    seq = SequenceLandmarks(
        left_hand=left_seq,
        right_hand=right_seq,
        left_detected=np.zeros(t, dtype=bool),
        right_detected=np.ones(t, dtype=bool),
        left_confidence=np.zeros(t, dtype=np.float32),
        right_confidence=np.ones(t, dtype=np.float32),
    )

    norm_seq = normalize_sequence_landmarks(seq, method="palm")
    assert norm_seq.num_frames == t
    for i in range(t):
        assert np.allclose(norm_seq.right_hand[i, WRIST_IDX], 0.0, atol=1e-6)
        palm_len = float(
            np.linalg.norm(
                norm_seq.right_hand[i, MIDDLE_MCP_IDX] - norm_seq.right_hand[i, WRIST_IDX]
            )
        )
        assert pytest.approx(palm_len, abs=1e-5) == 1.0


# -----------------------------------------------------------------------------
# 6. LandmarkNormalizer Pipeline Class Tests
# -----------------------------------------------------------------------------


def test_landmark_normalizer_class(sample_hand: np.ndarray) -> None:
    """Test LandmarkNormalizer class encapsulation."""
    normalizer = LandmarkNormalizer(method="palm", align_rotation=True)
    norm_hand, meta = normalizer.normalize_hand(sample_hand)
    assert pytest.approx(norm_hand[MIDDLE_MCP_IDX, 0], abs=1e-5) == 0.0

    denorm = normalizer.denormalize_hand(norm_hand, meta)
    assert np.allclose(denorm, sample_hand, atol=1e-5)


# -----------------------------------------------------------------------------
# 7. CLI Script Execution Tests
# -----------------------------------------------------------------------------


def test_cli_verify_invariance() -> None:
    """Test CLI script with --verify-invariance flag."""
    cmd = [sys.executable, "scripts/normalize_landmarks.py", "--verify-invariance"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert "SUCCESS: All coordinate invariance tests PASSED" in res.stderr or res.stdout


def test_cli_normalize_single_file(sample_hand: np.ndarray, tmp_path: Path) -> None:
    """Test CLI script normalizing a single .npz landmark file."""
    seq = SequenceLandmarks(
        left_hand=np.zeros((3, 21, 3), dtype=np.float32),
        right_hand=np.stack([sample_hand, sample_hand * 1.5, sample_hand * 0.8]),
        left_detected=np.zeros(3, dtype=bool),
        right_detected=np.ones(3, dtype=bool),
        left_confidence=np.zeros(3, dtype=np.float32),
        right_confidence=np.ones(3, dtype=np.float32),
    )
    raw_npz = tmp_path / "raw_landmarks.npz"
    save_landmarks_npz(seq, raw_npz)

    out_dir = tmp_path / "norm_out"
    cmd = [
        sys.executable,
        "scripts/normalize_landmarks.py",
        "--input",
        str(raw_npz),
        "--output-dir",
        str(out_dir),
        "--method",
        "palm",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"

    dest_file = out_dir / "raw_landmarks_norm.npz"
    assert dest_file.is_file()
    loaded_seq, meta = load_landmarks_npz(dest_file)
    assert loaded_seq.num_frames == 3
    assert meta["normalization_method"] == "palm"


def test_cli_normalize_batch_dir(sample_hand: np.ndarray, tmp_path: Path) -> None:
    """Test CLI script batch-normalizing a directory."""
    in_dir = tmp_path / "batch_in"
    in_dir.mkdir()
    seq = SequenceLandmarks(
        left_hand=np.zeros((2, 21, 3), dtype=np.float32),
        right_hand=np.stack([sample_hand, sample_hand]),
        left_detected=np.zeros(2, dtype=bool),
        right_detected=np.ones(2, dtype=bool),
        left_confidence=np.zeros(2, dtype=np.float32),
        right_confidence=np.ones(2, dtype=np.float32),
    )
    save_landmarks_npz(seq, in_dir / "clip1.npz")
    save_landmarks_npz(seq, in_dir / "clip2.npz")

    out_dir = tmp_path / "batch_norm_out"
    cmd = [
        sys.executable,
        "scripts/normalize_landmarks.py",
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert (out_dir / "clip1_norm.npz").is_file()
    assert (out_dir / "clip2_norm.npz").is_file()
