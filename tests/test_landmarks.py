"""Unit test suite for MediaPipe hand landmark extraction (SANKET-024)."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from sanket.vision.landmarks import (
    COORDINATE_DIM,
    HAND_FEATURE_DIM,
    NUM_HAND_LANDMARKS,
    TOTAL_FEATURE_DIM,
    FrameLandmarks,
    HandLandmarkExtractor,
    SequenceLandmarks,
    ensure_hand_landmarker_model,
    load_landmarks_npz,
    save_landmarks_npz,
)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a temporary valid MP4 video with 10 synthetic frames."""
    video_path = tmp_path / "test_clip.mp4"
    width, height = 160, 120
    fps = 10.0
    num_frames = 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20, 100, 150), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


# -----------------------------------------------------------------------------
# 1. FrameLandmarks Container Tests
# -----------------------------------------------------------------------------


def test_frame_landmarks_flatten() -> None:
    """Verify flat vector flattening and dimensions."""
    left = np.full((NUM_HAND_LANDMARKS, COORDINATE_DIM), 1.0, dtype=np.float32)
    right = np.full((NUM_HAND_LANDMARKS, COORDINATE_DIM), 2.0, dtype=np.float32)

    frame = FrameLandmarks(
        left_hand=left,
        right_hand=right,
        left_detected=True,
        right_detected=True,
        left_confidence=0.95,
        right_confidence=0.88,
    )

    vec = frame.to_flat_vector()
    assert vec.shape == (TOTAL_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(vec[:HAND_FEATURE_DIM] == 1.0)
    assert np.all(vec[HAND_FEATURE_DIM:] == 2.0)


# -----------------------------------------------------------------------------
# 2. SequenceLandmarks Imputation and Matrix Tests
# -----------------------------------------------------------------------------


def test_sequence_landmarks_flat_matrix() -> None:
    """Verify 2D matrix transformation for sequence of T frames."""
    t = 15
    left = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    seq = SequenceLandmarks(
        left_hand=left,
        right_hand=right,
        left_detected=np.ones(t, dtype=bool),
        right_detected=np.ones(t, dtype=bool),
        left_confidence=np.ones(t, dtype=np.float32),
        right_confidence=np.ones(t, dtype=np.float32),
    )

    assert seq.num_frames == t
    mat = seq.to_flat_matrix()
    assert mat.shape == (t, TOTAL_FEATURE_DIM)
    assert mat.dtype == np.float32


def test_impute_missing_zero_strategy() -> None:
    """Test 'zero' imputation strategy preserves zero-padding."""
    t = 5
    left = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    left[2] = 1.0

    seq = SequenceLandmarks(
        left_hand=left,
        right_hand=right,
        left_detected=np.array([False, False, True, False, False]),
        right_detected=np.zeros(t, dtype=bool),
        left_confidence=np.zeros(t, dtype=np.float32),
        right_confidence=np.zeros(t, dtype=np.float32),
    )

    imputed = seq.impute_missing(strategy="zero")
    assert np.all(imputed.left_hand[0] == 0.0)
    assert np.all(imputed.left_hand[2] == 1.0)
    assert np.all(imputed.left_hand[3] == 0.0)


def test_impute_missing_carry_forward_strategy() -> None:
    """Test 'carry_forward' temporal forward-fill and backfill."""
    t = 6
    left = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    # Frame 2 detected with value 10.0, Frame 4 detected with value 20.0
    left[2] = 10.0
    left[4] = 20.0
    mask = np.array([False, False, True, False, True, False])

    seq = SequenceLandmarks(
        left_hand=left,
        right_hand=right,  # Completely undetected hand
        left_detected=mask,
        right_detected=np.zeros(t, dtype=bool),
        left_confidence=np.zeros(t, dtype=np.float32),
        right_confidence=np.zeros(t, dtype=np.float32),
    )

    imputed = seq.impute_missing(strategy="carry_forward")

    # Initial frames 0 and 1 should backfill from frame 2 (value 10.0)
    assert np.all(imputed.left_hand[0] == 10.0)
    assert np.all(imputed.left_hand[1] == 10.0)
    # Frame 2 is original 10.0
    assert np.all(imputed.left_hand[2] == 10.0)
    # Frame 3 forward-fills from frame 2 (value 10.0)
    assert np.all(imputed.left_hand[3] == 10.0)
    # Frame 4 is original 20.0
    assert np.all(imputed.left_hand[4] == 20.0)
    # Frame 5 forward-fills from frame 4 (value 20.0)
    assert np.all(imputed.left_hand[5] == 20.0)

    # Completely undetected right hand must remain all zeros
    assert np.all(imputed.right_hand == 0.0)


def test_impute_missing_linear_strategy() -> None:
    """Test 'linear' interpolation between detected keyframes."""
    t = 5
    left = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
    right = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)

    left[0] = 0.0
    left[4] = 40.0
    mask = np.array([True, False, False, False, True])

    seq = SequenceLandmarks(
        left_hand=left,
        right_hand=right,
        left_detected=mask,
        right_detected=np.zeros(t, dtype=bool),
        left_confidence=np.zeros(t, dtype=np.float32),
        right_confidence=np.zeros(t, dtype=np.float32),
    )

    imputed = seq.impute_missing(strategy="linear")
    assert pytest.approx(imputed.left_hand[1, 0, 0]) == 10.0
    assert pytest.approx(imputed.left_hand[2, 0, 0]) == 20.0
    assert pytest.approx(imputed.left_hand[3, 0, 0]) == 30.0


def test_impute_missing_invalid_strategy() -> None:
    """Verify invalid strategy raises ValueError."""
    seq = SequenceLandmarks(
        left_hand=np.zeros((2, 21, 3), dtype=np.float32),
        right_hand=np.zeros((2, 21, 3), dtype=np.float32),
        left_detected=np.zeros(2, dtype=bool),
        right_detected=np.zeros(2, dtype=bool),
        left_confidence=np.zeros(2, dtype=np.float32),
        right_confidence=np.zeros(2, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Invalid strategy"):
        seq.impute_missing(strategy="unknown_strategy")


# -----------------------------------------------------------------------------
# 3. Serialization and Deserialization Tests
# -----------------------------------------------------------------------------


def test_save_and_load_landmarks_npz(tmp_path: Path) -> None:
    """Test saving SequenceLandmarks to compressed .npz archive and reloading."""
    t = 8
    left = np.random.rand(t, NUM_HAND_LANDMARKS, COORDINATE_DIM).astype(np.float32)
    right = np.random.rand(t, NUM_HAND_LANDMARKS, COORDINATE_DIM).astype(np.float32)
    detected = np.ones(t, dtype=bool)
    conf = np.full(t, 0.9, dtype=np.float32)

    seq = SequenceLandmarks(
        left_hand=left,
        right_hand=right,
        left_detected=detected,
        right_detected=detected,
        left_confidence=conf,
        right_confidence=conf,
    )

    archive_path = tmp_path / "landmarks.npz"
    meta_in = {"class_name": "HELLO", "signer": "Signer01", "fps": 30.0}

    saved = save_landmarks_npz(seq, archive_path, metadata=meta_in)
    assert saved.is_file()

    loaded_seq, meta_out = load_landmarks_npz(archive_path)
    assert loaded_seq.num_frames == t
    assert np.allclose(loaded_seq.left_hand, left)
    assert np.allclose(loaded_seq.right_hand, right)
    assert np.array_equal(loaded_seq.left_detected, detected)
    assert meta_out["class_name"] == "HELLO"
    assert meta_out["signer"] == "Signer01"
    assert meta_out["fps"] == 30.0


def test_save_and_load_landmarks_npz_no_metadata(tmp_path: Path) -> None:
    """Test saving SequenceLandmarks with no metadata dictionary."""
    seq = SequenceLandmarks(
        left_hand=np.zeros((3, 21, 3), dtype=np.float32),
        right_hand=np.zeros((3, 21, 3), dtype=np.float32),
        left_detected=np.zeros(3, dtype=bool),
        right_detected=np.zeros(3, dtype=bool),
        left_confidence=np.zeros(3, dtype=np.float32),
        right_confidence=np.zeros(3, dtype=np.float32),
    )
    archive_path = tmp_path / "no_meta.npz"
    saved = save_landmarks_npz(seq, archive_path, metadata=None)
    assert saved.is_file()
    loaded_seq, meta = load_landmarks_npz(archive_path)
    assert loaded_seq.num_frames == 3
    assert meta == {}


def test_load_landmarks_npz_missing(tmp_path: Path) -> None:
    """Test loading non-existent file raises FileNotFoundError."""
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError, match="Landmark archive not found"):
        load_landmarks_npz(missing)


# -----------------------------------------------------------------------------
# 4. Model Caching and Resolution Tests
# -----------------------------------------------------------------------------


def test_ensure_model_missing_custom() -> None:
    """Test that specifying a non-existent model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Specified model file does not exist"):
        ensure_hand_landmarker_model(model_path="non_existent_model.task")


def test_ensure_model_available() -> None:
    """Test that default model resolution returns a valid file."""
    model_path = ensure_hand_landmarker_model()
    assert model_path.is_file()
    assert model_path.stat().st_size > 1000000


# -----------------------------------------------------------------------------
# 5. HandLandmarkExtractor Pipeline Tests
# -----------------------------------------------------------------------------


def test_hand_landmark_extractor_blank_frame() -> None:
    """Test extraction on a blank RGB frame (no hands present)."""
    with HandLandmarkExtractor() as extractor:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = extractor.extract_frame(blank)

        assert isinstance(res, FrameLandmarks)
        assert res.left_detected is False
        assert res.right_detected is False
        assert res.left_confidence == 0.0
        assert res.right_confidence == 0.0
        assert res.left_hand.shape == (NUM_HAND_LANDMARKS, COORDINATE_DIM)
        assert res.right_hand.shape == (NUM_HAND_LANDMARKS, COORDINATE_DIM)
        assert np.all(res.left_hand == 0.0)
        assert np.all(res.right_hand == 0.0)


def test_hand_landmark_extractor_invalid_frame() -> None:
    """Verify invalid frame input raises ValueError."""
    with HandLandmarkExtractor() as extractor:
        with pytest.raises(ValueError, match="Cannot extract landmarks"):
            extractor.extract_frame(np.array([]))

        with pytest.raises(ValueError, match="Expected 3-channel RGB image"):
            extractor.extract_frame(np.zeros((100, 100), dtype=np.uint8))


def test_hand_landmark_extractor_sequence() -> None:
    """Test extraction over a temporal sequence of blank frames."""
    with HandLandmarkExtractor() as extractor:
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
        seq = extractor.extract_sequence(frames, impute_strategy="carry_forward")

        assert isinstance(seq, SequenceLandmarks)
        assert seq.num_frames == 5
        assert seq.left_hand.shape == (5, NUM_HAND_LANDMARKS, COORDINATE_DIM)
        assert seq.right_hand.shape == (5, NUM_HAND_LANDMARKS, COORDINATE_DIM)


def test_hand_landmark_extractor_empty_sequence() -> None:
    """Test empty frame sequence raises ValueError."""
    with HandLandmarkExtractor() as extractor:
        with pytest.raises(ValueError, match="Cannot extract landmarks from empty"):
            extractor.extract_sequence([])


def test_hand_landmark_extractor_video(synthetic_video: Path) -> None:
    """Test end-to-end video file extraction with SANKET-023 preprocessing integration."""
    with HandLandmarkExtractor() as extractor:
        seq = extractor.extract_video(
            video_path=synthetic_video,
            target_frames=15,
            target_size=(320, 240),
            letterbox=True,
            impute_strategy="carry_forward",
        )
        assert seq.num_frames == 15
        assert seq.left_hand.shape == (15, 21, 3)
        assert seq.right_hand.shape == (15, 21, 3)


# -----------------------------------------------------------------------------
# 6. CLI Script Execution Tests
# -----------------------------------------------------------------------------


def test_cli_extract_landmarks_single_video(synthetic_video: Path, tmp_path: Path) -> None:
    """Test CLI script extracting landmarks from a single video clip."""
    out_dir = tmp_path / "landmarks_out"
    cmd = [
        sys.executable,
        "scripts/extract_landmarks.py",
        "--video",
        str(synthetic_video),
        "--output-dir",
        str(out_dir),
        "--target-frames",
        "10",
        "--impute-strategy",
        "carry_forward",
        "--export-flat-npy",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"

    expected_npz = out_dir / f"{synthetic_video.stem}_landmarks.npz"
    expected_npy = out_dir / f"{synthetic_video.stem}_landmarks_flat.npy"
    assert expected_npz.is_file()
    assert expected_npy.is_file()

    # Verify reload
    seq, meta = load_landmarks_npz(expected_npz)
    assert seq.num_frames == 10
    assert meta["source_video"] == synthetic_video.name


def test_cli_extract_landmarks_batch_dir(synthetic_video: Path, tmp_path: Path) -> None:
    """Test CLI batch directory landmark extraction."""
    import shutil

    input_dir = tmp_path / "batch_vids"
    input_dir.mkdir()
    shutil.copy(synthetic_video, input_dir / "sign1.mp4")
    shutil.copy(synthetic_video, input_dir / "sign2.mp4")

    out_dir = tmp_path / "batch_landmarks"
    cmd = [
        sys.executable,
        "scripts/extract_landmarks.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(out_dir),
        "--target-frames",
        "5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert (out_dir / "sign1_landmarks.npz").is_file()
    assert (out_dir / "sign2_landmarks.npz").is_file()


def test_cli_extract_landmarks_missing_video(tmp_path: Path) -> None:
    """Test CLI script failure when video file is missing."""
    cmd = [
        sys.executable,
        "scripts/extract_landmarks.py",
        "--video",
        str(tmp_path / "missing.mp4"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
