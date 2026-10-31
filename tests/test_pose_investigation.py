"""Unit test suite for upper-body pose reference anchors and relative features (SANKET-025)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from sanket.vision.landmarks import FrameLandmarks
from sanket.vision.pose import (
    ANCHOR_NAMES,
    NUM_POSE_ANCHORS,
    HandBodyRelativeFeatures,
    PoseAnchors,
    PoseExtractor,
    compute_relative_spatial_features,
    compute_sternum,
    ensure_pose_landmarker_model,
)
from scripts.investigate_pose import simulate_minimal_pair_analysis


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a temporary valid MP4 video clip with 5 synthetic frames."""
    video_path = tmp_path / "test_pose_clip.mp4"
    width, height = 160, 120
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(5):
        frame = np.full((height, width, 3), (i * 30, 80, 120), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


# -----------------------------------------------------------------------------
# 1. Sternum and PoseAnchors Geometry Tests
# -----------------------------------------------------------------------------


def test_compute_sternum_midpoint() -> None:
    """Verify sternum midpoint arithmetic calculation."""
    left_sh = np.array([0.60, 0.40, 0.10], dtype=np.float32)
    right_sh = np.array([0.40, 0.40, 0.10], dtype=np.float32)
    sternum = compute_sternum(left_sh, right_sh)

    expected = np.array([0.50, 0.40, 0.10], dtype=np.float32)
    assert np.allclose(sternum, expected)


def test_pose_anchors_matrix() -> None:
    """Verify conversion of PoseAnchors to (5, 3) reference matrix."""
    anchors = PoseAnchors(
        nose=np.array([0.5, 0.2, 0.0], dtype=np.float32),
        mouth_center=np.array([0.5, 0.3, 0.0], dtype=np.float32),
        left_shoulder=np.array([0.7, 0.5, 0.0], dtype=np.float32),
        right_shoulder=np.array([0.3, 0.5, 0.0], dtype=np.float32),
        sternum=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        shoulder_width=0.4,
        is_detected=True,
        confidence=0.99,
    )
    mat = anchors.to_anchor_matrix()
    assert mat.shape == (NUM_POSE_ANCHORS, 3)
    assert np.allclose(mat[0], anchors.nose)
    assert np.allclose(mat[2], anchors.sternum)
    assert len(ANCHOR_NAMES) == NUM_POSE_ANCHORS


# -----------------------------------------------------------------------------
# 2. Hand-to-Body Relative Spatial Features Tests
# -----------------------------------------------------------------------------


def test_relative_spatial_features_detected() -> None:
    """Verify relative displacement and distance vectors for detected hands and pose."""
    pose = PoseAnchors(
        nose=np.array([0.5, 0.2, 0.0], dtype=np.float32),
        mouth_center=np.array([0.5, 0.3, 0.0], dtype=np.float32),
        left_shoulder=np.array([0.6, 0.5, 0.0], dtype=np.float32),
        right_shoulder=np.array([0.4, 0.5, 0.0], dtype=np.float32),
        sternum=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        shoulder_width=0.20,
        is_detected=True,
        confidence=1.0,
    )

    # Right hand with wrist positioned directly at mouth (0.5, 0.3, 0.0)
    r_hand = np.zeros((21, 3), dtype=np.float32)
    r_hand[0] = np.array([0.5, 0.3, 0.0], dtype=np.float32)  # Wrist at mouth
    r_hand[8] = np.array([0.5, 0.2, 0.0], dtype=np.float32)  # Index tip at nose

    hand_lm = FrameLandmarks(
        left_hand=np.zeros((21, 3), dtype=np.float32),
        right_hand=r_hand,
        left_detected=False,
        right_detected=True,
        left_confidence=0.0,
        right_confidence=0.9,
    )

    feat = compute_relative_spatial_features(hand_lm, pose, normalize_by_shoulder_width=True)
    assert isinstance(feat, HandBodyRelativeFeatures)
    assert feat.normalized_by_shoulder_width is True

    # Right wrist to mouth center (anchor 1) distance should be 0.0
    assert pytest.approx(feat.right_wrist_distances[1], abs=1e-5) == 0.0
    # Right index tip to nose (anchor 0) distance should be 0.0
    assert pytest.approx(feat.right_index_distances[0], abs=1e-5) == 0.0

    # Right wrist to sternum (anchor 2): unnormalized distance is 0.2. Normalized by width 0.2 = 1.0
    assert pytest.approx(feat.right_wrist_distances[2], abs=1e-5) == 1.0

    # Undetected left hand should be all zeros
    assert np.all(feat.left_wrist_distances == 0.0)
    assert np.all(feat.left_wrist_displacements == 0.0)

    # Vector flattening tests
    vec_full = feat.to_vector(include_displacements=True)
    assert vec_full.shape == (80,)
    assert vec_full.dtype == np.float32

    vec_dist = feat.to_vector(include_displacements=False)
    assert vec_dist.shape == (20,)


def test_relative_spatial_features_undetected() -> None:
    """Verify that undetected pose yields all-zero relative features."""
    pose = PoseAnchors(
        nose=np.zeros(3, dtype=np.float32),
        mouth_center=np.zeros(3, dtype=np.float32),
        left_shoulder=np.zeros(3, dtype=np.float32),
        right_shoulder=np.zeros(3, dtype=np.float32),
        sternum=np.zeros(3, dtype=np.float32),
        shoulder_width=0.0,
        is_detected=False,
        confidence=0.0,
    )
    hand_lm = FrameLandmarks(
        left_hand=np.ones((21, 3), dtype=np.float32),
        right_hand=np.ones((21, 3), dtype=np.float32),
        left_detected=True,
        right_detected=True,
        left_confidence=0.9,
        right_confidence=0.9,
    )
    feat = compute_relative_spatial_features(hand_lm, pose)
    assert np.all(feat.to_vector() == 0.0)


# -----------------------------------------------------------------------------
# 3. Model Caching and Resolution Tests
# -----------------------------------------------------------------------------


def test_ensure_pose_model_missing_custom() -> None:
    """Verify specifying non-existent model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Specified model file does not exist"):
        ensure_pose_landmarker_model(model_path="invalid_pose_model.task")


def test_ensure_pose_model_available() -> None:
    """Verify default resolution acquires valid pose model file."""
    p = ensure_pose_landmarker_model()
    assert p.is_file()
    assert p.stat().st_size > 1000000


# -----------------------------------------------------------------------------
# 4. PoseExtractor Pipeline Tests
# -----------------------------------------------------------------------------


def test_pose_extractor_blank_frame() -> None:
    """Verify PoseExtractor execution on blank image returns un-detected anchors."""
    with PoseExtractor() as extractor:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        anchors = extractor.extract_frame(blank)
        assert isinstance(anchors, PoseAnchors)
        assert anchors.is_detected is False
        assert anchors.confidence == 0.0
        assert anchors.nose.shape == (3,)
        assert anchors.sternum.shape == (3,)


def test_pose_extractor_mock_backend() -> None:
    """Verify PoseExtractor with explicit mock backend returns un-detected anchors."""
    with PoseExtractor(backend="mock") as extractor:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        anchors = extractor.extract_frame(blank)
        assert isinstance(anchors, PoseAnchors)
        assert anchors.is_detected is False
        assert anchors.confidence == 0.0
        assert extractor.backend == "mock"


def test_pose_extractor_mock_detection() -> None:
    """Verify PoseExtractor parsing when pose landmarks are returned."""
    from unittest.mock import MagicMock

    class MockLandmark:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x = x
            self.y = y
            self.z = z

    mock_landmarks = [MockLandmark(i * 0.01, i * 0.02, 0.0) for i in range(33)]
    mock_result = MagicMock()
    mock_result.pose_landmarks = [mock_landmarks]

    mock_detector = MagicMock()
    mock_detector.detect.return_value = mock_result

    with PoseExtractor(detector=mock_detector) as extractor:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        anchors = extractor.extract_frame(blank)
        assert anchors.is_detected is True
        assert anchors.confidence == 1.0
        assert pytest.approx(anchors.nose[0]) == 0.0
        assert anchors.shoulder_width > 0.0


def test_pose_extractor_invalid_inputs() -> None:
    """Verify error handling on invalid image frames."""
    with PoseExtractor() as extractor:
        with pytest.raises(ValueError, match="Cannot extract pose"):
            extractor.extract_frame(np.array([]))

        with pytest.raises(ValueError, match="Expected 3-channel RGB image"):
            extractor.extract_frame(np.zeros((100, 100), dtype=np.uint8))


# -----------------------------------------------------------------------------
# 5. Minimal Pair Simulation & CLI Execution Tests
# -----------------------------------------------------------------------------


def test_simulate_minimal_pair_analysis() -> None:
    """Verify empirical simulation quantifies substantial separation for confusable signs."""
    res = simulate_minimal_pair_analysis()
    assert "signs" in res
    assert "separations" in res
    assert "WATER" in res["signs"]
    assert "PLEASE" in res["signs"]

    # Assert significant distance margin (> 0.5 shoulder width) between chest and mouth signs
    margin_water_please = res["separations"]["WATER_vs_PLEASE_distance_margin"]
    assert margin_water_please > 0.5

    margin_food_hello = res["separations"]["FOOD_vs_HELLO_distance_margin"]
    assert margin_food_hello > 0.5


def test_cli_simulate(tmp_path: Path) -> None:
    """Test CLI script with --simulate and --output-json."""
    out_json = tmp_path / "sim_results.json"
    cmd = [
        sys.executable,
        "scripts/investigate_pose.py",
        "--simulate",
        "--output-json",
        str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert out_json.is_file()

    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)
    assert "WATER_vs_PLEASE_distance_margin" in data["separations"]


def test_cli_video(synthetic_video: Path, tmp_path: Path) -> None:
    """Test CLI script evaluating a video clip."""
    out_json = tmp_path / "video_results.json"
    cmd = [
        sys.executable,
        "scripts/investigate_pose.py",
        "--video",
        str(synthetic_video),
        "--output-json",
        str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert out_json.is_file()


def test_cli_missing_video(tmp_path: Path) -> None:
    """Test CLI script handling missing video."""
    cmd = [
        sys.executable,
        "scripts/investigate_pose.py",
        "--video",
        str(tmp_path / "non_existent.mp4"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
