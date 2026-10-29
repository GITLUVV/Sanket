"""Unit test suite for video and image preprocessing pipelines (SANKET-023)."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from sanket.vision.preprocessing import (
    VideoMetadata,
    VideoPreprocessor,
    compute_temporal_indices,
    get_video_metadata,
    letterbox_resize,
    read_video_frames,
    sample_video_frames,
    unletterbox_coordinates,
)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a temporary valid MP4 video with 20 distinct synthetic frames."""
    video_path = tmp_path / "test_clip.mp4"
    width, height = 160, 120
    fps = 10.0
    num_frames = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert out.isOpened(), "Failed to open VideoWriter for synthetic test video"

    for i in range(num_frames):
        # Draw a synthetic colored frame with distinct intensities per frame
        frame = np.full((height, width, 3), (i * 12, (num_frames - i) * 12, 100), dtype=np.uint8)
        cv2.putText(frame, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    return video_path


# -----------------------------------------------------------------------------
# 1. Temporal Index Computation Tests
# -----------------------------------------------------------------------------


def test_temporal_indices_identity() -> None:
    """Test identity temporal sampling when total_frames == target_frames."""
    indices = compute_temporal_indices(total_frames=30, target_frames=30)
    assert len(indices) == 30
    assert indices == list(range(30))


def test_temporal_indices_downsample() -> None:
    """Test temporal downsampling from 75 frames to 30 frames."""
    indices = compute_temporal_indices(total_frames=75, target_frames=30)
    assert len(indices) == 30
    assert indices[0] == 0
    assert indices[-1] == 74
    # Verify non-decreasing monotonicity
    assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
    # Verify all indices are strictly within bounds
    assert all(0 <= idx < 75 for idx in indices)


def test_temporal_indices_upsample() -> None:
    """Test temporal upsampling from 10 frames to 30 frames with frame replication."""
    indices = compute_temporal_indices(total_frames=10, target_frames=30)
    assert len(indices) == 30
    assert indices[0] == 0
    assert indices[-1] == 9
    assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
    assert all(0 <= idx < 10 for idx in indices)
    # Check that duplication occurs
    assert len(set(indices)) == 10


def test_temporal_indices_single_frame() -> None:
    """Test temporal sampling on a single-frame video."""
    indices = compute_temporal_indices(total_frames=1, target_frames=30)
    assert len(indices) == 30
    assert indices == [0] * 30


@pytest.mark.parametrize(
    ("total", "target"),
    [
        (0, 30),
        (-5, 30),
        (30, 0),
        (30, -1),
    ],
)
def test_temporal_indices_invalid(total: int, target: int) -> None:
    """Verify that non-positive values raise ValueError."""
    with pytest.raises(ValueError, match="must be >= 1"):
        compute_temporal_indices(total_frames=total, target_frames=target)


# -----------------------------------------------------------------------------
# 2. Letterbox Resizing and Coordinate Mapping Tests
# -----------------------------------------------------------------------------


def test_letterbox_landscape_padding() -> None:
    """Test letterboxing a 16:9 image into a 4:3 target window."""
    # 1920x1080 (16:9) resized to 640x480 (4:3)
    orig_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    orig_img[100:200, 100:200] = 255

    padded, meta = letterbox_resize(orig_img, target_size=(640, 480))
    assert padded.shape == (480, 640, 3)
    # Width fits exactly (640 / 1920 = 1/3 scale -> height becomes 360)
    # Remaining height padding is 480 - 360 = 120 (60 top, 60 bottom)
    assert meta["pad_left"] == 0
    assert meta["pad_right"] == 0
    assert meta["pad_top"] == 60
    assert meta["pad_bottom"] == 60
    assert pytest.approx(meta["scale"], 0.001) == (640 / 1920)


def test_letterbox_portrait_padding() -> None:
    """Test letterboxing a 9:16 portrait image into a 4:3 target window."""
    orig_img = np.zeros((1920, 1080, 3), dtype=np.uint8)
    padded, meta = letterbox_resize(orig_img, target_size=(640, 480))
    assert padded.shape == (480, 640, 3)
    # Height fits exactly (480 / 1920 = 0.25 scale -> width becomes 270)
    # Remaining width padding is 640 - 270 = 370 (185 left, 185 right)
    assert meta["pad_top"] == 0
    assert meta["pad_bottom"] == 0
    assert meta["pad_left"] == 185
    assert meta["pad_right"] == 185
    assert pytest.approx(meta["scale"], 0.001) == 0.25


def test_letterbox_custom_pad_color() -> None:
    """Verify custom border pad color application."""
    orig_img = np.zeros((100, 200, 3), dtype=np.uint8)
    pad_color = (42, 84, 126)
    padded, _ = letterbox_resize(orig_img, target_size=(100, 100), pad_color=pad_color)
    assert padded.shape == (100, 100, 3)
    # Corner pixel in padding border should equal pad_color
    assert tuple(padded[0, 0]) == pad_color


def test_unletterbox_coordinates_inversion() -> None:
    """Verify that unletterbox_coordinates accurately recovers original pixel locations."""
    orig_w, orig_h = 1280, 720
    test_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    _, meta = letterbox_resize(test_img, target_size=(640, 480))

    # Test coordinate at original center (640, 360)
    orig_x, orig_y = 640.0, 360.0
    letterbox_x = orig_x * meta["scale"] + meta["pad_left"]
    letterbox_y = orig_y * meta["scale"] + meta["pad_top"]

    recovered_x, recovered_y = unletterbox_coordinates(letterbox_x, letterbox_y, meta)
    assert pytest.approx(recovered_x, abs=1.0) == orig_x
    assert pytest.approx(recovered_y, abs=1.0) == orig_y


def test_letterbox_invalid_inputs() -> None:
    """Verify invalid image inputs raise ValueError."""
    with pytest.raises(ValueError, match="Cannot resize empty"):
        letterbox_resize(np.array([]))

    with pytest.raises(ValueError, match="Target size must be positive"):
        valid_img = np.zeros((50, 50, 3), dtype=np.uint8)
        letterbox_resize(valid_img, target_size=(0, 50))


# -----------------------------------------------------------------------------
# 3. Video Metadata and Frame Reading Tests
# -----------------------------------------------------------------------------


def test_get_video_metadata(synthetic_video: Path) -> None:
    """Test metadata extraction from a valid video."""
    meta = get_video_metadata(synthetic_video)
    assert isinstance(meta, VideoMetadata)
    assert meta.frame_count == 20
    assert meta.width == 160
    assert meta.height == 120
    assert pytest.approx(meta.fps, 0.1) == 10.0
    assert pytest.approx(meta.duration_seconds, 0.1) == 2.0


def test_get_video_metadata_missing_file(tmp_path: Path) -> None:
    """Test that missing video raises FileNotFoundError."""
    missing = tmp_path / "non_existent.mp4"
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        get_video_metadata(missing)


def test_get_video_metadata_corrupt_file(tmp_path: Path) -> None:
    """Test that zero-byte corrupt video raises ValueError."""
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_text("not a video")
    with pytest.raises(ValueError, match="no readable frames|Unable to open"):
        get_video_metadata(corrupt)


def test_read_video_frames_all(synthetic_video: Path) -> None:
    """Test sequential reading of all video frames."""
    frames = read_video_frames(synthetic_video, target_indices=None, to_rgb=True)
    assert len(frames) == 20
    assert frames[0].shape == (120, 160, 3)
    assert frames[0].dtype == np.uint8


def test_read_video_frames_subsampled(synthetic_video: Path) -> None:
    """Test targeted frame indexing."""
    target_indices = [0, 5, 10, 10, 19]
    frames = read_video_frames(synthetic_video, target_indices=target_indices, to_rgb=True)
    assert len(frames) == 5
    # Verify index 2 and index 3 (both requested frame 10) have identical pixels
    assert np.array_equal(frames[2], frames[3])


def test_sample_video_frames_letterboxed(synthetic_video: Path) -> None:
    """Test end-to-end sample_video_frames with letterboxing to (640, 480)."""
    tensor = sample_video_frames(
        video_path=synthetic_video,
        target_frames=30,
        target_size=(640, 480),
        letterbox=True,
    )
    assert tensor.shape == (30, 480, 640, 3)
    assert tensor.dtype == np.uint8


def test_sample_video_frames_direct_resize(synthetic_video: Path) -> None:
    """Test sample_video_frames with non-letterbox direct resize."""
    tensor = sample_video_frames(
        video_path=synthetic_video,
        target_frames=15,
        target_size=(320, 240),
        letterbox=False,
    )
    assert tensor.shape == (15, 240, 320, 3)


# -----------------------------------------------------------------------------
# 4. VideoPreprocessor Pipeline Class Tests
# -----------------------------------------------------------------------------


def test_video_preprocessor_uint8(synthetic_video: Path) -> None:
    """Test VideoPreprocessor with default uint8 output."""
    preprocessor = VideoPreprocessor(target_frames=30, target_size=(640, 480))
    tensor = preprocessor.preprocess(synthetic_video)
    assert tensor.shape == (30, 480, 640, 3)
    assert tensor.dtype == np.uint8


def test_video_preprocessor_normalized_float(synthetic_video: Path) -> None:
    """Test VideoPreprocessor with normalized float32 output."""
    preprocessor = VideoPreprocessor(
        target_frames=30,
        target_size=(640, 480),
        normalize_pixels=True,
    )
    tensor = preprocessor.preprocess(synthetic_video)
    assert tensor.shape == (30, 480, 640, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= 1.0
    assert 0.0 <= tensor.max() <= 1.0


def test_video_preprocessor_clip_memory() -> None:
    """Test VideoPreprocessor.preprocess_clip on in-memory list of frames."""
    preprocessor = VideoPreprocessor(target_frames=10, target_size=(640, 480))
    raw_frames = [np.full((100, 100, 3), i * 20, dtype=np.uint8) for i in range(5)]
    tensor = preprocessor.preprocess_clip(raw_frames)
    assert tensor.shape == (10, 480, 640, 3)


def test_video_preprocessor_clip_empty() -> None:
    """Test VideoPreprocessor.preprocess_clip raises ValueError on empty list."""
    preprocessor = VideoPreprocessor()
    with pytest.raises(ValueError, match="Cannot preprocess empty frames list"):
        preprocessor.preprocess_clip([])


def test_video_preprocessor_save_npy(synthetic_video: Path, tmp_path: Path) -> None:
    """Test preprocessor serialization to .npy file."""
    preprocessor = VideoPreprocessor(target_frames=15, target_size=(320, 240))
    output_npy = tmp_path / "processed" / "clip.npy"
    saved_path = preprocessor.save_preprocessed(synthetic_video, output_npy)

    assert saved_path.is_file()
    loaded = np.load(str(saved_path))
    assert loaded.shape == (15, 240, 320, 3)


# -----------------------------------------------------------------------------
# 5. CLI Script Execution Tests
# -----------------------------------------------------------------------------


def test_cli_single_video(synthetic_video: Path, tmp_path: Path) -> None:
    """Test CLI script processing a single video with preview frames."""
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "scripts/preprocess_videos.py",
        "--video",
        str(synthetic_video),
        "--output-dir",
        str(out_dir),
        "--target-frames",
        "15",
        "--width",
        "320",
        "--height",
        "240",
        "--export-preview",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"

    expected_npy = out_dir / f"{synthetic_video.stem}_preprocessed.npy"
    assert expected_npy.is_file()
    assert (out_dir / f"{synthetic_video.stem}_frame00.png").is_file()
    assert (out_dir / f"{synthetic_video.stem}_frame07.png").is_file()
    assert (out_dir / f"{synthetic_video.stem}_frame14.png").is_file()


def test_cli_input_dir(synthetic_video: Path, tmp_path: Path) -> None:
    """Test CLI script batch-processing a directory."""
    input_dir = tmp_path / "batch_input"
    input_dir.mkdir()
    # Copy synthetic video into input_dir
    import shutil

    shutil.copy(synthetic_video, input_dir / "clip1.mp4")
    shutil.copy(synthetic_video, input_dir / "clip2.mp4")

    out_dir = tmp_path / "batch_output"
    cmd = [
        sys.executable,
        "scripts/preprocess_videos.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(out_dir),
        "--target-frames",
        "10",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert (out_dir / "clip1_preprocessed.npy").is_file()
    assert (out_dir / "clip2_preprocessed.npy").is_file()


def test_cli_missing_video(tmp_path: Path) -> None:
    """Test CLI failure on missing video."""
    cmd = [
        sys.executable,
        "scripts/preprocess_videos.py",
        "--video",
        str(tmp_path / "missing.mp4"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
