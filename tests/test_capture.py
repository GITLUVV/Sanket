"""Comprehensive unit tests for camera capture and mock video streaming."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sanket.core.config import CameraConfig
from sanket.vision.capture import (
    BaseVideoCapture,
    CameraCapture,
    CameraNotFoundError,
    CapturedFrame,
    MockVideoCapture,
    find_available_cameras,
)


class TestCapturedFrame:
    """Tests for CapturedFrame container."""

    def test_captured_frame_creation(self) -> None:
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        captured = CapturedFrame(
            frame=frame_data,
            timestamp_ms=12345.6,
            frame_index=1,
            dropped_frames=0,
        )
        assert captured.frame.shape == (480, 640, 3)
        assert captured.timestamp_ms == 12345.6
        assert captured.frame_index == 1
        assert captured.dropped_frames == 0

    def test_captured_frame_immutability(self) -> None:
        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        captured = CapturedFrame(
            frame=frame_data,
            timestamp_ms=100.0,
            frame_index=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            captured.frame_index = 2  # type: ignore[misc]


class TestMockVideoCapture:
    """Tests for MockVideoCapture procedural and file-based modes."""

    def test_synthetic_moving_box_generation(self) -> None:
        with MockVideoCapture(
            width=320, height=240, fps=60, synthetic_pattern="moving_box"
        ) as cap:
            assert isinstance(cap, BaseVideoCapture)
            assert cap.is_running

            # Read several frames
            frames: list[CapturedFrame] = []
            for _ in range(5):
                frame = cap.read(timeout=1.0)
                assert frame is not None
                assert frame.frame.shape == (240, 320, 3)
                assert frame.frame.dtype == np.uint8
                frames.append(frame)

            # Assert sequential ordering and timestamps
            for i in range(1, len(frames)):
                assert frames[i].frame_index > frames[i - 1].frame_index
                assert frames[i].timestamp_ms >= frames[i - 1].timestamp_ms

        assert not cap.is_running

    @pytest.mark.parametrize("pattern", ["solid", "noise", "moving_box"])
    def test_synthetic_patterns(self, pattern: str) -> None:
        cap = MockVideoCapture(width=160, height=120, fps=100, synthetic_pattern=pattern)
        cap.start()
        try:
            frame = cap.read(timeout=1.0)
            assert frame is not None
            assert frame.frame.shape == (120, 160, 3)
        finally:
            cap.stop()

    def test_mock_video_read_non_blocking(self) -> None:
        cap = MockVideoCapture(width=160, height=120, fps=30)
        # Reading when not started returns None
        assert cap.read() is None

        cap.start()
        time.sleep(0.1)
        # Non-blocking read (timeout=None)
        frame = cap.read(timeout=None)
        cap.stop()
        assert frame is None or isinstance(frame, CapturedFrame)

    def test_mock_video_file_not_found(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "non_existent_video.mp4"
        cap = MockVideoCapture(video_path=missing_file)
        with pytest.raises(FileNotFoundError):
            cap.start()

    def test_mock_video_file_playback(self, tmp_path: Path) -> None:
        # Generate a small 10-frame synthetic test video file using OpenCV
        video_file = tmp_path / "test_clip.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_file), fourcc, 30.0, (160, 120))
        for i in range(10):
            canvas = np.full((120, 160, 3), fill_value=i * 20, dtype=np.uint8)
            writer.write(canvas)
        writer.release()

        # Playback test
        with MockVideoCapture(
            video_path=video_file, width=160, height=120, fps=60, loop=True
        ) as cap:
            assert cap.is_running
            frames_read = 0
            for _ in range(15):  # Read 15 frames to test loop
                f = cap.read(timeout=1.0)
                if f is not None:
                    frames_read += 1
            assert frames_read >= 5


class TestCameraCapture:
    """Tests for CameraCapture with mocked OpenCV hardware bindings."""

    def test_instantiation_from_config(self) -> None:
        config = CameraConfig(device_id=2, width=1280, height=720, fps=60)
        cap = CameraCapture.from_config(config)
        assert cap.device_id == 2
        assert cap.width == 1280
        assert cap.height == 720
        assert cap.fps == 60

    @patch("cv2.VideoCapture")
    def test_camera_not_found_raises_error(self, mock_cv_capture: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cv_capture.return_value = mock_instance

        cap = CameraCapture(device_id=99)
        with pytest.raises(CameraNotFoundError, match="Camera device index 99 could not be opened"):
            cap.start()

    @patch("cv2.VideoCapture")
    def test_camera_capture_successful_stream(self, mock_cv_capture: MagicMock) -> None:
        fake_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
        }.get(prop, 0)
        mock_instance.read.return_value = (True, fake_frame)
        mock_cv_capture.return_value = mock_instance

        with CameraCapture(device_id=0, width=640, height=480, fps=30) as cap:
            assert cap.is_running
            assert cap.actual_dimensions == (640, 480)

            # Wait for at least one frame
            frame = cap.read(timeout=1.0)
            assert frame is not None
            assert frame.frame.shape == (480, 640, 3)
            assert frame.frame_index >= 1
            assert cap.total_frames_captured >= 1

        assert not cap.is_running
        mock_instance.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_camera_capture_queue_backpressure_and_drops(self, mock_cv_capture: MagicMock) -> None:
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.return_value = 640
        mock_instance.read.return_value = (True, fake_frame)
        mock_cv_capture.return_value = mock_instance

        # Buffer size of 1 with fast producer
        cap = CameraCapture(device_id=0, max_buffer_size=1, fps=120)
        cap.start()
        try:
            # Let worker produce multiple frames without reading
            time.sleep(0.1)
            assert cap.total_frames_captured > 1
            # Dropped frames should be recorded when queue overflows
            assert cap.total_frames_dropped >= 1

            frame = cap.read(timeout=0.5)
            assert frame is not None
        finally:
            cap.stop()

    @patch("cv2.VideoCapture")
    def test_camera_read_error_recovery(self, mock_cv_capture: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        # First 2 reads fail, 3rd read succeeds
        fake_frame = np.ones((480, 640, 3), dtype=np.uint8)
        mock_instance.read.side_effect = [
            (False, None),
            (False, None),
            (True, fake_frame),
            (True, fake_frame),
        ]
        mock_cv_capture.return_value = mock_instance

        cap = CameraCapture(device_id=0, fps=60)
        cap.start()
        try:
            frame = cap.read(timeout=1.0)
            assert frame is not None
            assert frame.frame.shape == (480, 640, 3)
        finally:
            cap.stop()


class TestFindAvailableCameras:
    """Tests for camera discovery utility."""

    @patch("cv2.VideoCapture")
    def test_find_available_cameras(self, mock_cv_capture: MagicMock) -> None:
        def side_effect(dev_id: int) -> MagicMock:
            instance = MagicMock()
            # Devices 0 and 2 are open, others closed
            instance.isOpened.return_value = dev_id in (0, 2)
            return instance

        mock_cv_capture.side_effect = side_effect
        found = find_available_cameras(max_devices=4)
        assert found == [0, 2]
