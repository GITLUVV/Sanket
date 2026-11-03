"""Resilient, thread-safe camera and video input capture pipeline.

Provides asynchronous camera frame acquisition with drop-frame handling,
bounded in-memory queues (zero biometric disk persistence), and mock video
capture for deterministic headless CI execution.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from sanket.core.config import CameraConfig

logger = logging.getLogger(__name__)


class CaptureError(Exception):
    """Base exception for camera and video capture failures."""


class CameraNotFoundError(CaptureError):
    """Raised when the requested camera device index is inaccessible or invalid."""


@dataclass(frozen=True)
class CapturedFrame:
    """Immutable container representing a captured video frame.

    Attributes:
        frame: BGR image array of shape (height, width, 3) and dtype uint8.
        timestamp_ms: Monotonic timestamp in milliseconds when the frame was acquired.
        frame_index: Sequential zero-based frame counter.
        dropped_frames: Number of frames dropped prior to this frame due to queue backpressure.
    """

    frame: np.ndarray
    timestamp_ms: float
    frame_index: int
    dropped_frames: int = 0


class BaseVideoCapture(ABC):
    """Abstract base class for thread-safe video frame providers."""

    @abstractmethod
    def start(self) -> None:
        """Start the video acquisition stream."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the video acquisition stream and release resources."""

    @abstractmethod
    def read(self, timeout: float | None = 1.0) -> CapturedFrame | None:
        """Read the latest captured frame.

        Args:
            timeout: Maximum seconds to wait for a frame. None for non-blocking.

        Returns:
            CapturedFrame if a frame is available, otherwise None.
        """

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the capture stream is actively running."""

    def __enter__(self) -> BaseVideoCapture:
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point releasing hardware resources."""
        self.stop()


class CameraCapture(BaseVideoCapture):
    """Threaded camera capture worker wrapping OpenCV VideoCapture.

    Acquires video frames asynchronously in a dedicated worker thread to isolate
    camera I/O latency from the primary recognition pipeline. Implements bounded
    queue backpressure with drop-frame tracking to maintain minimum latency.

    Adheres to SANKET's Zero-Biometric-Retention mandate: frames remain strictly
    in volatile system memory and are never persisted to disk.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_buffer_size: int = 2,
    ) -> None:
        """Initialize CameraCapture with device parameters.

        Args:
            device_id: Operating system device index for webcam.
            width: Requested capture frame width.
            height: Requested capture frame height.
            fps: Requested capture frame rate.
            max_buffer_size: Maximum frames retained in volatile memory queue.
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.max_buffer_size = max(1, max_buffer_size)

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=self.max_buffer_size)

        self._frame_counter = 0
        self._dropped_counter = 0
        self._actual_width = width
        self._actual_height = height
        self._actual_fps = fps

    @classmethod
    def from_config(cls, config: CameraConfig) -> CameraCapture:
        """Instantiate CameraCapture from a validated CameraConfig model."""
        return cls(
            device_id=config.device_id,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )

    def start(self) -> None:
        """Open the camera device and launch the background acquisition thread."""
        if self.is_running:
            return

        logger.info(
            "Initializing CameraCapture on device %d (requested: %dx%d @ %d FPS)",
            self.device_id,
            self.width,
            self.height,
            self.fps,
        )

        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            raise CameraNotFoundError(
                f"Camera device index {self.device_id} could not be opened. "
                "Ensure device is connected, not locked, and permissions allow access."
            )

        # Negotiate resolution and frame rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        self._actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        self._cap = cap

        self._stop_event.clear()
        self._frame_counter = 0
        self._dropped_counter = 0

        self._thread = threading.Thread(
            target=self._capture_worker,
            name=f"CameraCaptureWorker-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "CameraCapture started successfully (negotiated: %dx%d @ %d FPS)",
            self._actual_width,
            self._actual_height,
            self._actual_fps,
        )

    def _capture_worker(self) -> None:
        """Background thread worker continuously pulling frames from camera hardware."""
        interval_sec = 1.0 / max(1, self.fps)

        while not self._stop_event.is_set():
            start_time = time.perf_counter()

            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(
                    "Camera read error or frame acquisition drop on device %d",
                    self.device_id,
                )
                time.sleep(0.01)
                continue

            timestamp_ms = time.perf_counter() * 1000.0
            self._frame_counter += 1

            # Handle bounded queue backpressure: discard oldest frame if buffer is full
            dropped = 0
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._dropped_counter += 1
                    dropped = 1
                except queue.Empty:
                    pass

            captured = CapturedFrame(
                frame=frame,
                timestamp_ms=timestamp_ms,
                frame_index=self._frame_counter,
                dropped_frames=dropped,
            )

            try:
                self._queue.put_nowait(captured)
            except queue.Full:
                self._dropped_counter += 1

            # Throttle loop to target FPS if camera hardware delivers faster than configured
            elapsed = time.perf_counter() - start_time
            sleep_time = interval_sec - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def read(self, timeout: float | None = 1.0) -> CapturedFrame | None:
        """Read the latest captured frame from volatile memory buffer.

        Args:
            timeout: Seconds to wait for frame. If None, performs non-blocking poll.

        Returns:
            Latest CapturedFrame or None if queue is empty.
        """
        if not self.is_running:
            return None

        try:
            if timeout is None:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal background thread to stop, wait for exit, and release camera device."""
        if not self.is_running and self._cap is None:
            return

        logger.info("Stopping CameraCapture on device %d...", self.device_id)
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Flush in-memory queue to guarantee zero biometric retention in memory
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        logger.info("CameraCapture stopped successfully.")

    @property
    def is_running(self) -> bool:
        """True if worker thread is active and capture device is open."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def actual_dimensions(self) -> tuple[int, int]:
        """Negotiated (width, height) of the video capture device."""
        return self._actual_width, self._actual_height

    @property
    def total_frames_captured(self) -> int:
        """Total number of frames acquired by the background thread."""
        return self._frame_counter

    @property
    def total_frames_dropped(self) -> int:
        """Total number of frames dropped due to consumer backpressure."""
        return self._dropped_counter


class MockVideoCapture(BaseVideoCapture):
    """Deterministic synthetic or file-based video capture for testing and CI.

    Supports generating procedural synthetic frames (solid color, gradient,
    moving geometric patterns) or reading pre-recorded video files.
    """

    def __init__(
        self,
        video_path: Path | str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        synthetic_pattern: str = "moving_box",
        loop: bool = True,
        max_buffer_size: int = 5,
    ) -> None:
        """Initialize MockVideoCapture.

        Args:
            video_path: Optional path to recorded video file. If None, operates procedurally.
            width: Frame width.
            height: Frame height.
            fps: Frame generation rate.
            synthetic_pattern: Procedural pattern: 'solid', 'moving_box', or 'noise'.
            loop: Whether to loop video file when reaching end.
            max_buffer_size: Maximum queue capacity.
        """
        self.video_path = Path(video_path) if video_path else None
        self.width = width
        self.height = height
        self.fps = fps
        self.synthetic_pattern = synthetic_pattern
        self.loop = loop
        self.max_buffer_size = max_buffer_size

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=self.max_buffer_size)
        self._frame_counter = 0
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        """Start the background generation or file playback thread."""
        if self.is_running:
            return

        if self.video_path is not None:
            if not self.video_path.exists():
                raise FileNotFoundError(f"Mock video file not found: {self.video_path}")
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise CaptureError(f"Failed to open mock video file: {self.video_path}")

        self._stop_event.clear()
        self._frame_counter = 0

        self._thread = threading.Thread(
            target=self._worker,
            name="MockVideoCaptureWorker",
            daemon=True,
        )
        self._thread.start()

    def _generate_synthetic_frame(self, index: int) -> np.ndarray:
        """Generate a procedural synthetic frame without disk I/O."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.synthetic_pattern == "solid":
            frame[:] = (40, 40, 40)
        elif self.synthetic_pattern == "noise":
            rng = np.random.default_rng(seed=42 + index)
            frame = rng.integers(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        else:  # 'moving_box'
            frame[:] = (30, 30, 30)
            # Procedural moving box simulating hand gesture motion
            box_size = 80
            pos_x = int((np.sin(index * 0.1) * 0.4 + 0.5) * (self.width - box_size))
            pos_y = int((np.cos(index * 0.1) * 0.4 + 0.5) * (self.height - box_size))
            cv2.rectangle(
                frame,
                (pos_x, pos_y),
                (pos_x + box_size, pos_y + box_size),
                (0, 220, 255),
                -1,
            )

        return frame

    def _worker(self) -> None:
        """Worker thread producing synthetic frames or reading video frames at controlled FPS."""
        interval = 1.0 / max(1, self.fps)

        while not self._stop_event.is_set():
            start_t = time.perf_counter()

            if self.video_path is not None and self._cap is not None:
                ret, frame = self._cap.read()
                if not ret:
                    if self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self._cap.read()
                        if not ret:
                            break
                    else:
                        break
                frame = cv2.resize(frame, (self.width, self.height))
            else:
                frame = self._generate_synthetic_frame(self._frame_counter)

            self._frame_counter += 1
            timestamp_ms = time.perf_counter() * 1000.0

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            captured = CapturedFrame(
                frame=frame,
                timestamp_ms=timestamp_ms,
                frame_index=self._frame_counter,
                dropped_frames=0,
            )

            try:
                self._queue.put_nowait(captured)
            except queue.Full:
                pass

            elapsed = time.perf_counter() - start_t
            sleep_time = interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def read(self, timeout: float | None = 1.0) -> CapturedFrame | None:
        """Read the next mock frame."""
        if not self.is_running:
            return None

        try:
            if timeout is None:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop worker and release mock resources."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        """True if mock worker thread is active."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )


def find_available_cameras(max_devices: int = 5) -> list[int]:
    """Discover accessible camera devices connected to the host system.

    Probes device indices from 0 up to max_devices - 1.

    Args:
        max_devices: Maximum number of device indices to test.

    Returns:
        List of integer device indices that opened successfully.
    """
    available: list[int] = []
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            available.append(device_id)
            cap.release()
    return available
