"""Unit tests for RealtimePreprocessor and FrameBuffer (SANKET-041)."""

import threading
import time

import numpy as np
import pytest

from sanket.vision.preprocessing import FrameBuffer, RealtimePreprocessor


class TestRealtimePreprocessor:
    """Test suite for RealtimePreprocessor."""

    def test_default_initialization(self) -> None:
        """Verify default configuration targets 640x480 with mirror and RGB enabled."""
        preprocessor = RealtimePreprocessor()
        assert preprocessor.target_size == (640, 480)
        assert preprocessor.mirror is True
        assert preprocessor.to_rgb is True
        assert preprocessor.pad_color == (0, 0, 0)

    def test_invalid_target_size(self) -> None:
        """Verify non-positive target dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            RealtimePreprocessor(target_size=(0, 480))
        with pytest.raises(ValueError, match="dimensions must be positive"):
            RealtimePreprocessor(target_size=(640, -10))

    def test_empty_or_none_frame_raises(self) -> None:
        """Verify empty or None frames raise ValueError."""
        preprocessor = RealtimePreprocessor()
        with pytest.raises(ValueError, match="Cannot process empty"):
            preprocessor.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="Cannot process empty"):
            preprocessor.process_frame(None)  # type: ignore[arg-type]

    def test_aspect_ratio_preservation_widescreen(self) -> None:
        """Verify 16:9 1080p frame is letterboxed without distortion to 640x480."""
        preprocessor = RealtimePreprocessor(target_size=(640, 480), mirror=False)
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        out, meta = preprocessor.process_frame(frame)

        assert out.shape == (480, 640, 3)
        assert meta["mirrored"] is False
        assert meta["raw_shape"] == (1080, 1920)
        # 1920x1080 scaled to max within 640x480: scale = 640/1920 = 1/3
        # new_h = round(1080 / 3) = 360, pad_top + pad_bottom = 120
        assert meta["resized_shape"] == (360, 640)
        assert meta["pad_top"] == 60
        assert meta["pad_bottom"] == 60
        assert meta["pad_left"] == 0
        assert meta["pad_right"] == 0

    def test_aspect_ratio_preservation_portrait(self) -> None:
        """Verify 9:16 portrait frame is letterboxed with pillarbox padding."""
        preprocessor = RealtimePreprocessor(target_size=(640, 480), mirror=False)
        frame = np.ones((1920, 1080, 3), dtype=np.uint8) * 128
        out, meta = preprocessor.process_frame(frame)

        assert out.shape == (480, 640, 3)
        assert meta["raw_shape"] == (1920, 1080)
        # scale = 480 / 1920 = 0.25, new_w = round(1080 * 0.25) = 270
        assert meta["resized_shape"] == (480, 270)
        assert meta["pad_top"] == 0
        assert meta["pad_bottom"] == 0
        assert meta["pad_left"] + meta["pad_right"] == 640 - 270

    def test_horizontal_mirror_toggle(self) -> None:
        """Verify horizontal mirroring correctly flips left and right columns."""
        preprocessor_mirror = RealtimePreprocessor(
            target_size=(100, 100), mirror=True, to_rgb=False
        )
        preprocessor_no_mirror = RealtimePreprocessor(
            target_size=(100, 100), mirror=False, to_rgb=False
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill left 10 columns with 255
        frame[:, :10, :] = 255

        out_mirror, meta_m = preprocessor_mirror.process_frame(frame)
        out_no_mirror, meta_nm = preprocessor_no_mirror.process_frame(frame)

        assert meta_m["mirrored"] is True
        assert meta_nm["mirrored"] is False

        # Mirrored: rightmost columns should be 255
        assert np.all(out_mirror[:, -10:, :] == 255)
        # Not mirrored: leftmost columns should be 255
        assert np.all(out_no_mirror[:, :10, :] == 255)

    def test_color_conversion_bgr_to_rgb(self) -> None:
        """Verify BGR channels are converted to RGB order."""
        preprocessor = RealtimePreprocessor(target_size=(64, 64), mirror=False, to_rgb=True)
        # Create pure BGR blue: B=255, G=0, R=0
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue in BGR

        out, _ = preprocessor.process_frame(frame)
        # In RGB, Red is index 0, Green index 1, Blue index 2
        assert np.all(out[:, :, 0] == 0)    # Red channel is 0
        assert np.all(out[:, :, 1] == 0)    # Green channel is 0
        assert np.all(out[:, :, 2] == 255)  # Blue channel is 255

    def test_sub_3ms_cpu_latency_benchmark(self) -> None:
        """Verify per-frame preprocessing completes under 3 ms on standard 720p/1080p inputs."""
        preprocessor = RealtimePreprocessor(target_size=(640, 480), mirror=True, to_rgb=True)
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        # Warm-up runs
        for _ in range(5):
            preprocessor.process_frame(frame)

        n_iterations = 30
        start = time.perf_counter()
        for _ in range(n_iterations):
            preprocessor.process_frame(frame)
        elapsed_sec = time.perf_counter() - start

        avg_latency_ms = (elapsed_sec / n_iterations) * 1000.0
        # Average latency must be < 3.0 ms per frame on CPU
        assert avg_latency_ms < 3.0, f"Average latency was {avg_latency_ms:.2f} ms (> 3.0 ms)"


class TestFrameBuffer:
    """Test suite for thread-safe circular FrameBuffer."""

    def test_initialization(self) -> None:
        """Verify buffer capacity and empty initialization."""
        buf = FrameBuffer(capacity=60)
        assert buf.capacity == 60
        assert len(buf) == 0
        assert buf.is_full() is False
        assert buf.is_ready(min_frames=30) is False

    def test_invalid_capacity(self) -> None:
        """Verify capacity < 1 raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be at least 1"):
            FrameBuffer(capacity=0)

    def test_empty_or_none_push_raises(self) -> None:
        """Verify pushing None or empty arrays raises ValueError."""
        buf = FrameBuffer(capacity=10)
        with pytest.raises(ValueError, match="Cannot push empty"):
            buf.push(np.zeros((0, 0, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="Cannot push empty"):
            buf.push(None)  # type: ignore[arg-type]

    def test_push_and_fifo_eviction(self) -> None:
        """Verify bounded capacity FIFO eviction when exceeding maxlen."""
        capacity = 5
        buf = FrameBuffer(capacity=capacity)

        for i in range(10):
            frame = np.full((10, 10, 3), fill_value=i, dtype=np.uint8)
            buf.push(frame, timestamp_ms=float(i * 33.3))

        assert len(buf) == capacity
        assert buf.is_full() is True

        # Oldest stored frame should be index 5, newest index 9
        recent = buf.get_recent(5)
        assert len(recent) == 5
        for offset, (f, ts) in enumerate(recent):
            expected_val = 5 + offset
            assert np.all(f == expected_val)
            assert pytest.approx(ts, rel=1e-2) == expected_val * 33.3

    def test_get_recent_bounds(self) -> None:
        """Verify get_recent handles requests exceeding current size gracefully."""
        buf = FrameBuffer(capacity=10)
        assert buf.get_recent(5) == []

        for i in range(3):
            buf.push(np.full((5, 5, 3), i, dtype=np.uint8))

        recent = buf.get_recent(10)
        assert len(recent) == 3

        with pytest.raises(ValueError, match="n_frames must be positive"):
            buf.get_recent(0)

    def test_get_window_and_tensor(self) -> None:
        """Verify get_window returns list and get_tensor returns stacked numpy array."""
        buf = FrameBuffer(capacity=30)
        for i in range(30):
            buf.push(np.full((20, 20, 3), i, dtype=np.uint8))

        assert buf.is_ready(30) is True

        window = buf.get_window(window_size=15)
        assert len(window) == 15
        assert isinstance(window, list)
        assert window[-1][0, 0, 0] == 29

        tensor = buf.get_tensor(window_size=30)
        assert tensor.shape == (30, 20, 20, 3)
        assert tensor[0, 0, 0, 0] == 0
        assert tensor[-1, 0, 0, 0] == 29

    def test_get_tensor_insufficient_frames_raises(self) -> None:
        """Verify get_tensor raises ValueError if requested window exceeds stored count."""
        buf = FrameBuffer(capacity=30)
        for _ in range(5):
            buf.push(np.ones((10, 10, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="fewer than requested window"):
            buf.get_tensor(window_size=10)

    def test_clear(self) -> None:
        """Verify clear resets buffer to zero frames."""
        buf = FrameBuffer(capacity=10)
        for _ in range(5):
            buf.push(np.ones((5, 5, 3), dtype=np.uint8))
        assert len(buf) == 5

        buf.clear()
        assert len(buf) == 0
        assert buf.is_full() is False

    def test_thread_safety(self) -> None:
        """Verify concurrent producer and consumer threads operate safely without corruption."""
        buf = FrameBuffer(capacity=50)
        stop_event = threading.Event()

        def producer() -> None:
            val = 0
            while not stop_event.is_set():
                frame = np.full((16, 16, 3), fill_value=(val % 256), dtype=np.uint8)
                buf.push(frame)
                val += 1
                time.sleep(0.001)

        def consumer() -> None:
            while not stop_event.is_set():
                _ = len(buf)
                _ = buf.is_full()
                _ = buf.get_recent(5)
                time.sleep(0.002)

        p_thread = threading.Thread(target=producer, daemon=True)
        c_thread = threading.Thread(target=consumer, daemon=True)

        p_thread.start()
        c_thread.start()

        time.sleep(0.1)
        stop_event.set()

        p_thread.join(timeout=1.0)
        c_thread.join(timeout=1.0)

        assert len(buf) > 0
        assert len(buf) <= 50
