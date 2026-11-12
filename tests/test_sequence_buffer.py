"""Unit tests for sliding temporal window sequence buffer (SANKET-063)."""

from __future__ import annotations

import numpy as np
import pytest

from sanket.realtime.sequence_buffer import SlidingSequenceBuffer, SlidingWindowOutput
from sanket.vision.landmarks import FrameLandmarks
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks


def _create_mock_frame(ts_ms: float = 100.0, val: float = 1.0) -> NormalizedFrameLandmarks:
    """Helper generating a mock NormalizedFrameLandmarks."""
    raw_lm = FrameLandmarks(
        left_hand=np.full((21, 3), val, dtype=np.float32),
        right_hand=np.zeros((21, 3), dtype=np.float32),
        left_detected=True,
        right_detected=False,
        left_confidence=0.9,
        right_confidence=0.0,
    )
    return NormalizedFrameLandmarks(
        raw_landmarks=raw_lm,
        left_hand_normalized=np.full((21, 3), val, dtype=np.float32),
        right_hand_normalized=np.zeros((21, 3), dtype=np.float32),
        left_metadata={},
        right_metadata={},
        timestamp_ms=ts_ms,
    )


class TestSlidingSequenceBuffer:
    """Test suite validating FIFO window buffering and stride behavior."""

    def test_buffer_initialization_validation(self) -> None:
        buf = SlidingSequenceBuffer(window_size=30, stride=3)
        assert buf.window_size == 30
        assert buf.stride == 3
        assert buf.current_length == 0
        assert buf.total_frames_added == 0

        with pytest.raises(ValueError, match="window_size must be positive"):
            SlidingSequenceBuffer(window_size=0)
        with pytest.raises(ValueError, match="stride must be positive"):
            SlidingSequenceBuffer(stride=0)

    def test_first_frame_triggers_stride_under_repeat_first(self) -> None:
        buf = SlidingSequenceBuffer(window_size=10, stride=3, warmup_policy="repeat_first")
        frame0 = _create_mock_frame(ts_ms=100.0, val=2.5)

        # First frame triggers initial output with warmup replication
        out = buf.add_frame(frame0)
        assert isinstance(out, SlidingWindowOutput)
        assert out.is_ready is True
        assert out.window_size == 10
        assert out.stride_index == 1
        assert out.sequence.num_frames == 10
        # All 10 frames should be replicated from frame0
        assert np.all(out.sequence.left_hand == 2.5)

    def test_stride_step_intervals(self) -> None:
        buf = SlidingSequenceBuffer(window_size=10, stride=3, warmup_policy="zero")

        # Frame 1: initial trigger (total=1)
        out1 = buf.add_frame(_create_mock_frame(ts_ms=10.0))
        assert out1 is not None

        # Frame 2, 3: awaiting stride 3
        assert buf.add_frame(_create_mock_frame(ts_ms=20.0)) is None
        assert buf.add_frame(_create_mock_frame(ts_ms=30.0)) is None

        # Frame 4: triggers (stride=3 after initial)
        out4 = buf.add_frame(_create_mock_frame(ts_ms=40.0))
        assert out4 is not None
        assert out4.stride_index == 2

    def test_zero_warmup_padding_policy(self) -> None:
        buf = SlidingSequenceBuffer(window_size=5, stride=1, warmup_policy="zero")
        buf.add_frame(_create_mock_frame(val=5.0))
        buf.add_frame(_create_mock_frame(val=6.0))

        # We have 2 frames, front 3 should be zeros
        seq = buf.get_current_window()
        assert seq.num_frames == 5
        assert np.all(seq.left_hand[:3] == 0.0)
        assert np.all(seq.left_hand[3] == 5.0)
        assert np.all(seq.left_hand[4] == 6.0)

    def test_partial_warmup_policy(self) -> None:
        buf = SlidingSequenceBuffer(window_size=4, stride=1, warmup_policy="partial")
        assert buf.add_frame(_create_mock_frame()) is None
        assert buf.add_frame(_create_mock_frame()) is None
        assert buf.add_frame(_create_mock_frame()) is None
        # 4th frame fills the buffer -> triggers output
        out = buf.add_frame(_create_mock_frame())
        assert out is not None
        assert out.sequence.num_frames == 4

    def test_buffer_rollover_bounded_memory(self) -> None:
        buf = SlidingSequenceBuffer(window_size=5, stride=2)
        for i in range(25):
            buf.add_frame(_create_mock_frame(ts_ms=float(i * 33), val=float(i)))

        assert buf.current_length == 5
        assert buf.total_frames_added == 25

        # Check most recent 5 frames in FIFO: [20, 21, 22, 23, 24]
        seq = buf.get_current_window()
        expected = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        for j in range(5):
            assert np.all(seq.left_hand[j] == expected[j])

    def test_reset_behavior(self) -> None:
        buf = SlidingSequenceBuffer(window_size=10, stride=3)
        buf.add_frame(_create_mock_frame())
        buf.add_frame(_create_mock_frame())
        assert buf.current_length == 2

        buf.reset()
        assert buf.current_length == 0
        assert buf.total_frames_added == 0

