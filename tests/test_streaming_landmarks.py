"""Unit tests for StreamingHandLandmarker and NormalizedFrameLandmarks (SANKET-042)."""

import numpy as np
import pytest

from sanket.vision.landmarks import FrameLandmarks
from sanket.vision.normalization import normalize_hand_landmarks
from sanket.vision.streaming_landmarks import (
    NormalizedFrameLandmarks,
    StreamingHandLandmarker,
)


class TestNormalizedFrameLandmarks:
    """Test suite for NormalizedFrameLandmarks container."""

    def test_container_properties_and_flat_vector(self) -> None:
        """Verify detected properties and flat vector concatenation."""
        raw_left = np.ones((21, 3), dtype=np.float32)
        raw_right = np.zeros((21, 3), dtype=np.float32)
        raw = FrameLandmarks(
            left_hand=raw_left,
            right_hand=raw_right,
            left_detected=True,
            right_detected=False,
            left_confidence=0.95,
            right_confidence=0.0,
        )

        norm_left = np.full((21, 3), 0.5, dtype=np.float32)
        norm_right = np.zeros((21, 3), dtype=np.float32)

        container = NormalizedFrameLandmarks(
            raw_landmarks=raw,
            left_hand_normalized=norm_left,
            right_hand_normalized=norm_right,
            left_metadata={"method": "palm"},
            right_metadata={"method": "palm"},
            timestamp_ms=100.0,
        )

        assert container.left_detected is True
        assert container.right_detected is False
        assert container.timestamp_ms == 100.0

        vec = container.to_flat_vector()
        assert vec.shape == (126,)
        assert np.all(vec[:63] == 0.5)
        assert np.all(vec[63:] == 0.0)


class TestStreamingHandLandmarker:
    """Test suite for StreamingHandLandmarker tracking and online normalization."""

    def test_initialization_defaults(self) -> None:
        """Verify default initialization parameters."""
        landmarker = StreamingHandLandmarker()
        assert landmarker.num_hands == 2
        assert landmarker.normalization_method == "palm"
        assert landmarker.align_rotation is False
        assert landmarker.impute_strategy == "carry_forward"
        assert landmarker.max_missing_interpolation_frames == 5
        landmarker.close()

    def test_invalid_parameters_raise(self) -> None:
        """Verify parameter validation raises ValueError."""
        with pytest.raises(ValueError, match="num_hands must be 1 or 2"):
            StreamingHandLandmarker(num_hands=3)

        with pytest.raises(ValueError, match="Invalid normalization_method"):
            StreamingHandLandmarker(normalization_method="invalid_norm")

        with pytest.raises(ValueError, match="Invalid impute_strategy"):
            StreamingHandLandmarker(impute_strategy="unsupported")

    def test_invalid_frame_input_raises(self) -> None:
        """Verify None, empty, or wrong shape frames raise ValueError."""
        with StreamingHandLandmarker() as landmarker:
            with pytest.raises(ValueError, match="None or empty"):
                landmarker.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))

            with pytest.raises(ValueError, match="None or empty"):
                landmarker.process_frame(None)  # type: ignore[arg-type]

            with pytest.raises(ValueError, match="Expected 3-channel"):
                landmarker.process_frame(np.zeros((100, 100), dtype=np.uint8))

    def test_mock_backend_process_frame(self) -> None:
        """Verify processing synthetic frame returns expected NormalizedFrameLandmarks structure."""
        with StreamingHandLandmarker() as landmarker:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = landmarker.process_frame(frame, timestamp_ms=33.3)

            assert isinstance(result, NormalizedFrameLandmarks)
            assert result.left_hand_normalized.shape == (21, 3)
            assert result.right_hand_normalized.shape == (21, 3)
            assert result.timestamp_ms == pytest.approx(33.3, rel=1e-1)
            assert isinstance(result.left_metadata, dict)
            assert isinstance(result.right_metadata, dict)

    def test_timestamp_monotonicity_enforcement(self) -> None:
        """Verify non-increasing or missing timestamps are monotonically advanced."""
        with StreamingHandLandmarker() as landmarker:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

            r1 = landmarker.process_frame(frame, timestamp_ms=100.0)
            # Reusing the same timestamp should advance monotonically
            r2 = landmarker.process_frame(frame, timestamp_ms=100.0)
            assert r2.timestamp_ms > r1.timestamp_ms

            # Non-specified timestamp advances automatically
            r3 = landmarker.process_frame(frame, timestamp_ms=None)
            assert r3.timestamp_ms > r2.timestamp_ms

    def test_reset(self) -> None:
        """Verify reset clears frame counters and temporal state."""
        with StreamingHandLandmarker() as landmarker:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = landmarker.process_frame(frame, timestamp_ms=500.0)
            assert landmarker._last_timestamp_ms == 500

            landmarker.reset()
            assert landmarker._last_timestamp_ms == -1
            assert landmarker._frame_index == 0
            assert landmarker._left_missing_count == 0
            assert landmarker._right_missing_count == 0

    def test_mathematical_equivalence_to_offline_normalization(self) -> None:
        """Verify online normalized coordinates match offline normalize_hand_landmarks math."""
        # Create a synthetic hand with known non-zero points
        raw_hand = np.zeros((21, 3), dtype=np.float32)
        raw_hand[0] = [0.5, 0.8, 0.0]  # Wrist
        raw_hand[9] = [0.5, 0.4, 0.0]  # Middle MCP (distance 0.4)
        for i in range(1, 21):
            if i != 9:
                raw_hand[i] = [0.5 + i * 0.01, 0.8 - i * 0.02, 0.01]

        # Offline normalization
        offline_norm, offline_meta = normalize_hand_landmarks(raw_hand, method="palm")

        # Mock online state handler injection
        with StreamingHandLandmarker(normalization_method="palm") as landmarker:
            # Test _handle_hand_state with detected hand
            history = landmarker._left_history
            handled = landmarker._handle_hand_state(
                coords=raw_hand,
                detected=True,
                history=history,
                missing_count_attr="_left_missing_count",
                frame_idx=0,
            )
            online_norm, online_meta = normalize_hand_landmarks(handled, method="palm")

            np.testing.assert_allclose(online_norm, offline_norm, atol=1e-6)
            assert online_meta["method"] == offline_meta["method"]
            assert pytest.approx(online_meta["scale_factor"]) == offline_meta["scale_factor"]

    def test_missing_hand_carry_forward_imputation(self) -> None:
        """Verify carry_forward holds coordinates during brief dropouts up to threshold."""
        with StreamingHandLandmarker(max_missing_interpolation_frames=3) as landmarker:
            sample_hand = np.ones((21, 3), dtype=np.float32) * 5.0
            history = landmarker._left_history

            # Frame 0: Hand detected
            out0 = landmarker._handle_hand_state(
                sample_hand, True, history, "_left_missing_count", 0
            )
            assert np.all(out0 == 5.0)

            # Frame 1: Hand dropped (missing frame 1 <= 3) -> should carry forward
            out1 = landmarker._handle_hand_state(
                np.zeros((21, 3), dtype=np.float32), False, history, "_left_missing_count", 1
            )
            assert np.all(out1 == 5.0)

            # Frame 2: Hand dropped (missing frame 2 <= 3) -> should carry forward
            out2 = landmarker._handle_hand_state(
                np.zeros((21, 3), dtype=np.float32), False, history, "_left_missing_count", 2
            )
            assert np.all(out2 == 5.0)

            # Frame 3: Hand dropped (missing frame 3 <= 3) -> should carry forward
            out3 = landmarker._handle_hand_state(
                np.zeros((21, 3), dtype=np.float32), False, history, "_left_missing_count", 3
            )
            assert np.all(out3 == 5.0)

            # Frame 4: Hand dropped (missing frame 4 > 3) -> falls back to zeros
            out4 = landmarker._handle_hand_state(
                np.zeros((21, 3), dtype=np.float32), False, history, "_left_missing_count", 4
            )
            assert np.all(out4 == 0.0)

            # Frame 5: Hand re-enters view -> immediately resumes detection
            new_hand = np.ones((21, 3), dtype=np.float32) * 9.0
            out5 = landmarker._handle_hand_state(new_hand, True, history, "_left_missing_count", 5)
            assert np.all(out5 == 9.0)
            assert landmarker._left_missing_count == 0

    def test_missing_hand_zero_strategy(self) -> None:
        """Verify zero strategy outputs zeros immediately without carry forward."""
        with StreamingHandLandmarker(impute_strategy="zero") as landmarker:
            sample_hand = np.ones((21, 3), dtype=np.float32) * 3.0
            history = landmarker._left_history

            _ = landmarker._handle_hand_state(sample_hand, True, history, "_left_missing_count", 0)

            # Dropped frame with zero strategy
            out_dropped = landmarker._handle_hand_state(
                np.zeros((21, 3), dtype=np.float32), False, history, "_left_missing_count", 1
            )
            assert np.all(out_dropped == 0.0)
