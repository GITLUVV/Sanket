"""Online sliding temporal window buffer for real-time sequence inference (SANKET-063).

Provides a bounded, thread-safe FIFO ring buffer holding T recent normalized landmark
frames. Enables streaming sequence model inference with:
- Configurable window capacity (e.g., T = 30 frames).
- Configurable step stride (e.g., trigger inference every S = 3 frames to avoid redundant compute).
- Startup warmup padding policies ('repeat_first', 'zero', or 'partial') to avoid cold-start delays.
- Extraction of SequenceLandmarks and BatchSequenceTensor ready for recurrent model forward passes.
- Strict memory bounds with zero frame retention.
"""

from __future__ import annotations

import collections
import logging
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sanket.vision.landmarks import COORDINATE_DIM, NUM_HAND_LANDMARKS, SequenceLandmarks
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks

logger = logging.getLogger(__name__)

WarmupPolicy = Literal["repeat_first", "zero", "partial"]


@dataclass(frozen=True)
class SlidingWindowOutput:
    """Output container produced when the sliding buffer reaches an inference stride step.

    Attributes:
        sequence: SequenceLandmarks object representing the temporal window of length T.
        is_ready: True if the buffer has satisfied minimum warmup criteria and stride step.
        window_size: Actual number of frames in the window.
        stride_index: Monotonic counter of inference steps triggered.
        timestamp_ms: Timestamp of the most recent frame in milliseconds.
    """

    sequence: SequenceLandmarks
    is_ready: bool
    window_size: int
    stride_index: int
    timestamp_ms: float


class SlidingSequenceBuffer:
    """Thread-safe sliding temporal window buffer for online sequence model inference.

    Maintains a bounded FIFO queue of normalized landmark frames, advancing by `stride` frames
    between model evaluations to balance inference latency with temporal overlap.
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 3,
        warmup_policy: WarmupPolicy = "repeat_first",
    ) -> None:
        """Initialize the SlidingSequenceBuffer.

        Args:
            window_size: Maximum temporal sequence capacity (frames), default 30.
            stride: Number of new incoming frames required before triggering next inference step.
            warmup_policy: How to populate the window before window_size frames have arrived:
                - 'repeat_first': Replicates the earliest frame to fill the window.
                - 'zero': Zero-pads earlier frames.
                - 'partial': Leaves earlier frames empty / not ready until buffer is full.

        Raises:
            ValueError: If window_size <= 0 or stride <= 0 or invalid warmup_policy.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if warmup_policy not in {"repeat_first", "zero", "partial"}:
            raise ValueError(f"Invalid warmup_policy '{warmup_policy}'")

        self.window_size = window_size
        self.stride = stride
        self.warmup_policy: WarmupPolicy = warmup_policy

        self._lock = threading.Lock()
        self._buffer: collections.deque[NormalizedFrameLandmarks] = collections.deque(
            maxlen=window_size
        )
        self._frames_since_last_stride = 0
        self._total_frames_added = 0
        self._stride_counter = 0

    @property
    def current_length(self) -> int:
        """Current number of frames held in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_frames_added(self) -> int:
        """Total lifetime frames ingested into this buffer."""
        with self._lock:
            return self._total_frames_added

    def reset(self) -> None:
        """Clear all buffered frames and reset stride counters."""
        with self._lock:
            self._buffer.clear()
            self._frames_since_last_stride = 0
            self._total_frames_added = 0
            self._stride_counter = 0

    def add_frame(
        self,
        frame: NormalizedFrameLandmarks,
    ) -> SlidingWindowOutput | None:
        """Ingest a single landmark frame and return SlidingWindowOutput if stride is met.

        Args:
            frame: NormalizedFrameLandmarks instance.

        Returns:
            SlidingWindowOutput if stride is triggered, or None if awaiting stride.
        """
        with self._lock:
            self._buffer.append(frame)
            self._total_frames_added += 1
            self._frames_since_last_stride += 1

            # Check whether an inference step is triggered
            is_stride_step = (self._frames_since_last_stride >= self.stride) or (
                self._total_frames_added == 1
            )
            if not is_stride_step:
                return None

            # Handle partial buffer under 'partial' warmup policy
            buf_len = len(self._buffer)
            if self.warmup_policy == "partial" and buf_len < self.window_size:
                return None

            self._frames_since_last_stride = 0
            self._stride_counter += 1
            stride_idx = self._stride_counter

            # Convert current buffer state into SequenceLandmarks
            sequence = self._materialize_sequence_locked()
            latest_ts = frame.timestamp_ms

            return SlidingWindowOutput(
                sequence=sequence,
                is_ready=True,
                window_size=self.window_size,
                stride_index=stride_idx,
                timestamp_ms=latest_ts,
            )

    def _materialize_sequence_locked(self) -> SequenceLandmarks:
        """Convert deque of NormalizedFrameLandmarks to SequenceLandmarks (locked)."""
        frames_list = list(self._buffer)
        n = len(frames_list)
        t = self.window_size

        left_hands = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        right_hands = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        left_det = np.zeros(t, dtype=bool)
        right_det = np.zeros(t, dtype=bool)
        left_conf = np.zeros(t, dtype=np.float32)
        right_conf = np.zeros(t, dtype=np.float32)

        if n == t:
            # Buffer is completely full
            for i, f in enumerate(frames_list):
                left_hands[i] = f.left_hand_normalized
                right_hands[i] = f.right_hand_normalized
                left_det[i] = f.left_detected
                right_det[i] = f.right_detected
                left_conf[i] = f.raw_landmarks.left_confidence
                right_conf[i] = f.raw_landmarks.right_confidence
        elif self.warmup_policy == "repeat_first" and n > 0:
            # Replicate earliest available frame across the missing front prefix
            pad_len = t - n
            first_f = frames_list[0]
            for i in range(pad_len):
                left_hands[i] = first_f.left_hand_normalized
                right_hands[i] = first_f.right_hand_normalized
                left_det[i] = first_f.left_detected
                right_det[i] = first_f.right_detected
                left_conf[i] = first_f.raw_landmarks.left_confidence
                right_conf[i] = first_f.raw_landmarks.right_confidence

            for j, f in enumerate(frames_list):
                idx = pad_len + j
                left_hands[idx] = f.left_hand_normalized
                right_hands[idx] = f.right_hand_normalized
                left_det[idx] = f.left_detected
                right_det[idx] = f.right_detected
                left_conf[idx] = f.raw_landmarks.left_confidence
                right_conf[idx] = f.raw_landmarks.right_confidence
        else:  # 'zero' or partial
            # Place existing frames at the tail end, leaving front as zero padding
            offset = t - n
            for j, f in enumerate(frames_list):
                idx = offset + j
                left_hands[idx] = f.left_hand_normalized
                right_hands[idx] = f.right_hand_normalized
                left_det[idx] = f.left_detected
                right_det[idx] = f.right_detected
                left_conf[idx] = f.raw_landmarks.left_confidence
                right_conf[idx] = f.raw_landmarks.right_confidence

        return SequenceLandmarks(
            left_hand=left_hands,
            right_hand=right_hands,
            left_detected=left_det,
            right_detected=right_det,
            left_confidence=left_conf,
            right_confidence=right_conf,
        )

    def get_current_window(self) -> SequenceLandmarks:
        """Retrieve the current temporal window as SequenceLandmarks without advancing stride."""
        with self._lock:
            return self._materialize_sequence_locked()

