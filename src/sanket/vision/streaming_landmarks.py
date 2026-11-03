"""Real-time streaming hand landmark extraction and online normalization (SANKET-042).

Provides continuous video-mode MediaPipe hand tracking (RunningMode.VIDEO), online
wrist-centering and palm-scale coordinate normalization, and missing-hand temporal
imputation for low-latency live camera execution.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sanket.vision.landmarks import (
    COORDINATE_DIM,
    NUM_HAND_LANDMARKS,
    FrameLandmarks,
    ensure_hand_landmarker_model,
)
from sanket.vision.normalization import normalize_hand_landmarks

logger = logging.getLogger(__name__)


@dataclass
class NormalizedFrameLandmarks:
    """Normalized hand landmarks for a single real-time frame with metadata."""

    raw_landmarks: FrameLandmarks
    left_hand_normalized: np.ndarray  # Shape: (21, 3), dtype: float32
    right_hand_normalized: np.ndarray  # Shape: (21, 3), dtype: float32
    left_metadata: dict[str, Any]
    right_metadata: dict[str, Any]
    timestamp_ms: float

    @property
    def left_detected(self) -> bool:
        """Whether left hand was detected."""
        return self.raw_landmarks.left_detected

    @property
    def right_detected(self) -> bool:
        """Whether right hand was detected."""
        return self.raw_landmarks.right_detected

    def to_flat_vector(self) -> np.ndarray:
        """Concatenate normalized left and right landmarks into a 1D vector (length 126)."""
        left_flat = self.left_hand_normalized.reshape(-1)
        right_flat = self.right_hand_normalized.reshape(-1)
        return np.concatenate([left_flat, right_flat]).astype(np.float32)


class StreamingHandLandmarker:
    """Real-time streaming hand landmark tracking and online normalization engine.

    Operates in MediaPipe continuous video tracking mode (`RunningMode.VIDEO`), tracking
    hands across consecutive frames with temporal continuity. Applies immediate online
    wrist-centering and palm-scale normalization, maintaining short sliding history
    for smooth missing-hand imputation during brief tracking dropouts.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_hands: int = 2,
        normalization_method: str = "palm",
        align_rotation: bool = False,
        max_missing_interpolation_frames: int = 5,
        impute_strategy: str = "carry_forward",
    ) -> None:
        """Initialize the real-time streaming hand landmarker.

        Args:
            model_path: Path to hand_landmarker.task model. If None, auto-resolved.
            min_detection_confidence: Minimum detection confidence threshold.
            min_tracking_confidence: Minimum tracking confidence threshold.
            num_hands: Maximum number of hands to track (1 or 2, default 2).
            normalization_method: Scale normalization metric ('palm', 'max_span', 'bbox').
            align_rotation: If True, applies 2D in-plane rotational alignment.
            max_missing_interpolation_frames: Max consecutive missing frames to bridge.
            impute_strategy: Fallback imputation strategy ('carry_forward' or 'zero').

        Raises:
            ValueError: If num_hands not in (1, 2) or invalid normalization method.
        """
        if num_hands not in (1, 2):
            raise ValueError(f"num_hands must be 1 or 2, got {num_hands}")
        if normalization_method not in {"palm", "max_span", "bbox"}:
            raise ValueError(f"Invalid normalization_method '{normalization_method}'")
        if impute_strategy not in {"carry_forward", "zero"}:
            raise ValueError(f"Invalid impute_strategy '{impute_strategy}'")

        self.model_path = ensure_hand_landmarker_model(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_hands = num_hands
        self.normalization_method = normalization_method
        self.align_rotation = align_rotation
        self.max_missing_interpolation_frames = max_missing_interpolation_frames
        self.impute_strategy = impute_strategy

        # Monotonic timestamp tracker (MediaPipe requires strictly increasing timestamps)
        self._last_timestamp_ms: int = -1

        # Tracking drop history for left and right hands
        self._left_history: collections.deque[tuple[np.ndarray, int]] = collections.deque(
            maxlen=max_missing_interpolation_frames + 2
        )
        self._right_history: collections.deque[tuple[np.ndarray, int]] = collections.deque(
            maxlen=max_missing_interpolation_frames + 2
        )
        self._left_missing_count: int = 0
        self._right_missing_count: int = 0
        self._frame_index: int = 0

        # Initialize MediaPipe backend
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self._mp = mp
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            self._backend = "mediapipe"
        except (ImportError, OSError) as exc:
            logger.warning(
                "MediaPipe native backend unavailable (%s). Falling back to mock extractor.",
                exc,
            )
            self._detector = None
            self._backend = "mock"

    def __enter__(self) -> StreamingHandLandmarker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release native MediaPipe detector resources."""
        if hasattr(self, "_detector") and self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def reset(self) -> None:
        """Reset temporal state, missing frame counters, and timestamp history."""
        self._last_timestamp_ms = -1
        self._frame_index = 0
        self._left_missing_count = 0
        self._right_missing_count = 0
        self._left_history.clear()
        self._right_history.clear()

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: float | None = None,
    ) -> NormalizedFrameLandmarks:
        """Extract continuous hand landmarks and apply online normalization.

        Args:
            frame_rgb: RGB image array of shape (H, W, 3) and dtype uint8.
            timestamp_ms: Monotonic timestamp in milliseconds. If None, derived
                from frame index or internal timer.

        Returns:
            NormalizedFrameLandmarks containing raw and normalized hand coordinates,
            metadata dictionaries, and timestamps.

        Raises:
            ValueError: If frame_rgb is None, empty, or incorrectly shaped.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            raise ValueError("Cannot extract landmarks from None or empty image frame")
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {frame_rgb.shape}")

        # Compute strictly increasing integer timestamp in milliseconds
        if timestamp_ms is None:
            calc_ts = int(self._last_timestamp_ms + 33 if self._last_timestamp_ms >= 0 else 0)
        else:
            calc_ts = int(round(timestamp_ms))

        if calc_ts <= self._last_timestamp_ms:
            calc_ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = calc_ts
        current_frame_idx = self._frame_index
        self._frame_index += 1

        left_raw = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        right_raw = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        left_detected = False
        right_detected = False
        left_conf = 0.0
        right_conf = 0.0

        if getattr(self, "_backend", "mediapipe") == "mediapipe" and self._detector is not None:
            frame_u8 = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_u8)

            result = self._detector.detect_for_video(mp_image, calc_ts)

            if result.hand_landmarks and result.handedness:
                for landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness, strict=False
                ):
                    category = handedness[0]
                    label = category.category_name
                    score = float(category.score)

                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in landmarks],
                        dtype=np.float32,
                    )

                    if label == "Left" and not left_detected:
                        left_raw = coords
                        left_detected = True
                        left_conf = score
                    elif label == "Right" and not right_detected:
                        right_raw = coords
                        right_detected = True
                        right_conf = score

        raw_container = FrameLandmarks(
            left_hand=left_raw,
            right_hand=right_raw,
            left_detected=left_detected,
            right_detected=right_detected,
            left_confidence=left_conf,
            right_confidence=right_conf,
        )

        # Handle missing hand imputation
        imputed_left = self._handle_hand_state(
            coords=left_raw,
            detected=left_detected,
            history=self._left_history,
            missing_count_attr="_left_missing_count",
            frame_idx=current_frame_idx,
        )

        imputed_right = self._handle_hand_state(
            coords=right_raw,
            detected=right_detected,
            history=self._right_history,
            missing_count_attr="_right_missing_count",
            frame_idx=current_frame_idx,
        )

        # Online normalization
        left_norm, left_meta = normalize_hand_landmarks(
            landmarks=imputed_left,
            method=self.normalization_method,
            align_rotation=self.align_rotation,
        )

        right_norm, right_meta = normalize_hand_landmarks(
            landmarks=imputed_right,
            method=self.normalization_method,
            align_rotation=self.align_rotation,
        )

        return NormalizedFrameLandmarks(
            raw_landmarks=raw_container,
            left_hand_normalized=left_norm,
            right_hand_normalized=right_norm,
            left_metadata=left_meta,
            right_metadata=right_meta,
            timestamp_ms=float(calc_ts),
        )

    def _handle_hand_state(
        self,
        coords: np.ndarray,
        detected: bool,
        history: collections.deque[tuple[np.ndarray, int]],
        missing_count_attr: str,
        frame_idx: int,
    ) -> np.ndarray:
        """Handle tracking history and imputation for a single hand."""
        if detected:
            setattr(self, missing_count_attr, 0)
            history.append((coords.copy(), frame_idx))
            return coords

        # Hand not detected in this frame
        current_missing = getattr(self, missing_count_attr) + 1
        setattr(self, missing_count_attr, current_missing)

        if self.impute_strategy == "carry_forward" and len(history) > 0:
            if current_missing <= self.max_missing_interpolation_frames:
                # Return most recent detected coordinates
                return history[-1][0].copy()

        # Beyond threshold or zero-imputation strategy
        return np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
