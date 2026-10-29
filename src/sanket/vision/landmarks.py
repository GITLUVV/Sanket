"""Hand landmark extraction subsystem using MediaPipe Hands (SANKET-024).

Extracts 21 3D coordinate keypoints (x, y, z) for dominant and non-dominant hands
across single images and standardized video sequences. Includes missing hand detection
handling, confidence scoring, temporal carry-forward imputation, and compressed .npz
serialization.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Constants
NUM_HAND_LANDMARKS = 21
COORDINATE_DIM = 3  # (x, y, z)
HAND_FEATURE_DIM = NUM_HAND_LANDMARKS * COORDINATE_DIM  # 63
TOTAL_FEATURE_DIM = HAND_FEATURE_DIM * 2  # 126 (left + right)

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class FrameLandmarks:
    """Skeletal hand landmark coordinates extracted from a single video frame."""

    left_hand: np.ndarray  # Shape: (21, 3), dtype: float32
    right_hand: np.ndarray  # Shape: (21, 3), dtype: float32
    left_detected: bool
    right_detected: bool
    left_confidence: float
    right_confidence: float

    def to_flat_vector(self) -> np.ndarray:
        """Concatenate left and right hand landmarks into a 1D feature vector of length 126."""
        left_flat = self.left_hand.reshape(-1)
        right_flat = self.right_hand.reshape(-1)
        return np.concatenate([left_flat, right_flat]).astype(np.float32)


@dataclass
class SequenceLandmarks:
    """Skeletal hand landmark coordinates across a temporal sequence of T frames."""

    left_hand: np.ndarray  # Shape: (T, 21, 3), dtype: float32
    right_hand: np.ndarray  # Shape: (T, 21, 3), dtype: float32
    left_detected: np.ndarray  # Shape: (T,), dtype: bool
    right_detected: np.ndarray  # Shape: (T,), dtype: bool
    left_confidence: np.ndarray  # Shape: (T,), dtype: float32
    right_confidence: np.ndarray  # Shape: (T,), dtype: float32

    @property
    def num_frames(self) -> int:
        """Number of temporal frames T in this sequence."""
        return int(self.left_hand.shape[0])

    def to_flat_matrix(self) -> np.ndarray:
        """Flatten sequence into a 2D feature matrix of shape (T, 126)."""
        t = self.num_frames
        left_flat = self.left_hand.reshape(t, -1)
        right_flat = self.right_hand.reshape(t, -1)
        return np.concatenate([left_flat, right_flat], axis=1).astype(np.float32)

    def impute_missing(self, strategy: str = "carry_forward") -> SequenceLandmarks:
        """Impute missing hand landmark coordinates across the temporal sequence.

        Supported strategies:
            - 'zero': Zero-padding (leaves missing hands as all zeros).
            - 'carry_forward': Forward-fills missing frames from the most recently
              detected keyframe. Backfills if initial frames are undetected.
              If a hand is never detected in the sequence, remains zeros.
            - 'linear': Linearly interpolates between detected keyframes along time.

        Args:
            strategy: Imputation method ('zero', 'carry_forward', 'linear').

        Returns:
            A new SequenceLandmarks instance with imputed coordinate arrays.
        """
        valid_strategies = {"zero", "carry_forward", "linear"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Choose from {valid_strategies}")

        if strategy == "zero":
            return SequenceLandmarks(
                left_hand=self.left_hand.copy(),
                right_hand=self.right_hand.copy(),
                left_detected=self.left_detected.copy(),
                right_detected=self.right_detected.copy(),
                left_confidence=self.left_confidence.copy(),
                right_confidence=self.right_confidence.copy(),
            )

        imputed_left = _impute_hand_sequence(self.left_hand, self.left_detected, strategy)
        imputed_right = _impute_hand_sequence(self.right_hand, self.right_detected, strategy)

        return SequenceLandmarks(
            left_hand=imputed_left,
            right_hand=imputed_right,
            left_detected=self.left_detected.copy(),
            right_detected=self.right_detected.copy(),
            left_confidence=self.left_confidence.copy(),
            right_confidence=self.right_confidence.copy(),
        )


def _impute_hand_sequence(
    coordinates: np.ndarray,
    detected_mask: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Helper to apply temporal imputation to a single hand coordinate sequence (T, 21, 3)."""
    t = coordinates.shape[0]
    result = coordinates.copy()

    # If completely undetected or completely detected, return as is
    detected_indices = np.where(detected_mask)[0]
    if len(detected_indices) == 0 or len(detected_indices) == t:
        return result

    if strategy == "carry_forward":
        # Forward fill
        last_known = None
        for i in range(t):
            if detected_mask[i]:
                last_known = result[i]
            elif last_known is not None:
                result[i] = last_known

        # Backfill initial frames if needed
        first_detected = detected_indices[0]
        if first_detected > 0:
            result[:first_detected] = result[first_detected]

    elif strategy == "linear":
        # Linear interpolation per coordinate across time
        for i in range(NUM_HAND_LANDMARKS):
            for dim in range(COORDINATE_DIM):
                values = coordinates[:, i, dim]
                # Interpolate using detected indices as x
                interp_vals = np.interp(
                    np.arange(t),
                    detected_indices,
                    values[detected_indices],
                )
                result[:, i, dim] = interp_vals

    return result.astype(np.float32)


def ensure_hand_landmarker_model(model_path: str | Path | None = None) -> Path:
    """Resolve or automatically download the official MediaPipe HandLandmarker task model.

    Args:
        model_path: Explicit path to the model file. If None, checks default locations.

    Returns:
        Path to verified hand_landmarker.task file.

    Raises:
        RuntimeError: If download fails and no model exists.
    """
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified model file does not exist: {model_path}")

    # Check candidates
    candidates = [
        Path("models/hand_landmarker.task"),
        Path.home() / ".cache" / "sanket" / "models" / "hand_landmarker.task",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Download to user cache
    target_path = Path.home() / ".cache" / "sanket" / "models" / "hand_landmarker.task"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading official MediaPipe HandLandmarker model to %s...", target_path)
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(target_path))
        logger.info(
            "Downloaded hand_landmarker.task successfully (%d bytes)",
            target_path.stat().st_size,
        )
        return target_path
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MediaPipe HandLandmarker model from {DEFAULT_MODEL_URL}: {exc}"
        ) from exc


def save_landmarks_npz(
    sequence: SequenceLandmarks,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save SequenceLandmarks to a compressed .npz archive on disk.

    Args:
        sequence: The SequenceLandmarks object.
        output_path: Target .npz file path.
        metadata: Optional dictionary of metadata attributes (e.g., class, signer, fps).

    Returns:
        Path to the saved .npz archive.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "left_hand": sequence.left_hand,
        "right_hand": sequence.right_hand,
        "left_detected": sequence.left_detected,
        "right_detected": sequence.right_detected,
        "left_confidence": sequence.left_confidence,
        "right_confidence": sequence.right_confidence,
    }

    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = v

    np.savez_compressed(str(dest), **save_dict)
    logger.info("Saved SequenceLandmarks archive: %s", dest)
    return dest


def load_landmarks_npz(path: str | Path) -> tuple[SequenceLandmarks, dict[str, Any]]:
    """Load SequenceLandmarks and any embedded metadata from a .npz archive.

    Args:
        path: Path to the .npz archive.

    Returns:
        Tuple of (SequenceLandmarks, metadata_dict).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Landmark archive not found: {path}")

    with np.load(str(p), allow_pickle=True) as data:
        seq = SequenceLandmarks(
            left_hand=data["left_hand"].astype(np.float32),
            right_hand=data["right_hand"].astype(np.float32),
            left_detected=data["left_detected"].astype(bool),
            right_detected=data["right_detected"].astype(bool),
            left_confidence=data["left_confidence"].astype(np.float32),
            right_confidence=data["right_confidence"].astype(np.float32),
        )
        metadata = {}
        for k in data.files:
            if k.startswith("meta_"):
                metadata[k[5:]] = data[k].item() if data[k].ndim == 0 else data[k]

    return seq, metadata


class HandLandmarkExtractor:
    """MediaPipe Tasks vision-based 3D hand landmark extraction engine."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_hands: int = 2,
    ) -> None:
        """Initialize MediaPipe HandLandmarker.

        Args:
            model_path: Path to hand_landmarker.task bundle. If None, auto-resolved.
            min_detection_confidence: Minimum hand detection confidence threshold.
            min_tracking_confidence: Minimum hand landmark tracking confidence threshold.
            num_hands: Maximum number of hands to detect (default: 2).
        """
        self.model_path = ensure_hand_landmarker_model(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_hands = num_hands

        try:
            # Lazy import to ensure compatibility
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self._mp = mp
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
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

    def __enter__(self) -> HandLandmarkExtractor:
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

    def extract_frame(self, frame_rgb: np.ndarray) -> FrameLandmarks:
        """Extract 21 3D hand landmarks from a single RGB image frame.

        Args:
            frame_rgb: RGB image array of shape (H, W, 3) and dtype uint8.

        Returns:
            FrameLandmarks container with left and right coordinates and confidence.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            raise ValueError("Cannot extract landmarks from None or empty image frame")
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {frame_rgb.shape}")

        if getattr(self, "_backend", "mediapipe") == "mock" or self._detector is None:
            return FrameLandmarks(
                left_hand=np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32),
                right_hand=np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32),
                left_detected=False,
                right_detected=False,
                left_confidence=0.0,
                right_confidence=0.0,
            )

        # Ensure contiguous uint8 array
        frame_u8 = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_u8)

        result = self._detector.detect(mp_image)

        # Initialize empty defaults
        left_hand = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        right_hand = np.zeros((NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        left_detected = False
        right_detected = False
        left_confidence = 0.0
        right_confidence = 0.0

        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(
                result.hand_landmarks, result.handedness, strict=False
            ):
                category = handedness[0]
                label = category.category_name  # "Left" or "Right"
                score = float(category.score)

                # Extract (x, y, z) coordinates into (21, 3) array
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32,
                )

                if label == "Left":
                    left_hand = coords
                    left_detected = True
                    left_confidence = score
                elif label == "Right":
                    right_hand = coords
                    right_detected = True
                    right_confidence = score

        return FrameLandmarks(
            left_hand=left_hand,
            right_hand=right_hand,
            left_detected=left_detected,
            right_detected=right_detected,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
        )

    def extract_sequence(
        self,
        frames: list[np.ndarray] | np.ndarray,
        impute_strategy: str | None = None,
    ) -> SequenceLandmarks:
        """Extract skeletal landmarks across a temporal sequence of video frames.

        Args:
            frames: Sequence of RGB frames as a list or a numpy array (T, H, W, 3).
            impute_strategy: Optional imputation strategy to fill missing hand detections
                ('zero', 'carry_forward', 'linear').

        Returns:
            SequenceLandmarks object containing all temporal landmark arrays.
        """
        if len(frames) == 0:
            raise ValueError("Cannot extract landmarks from empty frame sequence")

        frame_list = [f for f in frames]
        t = len(frame_list)

        left_hands = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        right_hands = np.zeros((t, NUM_HAND_LANDMARKS, COORDINATE_DIM), dtype=np.float32)
        left_detected = np.zeros(t, dtype=bool)
        right_detected = np.zeros(t, dtype=bool)
        left_conf = np.zeros(t, dtype=np.float32)
        right_conf = np.zeros(t, dtype=np.float32)

        for i, frame in enumerate(frame_list):
            res = self.extract_frame(frame)
            left_hands[i] = res.left_hand
            right_hands[i] = res.right_hand
            left_detected[i] = res.left_detected
            right_detected[i] = res.right_detected
            left_conf[i] = res.left_confidence
            right_conf[i] = res.right_confidence

        seq = SequenceLandmarks(
            left_hand=left_hands,
            right_hand=right_hands,
            left_detected=left_detected,
            right_detected=right_detected,
            left_confidence=left_conf,
            right_confidence=right_conf,
        )

        if impute_strategy:
            return seq.impute_missing(strategy=impute_strategy)
        return seq

    def extract_video(
        self,
        video_path: str | Path,
        target_frames: int = 30,
        target_size: tuple[int, int] = (640, 480),
        letterbox: bool = True,
        impute_strategy: str | None = "carry_forward",
    ) -> SequenceLandmarks:
        """Preprocess a video file and extract its skeletal hand landmark sequence.

        Integrates with the SANKET-023 video preprocessing pipeline.

        Args:
            video_path: Path to the video file.
            target_frames: Fixed sequence length T (default 30).
            target_size: Standardized spatial frame size (default (640, 480)).
            letterbox: Whether to use aspect-ratio preserving letterboxing (default True).
            impute_strategy: Missing hand imputation strategy (default 'carry_forward').

        Returns:
            SequenceLandmarks object.
        """
        # Late import to prevent circular dependencies
        from sanket.vision.preprocessing import sample_video_frames

        tensor = sample_video_frames(
            video_path=video_path,
            target_frames=target_frames,
            target_size=target_size,
            letterbox=letterbox,
            to_rgb=True,
        )
        return self.extract_sequence(tensor, impute_strategy=impute_strategy)
