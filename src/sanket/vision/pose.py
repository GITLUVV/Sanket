"""Auxiliary upper-body pose reference anchor extraction (SANKET-025).

Provides MediaPipe Pose integration to extract critical upper-body reference anchors
(nose, mouth center, shoulders, sternum) and calculate hand-to-body relative distance
and displacement vectors for disambiguating location-dependent sign language pairs.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sanket.vision.landmarks import FrameLandmarks

logger = logging.getLogger(__name__)

# Keypoint indices from 33-point MediaPipe Pose topology
NOSE_IDX = 0
MOUTH_LEFT_IDX = 9
MOUTH_RIGHT_IDX = 10
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

NUM_POSE_ANCHORS = 5
ANCHOR_NAMES = ["nose", "mouth_center", "sternum", "left_shoulder", "right_shoulder"]

DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class PoseAnchors:
    """Canonical upper-body reference anchors extracted from a video frame."""

    nose: np.ndarray  # Shape: (3,), dtype: float32
    mouth_center: np.ndarray  # Shape: (3,), dtype: float32
    left_shoulder: np.ndarray  # Shape: (3,), dtype: float32
    right_shoulder: np.ndarray  # Shape: (3,), dtype: float32
    sternum: np.ndarray  # Shape: (3,), dtype: float32 (midpoint of shoulders)
    shoulder_width: float  # Euclidean distance between left and right shoulders
    is_detected: bool
    confidence: float

    def to_anchor_matrix(self) -> np.ndarray:
        """Return a (5, 3) matrix containing all 5 canonical body reference points."""
        return np.stack(
            [
                self.nose,
                self.mouth_center,
                self.sternum,
                self.left_shoulder,
                self.right_shoulder,
            ],
            axis=0,
        ).astype(np.float32)


@dataclass(frozen=True)
class HandBodyRelativeFeatures:
    """Relative 3D distance and displacement vectors between hands and body anchors."""

    left_wrist_distances: np.ndarray  # Shape: (5,)
    left_wrist_displacements: np.ndarray  # Shape: (5, 3)
    right_wrist_distances: np.ndarray  # Shape: (5,)
    right_wrist_displacements: np.ndarray  # Shape: (5, 3)
    left_index_distances: np.ndarray  # Shape: (5,)
    left_index_displacements: np.ndarray  # Shape: (5, 3)
    right_index_distances: np.ndarray  # Shape: (5,)
    right_index_displacements: np.ndarray  # Shape: (5, 3)
    normalized_by_shoulder_width: bool

    def to_vector(self, include_displacements: bool = True) -> np.ndarray:
        """Flatten into a 1D feature vector.

        Args:
            include_displacements: If True, includes both scalar Euclidean distances
                and 3D directional displacement vectors (80 features total: 20 distances
                + 60 displacement components). If False, includes scalar distances only
                (20 features total).

        Returns:
            1D float32 numpy array.
        """
        distances = np.concatenate(
            [
                self.left_wrist_distances,
                self.right_wrist_distances,
                self.left_index_distances,
                self.right_index_distances,
            ]
        )  # Length: 20

        if not include_displacements:
            return distances.astype(np.float32)

        displacements = np.concatenate(
            [
                self.left_wrist_displacements.reshape(-1),
                self.right_wrist_displacements.reshape(-1),
                self.left_index_displacements.reshape(-1),
                self.right_index_displacements.reshape(-1),
            ]
        )  # Length: 60 (20 * 3)

        return np.concatenate([distances, displacements]).astype(np.float32)


def compute_sternum(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> np.ndarray:
    """Compute the anatomical sternum anchor point as the midpoint of both shoulders.

    Args:
        left_shoulder: 3D coordinates (3,) of the left shoulder.
        right_shoulder: 3D coordinates (3,) of the right shoulder.

    Returns:
        3D coordinates (3,) of the sternum midpoint.
    """
    return ((left_shoulder + right_shoulder) / 2.0).astype(np.float32)


def compute_relative_spatial_features(
    hand_landmarks: FrameLandmarks,
    pose_anchors: PoseAnchors,
    normalize_by_shoulder_width: bool = True,
) -> HandBodyRelativeFeatures:
    """Compute relative distances and 3D displacement vectors between hands and body anchors.

    Args:
        hand_landmarks: Hand landmarks for the current frame.
        pose_anchors: Canonical upper-body reference anchors for the current frame.
        normalize_by_shoulder_width: Whether to normalize Euclidean distances by the
            signer's shoulder width for scale and distance invariance.

    Returns:
        HandBodyRelativeFeatures container.
    """
    # Scale normalization factor (protect against divide-by-zero or non-detections)
    scale = 1.0
    if (
        normalize_by_shoulder_width
        and pose_anchors.is_detected
        and pose_anchors.shoulder_width > 1e-4
    ):
        scale = pose_anchors.shoulder_width

    anchor_matrix = pose_anchors.to_anchor_matrix()  # Shape: (5, 3)

    def _calculate_hand_anchor_rel(
        keypoint: np.ndarray, is_detected: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if not is_detected or not pose_anchors.is_detected:
            return (
                np.zeros(NUM_POSE_ANCHORS, dtype=np.float32),
                np.zeros((NUM_POSE_ANCHORS, 3), dtype=np.float32),
            )

        # 3D displacements: keypoint - anchor
        displacements = (keypoint - anchor_matrix) / scale
        distances = np.linalg.norm(displacements, axis=1)
        return distances.astype(np.float32), displacements.astype(np.float32)

    # Wrist: landmark 0, Index fingertip: landmark 8
    lw_dist, lw_disp = _calculate_hand_anchor_rel(
        hand_landmarks.left_hand[0], hand_landmarks.left_detected
    )
    rw_dist, rw_disp = _calculate_hand_anchor_rel(
        hand_landmarks.right_hand[0], hand_landmarks.right_detected
    )
    li_dist, li_disp = _calculate_hand_anchor_rel(
        hand_landmarks.left_hand[8], hand_landmarks.left_detected
    )
    ri_dist, ri_disp = _calculate_hand_anchor_rel(
        hand_landmarks.right_hand[8], hand_landmarks.right_detected
    )

    return HandBodyRelativeFeatures(
        left_wrist_distances=lw_dist,
        left_wrist_displacements=lw_disp,
        right_wrist_distances=rw_dist,
        right_wrist_displacements=rw_disp,
        left_index_distances=li_dist,
        left_index_displacements=li_disp,
        right_index_distances=ri_dist,
        right_index_displacements=ri_disp,
        normalized_by_shoulder_width=normalize_by_shoulder_width,
    )


def ensure_pose_landmarker_model(model_path: str | Path | None = None) -> Path:
    """Resolve or automatically download the official MediaPipe PoseLandmarker task model.

    Args:
        model_path: Explicit path to model file. If None, checks default locations.

    Returns:
        Path to verified pose_landmarker_lite.task file.

    Raises:
        RuntimeError: If download fails and no model exists.
    """
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified model file does not exist: {model_path}")

    candidates = [
        Path("models/pose_landmarker_lite.task"),
        Path.home() / ".cache" / "sanket" / "models" / "pose_landmarker_lite.task",
    ]
    for c in candidates:
        if c.is_file():
            return c

    target_path = Path.home() / ".cache" / "sanket" / "models" / "pose_landmarker_lite.task"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MediaPipe PoseLandmarker model to %s...", target_path)
    try:
        urllib.request.urlretrieve(DEFAULT_POSE_MODEL_URL, str(target_path))
        logger.info(
            "Downloaded pose_landmarker_lite.task successfully (%d bytes)",
            target_path.stat().st_size,
        )
        return target_path
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MediaPipe PoseLandmarker model from "
            f"{DEFAULT_POSE_MODEL_URL}: {exc}"
        ) from exc


class PoseExtractor:
    """Upper-body pose reference anchor extraction engine."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        backend: str = "auto",
        detector: Any = None,
    ) -> None:
        """Initialize MediaPipe PoseLandmarker.

        Args:
            model_path: Path to pose_landmarker_lite.task bundle. If None, auto-resolved.
            min_detection_confidence: Minimum detection confidence threshold.
            backend: Backend engine ('auto', 'mediapipe', or 'mock').
            detector: Optional injected detector object (primarily for testing).
        """
        self.min_detection_confidence = min_detection_confidence

        if detector is not None:
            self.model_path = Path(model_path) if model_path is not None else None
            self._detector = detector
            self._backend = "mediapipe"
            try:
                import mediapipe as mp

                self._mp = mp
            except (ImportError, OSError):
                from unittest.mock import MagicMock

                self._mp = MagicMock()
            return

        if backend == "mock":
            self.model_path = None
            self._detector = None
            self._backend = "mock"
            self._mp = None
            return

        self.model_path = ensure_pose_landmarker_model(model_path)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self._mp = mp
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
            )
            self._detector = vision.PoseLandmarker.create_from_options(options)
            self._backend = "mediapipe"
        except (ImportError, OSError) as exc:
            logger.warning(
                "MediaPipe Pose native backend unavailable (%s). Falling back to mock.",
                exc,
            )
            self._detector = None
            self._backend = "mock"

    @property
    def backend(self) -> str:
        """Return the active pose extraction backend ('mediapipe' or 'mock')."""
        return self._backend

    @property
    def detector(self) -> Any:
        """Return the underlying detector instance or None."""
        return self._detector

    def __enter__(self) -> PoseExtractor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release native detector resources."""
        if hasattr(self, "_detector") and self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def extract_frame(self, frame_rgb: np.ndarray) -> PoseAnchors:
        """Extract canonical upper-body pose reference anchors from an RGB frame.

        Args:
            frame_rgb: RGB image array of shape (H, W, 3) and dtype uint8.

        Returns:
            PoseAnchors container.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            raise ValueError("Cannot extract pose from None or empty image frame")
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {frame_rgb.shape}")

        if getattr(self, "_backend", "mediapipe") == "mock" or self._detector is None:
            return PoseAnchors(
                nose=np.zeros(3, dtype=np.float32),
                mouth_center=np.zeros(3, dtype=np.float32),
                left_shoulder=np.zeros(3, dtype=np.float32),
                right_shoulder=np.zeros(3, dtype=np.float32),
                sternum=np.zeros(3, dtype=np.float32),
                shoulder_width=0.0,
                is_detected=False,
                confidence=0.0,
            )

        frame_u8 = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_u8)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            return PoseAnchors(
                nose=np.zeros(3, dtype=np.float32),
                mouth_center=np.zeros(3, dtype=np.float32),
                left_shoulder=np.zeros(3, dtype=np.float32),
                right_shoulder=np.zeros(3, dtype=np.float32),
                sternum=np.zeros(3, dtype=np.float32),
                shoulder_width=0.0,
                is_detected=False,
                confidence=0.0,
            )

        def _to_coord(idx: int) -> np.ndarray:
            lm = result.pose_landmarks[0][idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        nose = _to_coord(NOSE_IDX)
        mouth_l = _to_coord(MOUTH_LEFT_IDX)
        mouth_r = _to_coord(MOUTH_RIGHT_IDX)
        mouth_center = ((mouth_l + mouth_r) / 2.0).astype(np.float32)

        left_sh = _to_coord(LEFT_SHOULDER_IDX)
        right_sh = _to_coord(RIGHT_SHOULDER_IDX)
        sternum = compute_sternum(left_sh, right_sh)
        shoulder_width = float(np.linalg.norm(left_sh - right_sh))

        confidence = 1.0
        return PoseAnchors(
            nose=nose,
            mouth_center=mouth_center,
            left_shoulder=left_sh,
            right_shoulder=right_sh,
            sternum=sternum,
            shoulder_width=shoulder_width,
            is_detected=True,
            confidence=confidence,
        )
