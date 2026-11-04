"""Confidence calibration, Shannon entropy gating, and out-of-vocabulary thresholding (SANKET-045).

Protects real-time sign recognition against false positives by detecting:
1. Low confidence / ambiguous predictions below dynamic thresholds.
2. High-entropy / flat probability distributions (out-of-vocabulary or transition noise).
3. Resting or idle hands via landmark kinetic energy / motion estimation.
"""

from __future__ import annotations

import collections
import enum
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sanket.realtime.engine import InferenceResult
from sanket.realtime.smoothing import SmoothedPrediction
from sanket.vision.landmarks import FrameLandmarks, SequenceLandmarks
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks

logger = logging.getLogger(__name__)

IDLE_GLOSS: str = "IDLE"
NO_SIGN_GLOSS: str = "NO_SIGN"
IDLE_CLASS_ID: int = -1
NO_SIGN_CLASS_ID: int = -2


class SensitivityProfile(str, enum.Enum):
    """Sensitivity configuration presets for thresholding and gating."""

    PERMISSIVE = "permissive"
    BALANCED = "balanced"
    STRICT = "strict"


@dataclass(frozen=True)
class CalibratedPrediction:
    """Confidence-calibrated and OOV-gated prediction result."""

    gloss: str
    class_id: int
    confidence: float
    entropy: float
    kinetic_energy: float
    is_valid_sign: bool
    is_resting: bool
    rejection_reason: str | None
    raw_gloss: str
    raw_confidence: float
    probabilities: dict[str, float]
    timestamp_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_shannon_entropy(
    probabilities: Sequence[float] | dict[str, float] | np.ndarray,
) -> float:
    """Compute the normalized Shannon entropy (base 2) of a probability distribution.

    Args:
        probabilities: Array, sequence, or dict of class probabilities summing to ~1.0.

    Returns:
        Entropy in bits: H(P) = -sum(p * log2(p)). If single class or empty, returns 0.0.
    """
    if isinstance(probabilities, dict):
        probs = list(probabilities.values())
    else:
        probs = list(probabilities)

    if not probs:
        return 0.0

    total = sum(probs)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for p in probs:
        p_norm = p / total
        if p_norm > 1e-12:
            entropy -= p_norm * math.log2(p_norm)

    return float(entropy)


def _extract_points(
    landmarks: FrameLandmarks | NormalizedFrameLandmarks | np.ndarray,
) -> np.ndarray:
    """Helper to extract active 3D landmark points as a float32 array of shape (N, 3)."""
    if isinstance(landmarks, NormalizedFrameLandmarks):
        pts = []
        if landmarks.left_detected:
            pts.append(landmarks.left_hand_normalized)
        if landmarks.right_detected:
            pts.append(landmarks.right_hand_normalized)
        if pts:
            return np.concatenate(pts, axis=0).astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    if isinstance(landmarks, FrameLandmarks):
        pts = []
        if landmarks.left_detected:
            pts.append(landmarks.left_hand)
        if landmarks.right_detected:
            pts.append(landmarks.right_hand)
        if pts:
            return np.concatenate(pts, axis=0).astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1 and arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    return np.zeros((0, 3), dtype=np.float32)


def compute_kinetic_energy(
    current_landmarks: FrameLandmarks | NormalizedFrameLandmarks | np.ndarray,
    previous_landmarks: FrameLandmarks | NormalizedFrameLandmarks | np.ndarray | None,
    delta_time_s: float | None = None,
) -> float:
    """Compute normalized kinetic motion energy of hand landmarks between consecutive frames.

    Args:
        current_landmarks: Current frame's landmark data.
        previous_landmarks: Prior frame's landmark data (if available).
        delta_time_s: Elapsed time between frames in seconds.

    Returns:
        Mean Euclidean displacement per tracked landmark (or velocity if delta_time_s provided).
    """
    if previous_landmarks is None:
        return 0.0

    curr_pts = _extract_points(current_landmarks)
    prev_pts = _extract_points(previous_landmarks)

    min_pts = min(len(curr_pts), len(prev_pts))
    if min_pts == 0:
        return 0.0

    dims = min(curr_pts.shape[1], prev_pts.shape[1], 3)
    diff = curr_pts[:min_pts, :dims] - prev_pts[:min_pts, :dims]
    distances = np.linalg.norm(diff, axis=1)

    mean_displacement = float(np.mean(distances))

    if delta_time_s is not None and delta_time_s > 0:
        return mean_displacement / delta_time_s

    return mean_displacement


class ConfidenceCalibrator:
    """Confidence calibration, Shannon entropy gating, and resting-hand detector.

    Prevents false-positive sign detections in unconstrained camera streams.
    """

    PROFILE_CONFIGS = {
        SensitivityProfile.PERMISSIVE: {
            "min_confidence": 0.50,
            "max_entropy": 2.80,
            "motion_energy_threshold": 0.005,
        },
        SensitivityProfile.BALANCED: {
            "min_confidence": 0.65,
            "max_entropy": 2.20,
            "motion_energy_threshold": 0.012,
        },
        SensitivityProfile.STRICT: {
            "min_confidence": 0.78,
            "max_entropy": 1.60,
            "motion_energy_threshold": 0.025,
        },
    }

    def __init__(
        self,
        min_confidence: float = 0.65,
        max_entropy: float = 2.20,
        motion_energy_threshold: float = 0.012,
        motion_window_size: int = 5,
        profile: SensitivityProfile | str = SensitivityProfile.BALANCED,
    ) -> None:
        """Initialize ConfidenceCalibrator with thresholds or preset profile.

        Args:
            min_confidence: Minimum probability required for top-1 class hypothesis.
            max_entropy: Maximum Shannon entropy permitted before rejecting as OOV.
            motion_energy_threshold: Minimum mean landmark velocity/energy to be considered active.
            motion_window_size: Number of frames to average motion energy over.
            profile: Sensitivity profile (permissive, balanced, strict).
        """
        if isinstance(profile, str):
            profile = SensitivityProfile(profile.lower())

        preset = self.PROFILE_CONFIGS[profile]
        self.profile = profile
        self.min_confidence = min_confidence if min_confidence != 0.65 else preset["min_confidence"]
        self.max_entropy = max_entropy if max_entropy != 2.20 else preset["max_entropy"]
        motion_thresh = (
            motion_energy_threshold
            if motion_energy_threshold != 0.012
            else preset["motion_energy_threshold"]
        )
        self.motion_energy_threshold = motion_thresh

        self.motion_window_size = max(1, motion_window_size)
        self._recent_motion: collections.deque[float] = collections.deque(
            maxlen=self.motion_window_size
        )
        self._prev_landmarks: FrameLandmarks | NormalizedFrameLandmarks | np.ndarray | None = None
        self._prev_timestamp_ms: float | None = None

    def set_profile(self, profile: SensitivityProfile | str) -> None:
        """Update calibration thresholds to match a sensitivity profile."""
        if isinstance(profile, str):
            profile = SensitivityProfile(profile.lower())
        self.profile = profile
        cfg = self.PROFILE_CONFIGS[profile]
        self.min_confidence = cfg["min_confidence"]
        self.max_entropy = cfg["max_entropy"]
        self.motion_energy_threshold = cfg["motion_energy_threshold"]

    def update_motion(
        self,
        landmarks: (
            FrameLandmarks | NormalizedFrameLandmarks | SequenceLandmarks | np.ndarray | None
        ),
        timestamp_ms: float | None = None,
    ) -> float:
        """Update internal motion energy tracker with latest landmark frame.

        Args:
            landmarks: Latest landmark frame or sequence.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            Smoothed mean motion energy across the recent window.
        """
        if landmarks is None:
            self._recent_motion.append(0.0)
            self._prev_landmarks = None
            return 0.0

        current_frame: FrameLandmarks | NormalizedFrameLandmarks | np.ndarray | None = None
        if isinstance(landmarks, SequenceLandmarks):
            num_frames = getattr(landmarks, "num_frames", None)
            if num_frames is None:
                num_frames = len(landmarks.left_hand)
            if num_frames > 0:
                # Extract last frame as FrameLandmarks
                idx = num_frames - 1
                current_frame = FrameLandmarks(
                    left_hand=landmarks.left_hand[idx],
                    right_hand=landmarks.right_hand[idx],
                    left_detected=bool(landmarks.left_detected[idx]),
                    right_detected=bool(landmarks.right_detected[idx]),
                    left_confidence=float(landmarks.left_confidence[idx]),
                    right_confidence=float(landmarks.right_confidence[idx]),
                )
        else:
            current_frame = landmarks

        if current_frame is None:
            self._recent_motion.append(0.0)
            self._prev_landmarks = None
            return 0.0

        delta_t_s = None
        if timestamp_ms is not None and self._prev_timestamp_ms is not None:
            delta_t_s = max(1e-4, (timestamp_ms - self._prev_timestamp_ms) / 1000.0)

        energy = compute_kinetic_energy(current_frame, self._prev_landmarks, delta_time_s=delta_t_s)
        self._recent_motion.append(energy)

        self._prev_landmarks = current_frame
        self._prev_timestamp_ms = timestamp_ms

        return float(np.mean(self._recent_motion)) if self._recent_motion else 0.0

    def get_average_motion_energy(self) -> float:
        """Return the running average kinetic motion energy."""
        return float(np.mean(self._recent_motion)) if self._recent_motion else 0.0

    def evaluate(
        self,
        prediction: InferenceResult | SmoothedPrediction,
        landmarks: (
            FrameLandmarks | NormalizedFrameLandmarks | SequenceLandmarks | np.ndarray | None
        ) = None,
        timestamp_ms: float | None = None,
    ) -> CalibratedPrediction:
        """Calibrate and gate an incoming raw or smoothed prediction.

        Args:
            prediction: Raw `InferenceResult` or temporally smoothed `SmoothedPrediction`.
            landmarks: Optional recent landmark frame to verify hand motion.
            timestamp_ms: Optional timestamp in ms.

        Returns:
            CalibratedPrediction with gated gloss (`NO_SIGN` or `IDLE` if rejected).
        """
        if isinstance(prediction, SmoothedPrediction):
            raw_gloss = prediction.active_gloss
            raw_class_id = prediction.active_class
            raw_conf = prediction.confidence
            probs = prediction.smoothed_probabilities
            ts_ms = timestamp_ms if timestamp_ms is not None else prediction.timestamp_ms
        elif isinstance(prediction, InferenceResult):
            raw_gloss = prediction.predicted_gloss
            raw_class_id = prediction.predicted_class
            raw_conf = prediction.confidence
            probs = prediction.probabilities
            ts_ms = timestamp_ms if timestamp_ms is not None else prediction.timestamp_ms
        else:
            raise TypeError(f"Unsupported prediction type: {type(prediction)}")

        if landmarks is not None:
            motion_energy = self.update_motion(landmarks, ts_ms)
        else:
            motion_energy = self.get_average_motion_energy()

        entropy = compute_shannon_entropy(probs)

        is_resting = False
        rejection_reason: str | None = None

        # 1. Motion energy check (resting / idle hands)
        if landmarks is not None and motion_energy < self.motion_energy_threshold:
            is_resting = True
            rejection_reason = "kinetic_energy_below_threshold"

        # 2. Entropy check (flat/uncertain distribution -> OOV or transition)
        elif entropy > self.max_entropy:
            rejection_reason = "entropy_exceeds_threshold"

        # 3. Confidence threshold check
        elif raw_conf < self.min_confidence:
            rejection_reason = "confidence_below_threshold"

        if rejection_reason is not None:
            if is_resting:
                gated_gloss = IDLE_GLOSS
                gated_class = IDLE_CLASS_ID
            else:
                gated_gloss = NO_SIGN_GLOSS
                gated_class = NO_SIGN_CLASS_ID
            is_valid_sign = False
        else:
            gated_gloss = raw_gloss
            gated_class = raw_class_id
            is_valid_sign = True

        return CalibratedPrediction(
            gloss=gated_gloss,
            class_id=gated_class,
            confidence=raw_conf,
            entropy=entropy,
            kinetic_energy=motion_energy,
            is_valid_sign=is_valid_sign,
            is_resting=is_resting,
            rejection_reason=rejection_reason,
            raw_gloss=raw_gloss,
            raw_confidence=raw_conf,
            probabilities=probs,
            timestamp_ms=ts_ms,
            metadata={
                "profile": self.profile.value,
                "min_confidence": self.min_confidence,
                "max_entropy": self.max_entropy,
                "motion_threshold": self.motion_energy_threshold,
            },
        )

    def reset(self) -> None:
        """Reset motion history and prior landmark states."""
        self._recent_motion.clear()
        self._prev_landmarks = None
        self._prev_timestamp_ms = None
