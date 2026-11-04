"""Unit tests for confidence calibration, Shannon entropy, and OOV thresholding (SANKET-045)."""

from __future__ import annotations

import math

import numpy as np

from sanket.realtime.engine import InferenceResult
from sanket.realtime.smoothing import SmoothedPrediction
from sanket.realtime.thresholding import (
    IDLE_CLASS_ID,
    IDLE_GLOSS,
    NO_SIGN_CLASS_ID,
    NO_SIGN_GLOSS,
    CalibratedPrediction,
    ConfidenceCalibrator,
    SensitivityProfile,
    compute_kinetic_energy,
    compute_shannon_entropy,
)
from sanket.vision.landmarks import FrameLandmarks, SequenceLandmarks


def _make_dummy_inference(
    gloss: str = "namaste",
    class_id: int = 1,
    confidence: float = 0.90,
    probs: dict[str, float] | None = None,
    ts_ms: float = 1000.0,
) -> InferenceResult:
    if probs is None:
        probs = {gloss: confidence, "other": 1.0 - confidence}
    return InferenceResult(
        predicted_gloss=gloss,
        predicted_class=class_id,
        confidence=confidence,
        probabilities=probs,
        feature_latency_ms=1.0,
        model_latency_ms=2.5,
        total_latency_ms=3.5,
        timestamp_ms=ts_ms,
    )


def _make_dummy_landmarks(coords_offset: float = 0.0) -> FrameLandmarks:
    # 21 hand landmarks with 3 coordinates
    pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        pts[i] = [i * 0.01 + coords_offset, i * 0.02 + coords_offset, 0.0]
    return FrameLandmarks(
        left_hand=pts,
        right_hand=pts,
        left_detected=True,
        right_detected=True,
        left_confidence=0.95,
        right_confidence=0.95,
    )


class TestEntropyAndEnergyCalculations:
    """Test mathematical formulas for entropy and landmark kinetic motion."""

    def test_shannon_entropy_deterministic_distribution(self) -> None:
        # A completely confident prediction (p=1.0) has 0 entropy
        entropy = compute_shannon_entropy({"A": 1.0, "B": 0.0, "C": 0.0})
        assert math.isclose(entropy, 0.0, abs_tol=1e-6)

    def test_shannon_entropy_uniform_distribution(self) -> None:
        # Uniform distribution over 4 classes: -4 * (0.25 * log2(0.25)) = -log2(0.25) = 2.0 bits
        entropy = compute_shannon_entropy([0.25, 0.25, 0.25, 0.25])
        assert math.isclose(entropy, 2.0, abs_tol=1e-5)

    def test_shannon_entropy_empty_or_zero(self) -> None:
        assert compute_shannon_entropy([]) == 0.0
        assert compute_shannon_entropy([0.0, 0.0]) == 0.0

    def test_kinetic_energy_static_landmarks(self) -> None:
        frame1 = _make_dummy_landmarks(0.0)
        frame2 = _make_dummy_landmarks(0.0)
        energy = compute_kinetic_energy(frame2, frame1, delta_time_s=0.033)
        assert math.isclose(energy, 0.0, abs_tol=1e-6)

    def test_kinetic_energy_moved_landmarks(self) -> None:
        frame1 = _make_dummy_landmarks(0.0)
        frame2 = _make_dummy_landmarks(0.05)
        # Displacement for each coord in x and y is 0.05
        # Euclidean = sqrt(0.05^2 + 0.05^2) = 0.07071
        energy = compute_kinetic_energy(frame2, frame1, delta_time_s=None)
        expected_disp = math.sqrt(0.05**2 + 0.05**2)
        assert math.isclose(energy, expected_disp, rel_tol=1e-4)

    def test_kinetic_energy_no_prior_frame(self) -> None:
        frame = _make_dummy_landmarks(0.0)
        assert compute_kinetic_energy(frame, None) == 0.0


class TestConfidenceCalibrator:
    """Test confidence calibration, thresholding, and gating rules."""

    def test_accepts_high_confidence_active_prediction(self) -> None:
        calibrator = ConfidenceCalibrator(profile=SensitivityProfile.BALANCED)
        pred = _make_dummy_inference(gloss="namaste", confidence=0.92)

        # Simulate movement
        f1 = _make_dummy_landmarks(0.0)
        f2 = _make_dummy_landmarks(0.05)
        calibrator.update_motion(f1, timestamp_ms=100.0)

        result = calibrator.evaluate(pred, landmarks=f2, timestamp_ms=133.0)

        assert isinstance(result, CalibratedPrediction)
        assert result.is_valid_sign is True
        assert result.gloss == "namaste"
        assert result.class_id == 1
        assert result.is_resting is False
        assert result.rejection_reason is None

    def test_rejects_resting_hand_as_idle(self) -> None:
        calibrator = ConfidenceCalibrator(
            motion_energy_threshold=0.01,
            motion_window_size=3,
        )
        pred = _make_dummy_inference(gloss="namaste", confidence=0.88)

        # Hand barely moving (< 0.001)
        f1 = _make_dummy_landmarks(0.0)
        f2 = _make_dummy_landmarks(0.0001)
        calibrator.update_motion(f1, timestamp_ms=100.0)

        result = calibrator.evaluate(pred, landmarks=f2, timestamp_ms=133.0)

        assert result.is_valid_sign is False
        assert result.is_resting is True
        assert result.gloss == IDLE_GLOSS
        assert result.class_id == IDLE_CLASS_ID
        assert result.rejection_reason == "kinetic_energy_below_threshold"

    def test_rejects_high_entropy_out_of_vocabulary_as_no_sign(self) -> None:
        calibrator = ConfidenceCalibrator(max_entropy=1.5, profile=SensitivityProfile.STRICT)

        # 8 uniformly distributed classes -> entropy = 3.0 bits > 1.5 threshold
        uniform_probs = {f"class_{i}": 0.125 for i in range(8)}
        pred = _make_dummy_inference(gloss="class_0", confidence=0.80, probs=uniform_probs)

        # Ensure motion is active
        f1 = _make_dummy_landmarks(0.0)
        f2 = _make_dummy_landmarks(0.1)
        calibrator.update_motion(f1, timestamp_ms=100.0)

        result = calibrator.evaluate(pred, landmarks=f2, timestamp_ms=133.0)

        assert result.is_valid_sign is False
        assert result.is_resting is False
        assert result.gloss == NO_SIGN_GLOSS
        assert result.class_id == NO_SIGN_CLASS_ID
        assert result.rejection_reason == "entropy_exceeds_threshold"

    def test_rejects_low_confidence_below_cutoff(self) -> None:
        calibrator = ConfidenceCalibrator(min_confidence=0.70)
        probs = {"namaste": 0.55, "dhanyavad": 0.45}
        pred = _make_dummy_inference(gloss="namaste", confidence=0.55, probs=probs)

        # Active movement
        f1 = _make_dummy_landmarks(0.0)
        f2 = _make_dummy_landmarks(0.1)
        calibrator.update_motion(f1, timestamp_ms=100.0)

        result = calibrator.evaluate(pred, landmarks=f2, timestamp_ms=133.0)

        assert result.is_valid_sign is False
        assert result.gloss == NO_SIGN_GLOSS
        assert result.rejection_reason == "confidence_below_threshold"

    def test_accepts_smoothed_prediction_input(self) -> None:
        calibrator = ConfidenceCalibrator()
        smoothed = SmoothedPrediction(
            active_gloss="alvida",
            active_class=3,
            confidence=0.89,
            raw_gloss="alvida",
            raw_class=3,
            smoothed_probabilities={"alvida": 0.89, "namaste": 0.11},
            state_changed=False,
            dwell_duration_ms=250.0,
            timestamp_ms=1500.0,
        )

        f1 = _make_dummy_landmarks(0.0)
        f2 = _make_dummy_landmarks(0.05)
        calibrator.update_motion(f1, timestamp_ms=1460.0)

        result = calibrator.evaluate(smoothed, landmarks=f2, timestamp_ms=1500.0)
        assert result.is_valid_sign is True
        assert result.gloss == "alvida"

    def test_sensitivity_profiles_switching(self) -> None:
        calibrator = ConfidenceCalibrator(profile="strict")
        assert calibrator.min_confidence == 0.78
        assert calibrator.max_entropy == 1.60

        calibrator.set_profile(SensitivityProfile.PERMISSIVE)
        assert calibrator.min_confidence == 0.50
        assert calibrator.max_entropy == 2.80

    def test_sequence_landmarks_motion_update(self) -> None:
        calibrator = ConfidenceCalibrator()
        l_arr = np.zeros((5, 21, 3), dtype=np.float32)
        r_arr = np.zeros((5, 21, 3), dtype=np.float32)
        det = np.ones(5, dtype=bool)
        conf = np.ones(5, dtype=np.float32)
        seq = SequenceLandmarks(
            left_hand=l_arr,
            right_hand=r_arr,
            left_detected=det,
            right_detected=det,
            left_confidence=conf,
            right_confidence=conf,
        )
        avg_motion = calibrator.update_motion(seq, timestamp_ms=200.0)
        assert avg_motion >= 0.0

    def test_reset_clears_state(self) -> None:
        calibrator = ConfidenceCalibrator()
        calibrator.update_motion(_make_dummy_landmarks(0.05), timestamp_ms=100.0)
        assert len(calibrator._recent_motion) > 0
        calibrator.reset()
        assert len(calibrator._recent_motion) == 0
        assert calibrator._prev_landmarks is None
