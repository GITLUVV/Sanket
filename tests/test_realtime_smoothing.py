"""Unit tests for TemporalPredictionSmoother and SmoothedPrediction (SANKET-044)."""

from __future__ import annotations

import time

import pytest

from sanket.realtime.engine import InferenceResult
from sanket.realtime.smoothing import TemporalPredictionSmoother


def _make_dummy_inference_result(
    pred_class: int,
    confidence: float = 0.9,
    timestamp_ms: float = 100.0,
    num_classes: int = 5,
    class_names: list[str] | None = None,
) -> InferenceResult:
    """Helper to synthesize test InferenceResult instances."""
    names = class_names if class_names is not None else [f"sign_{c}" for c in range(num_classes)]
    probas = {}
    for c in range(num_classes):
        name = names[c]
        if c == pred_class:
            probas[name] = confidence
        else:
            probas[name] = (1.0 - confidence) / (num_classes - 1)

    return InferenceResult(
        predicted_gloss=names[pred_class],
        predicted_class=pred_class,
        confidence=confidence,
        probabilities=probas,
        feature_latency_ms=1.0,
        model_latency_ms=2.0,
        total_latency_ms=3.0,
        timestamp_ms=timestamp_ms,
    )


class TestTemporalPredictionSmoother:
    """Test suite for TemporalPredictionSmoother."""

    def test_parameter_validation(self) -> None:
        """Verify invalid parameter values raise ValueError."""
        with pytest.raises(ValueError, match="ema_alpha"):
            TemporalPredictionSmoother(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            TemporalPredictionSmoother(ema_alpha=1.5)
        with pytest.raises(ValueError, match="window_size"):
            TemporalPredictionSmoother(window_size=0)
        with pytest.raises(ValueError, match="hysteresis_threshold"):
            TemporalPredictionSmoother(hysteresis_threshold=0)
        with pytest.raises(ValueError, match="min_dwell_ms"):
            TemporalPredictionSmoother(min_dwell_ms=-10.0)

    def test_initial_state_and_commit(self) -> None:
        """Verify initial state is UNCERTAIN and commits after initial hysteresis threshold."""
        smoother = TemporalPredictionSmoother(
            num_classes=5,
            classes=[f"sign_{i}" for i in range(5)],
            hysteresis_threshold=3,
        )
        assert smoother.active_gloss == "UNCERTAIN"
        assert smoother.active_class == -1

        # Frame 1: Candidate sign_0
        r1 = smoother.process(_make_dummy_inference_result(pred_class=0, timestamp_ms=0.0))
        # Initial threshold min(2, 3) is 2, so frame 1 shouldn't commit yet
        assert r1.active_class == -1
        assert r1.active_gloss == "UNCERTAIN"
        assert r1.state_changed is False

        # Frame 2: Candidate sign_0 again -> reaches 2 -> commits
        r2 = smoother.process(_make_dummy_inference_result(pred_class=0, timestamp_ms=33.3))
        assert r2.active_class == 0
        assert r2.active_gloss == "sign_0"
        assert r2.state_changed is True

    def test_noise_and_flicker_suppression(self) -> None:
        """Verify transient 1-2 frame spikes do not disrupt the active sign state."""
        smoother = TemporalPredictionSmoother(
            num_classes=5,
            classes=[f"sign_{i}" for i in range(5)],
            window_size=5,
            hysteresis_threshold=4,
            min_dwell_ms=100.0,
        )

        # Establish sign_1 as active sign
        for i in range(5):
            smoother.process(_make_dummy_inference_result(pred_class=1, timestamp_ms=i * 33.3))

        assert smoother.active_class == 1
        assert smoother.active_gloss == "sign_1"

        # Frame 6: Transient sensor noise flips raw prediction to sign_3
        r_noise1 = smoother.process(
            _make_dummy_inference_result(pred_class=3, timestamp_ms=6 * 33.3)
        )
        assert r_noise1.raw_class == 3
        # Active state MUST remain sign_1
        assert r_noise1.active_class == 1
        assert r_noise1.active_gloss == "sign_1"
        assert r_noise1.state_changed is False

        # Frame 7: Transient noise flips to sign_4
        r_noise2 = smoother.process(
            _make_dummy_inference_result(pred_class=4, timestamp_ms=7 * 33.3)
        )
        assert r_noise2.raw_class == 4
        assert r_noise2.active_class == 1
        assert r_noise2.active_gloss == "sign_1"
        assert r_noise2.state_changed is False

        # Frame 8: Resumes sign_1
        r_resume = smoother.process(
            _make_dummy_inference_result(pred_class=1, timestamp_ms=8 * 33.3)
        )
        assert r_resume.active_class == 1
        assert r_resume.state_changed is False

    def test_intentional_state_transition_with_hysteresis(self) -> None:
        """Verify intentional transition to new sign requires consecutive frames and dwell."""
        smoother = TemporalPredictionSmoother(
            num_classes=5,
            classes=[f"sign_{i}" for i in range(5)],
            window_size=5,
            hysteresis_threshold=3,
            min_dwell_ms=100.0,
        )

        # Establish sign_0 for 200 ms (exceeding 100 ms dwell)
        for i in range(6):
            smoother.process(_make_dummy_inference_result(pred_class=0, timestamp_ms=i * 35.0))

        assert smoother.active_class == 0

        # Now start emitting sign_2 consistently
        # Frame 1 of sign_2 (ts = 210 ms)
        r1 = smoother.process(_make_dummy_inference_result(pred_class=2, timestamp_ms=210.0))
        assert r1.active_class == 0  # Still old state

        # Frame 2 of sign_2 (ts = 245 ms)
        r2 = smoother.process(_make_dummy_inference_result(pred_class=2, timestamp_ms=245.0))
        assert r2.active_class == 0  # Still old state (count 2 < 3)

        # Frame 3 of sign_2 (ts = 280 ms) -> count reaches threshold 3, dwell satisfied
        r3 = smoother.process(_make_dummy_inference_result(pred_class=2, timestamp_ms=280.0))
        assert r3.active_class == 2
        assert r3.active_gloss == "sign_2"
        assert r3.state_changed is True

    def test_minimum_dwell_time_blocking(self) -> None:
        """Verify state change is blocked if min_dwell_ms has not elapsed."""
        smoother = TemporalPredictionSmoother(
            num_classes=5,
            classes=[f"sign_{i}" for i in range(5)],
            window_size=3,
            hysteresis_threshold=2,
            min_dwell_ms=200.0,
        )

        # Commit sign_0 at ts = 0 ms
        smoother.process(_make_dummy_inference_result(pred_class=0, timestamp_ms=0.0))
        smoother.process(_make_dummy_inference_result(pred_class=0, timestamp_ms=30.0))
        assert smoother.active_class == 0

        # Attempt to rapidly switch to sign_1 at ts = 60 ms and 90 ms (< 200 ms dwell)
        smoother.process(_make_dummy_inference_result(pred_class=1, timestamp_ms=60.0))
        r_blocked = smoother.process(_make_dummy_inference_result(pred_class=1, timestamp_ms=90.0))

        # Blocked due to min_dwell_ms
        assert r_blocked.active_class == 0

        # After dwell time elapses at ts = 250 ms, transition succeeds
        r_allowed = smoother.process(_make_dummy_inference_result(pred_class=1, timestamp_ms=250.0))
        assert r_allowed.active_class == 1
        assert r_allowed.state_changed is True

    def test_ema_probability_smoothing(self) -> None:
        """Verify EMA smoothly blends probability distributions across frames."""
        smoother = TemporalPredictionSmoother(
            num_classes=3,
            classes=["A", "B", "C"],
            ema_alpha=0.5,
        )

        res1 = _make_dummy_inference_result(
            pred_class=0,
            confidence=1.0,
            timestamp_ms=0.0,
            num_classes=3,
            class_names=["A", "B", "C"],
        )
        res1_smoothed = smoother.process(res1)
        assert res1_smoothed.smoothed_probabilities["A"] == pytest.approx(1.0, rel=1e-3)

        # Frame 2: sudden shift to class B (p(B)=1.0, p(A)=0.0)
        res2 = _make_dummy_inference_result(
            pred_class=1,
            confidence=1.0,
            timestamp_ms=33.0,
            num_classes=3,
            class_names=["A", "B", "C"],
        )
        res2_smoothed = smoother.process(res2)

        # With alpha=0.5: p(A) should be 0.5 * 0 + 0.5 * 1.0 = 0.5
        # p(B) should be 0.5 * 1.0 + 0.5 * 0 = 0.5
        assert res2_smoothed.smoothed_probabilities["A"] == pytest.approx(0.5, rel=1e-2)
        assert res2_smoothed.smoothed_probabilities["B"] == pytest.approx(0.5, rel=1e-2)

    def test_reset(self) -> None:
        """Verify reset clears history and resets active gloss to UNCERTAIN."""
        smoother = TemporalPredictionSmoother(num_classes=5)
        for i in range(5):
            smoother.process(_make_dummy_inference_result(pred_class=2, timestamp_ms=i * 30.0))
        assert smoother.active_class == 2

        smoother.reset()
        assert smoother.active_class == -1
        assert smoother.active_gloss == "UNCERTAIN"

    def test_processing_latency(self) -> None:
        """Verify smoothing processing executes under 1 ms per evaluation."""
        smoother = TemporalPredictionSmoother(num_classes=16)
        res = _make_dummy_inference_result(pred_class=5, num_classes=16)

        t0 = time.perf_counter()
        n_iter = 100
        for i in range(n_iter):
            smoother.process(res, timestamp_ms=float(i * 30))
        elapsed_s = time.perf_counter() - t0

        avg_latency_ms = (elapsed_s / n_iter) * 1000.0
        assert avg_latency_ms < 1.0, f"Average latency was {avg_latency_ms:.3f} ms (> 1.0 ms)"
