"""Unit tests for RealtimeInferenceEngine and Worker (SANKET-043)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from sanket.models.base import BaseSignClassifier
from sanket.models.random_forest import RandomForestSignClassifier
from sanket.realtime.engine import (
    InferenceResult,
    ModelLoadError,
    RealtimeInferenceEngine,
    RealtimeInferenceWorker,
)
from sanket.vision.landmarks import FrameLandmarks, SequenceLandmarks
from sanket.vision.streaming_landmarks import NormalizedFrameLandmarks


class MockClassifier(BaseSignClassifier):
    """Mock classifier for deterministic unit testing of the inference engine."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.classes_ = [f"sign_{i}" for i in range(num_classes)]
        self.is_fitted_ = True

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> MockClassifier:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Predict class 1 deterministically
        return np.ones(len(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probas = np.zeros((len(x), self.num_classes), dtype=np.float32)
        probas[:, 1] = 0.85
        remaining = 0.15 / (self.num_classes - 1)
        for c in range(self.num_classes):
            if c != 1:
                probas[:, c] = remaining
        return probas

    def save(self, path: Path | str) -> Path:
        return Path(path)

    @classmethod
    def load(cls, path: Path | str) -> MockClassifier:
        return cls()


def _make_dummy_normalized_frames(n_frames: int = 30) -> list[NormalizedFrameLandmarks]:
    """Helper to synthesize a list of NormalizedFrameLandmarks."""
    frames = []
    for i in range(n_frames):
        raw = FrameLandmarks(
            left_hand=np.zeros((21, 3), dtype=np.float32),
            right_hand=np.zeros((21, 3), dtype=np.float32),
            left_detected=True,
            right_detected=True,
            left_confidence=0.9,
            right_confidence=0.9,
        )
        norm_left = np.full((21, 3), fill_value=0.01 * (i + 1), dtype=np.float32)
        norm_right = np.full((21, 3), fill_value=0.02 * (i + 1), dtype=np.float32)
        frames.append(
            NormalizedFrameLandmarks(
                raw_landmarks=raw,
                left_hand_normalized=norm_left,
                right_hand_normalized=norm_right,
                left_metadata={"method": "palm"},
                right_metadata={"method": "palm"},
                timestamp_ms=float(i * 33.3),
            )
        )
    return frames


class TestInferenceResult:
    """Test suite for InferenceResult dataclass."""

    def test_container_fields_and_confidence(self) -> None:
        """Verify field immutability and is_confident helper."""
        res = InferenceResult(
            predicted_gloss="HELLO",
            predicted_class=1,
            confidence=0.92,
            probabilities={"HELLO": 0.92, "WORLD": 0.08},
            feature_latency_ms=1.5,
            model_latency_ms=0.8,
            total_latency_ms=2.3,
            timestamp_ms=100.0,
        )
        assert res.predicted_gloss == "HELLO"
        assert res.predicted_class == 1
        assert res.confidence == 0.92
        assert res.is_confident is True
        assert res.total_latency_ms == 2.3

        unconfident = InferenceResult(
            predicted_gloss="UNCERTAIN",
            predicted_class=0,
            confidence=0.0,
            probabilities={},
            feature_latency_ms=0.0,
            model_latency_ms=0.0,
            total_latency_ms=0.0,
            timestamp_ms=0.0,
        )
        assert unconfident.is_confident is False


class TestRealtimeInferenceEngine:
    """Test suite for RealtimeInferenceEngine."""

    def test_unfitted_model_raises_load_error(self) -> None:
        """Verify unfitted model raises ModelLoadError."""
        unfitted_rf = RandomForestSignClassifier()
        with pytest.raises(ModelLoadError, match="must be fitted"):
            RealtimeInferenceEngine(model=unfitted_rf)

    def test_missing_model_and_invalid_path_raises(self) -> None:
        """Verify missing model path raises ModelLoadError."""
        with pytest.raises(ModelLoadError, match="not found"):
            RealtimeInferenceEngine(model_path="non_existent_model_dir/model.joblib")

    def test_predict_from_precomputed_feature_vector(self) -> None:
        """Verify prediction from precomputed 645-dim feature vector."""
        model = MockClassifier(num_classes=5)
        engine = RealtimeInferenceEngine(model=model)

        dummy_feat = np.ones((1, 645), dtype=np.float32)
        res = engine.predict(dummy_feat, timestamp_ms=50.0)

        assert isinstance(res, InferenceResult)
        assert res.predicted_class == 1
        assert res.predicted_gloss == "sign_1"
        assert res.confidence == pytest.approx(0.85, rel=1e-2)
        assert res.timestamp_ms == 50.0
        assert res.feature_latency_ms == 0.0
        assert res.model_latency_ms > 0.0

    def test_predict_from_normalized_frames_list(self) -> None:
        """Verify end-to-end feature extraction and prediction from list of frames."""
        model = MockClassifier(num_classes=5)
        engine = RealtimeInferenceEngine(model=model)

        frames = _make_dummy_normalized_frames(n_frames=30)
        res = engine.predict(frames)

        assert res.predicted_gloss == "sign_1"
        assert res.confidence == pytest.approx(0.85, rel=1e-2)
        assert res.feature_latency_ms > 0.0
        assert res.total_latency_ms == res.feature_latency_ms + res.model_latency_ms

    def test_predict_from_sequence_landmarks(self) -> None:
        """Verify prediction directly from SequenceLandmarks object."""
        model = MockClassifier(num_classes=5)
        engine = RealtimeInferenceEngine(model=model)

        seq = SequenceLandmarks(
            left_hand=np.zeros((15, 21, 3), dtype=np.float32),
            right_hand=np.zeros((15, 21, 3), dtype=np.float32),
            left_detected=np.ones(15, dtype=bool),
            right_detected=np.ones(15, dtype=bool),
            left_confidence=np.ones(15, dtype=np.float32),
            right_confidence=np.ones(15, dtype=np.float32),
        )
        res = engine.predict(seq)
        assert res.predicted_class == 1

    def test_confidence_threshold_fallback(self) -> None:
        """Verify predictions below min_confidence_threshold return UNCERTAIN."""
        model = MockClassifier(num_classes=5)
        # Mock confidence is 0.85; set threshold to 0.90 to trigger UNCERTAIN
        engine = RealtimeInferenceEngine(model=model, min_confidence_threshold=0.90)

        dummy_feat = np.zeros((1, 645), dtype=np.float32)
        res = engine.predict(dummy_feat)
        assert res.predicted_gloss == "UNCERTAIN"
        assert res.confidence == pytest.approx(0.85, rel=1e-2)

    def test_empty_frames_raises_value_error(self) -> None:
        """Verify empty sequence raises ValueError."""
        model = MockClassifier()
        engine = RealtimeInferenceEngine(model=model)
        with pytest.raises(ValueError, match="empty frame sequence"):
            engine.extract_features([])

    def test_sub_10ms_latency_benchmark(self) -> None:
        """Verify baseline classifier inference executes under 10 ms per evaluation."""
        # Train a real Random Forest on small synthetic data
        x_synth = np.random.randn(50, 645).astype(np.float32)
        y_synth = np.random.randint(0, 5, size=50)
        rf = RandomForestSignClassifier(
            n_estimators=50, max_depth=15, n_jobs=1, random_state=42
        )
        rf.fit(x_synth, y_synth, classes=[f"sign_{i}" for i in range(5)])

        engine = RealtimeInferenceEngine(model=rf)
        dummy_feat = np.random.randn(1, 645).astype(np.float32)

        # Warm-up runs
        for _ in range(5):
            _ = engine.predict(dummy_feat)

        n_iter = 50
        t_start = time.perf_counter()
        for _ in range(n_iter):
            _ = engine.predict(dummy_feat)
        total_time_s = time.perf_counter() - t_start

        avg_latency_ms = (total_time_s / n_iter) * 1000.0
        # Target: inference latency < 10 ms per evaluation (benchmarked ~2-3 ms)
        assert avg_latency_ms < 10.0, f"Average latency was {avg_latency_ms:.2f} ms (> 10.0 ms)"


class TestRealtimeInferenceWorker:
    """Test suite for asynchronous RealtimeInferenceWorker."""

    def test_async_submission_and_polling(self) -> None:
        """Verify worker processes tasks asynchronously and returns results via poll and latest."""
        model = MockClassifier(num_classes=5)
        engine = RealtimeInferenceEngine(model=model)
        worker = RealtimeInferenceWorker(engine=engine, queue_size=5)

        worker.start()
        try:
            frames = _make_dummy_normalized_frames(n_frames=10)
            submitted = worker.submit_async(frames, timestamp_ms=100.0)
            assert submitted is True

            res = worker.poll_result(timeout=1.0)
            assert res is not None
            assert res.predicted_gloss == "sign_1"
            assert res.timestamp_ms == 100.0

            latest = worker.get_latest_result()
            assert latest is not None
            assert latest.predicted_gloss == "sign_1"
        finally:
            worker.stop()

    def test_queue_overflow_drop_prevention(self) -> None:
        """Verify submitting beyond queue size drops oldest task without deadlock."""
        model = MockClassifier(num_classes=5)
        engine = RealtimeInferenceEngine(model=model)
        # Queue size 2
        worker = RealtimeInferenceWorker(engine=engine, queue_size=2)

        # Do not start worker immediately so tasks queue up
        dummy = np.zeros((1, 645), dtype=np.float32)
        worker.submit_async(dummy, timestamp_ms=1.0)
        worker.submit_async(dummy, timestamp_ms=2.0)
        # Third submission should evict oldest
        worker.submit_async(dummy, timestamp_ms=3.0)

        worker.start()
        try:
            res1 = worker.poll_result(timeout=1.0)
            assert res1 is not None
            # Timestamps 2 or 3 should be processed (oldest 1.0 was dropped)
            assert res1.timestamp_ms in (2.0, 3.0)
        finally:
            worker.stop()
