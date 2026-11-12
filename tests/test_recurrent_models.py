"""Unit tests for recurrent neural sequence models (SANKET-064)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from sanket.models.recurrent import (
    RecurrentModelConfig,
    RecurrentSignClassifier,
)


def _generate_synthetic_sequences(
    num_samples: int = 16,
    seq_len: int = 30,
    feat_dim: int = 126,
    num_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper generating deterministic synthetic sequence batches."""
    rng = np.random.RandomState(42)
    x = rng.randn(num_samples, seq_len, feat_dim).astype(np.float32)
    y = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    return x, y


class TestRecurrentSignClassifier:
    """Test suite validating GRU and LSTM sequence classifiers."""

    def test_recurrent_gru_bidirectional_shapes(self) -> None:
        cfg = RecurrentModelConfig(
            cell_type="gru",
            input_dim=64,
            hidden_dim=32,
            bidirectional=True,
            num_classes=5,
        )
        model = RecurrentSignClassifier(config=cfg)

        assert model.num_parameters < 500_000
        x, _ = _generate_synthetic_sequences(num_samples=4, seq_len=20, feat_dim=64, num_classes=5)

        logits = model.forward(x)
        assert logits.shape == (4, 5)

        probas = model.predict_proba(x)
        assert probas.shape == (4, 5)
        # Softmax rows must sum to 1.0
        assert np.allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

        preds = model.predict(x)
        assert preds.shape == (4,)
        assert np.all((preds >= 0) & (preds < 5))

    def test_recurrent_lstm_unidirectional_shapes(self) -> None:
        cfg = RecurrentModelConfig(
            cell_type="lstm",
            input_dim=48,
            hidden_dim=24,
            bidirectional=False,
            num_classes=3,
        )
        model = RecurrentSignClassifier(config=cfg)

        x, _ = _generate_synthetic_sequences(num_samples=2, seq_len=15, feat_dim=48, num_classes=3)
        logits = model.forward(x)
        assert logits.shape == (2, 3)

        preds = model.predict(x)
        assert preds.shape == (2,)

    def test_single_sequence_inference(self) -> None:
        cfg = RecurrentModelConfig(cell_type="gru", input_dim=32, hidden_dim=16, num_classes=2)
        model = RecurrentSignClassifier(config=cfg)

        # Single sequence (T, D)
        rng = np.random.RandomState(42)
        single_x = rng.randn(25, 32).astype(np.float32)

        pred = model.predict(single_x)
        assert isinstance(pred, (int, np.integer))

    def test_recurrent_fit_and_loss_decrease(self) -> None:
        cfg = RecurrentModelConfig(
            cell_type="gru",
            input_dim=32,
            hidden_dim=16,
            num_classes=2,
            seed=42,
        )
        model = RecurrentSignClassifier(config=cfg)

        x, y = _generate_synthetic_sequences(num_samples=20, seq_len=15, feat_dim=32, num_classes=2)

        # Initial cross entropy
        probs_init = model.predict_proba(x)
        loss_init = -np.mean(np.log(np.clip(probs_init[np.arange(len(y)), y], 1e-7, 1.0)))

        # Train 15 epochs
        model.fit(x, y, epochs=15, lr=0.05, batch_size=5)
        assert model.is_fitted is True

        probs_final = model.predict_proba(x)
        loss_final = -np.mean(np.log(np.clip(probs_final[np.arange(len(y)), y], 1e-7, 1.0)))

        # Loss should decrease
        assert loss_final <= loss_init

    def test_forward_latency_benchmark(self) -> None:
        """Verify CPU forward pass latency satisfies < 15 ms target."""
        cfg = RecurrentModelConfig(
            cell_type="gru",
            input_dim=126,
            hidden_dim=64,
            bidirectional=True,
            num_classes=16,
        )
        model = RecurrentSignClassifier(config=cfg)

        # Sequence of length 30 at feature dim 126
        x, _ = _generate_synthetic_sequences(
            num_samples=1, seq_len=30, feat_dim=126, num_classes=16
        )

        # Warm up
        for _ in range(3):
            model.forward(x)

        # Benchmark 10 iterations
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            model.forward(x)
            times.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(times))
        # Ensure fast CPU inference (< 15 ms target)
        assert mean_ms < 15.0

    def test_model_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        cfg = RecurrentModelConfig(cell_type="gru", input_dim=32, hidden_dim=16, num_classes=3)
        classes = ["HELLO", "THANK_YOU", "YES"]
        model = RecurrentSignClassifier(config=cfg, classes=classes)

        x, _ = _generate_synthetic_sequences(num_samples=2, seq_len=10, feat_dim=32, num_classes=3)
        orig_probs = model.predict_proba(x)

        model_file = tmp_path / "test_recurrent.joblib"
        model.save(model_file)
        assert model_file.exists()

        loaded_model = RecurrentSignClassifier.load(model_file)
        assert loaded_model.classes_ == classes
        assert loaded_model.config.cell_type == "gru"
        loaded_probs = loaded_model.predict_proba(x)

        assert np.allclose(orig_probs, loaded_probs, atol=1e-6)

    def test_invalid_input_shapes_raise_error(self) -> None:
        cfg = RecurrentModelConfig(input_dim=32)
        model = RecurrentSignClassifier(config=cfg)

        with pytest.raises(ValueError, match="Expected 3D input"):
            model.forward(np.zeros((2, 10, 32, 1)))

        with pytest.raises(ValueError, match="does not match model input_dim"):
            model.forward(np.zeros((2, 10, 64)))
