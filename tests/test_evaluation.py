"""Unit tests for the unified model evaluation pipeline (SANKET-031).

Tests metric computations, runtime latency profiling, manifest generation,
SignModelEvaluator integration with BaseSignClassifier, and evaluate_model CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sanket.evaluation import (
    EvaluationReport,
    ExperimentManifest,
    SignModelEvaluator,
    compute_data_checksum,
    compute_metrics,
    get_git_provenance,
    get_sanitized_environment_info,
    lock_seed,
    profile_latency,
)
from sanket.models import RandomForestSignClassifier
from scripts.evaluate_model import (
    generate_synthetic_test_data,
    load_model_from_file,
)
from scripts.evaluate_model import (
    main as cli_main,
)


class TestMetrics:
    """Test classification metrics computation and container methods."""

    def test_compute_metrics_perfect_predictions(self) -> None:
        """Verify metric outputs when predictions perfectly match ground truth."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        classes = ["A", "B", "C"]

        metrics = compute_metrics(y_true, y_pred, classes=classes)

        assert metrics.accuracy == 1.0
        assert metrics.macro_precision == 1.0
        assert metrics.macro_recall == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.weighted_f1 == 1.0
        assert metrics.matthews_corrcoef == 1.0
        assert metrics.num_samples == 6
        assert metrics.num_classes == 3
        assert metrics.classes == classes

        # Check per-class metrics
        for c in classes:
            assert metrics.per_class[c]["f1"] == 1.0
            assert metrics.per_class[c]["support"] == 2

        # Check normalized confusion matrix diagonal
        norm_cm = np.array(metrics.normalized_confusion_matrix)
        assert np.allclose(norm_cm, np.eye(3))

    def test_compute_metrics_with_probabilities_top_k(self) -> None:
        """Verify Top-3 and Top-5 accuracy calculation from predicted probabilities."""
        # 6 classes
        y_true = np.array([0, 1, 2, 3, 4, 5])
        # y_pred gets first 3 correct, last 3 wrong
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        # Probability distribution where true class is in top-3
        y_proba = np.zeros((6, 6), dtype=float)
        # Class 0-2: rank 1
        y_proba[0, 0] = 0.9
        y_proba[1, 1] = 0.9
        y_proba[2, 2] = 0.9
        # Class 3: predicted 0 (prob 0.6), but true class 3 has prob 0.3 (rank 2)
        y_proba[3, 0] = 0.6
        y_proba[3, 3] = 0.3
        # Class 4: predicted 1 (prob 0.6), but true class 4 has prob 0.3 (rank 2)
        y_proba[4, 1] = 0.6
        y_proba[4, 4] = 0.3
        # Class 5: true class 5 has 0 prob, top 5 are classes 2, 0, 1, 3, 4
        y_proba[5, 2] = 0.4
        y_proba[5, 0] = 0.2
        y_proba[5, 1] = 0.2
        y_proba[5, 3] = 0.1
        y_proba[5, 4] = 0.1

        classes = [f"sign_{i}" for i in range(6)]
        metrics = compute_metrics(y_true, y_pred, y_proba=y_proba, classes=classes)

        assert metrics.accuracy == pytest.approx(3 / 6)
        assert metrics.top_3_accuracy == pytest.approx(5 / 6)
        assert metrics.top_5_accuracy == pytest.approx(5 / 6)

    def test_compute_metrics_top_k_fallback(self) -> None:
        """Verify Top-k falls back gracefully when num_classes < k."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])
        y_proba = np.array([[0.8, 0.2], [0.6, 0.4]])

        metrics = compute_metrics(y_true, y_pred, y_proba=y_proba, classes=["A", "B"])
        assert metrics.top_3_accuracy == metrics.accuracy
        assert metrics.top_5_accuracy == metrics.accuracy

    def test_compute_metrics_empty_raises(self) -> None:
        """Verify ValueError is raised on empty evaluation inputs."""
        with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
            compute_metrics([], [])

    def test_compute_metrics_length_mismatch_raises(self) -> None:
        """Verify ValueError is raised when prediction and target lengths differ."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_metrics([0, 1], [0])

    def test_compute_metrics_auto_classes(self) -> None:
        """Verify automatic class identification when classes argument is None."""
        y_true = np.array([0, 2])
        y_pred = np.array([0, 2])
        metrics = compute_metrics(y_true, y_pred, classes=None)

        assert metrics.num_classes == 3
        assert "class_0" in metrics.per_class
        assert "class_2" in metrics.per_class

    def test_metrics_to_dict_serializability(self) -> None:
        """Verify metrics serialize to valid JSON dictionary."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_metrics(y_true, y_pred, classes=["A", "B"])
        data = metrics.to_dict()

        assert isinstance(data, dict)
        serialized = json.dumps(data)
        assert "accuracy" in serialized
        assert "normalized_confusion_matrix" in serialized


class TestLatency:
    """Test operational latency and throughput benchmarking."""

    def test_profile_latency_success(self) -> None:
        """Verify latency profiling calculates expected percentiles and throughput."""
        call_count = 0

        def dummy_predict(x: np.ndarray) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.array([0])

        sample = np.ones((1, 645), dtype=np.float32)
        metrics = profile_latency(
            predict_fn=dummy_predict,
            sample=sample,
            num_trials=50,
            warm_up=5,
            track_memory=True,
        )

        assert call_count == 55
        assert metrics.num_trials == 50
        assert metrics.min_ms <= metrics.median_ms <= metrics.max_ms
        assert metrics.median_ms <= metrics.p95_ms <= metrics.p99_ms
        assert metrics.throughput_fps > 0
        assert metrics.peak_memory_mb is not None

        d = metrics.to_dict()
        assert "median_ms" in d
        assert "throughput_fps" in d
        assert "peak_memory_mb" in d

    def test_profile_latency_1d_sample(self) -> None:
        """Verify 1D sample input is reshaped to 2D automatically."""
        def dummy_predict(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 2
            return np.array([0])

        sample_1d = np.zeros(645, dtype=np.float32)
        metrics = profile_latency(dummy_predict, sample_1d, num_trials=10, warm_up=2)
        assert metrics.num_trials == 10

    def test_profile_latency_invalid_trials_raises(self) -> None:
        """Verify ValueError is raised if num_trials < 1."""
        with pytest.raises(ValueError, match="num_trials must be at least 1"):
            profile_latency(lambda x: x, np.array([1.0]), num_trials=0)

    def test_profile_latency_empty_sample_raises(self) -> None:
        """Verify ValueError is raised if sample array is empty."""
        with pytest.raises(ValueError, match="Sample array cannot be empty"):
            profile_latency(lambda x: x, np.array([]), num_trials=10)


class TestManifest:
    """Test structured experiment manifest creation, Git provenance, and serialization."""

    def test_get_git_provenance(self) -> None:
        """Verify Git provenance returns a commit hash and boolean dirty flag."""
        sha, is_dirty = get_git_provenance()
        assert isinstance(sha, str)
        assert len(sha) > 0
        assert isinstance(is_dirty, bool)

    def test_compute_data_checksum(self, tmp_path: Path) -> None:
        """Verify SHA-256 data checksum across array, bytes, and file inputs."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        chk_arr = compute_data_checksum(arr)
        assert len(chk_arr) == 64

        raw_bytes = b"sanket-test-bytes"
        chk_bytes = compute_data_checksum(raw_bytes)
        assert len(chk_bytes) == 64

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(raw_bytes)
        chk_file = compute_data_checksum(test_file)
        assert chk_file == chk_bytes

    def test_get_sanitized_environment_info(self) -> None:
        """Verify environment metadata contains essential version info without private paths."""
        env = get_sanitized_environment_info()
        assert "python_version" in env
        assert "numpy_version" in env
        assert "scikit_learn_version" in env
        assert "platform_system" in env

        # Ensure no personal Windows directory paths are present
        env_str = json.dumps(env)
        assert "C:\\Users\\" not in env_str
        assert "C:/Users/" not in env_str

    def test_manifest_roundtrip_save_and_load(self, tmp_path: Path) -> None:
        """Verify ExperimentManifest serializes and deserializes accurately."""
        manifest = ExperimentManifest.create(
            experiment_id="EXP-TEST-001",
            active_vocabulary=["HELLO", "WATER"],
            metrics={"accuracy": 0.95, "macro_f1": 0.94},
            runtime={"median_ms": 2.5, "throughput_fps": 400.0},
            dataset_checksum="abc123sha",
            signer_partitions={"train": ["S1"], "test": ["S2"]},
            model_metadata={"model_class": "RandomForestSignClassifier"},
        )

        manifest_path = tmp_path / "manifest.json"
        saved_path = manifest.save(manifest_path)
        assert saved_path.exists()

        loaded = ExperimentManifest.load(saved_path)
        assert loaded.experiment_id == "EXP-TEST-001"
        assert loaded.active_vocabulary == ["HELLO", "WATER"]
        assert loaded.metrics["accuracy"] == 0.95
        assert loaded.runtime["median_ms"] == 2.5
        assert loaded.signer_partitions["test"] == ["S2"]
        assert loaded.model_metadata["model_class"] == "RandomForestSignClassifier"


class TestEvaluator:
    """Test SignModelEvaluator integration with BaseSignClassifier implementations."""

    @pytest.fixture
    def fitted_rf(self) -> tuple[RandomForestSignClassifier, np.ndarray, np.ndarray, list[str]]:
        """Fixture providing a fitted RandomForestSignClassifier and test split."""
        classes = ["HELLO", "WATER", "FOOD", "HELP"]
        x_train, y_train, _ = generate_synthetic_test_data(
            num_classes=4, num_features=64, samples_per_class=10, seed=42
        )
        x_test, y_test, _ = generate_synthetic_test_data(
            num_classes=4, num_features=64, samples_per_class=5, seed=100
        )

        model = RandomForestSignClassifier(n_estimators=20, random_state=42)
        model.fit(x_train, y_train, classes=classes)
        return model, x_test, y_test, classes

    def test_evaluator_success(
        self,
        fitted_rf: tuple[RandomForestSignClassifier, np.ndarray, np.ndarray, list[str]],
        tmp_path: Path,
    ) -> None:
        """Verify end-to-end evaluation, reporting, and manifest output."""
        model, x_test, y_test, classes = fitted_rf
        evaluator = SignModelEvaluator(seed=42, latency_trials=20, warm_up_trials=5)

        report = evaluator.evaluate(
            model=model,
            x_test=x_test,
            y_test=y_test,
            classes=classes,
            experiment_id="EXP-UNIT-TEST-RF",
            save_manifest_dir=tmp_path,
        )

        assert isinstance(report, EvaluationReport)
        assert report.model_name == "RandomForestSignClassifier"
        assert report.metrics.num_samples == len(x_test)
        assert report.metrics.num_classes == 4
        assert report.latency.median_ms > 0
        assert report.latency.throughput_fps > 0

        # Manifest written to disk
        expected_manifest = tmp_path / "EXP-UNIT-TEST-RF.json"
        assert expected_manifest.exists()

        # Summary text formatting
        summary = report.summary()
        assert "SANKET EVALUATION REPORT" in summary
        assert "Top-1 Accuracy" in summary
        assert "Macro F1-Score" in summary
        assert "OPERATIONAL RUNTIME BENCHMARKS" in summary

    def test_evaluator_unfitted_model_raises(self) -> None:
        """Verify evaluating unfitted model raises RuntimeError."""
        model = RandomForestSignClassifier()
        evaluator = SignModelEvaluator()

        with pytest.raises(RuntimeError, match="Cannot evaluate unfitted model"):
            evaluator.evaluate(model, np.ones((5, 10)), np.array([0, 1, 0, 1, 0]))

    def test_evaluator_empty_dataset_raises(
        self, fitted_rf: tuple[RandomForestSignClassifier, np.ndarray, np.ndarray, list[str]]
    ) -> None:
        """Verify evaluating with empty feature dataset raises ValueError."""
        model, _, _, _ = fitted_rf
        evaluator = SignModelEvaluator()

        with pytest.raises(ValueError, match="Evaluation feature dataset is empty"):
            evaluator.evaluate(model, np.empty((0, 64)), np.empty(0))

    def test_evaluator_length_mismatch_raises(
        self, fitted_rf: tuple[RandomForestSignClassifier, np.ndarray, np.ndarray, list[str]]
    ) -> None:
        """Verify evaluating with mismatched x and y lengths raises ValueError."""
        model, x_test, _, _ = fitted_rf
        evaluator = SignModelEvaluator()

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate(model, x_test, np.array([0, 1]))

    def test_lock_seed_determinism(self) -> None:
        """Verify lock_seed enforces reproducible random generations."""
        lock_seed(123)
        v1 = np.random.rand(5)
        lock_seed(123)
        v2 = np.random.rand(5)
        assert np.allclose(v1, v2)


class TestEvaluateCLI:
    """Test CLI execution and argument parsing in scripts/evaluate_model.py."""

    def test_cli_synthetic_demo_mode(self, tmp_path: Path) -> None:
        """Verify CLI executes successfully in synthetic demonstration mode."""
        ret = cli_main(
            [
                "--synthetic-demo",
                "--latency-trials",
                "15",
                "--manifest-dir",
                str(tmp_path),
                "--experiment-id",
                "EXP-CLI-TEST",
            ]
        )
        assert ret == 0
        manifest_file = tmp_path / "EXP-CLI-TEST.json"
        assert manifest_file.exists()

    def test_cli_with_saved_model_and_npz(self, tmp_path: Path) -> None:
        """Verify CLI evaluates a saved model artifact against an NPZ dataset."""
        classes = ["HELLO", "THANK_YOU"]
        x_train, y_train, _ = generate_synthetic_test_data(
            num_classes=2, num_features=32, samples_per_class=10, seed=42
        )
        x_test, y_test, _ = generate_synthetic_test_data(
            num_classes=2, num_features=32, samples_per_class=5, seed=100
        )

        model = RandomForestSignClassifier(n_estimators=10, random_state=42)
        model.fit(x_train, y_train, classes=classes)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)

        data_path = tmp_path / "test_data.npz"
        np.savez_compressed(data_path, x_test=x_test, y_test=y_test, classes=classes)

        ret = cli_main(
            [
                "--model-path",
                str(model_path),
                "--data-path",
                str(data_path),
                "--latency-trials",
                "10",
                "--manifest-dir",
                str(tmp_path),
                "--experiment-id",
                "EXP-NPZ-TEST",
            ]
        )
        assert ret == 0
        manifest_file = tmp_path / "EXP-NPZ-TEST.json"
        assert manifest_file.exists()

    def test_cli_missing_data_path_fails(self, tmp_path: Path) -> None:
        """Verify CLI exits with non-zero code when data path is missing in non-demo mode."""
        model_path = tmp_path / "dummy.joblib"
        model_path.write_bytes(b"dummy")

        ret = cli_main(["--model-path", str(model_path)])
        assert ret == 1

    def test_load_model_not_found(self, tmp_path: Path) -> None:
        """Verify load_model_from_file raises FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_model_from_file(tmp_path / "non_existent.joblib")

    def test_load_model_invalid_type(self, tmp_path: Path) -> None:
        """Verify load_model_from_file raises TypeError for non-BaseSignClassifier objects."""
        import joblib

        bad_file = tmp_path / "invalid.joblib"
        joblib.dump({"not": "a model"}, bad_file)

        with pytest.raises(TypeError, match="does not implement BaseSignClassifier"):
            load_model_from_file(bad_file)
