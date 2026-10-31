"""Unit tests for Random Forest baseline classifier and training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sanket.models.base import BaseSignClassifier, SignModelMetrics
from sanket.models.random_forest import RandomForestSignClassifier
from scripts.train_random_forest import (
    generate_synthetic_benchmark_dataset,
    main,
    train_and_evaluate_rf,
)


class DummyClassifier(BaseSignClassifier):
    """Minimal concrete subclass for testing BaseSignClassifier ABC contract."""

    def __init__(self) -> None:
        super().__init__()
        self.is_fitted_ = True

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> DummyClassifier:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 2)) * 0.5

    def save(self, path: Path | str) -> None:
        pass

    @classmethod
    def load(cls, path: Path | str) -> DummyClassifier:
        return cls()


def test_sign_model_metrics_to_dict() -> None:
    """Verify SignModelMetrics fields and dictionary serialization."""
    metrics = SignModelMetrics(
        accuracy=0.92,
        top_3_accuracy=0.98,
        macro_precision=0.90,
        macro_recall=0.91,
        macro_f1=0.91,
        weighted_f1=0.92,
        num_samples=22,
        inference_latency_ms=1.25,
        per_class_f1={"A": 0.90, "B": 0.94},
        confusion_matrix=[[10, 1], [0, 11]],
        classes=["A", "B"],
    )
    d = metrics.to_dict()
    assert d["accuracy"] == 0.92
    assert d["macro_f1"] == 0.91
    assert d["top_3_accuracy"] == 0.98
    assert d["classes"] == ["A", "B"]
    assert d["confusion_matrix"] == [[10, 1], [0, 11]]


def test_base_classifier_abstract_contract() -> None:
    """Verify BaseSignClassifier cannot be instantiated directly without implementations."""
    with pytest.raises(TypeError):
        BaseSignClassifier()  # type: ignore[abstract]

    dummy = DummyClassifier()
    assert dummy.is_fitted is True


def test_random_forest_initial_state() -> None:
    """Verify initial unfitted state and guard rails."""
    clf = RandomForestSignClassifier(n_estimators=50, random_state=42)
    assert clf.is_fitted is False
    assert clf.estimator_ is None

    x_dummy = np.zeros((5, 10))
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(x_dummy)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(x_dummy)

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.get_top_features()

    params = clf.get_params()
    assert params["n_estimators"] == 50
    assert params["random_state"] == 42


def test_random_forest_fit_predict() -> None:
    """Verify training, inference shapes, and probability distributions."""
    rng = np.random.default_rng(42)
    n_samples, n_features, n_classes = 100, 20, 4
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    classes = [f"sign_{i}" for i in range(n_classes)]
    y = rng.choice(classes, size=n_samples)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    clf = RandomForestSignClassifier(n_estimators=20, random_state=42)
    clf.fit(x, y, classes=classes, feature_names=feature_names)

    assert clf.is_fitted is True
    assert clf.classes == classes
    assert clf.feature_names == feature_names

    # Test predict
    y_pred = clf.predict(x)
    assert y_pred.shape == (n_samples,)
    assert set(np.unique(y_pred)).issubset(set(classes))

    # Test predict_proba
    probas = clf.predict_proba(x)
    assert probas.shape == (n_samples, n_classes)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)


def test_random_forest_evaluate() -> None:
    """Verify evaluation metric computation against test split."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 20)).astype(np.float32)
    y = np.repeat(["hello", "thank_you", "please", "yes"], 25)

    clf = RandomForestSignClassifier(n_estimators=25, random_state=42)
    clf.fit(x, y)

    metrics = clf.evaluate(x, y)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.top_3_accuracy <= 1.0
    assert metrics.inference_latency_ms >= 0.0
    assert len(metrics.confusion_matrix) == len(clf.classes)
    assert len(metrics.per_class_f1) == len(clf.classes)


def test_random_forest_feature_importances() -> None:
    """Verify feature importance extraction and ordering."""
    rng = np.random.default_rng(42)
    n_features = 15
    x = rng.standard_normal((80, n_features)).astype(np.float32)
    # Make feature 3 strongly predictive
    y = (x[:, 3] > 0).astype(int)
    feature_names = [f"dim_{i}" for i in range(n_features)]

    clf = RandomForestSignClassifier(n_estimators=30, random_state=42)
    clf.fit(x, y, feature_names=feature_names)

    top_features = clf.get_top_features(k=5)
    assert len(top_features) == 5
    # Most important feature should have highest score
    assert top_features[0][1] >= top_features[1][1]
    # Feature 3 should rank in the top features
    top_feature_names = [feat[0] for feat in top_features]
    assert "dim_3" in top_feature_names


def test_random_forest_save_and_load(tmp_path: Path) -> None:
    """Verify serialization and identical inference on reload."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((60, 10)).astype(np.float32)
    y = rng.choice(["sign_a", "sign_b", "sign_c"], size=60)

    clf = RandomForestSignClassifier(n_estimators=15, random_state=42)
    clf.fit(x, y)
    preds_before = clf.predict(x)
    probs_before = clf.predict_proba(x)

    model_path = tmp_path / "rf_model.joblib"
    clf.save(model_path)
    assert model_path.exists()

    loaded_clf = RandomForestSignClassifier.load(model_path)
    assert loaded_clf.is_fitted is True
    assert loaded_clf.classes == clf.classes
    assert loaded_clf.feature_names == clf.feature_names

    preds_after = loaded_clf.predict(x)
    probs_after = loaded_clf.predict_proba(x)
    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(probs_before, probs_after, atol=1e-5)


def test_random_forest_save_unfitted_raises(tmp_path: Path) -> None:
    """Verify error when saving unfitted classifier."""
    clf = RandomForestSignClassifier()
    with pytest.raises(RuntimeError, match="unfitted model"):
        clf.save(tmp_path / "bad.joblib")


def test_random_forest_tune_hyperparameters() -> None:
    """Verify validation search explores param grid and selects best estimator."""
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((80, 12)).astype(np.float32)
    y_train = rng.choice(["c1", "c2"], size=80)
    x_val = rng.standard_normal((30, 12)).astype(np.float32)
    y_val = rng.choice(["c1", "c2"], size=30)

    clf = RandomForestSignClassifier(random_state=42)
    param_grid: dict[str, list[Any]] = {
        "n_estimators": [10, 20],
        "max_depth": [3, 5],
    }

    res = clf.tune_hyperparameters(
        x_train,
        y_train,
        x_val,
        y_val,
        param_grid=param_grid,
    )

    best_params = res["best_params"]
    trials = res["trials"]
    assert len(trials) == 4
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert clf.is_fitted is True


def test_generate_synthetic_benchmark_dataset() -> None:
    """Verify synthetic dataset generator produces correct splits and shapes."""
    x_train, y_train, x_val, y_val, x_test, y_test, classes = (
        generate_synthetic_benchmark_dataset(
            num_classes=5,
            num_features=50,
            random_seed=42,
        )
    )
    assert x_train.shape == (15 * 5, 50)
    assert y_train.shape == (15 * 5,)
    assert x_val.shape == (4 * 5, 50)
    assert x_test.shape == (4 * 5, 50)
    assert len(classes) == 5


def test_train_and_evaluate_rf_with_features_dir(tmp_path: Path) -> None:
    """Verify training pipeline when loading pre-extracted features from directory."""
    feat_dir = tmp_path / "features"
    feat_dir.mkdir(parents=True)
    out_dir = tmp_path / "model_out"

    rng = np.random.default_rng(42)
    n_features = 30
    x_train = rng.standard_normal((40, n_features)).astype(np.float32)
    y_train = rng.choice(["HELLO", "YES"], size=40)
    x_val = rng.standard_normal((10, n_features)).astype(np.float32)
    y_val = rng.choice(["HELLO", "YES"], size=10)
    x_test = rng.standard_normal((10, n_features)).astype(np.float32)
    y_test = rng.choice(["HELLO", "YES"], size=10)

    np.save(feat_dir / "X_train.npy", x_train)
    np.save(feat_dir / "y_train.npy", y_train)
    np.save(feat_dir / "X_val.npy", x_val)
    np.save(feat_dir / "y_val.npy", y_val)
    np.save(feat_dir / "X_test.npy", x_test)
    np.save(feat_dir / "y_test.npy", y_test)

    with open(feat_dir / "class_to_index.json", "w", encoding="utf-8") as f:
        json.dump({"HELLO": 0, "YES": 1}, f)

    with open(feat_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump([f"dim_{i}" for i in range(n_features)], f)

    report = train_and_evaluate_rf(
        features_dir=feat_dir,
        output_dir=out_dir,
        tune=True,
        n_estimators=10,
        max_depth=4,
    )

    assert "train_metrics" in report
    assert "val_metrics" in report
    assert "test_metrics" in report
    assert (out_dir / "random_forest_baseline.joblib").exists()
    assert (out_dir / "rf_feature_importances.json").exists()
    assert (out_dir / "rf_metrics.json").exists()


def test_train_random_forest_cli_synthetic_demo(tmp_path: Path) -> None:
    """Verify full end-to-end training CLI execution with --synthetic-demo."""
    output_dir = tmp_path / "models" / "rf_test"

    args = [
        "--synthetic-demo",
        "--output-dir",
        str(output_dir),
        "--n-estimators",
        "15",
        "--max-depth",
        "4",
    ]

    ret = main(args)
    assert ret == 0

    assert (output_dir / "random_forest_baseline.joblib").exists()
    assert (output_dir / "rf_feature_importances.json").exists()
    assert (output_dir / "rf_metrics.json").exists()

    with open(output_dir / "rf_metrics.json", encoding="utf-8") as f:
        report = json.load(f)
        assert report["model"] == "RandomForestSignClassifier"
        assert "train_metrics" in report
        assert "test_metrics" in report
        assert report["test_metrics"]["accuracy"] >= 0.0
        assert "train_metrics" in report
        assert "test_metrics" in report
        assert report["test_metrics"]["accuracy"] >= 0.0
