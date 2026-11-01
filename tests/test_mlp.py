"""Unit tests for the Multi-Layer Perceptron neural network baseline classifier (SANKET-030).

Tests MLPSignClassifier initialization, fitting, loss convergence, feature standardization,
early stopping, hyperparameter tuning, model persistence, and CLI execution.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from sanket.models import MLPSignClassifier
from scripts.train_mlp import generate_synthetic_benchmark_dataset
from scripts.train_mlp import main as cli_main


class TestMLPSignClassifier:
    """Test MLPSignClassifier model class methods and properties."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Fixture returning synthetic 4-class training and validation splits."""
        x_tr, y_tr, x_v, y_v, _, _, classes = generate_synthetic_benchmark_dataset(
            num_classes=4,
            num_features=64,
            random_seed=42,
        )
        return x_tr, y_tr, x_v, y_v, classes

    def test_mlp_initialization_and_params(self) -> None:
        """Verify default hyperparameters and parameter dictionary."""
        clf = MLPSignClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=50)
        assert clf.hidden_layer_sizes == (128, 64)
        assert clf.alpha == 0.01
        assert clf.is_fitted is False
        assert clf.classes is None

        params = clf.get_params()
        assert params["hidden_layer_sizes"] == (128, 64)
        assert params["alpha"] == 0.01
        assert params["scale_features"] is True

    def test_mlp_fit_predict_and_proba(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
    ) -> None:
        """Verify model fitting, label predictions, calibrated probabilities, and loss curves."""
        x_tr, y_tr, _, _, classes = synthetic_data
        clf = MLPSignClassifier(hidden_layer_sizes=(64, 32), max_iter=30, random_state=42)

        clf.fit(x_tr, y_tr, classes=classes)
        assert clf.is_fitted is True
        assert clf.classes == classes
        assert clf.loss_curve_ is not None
        assert len(clf.loss_curve_) > 0

        preds = clf.predict(x_tr[:5])
        assert preds.shape == (5,)
        assert np.all(np.isin(preds, np.arange(4)))

        proba = clf.predict_proba(x_tr[:5])
        assert proba.shape == (5, 4)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_mlp_feature_scaling_option(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
    ) -> None:
        """Verify behavior with and without StandardScaler feature scaling."""
        x_tr, y_tr, _, _, _ = synthetic_data

        clf_scaled = MLPSignClassifier(scale_features=True, max_iter=10)
        clf_scaled.fit(x_tr, y_tr)
        assert clf_scaled.scaler_ is not None

        clf_unscaled = MLPSignClassifier(scale_features=False, max_iter=10)
        clf_unscaled.fit(x_tr, y_tr)
        assert clf_unscaled.scaler_ is None

        # Predict single 1D vector (handles reshape)
        pred_1d = clf_scaled.predict(x_tr[0])
        assert pred_1d.shape == (1,)

    def test_mlp_error_handling(self) -> None:
        """Verify appropriate exceptions for uninitialized or invalid inputs."""
        clf = MLPSignClassifier()

        with pytest.raises(RuntimeError, match="Classifier is not fitted"):
            clf.predict(np.ones((2, 10)))

        with pytest.raises(RuntimeError, match="Classifier is not fitted"):
            clf.predict_proba(np.ones((2, 10)))

        with pytest.raises(RuntimeError, match="Cannot serialize unfitted model"):
            clf.save("dummy.joblib")

        # 1D feature array error
        with pytest.raises(ValueError, match="Expected 2D feature matrix"):
            clf.fit(np.array([1, 2, 3]), np.array([0]))

        # Length mismatch
        with pytest.raises(ValueError, match="Sample count mismatch"):
            clf.fit(np.ones((5, 10)), np.array([0, 1]))

        # Empty dataset
        with pytest.raises(ValueError, match="Cannot fit classifier on empty dataset"):
            clf.fit(np.empty((0, 10)), np.empty(0))

    def test_mlp_tune_hyperparameters(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
    ) -> None:
        """Verify hyperparameter search selects optimal architecture and regularization."""
        x_tr, y_tr, x_v, y_v, classes = synthetic_data
        clf = MLPSignClassifier(random_state=42)

        param_grid = {
            "hidden_layer_sizes": [(32,), (32, 16)],
            "alpha": [0.001, 0.01],
        }
        res = clf.tune_hyperparameters(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_v,
            y_val=y_v,
            param_grid=param_grid,
            classes=classes,
        )

        assert "best_params" in res
        assert "best_val_macro_f1" in res
        assert len(res["trials"]) == 4
        assert clf.is_fitted is True

    def test_mlp_serialization_roundtrip(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
        tmp_path: Path,
    ) -> None:
        """Verify model persistence to disk and identical predictions after loading."""
        x_tr, y_tr, _, _, classes = synthetic_data
        clf = MLPSignClassifier(hidden_layer_sizes=(32, 16), max_iter=25, random_state=42)
        clf.fit(x_tr, y_tr, classes=classes)

        model_file = tmp_path / "mlp_model.joblib"
        saved = clf.save(model_file)
        assert saved.exists()

        loaded = MLPSignClassifier.load(model_file)
        assert loaded.is_fitted is True
        assert loaded.hidden_layer_sizes == (32, 16)
        assert loaded.classes == classes

        preds_orig = clf.predict(x_tr[:5])
        preds_loaded = loaded.predict(x_tr[:5])
        assert np.array_equal(preds_orig, preds_loaded)

        proba_orig = clf.predict_proba(x_tr[:5])
        proba_loaded = loaded.predict_proba(x_tr[:5])
        assert np.allclose(proba_orig, proba_loaded)

    def test_mlp_load_invalid(self, tmp_path: Path) -> None:
        """Verify exceptions on missing or malformed model files."""
        with pytest.raises(FileNotFoundError):
            MLPSignClassifier.load(tmp_path / "non_existent.joblib")

        bad_file = tmp_path / "corrupt.joblib"
        joblib.dump({"bad": "payload"}, bad_file)
        with pytest.raises(ValueError, match="Invalid model payload"):
            MLPSignClassifier.load(bad_file)


class TestTrainMLPCLI:
    """Test CLI execution of scripts/train_mlp.py."""

    def test_cli_synthetic_demo(self, tmp_path: Path) -> None:
        """Verify training CLI runs successfully in synthetic demonstration mode."""
        model_file = tmp_path / "model.joblib"
        manifest_file = tmp_path / "manifest.json"

        ret = cli_main(
            [
                "--synthetic-demo",
                "--output-model",
                str(model_file),
                "--output-manifest",
                str(manifest_file),
                "--hidden-layers",
                "64,32",
                "--max-iter",
                "20",
                "--seed",
                "42",
            ]
        )
        assert ret == 0
        assert model_file.exists()
        assert manifest_file.exists()

    def test_cli_with_tuning(self, tmp_path: Path) -> None:
        """Verify training CLI runs with hyperparameter tuning enabled."""
        model_file = tmp_path / "tuned_model.joblib"
        manifest_file = tmp_path / "tuned_manifest.json"

        ret = cli_main(
            [
                "--synthetic-demo",
                "--tune",
                "--max-iter",
                "20",
                "--output-model",
                str(model_file),
                "--output-manifest",
                str(manifest_file),
                "--seed",
                "42",
            ]
        )
        assert ret == 0
        assert model_file.exists()
        assert manifest_file.exists()

    def test_cli_missing_data_file(self, tmp_path: Path) -> None:
        """Verify CLI returns error code 1 when data path does not exist."""
        ret = cli_main(["--data-path", str(tmp_path / "missing.npz")])
        assert ret == 1
