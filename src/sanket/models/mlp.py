"""Multi-Layer Perceptron (MLP) neural network baseline classifier for isolated sign recognition."""

from __future__ import annotations

import logging
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sanket.models.base import BaseSignClassifier

logger = logging.getLogger("sanket.models.mlp")


class MLPSignClassifier(BaseSignClassifier):
    """Multi-Layer Perceptron neural network classifier for isolated sign landmark features."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: int | str = 32,
        learning_rate: str = "adaptive",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        n_iter_no_change: int = 15,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        scale_features: bool = True,
    ) -> None:
        """Initialize Multi-Layer Perceptron classifier hyperparameters.

        Args:
            hidden_layer_sizes: Tuple representing number of neurons per hidden layer.
            activation: Activation function ('identity', 'logistic', 'tanh', 'relu').
            solver: Solver for weight optimization ('lbfgs', 'sgd', 'adam').
            alpha: L2 penalty (regularization term) parameter.
            batch_size: Size of minibatches for stochastic optimizers.
            learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive').
            learning_rate_init: Initial learning rate used.
            max_iter: Maximum number of epochs.
            early_stopping: Whether to terminate training when validation score is not improving.
            n_iter_no_change: Maximum epochs with no improvement before early stopping triggers.
            validation_fraction: Proportion of training data set aside for early stopping.
            random_state: Random number seed for reproducible weight initialization.
            scale_features: Whether to fit and apply StandardScaler to input features.
        """
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = float(validation_fraction)
        self.random_state = random_state
        self.scale_features = scale_features

        self.estimator_: MLPClassifier | None = None
        self.scaler_: StandardScaler | None = None
        self.loss_curve_: list[float] | None = None
        self.validation_scores_: list[float] | None = None

    def get_params(self) -> dict[str, Any]:
        """Return dictionary of current model hyperparameters."""
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "scale_features": self.scale_features,
        }

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> MLPSignClassifier:
        """Fit MLP neural network on feature matrix X and integer class targets y.

        StandardScaler is fitted strictly on the input training features when
        scale_features is True to prevent data leakage from validation/test sets.

        Args:
            x: 2D feature array of shape (N, D).
            y: 1D array of integer class labels (N,).
            classes: Optional human-readable class names.
            feature_names: Optional feature column names.

        Returns:
            self.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int).ravel()

        if x_arr.ndim != 2:
            msg = f"Expected 2D feature matrix (N, D), got shape {x_arr.shape}"
            raise ValueError(msg)
        if len(x_arr) != len(y_arr):
            msg = f"Sample count mismatch: X has {len(x_arr)}, y has {len(y_arr)}"
            raise ValueError(msg)
        if len(x_arr) == 0:
            msg = "Cannot fit classifier on empty dataset."
            raise ValueError(msg)

        self.classes_ = list(classes) if classes is not None else [str(c) for c in np.unique(y_arr)]
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        # Fit feature standardization on training partition
        if self.scale_features:
            self.scaler_ = StandardScaler()
            x_processed = self.scaler_.fit_transform(x_arr)
        else:
            self.scaler_ = None
            x_processed = x_arr

        # Handle batch size if larger than samples
        effective_batch_size = (
            min(int(self.batch_size), len(x_arr))
            if isinstance(self.batch_size, int)
            else self.batch_size
        )

        self.estimator_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=effective_batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

        t0 = time.perf_counter()
        self.estimator_.fit(x_processed, y_arr)
        train_time = time.perf_counter() - t0

        self.loss_curve_ = getattr(self.estimator_, "loss_curve_", None)
        self.validation_scores_ = getattr(self.estimator_, "validation_scores_", None)
        self.is_fitted_ = True

        epochs = getattr(self.estimator_, "n_iter_", self.max_iter)
        logger.info(
            "Fitted MLPSignClassifier (layers=%s, epochs=%d) on %d samples in %.2fs",
            self.hidden_layer_sizes,
            epochs,
            len(x_arr),
            train_time,
        )
        return self

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        """Apply feature scaling if configured."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if self.scale_features and self.scaler_ is not None:
            return self.scaler_.transform(x_arr)
        return x_arr

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer class indices for input feature matrix X."""
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Classifier is not fitted. Call fit() before predict()."
            raise RuntimeError(msg)

        x_proc = self._transform_features(x)
        return self.estimator_.predict(x_proc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability distributions for input feature matrix X."""
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Classifier is not fitted. Call fit() before predict_proba()."
            raise RuntimeError(msg)

        x_proc = self._transform_features(x)
        return self.estimator_.predict_proba(x_proc)

    def tune_hyperparameters(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: dict[str, list[Any]] | None = None,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform validation grid search optimizing Macro F1 score on validation split.

        Args:
            x_train: Training feature matrix.
            y_train: Training integer labels.
            x_val: Validation feature matrix.
            y_val: Validation integer labels.
            param_grid: Dictionary of candidate hyperparameters (hidden_layer_sizes, alpha, lr).
            classes: Optional human-readable class names.
            feature_names: Optional feature column names.

        Returns:
            Dictionary containing best parameters, best validation F1, and all trials.
        """
        grid = param_grid or {
            "hidden_layer_sizes": [(128, 64), (256, 128, 64)],
            "alpha": [0.0001, 0.01],
            "learning_rate_init": [0.001],
        }

        keys = list(grid.keys())
        combos = list(product(*grid.values()))
        logger.info("Starting MLP hyperparameter grid search over %d combinations...", len(combos))

        best_score = -1.0
        best_params: dict[str, Any] = {}
        trials: list[dict[str, Any]] = []

        from sklearn.metrics import f1_score

        for combo in combos:
            params = dict(zip(keys, combo, strict=True))
            clf = MLPSignClassifier(
                hidden_layer_sizes=params.get("hidden_layer_sizes", self.hidden_layer_sizes),
                activation=self.activation,
                solver=self.solver,
                alpha=params.get("alpha", self.alpha),
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                learning_rate_init=params.get("learning_rate_init", self.learning_rate_init),
                max_iter=min(30, self.max_iter),  # Fast tuning iterations
                early_stopping=True,
                n_iter_no_change=5,
                random_state=self.random_state,
                scale_features=self.scale_features,
            )
            clf.fit(x_train, y_train, classes=classes, feature_names=feature_names)
            preds = clf.predict(x_val)
            macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

            trial_record = {"params": params, "val_macro_f1": macro_f1}
            trials.append(trial_record)

            if macro_f1 > best_score:
                best_score = macro_f1
                best_params = params

        logger.info("Best MLP Validation Macro F1: %.4f with %s", best_score, best_params)

        # Update instance hyperparameters to optimal values and re-fit on training set
        if "hidden_layer_sizes" in best_params:
            self.hidden_layer_sizes = best_params["hidden_layer_sizes"]
        if "alpha" in best_params:
            self.alpha = float(best_params["alpha"])
        if "learning_rate_init" in best_params:
            self.learning_rate_init = float(best_params["learning_rate_init"])

        self.fit(x_train, y_train, classes=classes, feature_names=feature_names)

        return {
            "best_params": best_params,
            "best_val_macro_f1": best_score,
            "trials": trials,
        }

    def save(self, model_path: str | Path) -> Path:
        """Serialize model, fitted scaler, convergence curves, and metadata to disk using joblib.

        Args:
            model_path: Destination path (.joblib).

        Returns:
            Resolved Path object.
        """
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Cannot serialize unfitted model. Call fit() first."
            raise RuntimeError(msg)

        p = Path(model_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "estimator": self.estimator_,
            "scaler": self.scaler_,
            "classes": self.classes_,
            "feature_names": self.feature_names_,
            "loss_curve": self.loss_curve_,
            "validation_scores": self.validation_scores_,
            "hyperparameters": self.get_params(),
            "model_version": "1.0.0",
        }
        joblib.dump(payload, p)
        logger.info("Saved MLPSignClassifier artifact to %s", p)
        return p

    @classmethod
    def load(cls, model_path: str | Path) -> MLPSignClassifier:
        """Load serialized model artifact, scaler, and metadata from disk.

        Args:
            model_path: Path to serialized .joblib file.

        Returns:
            Restored MLPSignClassifier instance.
        """
        p = Path(model_path).resolve()
        if not p.is_file():
            msg = f"Model artifact not found: {p}"
            raise FileNotFoundError(msg)

        payload = joblib.load(p)
        if not isinstance(payload, dict) or "estimator" not in payload:
            msg = f"Invalid model payload in {p}: expected dictionary with 'estimator'"
            raise ValueError(msg)

        params = payload.get("hyperparameters", {})
        instance = cls(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (256, 128, 64))),
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=params.get("alpha", 0.0001),
            batch_size=params.get("batch_size", 32),
            learning_rate=params.get("learning_rate", "adaptive"),
            learning_rate_init=params.get("learning_rate_init", 0.001),
            max_iter=params.get("max_iter", 200),
            early_stopping=params.get("early_stopping", True),
            n_iter_no_change=params.get("n_iter_no_change", 15),
            validation_fraction=params.get("validation_fraction", 0.15),
            random_state=params.get("random_state", 42),
            scale_features=params.get("scale_features", True),
        )

        instance.estimator_ = payload["estimator"]
        instance.scaler_ = payload.get("scaler")
        instance.classes_ = payload.get("classes")
        instance.feature_names_ = payload.get("feature_names")
        instance.loss_curve_ = payload.get("loss_curve")
        instance.validation_scores_ = payload.get("validation_scores")
        instance.is_fitted_ = True
        return instance
