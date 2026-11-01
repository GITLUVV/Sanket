"""Support Vector Machine (SVM) baseline classifier for isolated sign recognition."""

from __future__ import annotations

import logging
import time
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sanket.models.base import BaseSignClassifier

logger = logging.getLogger("sanket.models.svm")


class SVMSignClassifier(BaseSignClassifier):
    """Support Vector Machine classifier with RBF kernel and feature standardization."""

    def __init__(
        self,
        c: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        probability: bool = True,
        class_weight: str | dict[Any, float] | None = "balanced",
        random_state: int = 42,
        scale_features: bool = True,
    ) -> None:
        """Initialize SVM classifier hyperparameters.

        Args:
            c: Regularization parameter. Greater values penalize misclassifications more.
            kernel: Kernel specification ('rbf', 'linear', 'poly', 'sigmoid').
            gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
            degree: Degree of polynomial kernel function ('poly').
            probability: Whether to enable Platt scaling probability estimates.
            class_weight: Set the parameter C of class i to class_weight[i]*C.
            random_state: Pseudo-random number generator seed for probability calibration.
            scale_features: Whether to fit and apply StandardScaler to input features.
        """
        super().__init__()
        self.c = float(c)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.class_weight = class_weight
        self.random_state = random_state
        self.scale_features = scale_features

        self.estimator_: SVC | None = None
        self.scaler_: StandardScaler | None = None

    def get_params(self) -> dict[str, Any]:
        """Return dictionary of current model hyperparameters."""
        return {
            "c": self.c,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "probability": self.probability,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "scale_features": self.scale_features,
        }

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> SVMSignClassifier:
        """Fit SVM model on feature matrix X and integer class targets y.

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

        self.estimator_ = SVC(
            C=self.c,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

        t0 = time.perf_counter()
        self.estimator_.fit(x_processed, y_arr)
        train_time = time.perf_counter() - t0

        self.is_fitted_ = True
        logger.info(
            "Fitted SVMSignClassifier (kernel=%s, C=%.2f) on %d samples in %.2fs",
            self.kernel,
            self.c,
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
        """Predict calibrated class probability distributions for input feature matrix X."""
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Classifier is not fitted. Call fit() before predict_proba()."
            raise RuntimeError(msg)

        if not self.probability:
            msg = "Probability estimation is disabled. Re-fit with probability=True."
            raise RuntimeError(msg)

        x_proc = self._transform_features(x)
        return self.estimator_.predict_proba(x_proc)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Evaluate decision function distance to separating hyperplane for input X."""
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Classifier is not fitted. Call fit() before decision_function()."
            raise RuntimeError(msg)

        x_proc = self._transform_features(x)
        return self.estimator_.decision_function(x_proc)

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
            param_grid: Dictionary of candidate hyperparameters (C, gamma, kernel).
            classes: Optional human-readable class names.
            feature_names: Optional feature column names.

        Returns:
            Dictionary containing best parameters, best validation F1, and all trials.
        """
        grid = param_grid or {
            "c": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.001],
            "kernel": ["rbf"],
        }

        keys = list(grid.keys())
        combos = list(product(*grid.values()))
        logger.info("Starting SVM hyperparameter grid search over %d combinations...", len(combos))

        best_score = -1.0
        best_params: dict[str, Any] = {}
        trials: list[dict[str, Any]] = []

        from sklearn.metrics import f1_score

        for combo in combos:
            params = dict(zip(keys, combo, strict=True))
            clf = SVMSignClassifier(
                c=params.get("c", self.c),
                kernel=params.get("kernel", self.kernel),
                gamma=params.get("gamma", self.gamma),
                degree=params.get("degree", self.degree),
                probability=False,  # Faster evaluation during search
                class_weight=self.class_weight,
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

        logger.info("Best SVM Validation Macro F1: %.4f with %s", best_score, best_params)

        # Update instance hyperparameters to optimal values and re-fit with probability=True
        self.c = float(best_params.get("c", self.c))
        self.gamma = best_params.get("gamma", self.gamma)
        self.kernel = best_params.get("kernel", self.kernel)
        self.fit(x_train, y_train, classes=classes, feature_names=feature_names)

        return {
            "best_params": best_params,
            "best_val_macro_f1": best_score,
            "trials": trials,
        }

    def save(self, model_path: str | Path) -> Path:
        """Serialize model, fitted scaler, and metadata to disk using joblib.

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
            "hyperparameters": self.get_params(),
            "model_version": "1.0.0",
        }
        joblib.dump(payload, p)
        logger.info("Saved SVMSignClassifier artifact to %s", p)
        return p

    @classmethod
    def load(cls, model_path: str | Path) -> SVMSignClassifier:
        """Load serialized model artifact, scaler, and metadata from disk.

        Args:
            model_path: Path to serialized .joblib file.

        Returns:
            Restored SVMSignClassifier instance.
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
            c=params.get("c", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            degree=params.get("degree", 3),
            probability=params.get("probability", True),
            class_weight=params.get("class_weight", "balanced"),
            random_state=params.get("random_state", 42),
            scale_features=params.get("scale_features", True),
        )

        instance.estimator_ = payload["estimator"]
        instance.scaler_ = payload.get("scaler")
        instance.classes_ = payload.get("classes")
        instance.feature_names_ = payload.get("feature_names")
        instance.is_fitted_ = True
        return instance
