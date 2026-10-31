"""Random Forest baseline classifier for isolated Indian Sign Language recognition."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sanket.models.base import BaseSignClassifier

logger = logging.getLogger("sanket.models.random_forest")


class RandomForestSignClassifier(BaseSignClassifier):
    """Random Forest classifier for skeletal sign recognition."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int | None = 20,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        class_weight: str | dict[Any, float] | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Random Forest hyper-parameters.

        Args:
            n_estimators: Number of decision trees in the forest.
            max_depth: Maximum tree depth. If None, nodes expand until purity.
            min_samples_split: Minimum number of samples required to split a node.
            min_samples_leaf: Minimum number of samples required at a leaf node.
            max_features: Number of features to consider when looking for the best split.
            class_weight: Weighting strategy for class balancing ('balanced' or dict).
            random_state: Seed for reproducible tree bootstrap and feature splitting.
            n_jobs: Number of CPU workers for parallel tree training (-1 for all cores).
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimator_: RandomForestClassifier | None = None
        self.feature_importances_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return dictionary of current model hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> RandomForestSignClassifier:
        """Fit Random Forest on feature matrix X and integer class targets y.

        Args:
            x: 2D array of shape (N, D).
            y: 1D array of integer class labels (N,).
            classes: Optional human-readable class names.
            feature_names: Optional feature column names.

        Returns:
            self.
        """
        if x.ndim != 2:
            msg = f"Expected 2D feature matrix (N, D), got shape {x.shape}"
            raise ValueError(msg)
        if len(x) != len(y):
            msg = f"Sample count mismatch: X has {len(x)}, y has {len(y)}"
            raise ValueError(msg)

        self.classes_ = list(classes) if classes is not None else [str(c) for c in np.unique(y)]
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        t0 = time.perf_counter()
        self.estimator_.fit(x, y)
        train_time = time.perf_counter() - t0

        self.feature_importances_ = self.estimator_.feature_importances_
        self.is_fitted_ = True
        logger.info(
            "Fitted RandomForestSignClassifier with %d trees in %.2fs",
            self.n_estimators,
            train_time,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer class indices for input feature matrix X."""
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Classifier is not fitted. Call fit() before predict()."
            raise RuntimeError(msg)

        return self.estimator_.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability distributions for input feature matrix X."""
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Classifier is not fitted. Call fit() before predict_proba()."
            raise RuntimeError(msg)

        return self.estimator_.predict_proba(x)

    def get_top_features(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the top-k most discriminative features ranked by Gini importance.

        Args:
            k: Number of top features to return.

        Returns:
            List of (feature_name, importance_score) tuples sorted in descending order.
        """
        if not self.is_fitted_ or self.feature_importances_ is None:
            msg = "Classifier is not fitted."
            raise RuntimeError(msg)

        indices = np.argsort(self.feature_importances_)[::-1][:k]
        result: list[tuple[str, float]] = []
        for idx in indices:
            score = float(self.feature_importances_[idx])
            name = (
                self.feature_names_[idx]
                if self.feature_names_ and idx < len(self.feature_names_)
                else f"feature_{idx}"
            )
            result.append((name, score))

        return result

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
        """Perform validation search over hyperparameter combinations.

        Args:
            x_train: Training feature matrix.
            y_train: Training labels.
            x_val: Validation feature matrix.
            y_val: Validation labels.
            param_grid: Dictionary mapping parameter names to lists of candidate values.
            classes: Optional class names.
            feature_names: Optional feature names.

        Returns:
            Dictionary containing best parameters, best validation F1, and all trials.
        """
        grid = param_grid or {
            "n_estimators": [50, 100, 150],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }

        from itertools import product

        keys = list(grid.keys())
        combos = list(product(*grid.values()))

        logger.info("Starting hyperparameter tuning over %d combinations...", len(combos))
        best_f1 = -1.0
        best_params: dict[str, Any] = {}
        trials: list[dict[str, Any]] = []

        for values in combos:
            params = dict(zip(keys, values, strict=True))
            clf = RandomForestSignClassifier(
                **params,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            clf.fit(x_train, y_train, classes=classes, feature_names=feature_names)
            metrics = clf.evaluate(x_val, y_val, classes=classes)

            val_f1 = metrics.macro_f1
            trials.append({
                "params": params,
                "val_macro_f1": val_f1,
                "val_accuracy": metrics.accuracy,
            })

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = params

        logger.info(
            "Hyperparameter tuning completed. Best Val Macro F1: %.4f with params: %s",
            best_f1,
            best_params,
        )

        # Update self with best params and fit on training set
        for k, v in best_params.items():
            setattr(self, k, v)

        self.fit(x_train, y_train, classes=classes, feature_names=feature_names)

        return {
            "best_params": best_params,
            "best_val_macro_f1": best_f1,
            "trials": trials,
        }

    def save(self, model_path: str | Path) -> Path:
        """Serialize model weights and metadata using joblib.

        Args:
            model_path: Destination path for .joblib artifact.

        Returns:
            Verified Path to written file.
        """
        if not self.is_fitted_ or self.estimator_ is None:
            msg = "Cannot save an unfitted model."
            raise RuntimeError(msg)

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "architecture": "random_forest",
            "estimator": self.estimator_,
            "classes_": self.classes_,
            "feature_names_": self.feature_names_,
            "feature_importances_": self.feature_importances_,
            "params": self.get_params(),
            "timestamp": time.time(),
        }

        joblib.dump(bundle, str(path))
        logger.info("Saved RandomForestSignClassifier model to %s", path)
        return path

    @classmethod
    def load(cls, model_path: str | Path) -> RandomForestSignClassifier:
        """Load serialized Random Forest model from disk.

        Args:
            model_path: Path to .joblib file.

        Returns:
            Restored RandomForestSignClassifier instance.
        """
        path = Path(model_path)
        if not path.is_file():
            msg = f"Model artifact not found: {path}"
            raise FileNotFoundError(msg)

        bundle = joblib.load(str(path))
        if not isinstance(bundle, dict) or bundle.get("architecture") != "random_forest":
            msg = f"Invalid model artifact in {path}"
            raise ValueError(msg)

        params = bundle.get("params", {})
        instance = cls(**params)
        instance.estimator_ = bundle["estimator"]
        instance.classes_ = bundle.get("classes_")
        instance.feature_names_ = bundle.get("feature_names_")
        instance.feature_importances_ = bundle.get("feature_importances_")
        instance.is_fitted_ = True

        logger.info("Loaded RandomForestSignClassifier from %s", path)
        return instance
