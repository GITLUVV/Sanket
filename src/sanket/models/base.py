"""Abstract base classes and evaluation metrics data models for sign recognition classifiers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


@dataclass
class SignModelMetrics:
    """Standardized performance evaluation metrics for sign language classifiers."""

    accuracy: float
    top_3_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    num_samples: int
    inference_latency_ms: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "accuracy": float(self.accuracy),
            "top_3_accuracy": float(self.top_3_accuracy),
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "num_samples": int(self.num_samples),
            "inference_latency_ms": float(self.inference_latency_ms),
            "per_class_f1": {k: float(v) for k, v in self.per_class_f1.items()},
            "confusion_matrix": self.confusion_matrix,
            "classes": self.classes,
            "metadata": self.metadata,
        }


class BaseSignClassifier(ABC):
    """Abstract base class defining the standardized sign recognition model interface."""

    def __init__(self) -> None:
        self.classes_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.is_fitted_: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        return self.is_fitted_

    @property
    def classes(self) -> list[str] | None:
        """Return target class names if provided."""
        return self.classes_

    @property
    def feature_names(self) -> list[str] | None:
        """Return feature names if provided."""
        return self.feature_names_

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> BaseSignClassifier:
        """Fit the model to training features X and integer class labels y."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer class labels for feature array X."""

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability distributions for feature array X."""

    @abstractmethod
    def save(self, model_path: str | Path) -> Path:
        """Serialize model weights and metadata to disk."""

    @classmethod
    @abstractmethod
    def load(cls, model_path: str | Path) -> BaseSignClassifier:
        """Load serialized model from disk."""

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        classes: list[str] | None = None,
    ) -> SignModelMetrics:
        """Evaluate classifier performance against ground truth labels.

        Args:
            x: Feature matrix of shape (N, D).
            y: Integer ground truth labels of shape (N,).
            classes: Optional class names. If None, uses self.classes_.

        Returns:
            SignModelMetrics container.
        """
        if not self.is_fitted_:
            msg = "Cannot evaluate unfitted model. Call fit() first."
            raise RuntimeError(msg)

        if len(x) == 0:
            msg = "Evaluation dataset is empty."
            raise ValueError(msg)

        class_names = classes or self.classes_
        num_classes = len(class_names) if class_names else len(np.unique(y))

        # Benchmark inference latency
        t0 = time.perf_counter()
        y_pred = self.predict(x)
        y_proba = self.predict_proba(x)
        elapsed_s = time.perf_counter() - t0
        latency_ms = (elapsed_s / len(x)) * 1000.0

        acc = float(accuracy_score(y, y_pred))

        # Top-3 accuracy (if at least 3 classes are present)
        if num_classes >= 3 and y_proba.shape[1] >= 3:
            try:
                # all_classes ensures top_k_accuracy handles subsets of classes safely
                labels_range = np.arange(num_classes)
                top3 = float(top_k_accuracy_score(y, y_proba, k=3, labels=labels_range))
            except Exception:
                top3 = acc
        else:
            top3 = acc

        macro_p = float(precision_score(y, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y, y_pred, average="weighted", zero_division=0))

        # Per-class F1 metrics
        per_class_scores = f1_score(y, y_pred, average=None, zero_division=0)
        per_class_dict: dict[str, float] = {}
        if class_names is not None and len(class_names) == len(per_class_scores):
            for c_name, score in zip(class_names, per_class_scores, strict=True):
                per_class_dict[c_name] = float(score)
        else:
            for idx, score in enumerate(per_class_scores):
                per_class_dict[str(idx)] = float(score)

        cm = confusion_matrix(y, y_pred).tolist()

        return SignModelMetrics(
            accuracy=acc,
            top_3_accuracy=top3,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            num_samples=len(x),
            inference_latency_ms=latency_ms,
            per_class_f1=per_class_dict,
            confusion_matrix=cm,
            classes=class_names or [],
        )
