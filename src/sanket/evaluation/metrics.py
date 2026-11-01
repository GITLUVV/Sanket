"""Standardized quantitative performance metric calculation for sign language recognition.

Implements formal classification metrics codified in docs/research/EVALUATION_METHODOLOGY.md:
- Top-1 Accuracy
- Top-3 and Top-5 Accuracy
- Macro-Averaged Precision, Recall, and F1-Score
- Weighted-Averaged Precision, Recall, and F1-Score
- Multi-Class Matthew's Correlation Coefficient (MCC)
- Per-Class Precision, Recall, F1, and Support
- Row-Normalized Confusion Matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


@dataclass
class EvaluationMetrics:
    """Comprehensive performance metrics container for sign language classifiers."""

    accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    matthews_corrcoef: float
    num_samples: int
    num_classes: int
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    normalized_confusion_matrix: list[list[float]] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return {
            "accuracy": round(float(self.accuracy), 6),
            "top_3_accuracy": round(float(self.top_3_accuracy), 6),
            "top_5_accuracy": round(float(self.top_5_accuracy), 6),
            "macro_precision": round(float(self.macro_precision), 6),
            "macro_recall": round(float(self.macro_recall), 6),
            "macro_f1": round(float(self.macro_f1), 6),
            "weighted_precision": round(float(self.weighted_precision), 6),
            "weighted_recall": round(float(self.weighted_recall), 6),
            "weighted_f1": round(float(self.weighted_f1), 6),
            "matthews_corrcoef": round(float(self.matthews_corrcoef), 6),
            "num_samples": int(self.num_samples),
            "num_classes": int(self.num_classes),
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "normalized_confusion_matrix": [
                [round(float(val), 6) for val in row] for row in self.normalized_confusion_matrix
            ],
            "classes": self.classes,
        }


def compute_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_proba: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> EvaluationMetrics:
    """Compute primary classification metrics, MCC, per-class stats, and confusion matrix.

    Args:
        y_true: Ground truth integer class labels of shape (N,).
        y_pred: Predicted integer class labels of shape (N,).
        y_proba: Optional predicted class probability matrix of shape (N, C).
        classes: Optional list of class names corresponding to integer indices.

    Returns:
        EvaluationMetrics dataclass populated with computed metrics.

    Raises:
        ValueError: If y_true and y_pred have mismatched lengths or are empty.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=int).ravel()

    if len(y_true_arr) == 0:
        msg = "Cannot compute metrics on empty arrays."
        raise ValueError(msg)

    if len(y_true_arr) != len(y_pred_arr):
        msg = f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}."
        raise ValueError(msg)

    num_samples = len(y_true_arr)

    # Determine class universe
    if classes is not None and len(classes) > 0:
        class_names = list(classes)
        num_classes = len(class_names)
        labels_universe = np.arange(num_classes)
    else:
        unique_labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        num_classes = max(int(np.max(unique_labels)) + 1, len(unique_labels))
        class_names = [f"class_{i}" for i in range(num_classes)]
        labels_universe = np.arange(num_classes)

    # Top-1 Accuracy
    top1_acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Top-3 and Top-5 Accuracy
    if y_proba is not None and y_proba.ndim == 2:
        proba_classes = y_proba.shape[1]
        # Top-3
        if proba_classes >= 3 and num_classes >= 3:
            try:
                top3_acc = float(
                    top_k_accuracy_score(y_true_arr, y_proba, k=3, labels=labels_universe)
                )
            except Exception:
                top3_acc = top1_acc
        else:
            top3_acc = top1_acc

        # Top-5
        if proba_classes >= 5 and num_classes >= 5:
            try:
                top5_acc = float(
                    top_k_accuracy_score(y_true_arr, y_proba, k=5, labels=labels_universe)
                )
            except Exception:
                top5_acc = top1_acc
        else:
            top5_acc = top1_acc
    else:
        top3_acc = top1_acc
        top5_acc = top1_acc

    # Macro and Weighted Precision, Recall, F1
    macro_p = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            zero_division=0,
            labels=labels_universe,
        )
    )
    macro_r = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            zero_division=0,
            labels=labels_universe,
        )
    )
    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            zero_division=0,
            labels=labels_universe,
        )
    )

    weighted_p = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            zero_division=0,
            labels=labels_universe,
        )
    )
    weighted_r = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            zero_division=0,
            labels=labels_universe,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            average="weighted",
            zero_division=0,
            labels=labels_universe,
        )
    )

    # Matthews Correlation Coefficient
    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    except Exception:
        mcc = 0.0

    # Per-class metrics
    p_per_class = precision_score(
        y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels_universe
    )
    r_per_class = recall_score(
        y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels_universe
    )
    f1_per_class = f1_score(
        y_true_arr, y_pred_arr, average=None, zero_division=0, labels=labels_universe
    )

    per_class_dict: dict[str, dict[str, float]] = {}
    for idx, c_name in enumerate(class_names):
        support_count = int(np.sum(y_true_arr == idx))
        per_class_dict[c_name] = {
            "precision": round(float(p_per_class[idx]), 6),
            "recall": round(float(r_per_class[idx]), 6),
            "f1": round(float(f1_per_class[idx]), 6),
            "support": support_count,
        }

    # Confusion matrix (raw and row-normalized)
    cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_universe)
    cm_raw_list = cm_raw.tolist()

    row_sums = cm_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm_raw / row_sums, 0.0)
    cm_norm_list = cm_norm.tolist()

    return EvaluationMetrics(
        accuracy=top1_acc,
        top_3_accuracy=top3_acc,
        top_5_accuracy=top5_acc,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_precision=weighted_p,
        weighted_recall=weighted_r,
        weighted_f1=weighted_f1,
        matthews_corrcoef=mcc,
        num_samples=num_samples,
        num_classes=num_classes,
        per_class=per_class_dict,
        confusion_matrix=cm_raw_list,
        normalized_confusion_matrix=cm_norm_list,
        classes=class_names,
    )
