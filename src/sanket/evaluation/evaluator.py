"""Unified evaluation engine for sign recognition models.

Provides standardized benchmarking, seed locking, metric computation, runtime profiling,
and structured manifest generation across Random Forest, SVM, and Neural Network (MLP) baselines.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sanket.evaluation.latency import LatencyMetrics, profile_latency
from sanket.evaluation.manifest import ExperimentManifest, compute_data_checksum
from sanket.evaluation.metrics import EvaluationMetrics, compute_metrics
from sanket.models.base import BaseSignClassifier


def lock_seed(seed: int = 42) -> None:
    """Lock all pseudo-random number generator seeds for strict determinism.

    Satisfies docs/research/EVALUATION_METHODOLOGY.md Section 6.1.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


@dataclass
class EvaluationReport:
    """Complete evaluation report combining classification metrics, latency, and manifest."""

    model_name: str
    metrics: EvaluationMetrics
    latency: LatencyMetrics
    manifest: ExperimentManifest | None = None

    def summary(self) -> str:
        """Generate human-readable summary table of benchmark results."""
        m = self.metrics
        lat = self.latency

        # Target benchmarks from EVALUATION_METHODOLOGY.md Section 5
        p50_status = "PASS" if lat.median_ms <= 15.0 else "WARN"
        p95_status = "PASS" if lat.p95_ms <= 25.0 else "WARN"
        fps_status = "PASS" if lat.throughput_fps >= 25.0 else "WARN"

        lines = [
            "=" * 68,
            f"  SANKET EVALUATION REPORT — {self.model_name.upper()}",
            "=" * 68,
            f"  Samples Evaluated    : {m.num_samples}",
            f"  Vocabulary Classes   : {m.num_classes}",
            "-" * 68,
            "  CLASSIFICATION PERFORMANCE",
            "-" * 68,
            f"  Top-1 Accuracy       : {m.accuracy:.4f} ({m.accuracy * 100:.2f}%)",
            f"  Top-3 Accuracy       : {m.top_3_accuracy:.4f} ({m.top_3_accuracy * 100:.2f}%)",
            f"  Top-5 Accuracy       : {m.top_5_accuracy:.4f} ({m.top_5_accuracy * 100:.2f}%)",
            f"  Macro Precision      : {m.macro_precision:.4f}",
            f"  Macro Recall         : {m.macro_recall:.4f}",
            f"  Macro F1-Score       : {m.macro_f1:.4f}",
            f"  Weighted F1-Score    : {m.weighted_f1:.4f}",
            f"  Matthews Corr (MCC)  : {m.matthews_corrcoef:.4f}",
            "-" * 68,
            "  OPERATIONAL RUNTIME BENCHMARKS (EVALUATION_METHODOLOGY.md)",
            "-" * 68,
            f"  Latency Median (p50) : {lat.median_ms:.2f} ms  [Target <= 15.0 ms: {p50_status}]",
            f"  Latency p95          : {lat.p95_ms:.2f} ms  [Target <= 25.0 ms: {p95_status}]",
            f"  Latency p99          : {lat.p99_ms:.2f} ms",
            f"  Throughput (FPS)     : {lat.throughput_fps:.1f} FPS "
            f"[Target >= 25.0 FPS: {fps_status}]",
            f"  Benchmark Iterations : {lat.num_trials} trials",
        ]

        if lat.peak_memory_mb is not None:
            mem_status = "PASS" if lat.peak_memory_mb <= 200.0 else "WARN"
            lines.append(
                f"  Peak RAM Footprint   : {lat.peak_memory_mb:.2f} MB "
                f"[Target <= 200.0 MB: {mem_status}]"
            )

        if self.manifest:
            dirty_str = f" (dirty={self.manifest.git_dirty})"
            lines.extend(
                [
                    "-" * 68,
                    f"  Experiment ID        : {self.manifest.experiment_id}",
                    f"  Git Commit SHA       : {self.manifest.git_commit_sha[:10]}{dirty_str}",
                    f"  Dataset Checksum     : {self.manifest.dataset_checksum[:16]}...",
                ]
            )

        lines.append("=" * 68)
        return "\n".join(lines)


class SignModelEvaluator:
    """Standardized evaluation harness executing model testing across all Phase 1 baselines."""

    def __init__(
        self,
        seed: int = 42,
        latency_trials: int = 100,
        warm_up_trials: int = 10,
    ) -> None:
        """Initialize the evaluation harness.

        Args:
            seed: Seed locked across evaluation operations.
            latency_trials: Number of benchmark trials for latency estimation.
            warm_up_trials: Number of initial iterations discarded during profiling.
        """
        self.seed = seed
        self.latency_trials = latency_trials
        self.warm_up_trials = warm_up_trials

    def evaluate(
        self,
        model: BaseSignClassifier,
        x_test: np.ndarray,
        y_test: np.ndarray,
        classes: list[str] | None = None,
        experiment_id: str | None = None,
        signer_partitions: dict[str, list[str]] | None = None,
        dataset_checksum: str | None = None,
        save_manifest_dir: str | Path | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Run standardized evaluation on a trained sign classifier.

        Args:
            model: Trained classifier implementing BaseSignClassifier contract.
            x_test: Feature matrix of shape (N, D).
            y_test: Ground truth integer class labels of shape (N,).
            classes: Optional class gloss names. Defaults to model.classes or generic names.
            experiment_id: Unique identifier for experiment manifest tracking.
            signer_partitions: Optional mapping of split names to signer IDs.
            dataset_checksum: SHA-256 hash of test data (computed if None).
            save_manifest_dir: Directory where structured JSON manifest will be written.
            model_metadata: Additional model hyperparameters or architecture info.

        Returns:
            EvaluationReport containing metrics, latency, and manifest.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If test data is empty.
        """
        if not model.is_fitted:
            msg = "Cannot evaluate unfitted model. Train model before evaluation."
            raise RuntimeError(msg)

        x_arr = np.asarray(x_test, dtype=np.float32)
        y_arr = np.asarray(y_test, dtype=int).ravel()

        if len(x_arr) == 0:
            msg = "Evaluation feature dataset is empty."
            raise ValueError(msg)

        if len(x_arr) != len(y_arr):
            msg = f"Length mismatch: len(x)={len(x_arr)} != len(y)={len(y_arr)}."
            raise ValueError(msg)

        # Enforce deterministic environment
        lock_seed(self.seed)

        active_classes = classes or model.classes
        model_name = getattr(model, "__class__", type(model)).__name__

        # Compute predictions
        y_pred = model.predict(x_arr)

        y_proba: np.ndarray | None = None
        try:
            y_proba = model.predict_proba(x_arr)
        except Exception:
            pass

        # Compute classification metrics
        metrics = compute_metrics(
            y_true=y_arr,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=active_classes,
        )

        # Profile runtime latency on single sample
        test_sample = x_arr[0]
        latency = profile_latency(
            predict_fn=model.predict,
            sample=test_sample,
            num_trials=self.latency_trials,
            warm_up=self.warm_up_trials,
        )

        # Checksum of evaluation data
        checksum = dataset_checksum or compute_data_checksum(x_arr)

        exp_id = experiment_id or f"EXP-{model_name.upper()}-EVAL"
        metadata = model_metadata or {}
        if hasattr(model, "get_params"):
            try:
                metadata.setdefault("hyperparameters", model.get_params())
            except Exception:
                pass
        metadata.setdefault("model_class", model_name)

        manifest = ExperimentManifest.create(
            experiment_id=exp_id,
            active_vocabulary=metrics.classes,
            metrics=metrics.to_dict(),
            runtime=latency.to_dict(),
            dataset_checksum=checksum,
            signer_partitions=signer_partitions or {},
            model_metadata=metadata,
        )

        if save_manifest_dir is not None:
            manifest_dir = Path(save_manifest_dir)
            manifest_file = manifest_dir / f"{exp_id}.json"
            manifest.save(manifest_file)

        return EvaluationReport(
            model_name=model_name,
            metrics=metrics,
            latency=latency,
            manifest=manifest,
        )
