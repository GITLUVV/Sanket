"""Unified evaluation module for Sanket sign language recognition models.

Provides standardized metric computation, operational latency profiling, structured
experiment manifests, evaluation harnesses, and confusion matrix diagnostics across
all baseline models.
"""

from sanket.evaluation.diagnostics import (
    ISL_16_PHONOLOGY,
    ConfusablePair,
    ConfusionMatrixDiagnostics,
    ErrorBreakdown,
    PhonologicalErrorType,
    SignPhonology,
    classify_phonological_error,
)
from sanket.evaluation.evaluator import EvaluationReport, SignModelEvaluator, lock_seed
from sanket.evaluation.latency import LatencyMetrics, profile_latency
from sanket.evaluation.manifest import (
    ExperimentManifest,
    compute_data_checksum,
    get_git_provenance,
    get_sanitized_environment_info,
)
from sanket.evaluation.metrics import EvaluationMetrics, compute_metrics

__all__ = [
    "ConfusablePair",
    "ConfusionMatrixDiagnostics",
    "ErrorBreakdown",
    "EvaluationMetrics",
    "EvaluationReport",
    "ExperimentManifest",
    "ISL_16_PHONOLOGY",
    "LatencyMetrics",
    "PhonologicalErrorType",
    "SignModelEvaluator",
    "SignPhonology",
    "classify_phonological_error",
    "compute_data_checksum",
    "compute_metrics",
    "get_git_provenance",
    "get_sanitized_environment_info",
    "lock_seed",
    "profile_latency",
]
