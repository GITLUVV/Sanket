"""Unit tests for Phase 1 baseline model comparison and ADR-0003 (SANKET-033)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sanket.models import RandomForestSignClassifier
from scripts import compare_baselines


def test_generate_synthetic_benchmark_data() -> None:
    """Verify synthetic benchmark dataset generator creates proper clusters and types."""
    x, y, classes = compare_baselines.generate_synthetic_benchmark_data(
        num_classes=16,
        num_features=645,
        samples_per_class=10,
        seed=42,
    )
    assert x.shape == (160, 645)
    assert y.shape == (160,)
    assert len(classes) == 16
    assert x.dtype == np.float32
    assert y.dtype == np.int64


def test_evaluate_candidate_model() -> None:
    """Verify single candidate model evaluation generates a valid EvaluationReport."""
    x_train, y_train, classes = compare_baselines.generate_synthetic_benchmark_data(
        num_classes=4,
        num_features=20,
        samples_per_class=15,
        seed=42,
    )
    x_test, y_test, _ = compare_baselines.generate_synthetic_benchmark_data(
        num_classes=4,
        num_features=20,
        samples_per_class=5,
        seed=43,
    )

    model = RandomForestSignClassifier(n_estimators=10, n_jobs=1, random_state=42)
    model.fit(x_train, y_train, classes=classes)

    rep = compare_baselines.evaluate_candidate_model(
        model=model,
        name="TestRF",
        x_test=x_test,
        y_test=y_test,
        classes=classes,
        seed=42,
        latency_trials=10,
    )
    assert rep.model_name == "RandomForestSignClassifier"
    assert rep.metrics.accuracy >= 0.0
    assert rep.latency.median_ms >= 0.0
    assert rep.latency.throughput_fps > 0.0


def test_format_comparison_markdown_table() -> None:
    """Verify Markdown table generation formats rows, columns, and target checks correctly."""
    x, y, classes = compare_baselines.generate_synthetic_benchmark_data(
        num_classes=4,
        num_features=20,
        samples_per_class=10,
        seed=42,
    )
    model = RandomForestSignClassifier(n_estimators=10, n_jobs=1, random_state=42)
    model.fit(x, y, classes=classes)

    rep = compare_baselines.evaluate_candidate_model(
        model=model,
        name="CandidateRF",
        x_test=x,
        y_test=y,
        classes=classes,
        latency_trials=10,
    )

    table = compare_baselines.format_comparison_markdown_table([("CandidateRF", rep)])
    assert "| Model Architecture | Top-1 Acc |" in table
    assert "**CandidateRF**" in table
    assert "FPS" in table
    assert "MB" in table


def test_generate_baseline_comparison_document(tmp_path: Path) -> None:
    """Verify document authoring outputs formal EXP-P1-001 content."""
    x, y, classes = compare_baselines.generate_synthetic_benchmark_data(
        num_classes=4,
        num_features=20,
        samples_per_class=10,
        seed=42,
    )
    model = RandomForestSignClassifier(n_estimators=10, n_jobs=1, random_state=42)
    model.fit(x, y, classes=classes)

    rep = compare_baselines.evaluate_candidate_model(
        model=model,
        name="ModelA",
        x_test=x,
        y_test=y,
        classes=classes,
        latency_trials=10,
    )

    out_file = tmp_path / "comparison.md"
    content = compare_baselines.generate_baseline_comparison_document(
        reports=[("ModelA", rep)],
        output_path=out_file,
    )
    assert out_file.exists()
    assert "SANKET Phase 1 — Comprehensive Baseline Model Comparison" in content
    assert "Research Question 1 (RQ1)" in content
    assert "Research Question 3 (RQ3)" in content
    assert "Phase 1 Deliverable Certification & Sign-off" in content


def test_compare_baselines_cli(tmp_path: Path) -> None:
    """Verify CLI tool executes end-to-end with custom output path."""
    out_file = tmp_path / "cli_comparison.md"
    exit_code = compare_baselines.main([
        "--output-report",
        str(out_file),
        "--latency-trials",
        "10",
        "--seed",
        "42",
    ])
    assert exit_code == 0
    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8")
    assert "Random Forest" in text
    assert "SVM (RBF Kernel)" in text
    assert "MLP Neural Net" in text


def test_adr_0003_content_integrity() -> None:
    """Verify ADR-0003 exists and codifies baseline classifier selection."""
    adr_path = Path("docs/decisions/ADR-0003-baseline-classifier-selection.md")
    assert adr_path.exists(), "ADR-0003 document must exist."
    content = adr_path.read_text(encoding="utf-8")
    assert "- **Status**: Accepted" in content
    assert "RandomForestSignClassifier" in content
    assert "SVMSignClassifier" in content
    assert "MLPSignClassifier" in content
    assert "Phase 2 Real-Time Camera MVP" in content
