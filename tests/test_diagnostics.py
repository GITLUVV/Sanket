"""Unit tests for confusion matrix diagnostics and phonological error analysis (SANKET-032)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sanket.evaluation import (
    ConfusablePair,
    ConfusionMatrixDiagnostics,
    ErrorBreakdown,
    PhonologicalErrorType,
    classify_phonological_error,
    compute_metrics,
)
from sanket.models import RandomForestSignClassifier
from scripts import analyze_errors


def test_phonological_error_types_exist() -> None:
    """Verify all 5 linguistic taxonomy categories are defined with correct designations."""
    expected_categories = {
        PhonologicalErrorType.HANDSHAPE: "Handshape (E_H)",
        PhonologicalErrorType.LOCATION: "Location (E_L)",
        PhonologicalErrorType.ORIENTATION: "Orientation (E_O)",
        PhonologicalErrorType.MOVEMENT: "Movement Trajectory (E_M)",
        PhonologicalErrorType.TRACKING: "Tracking / Occlusion (E_track)",
    }
    for err_enum, expected_val in expected_categories.items():
        assert err_enum.value == expected_val


def test_classify_phonological_error_known_pairs() -> None:
    """Verify linguistic error attribution on verified Phase 1 confusable pairs."""
    # Handshape errors
    err_type, explanation = classify_phonological_error("DOCTOR", "TIME")
    assert err_type == PhonologicalErrorType.HANDSHAPE
    assert "wrist" in explanation.lower()

    err_type, explanation = classify_phonological_error("WATER", "FOOD")
    assert err_type == PhonologicalErrorType.HANDSHAPE
    assert "mouth" in explanation.lower() or "lips" in explanation.lower()

    # Movement trajectory errors
    err_type, explanation = classify_phonological_error("HELLO", "PLEASE")
    assert err_type == PhonologicalErrorType.MOVEMENT
    assert "trajectory" in explanation.lower() or "static" in explanation.lower()

    # Orientation errors
    err_type, explanation = classify_phonological_error("GOOD", "BAD")
    assert err_type == PhonologicalErrorType.ORIENTATION
    assert "thumb" in explanation.lower() or "orientation" in explanation.lower()

    # Location errors
    err_type, explanation = classify_phonological_error("ME", "TIME")
    assert err_type == PhonologicalErrorType.LOCATION
    assert "wrist" in explanation.lower() or "chest" in explanation.lower()

    # Tracking / Occlusion errors
    err_type, explanation = classify_phonological_error("WHERE", "PLEASE")
    assert err_type == PhonologicalErrorType.TRACKING
    assert "handedness" in explanation.lower() or "detection" in explanation.lower()


def test_classify_phonological_error_heuristics() -> None:
    """Verify fallback heuristics for unseen class combinations."""
    # Identical signs
    err_type, explanation = classify_phonological_error("HELLO", "HELLO")
    assert err_type == PhonologicalErrorType.HANDSHAPE
    assert "correct" in explanation.lower()

    # Unseen classes not in vocabulary
    err_type, explanation = classify_phonological_error("UNKNOWN_A", "UNKNOWN_B")
    assert err_type == PhonologicalErrorType.HANDSHAPE
    assert "UNKNOWN_A" in explanation and "UNKNOWN_B" in explanation


def test_diagnostics_initialization_and_normalization() -> None:
    """Test raw matrix validation, row normalization, and zero row handling."""
    # 3x3 matrix: class 0 has 10 samples (8 correct, 2 confused with 1)
    # class 1 has 5 samples (5 correct)
    # class 2 has 0 samples (row sum = 0)
    cm_raw = np.array([
        [8, 2, 0],
        [0, 5, 0],
        [0, 0, 0],
    ])
    classes = ["CLASS_A", "CLASS_B", "CLASS_C"]

    diag = ConfusionMatrixDiagnostics(raw_matrix=cm_raw, classes=classes)
    assert diag.num_classes == 3
    assert diag.total_samples == 15
    assert diag.total_errors == 2
    assert pytest.approx(diag.overall_accuracy, rel=1e-4) == 13 / 15

    # Check row normalization
    assert pytest.approx(diag.normalized_matrix[0, 0], rel=1e-4) == 0.80
    assert pytest.approx(diag.normalized_matrix[0, 1], rel=1e-4) == 0.20
    assert pytest.approx(diag.normalized_matrix[1, 1], rel=1e-4) == 1.00
    # Zero row should not produce NaNs or infinities
    assert np.all(diag.normalized_matrix[2] == 0.0)


def test_diagnostics_invalid_shapes() -> None:
    """Ensure non-square matrices raise ValueError."""
    with pytest.raises(ValueError, match="Expected square 2D confusion matrix"):
        ConfusionMatrixDiagnostics(raw_matrix=np.ones((3, 4)))


def test_diagnostics_from_predictions() -> None:
    """Test factory constructor from y_true and y_pred."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    classes = ["SIGN_X", "SIGN_Y", "SIGN_Z"]

    diag = ConfusionMatrixDiagnostics.from_predictions(y_true, y_pred, classes=classes)
    assert diag.total_samples == 6
    assert diag.total_errors == 2
    assert diag.classes == classes

    # Test error cases
    with pytest.raises(ValueError, match="Cannot compute diagnostics on empty"):
        ConfusionMatrixDiagnostics.from_predictions([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        ConfusionMatrixDiagnostics.from_predictions([0, 1], [0])


def test_diagnostics_from_metrics() -> None:
    """Test factory constructor from EvaluationMetrics."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 1, 1, 2]
    metrics = compute_metrics(y_true, y_pred, classes=["A", "B", "C"])

    diag = ConfusionMatrixDiagnostics.from_metrics(metrics)
    assert diag.num_classes == 3
    assert diag.total_samples == 6
    assert diag.total_errors == 1


def test_top_confusable_pairs_ranking() -> None:
    """Test extracting off-diagonal misclassifications ranked by frequency."""
    cm = np.array([
        [10, 5, 2],
        [1, 12, 0],
        [4, 1, 15],
    ])
    classes = ["DOCTOR", "TIME", "WATER"]
    diag = ConfusionMatrixDiagnostics(raw_matrix=cm, classes=classes)

    pairs = diag.get_top_confusable_pairs(top_k=2)
    assert len(pairs) == 2
    assert isinstance(pairs[0], ConfusablePair)
    # Pair 1: DOCTOR -> TIME (5 errors)
    assert pairs[0].true_class == "DOCTOR"
    assert pairs[0].predicted_class == "TIME"
    assert pairs[0].error_count == 5
    assert pairs[0].error_type == PhonologicalErrorType.HANDSHAPE

    # Pair 2: WATER -> DOCTOR (4 errors)
    assert pairs[1].true_class == "WATER"
    assert pairs[1].predicted_class == "DOCTOR"
    assert pairs[1].error_count == 4

    # Test min_error_count filter
    high_thresh = diag.get_top_confusable_pairs(top_k=10, min_error_count=5)
    assert len(high_thresh) == 1
    assert high_thresh[0].error_count == 5


def test_phonological_breakdown() -> None:
    """Verify aggregation across 5 error categories."""
    cm = np.array([
        [20, 6, 0],  # DOCTOR -> TIME (6 handshape errors)
        [0, 20, 4],  # TIME -> WATER (4 location errors)
        [2, 0, 20],  # WATER -> DOCTOR (2 errors)
    ])
    classes = ["DOCTOR", "TIME", "WATER"]
    diag = ConfusionMatrixDiagnostics(raw_matrix=cm, classes=classes)

    breakdown = diag.get_phonological_breakdown()
    assert isinstance(breakdown, ErrorBreakdown)
    assert breakdown.total_errors == 12
    assert sum(breakdown.counts_by_type.values()) == 12
    assert pytest.approx(sum(breakdown.proportions_by_type.values()), rel=1e-4) == 1.0
    assert breakdown.top_category in breakdown.counts_by_type


def test_hardest_and_easiest_classes() -> None:
    """Verify extraction of lowest and highest recall classes."""
    cm = np.array([
        [10, 10],  # Class A: 10/20 = 50%
        [2, 18],   # Class B: 18/20 = 90%
    ])
    classes = ["DIFFICULT", "EASY"]
    diag = ConfusionMatrixDiagnostics(raw_matrix=cm, classes=classes)

    hardest = diag.get_hardest_classes(bottom_k=1)
    assert hardest[0][0] == "DIFFICULT"
    assert pytest.approx(hardest[0][1], rel=1e-4) == 0.50

    easiest = diag.get_easiest_classes(top_k=1)
    assert easiest[0][0] == "EASY"
    assert pytest.approx(easiest[0][1], rel=1e-4) == 0.90


def test_markdown_and_ascii_rendering(tmp_path: Path) -> None:
    """Test rendering of markdown tables, ASCII heatmap, and report generation."""
    cm = np.array([
        [8, 2],
        [1, 9],
    ])
    classes = ["HELLO", "THANK_YOU"]
    diag = ConfusionMatrixDiagnostics(raw_matrix=cm, classes=classes)

    md_table = diag.render_markdown_table()
    assert "| ID | Sign Gloss |" in md_table
    assert "`HELLO`" in md_table
    assert "`THANK_YOU`" in md_table

    ascii_art = diag.render_ascii_heatmap()
    assert "True \\ Pred" in ascii_art
    assert "HELLO" in ascii_art
    assert "(* indicates diagonal recall)" in ascii_art

    report = diag.generate_full_report(model_name="TestClassifier")
    assert "# SANKET Phase 1 — Linguistic Error Analysis & Diagnostics Report" in report
    assert "TestClassifier" in report
    assert "Phonological Linguistic Error Taxonomy Breakdown" in report

    out_file = tmp_path / "diagnostics_report.md"
    saved_path = diag.save_report(out_file, model_name="TestClassifier")
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == report

    # Test serialization to dict
    d = diag.to_dict()
    assert "overall_accuracy" in d
    assert "phonological_breakdown" in d
    assert "top_confusable_pairs" in d


def test_analyze_errors_cli_synthetic_demo(tmp_path: Path) -> None:
    """Test CLI script in self-contained demonstration mode."""
    report_file = tmp_path / "cli_report.md"
    exit_code = analyze_errors.main([
        "--synthetic-demo",
        "--output-report",
        str(report_file),
        "--top-k",
        "5",
    ])
    assert exit_code == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "SANKET Phase 1" in content
    assert "RandomForestSignClassifier" in content


def test_analyze_errors_cli_with_saved_model(tmp_path: Path) -> None:
    """Test CLI script loading a serialized model artifact and evaluation data."""
    # Generate and save model
    rng = np.random.RandomState(42)
    x = rng.randn(40, 20).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    classes = ["YES", "NO"]

    model = RandomForestSignClassifier(n_estimators=10, random_state=42)
    model.fit(x, y, classes=classes)
    model_path = tmp_path / "test_rf.joblib"
    model.save(model_path)

    # Save NPZ test split
    data_path = tmp_path / "test_data.npz"
    np.savez(data_path, x_test=x, y_test=y, classes=classes)

    report_file = tmp_path / "model_report.md"
    exit_code = analyze_errors.main([
        "--model-path",
        str(model_path),
        "--data-path",
        str(data_path),
        "--output-report",
        str(report_file),
    ])
    assert exit_code == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "RandomForestSignClassifier" in content
    assert "YES" in content and "NO" in content
