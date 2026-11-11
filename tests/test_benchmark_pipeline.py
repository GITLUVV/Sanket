"""Unit tests for the real-time MVP benchmark harness (SANKET-049)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_realtime_pipeline import (
    BenchmarkReport,
    compute_percentiles,
    print_report,
    run_benchmark,
)


def test_compute_percentiles_empty() -> None:
    """Empty list returns zeroed percentiles container."""
    p = compute_percentiles([])
    assert p.min_ms == 0.0
    assert p.p50_ms == 0.0
    assert p.p95_ms == 0.0
    assert p.mean_ms == 0.0


def test_compute_percentiles_values() -> None:
    """Percentile calculation matches expected statistics."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    p = compute_percentiles(values)
    assert p.min_ms == 10.0
    assert p.max_ms == 50.0
    assert p.p50_ms == 30.0
    assert p.mean_ms == 30.0
    assert np.isclose(p.p95_ms, 48.0)


def test_benchmark_short_run_generates_report(tmp_path: Path) -> None:
    """Test that run_benchmark executes headless for 1 second and produces valid report."""
    report = run_benchmark(duration_s=1.0, fps=30)
    assert isinstance(report, BenchmarkReport)
    assert report.stability.total_frames > 0
    assert report.throughput_fps > 0.0
    assert report.latency.p95_ms > 0.0

    # Ensure print_report executes cleanly
    print_report(report)

    # Test JSON serialization
    from dataclasses import asdict

    out_file = tmp_path / "test_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f)

    assert out_file.exists()
    assert out_file.stat().st_size > 0

