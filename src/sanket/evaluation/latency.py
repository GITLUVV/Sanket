"""Operational runtime and inference latency benchmarking for sign recognition models.

Implements latency and throughput benchmarks defined in docs/research/EVALUATION_METHODOLOGY.md:
- Inference Latency (T_infer): Median (p50) <= 15.0 ms, p95 <= 25.0 ms
- Percentiles: p50, p95, p99, mean, std, min, max
- Sustainable Throughput: Frames per second (FPS)
- Memory Footprint: Peak RSS memory tracking
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LatencyMetrics:
    """Quantitative runtime latency and throughput measurements."""

    median_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_fps: float
    num_trials: int
    peak_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize latency metrics to a JSON-compatible dictionary."""
        return {
            "median_ms": round(float(self.median_ms), 4),
            "p95_ms": round(float(self.p95_ms), 4),
            "p99_ms": round(float(self.p99_ms), 4),
            "mean_ms": round(float(self.mean_ms), 4),
            "std_ms": round(float(self.std_ms), 4),
            "min_ms": round(float(self.min_ms), 4),
            "max_ms": round(float(self.max_ms), 4),
            "throughput_fps": round(float(self.throughput_fps), 2),
            "num_trials": int(self.num_trials),
            "peak_memory_mb": (
                round(float(self.peak_memory_mb), 4) if self.peak_memory_mb is not None else None
            ),
        }


def profile_latency(
    predict_fn: Callable[[np.ndarray], Any],
    sample: np.ndarray,
    num_trials: int = 100,
    warm_up: int = 10,
    track_memory: bool = True,
) -> LatencyMetrics:
    """Profile inference latency and throughput across repeated forward passes.

    Args:
        predict_fn: Callable accepting a feature array and returning predictions.
        sample: Single evaluation feature vector of shape (D,) or (1, D).
        num_trials: Number of benchmark trials to execute after warmup.
        warm_up: Number of initial iterations discarded to prime CPU/caches.
        track_memory: If True, tracks peak memory allocated during profiling.

    Returns:
        LatencyMetrics container with timing percentiles and throughput.

    Raises:
        ValueError: If num_trials < 1 or sample is empty.
    """
    if num_trials < 1:
        msg = "num_trials must be at least 1."
        raise ValueError(msg)

    sample_arr = np.asarray(sample)
    if sample_arr.size == 0:
        msg = "Sample array cannot be empty."
        raise ValueError(msg)

    # Ensure 2D shape (1, D) for standard scikit-learn estimators
    if sample_arr.ndim == 1:
        sample_input = sample_arr.reshape(1, -1)
    else:
        sample_input = sample_arr[:1]

    # Warm-up phase
    for _ in range(max(0, warm_up)):
        predict_fn(sample_input)

    # Benchmarking phase with high-resolution timer
    latencies_ms: list[float] = []

    if track_memory:
        tracemalloc.start()

    for _ in range(num_trials):
        t0 = time.perf_counter_ns()
        predict_fn(sample_input)
        elapsed_ns = time.perf_counter_ns() - t0
        latencies_ms.append(elapsed_ns / 1_000_000.0)

    peak_memory_mb: float | None = None
    if track_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_bytes / (1024.0 * 1024.0)

    lat_arr = np.array(latencies_ms, dtype=float)
    p50, p95, p99 = np.percentile(lat_arr, [50, 95, 99])
    mean_val = float(np.mean(lat_arr))
    std_val = float(np.std(lat_arr))
    min_val = float(np.min(lat_arr))
    max_val = float(np.max(lat_arr))

    # Sustainable throughput FPS calculated from median inference latency
    throughput_fps = (1000.0 / p50) if p50 > 0 else 0.0

    return LatencyMetrics(
        median_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        mean_ms=mean_val,
        std_ms=std_val,
        min_ms=min_val,
        max_ms=max_val,
        throughput_fps=float(throughput_fps),
        num_trials=num_trials,
        peak_memory_mb=peak_memory_mb,
    )
