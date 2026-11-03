# Testing & Benchmarking Rules

## Automated Test Requirements
- Maintain 100% test passing status across all test suites (`pytest -q`).
- Any new module in `src/sanket/` must have a corresponding test module in `tests/`.
- Repository test coverage must remain >= 80% overall, and 100% on mathematical transformations (`normalization.py`, `geometry.py`).

## Determinism & Fixtures
- All test fixtures generating random numbers or synthetic landmarks must specify an explicit seed: `np.random.default_rng(seed=42)`.
- Use `MockVideoCapture` for video testing in CI environments to avoid hardware camera dependencies.
- Do not commit multi-megabyte binary fixtures into version control.

## Latency & Performance Thresholds
- High-resolution timing must be performed using `time.perf_counter_ns()`.
- Always perform 50 warmup iterations before recording latency measurements.
- Real-time latency targets:
  - Feature extraction + inference: p95 <= 25 ms, Median <= 15 ms.
  - End-to-end pipeline: p95 < 100 ms.
  - Throughput: >= 20 FPS on CPU.

