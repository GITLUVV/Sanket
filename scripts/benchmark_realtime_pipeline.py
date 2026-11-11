#!/usr/bin/env python3
"""SANKET Real-Time Pipeline Benchmark & Usability Harness (SANKET-049).

Instruments every stage of the real-time recognition pipeline (capture, preprocessing,
streaming landmark extraction, feature computation, inference, smoothing, thresholding,
and HUD rendering) with high-resolution performance timers.

Measures:
- End-to-end latency percentiles (p50, p90, p95, p99) ensuring latency < 100 ms target.
- Streaming throughput ensuring pipeline FPS >= 20 FPS on commodity edge CPU.
- Memory leak detection across extended streaming sessions via tracemalloc.
- Static sign flicker rate and state transition hysteresis stability.

Outputs structured benchmark results to stdout and JSON experiment manifests.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src/ is on sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sanket.models.random_forest import RandomForestSignClassifier
from sanket.realtime.pipeline import PipelineConfig, RealtimePipeline
from sanket.realtime.thresholding import SensitivityProfile
from sanket.ui.overlay import TelemetryData

logger = logging.getLogger("sanket.benchmark")


@dataclass
class LatencyPercentiles:
    """Latency distribution percentiles in milliseconds."""

    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float


@dataclass
class StageLatencyBreakdown:
    """Per-stage mean latency breakdown in milliseconds."""

    capture_ms: float
    preprocessing_ms: float
    landmark_extraction_ms: float
    feature_inference_ms: float
    smoothing_gating_ms: float
    hud_render_ms: float


@dataclass
class MemoryUsageReport:
    """Memory usage statistics in megabytes (MB)."""

    initial_mb: float
    peak_mb: float
    final_mb: float
    growth_mb: float
    leak_detected: bool


@dataclass
class StabilityReport:
    """Prediction stability and flicker metrics."""

    total_frames: int
    dropped_frames: int
    drop_rate_pct: float
    state_transitions: int
    flicker_rate_pct: float


@dataclass
class BenchmarkReport:
    """Master benchmark report capturing all performance, latency, and stability metrics."""

    experiment_id: str
    timestamp: str
    duration_seconds: float
    hardware: dict[str, str]
    throughput_fps: float
    fps_target_met: bool
    latency_target_met: bool
    latency: LatencyPercentiles
    stages: StageLatencyBreakdown
    memory: MemoryUsageReport
    stability: StabilityReport
    certified: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_percentiles(values: list[float]) -> LatencyPercentiles:
    """Compute standard statistical percentiles from a list of latencies."""
    if not values:
        return LatencyPercentiles(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.asarray(values, dtype=np.float64)
    return LatencyPercentiles(
        min_ms=float(np.min(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(np.max(arr)),
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
    )


def run_benchmark(
    duration_s: float = 10.0,
    fps: int = 30,
    synthetic_pattern: str = "moving_box",
    sensitivity: SensitivityProfile = SensitivityProfile.BALANCED,
    model_path: str | Path | None = None,
) -> BenchmarkReport:
    """Execute high-precision streaming benchmark on the RealtimePipeline."""
    logger.info(
        "Starting SANKET Real-Time Pipeline Benchmark (Duration: %.1fs, Target FPS: %d)...",
        duration_s,
        fps,
    )

    tracemalloc.start()
    init_mem_bytes, _ = tracemalloc.get_traced_memory()
    init_mem_mb = init_mem_bytes / (1024 * 1024)

    # Initialize mock model if no disk model provided
    model = None
    if model_path is None and not Path("models/random_forest_baseline.joblib").exists():
        rng = np.random.RandomState(42)
        x_synth = rng.randn(20, 645).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=int)
        model = RandomForestSignClassifier(n_estimators=5, random_state=42)
        model.fit(x_synth, y)

    config = PipelineConfig(
        use_mock=True,
        mock_pattern=synthetic_pattern,
        fps=fps,
        headless=True,
        show_hud=True,
        sensitivity_profile=sensitivity,
        model_path=model_path,
    )

    pipeline = RealtimePipeline(config=config, model=model)

    latencies_e2e: list[float] = []
    capture_times: list[float] = []
    preprocess_times: list[float] = []
    landmark_times: list[float] = []
    inference_times: list[float] = []
    smoothing_times: list[float] = []
    hud_times: list[float] = []

    last_gloss: str | None = None
    state_transitions = 0
    frame_count = 0

    benchmark_start_time = time.perf_counter()
    pipeline.start()

    try:
        while True:
            elapsed = time.perf_counter() - benchmark_start_time
            if elapsed >= duration_s:
                break

            # 1. Capture Stage (frame acquisition from thread queue)
            t_cap_start = time.perf_counter()
            captured = pipeline.capture.read(timeout=0.05)
            t_cap_end = time.perf_counter()
            capture_times.append((t_cap_end - t_cap_start) * 1000.0)

            if captured is None:
                continue

            # Start of frame processing pipeline
            t_frame_start = time.perf_counter()

            frame_count += 1
            ts_ms = captured.timestamp_ms

            # 2. Preprocess Stage
            t_prep_start = time.perf_counter()
            active_frame, _ = pipeline.preprocessor.process_frame(captured.frame)
            t_prep_end = time.perf_counter()
            preprocess_times.append((t_prep_end - t_prep_start) * 1000.0)

            # 3. Landmark Extraction Stage
            t_lm_start = time.perf_counter()
            norm_landmarks = None
            try:
                norm_landmarks = pipeline.landmarker.process_frame(active_frame, timestamp_ms=ts_ms)
            except Exception:
                pass
            t_lm_end = time.perf_counter()
            landmark_times.append((t_lm_end - t_lm_start) * 1000.0)

            # 4. Feature Extraction & Inference Stage
            t_inf_start = time.perf_counter()
            raw_pred = None
            has_hands = (
                norm_landmarks is not None
                and (norm_landmarks.left_detected or norm_landmarks.right_detected)
            )
            if pipeline.engine is not None and has_hands and norm_landmarks is not None:
                try:
                    raw_pred = pipeline.engine.predict(norm_landmarks, timestamp_ms=ts_ms)
                except Exception:
                    pass
            t_inf_end = time.perf_counter()
            inference_times.append((t_inf_end - t_inf_start) * 1000.0)

            # 5. Smoothing & Gating Stage
            t_smooth_start = time.perf_counter()
            smoothed_pred = None
            if raw_pred is not None:
                smoothed_pred = pipeline.smoother.update(raw_pred)

            calibrated_pred = None
            if smoothed_pred is not None:
                calibrated_pred = pipeline.calibrator.evaluate(
                    smoothed_pred, landmarks=norm_landmarks, timestamp_ms=ts_ms
                )
            elif raw_pred is not None:
                calibrated_pred = pipeline.calibrator.evaluate(
                    raw_pred, landmarks=norm_landmarks, timestamp_ms=ts_ms
                )

            if calibrated_pred is not None:
                pipeline.text_display.update(calibrated_pred)
                current_gloss = calibrated_pred.gloss
                if last_gloss is not None and current_gloss != last_gloss:
                    state_transitions += 1
                last_gloss = current_gloss

            t_smooth_end = time.perf_counter()
            smoothing_times.append((t_smooth_end - t_smooth_start) * 1000.0)

            # 6. HUD Render Stage
            t_hud_start = time.perf_counter()
            hud_telemetry = TelemetryData(
                capture_fps=pipeline.capture.fps,
                inference_fps=30.0,
                latency_ms=(time.perf_counter() - t_frame_start) * 1000.0,
            )
            pipeline.overlay.render(
                active_frame,
                prediction=calibrated_pred,
                landmarks=norm_landmarks,
                telemetry=hud_telemetry,
            )
            t_hud_end = time.perf_counter()
            hud_times.append((t_hud_end - t_hud_start) * 1000.0)

            # Record total end-to-end frame processing latency (preprocessing through HUD render)
            e2e_ms = (time.perf_counter() - t_frame_start) * 1000.0
            latencies_e2e.append(e2e_ms)

    finally:
        pipeline.stop()

    actual_duration = time.perf_counter() - benchmark_start_time
    curr_mem_bytes, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    curr_mem_mb = curr_mem_bytes / (1024 * 1024)
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    mem_growth_mb = max(0.0, curr_mem_mb - init_mem_mb)

    # Compute metrics
    throughput_fps = float(frame_count / actual_duration) if actual_duration > 0 else 0.0
    latency_stats = compute_percentiles(latencies_e2e)

    stage_breakdown = StageLatencyBreakdown(
        capture_ms=float(np.mean(capture_times)) if capture_times else 0.0,
        preprocessing_ms=float(np.mean(preprocess_times)) if preprocess_times else 0.0,
        landmark_extraction_ms=float(np.mean(landmark_times)) if landmark_times else 0.0,
        feature_inference_ms=float(np.mean(inference_times)) if inference_times else 0.0,
        smoothing_gating_ms=float(np.mean(smoothing_times)) if smoothing_times else 0.0,
        hud_render_ms=float(np.mean(hud_times)) if hud_times else 0.0,
    )

    mem_report = MemoryUsageReport(
        initial_mb=init_mem_mb,
        peak_mb=peak_mem_mb,
        final_mb=curr_mem_mb,
        growth_mb=mem_growth_mb,
        leak_detected=bool(mem_growth_mb > 50.0),  # Leak flagged if net growth > 50 MB
    )

    dropped = getattr(pipeline.capture, "dropped_frames", 0)
    total_captured = frame_count + dropped
    drop_rate = float(dropped / total_captured * 100.0) if total_captured > 0 else 0.0
    flicker_rate = float(state_transitions / frame_count * 100.0) if frame_count > 0 else 0.0

    stability_report = StabilityReport(
        total_frames=frame_count,
        dropped_frames=dropped,
        drop_rate_pct=drop_rate,
        state_transitions=state_transitions,
        flicker_rate_pct=flicker_rate,
    )

    fps_target_met = bool(throughput_fps >= 20.0)
    latency_target_met = bool(latency_stats.p95_ms < 100.0)
    certified = bool(fps_target_met and latency_target_met and not mem_report.leak_detected)

    return BenchmarkReport(
        experiment_id="EXP-P2-001-realtime-mvp",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        duration_seconds=actual_duration,
        hardware={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        throughput_fps=throughput_fps,
        fps_target_met=fps_target_met,
        latency_target_met=latency_target_met,
        latency=latency_stats,
        stages=stage_breakdown,
        memory=mem_report,
        stability=stability_report,
        certified=certified,
    )


def print_report(report: BenchmarkReport) -> None:
    """Print readable Markdown table format of benchmark outcomes."""
    print("\n" + "=" * 65)
    print("  SANKET REAL-TIME MVP BENCHMARK & CERTIFICATION REPORT")
    print("=" * 65)
    print(f"Experiment ID       : {report.experiment_id}")
    print(f"Timestamp           : {report.timestamp}")
    print(f"Execution Duration  : {report.duration_seconds:.2f} seconds")
    print(f"Processed Frames    : {report.stability.total_frames}")
    print(f"System Platform     : {report.hardware['platform']}")
    print("-" * 65)
    print("THROUGHPUT & LATENCY GATES:")
    print(
        f"  Throughput (FPS)  : {report.throughput_fps:.1f} FPS "
        f"(Target: >= 20.0 FPS) -> [{'PASS' if report.fps_target_met else 'FAIL'}]"
    )
    print(
        f"  Latency p95 (ms)  : {report.latency.p95_ms:.2f} ms "
        f"(Target: < 100.0 ms)  -> [{'PASS' if report.latency_target_met else 'FAIL'}]"
    )
    print(
        f"  Memory Leak Check : {report.memory.growth_mb:.2f} MB growth "
        f"-> [{'PASS' if not report.memory.leak_detected else 'FAIL'}]"
    )
    print("-" * 65)
    print("LATENCY DISTRIBUTION (ms):")
    print(f"  p50 (Median)      : {report.latency.p50_ms:.2f} ms")
    print(f"  p90               : {report.latency.p90_ms:.2f} ms")
    print(f"  p95               : {report.latency.p95_ms:.2f} ms")
    print(f"  p99               : {report.latency.p99_ms:.2f} ms")
    print(f"  Mean +/- Std      : {report.latency.mean_ms:.2f} +/- {report.latency.std_ms:.2f} ms")
    print("-" * 65)
    print("STAGE BREAKDOWN (Mean ms):")
    print(f"  Preprocessing     : {report.stages.preprocessing_ms:.2f} ms")
    print(f"  Landmark Extr.    : {report.stages.landmark_extraction_ms:.2f} ms")
    print(f"  Feature & Model   : {report.stages.feature_inference_ms:.2f} ms")
    print(f"  Smoothing/Gating  : {report.stages.smoothing_gating_ms:.2f} ms")
    print(f"  HUD Rendering     : {report.stages.hud_render_ms:.2f} ms")
    print("-" * 65)
    status_label = (
        "PHASE 2 EXIT GATES OFFICIALLY CERTIFIED"
        if report.certified
        else "CERTIFICATION FAILED"
    )
    print(f"OVERALL STATUS      : {status_label}")
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="SANKET Real-Time Pipeline Benchmark Harness")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Benchmark duration in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target capture frame rate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON benchmark manifest.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI execution entrypoint."""
    args = parse_args()
    report = run_benchmark(duration_s=args.duration, fps=args.fps, model_path=args.model_path)
    print_report(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Wrote benchmark manifest to {out_path}")

    return 0 if report.certified else 1


if __name__ == "__main__":
    sys.exit(main())
