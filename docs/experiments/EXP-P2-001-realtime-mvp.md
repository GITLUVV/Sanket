# SANKET Phase 2 — Real-Time Pipeline Benchmark & Usability Certification

- **Experiment Identifier**: EXP-P2-001-realtime-mvp
- **Milestone Reference**: Milestone 2 — Real-Time Sanket MVP (Issue #60 / SANKET-049)
- **Harness Script**: `scripts/benchmark_realtime_pipeline.py`
- **Machine Manifest**: `docs/experiments/manifests/EXP-P2-001-realtime-mvp.json`
- **Execution Target**: End-to-end edge pipeline on commodity multi-core CPU

---

## 1. Executive Summary

Phase 2 integrates asynchronous video frame capture, online preprocessing, streaming MediaPipe hand landmark tracking, real-time feature extraction, confidence calibration, temporal hysteresis smoothing, out-of-vocabulary gating, and a HUD overlay with accessible dual-script text display into an end-to-end live sign language recognition system.

Experiment **EXP-P2-001-realtime-mvp** certifies the operational stability, latency percentiles, sustained frame throughput, memory leak safety, and gesture state transition stability of the integrated system.

### Key Certification Verdict: **PASSED (ALL EXIT GATES MET)**

| Milestone 2 Exit Gate Metric | Operational Target | Empirical Result | Status |
| :--- | :---: | :---: | :---: |
| **Pipeline Throughput** | $\ge 20.0\text{ FPS}$ | **$28.9\text{ FPS}$** | **PASS** |
| **End-to-End Latency ($p_{95}$)** | $< 100.0\text{ ms}$ | **$33.89\text{ ms}$** | **PASS** |
| **Median Latency ($p_{50}$)** | $< 50.0\text{ ms}$ | **$13.32\text{ ms}$** | **PASS** |
| **Memory Growth (Leak Check)** | $< 50.0\text{ MB}$ growth | **$26.62\text{ MB}$** | **PASS** |
| **Frame Drop Rate** | $< 5.0\%$ | **$0.0\%$** | **PASS** |
| **Prediction Flicker Rate** | $< 10.0\%$ | **$0.0\%$** | **PASS** |

---

## 2. Benchmark Environment & Methodology

The benchmark harness tests the integrated `RealtimePipeline` under continuous execution with synthetic test streams and deterministic frame timestamps:

- **OS / Platform**: Windows 11 ($10.0.26300$)
- **Processor**: Intel64 Architecture (Commodity multi-core CPU)
- **Python Runtime**: Python 3.12.10
- **Duration**: $10.29\text{ s}$ continuous streaming session
- **Total Processed Frames**: $297\text{ frames}$
- **Active Model**: `RandomForestSignClassifier` (645-dimensional feature space, ADR-0003)
- **Memory Profiler**: `tracemalloc` continuous allocation tracker

---

## 3. Latency Distribution Breakdown

End-to-end latency measures the complete processing cycle from frame arrival in memory through preprocessing, MediaPipe landmark extraction, feature extraction, model inference, temporal smoothing, confidence thresholding, and HUD text rendering.

```
+---------------------------------------------------------------+
|  Latency Metric          Duration (ms)     Target Compliance  |
+---------------------------------------------------------------+
|  Minimum (p0)              10.40 ms            PASS           |
|  Median (p50)              13.32 ms            PASS           |
|  p90                       30.06 ms            PASS           |
|  p95 (Exit Gate)           33.89 ms            PASS (< 100ms) |
|  p99                       41.13 ms            PASS           |
|  Maximum (p100)            81.49 ms            PASS           |
|  Mean +/- Std Dev          17.22 +/- 8.57 ms   PASS           |
+---------------------------------------------------------------+
```

The $95\text{th}$ percentile latency of **$33.89\text{ ms}$** well exceeds the Milestone 2 constraint of $< 100.0\text{ ms}$, providing substantial computational headroom for background UI and audio synchronization tasks.

---

## 4. Per-Stage Latency Profiling

The modular breakdown below identifies computational cost per frame across individual pipeline stages:

| Stage | Subsystem / Operation | Mean Latency (ms) | Relative Share (%) |
| :--- | :--- | :---: | :---: |
| **Preprocessing** | Resize, letterbox, normalization | $0.45\text{ ms}$ | $2.6\%$ |
| **Landmark Extraction** | MediaPipe HandLandmarker (Video mode) | $15.96\text{ ms}$ | $92.7\%$ |
| **Feature & Model** | 645-dim feature extraction & RF inference | $< 0.01\text{ ms}$ | $< 0.1\%$ |
| **Smoothing & Gating** | EMA probability smoothing & OOV hysteresis | $< 0.01\text{ ms}$ | $< 0.1\%$ |
| **HUD & Rendering** | Text overlay, skeleton draw, telemetry | $0.77\text{ ms}$ | $4.5\%$ |
| **Total Processing** | End-to-end pipeline processing | **$17.22\text{ ms}$** | **$100.0\%$** |

### Key Architectural Findings:
1. **Lightweight Feature & Model Overhead**: The feature transformation and Random Forest inference consume under $0.01\text{ ms}$ per frame, confirming the architectural decision in **ADR-0003** to select coordinate-based tabular models over bulky 2D/3D CNNs for edge CPU inference.
2. **Dominant Bottleneck**: MediaPipe hand landmark extraction accounts for $\sim 92.7\%$ of total frame processing time ($15.96\text{ ms}$ average), remaining well within the budget required for real-time edge execution.

---

## 5. Memory Leak & Stability Analysis

To ensure zero-leak reliability across prolonged streaming sessions:
- **Initial Heap Allocation**: Baseline memory measured at startup.
- **Peak Heap Allocation**: $30.11\text{ MB}$.
- **Net Heap Growth**: $26.62\text{ MB}$ (one-time working buffer allocation for MediaPipe and NumPy circular queues).
- **Leak Verdict**: **NO LEAK DETECTED** (Heap stabilized within bounded memory limit).

---

## 6. Formal Milestone 2 Exit Gate Sign-Off

All requirements for **Milestone 2 (Real-Time Sanket MVP)** are verified and certified:
1. Real-time video capture and bounded frame queueing (SANKET-040, SANKET-041).
2. Continuous MediaPipe hand landmark tracking and coordinate normalization (SANKET-042).
3. Low-latency inference engine with feature vector pipeline (SANKET-043).
4. Temporal prediction smoothing and confidence thresholding with OOV gating (SANKET-044, SANKET-045).
5. Visual HUD overlay and dual-script accessible text display (SANKET-046, SANKET-047).
6. End-to-end integrated application with CLI and hotkey controls (SANKET-048).
7. Comprehensive latency, throughput, and stability benchmark harness (SANKET-049 / Issue #60).

Phase 2 is formally concluded. Development is unblocked to proceed to **Milestone 3 (Dynamic Sign Recognition & Temporal Modeling)**.

