# Evaluation Methodology

Refer to `docs/research/EVALUATION_METHODOLOGY.md` for complete mathematical definitions.

## Primary Classification Metrics
- **Top-1 Accuracy**: Fraction of samples where the highest confidence prediction matches the ground truth.
- **Top-5 Accuracy**: Fraction of samples where the ground truth is within the top 5 predictions.
- **Macro-Averaged Precision & Recall**: Unweighted average across all classes, treating rare and frequent signs equally.
- **Macro-Averaged F1 Score**: Primary model ranking criterion:
  $$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Matthews Correlation Coefficient (MCC)**: Multi-class correlation metric robust to label imbalance.

## Computational Runtime Metrics
- **Inference Latency**: Evaluated per sample with 50-run warm-up: p50 (median), p95, and p99 percentiles in milliseconds.
- **Throughput**: Frames/samples processed per second (FPS).
- **Memory Footprint**: Serialized model artifact size on disk (MB).
