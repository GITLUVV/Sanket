# ADR-0003: Selection of Baseline Sign Classifier for Phase 2 Architecture & MVP Deployment

- **Status**: Accepted
- **Date**: 2026-11-01
- **Deciders**: Sanket Core AI, Vision & Governance Team
- **Related Issue**: SANKET-033 (Issue #35)

---

## Decision

1. Select **`RandomForestSignClassifier`** as the primary classification engine for the
   **Phase 2 Real-Time Camera MVP** (`SANKET-040` through `SANKET-048`).
2. Retain **`SVMSignClassifier`** (RBF kernel with standard feature scaling) as the high-margin
   secondary verification baseline for ambiguity resolution.
3. Establish **`MLPSignClassifier`** (3-layer feedforward architecture) as the foundational
   architectural bridge for Phase 3 deep sequence modeling (BiLSTM / Transformer encoder backbones).

---

## Context & Problem Statement

Phase 1 evaluated three classical and shallow neural machine learning architectures on the
16-sign constrained Indian Sign Language (ISL) vocabulary using identical 645-dimensional hybrid
feature representations (ADR-0002) and Signer-Independent partitions:
- **Random Forest (`RandomForestSignClassifier`)**: Gini-impurity tree ensemble with 100 estimators.
- **Support Vector Machine (`SVMSignClassifier`)**: Soft-margin classifier with RBF kernel and
  training-partition feature standardization.
- **Multi-Layer Perceptron (`MLPSignClassifier`)**: 3-layer neural network ($645 \to 256 \to 128 \to 64 \to 16$)
  with ReLU non-linearities and Adam weight optimization.

The system must satisfy strict edge-computing operational criteria codified in
`docs/research/EVALUATION_METHODOLOGY.md` Section 5:
- Inference forward-pass latency $T_{\text{infer}} \le 15.0\text{ ms}$ (median) and $\le 25.0\text{ ms}$ (p95).
- Sustainable throughput $\Phi \ge 25.0\text{ FPS}$ on single commodity CPU cores.
- Memory footprint $\le 200.0\text{ MB}$ RSS.
- Model serialized disk footprint $\le 30.0\text{ MB}$.

---

## Options Considered

### Option 1: Random Forest Classifier (`RandomForestSignClassifier`)
- **Pros**:
  - Extremely fast single-sample evaluation on CPU.
  - Scale-invariant: Does not strictly require feature standardization, making it resilient to
    sensor jitter or un-normalized landmark coordinate outliers.
  - Interpretable feature importances: Gini scores directly identify critical anatomical joints.
  - Zero gradient explosion/vanishing risks during real-time edge operation.
- **Cons**:
  - Step-wise axis-aligned decision boundaries can exhibit boundary jitter on continuous rotations.
  - Disk storage size grows linearly with number of trees (~8–12 MB).

### Option 2: Support Vector Machine with RBF Kernel (`SVMSignClassifier`)
- **Pros**:
  - Maximum-margin separation ensures optimal generalization when feature scale is normalized.
  - Platt probability calibration produces smooth posterior distributions.
  - Robust against high-dimensional collinearity.
- **Cons**:
  - Requires persistent `StandardScaler` pipeline during inference (data leakage risk if mismanaged).
  - Quadratic scaling with sample size during retraining.

### Option 3: Multi-Layer Perceptron (`MLPSignClassifier`)
- **Pros**:
  - Learns smooth non-linear latent manifolds across joint angle features.
  - Direct architectural compatibility with PyTorch/ONNX runtime graphs.
  - Natural stepping stone toward recurrent and attention-based sequence modeling (Phase 3).
- **Cons**:
  - Slightly higher inference latency than tree traversal.
  - Sensitive to weight initialization and requires careful feature scaling.

---

## Chosen Approach & Rationale

We selected **Option 1 (`RandomForestSignClassifier`)** as the primary edge deployment baseline,
complemented by **Option 2 (`SVMSignClassifier`)** for dual-inference verification, and
**Option 3 (`MLPSignClassifier`)** for deep sequence evolution.

### Primary Drivers:
1. **Latency Budget Headroom**: Random Forest inference executes well within our real-time budget,
   leaving $> 85\%$ of the 40 ms frame budget available for video capture, MediaPipe landmark
   extraction, coordinate normalization, and Generative UI rendering.
2. **Operational Robustness**: In production camera streams, signer movement occasionally produces
   incomplete landmarks or extreme scale ratios. Random Forest decision trees handle non-linear
   coordinate ranges gracefully without numerical overflow.
3. **Reproducibility**: Satisfies Research Questions RQ1 and RQ3, providing an empirical standard
   against which Phase 3 deep sequential models (BiLSTM / Transformer) will be measured.

---

## Consequences

- **Positive**:
  - Enables zero-latency local CPU inference in Phase 2 without requiring dedicated GPU acceleration.
  - Decouples UI development and camera frame streaming from heavy deep learning inference servers.
  - Preserves full architectural modularity via the standardized `BaseSignClassifier` interface.
- **Negative / Trade-offs**:
  - Temporal movement trajectory errors ($E_{\mathcal{M}}$) documented in SANKET-032 remain
    unresolved in isolated static frames, requiring temporal filtering in the Phase 2 buffer pipeline.
- **Risk Mitigation**:
  - Implement temporal sliding-window consensus voting in `sanket.vision` during Phase 2 to stabilize
    frame-by-frame predictions.
