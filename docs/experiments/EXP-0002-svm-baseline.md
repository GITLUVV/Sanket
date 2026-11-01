# EXP-0002: Baseline Support Vector Machine (SVM) Classifier

- **Date**: 2026-11-01
- **Related Issue**: SANKET-029 (#31)
- **Author**: Anveshna
- **Status**: Completed

---

## 1. Objective & Hypothesis

- **Objective**: Establish a maximum-margin kernel baseline classifier to evaluate linear versus non-linear separability of the 645-dimensional normalized skeletal feature vectors.
- **Hypothesis**: The non-linear Radial Basis Function (RBF) kernel with feature standardization ($z$-score scaling) will yield high discriminative margins for isolated Indian Sign Language recognition while maintaining sub-15 ms CPU inference latency on commodity hardware.

---

## 2. Experimental Setup

- **Dataset / Version**: INCLUDE-Subset-v1 (16 isolated signs).
- **Classes / Vocabulary Size**: 16 vocabulary classes (`HELLO`, `THANK_YOU`, `YES`, `NO`, `PLEASE`, `WATER`, `FOOD`, `HELP`, `GOOD`, `BAD`, `FAMILY`, `FRIEND`, `WORK`, `NAME`, `WHERE`, `HOW`).
- **Dataset Split**: Signer-Independent Disjoint Split (70% Train, 15% Validation, 15% Test).
- **Feature Pipeline**:
  - Landmark Extractor: MediaPipe Hands (wrist-centered, anatomical scale normalized).
  - Feature Dimension: 645 float values per sequence (normalized landmark coordinates, pairwise distances, joint angles, and temporal pooling statistics).
  - Feature Preprocessing: `StandardScaler` fitted strictly on the training partition to prevent biometric data leakage.
- **Model / Classifier**: `SVMSignClassifier` (`sklearn.svm.SVC`) with Platt probability scaling (`probability=True`).
- **Hyperparameters**:
  - Regularization parameter $C$: 10.0 (tuned via validation grid search over $[0.1, 1.0, 10.0, 100.0]$).
  - Kernel: RBF (`kernel="rbf"`).
  - Kernel coefficient $\gamma$: `scale` ($\frac{1}{D \cdot \text{Var}(X)}$).
  - Class weighting: `balanced` (mitigating class articulation frequency imbalance).
  - Random Seed: 42.

---

## 3. Quantitative Results

| Model / Run | Kernel | C | Scaling | Latency (p50 ms) | Target Threshold (<= 15 ms) | Memory (RAM) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM RBF Baseline** | RBF | 10.0 | StandardScaler | ~3.8 ms | PASS | < 5 MB |

---

## 4. Key Findings

1. **StandardScaler Preprocessing is Essential**: Unscaled landmark features contain widely varying ranges (angles in $[0, \pi]$ vs coordinates in $[-1, 1]$ vs temporal pooling variances). Standardizing features strictly on the training partition prevents dominant features from biasing the RBF kernel distance metric.
2. **Maximum Margin Robustness**: The RBF kernel provides effective non-linear decision boundaries for subtle hand configurations that distinguish sign pairs with identical palm orientations.
3. **Operational Latency**: Single-sample CPU inference forward pass latency median ($p50 \approx 3.8\text{ ms}$) satisfies the operational latency requirement ($\le 15.0\text{ ms}$) defined in `docs/research/EVALUATION_METHODOLOGY.md`.

---

## 5. Next Actions

- Proceed to SANKET-030: Implement Multi-Layer Perceptron (`MLPSignClassifier`) neural network baseline.
- Proceed to SANKET-032: Execute detailed error analysis and phonological confusion matrix investigation.
