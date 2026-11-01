# EXP-0003: Baseline Multi-Layer Perceptron (MLP) Neural Network

- **Date**: 2026-11-01
- **Related Issue**: SANKET-030 (#32)
- **Author**: Anveshna
- **Status**: Completed

---

## 1. Objective & Hypothesis

- **Objective**: Establish a neural representation learning baseline (Multi-Layer Perceptron) to benchmark learned non-linear feature projections against tree ensembles (Random Forest, SANKET-028) and maximum-margin kernel methods (SVM, SANKET-029).
- **Hypothesis**: A 3-layer feedforward network with ReLU non-linearities, input feature standardization, and early stopping regularization will match or exceed tree ensemble accuracy while sustaining real-time inference latency ($\le 15.0\text{ ms}$) on commodity CPU environments.

---

## 2. Experimental Setup

- **Dataset / Version**: INCLUDE-Subset-v1 (16 isolated signs).
- **Classes / Vocabulary Size**: 16 vocabulary classes (`HELLO`, `THANK_YOU`, `YES`, `NO`, `PLEASE`, `WATER`, `FOOD`, `HELP`, `GOOD`, `BAD`, `FAMILY`, `FRIEND`, `WORK`, `NAME`, `WHERE`, `HOW`).
- **Dataset Split**: Signer-Independent Disjoint Split (70% Train, 15% Validation, 15% Test).
- **Feature Pipeline**:
  - Landmark Extractor: MediaPipe Hands (wrist-centered, anatomical scale normalized).
  - Feature Dimension: 645 float values per sequence (normalized landmark coordinates, pairwise distances, joint angles, and temporal pooling statistics).
  - Feature Preprocessing: `StandardScaler` fitted strictly on the training partition.
- **Model Architecture**:
  - Input Layer: 645 neurons.
  - Hidden Layer 1: 256 neurons + ReLU.
  - Hidden Layer 2: 128 neurons + ReLU.
  - Hidden Layer 3: 64 neurons + ReLU.
  - Output Layer: 16 neurons + Softmax.
  - Regularization: L2 weight penalty ($\alpha = 10^{-4}$), validation early stopping with 15-epoch patience.
  - Optimizer: Adam with adaptive learning rate scheduling ($\eta_0 = 10^{-3}$).
  - Random Seed: 42.

---

## 3. Quantitative Results

| Model / Run | Architecture | Alpha | Learning Rate | Latency (p50 ms) | Target Threshold (<= 15 ms) | Memory (RAM) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP Baseline** | 256-128-64 | 1e-4 | 1e-3 (adaptive) | ~1.8 ms | PASS | < 10 MB |

---

## 4. Key Findings

1. **Layered Representation Capacity**: The 3-layer architecture compresses the 645-dimensional feature space through successive abstractions ($645 \to 256 \to 128 \to 64 \to 16$), creating compact representations that separate phonological variants.
2. **Early Stopping Mitigates Overfitting**: Monitoring validation loss with 15-epoch patience halts training before over-parameterization memorizes idiosyncratic signer articulation nuances.
3. **High Operational Throughput**: With an inference latency median ($p50 \approx 1.8\text{ ms}$), the MLP baseline is over $8\times$ faster than the $15.0\text{ ms}$ budget, achieving $> 500\text{ FPS}$ sustained throughput on CPU.

---

## 5. Next Actions

- Proceed to SANKET-032: Execute confusion matrix analysis and phonological error diagnosis across baseline predictions.
- Proceed to SANKET-033: Conduct comprehensive comparative benchmark (Random Forest vs SVM vs MLP) and author ADR-0003.
