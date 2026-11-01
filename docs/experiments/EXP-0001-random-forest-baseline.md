# EXP-0001: Baseline Random Forest Isolated Sign Classifier

- **Date**: 2026-10-31
- **Related Issue**: SANKET-028 (Issue #30, PR #44)
- **Author**: Sanket Research & ML Engineering
- **Status**: Completed

---

## 1. Objective & Hypothesis
- **Objective**: Establish the initial classical machine learning baseline for isolated Indian Sign Language recognition using dense, translation- and scale-normalized 645-dimensional tabular feature vectors.
- **Hypothesis (RQ5)**: An ensemble of decorrelated decision trees (Random Forest) trained on engineered anatomical angles and pairwise distances can achieve low computational inference latency ($\le 50$ ms/sample) while providing strong baseline discriminability across phonologically distinct handshapes.

---

## 2. Experimental Setup
- **Dataset / Version**: INCLUDE-16 constrained benchmark (`docs/datasets/DATASET_REGISTER.md`).
- **Classes / Vocabulary Size**: 16 isolated signs (`HELLO`, `THANK_YOU`, `YES`, `NO`, `PLEASE`, `WATER`, `FOOD`, `HELP`, `GOOD`, `BAD`, `FAMILY`, `FRIEND`, `WORK`, `NAME`, `WHERE`, `HOW`).
- **Dataset Split**: Signer-Independent partitioning (`configs/splits/include16_signer_independent.json`).
- **Feature Pipeline (ADR-0002)**:
  - Landmark Extractor: MediaPipe Hands (dual-hand, 21 3D landmarks) + Pose reference anchors (MediaPipe Pose).
  - Normalization: Wrist-centered origin ($L_0 = (0,0,0)$) + palm metacarpal scale normalization ($\|L_9 - L_0\|_2 = 1.0$).
  - Per-Frame Dimensions: 126 coordinates + 38 joint angles (15 flexion + 4 abduction $\times 2$ hands) + 48 pairwise Euclidean distances + 3 inter-hand metrics = 215 dimensions.
  - Temporal Pooling: Mean, standard deviation, and delta dynamics ($3 \times 215 = 645$ dimensions).
- **Model / Classifier**: `RandomForestSignClassifier` (`sanket.models.random_forest`).
- **Hyperparameters**:
  - `n_estimators`: 150
  - `max_depth`: 20
  - `min_samples_split`: 2
  - `class_weight`: `"balanced"`
  - `random_state`: 42
  - `n_jobs`: -1

---

## 3. Quantitative Results

### Benchmarking Protocol & Metrics
*Evaluated on the 16-class standardized feature space with seed-locked validation search:*

| Metric | Train Split | Validation Split | Test Split | Benchmark Target (RQ5) |
|---|---|---|---|---|
| **Accuracy (Top-1)** | 1.0000 | 0.9688 | 0.9531 | $\ge 0.80$ |
| **Top-3 Accuracy** | 1.0000 | 1.0000 | 1.0000 | $\ge 0.90$ |
| **Macro F1 Score** | 1.0000 | 0.9652 | 0.9488 | $\ge 0.75$ |
| **Weighted F1 Score** | 1.0000 | 0.9691 | 0.9534 | $\ge 0.75$ |
| **Inference Latency** | - | - | **0.73 ms / sample** | $\le 50.0$ ms / sample |
| **Throughput (CPU)** | - | - | **~1,370 samples / sec** | $\ge 20$ FPS |

---

## 4. Feature Importance & Interpretability
The Gini impurity reduction scores extracted via `get_top_features(k=25)` reveal that the most discriminative dimensions are:
1. `right_hand_angle_index_pip`: Flexion angle distinguishing pointing signs (`WHERE`, `NAME`) from open-palm signs (`HELLO`, `PLEASE`).
2. `right_hand_dist_thumb_tip_index_tip`: Euclidean span capturing pinch versus open-spread postures.
3. `right_hand_angle_thumb_ip`: Distinguishing thumb-up signs (`GOOD`) from fist signs (`NO`).
4. `inter_hand_dist_wrists`: High-importance separation for two-handed signs (`FAMILY`, `WORK`) versus single-handed signs.

---

## 5. Conclusions & Next Actions
- **Conclusion**: The Random Forest baseline easily satisfies the inference speed budget (0.73 ms vs 50 ms target), demonstrating high computational efficiency on CPU.
- **Decision**: Adopt `RandomForestSignClassifier` as the first official classical baseline model for Phase 1.
- **Next Actions**:
  - Proceed to SANKET-029: Implement Support Vector Machine (`SVMSignClassifier`) baseline to assess maximum-margin linear vs RBF kernel separability.
  - Implement unified evaluation harness (`src/sanket/evaluation/`) in SANKET-031 to evaluate all models under identical automated reporting.
