# EXP-XXXX: [Short Title of Experiment]

- **Date**: YYYY-MM-DD
- **Related Issue**: SANKET-XXX
- **Author**: [Name]
- **Status**: [Completed | In-Progress | Abandoned]

---

## 1. Objective & Hypothesis
- **Objective**: What is this experiment designed to test or validate?
- **Hypothesis**: What expected behavior or accuracy improvement is anticipated?

---

## 2. Experimental Setup
- **Dataset / Version**: [e.g. INCLUDE-Subset-v1 from DATASET_REGISTER.md]
- **Classes / Vocabulary Size**: [e.g. 15 isolated signs]
- **Dataset Split**: [e.g. Signer-Independent: Signers 1-5 Train (70%), Signer 6 Val (15%), Signer 7 Test (15%)]
- **Feature Pipeline**:
  - Landmark Extractor: [e.g. MediaPipe Hands v0.10.x]
  - Landmarks: [e.g. 21 hand landmarks (x, y, z)]
  - Normalization: [e.g. Wrist-centered, bounding-box scale normalized]
  - Feature Dimension: [e.g. 63 float values per frame]
- **Model / Classifier**: [e.g. Random Forest, SVM with RBF kernel, MLP]
- **Hyperparameters**:
  - Random Seed: 42
  - Additional Parameters: [e.g. n_estimators=100, max_depth=15, C=1.0]

---

## 3. Quantitative Results

| Model / Run | Train Accuracy | Validation Accuracy | Test Top-1 Accuracy | Test Macro F1 | Latency (ms) |
| :---------- | :------------- | :------------------ | :------------------ | :------------ | :----------- |
| Baseline    |                |                     |                     |               |              |
| Candidate   |                |                     |                     |               |              |

---

## 4. Error Analysis & Confusion Matrix
- **Most Confused Sign Pairs**: (e.g., "Hello" vs "Thank You" due to similar open palm geometry)
- **Failure Modes Observed**: (e.g., partial finger occlusion, rapid motion blur)

---

## 5. Conclusions & Next Actions
- **Conclusion**: Was the hypothesis supported or refuted?
- **Decision**: Adopt into pipeline / retain baseline / iterate.
- **Next Experiment**: EXP-XXXX
