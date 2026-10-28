# Core Research Questions & Hypotheses — Sanket

This document outlines the scientific questions, theoretical assumptions, and empirical hypotheses guiding the research trajectory of **Sanket**.

---

## Research Question 1 (RQ1): Geometric Landmarks vs Raw Pixel CNNs

**Question**: *Can a compact geometric landmark representation (21 3D hand keypoints extracted via MediaPipe) achieve comparable or superior recognition accuracy on isolated Indian Sign Language signs compared to end-to-end 2D Convolutional Neural Networks, while operating under constrained compute budgets (< 50ms latency on CPU)?*

- **Hypothesis $H_1$**: Geometric landmark features will demonstrate equal or superior generalization compared to raw pixel CNNs when evaluated on unseen signers, because skeletal landmarks naturally isolate articulatory morphology and strip away spurious environmental distractors (background textures, skin tone, clothing, illumination).
- **Evaluation Criteria**: Test Top-1 Accuracy, Macro F1, Inference Latency (ms), Model Memory Footprint (MB).
- **Planned Verification**: Phase 1 baseline experiments (SANKET-028 through SANKET-033).

---

## Research Question 2 (RQ2): Geometric Normalization Resilience

**Question**: *What coordinate normalization algorithm provides the highest robustness against camera-distance variations and planar rotation without degrading distinguishing sign features?*

- **Hypothesis $H_2$**: Wrist-centered origin translation combined with palm-span scale normalization will yield higher recognition accuracy across varying camera distances (0.5m to 2.0m) than unnormalized screen-space pixel coordinates.
- **Evaluation Criteria**: Signer accuracy degradation across distance tiers (Near, Medium, Far).
- **Planned Verification**: Phase 1 landmark normalization (SANKET-026) and Phase 5 distance benchmark (SANKET-103).

---

## Research Question 3 (RQ3): Static Classifiers vs Temporal Recurrent Models

**Question**: *At what vocabulary scale and degree of dynamic movement do temporal recurrent models (LSTM / GRU) justify their added computational overhead over static single-frame classifiers for Indian Sign Language recognition?*

- **Hypothesis $H_3$**: Static classifiers (Random Forest / SVM) will achieve adequate accuracy (> 85%) on purely static ISL signs (e.g. manual alphabets and static numerals), but will degrade significantly (< 60%) on signs whose lexical identity depends on movement trajectory (e.g. "Come", "Go", "Help"), where sequence models will achieve a statistically significant margin of improvement (> 20% accuracy gain).
- **Evaluation Criteria**: Relative accuracy gain per sign category (static vs dynamic), FPS impact.
- **Planned Verification**: Phase 3 comparison study (SANKET-066).

---

## Research Question 4 (RQ4): Signer-Independent Generalization Bounds

**Question**: *How severe is the performance penalty when evaluating ISL recognition models on completely unseen native signers compared to random frame-level train/test splits?*

- **Hypothesis $H_4$**: Random frame-level splits will exhibit artificial performance inflation (reporting > 95% accuracy) due to temporal and visual leakage, whereas genuine Signer-Independent validation will reveal an initial 15–25% accuracy drop, establishing a realistic foundation for generalizable research.
- **Evaluation Criteria**: Signer-dependent vs Signer-independent cross-validation delta.
- **Planned Verification**: Phase 1 evaluation (SANKET-031) and Phase 5 generalization benchmark (SANKET-100).
