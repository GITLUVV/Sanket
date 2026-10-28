# Core Research Questions & Hypotheses — Sanket

This document outlines the formal scientific research questions, theoretical foundations, mathematical formulations, and empirical hypotheses guiding the research trajectory of **Sanket — AI-Assisted Indian Sign Language Accessibility System**.

---

## 1. Theoretical Framing & Problem Context

Indian Sign Language (ISL) is a rich, natural visuo-spatial language characterized by simultaneous multi-channel articulation. Following formal linguistic sign models (Stokoe, 1960; Battison, 1978), every ISL sign is composed of five phonological parameters:
1. **Handshape ($\mathcal{H}$)**: Specific geometric configuration and flexion/extension of fingers.
2. **Location ($\mathcal{L}$)**: Spatial position relative to the signer's body (neutral signing space, face, chest, torso, arm).
3. **Orientation ($\mathcal{O}$)**: 3D vector orientation of the palm and metacarpals relative to the signer and observer.
4. **Movement ($\mathcal{M}$)**: Dynamic spatiotemporal trajectory of hands (static hold, linear translation, circular arc, oscillating wag, sudden stop).
5. **Non-Manual Features ($\mathcal{N}$)**: Head orientation, facial expressions, eye gaze, and mouth gestures.

Computational recognition of ISL faces significant challenges:
- High intra-signer speed variability and inter-signer physical variations (hand dimensions, signing space scale).
- Complex hand-hand and hand-body self-occlusions during two-handed interactions.
- Computational constraints of edge-deployable accessibility devices (laptops, mobile phones) requiring real-time operation ($\ge 25$ FPS, latency $\le 50$ ms) without high-end dedicated GPU clusters.

To prevent ad-hoc trial-and-error engineering, Sanket structures all experimental phases around five verifiable research questions and empirical hypotheses.

---

## 2. Formal Research Questions & Hypotheses

### Research Question 1 (RQ1): Representation Fidelity — Skeletal Coordinate Tensors vs End-to-End Pixels

**Question**: *Can a compact geometric skeletal representation ($21 \times 3$ coordinates per hand extracted via MediaPipe) achieve comparable or superior recognition accuracy on isolated ISL signs compared to end-to-end 2D Convolutional Neural Networks on raw pixel frames, while operating within edge CPU compute budgets?*

#### Mathematical Formulation
Let raw input video be $V = \{I_1, I_2, \dots, I_T\}$ where $I_t \in \mathbb{R}^{H \times W \times 3}$.  
Let geometric feature extractor $f_{\text{MP}}: I_t \mapsto X_t \in \mathbb{R}^{K \times 3}$ map each frame to $K=21$ skeletal landmarks per hand (plus face/pose anchors).  
Let $M_{\text{CNN}}: V \to \hat{Y}$ be an end-to-end pixel baseline (e.g. MobileNetV3 / ResNet-18) with parameter count $|\theta_{\text{CNN}}| \sim 10^7$.  
Let $M_{\text{geom}}: \{X_1, \dots, X_T\} \to \hat{Y}$ be a landmark-based classifier with parameter count $|\theta_{\text{geom}}| \sim 10^4 - 10^5$.

- **Null Hypothesis ($H_{0,1}$)**: On a held-out Signer-Independent test partition $\mathcal{D}_{\text{test}}$, the skeletal representation model $M_{\text{geom}}$ produces a Top-1 recognition accuracy statistically inferior or indistinguishable from the raw pixel CNN model $M_{\text{CNN}}$ ($p \ge 0.05$ under a paired two-tailed Wilcoxon signed-rank test across sign categories).
$$\mathbb{E}[\text{Acc}(M_{\text{geom}})] \le \mathbb{E}[\text{Acc}(M_{\text{CNN}})]$$

- **Alternative Hypothesis ($H_{1,1}$)**: The geometric skeletal representation $M_{\text{geom}}$ achieves Top-1 recognition accuracy within $\pm 2.5\%$ of, or superior to, $M_{\text{CNN}}$ on unseen signers, while reducing per-frame inference latency by at least an order of magnitude ($\ge 10\times$ speedup on CPU) and reducing model parameter memory footprint by $> 90\%$.
$$\text{Acc}(M_{\text{geom}}) \ge \text{Acc}(M_{\text{CNN}}) - 0.025 \quad \text{and} \quad \text{Latency}(M_{\text{geom}}) \le 0.10 \times \text{Latency}(M_{\text{CNN}})$$

#### Evaluation Metrics & Statistical Verification
- Primary Metrics: Top-1 Accuracy, Macro-averaged F1 Score, Inference Latency (ms/sample on single CPU core), Memory Footprint (MB).
- Verification Protocol: 5-fold Leave-Signers-Out cross-validation on the Phase 1 16-class INCLUDE benchmark.
- Mapping: Phase 1 experiments SANKET-028, SANKET-029, and SANKET-030.

---

### Research Question 2 (RQ2): Coordinate Normalization Resilience to Distance and Perspective

**Question**: *What coordinate normalization algorithm provides the highest robustness against camera-distance variations and planar rotation without degrading distinguishing linguistic features?*

#### Mathematical Formulation
Let raw landmark coordinate for joint $i \in \{0, \dots, 20\}$ at time $t$ be $p_i(t) = (x_i, y_i, z_i) \in [0, 1]^3$ in screen viewport space.  
Consider candidate normalization transforms $\mathcal{T}$:
1. **Raw Screen Coordinates ($\mathcal{T}_{\text{raw}}$)**: $p_i'(t) = p_i(t)$.
2. **Wrist-Centered Translation ($\mathcal{T}_{\text{trans}}$)**: $p_i'(t) = p_i(t) - p_0(t)$, where $p_0$ is the wrist landmark.
3. **Wrist-Centered + Palm-Scale Normalization ($\mathcal{T}_{\text{scale}}$)**:
   $$p_i'(t) = \frac{p_i(t) - p_0(t)}{\|p_9(t) - p_0(t)\|_2}$$
   where $p_9$ is the metacarpophalangeal (MCP) joint of the middle finger.
4. **Full Procrustes Alignment ($\mathcal{T}_{\text{procrustes}}$)**: Orthogonal rotational alignment to canonical hand frame.

- **Null Hypothesis ($H_{0,2}$)**: Wrist-centered palm-scale normalization $\mathcal{T}_{\text{scale}}$ yields no statistically significant improvement in recognition accuracy over raw coordinates $\mathcal{T}_{\text{raw}}$ when evaluated on signers recorded at varying camera distances ($d_1 = 0.6\text{m}$, $d_2 = 1.2\text{m}$, $d_3 = 2.0\text{m}$).
$$\text{Acc}(\mathcal{T}_{\text{scale}} \mid d \in \{d_1, d_2, d_3\}) = \text{Acc}(\mathcal{T}_{\text{raw}} \mid d \in \{d_1, d_2, d_3\})$$

- **Alternative Hypothesis ($H_{1,2}$)**: Normalization $\mathcal{T}_{\text{scale}}$ reduces intra-class Euclidean landmark variance across distance tiers by $> 40\%$ and improves cross-distance test accuracy by $\ge 15\%$ percentage points compared to $\mathcal{T}_{\text{raw}}$, with negligible computational overhead ($< 0.1\text{ ms}$ per frame).
$$\text{Var}_{\text{inter-dist}}(\mathcal{T}_{\text{scale}}) \le 0.60 \times \text{Var}_{\text{inter-dist}}(\mathcal{T}_{\text{raw}}) \quad \text{and} \quad \Delta\text{Acc} \ge 0.15$$

#### Evaluation Metrics & Statistical Verification
- Primary Metrics: Intra-class distance dispersion $\sigma^2_{\text{intra}}$, Cross-distance Accuracy drop $\Delta_{\text{dist}} = \text{Acc}_{\text{near}} - \text{Acc}_{\text{far}}$, Execution latency ($\mu\text{s}$).
- Verification Protocol: Multi-distance synthetic scaling test (Phase 1, SANKET-026) followed by real multi-distance signer benchmark (Phase 5, SANKET-103).

---

### Research Question 3 (RQ3): Inductive Bias — Classical Shallow Classifiers vs Deep Sequential Architectures

**Question**: *At what vocabulary scale and degree of dynamic movement do temporal sequence models (1D-CNN, BiLSTM, GRU) justify their added computational overhead over classical ensemble classifiers (Random Forest, SVM) trained on temporal summary statistics for isolated ISL recognition?*

#### Mathematical Formulation
Partition the 16-sign vocabulary $\mathcal{V}$ into two disjoint linguistic categories:
- **Static Signs ($\mathcal{V}_{\text{stat}}$)**: Signs defined primarily by static handshape and location with near-zero trajectory movement ($\|\frac{d}{dt} p_0\|_2 \approx 0$; e.g. `GOOD`, `YES`, `NAME`).
- **Dynamic Signs ($\mathcal{V}_{\text{dyn}}$)**: Signs defined by significant spatial translation, stroke trajectory, or phase interaction ($\|\frac{d}{dt} p_0\|_2 \gg 0$; e.g. `THANK_YOU`, `HELP`, `STOP`, `PLEASE`, `WATER`).

Let $M_{\text{stat}}$ be a Random Forest / SVM classifier trained on aggregated feature vectors (mean, variance, min, max, velocity over temporal window).  
Let $M_{\text{seq}}$ be a BiLSTM / 1D-CNN temporal sequence model trained on the frame-by-frame coordinate sequence.

- **Null Hypothesis ($H_{0,3}$)**: For dynamic signs $\mathcal{V}_{\text{dyn}}$, temporal sequence models $M_{\text{seq}}$ offer no statistically significant accuracy improvement ($p \ge 0.05$) over classical summary classifiers $M_{\text{stat}}$.
$$\text{Acc}(M_{\text{seq}} \mid \mathcal{V}_{\text{dyn}}) \le \text{Acc}(M_{\text{stat}} \mid \mathcal{V}_{\text{dyn}})$$

- **Alternative Hypothesis ($H_{1,3}$)**: 
  1. For dynamic signs $\mathcal{V}_{\text{dyn}}$, temporal sequence models outperform classical summary classifiers by $\ge 12\%$ Macro F1 score due to preservation of directional phase information.
  2. For static signs $\mathcal{V}_{\text{stat}}$, classical ensemble models match temporal models within $\pm 2.0\%$ accuracy while achieving $> 3\times$ faster inference speed ($< 5\text{ ms}$ vs $> 20\text{ ms}$).

#### Evaluation Metrics & Statistical Verification
- Primary Metrics: Disaggregated Macro F1 ($\mathcal{V}_{\text{stat}}$ vs $\mathcal{V}_{\text{dyn}}$), Confusion matrices for dynamic minimal pairs (`STOP` vs `HELP`), Per-sample inference time.
- Verification Protocol: Stratified sign-category comparison under identical feature inputs (Phase 1, SANKET-028 vs SANKET-029).

---

### Research Question 4 (RQ4): Biometric Leakage & Signer-Independent Generalization Bounds

**Question**: *What is the quantitative performance degradation when evaluating ISL models under strict Signer-Independent partitioning compared to naive random frame- or sample-level train/test splits?*

#### Mathematical Formulation
Let dataset $\mathcal{D} = \bigcup_{s=1}^S \mathcal{D}_s$ contain samples from $S$ distinct native signers.
- **Random Stratified Partitioning ($\mathcal{P}_{\text{rand}}$)**: Samples randomly distributed across train and test sets such that $P(\text{signer } s \in \mathcal{D}_{\text{test}} \mid s \in \mathcal{D}_{\text{train}}) = 1.0$.
- **Signer-Independent Partitioning ($\mathcal{P}_{\text{indep}}$)**: Signers strictly partitioned such that $S_{\text{train}} \cap S_{\text{test}} = \emptyset$.

- **Null Hypothesis ($H_{0,4}$)**: The difference in Top-1 test accuracy between random sample splitting and Signer-Independent splitting is negligible ($|\text{Acc}(\mathcal{P}_{\text{rand}}) - \text{Acc}(\mathcal{P}_{\text{indep}})| \le 3.0\%$), indicating that model features are inherently invariant to signer identity.

- **Alternative Hypothesis ($H_{1,4}$)**: Random sample-level splitting exhibits substantial optimistic bias, inflating reported Macro F1 by $\ge 15.0\%$ percentage points due to visual background correlation, anatomical biometric familiarity, and signer idiosyncrasies. Signer-Independent cross-validation reveals the genuine operational generalization performance.
$$\text{F1}(\mathcal{P}_{\text{rand}}) - \text{F1}(\mathcal{P}_{\text{indep}}) \ge 0.15$$

#### Evaluation Metrics & Statistical Verification
- Primary Metrics: Generalization Gap $\Delta_{\text{gen}} = \text{Acc}(\mathcal{P}_{\text{rand}}) - \text{Acc}(\mathcal{P}_{\text{indep}})$, Out-of-distribution signer variance $\sigma^2_s$.
- Verification Protocol: Direct empirical comparison in SANKET-031 using the Leave-Signers-Out protocol formulated in `docs/research/EVALUATION_METHODOLOGY.md`.

---

### Research Question 5 (RQ5): Real-Time Pipeline Latency Budget Under Edge Constraints

**Question**: *Can a complete end-to-end ISL recognition pipeline (video capture $\to$ landmark detection $\to$ normalization $\to$ temporal buffering $\to$ neural inference $\to$ text/speech output) operate within a strict 50 ms per-frame latency budget ($\ge 25\text{ FPS}$) on a commodity non-GPU CPU system?*

#### Mathematical Formulation
Total frame-to-output pipeline latency $T_{\text{total}}$ is decomposed into component stages:
$$T_{\text{total}} = T_{\text{capture}} + T_{\text{mediapipe}} + T_{\text{normalize}} + T_{\text{infer}} + T_{\text{render}}$$
Target constraints:
- Frame throughput: $\Phi = \frac{1000}{T_{\text{total}}} \ge 25.0 \text{ frames/second}$
- Per-frame latency budget: $T_{\text{total}} \le 40.0\text{ ms}$ (leaving 10 ms headroom for OS jitter).

- **Null Hypothesis ($H_{0,5}$)**: Total pipeline latency on a reference 4-core x86 CPU exceeds 50 ms ($T_{\text{total}} > 50\text{ ms}$, $\Phi < 20\text{ FPS}$), rendering pure CPU real-time interaction infeasible without hardware acceleration.

- **Alternative Hypothesis ($H_{1,5}$)**: Through efficient landmark caching, sliding-window array slicing, and lightweight ONNX/Quantized inference, the pipeline sustains $T_{\text{total}} \le 35.0\text{ ms}$ (throughput $\ge 28.5\text{ FPS}$) on standard commodity CPU hardware.

#### Evaluation Metrics & Statistical Verification
- Primary Metrics: 50th, 95th, and 99th percentile component latencies ($T_{\text{mediapipe}}$, $T_{\text{infer}}$), Frame throughput (FPS), CPU utilization percentage (%).
- Verification Protocol: Dedicated latency microbenchmarking suite in Phase 1 (SANKET-032) and Phase 5 (SANKET-102).

---

## 3. Research Questions to Planned Experiments Mapping Matrix

| Research Question | Primary Hypothesis | Tested Variables | Baseline Model | Verification Issue | Milestone |
| :---------------- | :----------------- | :--------------- | :------------- | :----------------- | :-------- |
| **RQ1 (Representation)** | Skeletal coordinates match CNN accuracy within $\pm 2.5\%$ at $10\times$ speedup | Landmark tensors vs raw RGB frames | MobileNetV3 / ResNet-18 | **SANKET-028**, **SANKET-030** | Phase 1 (M1) |
| **RQ2 (Normalization)** | Wrist + palm-scale normalization cuts distance error by $\ge 15\%$ | $\mathcal{T}_{\text{raw}}$ vs $\mathcal{T}_{\text{trans}}$ vs $\mathcal{T}_{\text{scale}}$ | Unnormalized coordinates | **SANKET-026**, **SANKET-103** | Phase 1 & 5 |
| **RQ3 (Inductive Bias)** | Sequence models outperform static models on dynamic signs by $\ge 12\%$ | Summary stats vs temporal frames | Random Forest / Linear SVM | **SANKET-028**, **SANKET-029** | Phase 1 (M1) |
| **RQ4 (Generalization)** | Signer-Independent splits reveal $\ge 15\%$ generalization gap over random splits | $\mathcal{P}_{\text{rand}}$ vs $\mathcal{P}_{\text{indep}}$ | Random K-Fold Split | **SANKET-031**, **SANKET-100** | Phase 1 & 5 |
| **RQ5 (Latency Budget)** | Full CPU pipeline completes within $\le 35\text{ ms}$ ($\ge 28\text{ FPS}$) | Capture, MediaPipe, Model, Render latency | Unoptimized sequential loop | **SANKET-032**, **SANKET-102** | Phase 1 & 5 |

---

## 4. Scientific Rigor & Negative Result Policy

In accordance with rigorous academic engineering principles:
1. **Reporting Integrity**: Every hypothesis test will be reported in `docs/experiments/` regardless of whether the hypothesis is supported or rejected.
2. **Negative Result Documentation**: If a null hypothesis cannot be rejected (e.g. if deep sequence models provide no statistically significant gain over Random Forest across all 16 signs), the negative result will be formally documented in an Architecture Decision Record (ADR) and incorporated into the design recommendations.
3. **Reproducibility Guarantee**: Every experiment script will fix random seeds (`random`, `numpy`, `torch`), record the exact git commit SHA, document package dependency versions, and log dataset hash checksums.
