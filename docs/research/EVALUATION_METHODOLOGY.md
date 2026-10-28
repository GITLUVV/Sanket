# Evaluation Methodology & Benchmarking Standards — Sanket

## 1. Overview
A common flaw in computer vision sign language research is evaluating models on randomly shuffled frames from the same recording sessions. This creates severe visual and biometric leakage, falsely reporting near-perfect accuracy while completely failing when tested on unseen individuals.

Sanket institutes a **strict, peer-review quality evaluation methodology**.

---

## 2. Partitioning Strategy: Signer-Independent Validation

All experimental splits must follow a **Signer-Independent (Leave-Signers-Out)** partitioning protocol:

- **Training Set (approx. 70%)**: Signers $S_1, S_2, \dots, S_k$.
- **Validation Set (approx. 15%)**: Signer $S_{k+1}$ (unseen during training). Used exclusively for hyperparameter tuning, learning rate scheduling, and early stopping.
- **Test Set (approx. 15%)**: Signers $S_{k+2}, \dots, S_n$ (completely unseen during both training and validation). Evaluated once per candidate architecture to generate final reported metrics.

> [!CAUTION]
> Under no circumstances may frames or clips from the same signer appear in both training and test sets.

---

## 3. Quantitative Performance Metrics

### 3.1. Classification Accuracy
- **Top-1 Accuracy**: Proportion of test instances where the highest-scoring class matches ground truth.
$$\text{Top-1 Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$
- **Top-5 Accuracy**: Proportion of test instances where ground truth is within the top-5 predicted probabilities (applicable for larger vocabularies).

### 3.2. Class-Balanced Metrics
To prevent majority-class bias from masking recognition failure on rare signs:
- **Macro-Averaged Precision**:
$$\text{Precision}_{\text{macro}} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FP_c}$$
- **Macro-Averaged Recall**:
$$\text{Recall}_{\text{macro}} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$$
- **Macro-Averaged F1-Score**: Harmonic mean of macro precision and macro recall.
$$\text{F1}_{\text{macro}} = \frac{2 \cdot \text{Precision}_{\text{macro}} \cdot \text{Recall}_{\text{macro}}}{\text{Precision}_{\text{macro}} + \text{Recall}_{\text{macro}}}$$

### 3.3. Confusion Matrix Analysis
Every official evaluation report must include a normalized confusion matrix heatmap identifying systematic misclassification pairs.

---

## 4. Operational & Runtime Metrics

To validate usability as an accessibility interface, models must be benchmarked on standard consumer hardware (e.g. 8-core CPU without dedicated GPU):

| Metric | Target Threshold | Description |
| :----- | :--------------- | :---------- |
| **Inference Latency** | $< 50\text{ ms}$ per frame | Time from raw image input to predicted sign probabilities. |
| **Throughput (FPS)** | $\ge 20\text{ FPS}$ | Sustainable real-time pipeline processing rate. |
| **Landmark Drop Rate** | $< 5\%$ | Frequency of failed landmark detection events under standard illumination. |
| **Model Size** | $< 50\text{ MB}$ | Disk footprint of serialised model weights for edge deployment. |
