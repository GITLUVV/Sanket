# Evaluation Methodology & Benchmarking Standards — Sanket

This document defines the formal, peer-review quality evaluation methodology, experimental partitioning architecture, quantitative metric definitions, and operational runtime benchmarking standards for **Sanket — AI-Assisted Indian Sign Language Accessibility System**.

---

## 1. Scientific Imperative & Critique of Prior Literature

A pervasive methodological flaw in published sign language recognition literature is evaluating models using randomly shuffled video clips or individual frames drawn from the same recording sessions. This practice introduces fatal methodological errors:
1. **Biometric Leakage**: The model learns individual signer anatomy (skin tone, hand proportions, jewelry, clothing) rather than true linguistic phonology.
2. **Environmental & Background Leakage**: Fixed camera backgrounds, static lighting shadows, and wall textures allow models to achieve artificial performance (> 95% accuracy) by memorizing session artifacts.
3. **Temporal Auto-Correlation**: Consecutive video frames are near-identical; sampling frames across train and test sets directly leaks future states to the classifier.

Consequently, models that report near-perfect academic benchmark accuracy completely fail when deployed in front of a new, unseen deaf or hard-of-hearing user.

**Sanket institutes a strict, non-negotiable Signer-Independent (Leave-Signers-Out) evaluation standard across all development phases.**

---

## 2. Experimental Data Partitioning Architecture

### 2.1. Strict Signer-Independent Disjointness

Let the benchmark dataset $\mathcal{D}$ comprise $N$ samples articulated by a set of $S$ distinct native signers:
$$\mathcal{D} = \bigcup_{s \in S} \mathcal{D}_s, \quad S = \{s_1, s_2, \dots, s_{|S|}\}$$

All data splits must strictly partition the set of signers $S$ into mutually exclusive subsets:
$$S = S_{\text{train}} \cup S_{\text{val}} \cup S_{\text{test}}$$
$$\text{subject to} \quad S_{\text{train}} \cap S_{\text{val}} = \emptyset, \quad S_{\text{train}} \cap S_{\text{test}} = \emptyset, \quad S_{\text{val}} \cap S_{\text{test}} = \emptyset$$

The corresponding sample partitions are defined as:
- **Training Partition ($\mathcal{D}_{\text{train}} \approx 70\%$)**: $\mathcal{D}_{\text{train}} = \bigcup_{s \in S_{\text{train}}} \mathcal{D}_s$. Used exclusively for parameter gradient updates and feature extraction calibration.
- **Validation Partition ($\mathcal{D}_{\text{val}} \approx 15\%$)**: $\mathcal{D}_{\text{val}} = \bigcup_{s \in S_{\text{val}}} \mathcal{D}_s$. Consisting of signers completely unseen during training. Used strictly for hyperparameter optimization, model selection, regularization tuning, and early stopping.
- **Test Partition ($\mathcal{D}_{\text{test}} \approx 15\%$)**: $\mathcal{D}_{\text{test}} = \bigcup_{s \in S_{\text{test}}} \mathcal{D}_s$. Consisting of signers completely unseen during both training and validation. Evaluated exactly once per finalized architecture to generate reported benchmark claims.

> [!CAUTION]
> Under no circumstances may any frame, video clip, or feature tensor from signers in $S_{\text{test}}$ or $S_{\text{val}}$ appear in the training pipeline or normalization estimators.

### 2.2. Class-Stratified Signer Partitioning

To ensure balanced evaluation across all classes $c \in \mathcal{V}$:
- Every signer partition ($S_{\text{train}}, S_{\text{val}}, S_{\text{test}}$) must contain at least $k \ge 1$ articulation instances of every sign in the active vocabulary.
- If a dataset exhibits incomplete signer-class coverage, signers must be partitioned using a greedy constrained combinatorial optimization ensuring minimal cross-class variance:
  $$\min \sum_{c \in \mathcal{V}} \left( \frac{|\mathcal{D}_{\text{test}, c}|}{|\mathcal{D}_{\text{test}}|} - \frac{1}{|\mathcal{V}|} \right)^2$$

### 2.3. Leave-Signers-Out Cross-Validation (LSO-CV)

For robust statistical generalization testing, Phase 1 and Phase 2 will execute $K$-Fold Leave-Signers-Out Cross-Validation:
- Signer pool $S$ is divided into $K$ disjoint signer groups $\{G_1, G_2, \dots, G_K\}$.
- In iteration $k$, group $G_k$ serves as the unseen test set, group $G_{(k \bmod K) + 1}$ serves as validation, and the remaining $K-2$ groups serve as training.
- Reported metrics represent the mean and standard error of the mean (SEM) over all $K$ folds:
  $$\mu = \frac{1}{K} \sum_{k=1}^K M_k, \quad \text{SEM} = \frac{\sigma}{\sqrt{K}}$$

---

## 3. Quantitative Performance Metrics

### 3.1. Primary Classification Metrics

Let $N$ be the total number of evaluation samples, $|\mathcal{V}|$ be the vocabulary size, $y_i$ be the ground truth label, and $\hat{y}_i$ be the top predicted label.

#### 1. Top-1 Accuracy
Proportion of test samples correctly classified:
$$\text{Top-1 Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$

#### 2. Top-k Accuracy (Applicable for $|\mathcal{V}| \ge 16$)
Proportion of test samples where ground truth $y_i$ is within the top $k$ predicted probabilities:
$$\text{Top-k Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \text{argtop}_k(P(\hat{Y} \mid X_i)))$$

#### 3. Macro-Averaged Precision, Recall, and F1-Score
In accessibility systems, minority emergency signs (e.g. `HELP`, `DOCTOR`) are just as vital as frequent conversational greetings. Macro-averaging computes metrics independently for each class before averaging, ensuring equal weight regardless of sample counts:
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
$$\text{Precision}_{\text{macro}} = \frac{1}{|\mathcal{V}|} \sum_{c \in \mathcal{V}} \text{Precision}_c, \quad \text{Recall}_{\text{macro}} = \frac{1}{|\mathcal{V}|} \sum_{c \in \mathcal{V}} \text{Recall}_c$$
$$\text{Macro F1} = \frac{2 \cdot \text{Precision}_{\text{macro}} \cdot \text{Recall}_{\text{macro}}}{\text{Precision}_{\text{macro}} + \text{Recall}_{\text{macro}}}$$

#### 4. Multi-Class Matthew's Correlation Coefficient (MCC)
MCC provides the most resilient metric against class imbalance by considering true/false positives and negatives:
$$\text{MCC} = \frac{c \cdot s - \sum_k p_k t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}}$$
where $c$ is the total correct predictions, $s$ is the total samples, $p_k$ is the number of times class $k$ was predicted, and $t_k$ is the number of times class $k$ truly occurred.

---

## 4. Error Diagnostics & Qualitative Analysis

### 4.1. Normalized Confusion Matrix Heatmaps
Every evaluation report must include a row-normalized confusion matrix:
$$C_{i, j} = \frac{N_{i, j}}{\sum_{m \in \mathcal{V}} N_{i, m}}$$
where $C_{i, j}$ represents the probability that sign $i$ is recognized as sign $j$. Ideal performance yields an identity matrix $I$.

### 4.2. Phonological Linguistic Error Taxonomy
Misclassifications must be categorized by linguistic source:
- **Handshape Error ($E_{\mathcal{H}}$)**: Correct location and trajectory, but misclassified finger flexion/extension (e.g. mistaking 5-finger open hand for cupped C-hand).
- **Location Error ($E_{\mathcal{L}}$)**: Hand articulated in wrong signing space (e.g. chest level vs face level).
- **Orientation Error ($E_{\mathcal{O}}$)**: Palm or metacarpal normal vector rotated away from target canonical plane.
- **Movement Trajectory Error ($E_{\mathcal{M}}$)**: Stroke vector velocity or shape incorrect (e.g. circular vs linear).
- **Tracking / Occlusion Failure ($E_{\text{track}}$)**: MediaPipe landmark confidence drop ($< 0.5$) causing synthetic zeros or jitter.

---

## 5. Operational Runtime & Usability Benchmarks

Accessibility solutions must maintain fluid, continuous interactivity on commodity, non-GPU user devices. All models must be profiled against the following operational criteria:

| Metric | Target Specification | Measurement Protocol |
| :----- | :------------------- | :------------------- |
| **Inference Latency ($T_{\text{infer}}$)** | $\le 15.0\text{ ms}$ (median), $\le 25.0\text{ ms}$ (p95) | Measured across 1,000 warm-up-isolated model inference forward passes. |
| **Total Pipeline Latency ($T_{\text{total}}$)** | $\le 35.0\text{ ms}$ (median), $\le 50.0\text{ ms}$ (p95) | Frame capture $\to$ MediaPipe $\to$ Normalize $\to$ Model $\to$ Render. |
| **Sustainable Throughput ($\Phi$)** | $\ge 25.0\text{ FPS}$ sustained for $\ge 10\text{ minutes}$ | End-to-end processing frame rate under typical webcam input stream. |
| **Landmark Drop Rate (LDR)** | $< 3.0\%$ of operational frames | Proportion of frames where MediaPipe hands confidence drops below detection threshold. |
| **Memory Footprint (RAM)** | $\le 200\text{ MB}$ Resident Set Size (RSS) | Measured via Python `tracemalloc` / system memory profiler. |
| **Disk Storage / Model Size** | $\le 30\text{ MB}$ total serialized size | Combined disk footprint of trained weights, scalers, and configuration files. |
| **Energy / CPU Utilization** | $\le 40\%$ CPU utilization across 4 cores | Prevents thermal throttling on standard portable laptops. |

---

## 6. Reproducibility & Benchmark Logging Protocol

To satisfy academic reproducibility standards:

### 6.1. Seed Locking Specification
All pseudo-random number generators must be deterministically initialized prior to dataset partitioning, landmark augmentation, and model training:
```python
import os
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
```

### 6.2. Experiment Artifact Manifest (`experiment_manifest.json`)
Every formal experiment must generate an immutable structured JSON manifest logged in `docs/experiments/` containing:
- `experiment_id`: Formal identifier (e.g., `EXP-P1-001-RF-BASELINE`).
- `timestamp_utc`: ISO 8601 execution timestamp.
- `git_commit_sha`: Git commit hash representing exact code state.
- `dataset_checksum`: SHA-256 hash of the ingested dataset coordinate tensors.
- `active_vocabulary`: List of class glosses evaluated.
- `signer_partitions`: Explicit list of signer IDs allocated to `train`, `val`, and `test`.
- `metrics`: Dictionary of Top-1 accuracy, Macro F1, Per-Class F1, and confusion matrix.
- `runtime`: Median, p95, and p99 latencies, peak RSS memory.
- `environment`: OS, CPU model, Python version, library dependencies.

### 6.3. Peer-Review Sign-Off Checklist
Before any experimental result is incorporated into project roadmaps or published reports:
- [ ] Strict Signer-Independent split verified ($S_{\text{train}} \cap S_{\text{test}} = \emptyset$).
- [ ] Zero raw biometric video committed to git repository.
- [ ] Random seeds fixed and documented.
- [ ] Macro F1 and per-class confusion matrix generated.
- [ ] Latency profiled on reference non-GPU CPU hardware.
- [ ] Manifest saved and referenced in `docs/development/PROJECT_LOG.md`.
