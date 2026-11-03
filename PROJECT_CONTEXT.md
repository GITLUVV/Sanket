# SANKET (संकेत) — Project Context & Engineering Foundations

> **Engineering & Research Context for Developers, AI Agents, and Contributors**  
> *Authoritative architectural boundaries, linguistic definitions, and scientific methodology.*

---

## 1. Domain Background & Linguistic Principles

### 1.1 Indian Sign Language (ISL) vs. Western Sign Languages
Indian Sign Language (ISL) is a complete natural visual-gestural language with its own complex grammar, syntax, and morphology, fundamentally distinct from spoken languages used in India (such as Hindi, English, or Tamil) and other sign languages (such as American Sign Language or British Sign Language):

1. **Dual-Hand Dominance**: While ASL utilizes a one-handed alphabet for fingerspelling, ISL utilizes a **two-handed fingerspelling system** where a dominant hand performs actions upon or in tandem with a non-dominant base hand.
2. **Spatial Reference and Shared Coordinates**: In two-handed signs, the relative Euclidean distance and orientation between dominant and non-dominant wrist joints encode semantic distinctions.
3. **Phonological Structure (Stokoe & Battison Model)**:
   - **Handshape (Designator / DEZ)**: Configuration of fingers and thumb (e.g., fist, open palm, index point).
   - **Orientation (Signation / SIG)**: Direction toward which the palm and fingertips face.
   - **Location (Tabula / TAB)**: Spatial coordinate region of sign execution (neutral space, chest, chin, forehead, non-dominant palm).
   - **Movement (Kinematic / KIN)**: Trajectory and dynamic velocity profile.
   - **Non-Manual Features (NMF)**: Facial affect, head orientation, and mouth gestures.

---

## 2. Dataset Governance & Privacy Architecture

### 2.1 The Zero-Biometric-Retention Mandate
Sign language video streams inherently capture facial images, eye movements, and body morphology, constituting sensitive biometric data under international and Indian privacy frameworks (Digital Personal Data Protection Act, 2023).

**Mandatory Repository Policy**:
- **Never commit raw video footage, unanonymized image frames, or facial biometric encodings to version control.**
- In live operation, camera frames are retained **strictly in volatile system memory (RAM)** inside bounded circular buffers and discarded immediately after landmark coordinate extraction.
- Only anonymized, abstract numerical skeletal coordinates (`x, y, z` coordinates normalized relative to the wrist) and computed geometrical features are stored for model training.

### 2.2 Dataset Survey & Access Governance
As documented in `docs/datasets/DATASET_REGISTER.md` and `docs/research/DATASET_SURVEY.md`:
- **INCLUDE Dataset** (IIT Madras): 0.27M frames across 263 isolated signs recorded from 7 signers. Requires explicit formal academic request and owner-approval before download.
- **ISL-CSLTR Dataset**: Continuous sentence-level sign language dataset.
- **Licensing Gate**: No third-party dataset may be integrated into SANKET without an audited entry in `DATASET_REGISTER.md` recording dataset owner authorization, citation, license terms, and SHA-256 verification hashes.

---

## 3. Initial Vocabulary Definition (Phase 1 Baseline)

To establish an empirically verifiable recognition baseline before expanding to continuous sentence-level recognition, SANKET standardized an initial closed vocabulary of **16 core isolated signs** covering daily communication and emergency concepts (`docs/research/INITIAL_VOCABULARY.md`):

| Index | English Gloss | Hindi Gloss (देवनागरी) | Phonological Category | Hand Dominance |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `HELLO` | नमस्ते | Greeting | Single-Hand |
| **1** | `THANK_YOU` | धन्यवाद | Courtesy | Single-Hand |
| **2** | `PLEASE` | कृपया | Courtesy | Dual-Hand |
| **3** | `YES` | हाँ | Affirmation | Single-Hand |
| **4** | `NO` | नहीं | Negation | Single-Hand |
| **5** | `HELP` | मदद | Emergency | Dual-Hand |
| **6** | `WATER` | पानी | Daily Need | Single-Hand |
| **7** | `FOOD` | खाना | Daily Need | Single-Hand |
| **8** | `NAME` | नाम | Identification | Dual-Hand |
| **9** | `WHERE` | कहाँ | Interrogative | Single-Hand |
| **10**| `WHEN` | कब | Interrogative | Single-Hand |
| **11**| `GOOD` | अच्छा | Evaluation | Single-Hand |
| **12**| `BAD` | बुरा | Evaluation | Single-Hand |
| **13**| `DOCTOR` | डॉक्टर | Medical | Dual-Hand |
| **14**| `EMERGENCY` | आपातकाल | Emergency | Dual-Hand |
| **15**| `FAMILY` | परिवार | Social | Dual-Hand |

---

## 4. Formal Research Questions (RQs) & Hypotheses

The development of SANKET is driven by four core research questions codified in `docs/research/RESEARCH_QUESTIONS.md`:

- **RQ1 (Landmark Feature Sufficiency)**:  
  *Can 3D skeletal landmarks alone (without raw RGB texture) provide sufficient discriminative power to classify isolated ISL signs with Macro F1 >= 0.85?*  
  **Status**: **Validated (Phase 1)**. Random Forest achieved Macro F1 = 0.9333 using 645-dim landmark geometry and temporal moments.
- **RQ2 (Pose Anchor Utility)**:  
  *Does augmenting hand landmarks with upper-body pose reference anchors (shoulders, elbows, nose) significantly improve discrimination for signs executed in shared spatial regions?*  
  **Status**: **Investigated (SANKET-025, SANKET-027)**. Confirmed upper-body shoulders serve as vital spatial anchors for location-variant signs (`DOCTOR`, `FOOD`).
- **RQ3 (Real-Time Edge Feasibility)**:  
  *Can feature extraction and inference operate on commodity multi-core CPUs with end-to-end latency < 100 ms (p95) and throughput >= 20 FPS?*  
  **Status**: **Validated (Phase 1 baseline benchmarks)**. Feature extraction + Random Forest inference executes in 0.73 ms on standard CPU hardware (> 1,500 FPS throughput).
- **RQ4 (Signer Invariance)**:  
  *Does wrist-centered and palm-scale coordinate normalization prevent model performance degradation when evaluated on unseen signers?*  
  **Status**: **Validated (ADR-0002, EXP-P1-001)**. Signer-independent splitting verified zero leakage, achieving 93.3% test accuracy on unseen signers.

---

## 5. Architectural Contracts & Decisions

All engineering implementations must adhere to established Architecture Decision Records (`docs/decisions/`):

1. **ADR-0001 (Project Foundations & Tooling)**:
   - Python 3.10+ language baseline.
   - MediaPipe Hands and Pose for vision extraction.
   - Pytest for test automation with 100% passing build gate.
   - Scikit-learn for initial classical ML baselines.
2. **ADR-0002 (Feature Representation & Coordinate Normalization)**:
   - **Translation Normalization**: Origin translated to wrist landmark (index 0).
   - **Scale Normalization**: Coordinates divided by palm size (Euclidean distance between wrist and middle MCP joint index 9).
   - **Feature Vector (645-dim)**: 126 raw coordinates + 126 normalized coordinates + 60 joint angles + 210 pairwise Euclidean distances + 123 temporal statistical moments.
3. **ADR-0003 (Baseline Classifier Selection for Real-Time MVP)**:
   - Selected **Random Forest** over SVM and MLP as the primary inference engine for Phase 2 due to superior Top-1 accuracy (93.33%), sub-millisecond CPU latency (0.73 ms p95), robust feature splitting across 645 dimensions, and native multi-class probabilistic output.

