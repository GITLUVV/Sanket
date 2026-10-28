# Indian Sign Language Dataset Survey — Sanket

## 1. Survey Objective
This survey investigates available Indian Sign Language (ISL) datasets to establish an empirical, legally unencumbered basis for model development in Phase 1 and beyond.

---

## 2. Comparative Matrix

| Dataset | Institution | Modality | Classes | Clips / Sequences | Signers | Environment | License |
| :------ | :---------- | :------- | :------ | :---------------- | :------ | :---------- | :------ |
| **INCLUDE** | IIT Madras / AIISC | RGB Video (MP4) | 263 isolated signs | 4,287 clips | 7 signers | Varied unconstrained | CC BY 4.0 / Academic |
| **INCLUDE-50** | IIT Madras | RGB Video (MP4) | 50 isolated signs | ~1,200 clips | 7 signers | Varied unconstrained | CC BY 4.0 / Academic |
| **ISL-CSLTR** | CDAC / IIIT-H | RGB Video | 100 isolated + 500 sentences | 5,000+ sequences | 7 signers | Controlled studio | Academic Research Only |
| **CISLR** | Panjab University | RGB Video | 100 signs | ~2,000 clips | Multiple | Controlled studio | Academic Research Only |
| **ISL Alphabet Datasets** | Kaggle Community | Static RGB Images | 26 A-Z / 10 Numbers | ~30,000 images | Varied | Web cameras | Open / Varied |

---

## 3. Detailed Dataset Profiles

### 3.1. INCLUDE & INCLUDE-50
- **Authors**: Sridhar et al. (ACM Multimedia 2020)
- **Strengths**: 
  - Standardized benchmark across recent ISL research literature.
  - Natural signer variance: Recorded with multiple lighting conditions, distances, and clothing styles.
  - Good lexical coverage spanning 15 domains: Education, Emergency, Healthcare, Daily activities.
  - Pre-defined official splits (Train, Validation, Test) enabling direct comparison with state-of-the-art papers.
- **Weaknesses**:
  - Contains full HD video files that require local frame extraction and preprocessing.
  - Some signs involve full upper-body motion rather than isolated hands alone.
- **Recommendation for Sanket**: **Primary candidate** for Phase 1 isolated baseline. We recommend extracting a constrained subset of 15–20 high-frequency signs for initial baseline training.

### 3.2. ISL-CSLTR
- **Authors**: Central DAC / IIIT Hyderabad
- **Strengths**: High-fidelity continuous sentence recordings, useful for grammar and temporal modeling.
- **Weaknesses**: Strict academic access requirements and controlled studio background that does not test real-world camera resilience.
- **Recommendation for Sanket**: Defer to Phase 3 (Dynamic signs) and Phase 4 (Sentence formation).

### 3.3. Static ISL Alphabet Datasets (A-Z)
- **Strengths**: Clean single-frame handshapes, fast extraction.
- **Weaknesses**: Static alphabetic finger-spelling represents only a tiny fraction of natural Indian Sign Language communication, which is predominantly word-level gesture.
- **Recommendation for Sanket**: Good for early pipeline verification, but must not be confused with true word-level ISL.
