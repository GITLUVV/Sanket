# Dataset Register & Legal Governance — Sanket

- **Issue Reference**: SANKET-009 (Dataset licensing/access review)
- **Status**: Formally Audited & Gated
- **Date**: 2026-10-28
- **Audit Lead**: Sanket Research & Compliance

---

## 1. Project Licensing Non-Interference Policy

> [!IMPORTANT]
> **Strict Non-Interference Clause**: The Sanket software repository is licensed under the **Apache License 2.0**. However, the Apache-2.0 license applies **strictly to the original software source code, architectural documentation, and build configurations** authored by the Sanket maintainers.
> 
> Under no circumstances does the Apache-2.0 license grant, imply, or extend sub-licensing rights over third-party Indian Sign Language datasets, participant video recordings, or proprietary corpus annotations. Every external dataset integrated into Sanket retains its original author copyright, licensing constraints, and ethical usage boundaries.

---

## 2. Human Project-Owner Approval Gate

Before any automated script, developer, or research assistant downloads, ingests, processes, or caches raw video or image frames from any dataset registered below:

1. **Explicit Human Sign-Off**: The project owner (`anveshna25`) must formally approve the specific dataset tier in writing or via an explicit issue response.
2. **Local Storage Compliance**: Downloaded raw video files must reside in uncommitted storage directories (`datasets/raw/` or `data/`), which are strictly excluded by `.gitignore`.
3. **No Upstream Video Pushing**: Raw video files containing identifiable signers must **never be committed or pushed** to GitHub.
4. **Derived Feature Validation**: Only derived, anonymized numerical feature arrays (normalized 3D MediaPipe landmark coordinates) may be shared or versioned, provided the source dataset license permits derivative mathematical representations.

---

## 3. Dataset Audit Matrix

| ID | Dataset Name | Primary Institution | Governing License | Commercial Use | Derivative Features Permitted | Raw Video Redistribution | Approval Status |
| :- | :----------- | :------------------ | :---------------- | :------------- | :---------------------------- | :----------------------- | :-------------- |
| **DS-001** | INCLUDE / INCLUDE-50 | IIT Madras & AIISC | Creative Commons Attribution 4.0 (CC BY 4.0) | Allowed with Attribution | Yes (Anonymized coordinates) | Allowed with original attribution | **Recommended for Phase 1 Ingestion** |
| **DS-002** | ISL-CSLTR | CDAC Kolkata & IIIT-H | Custom Academic Research MoU | Prohibited | Research Only | Strictly Prohibited | **Deferred to Phase 3/4 (MoU Required)** |
| **DS-003** | CISLR | Panjab University | Academic Non-Commercial | Prohibited | Research Only | Prohibited without Permission | **Restricted / Auxiliary Only** |
| **DS-004** | IIIT-Delhi ISL | IIIT Delhi | Custom Academic | Prohibited | Research Only | Prohibited | **Restricted / Auxiliary Only** |

---

## 4. Comprehensive Dataset Audits

### DS-001: INCLUDE (Indian Sign Language Dataset)
- **Citation**: *Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition", ACM Multimedia 2020.*
- **Host / Source**: AI Institute of South Carolina & Indian Institute of Technology Madras (GitHub / ACM Digital Library).
- **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **License Terms**:
  - *Sharing*: You are free to copy and redistribute the material in any medium or format.
  - *Adaptation*: You are free to remix, transform, and build upon the material for any purpose, even commercially.
  - *Attribution Condition*: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **Access Requirements**: Openly accessible via academic repository links; no NDAs or signed MoUs required.
- **Signer Privacy & Biometric Consent**:
  - Videos feature deaf students and certified educators from the St. Louis Institute for the Deaf and Blind, Chennai.
  - Recorded with institutional consent for academic open-access research.
  - Faces and hands are visible. Because CC BY 4.0 permits adaptation, extracting abstract skeletal landmarks and discarding raw facial pixel matrices is fully compliant.
- **Redistribution Policy for Sanket**:
  - Raw video clips: May be downloaded locally to `data/include/` (ignored by git).
  - Derived landmarks: Normalized keypoint arrays (`.npz` / `.parquet`) may be tracked and redistributed under CC BY 4.0 attribution.
- **Intended Sanket Use**:
  - Primary training corpus for Phase 1 isolated baseline using the 15-class subset specified in `docs/research/INITIAL_VOCABULARY.md`.
- **Audit Date**: 2026-10-28
- **Audit Conclusion**: **APPROVED FOR INGESTION** (Subject to human project-owner approval).

---

### DS-002: ISL-CSLTR (Continuous & Isolated Sign Language Translation/Recognition)
- **Host / Source**: CDAC Kolkata & IIIT Hyderabad.
- **License**: **Restricted Academic Research MoU / End-User License Agreement**.
- **License Terms**:
  - Strict non-commercial, academic educational research use only.
  - Explicit prohibition against re-hosting, sub-licensing, or publicly redistributing raw footage.
  - Commercial exploitation or deployment in commercial software without separate licensing is explicitly prohibited.
- **Access Requirements**:
  - Requires institutional affiliation, written request to CDAC/IIIT-H, and signed agreement.
- **Signer Privacy & Biometric Consent**:
  - Native signers recorded in controlled studio settings with signed institutional consent.
- **Redistribution Policy for Sanket**:
  - Zero raw footage may be hosted or distributed in the Sanket repository.
- **Intended Sanket Use**:
  - Deferred to Phase 3 (Dynamic Sign Modeling) and Phase 4 (Sentence Formation) for continuous sentence benchmarking.
- **Audit Date**: 2026-10-28
- **Audit Conclusion**: **DEFERRED**. Ingestion postponed until Phase 3 and formal MoU execution.

---

### DS-003: CISLR Dataset
- **Host / Source**: Panjab University, Chandigarh.
- **License**: **Academic Research Only**.
- **License Terms**: Non-commercial use for academic benchmarking. No public raw redistribution without permission.
- **Signer Privacy**: Institutional consent granted for academic publication.
- **Audit Conclusion**: **AUXILIARY ONLY**. Restricted to secondary cross-validation in Phase 5.

---

### DS-004: IIIT-Delhi Gesture Dataset
- **Host / Source**: Indraprastha Institute of Information Technology Delhi.
- **License**: **Custom Academic Research Agreement**.
- **License Terms**: Restricted academic evaluation.
- **Audit Conclusion**: **AUXILIARY ONLY**. Retained for robustness benchmarking in Phase 5.

---

## 5. Dataset Audit Procedure for Future Ingestion

Whenever an additional external dataset is proposed for Sanket:

1. **Create Issue**: File an issue using `.github/ISSUE_TEMPLATE/research.md`.
2. **Review License Terms**: Confirm whether the license is permissive (CC BY, MIT, Apache) or restrictive (CC BY-NC, non-redistributable academic MoU).
3. **Audit Biometric Consent**: Verify that participants provided consent for video recording and computer vision analysis.
4. **Append Record**: Add the formal profile to this document (`docs/datasets/DATASET_REGISTER.md`).
5. **Project Owner Gate**: Obtain explicit project owner approval before any download script is merged.
