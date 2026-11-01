# Datasets & Data Governance

## Privacy & Biometric Protection
Sign language videos contain identifiable human faces and bodies. Sanket strictly adheres to:
- **Zero Raw Media in Git**: No video files (`.mp4`, `.avi`, `.mov`) or facial images are committed to GitHub.
- **Local Isolated Storage**: Datasets reside locally in `data/` or `datasets/raw/`, excluded by `.gitignore`.
- **Derived Anonymized Features**: Only abstract numerical landmark coordinates are processed for model training.

## Dataset Audit Matrix
- **INCLUDE (DS-001)**: Primary corpus for Phase 1 (CC BY 4.0).
- **ISL-CSLTR (DS-002)**: Continuous corpus, deferred to Phase 3/4 under academic MoU.
- **CISLR (DS-003)**: Auxiliary dataset for Phase 5 cross-validation.
- **IIIT-Delhi (DS-004)**: Auxiliary dataset for Phase 5 robustness testing.

Refer to `docs/datasets/DATASET_REGISTER.md` for complete legal audits.
