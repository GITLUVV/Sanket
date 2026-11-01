# Sanket (संकेत) — AI-Assisted Indian Sign Language Accessibility System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-1_Data_%26_Isolated_Signs-blue.svg)](docs/roadmap/SANKET_ROADMAP.md)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen)](https://github.com/GITLUVV/Sanket/actions)
[![Security: Protected](https://img.shields.io/badge/Security-Dependabot%20%7C%20Secret%20Scan-brightgreen)](SECURITY.md)

**Sanket** is an academic and research-oriented AI accessibility system designed to recognize Indian Sign Language (ISL) from camera video input and translate recognized signs into accessible text and speech.

---

## Current Project Status
- **Active Phase**: **Phase 1 — Data & Isolated Sign Recognition**
- **Milestone**: [Milestone 1 — Data & Isolated Sign Recognition](https://github.com/GITLUVV/Sanket/milestone/2)
- **Current Objective**: Build an offline, reproducible pipeline for recognizing a constrained vocabulary of isolated ISL signs using skeletal landmark features and classical/neural baseline classifiers.
- **Phase 0 Status**: **Completed and Verified** (Repository foundations, security policies, dataset licensing audits, initial 16-sign vocabulary, and signer-independent evaluation protocols established).
- **Phase 1 Implementation Progress (9 of 14 tasks merged)**:
  - Dataset ingestion with SHA-256 integrity verification (`sanket.data.ingestion`).
  - Dataset inspection and label balance profiling (`sanket.data.inspection`).
  - Signer-independent train/validation/test splitting (`sanket.data.splits`).
  - Uniform temporal subsampling and letterbox video preprocessing (`sanket.vision.preprocessing`).
  - MediaPipe 3D hand landmark extraction with temporal imputation (`sanket.vision.landmarks`).
  - Upper-body pose reference anchor investigation (`sanket.vision.pose`).
  - Wrist-centered and metacarpal palm-scaled landmark normalization (`sanket.vision.normalization`).
  - Anatomical joint angles, pairwise distances, and temporal feature pooling (`sanket.features`).
  - Baseline Random Forest classifier with validation tuning and latency profiling (`sanket.models`).
- **Phase 1 Remaining Backlog (5 outstanding tasks)**:
  - SANKET-029: Baseline SVM model (Issue #31).
  - SANKET-030: Baseline neural network / MLP (Issue #32).
  - SANKET-031: Model evaluation pipeline and manifest generation (Issue #33).
  - SANKET-032: Confusion matrix and qualitative phonological error analysis (Issue #34).
  - SANKET-033: Baseline model comparison and ADR-0003 selection (Issue #35).

---

## Research Disclaimer
> [!NOTE]
> Sanket is currently scoped to recognize a **constrained vocabulary of 16 isolated signs** for empirical research and baseline validation. The project does not currently claim or implement continuous open-vocabulary translation. Any vocabulary expansion or temporal sequence modeling follows strict empirical phase gates.

---

## System Architecture

Sanket follows a modular, feed-forward pipeline separating vision ingestion from inference and presentation:

```text
Camera / Video Stream
       ↓
Landmark Extraction (MediaPipe Hands & Pose Anchors)
       ↓
Coordinate Normalization (Wrist-Origin & Palm-Scale Invariant)
       ↓
Feature Engineering (3D Angles, Distances & Temporal Pooling)
       ↓
Recognition Engine (Baseline RF / SVM / MLP → Sequential Models)
       ↓
Prediction Smoothing & Confidence Filter
       ↓
Accessible Text Display & Speech Output
```

### Implemented Subsystems (`src/sanket/`)
- **`sanket.core`**: Type-safe configuration subsystem backed by Pydantic and YAML.
- **`sanket.data`**: Dataset ingestion with checksum verification, inspection engines, and signer-independent split generation.
- **`sanket.vision`**: Temporal subsampling, letterboxing, MediaPipe hand landmark extraction, pose anchor tracking, and coordinate normalization.
- **`sanket.features`**: Kinematic modeling computing 38 joint angles, 48 pairwise Euclidean distances, 3 inter-hand metrics, and 645-dimensional pooled tabular representations.
- **`sanket.models`**: Unified `BaseSignClassifier` contract, `SignModelMetrics` evaluation data structures, and `RandomForestSignClassifier`.

### Planned Subsystems (Future Phases)
- **Phase 2**: Real-time webcam ingestion, temporal sliding window smoothing, and responsive UI.
- **Phase 3**: Dynamic sequential modeling (LSTM / GRU / Transformer).
- **Phase 4**: Language grammar structuring and Text-to-Speech (TTS) synthesis.

---

## Documentation Quick Links
- **Master Roadmap**: [`docs/roadmap/SANKET_ROADMAP.md`](docs/roadmap/SANKET_ROADMAP.md)
- **Recovery Audit Report**: [`docs/audits/PROJECT_RECOVERY_AUDIT.md`](docs/audits/PROJECT_RECOVERY_AUDIT.md)
- **Development Log**: [`docs/development/PROJECT_LOG.md`](docs/development/PROJECT_LOG.md)
- **Architecture Decisions (ADRs)**: [`docs/decisions/`](docs/decisions/)
  - [ADR-0001: Project Foundations & Structure](docs/decisions/ADR-0001-project-foundations.md)
  - [ADR-0002: Skeletal & Kinematic Feature Representation](docs/decisions/ADR-0002-feature-representation.md)
- **Dataset Register & Legal Governance**: [`docs/datasets/DATASET_REGISTER.md`](docs/datasets/DATASET_REGISTER.md)
- **Research Questions & Hypotheses**: [`docs/research/RESEARCH_QUESTIONS.md`](docs/research/RESEARCH_QUESTIONS.md)
- **Initial 16-Sign Vocabulary**: [`docs/research/INITIAL_VOCABULARY.md`](docs/research/INITIAL_VOCABULARY.md)
- **Evaluation Methodology**: [`docs/research/EVALUATION_METHODOLOGY.md`](docs/research/EVALUATION_METHODOLOGY.md)
- **Experiment Records**: [`docs/experiments/`](docs/experiments/)
- **Wiki Structure**: [`docs/architecture/WIKI_STRUCTURE.md`](docs/architecture/WIKI_STRUCTURE.md)

---

## Development Setup

```bash
# 1. Clone the canonical repository
git clone https://github.com/GITLUVV/Sanket.git
cd Sanket

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# 4. Verify installation and foundation checks
python scripts/verify_setup.py
pytest
```

---

## Contributing
We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).  
Direct pushes to `main` are restricted. All contributions must proceed through issue-linked branches and human-reviewed Pull Requests.

---

## Security & Privacy
Please review our [Security Policy](SECURITY.md) for vulnerability disclosure guidelines.  
**Strict rule**: Never commit private datasets, camera footage of identifiable individuals, credentials, or API keys.

---

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. Third-party datasets retain their respective original licenses as recorded in the [Dataset Register](docs/datasets/DATASET_REGISTER.md).
