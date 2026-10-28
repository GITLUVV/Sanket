# Sanket (संकेत) — AI-Assisted Indian Sign Language Accessibility System

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-0_Engineering_Foundation-orange.svg)](docs/roadmap/SANKET_ROADMAP.md)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Security: Protected](https://img.shields.io/badge/Security-Dependabot%20%7C%20Secret%20Scan-brightgreen)](SECURITY.md)

**Sanket** is an academic and research-oriented AI accessibility system designed to recognize Indian Sign Language (ISL) from camera video input and translate recognized signs into accessible text and speech.

---

## Current Project Status
- **Active Phase**: **Phase 0 — Engineering & Research Foundation**
- **Milestone**: [Milestone 0 — Engineering Foundation](https://github.com/anveshna25/Sanket/milestone/1)
- **Current Objective**: Establish repository governance, reproducible documentation, dataset licensing audits, security policies, and initial vocabulary definitions.
- **Model Development**: Not yet started. ML training begins in Phase 1 following human sign-off on the Phase 0 deliverables.

---

## Research Disclaimer
> [!NOTE]
> Sanket is currently scoped to recognize a **constrained vocabulary of isolated signs** for empirical validation. The project does not currently claim or implement continuous open-vocabulary translation. Any vocabulary expansion or temporal sequence modeling follows strict empirical phase gates.

---

## High-Level Architecture
Sanket follows a modular, feed-forward pipeline separating vision ingestion from inference and presentation:

```text
Camera / Video Stream
       ↓
Landmark Extraction (MediaPipe Hands & Pose)
       ↓
Coordinate Normalization (Scale & Origin Invariant)
       ↓
Recognition Engine (Baseline RF/SVM → Sequence Models)
       ↓
Prediction Smoothing & Confidence Filter
       ↓
Accessible Text Display & Speech Output
```

For detailed architectural diagrams and subsystem designs, see [System Overview](docs/architecture/SYSTEM_OVERVIEW.md).

---

## Documentation Quick Links
- **Master Roadmap**: [`docs/roadmap/SANKET_ROADMAP.md`](docs/roadmap/SANKET_ROADMAP.md)
- **Development Log**: [`docs/development/PROJECT_LOG.md`](docs/development/PROJECT_LOG.md)
- **Architecture Decisions (ADRs)**: [`docs/decisions/`](docs/decisions/)
- **Dataset Register**: [`docs/datasets/DATASET_REGISTER.md`](docs/datasets/DATASET_REGISTER.md)
- **Experiment Records**: [`docs/experiments/`](docs/experiments/)
- **Research Questions**: [`docs/research/RESEARCH_QUESTIONS.md`](docs/research/RESEARCH_QUESTIONS.md)
- **Wiki Structure**: [`docs/architecture/WIKI_STRUCTURE.md`](docs/architecture/WIKI_STRUCTURE.md)

---

## Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/anveshna25/Sanket.git
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