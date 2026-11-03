# AGENTS.md — SANKET AI Engineering Operational Manual

> **Authoritative Operational Guide for AI Assistants, Pair Programmers, and Autonomous Agents**  
> *SANKET (संकेत) — AI-Assisted Indian Sign Language Accessibility System (`https://github.com/GITLUVV/Sanket`)*

---

## 1. Project Mission & Identity

SANKET is an open-source, privacy-preserving, low-latency Indian Sign Language (ISL) recognition system engineered for edge execution on commodity CPUs.

- **Current Active Milestone**: `Milestone 2 — Real-Time Sanket MVP` (Issues #51–#60).
- **Core Technology Stack**: Python 3.10+, MediaPipe Hands & Pose, OpenCV, Scikit-learn, NumPy, Joblib, Pytest, Ruff.
- **Primary Model Baseline**: Random Forest classifier on 645-dimensional geometric and temporal landmark features (ADR-0003).

---

## 2. Strict Non-Negotiable Operational Rules

1. **No Direct Pushes to `main`**: All work proceeds via issue-linked branches (`feat/SANKET-XXX-...`, `fix/...`, `research/...`, `docs/...`) and small, reviewable Pull Requests. Never independently merge PRs.
2. **Zero-Biometric-Retention Mandate**: Camera frames and biometric media must remain in volatile memory. Never write raw video recordings, face images, or unanonymized biometric data to disk.
3. **Privacy & Path Hygiene**: Never include personal machine paths (e.g., `C:\Users\...`) or developer usernames in commits, PR descriptions, issue bodies, or documentation. Always use repository-relative paths (`src/sanket/...`).
4. **Third-Party Dataset Licensing**: Never download, ingest, or process external datasets without verifying entries in `docs/datasets/DATASET_REGISTER.md` and confirming documented owner-approval.
5. **Deterministic Mathematics**: Always lock pseudo-random seeds (`seed=42`, `random_state=42`) across all split generators, feature extractors, and model training routines.
6. **Zero Quality Regressions**: All 206+ unit tests must pass (`pytest -q`) and code style must produce zero errors (`ruff check .`) before any PR is submitted.

---

## 3. Modular Agent Rules Discovery (`.agents/rules/`)

Consult these specialized rule files when executing tasks in specific domains:

| Rule Topic | Path | Scope & Mandatory Constraints |
| :--- | :--- | :--- |
| **Coding Standards** | [`.agents/rules/coding-standards.md`](.agents/rules/coding-standards.md) | Python 3.10+ typing, Google docstrings, vectorization, error handling. |
| **Architecture Constraints** | [`.agents/rules/architecture-constraints.md`](.agents/rules/architecture-constraints.md) | Layer decoupling (`core` $\rightarrow$ `data` $\rightarrow$ `vision` $\rightarrow$ `features` $\rightarrow$ `models`), no circular imports. |
| **Security & Privacy** | [`.agents/rules/security-and-privacy.md`](.agents/rules/security-and-privacy.md) | Biometric data protection, secret avoidance, dependency immutability. |
| **Testing & Benchmarks** | [`.agents/rules/testing-and-benchmarking.md`](.agents/rules/testing-and-benchmarking.md) | Pytest practices, synthetic mocks, latency percentiles (< 100 ms target). |
| **Git Governance** | [`.agents/rules/git-and-governance.md`](.agents/rules/git-and-governance.md) | Branch naming, Conventional Commits PR titles, 4-tier labels, closing issues. |

---

## 4. Reusable Agent Skills (`.agents/skills/`)

- [`verify-pipeline`](.agents/skills/verify-pipeline/SKILL.md): Executes environment verification (`scripts/verify_setup.py`), Ruff linting, and Pytest test suite.
- [`model-evaluation`](.agents/skills/model-evaluation/SKILL.md): Executes model evaluation harness (`scripts/evaluate_model.py`) and validates structured JSON experiment manifests.

---

## 5. Reference Documentation Index

Do not load every document into context at once. Consult only when needed:

- **Architecture Details**: [`ARCHITECTURE.md`](ARCHITECTURE.md) & [`docs/architecture/SYSTEM_OVERVIEW.md`](docs/architecture/SYSTEM_OVERVIEW.md)
- **Technology Stack**: [`TECH_STACK.md`](TECH_STACK.md)
- **Developer Workflow**: [`DEVELOPMENT.md`](DEVELOPMENT.md) & [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Coding & Typing Guidelines**: [`CODING_STANDARDS.md`](CODING_STANDARDS.md)
- **Testing Protocols**: [`TESTING.md`](TESTING.md) & [`docs/research/EVALUATION_METHODOLOGY.md`](docs/research/EVALUATION_METHODOLOGY.md)
- **Technical Decisions**: [`docs/decisions/`](docs/decisions/) (`ADR-0001`, `ADR-0002`, `ADR-0003`)
- **Master Roadmap**: [`docs/roadmap/SANKET_ROADMAP.md`](docs/roadmap/SANKET_ROADMAP.md)
- **Project History & Audit**: [`docs/development/PROJECT_LOG.md`](docs/development/PROJECT_LOG.md) & [`docs/audits/PROJECT_RECOVERY_AUDIT.md`](docs/audits/PROJECT_RECOVERY_AUDIT.md)

