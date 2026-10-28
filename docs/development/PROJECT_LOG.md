# Project Development Log — Sanket

This log maintains a chronological record of significant engineering progress, architectural decisions, experimental runs, and milestone transitions for the Sanket project.

---

## Log Entries

### 2026-10-28 — Engineering Foundation & Milestone 0 Setup

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Completed Work**:
  - Initialized formal repository structure: `docs/`, `src/sanket/`, `tests/`, `configs/`, `scripts/`.
  - Added Apache License 2.0 `LICENSE`.
  - Configured comprehensive security policy `SECURITY.md` covering camera/biometric privacy and vulnerability disclosures.
  - Enabled GitHub Dependabot vulnerability alerts and automated security updates.
  - Enabled GitHub Secret Scanning and Push Protection.
  - Configured GitHub Ruleset protecting `main` from force pushes and requiring PR reviews.
  - Established standardized GitHub label taxonomy (27 labels across Type, Area, Priority, Phase).
  - Authored standard GitHub Issue templates (Feature, Bug, Research, Experiment, Docs) and PR template.
  - Established Master Development Roadmap `docs/roadmap/SANKET_ROADMAP.md`.
  - Initialized Architecture Decision Record system with ADR-0001 `docs/decisions/ADR-0001-project-foundations.md`.
  - Established Dataset Register `docs/datasets/DATASET_REGISTER.md` and Experiment tracking framework.
  - Configured Python project packaging with `pyproject.toml`, `requirements.txt`, and GitHub Actions CI.
  - Created Milestone 0 on GitHub and populated issues `SANKET-001` through `SANKET-012`.
- **Decisions**:
  - Adopted strict sequential Phase gating (ADR-0001).
  - Deferred all ML and model training until Phase 0 foundation, dataset audit, and vocabulary definitions are frozen.
  - Enforced Signer-Independent evaluation strategy to prevent biometric data leakage.
- **Experiments**:
  - No ML experiments conducted yet (Phase 0).
- **Problems & Mitigations**:
  - *Problem*: GitHub Wiki git repository is not provisioned until the first page is created.  
    *Mitigation*: Authored comprehensive Wiki information architecture and drafting specifications in `docs/architecture/WIKI_STRUCTURE.md`.
- **Results**:
  - Repository security posture validated (Dependabot, rulesets, secret scanning).
  - Unit tests verify packaging and baseline environment cleanly.
- **Next Actions**:
  - Review and merge PR #13 on GitHub.
  - Complete SANKET-008 (Dataset research and comparison).

---

### 2026-10-28 — SANKET-008: ISL Dataset Research & Comparison

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-008
- **Completed Work**:
  - Conducted a multi-dimensional comparative survey of 6 candidate ISL datasets: INCLUDE, INCLUDE-50, ISL-CSLTR, CISLR, IIIT-Delhi ISL, and static finger-spelling benchmarks.
  - Benchmarked datasets across modality, resolution, framerate, vocabulary scope, signer demographics, background clutter, and licensing terms in `docs/research/DATASET_SURVEY.md`.
  - Evaluated MediaPipe tracking feasibility on one-handed, two-handed, and overlapping sign morphology.
  - Authored empirical trade-off analysis comparing full-scale datasets vs constrained subsets.
- **Decisions**:
  - Selected **INCLUDE** (specifically a 15-class high-frequency subset matching `INITIAL_VOCABULARY.md`) as the primary training and evaluation benchmark for Phase 1.
  - Deferred **ISL-CSLTR** to Phase 3/4 due to controlled studio recording conditions and continuous sentence focus.
  - Excluded static alphabet datasets from primary research claims to ensure genuine ISL vocabulary fidelity.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Problems & Mitigations**:
  - *Problem*: Full INCLUDE dataset is ~15 GB and includes 263 classes, which is too complex for an initial isolated baseline.  
    *Mitigation*: Restrict Phase 1 ingestion to a 15-class subset (~1.2 GB), enabling fast, reproducible CPU baseline training while preserving authentic linguistic vocabulary.
- **Results**:
  - Comprehensive survey and recommendation completed and recorded in `docs/research/DATASET_SURVEY.md`.
- **Next Actions**:
  - Submit Pull Request for SANKET-008.
  - Proceed to SANKET-009 (Dataset licensing & access review) to formally audit INCLUDE's CC BY 4.0 terms in `DATASET_REGISTER.md`.


