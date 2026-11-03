# SANKET (संकेत) — Changelog

> **All notable changes to the SANKET project are documented in this file.**  
> *The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).*

---

## [Unreleased] — Phase 2: Real-Time Sanket MVP

### Active Milestone
- Assigned to `Milestone 2 — Real-Time Sanket MVP` (Issues #51–#60).
- Building live webcam pipeline, real-time MediaPipe streaming tracker, inference worker, temporal smoothing, confidence thresholding, accessible HUD overlay, and latency benchmarking (< 100 ms).

---

## [0.1.0] — 2026-11-01: Phase 1 Certification & Baseline Classifiers

### Added
- **Feature Extraction Pipeline** (`sanket.features`):
  - 645-dimensional feature representation (raw coords, normalized coords, joint angles, pairwise distances, temporal moments) codified in ADR-0002.
- **Baseline Classifiers** (`sanket.models`):
  - `RandomForestSignClassifier` (ADR-0003 selected baseline, 93.3% test accuracy, 0.73 ms p95 latency).
  - `SVMSignClassifier` with RBF kernel and integrated `StandardScaler`.
  - `MLPSignClassifier` with Adam optimizer and ReLU activation.
- **Evaluation & Diagnostics Suite** (`sanket.evaluation`):
  - Unified evaluation harness in `sanket.evaluation.evaluator.ModelEvaluator`.
  - High-resolution CPU latency profiling (`sanket.evaluation.latency`).
  - Structured JSON evaluation manifest schema (`sanket.evaluation.manifest`).
  - Confusion matrix and phonological error analysis diagnostics (`sanket.evaluation.diagnostics`).
- **Research Decisions & Empirical Reports**:
  - `docs/decisions/ADR-0002-feature-representation.md`.
  - `docs/decisions/ADR-0003-baseline-classifier-selection.md`.
  - `docs/experiments/EXP-P1-001-baseline-comparison.md`.
  - `docs/experiments/ERROR_ANALYSIS_PHASE1.md`.
- **Engineering Governance & Hardening**:
  - Automated PR metadata validation workflow (`.github/workflows/pr-metadata.yml`).
  - Least-privilege permissions and immutable commit SHA pinning in `.github/workflows/ci.yml`.
  - 18 educational Wiki draft pages in `docs/wiki_drafts/`.

---

## [0.0.1] — 2026-10-29: Phase 0 Foundation & Architecture

### Added
- **Repository Scaffolding**: Standard Python packaging with `pyproject.toml`, Ruff linting, and Pytest configuration.
- **Dataset Governance**: Established `docs/datasets/DATASET_REGISTER.md` with SHA-256 integrity tracking and owner-approval requirements.
- **Initial Vocabulary**: Codified 16-sign initial ISL vocabulary in `docs/research/INITIAL_VOCABULARY.md`.
- **Signer-Independent Split Strategy**: Implemented zero-leakage signer-independent splitting algorithm in `sanket.data.splits`.
- **Landmark Extraction**: Integrated MediaPipe Hands (21 keypoints) and investigated MediaPipe Pose upper-body anchors.
- **Foundational ADR**: Documented architecture decisions in `docs/decisions/ADR-0001-project-foundations.md`.

