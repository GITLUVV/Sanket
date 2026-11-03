# SANKET PROJECT RECOVERY AUDIT REPORT

- **Audit Date**: 2026-11-01
- **Auditor**: Sanket Team
- **Canonical Repository**: `https://github.com/GITLUVV/Sanket.git`
- **Audit Target Branch / Commit**: `main` @ `3abef44`
- **Total Test Suite Health**: 140/140 unit tests passing (100% pass rate)

---

## 1. Actual Project Status

| Dimension | Declared in README | Actual Implementation State | Status Assessment |
|---|---|---|---|
| **Active Phase** | Phase 0 — Engineering Foundation | **Phase 1 — Data & Isolated Sign Recognition** | **Discrepancy (Outdated README)** |
| **Active Milestone** | Milestone 0 (Closed) | **Milestone 1 — Data & Isolated Sign Recognition** (Open, 18 closed, 5 open) | **Active in Phase 1** |
| **Model Development** | "Not yet started" | **Baseline Random Forest implemented, tested, and merged** (`sanket.models`) | **Discrepancy (Outdated README)** |
| **Codebase Size** | Initial skeleton | 23 core modules, 10 CLI scripts, 10 test modules, 140 unit tests | **Substantially Advanced** |
| **Remote Repository URL** | `anveshna25/Sanket` | `GITLUVV/Sanket` | **Discrepancy (Repository migration)** |

The repository is **not** at the beginning of Phase 0. Phase 0 has been completed and verified. The repository is actively halfway through **Phase 1 (Data & Isolated Sign Recognition)**. 9 of the 14 Phase 1 tasks (SANKET-020 through SANKET-028) are implemented, verified by unit tests, and merged into `main`. Exactly 5 tasks remain in Phase 1 (SANKET-029 through SANKET-033).

---

## 2. Completed and Verified Work

Every item below has been verified against committed code on `main`, execution of corresponding unit tests, and Git merge records.

### Phase 0: Engineering & Research Foundation (100% Merged & Verified)
- **SANKET-001 (PR #13)**: Core project layout, packaging configuration (`pyproject.toml`), `scripts/verify_setup.py`, and base test suite.
- **SANKET-002 (Issue #2)**: GitHub Wiki information architecture formalized in `docs/architecture/WIKI_STRUCTURE.md`.
- **SANKET-003 (Issue #3)**: Security & vulnerability disclosure policy in `SECURITY.md`.
- **SANKET-004 (Issue #4)**: Branching policy, PR workflow, and human merge control in `CONTRIBUTING.md`.
- **SANKET-005 (Issue #5)**: Issue templates (`bug`, `feature`, `research`, `experiment`, `documentation`) and PR template in `.github/`.
- **SANKET-006 (Issue #6)**: Standardized label taxonomy across Type, Area, Priority, and Phase dimensions.
- **SANKET-007 (Issue #7)**: Documentation framework establishing `docs/` hierarchy, ADR template, and project log.
- **SANKET-008 (PR #14)**: Comprehensive survey of ISL corpora (INCLUDE, ISL-CSLTR, CISLR, IIIT-Delhi) in `docs/research/DATASET_SURVEY.md`.
- **SANKET-009 (PR #15)**: Dataset legal licensing audit, ethical sign-off protocol, and non-interference clause in `docs/datasets/DATASET_REGISTER.md`.
- **SANKET-010 (PR #16)**: Constrained 16-sign isolated ISL vocabulary baseline in `docs/research/INITIAL_VOCABULARY.md`.
- **SANKET-011 (PR #17)**: Formal research questions (RQ1–RQ6) and empirical hypotheses in `docs/research/RESEARCH_QUESTIONS.md`.
- **SANKET-012 (PR #18)**: Signer-independent evaluation protocol and metric hierarchy in `docs/research/EVALUATION_METHODOLOGY.md`.

### Phase 1: Data & Isolated Sign Recognition (64% Merged & Verified)
- **SANKET-020 (PR #36)**: Automated dataset ingestion pipeline (`sanket.data.ingestion`), directory structure validation, SHA-256 integrity checks, and CLI (`scripts/ingest_dataset.py`). 9/9 tests pass.
- **SANKET-021 (PR #37)**: Dataset inspection and label balance analysis engine (`sanket.data.inspection`), video metadata validation, and CLI (`scripts/inspect_dataset.py`). 6/6 tests pass.
- **SANKET-022 (PR #38)**: Signer-independent partitioning (`sanket.data.splits`), strict signer isolation preventing leakage, reproducible manifest generator (`configs/splits/include16_signer_independent.json`), and CLI (`scripts/create_splits.py`). 8/8 tests pass.
- **SANKET-023 (PR #39)**: Video preprocessing pipeline (`sanket.vision.preprocessing`), uniform temporal subsampling (16/32 frames), aspect-ratio-preserving letterbox resizing (224x224), and CLI (`scripts/preprocess_videos.py`). 28/28 tests pass.
- **SANKET-024 (PR #40)**: MediaPipe 3D hand landmark extraction (`sanket.vision.landmarks`), dual-hand tracking, linear temporal imputation for dropped detections, and CLI (`scripts/extract_landmarks.py`). 19/19 tests pass.
- **SANKET-025 (PR #41)**: Upper-body pose landmark investigation (`sanket.vision.pose`), anatomical reference anchors (nose, shoulders, elbows, wrists), research report `docs/research/POSE_LANDMARK_INVESTIGATION.md`, and CLI (`scripts/investigate_pose.py`). 14/14 tests pass.
- **SANKET-026 (PR #42)**: Skeletal coordinate normalization (`sanket.vision.normalization`), wrist-centering ($L_0$ origin), metacarpal palm scaling ($\|L_9 - L_0\|_2 = 1.0$), optional 2D in-plane rotation alignment, exact inverse denormalization, and CLI (`scripts/normalize_landmarks.py`). 20/20 tests pass.
- **SANKET-027 (PR #43)**: Feature vector generation subsystem (`sanket.features`):
  - Joint angles: 15 flexion + 4 abduction per hand ($38$ angles for two hands) in $[0, \pi]$.
  - Anatomical pairwise distances: 24 per hand ($48$ total) with proven translation invariance ($atol = 10^{-4}$).
  - Inter-hand spatial metrics: 3 distances (inter-wrist, inter-index-tip, inter-thumb-tip).
  - Temporal pooling: mean, standard deviation, and delta dynamics ($3 \times 215 = 645$-dim pooled tabular representation).
  - Operational CLI: `scripts/generate_features.py`.
  - Architecture Decision Record: `docs/decisions/ADR-0002-feature-representation.md`.
  - 20/20 tests pass.
- **SANKET-028 (PR #44)**: Baseline Random Forest model subsystem (`sanket.models`):
  - Base architecture contract: `BaseSignClassifier` ABC and `SignModelMetrics` dataclass (`src/sanket/models/base.py`).
  - Model implementation: `RandomForestSignClassifier` (`src/sanket/models/random_forest.py`) with probability calibration, top-k ranking, Gini feature importance ranking (`get_top_features`), validation hyperparameter search (`tune_hyperparameters`), and atomic joblib persistence (`save`/`load`).
  - Operational CLI: `scripts/train_random_forest.py` supporting training from extracted feature directories and `--synthetic-demo` mode.
  - Benchmarked latency: ~0.73 ms/sample (well below the $\le 50$ ms budget for RQ5).
  - 12/12 tests pass.

---

## 3. Partially Completed Work

1. **GitHub Wiki Content**:
   - `docs/architecture/WIKI_STRUCTURE.md` outlines a 13-page taxonomy.
   - However, the GitHub Wiki Git repository (`https://github.com/GITLUVV/Sanket.wiki.git`) is currently uninitialized because GitHub requires the first page to be created via the web interface. Draft page files exist in documentation specifications but have not yet been published to the live Wiki.
2. **Master Development Roadmap Checklist**:
   - `docs/roadmap/SANKET_ROADMAP.md` contains the complete 7-phase roadmap, but the phase gate checkboxes for Phase 0 and completed Phase 1 tasks were never checked off.
3. **Experiment Records**:
   - `docs/experiments/` contains `README.md` and `EXPERIMENT_TEMPLATE.md`, but no formal experiment logs (e.g. `EXP-0001-random-forest-baseline.md`) have been committed.

---

## 4. Outstanding Work (Phase 1 Remaining Scope)

Exactly five open issues remain to conclude Phase 1 (Milestone 1):

| Issue | Title | Type | Area | Priority | Phase |
|---|---|---|---|---|---|
| **#31** | **SANKET-029**: Baseline SVM model | `type:feature` | `area:ml` | `priority:P1` | `phase:1` |
| **#32** | **SANKET-030**: Baseline neural network (MLP) | `type:feature` | `area:ml` | `priority:P1` | `phase:1` |
| **#33** | **SANKET-031**: Model evaluation pipeline | `type:feature` | `area:testing` | `priority:P0` | `phase:1` |
| **#34** | **SANKET-032**: Confusion matrix and error analysis | `type:research` | `area:testing` | `priority:P1` | `phase:1` |
| **#35** | **SANKET-033**: Baseline model comparison | `type:research` | `area:ml` | `priority:P0` | `phase:1` |

Completing these five issues satisfies the Phase 1 Gate criteria (offline baselines trained on identical signer-independent splits, compared, and analyzed with confusion matrices).

---

## 5. Broken or Missing Functionality

- **No broken code detected**: All 140 unit tests pass on Python 3.12 locally and across Python 3.10, 3.11, and 3.12 on GitHub Actions CI.
- **Ruff linting**: 0 errors across the entire codebase.
- **Foundation verification**: `scripts/verify_setup.py` passes all 4/4 checks.
- **Missing**:
  - Standalone PR metadata validation GitHub Action workflow.
  - Experiment report markdown files in `docs/experiments/`.
  - Initialized GitHub Wiki git repository.

---

## 6. Documentation Inconsistencies (Reconciliation Analysis)

### Discrepancy A: README Declares Phase 0
- **Evidence**: `README.md` lines 4 and 12–16 state:
  - Badge: `Phase-0_Engineering_Foundation`
  - Active Phase: `Phase 0 — Engineering & Research Foundation`
  - Model Development: `Not yet started. ML training begins in Phase 1...`
- **Fact**: Milestone 0 is closed on GitHub. Phase 1 has 9 merged issues/PRs including Random Forest baseline model code.
- **Remediation**: Update `README.md` to reflect Phase 1 actively in progress, Milestone 1 open, Random Forest baseline merged, and 5 outstanding Phase 1 issues.

### Discrepancy B: PROJECT_LOG.md vs Roadmap Sync
- **Evidence**: `docs/development/PROJECT_LOG.md` accurately logs Phase 0 and Phase 1 up to SANKET-028. However, `docs/roadmap/SANKET_ROADMAP.md` still lists Phase 0 phase gate checkboxes as uncompleted (`- [ ]`).
- **Remediation**: Synchronize `docs/roadmap/SANKET_ROADMAP.md` phase gate checklists with verified reality.

### Discrepancy C: Five Open Issues in Phase 1
- **Evidence**: `gh issue list` confirms issues #31, #32, #33, #34, and #35 are open and assigned to Milestone 1.
- **Fact**: These correspond exactly to the second half of Phase 1 (SVM, MLP, evaluation pipeline, confusion matrix/error analysis, and baseline model comparison).
- **Remediation**: Retain these 5 issues as the authoritative remaining backlog for Phase 1.

### Discrepancy D: References to PRs #13–#18
- **Evidence**: `PROJECT_LOG.md` refers to PRs #13–#18.
- **Fact**: PRs #13–#18 exist on GitHub, are merged into `main`, and represent the Phase 0 foundational deliverables.

### Discrepancy E: Canonical Repository URL
- **Evidence**: `README.md` (lines 14, 62) and `WIKI_STRUCTURE.md` reference `https://github.com/anveshna25/Sanket.git` and `anveshna25/Sanket`.
- **Fact**: The canonical repository was moved to `https://github.com/GITLUVV/Sanket.git`.
- **Remediation**: Update all repository clone commands, badges, and documentation links to `GITLUVV/Sanket`.

---

## 7. GitHub Governance Gaps

1. **Branch Protection / Ruleset Enforcement**:
   - Ruleset `Protect main branch` (ID `26011645`) is active and protects `refs/heads/main` against deletion and non-fast-forward pushes.
   - **Gap**: The ruleset does not enforce required status checks (e.g. `lint-and-test`). CI runs, but technically a merge is not blocked if CI fails.
   - **Remediation**: Add required status check rule (`lint-and-test`) to the main branch ruleset.
2. **Automated PR Metadata Enforcement**:
   - **Gap**: There is no automated GitHub Action workflow checking that a PR links to a valid issue (`Closes #XX`), has a milestone, has type/area/priority/phase labels, and follows the title convention.
   - **Remediation**: Introduce `.github/workflows/pr-metadata.yml` to validate PR title, milestone, labels, linked issue, and template compliance.
3. **Branch Naming Standard**:
   - `CONTRIBUTING.md` specifies `feature/SANKET-XXX-...` while user prompt suggests `feat/ISSUE-ID-...`.
   - **Harmonization**: Standardize on `feat/SANKET-XXX-short-description` and `research/SANKET-XXX-short-description`, maintaining traceability to SANKET issue identifiers.

---

## 8. Security Gaps & Analysis

1. **GitHub Actions Security**:
   - In `.github/workflows/ci.yml`:
     - Missing explicit least-privilege `permissions: contents: read`.
     - Action steps use mutable version tags (`@v7`) instead of pinned immutable commit SHAs.
2. **Deserialization Safety**:
   - `joblib` is used for `RandomForestSignClassifier` serialization. While standard in scikit-learn, joblib uses pickle internally. `save()` and `load()` must restrict loading to trusted local paths and document deserialization boundaries.
3. **Biometric & Camera Privacy**:
   - `.gitignore` strictly ignores raw video formats (`*.mp4`, `*.avi`, `*.mov`, `*.webm`), dataset directories (`/data/`, `datasets/raw/`), and image files.
   - Verified: Zero camera video files, facial pixel arrays, or identifiable biometric media files exist in Git history.
4. **Supply Chain & Dependencies**:
   - Dependabot is configured (`.github/dependabot.yml`) for pip and GitHub Actions. PRs #19, #20, #21 bumped actions and types dependencies.

---

## 9. Recommended Next Actions (Prioritized Sequence)

1. **Stage B**: Resolve documentation discrepancies:
   - Update `README.md` to reflect `GITLUVV/Sanket`, Phase 1 active status, and implemented components.
   - Update `docs/roadmap/SANKET_ROADMAP.md` phase gates for Phase 0 and completed Phase 1 tasks.
   - Author `docs/experiments/EXP-0001-random-forest-baseline.md` capturing SANKET-028 metrics and benchmark latency.
2. **Stage C**: Governance and CI hardening:
   - Introduce `.github/workflows/pr-metadata.yml` for automated PR metadata validation.
   - Add least-privilege `permissions: contents: read` to `.github/workflows/ci.yml` and pin action SHAs.
   - Update `.github/pull_request_template.md` and `CONTRIBUTING.md`.
3. **Stage D**: Wiki preparation:
   - Prepare complete draft content for the 18 required Wiki pages in `docs/wiki_drafts/` so they are ready for publishing once the Wiki git repository is initialized.
4. **Stage E**: Phase 0 formal sign-off report.
5. **Stage F**: Sequence and execute the 5 remaining Phase 1 issues:
   - SANKET-029 (SVM baseline)
   - SANKET-030 (MLP baseline)
   - SANKET-031 (Model evaluation pipeline)
   - SANKET-032 (Confusion matrix and error analysis)
   - SANKET-033 (Baseline model comparison)

