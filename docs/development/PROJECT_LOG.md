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
  - Review and merge PR #14 (SANKET-008).
  - Complete SANKET-009 (Dataset licensing & access review).

---

### 2026-10-28 — SANKET-009: Dataset Licensing & Access Review

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-009
- **Completed Work**:
  - Formally audited 4 candidate datasets in `docs/datasets/DATASET_REGISTER.md`: INCLUDE / INCLUDE-50, ISL-CSLTR, CISLR, and IIIT-Delhi ISL.
  - Formulated the Project Licensing Non-Interference Policy, codifying that Sanket's Apache-2.0 license applies solely to software/docs and does not supersede third-party dataset restrictions.
  - Established a mandatory Human Project-Owner Approval Gate prior to raw video download or local dataset ingestion.
  - Audited biometric consent, facial identification risks, and derivative mathematical feature rights (MediaPipe 3D coordinate tensors).
- **Decisions**:
  - Approved **INCLUDE** under Creative Commons Attribution 4.0 (CC BY 4.0) as legally and ethically sound for Phase 1 isolated baseline training.
  - Deferred **ISL-CSLTR** to Phase 3/4 pending formal academic MoU execution.
  - Codified strict policy: No raw biometric video footage may ever be committed or pushed to the GitHub repository.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Results**:
  - `docs/datasets/DATASET_REGISTER.md` fully audited, verified, and gated.
- **Next Actions**:
  - Review and merge PR #15 (SANKET-009).
  - Complete SANKET-010 (Initial ISL vocabulary definition).

---

### 2026-10-28 — SANKET-010: Initial ISL Vocabulary Definition

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-010
- **Completed Work**:
  - Defined a frozen 16-sign constrained vocabulary of high-frequency isolated Indian Sign Language signs in `docs/research/INITIAL_VOCABULARY.md`.
  - Balanced vocabulary across 3 essential communicative domains: Greetings & Social Interaction, Core Needs & Assistance, and Emergency & Descriptive Signs.
  - Documented complete phonological and articulatory profiles for each of the 16 signs, specifying handedness, Stokoe-style handshape classes, spatial location/signing space, orientation vectors, and movement dynamics.
  - Formulated specific MediaPipe skeletal landmark extraction pipelines for each sign, mapping landmark geometric relationships (e.g. wrist-to-palm distance, finger joint angles, face anchor proximity).
  - Performed pairwise confusion risk analysis across morphologically adjacent signs (e.g., `DOCTOR` vs `TIME`, `WATER` vs `FOOD`, `HELLO` vs `PLEASE`, `GOOD` vs `YES`, `STOP` vs `HELP`), detailing primary geometric disambiguation features.
  - Established a formal vocabulary freeze protocol and project owner sign-off gate before dataset ingestion.
- **Decisions**:
  - Selected 16 signs aligning directly with the classes available in the CC BY 4.0 INCLUDE benchmark dataset to ensure unblocked data availability.
  - Excluded static 2D finger-spelling / alphabet symbols from vocabulary to prioritize authentic dynamic ISL lexical signs.
  - Mandated dual-hand + face anchor keypoint tracking (MediaPipe Hands + Pose) to disambiguate facial proximity and two-handed interaction.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Problems & Mitigations**:
  - *Problem*: High geometric similarity between signs targeting similar body regions (e.g., `DOCTOR` and `TIME` both contacting the non-dominant wrist).  
    *Mitigation*: Formulated precise vector-angle and palmar vs dorsal surface approach rules in the feature pipeline to differentiate volar arterial contact from dorsal wristwatch tapping.
- **Results**:
  - `docs/research/INITIAL_VOCABULARY.md` fully specified with 16 linguistic profiles and confusion matrices.
- **Next Actions**:
  - Submit Pull Request for SANKET-010.
  - Proceed to SANKET-011 (Research questions and hypotheses formulation).



