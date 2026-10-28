# ADR-0001: Project Architecture Foundations, Sequential Gating & Modular Layout

- **Status**: Accepted
- **Date**: 2026-10-28
- **Deciders**: Anveshna (Project Owner), Antigravity (AI Assistant)
- **Related Issue**: SANKET-001, SANKET-004, SANKET-007

---

## Decision
Establish a modular repository structure (`src/sanket/`, `docs/`, `tests/`, `configs/`, `scripts/`), protect the `main` branch with human-reviewed Pull Requests, enforce sequential phase progression gates, and defer all machine learning model training until the research problem, dataset licensing, and vocabulary baseline are formally validated.

---

## Context & Problem Statement
Sign language recognition is a technically challenging domain fraught with common academic pitfalls:
1. Overclaiming full translation when only recognizing a handful of isolated signs.
2. Data leakage across signers in random train/test splits.
3. Premature deep learning experimentation without baseline reproducibility.
4. Privacy and copyright liabilities from unvetted video datasets.

To build Sanket as an impactful, rigorous research software system, we must establish our engineering foundations before writing model code.

---

## Options Considered

1. **Option 1: Ad-hoc Rapid Prototyping (Monolithic Scripts)**
   - Train models directly in loose Jupyter notebooks or single scripts without structured packaging, ADRs, or phase gating.
   - *Pros*: Faster initial hackathon demo.
   - *Cons*: High technical debt, irreproducible results, high probability of signer leakage, lack of peer-review defensibility.

2. **Option 2: Modular Architecture with Sequential Phase Gating (Selected)**
   - Implement a modular Python package with distinct submodules (`sanket.vision`, `sanket.models`, `sanket.utils`), version-controlled documentation, formal ADRs, signer-independent evaluation, and human merge gates.
   - *Pros*: Reproducible, audit-defensible, modular interfaces allowing seamless swapping between classic ML and deep sequence models, strict privacy controls.
   - *Cons*: Requires upfront discipline and foundational engineering before initial model training.

---

## Chosen Approach
Option 2: Modular Architecture with Sequential Phase Gating.

---

## Reasons & Justification
- **Modularity**: Separating camera ingestion, landmark extraction (MediaPipe), feature normalization, classifier backends, and language rendering ensures changes to one stage (e.g. upgrading MediaPipe to holistic pose or swapping SVM for an MLP) do not destabilize the pipeline.
- **Scientific Rigor**: The ADR and Experiment tracking frameworks ensure research hypotheses can be tested, compared against reproducible baselines, and properly recorded.
- **Responsible AI**: Privacy safeguards and dataset licensing registers prevent legal and ethical liabilities regarding biometric video data.

---

## Consequences
- **Positive Consequences**:
  - Clear ownership and progress tracking via GitHub Milestones and Issues.
  - Safe, protected `main` branch preventing regression.
  - Transparent documentation for peer-reviewers, collaborators, and future contributors.
- **Negative Consequences / Trade-offs**:
  - Additional operational overhead to document experiments and open PRs.
- **Risks & Mitigation**:
  - *Risk*: Slower perceived early velocity.
  - *Mitigation*: The structured foundation eliminates rewrites and debugging chaos in later phases.
