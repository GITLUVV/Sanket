# Machine Learning Experiment Records — Sanket

This directory preserves the empirical research records of all machine learning experiments conducted in the Sanket project.

## Rules for Experimentation
1. **Never Overwrite**: Never overwrite or delete historical experiment results. Create a new markdown file for each experiment (`EXP-0001-...md`, `EXP-0002-...md`).
2. **Reproducibility**: Every recorded experiment must state the exact dataset version, random seed, feature configuration, and hyperparameters.
3. **Signer Independence**: Note whether the experiment used a Signer-Independent split to ensure fair peer review.

---

## Experiment Registry

| Experiment ID | Date | Objective | Model | Dataset | Top-1 Test Acc | Latency | Status |
| :------------ | :--- | :-------- | :---- | :------ | :------------- | :------ | :----- |
| [**EXP-0001**](EXP-0001-random-forest-baseline.md) | 2026-10-31 | Baseline Random Forest isolated sign classifier | RandomForest | INCLUDE-16 (Signer-Independent) | 0.9531 | 0.73 ms | Completed |

---

## How to Record an Experiment
1. Copy `docs/experiments/EXPERIMENT_TEMPLATE.md` to `EXP-XXXX-<slug>.md`.
2. Fill out all sections honestly, including negative results, confusion errors, and latency numbers.
3. Add a row to the table above.
4. Link the record in the corresponding GitHub Issue or Pull Request.
