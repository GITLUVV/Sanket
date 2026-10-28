---
name: ML Experiment Record
about: Plan, execute, and record an empirical machine learning experiment.
title: "[EXPERIMENT] EXP-"
labels: ["type:experiment"]
assignees: ""
---

## Experiment ID
<!-- e.g. EXP-001-rf-vs-svm-isolated -->

## Objective
<!-- What are we testing in this specific experiment run? -->

## Baseline
<!-- What existing model, metric, or heuristic serves as the comparator? -->

## Variables
- **Independent Variables** (changed):
- **Controlled Variables** (kept constant):

## Dataset
- Dataset Name & Version:
- Split Strategy (e.g., Signer-Independent 70/15/15):
- Number of Classes:
- Total Samples (Train/Val/Test):

## Model & Architecture
- Algorithm / Backbone:
- Feature Representation (e.g. 21 hand landmarks, normalized x/y/z coordinates):
- Hyperparameters:

## Metrics
- Primary Metric: [e.g. Test Top-1 Accuracy]
- Secondary Metrics: [e.g. Macro F1, Inference Latency per frame (ms), Model file size (MB)]

## Results
| Model / Configuration | Val Accuracy | Test Accuracy | Macro F1 | Latency (ms) |
| :-------------------- | :----------- | :------------ | :------- | :----------- |
| Baseline              |              |               |          |              |
| Candidate Run         |              |               |          |              |

## Interpretation & Error Analysis
<!-- Discussion of confusion matrices, failure modes, misclassified signs, and edge cases. -->

## Decision
- [ ] Adopt candidate model into pipeline
- [ ] Retain baseline model
- [ ] Conduct follow-up experiment
- **Rationale**:
