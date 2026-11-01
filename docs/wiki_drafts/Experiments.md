# Experiment Records

All machine learning experiments in Sanket are permanently recorded in `docs/experiments/`.

## Registry Overview
- **EXP-0001**: Baseline Random Forest classifier on INCLUDE-16 (Accuracy: 0.9531, Macro F1: 0.9488, Latency: 0.73 ms).
- **EXP-0002** (Planned): Support Vector Machine baseline.
- **EXP-0003** (Planned): Multi-Layer Perceptron neural network baseline.
- **EXP-P1-001** (Planned): Definite Phase 1 comparative synthesis.

## Experiment Reporting Rules
1. Never overwrite historical records.
2. Always report both Top-1 Accuracy and Macro-averaged F1 score.
3. Record inference latency percentiles and throughput on CPU.
4. Identify top confused sign pairs and qualitative failure causes.
