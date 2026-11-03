---
name: model-evaluation
description: Executes the standardized SANKET model evaluation harness, measures CPU latency percentiles, and generates structured JSON experiment manifests.
---

# Model Evaluation Skill

Use this procedure to evaluate trained model artifacts (`.joblib`) against the Signer-Independent test benchmark according to `docs/research/EVALUATION_METHODOLOGY.md`.

## Execution Steps

1. **Run Model Evaluation**:
   ```bash
   python scripts/evaluate_model.py \
       --model-path models/baseline_random_forest.joblib \
       --output-manifest docs/experiments/manifests/EXP-P2-EVAL-RF.json
   ```

2. **Verify Benchmark Gates**:
   Ensure the output report satisfies the Phase 1 & Phase 2 exit criteria:
   - Macro F1 >= 0.85
   - Top-1 Accuracy >= 90.0%
   - Latency: p95 <= 25.0 ms, Median <= 15.0 ms
   - CPU Throughput: >= 25 FPS

3. **Inspect Output Manifest**:
   Confirm that the generated manifest file in `docs/experiments/manifests/` complies with the schema defined in `sanket.evaluation.manifest.EvaluationManifest`.

