# SANKET (संकेत) — Backup, Artifact Retention & Disaster Recovery

> **Model Checkpoint Backups, Split Integrity, Dataset Hashes, and Environment Recovery**  
> *Authoritative procedures for safeguarding research assets and recovering operational baselines.*

---

## 1. Research Asset Inventory & Retention Strategy

| Asset Class | Primary Location | Backup / Archival Strategy | Integrity Check |
| :--- | :--- | :--- | :--- |
| **Source Code & Docs** | Git Repository (`main`) | Remote mirrored on GitHub (`GITLUVV/Sanket`). | Git tree SHA-1/SHA-256. |
| **Model Weights** | `models/*.joblib` | Versioned model checkpoints tagged with git commit hashes in GitHub Releases. | SHA-256 hash in experiment manifest. |
| **Split Configurations** | `configs/splits/*.json` | Version-controlled in repository. | Cryptographic hash recorded in split JSON. |
| **Experiment Manifests** | `docs/experiments/manifests/` | Version-controlled in repository. | Verified against `EvaluationManifest` schema. |
| **Licensed Datasets** | Local external storage (`data/`) | Re-creatable from authoritative upstream sources registered in `DATASET_REGISTER.md`. | Verified via `sha256sum`. |

---

## 2. Model Weight Checkpoint Backups

Trained model artifacts are deterministic and reproducible. To preserve validated baselines:

1. **Local Checkpoint Copy**:
   ```bash
   # Create dated backup of trained models
   mkdir -p backups/models
   cp models/*.joblib backups/models/
   ```

2. **Publishing Checkpoints to GitHub Releases**:
   When tagging a release (e.g., `v0.1.0`), attach serialized `.joblib` model artifacts to the GitHub Release:
   ```bash
   gh release upload v0.1.0 models/baseline_random_forest.joblib models/baseline_svm.joblib models/baseline_mlp.joblib
   ```

---

## 3. Disaster Recovery Procedures

### 3.1 Corrupted Virtual Environment
If Python dependencies become mismatched or corrupted:
```bash
# Deactivate and delete corrupted venv
deactivate
rm -rf venv/

# Rebuild fresh virtual environment
python -m venv venv
# Activate (Windows: .\venv\Scripts\Activate.ps1; Linux: source venv/bin/activate)
pip install --upgrade pip
pip install -e ".[dev]"
python scripts/verify_setup.py
```

### 3.2 Corrupted Model Artifact Recovery
If a model artifact (`.joblib`) is accidentally deleted or corrupted:
1. Re-train from feature arrays:
   ```bash
   python scripts/train_random_forest.py \
       --features-train data/features/features_train.npz \
       --output-model models/baseline_random_forest.joblib
   ```
2. Verify against the benchmark:
   ```bash
   python scripts/evaluate_model.py \
       --model-path models/baseline_random_forest.joblib \
       --output-manifest docs/experiments/manifests/EXP-RECOVERY.json
   ```

