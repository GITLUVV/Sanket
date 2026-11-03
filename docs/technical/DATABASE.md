# SANKET (संकेत) — Data Persistence & Storage Architecture

> **Dataset Storage, Structured Artifacts, Split Schemas, and Model Serialization**  
> *Authoritative specification of file-based data structures and artifact persistence.*

---

## 1. Storage Paradigm

> [!NOTE]
> SANKET does not utilize a traditional relational database (RDBMS) or NoSQL document store. As a real-time computer vision and machine learning system, its persistence layer is entirely **file-based, structured artifact-driven, and version-controlled**.

This architecture guarantees:
- **Portability**: Zero external database server dependencies; runs locally on any workstation or edge device.
- **Reproducibility**: Datasets, train/val/test splits, and evaluation manifests are represented as human-readable JSON/YAML files with explicit cryptographic checksums.
- **Privacy Compliance**: Raw video frames remain transient in memory; only abstract numerical coordinates and features are saved.

---

## 2. Directory & Storage Hierarchy

```text
Sanket/
├── configs/
│   ├── default.yaml                 # Primary system configuration
│   └── splits/
│       └── signer_independent.json   # Deterministic split definition
├── data/                            # Local working data (Git-ignored)
│   ├── raw/                         # Raw downloaded video archives (licensed)
│   ├── processed/                   # Standardized frames / subclips
│   ├── landmarks/                   # Extracted MediaPipe NumPy arrays (.npy)
│   └── features/                    # Computed 645-dim feature matrices (.npz)
├── docs/
│   ├── datasets/
│   │   ├── DATASET_REGISTER.md      # Authoritative dataset registry
│   │   └── INCLUDE_INSPECTION.md    # Class balance and signer metadata
│   └── experiments/
│       └── manifests/               # Standardized evaluation JSON manifests
└── models/                          # Serialized trained model estimators
    ├── baseline_random_forest.joblib
    ├── baseline_svm.joblib
    └── baseline_mlp.joblib
```

---

## 3. Structured File Schemas

### 3.1 Signer-Independent Split Schema (`configs/splits/signer_independent.json`)
Guarantees zero data leakage by strictly partitioning signers across sets:

```json
{
  "dataset_name": "INCLUDE_ISOLATED_16",
  "version": "1.0.0",
  "seed": 42,
  "signer_split": {
    "train_signers": ["signer_01", "signer_02", "signer_03", "signer_04", "signer_05"],
    "val_signers": ["signer_06"],
    "test_signers": ["signer_07"]
  },
  "sample_counts": {
    "train": 480,
    "val": 96,
    "test": 96
  },
  "vocabulary_size": 16,
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

### 3.2 Evaluation Manifest Schema (`docs/experiments/manifests/EXP-*.json`)
Constructed by `sanket.evaluation.manifest.EvaluationManifest`:

```json
{
  "experiment_id": "EXP-P1-001-RF",
  "timestamp": "2026-11-01T15:30:00Z",
  "git_commit": "bd7aa10255c1b0a08295e407f5442bcc2b05e32e",
  "model_type": "RandomForestSignClassifier",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": 42
  },
  "dataset": {
    "split_id": "signer_independent_v1",
    "test_samples": 96,
    "num_classes": 16
  },
  "metrics": {
    "top_1_accuracy": 0.9333,
    "top_5_accuracy": 1.0000,
    "macro_precision": 0.9412,
    "macro_recall": 0.9333,
    "macro_f1": 0.9333,
    "mcc": 0.9306
  },
  "latency": {
    "median_ms": 0.56,
    "p90_ms": 0.68,
    "p95_ms": 0.73,
    "p99_ms": 0.91,
    "throughput_fps": 1507.2
  }
}
```

---

## 4. Model Artifact Serialization (`.joblib`)

- Trained models are serialized using `joblib.dump(model, path, compress=3)`.
- Compression level 3 balances disk size reduction with fast sub-second deserialization during runtime startup.
- Deserialization is protected by strict typing checks verifying that the unpickled object inherits from `BaseSignClassifier`.

