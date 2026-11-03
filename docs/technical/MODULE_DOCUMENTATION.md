# SANKET (संकेत) — Comprehensive Module Documentation

> **In-Depth Engineering Breakdown of All Subsystems, Classes, and Internal Dependencies**  
> *Authoritative architectural catalog of `src/sanket/` packages.*

---

## 1. Subsystem: `sanket.core`

Foundation services, configuration schemas, and shared utilities.

### `sanket/core/config.py`
- **Role**: Validates, deserializes, and provides typed access to system configurations.
- **Classes**:
  - `CameraConfig`: Capture device ID, width, height, target FPS.
  - `LandmarkConfig`: Number of hands, detection/tracking confidence thresholds.
  - `ModelConfig`: Classifier selection, confidence cutoff, smoothing window size.
  - `SanketConfig`: Composite root configuration uniting all sub-configs.
- **Functions**:
  - `load_config(path: str | Path | None = None) -> SanketConfig`: Reads YAML with environment fallback.
- **Test Mapping**: `tests/test_foundation.py`.

---

## 2. Subsystem: `sanket.data`

Dataset ingestion, label distribution inspection, and signer-independent partitioning.

### `sanket/data/ingestion.py`
- **Role**: Manages download integrity, checksum verification, and archive extraction for licensed sign datasets.
- **Functions**:
  - `verify_checksum(file_path: Path, expected_sha256: str) -> bool`: Computes SHA-256 hash.
  - `extract_archive(archive_path: Path, output_dir: Path) -> None`: Safely extracts zip/tar archives.
- **Test Mapping**: `tests/test_data_ingestion.py`.

### `sanket/data/inspection.py`
- **Role**: Profiles dataset distributions, detects class imbalances, and verifies signer representations.
- **Functions**:
  - `inspect_dataset(data_dir: Path) -> DatasetReport`: Analyzes sample counts, unique classes, and signer distribution.
- **Test Mapping**: `tests/test_data_inspection.py`.

### `sanket/data/splits.py`
- **Role**: Generates reproducible Signer-Independent train, validation, and test splits preventing visual memorization.
- **Functions**:
  - `create_signer_independent_split(dataset_dir: Path, output_file: Path, seed: int = 42) -> dict`: Partitions signers with zero overlap.
- **Test Mapping**: `tests/test_data_splits.py`.

---

## 3. Subsystem: `sanket.vision`

Computer vision processing, camera acquisition, and landmark normalization.

### `sanket/vision/capture.py`
- **Role**: Acquires video frames asynchronously from physical webcams or deterministic mock video streams.
- **Classes**:
  - `CameraCapture`: Multi-threaded OpenCV capture worker with drop-frame detection.
  - `MockVideoCapture`: Deterministic frame feeder for headless CI testing.

### `sanket/vision/preprocessing.py`
- **Role**: Frame standardization, aspect-ratio-preserving letterboxing, and horizontal mirroring.
- **Functions**:
  - `letterbox_resize(image: np.ndarray, target_size=(640, 480)) -> np.ndarray`: Scales with padding.
  - `mirror_frame(frame: np.ndarray) -> np.ndarray`: Performs horizontal flip (`cv2.flip(frame, 1)`).
- **Test Mapping**: `tests/test_video_preprocessing.py`.

### `sanket/vision/normalization.py`
- **Role**: Mathematical coordinate normalization delivering translation and scale invariance.
- **Functions**:
  - `normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, float]`: Translates wrist to $(0,0,0)$ and scales by palm size.
  - `calculate_palm_scale(landmarks: np.ndarray) -> float`: Distance between wrist (0) and middle MCP (9).
- **Test Mapping**: `tests/test_landmark_normalization.py`.

### `sanket/vision/landmarks.py`
- **Role**: Extracts 21 3D coordinates per hand using Google MediaPipe Hands.
- **Functions**:
  - `extract_hand_landmarks(frame_rgb: np.ndarray, static_image_mode: bool) -> np.ndarray | None`: Produces `(2, 21, 3)` tensor.
- **Test Mapping**: `tests/test_landmarks.py`.

### `sanket/vision/pose.py`
- **Role**: Extracts upper-body spatial anchors (shoulders, elbows, nose) via MediaPipe Pose.
- **Functions**:
  - `extract_pose_landmarks(frame_rgb: np.ndarray) -> np.ndarray | None`: Produces upper-body anchor points.
- **Test Mapping**: `tests/test_pose_investigation.py`.

---

## 4. Subsystem: `sanket.features`

Feature vector engineering producing the standard 645-dimensional feature space (ADR-0002).

### `sanket/features/geometry.py`
- **Role**: Geometric feature generation across anatomical keypoints.
- **Functions**:
  - `calculate_joint_angles(landmarks: np.ndarray) -> np.ndarray`: Computes 60 joint angles.
  - `calculate_pairwise_distances(landmarks: np.ndarray) -> np.ndarray`: Computes 210 Euclidean distances.
- **Test Mapping**: `tests/test_features.py`.

### `sanket/features/temporal.py`
- **Role**: Temporal statistical pooling across sliding temporal windows.
- **Functions**:
  - `calculate_temporal_moments(sequence: np.ndarray) -> np.ndarray`: Computes 123 moment features.
- **Test Mapping**: `tests/test_features.py`.

### `sanket/features/extractor.py`
- **Role**: Unified feature extraction pipeline concatenating raw, normalized, geometric, and temporal features.
- **Classes**:
  - `FeatureExtractor`: Produces 645-dim feature vector per sample.
- **Test Mapping**: `tests/test_features.py`.

---

## 5. Subsystem: `sanket.models`

Machine learning baseline classifiers adhering to the `BaseSignClassifier` contract.

### `sanket/models/base.py`
- **Role**: Abstract Base Class establishing the model interface.
- **Classes**:
  - `BaseSignClassifier(ABC)`: Declares `fit`, `predict`, `predict_proba`, `save`, `load`.

### `sanket/models/random_forest.py`
- **Role**: Random Forest baseline classifier (ADR-0003 selected baseline).
- **Classes**:
  - `RandomForestSignClassifier(BaseSignClassifier)`: Scikit-learn Random Forest ensemble.
- **Test Mapping**: `tests/test_random_forest.py`.

### `sanket/models/svm.py`
- **Role**: Support Vector Machine baseline with RBF kernel and standard scaling.
- **Classes**:
  - `SVMSignClassifier(BaseSignClassifier)`: SVC with `StandardScaler` pipeline.
- **Test Mapping**: `tests/test_svm.py`.

### `sanket/models/mlp.py`
- **Role**: Multi-Layer Perceptron neural network baseline.
- **Classes**:
  - `MLPSignClassifier(BaseSignClassifier)`: Neural network classifier.
- **Test Mapping**: `tests/test_mlp.py`.

---

## 6. Subsystem: `sanket.evaluation`

Harness, metrics, latency profiling, and diagnostic error analysis.

### `sanket/evaluation/metrics.py`
- **Role**: Classification metric calculations (Top-1, Top-5, Macro F1, MCC).
- **Test Mapping**: `tests/test_evaluation.py`.

### `sanket/evaluation/latency.py`
- **Role**: High-resolution latency profiling (p50, p90, p95, p99, FPS).
- **Test Mapping**: `tests/test_evaluation.py`.

### `sanket/evaluation/manifest.py`
- **Role**: Structured JSON experiment manifest generation.
- **Test Mapping**: `tests/test_evaluation.py`.

### `sanket/evaluation/diagnostics.py`
- **Role**: Confusion matrix generation and phonological error breakdown.
- **Test Mapping**: `tests/test_diagnostics.py`.

