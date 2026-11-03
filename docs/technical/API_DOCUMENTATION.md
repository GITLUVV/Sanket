# SANKET (संकेत) — Python API & CLI Reference Manual

> **Comprehensive Reference for Core Python Packages, Class Interfaces, and CLI Commands**  
> *Authoritative technical specification for developers and automated integrations.*

---

## 1. Package: `sanket.core.config`

Central configuration system using Pydantic schemas.

### Classes

#### `CameraConfig(BaseModel)`
- `device_id: int = 0`: Index of the video capture device.
- `width: int = 640`: Target frame width.
- `height: int = 480`: Target frame height.
- `fps: int = 30`: Target camera capture rate.

#### `LandmarkConfig(BaseModel)`
- `max_num_hands: int = 2`: Maximum hands to detect (1 or 2).
- `min_detection_confidence: float = 0.7`: MediaPipe palm detection threshold.
- `min_tracking_confidence: float = 0.5`: MediaPipe landmark tracking threshold.
- `normalize_coordinates: bool = True`: Whether to apply wrist-centering and palm-scaling.

#### `ModelConfig(BaseModel)`
- `architecture: str = "random_forest"`: Selected classifier model architecture.
- `confidence_threshold: float = 0.75`: Minimum confidence cutoff for predictions.
- `smoothing_window_size: int = 5`: Temporal smoothing window length in frames.
- `weights_path: str | None = None`: Path to serialized `.joblib` model artifact.

#### `SanketConfig(BaseModel)`
Top-level configuration class uniting `camera`, `landmarks`, and `model`.

### Functions
- `load_config(path: str | Path | None = None) -> SanketConfig`: Loads configuration from YAML with fallback to `configs/default.yaml`.

---

## 2. Package: `sanket.vision`

Computer vision, image standardization, and landmark extraction.

### Modules

#### `sanket.vision.normalization`
- `normalize_landmarks(landmarks: np.ndarray, scale_factor: float | None = None) -> tuple[np.ndarray, float]`:
  Translates landmarks such that wrist (index 0) is at $(0, 0, 0)$ and scales coordinates by palm Euclidean distance.
- `calculate_palm_scale(landmarks: np.ndarray) -> float`:
  Computes Euclidean distance between wrist (landmark 0) and middle finger MCP joint (landmark 9).

#### `sanket.vision.preprocessing`
- `letterbox_resize(image: np.ndarray, target_size: tuple[int, int] = (640, 480)) -> np.ndarray`:
  Resizes image to target dimensions while maintaining aspect ratio via padding.
- `mirror_frame(frame: np.ndarray) -> np.ndarray`:
  Flips frame horizontally (`cv2.flip(frame, 1)`) for intuitive selfie camera mode.

#### `sanket.vision.landmarks`
- `extract_hand_landmarks(frame_rgb: np.ndarray, static_image_mode: bool = False) -> np.ndarray | None`:
  Executes MediaPipe Hands. Returns array of shape `(2, 21, 3)` containing 3D coordinates for dominant and non-dominant hands.

---

## 3. Package: `sanket.features`

Feature vector extraction generating the standard 645-dimensional feature space.

### Modules

#### `sanket.features.geometry`
- `calculate_joint_angles(landmarks: np.ndarray) -> np.ndarray`:
  Computes 3D joint angles across 10 finger joints for both hands (60 dimensions).
- `calculate_pairwise_distances(landmarks: np.ndarray) -> np.ndarray`:
  Computes normalized Euclidean distances between key anatomical landmarks (210 dimensions).

#### `sanket.features.temporal`
- `calculate_temporal_moments(sequence: np.ndarray) -> np.ndarray`:
  Computes mean, standard deviation, and velocity vectors across sliding temporal window (123 dimensions).

#### `sanket.features.extractor.FeatureExtractor`
- `extract(landmarks: np.ndarray, temporal_history: list[np.ndarray] | None = None) -> np.ndarray`:
  Produces the unified 645-dimensional feature vector:
  $$\mathbf{x} = [\mathbf{x}_{\text{raw}} (126), \mathbf{x}_{\text{norm}} (126), \mathbf{x}_{\text{angles}} (60), \mathbf{x}_{\text{dist}} (210), \mathbf{x}_{\text{temp}} (123)] \in \mathbb{R}^{645}$$

---

## 4. Package: `sanket.models`

Machine learning classifiers conforming to the `BaseSignClassifier` contract.

### Classes

#### `BaseSignClassifier(ABC)`
- `fit(X: np.ndarray, y: np.ndarray) -> BaseSignClassifier`: Abstract method for model training.
- `predict(X: np.ndarray) -> np.ndarray`: Abstract method returning predicted class string.
- `predict_proba(X: np.ndarray) -> np.ndarray`: Abstract method returning class probability distribution.
- `save(path: str | Path) -> None`: Serializes estimator via `joblib.dump`.
- `load(path: str | Path) -> BaseSignClassifier`: Deserializes estimator from `.joblib` file.

#### `RandomForestSignClassifier(BaseSignClassifier)`
- Selected baseline from ADR-0003.
- Parameters: `n_estimators=100`, `max_depth=15`, `random_state=42`, `n_jobs=-1`.

#### `SVMSignClassifier(BaseSignClassifier)`
- Support Vector Classifier with RBF kernel and integrated `StandardScaler`.
- Parameters: `c=2.0`, `kernel="rbf"`, `gamma="scale"`, `probability=True`.

#### `MLPSignClassifier(BaseSignClassifier)`
- Multi-Layer Perceptron neural network.
- Parameters: `hidden_layer_sizes=(256, 128)`, `activation="relu"`, `solver="adam"`, `max_iter=200`.

---

## 5. Package: `sanket.evaluation`

Harness and diagnostic reporting tools.

### Modules

#### `sanket.evaluation.metrics`
- `compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: list[str]) -> dict[str, float]`:
  Computes Top-1 Accuracy, Top-5 Accuracy, Macro Precision, Macro Recall, Macro F1, and MCC.

#### `sanket.evaluation.latency`
- `measure_latency(model: BaseSignClassifier, sample: np.ndarray, n_iterations: int = 500) -> LatencyReport`:
  Executes 50 warmup rounds and measures per-sample CPU latency (p50, p90, p95, p99, FPS).

#### `sanket.evaluation.evaluator.ModelEvaluator`
- `evaluate(model: BaseSignClassifier, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationReport`:
  Runs full benchmark suite and constructs structured evaluation report.

---

## 6. Command-Line Interface (CLI) Scripts

| Script | Purpose | Essential Arguments |
| :--- | :--- | :--- |
| `scripts/verify_setup.py` | Validates environment and dependencies. | `--quiet` |
| `scripts/create_splits.py` | Generates signer-independent train/val/test splits. | `--input-dir`, `--output-dir`, `--seed 42` |
| `scripts/extract_landmarks.py` | Extracts MediaPipe landmarks from video directory. | `--video-dir`, `--output-dir`, `--static-mode` |
| `scripts/generate_features.py` | Generates 645-dim feature arrays from landmarks. | `--landmarks-dir`, `--output-features` |
| `scripts/train_random_forest.py`| Trains baseline Random Forest model. | `--features-train`, `--output-model` |
| `scripts/train_svm.py` | Trains baseline SVM classifier. | `--features-train`, `--output-model`, `--c 2.0` |
| `scripts/train_mlp.py` | Trains baseline MLP neural network. | `--features-train`, `--output-model`, `--epochs 200` |
| `scripts/evaluate_model.py` | Evaluates model and writes JSON manifest. | `--model-path`, `--output-manifest` |
| `scripts/analyze_errors.py` | Computes confusion matrix and phonological errors. | `--manifest-path`, `--output-dir` |
| `scripts/compare_baselines.py` | Generates comparative study across baselines. | `--manifests-dir`, `--output-report` |

