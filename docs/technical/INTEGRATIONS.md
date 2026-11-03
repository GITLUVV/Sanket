# SANKET (संकेत) — Upstream Framework & Library Integrations

> **Technical Integration Specifications for External Libraries, Frameworks, and Toolchains**  
> *Configuration guidelines, operational parameters, and adapter patterns.*

---

## 1. Google MediaPipe Integration

SANKET integrates Google MediaPipe for high-performance, real-time landmark localization.

### 1.1 MediaPipe Hands
- **Interface**: `mp.solutions.hands.Hands`
- **Landmark Topology**: 21 3D coordinates per hand $(x, y, z)$.
  - $x, y \in [0.0, 1.0]$: Normalized image space.
  - $z$: Relative depth anchored at the wrist keypoint.
- **Operational Modes**:
  - `static_image_mode=True` (Phase 1): Runs full palm detection on every frame. Used for offline dataset processing and invariant testing.
  - `static_image_mode=False` (Phase 2): Tracks landmarks across frames using temporal bounding box tracking, reducing latency from ~35 ms to < 12 ms per frame.
- **Configured Parameters** (from `configs/default.yaml`):
  ```yaml
  max_num_hands: 2
  min_detection_confidence: 0.7
  min_tracking_confidence: 0.5
  ```

### 1.2 MediaPipe Pose (Upper-Body Investigation)
- **Interface**: `mp.solutions.pose.Pose`
- **Landmark Topology**: 33 full-body landmarks. SANKET filters this down to upper-body spatial anchors (nose, left/right shoulders, left/right elbows) to provide reference coordinate planes for location-dependent signs.

---

## 2. OpenCV (`opencv-python`) Integration

- **Video Acquisition**: `cv2.VideoCapture(device_id)`.
  - Configured for non-blocking read in dedicated worker threads (`CameraCapture`).
  - Sets capture properties: `CAP_PROP_FRAME_WIDTH = 640`, `CAP_PROP_FRAME_HEIGHT = 480`, `CAP_PROP_FPS = 30`.
- **Image Preprocessing**:
  - `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`: Converts frames for MediaPipe ingestion.
  - `cv2.flip(frame, 1)`: Provides mirror selfie mode for live user feedback.
  - `cv2.copyMakeBorder()`: Generates letterbox padding to maintain native aspect ratios.
- **Visual HUD Rendering**:
  - `cv2.line()` and `cv2.circle()`: Draws skeletal connections with anti-aliasing (`LINE_AA`).
  - `cv2.rectangle()`: Draws translucent HUD backing banners for text contrast.
  - `cv2.putText()`: Renders prediction glosses and telemetry.

---

## 3. Scikit-learn Integration

- **Classifier Contracts**: Wraps `sklearn.ensemble.RandomForestClassifier`, `sklearn.svm.SVC`, and `sklearn.neural_network.MLPClassifier`.
- **Pipelines**: Integrates `sklearn.preprocessing.StandardScaler` within the SVM pipeline to standardize 645-dim feature vectors before kernel evaluation.
- **Metrics Computation**: Leverages `sklearn.metrics` for `accuracy_score`, `precision_recall_fscore_support`, `matthews_corrcoef`, and `confusion_matrix`.

---

## 4. Joblib Serialization Integration

- **Serialization Protocol**:
  ```python
  import joblib
  joblib.dump(model, artifact_path, compress=3)
  ```
- **Safety Checks**: Prior to invoking methods on unpickled objects, SANKET asserts `isinstance(loaded_obj, BaseSignClassifier)`.

