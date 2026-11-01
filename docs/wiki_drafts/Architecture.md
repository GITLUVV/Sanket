# System Architecture

Sanket adopts a modular, feed-forward architecture separating vision ingestion from inference and presentation:

```text
Camera / Video Stream
       ↓
Landmark Extraction (MediaPipe Hands & Pose Anchors)
       ↓
Coordinate Normalization (Wrist Origin & Palm Scale Invariance)
       ↓
Feature Engineering (3D Angles, Distances & Temporal Dynamics)
       ↓
Classification Engine (Baseline RF / SVM / MLP → Sequential Models)
       ↓
Temporal Smoothing & Confidence Thresholding
       ↓
Accessible Text Output & Speech Synthesis
```

## Subsystems
- **`sanket.core`**: Configuration schema and runtime validation.
- **`sanket.data`**: Dataset ingestion, SHA-256 verification, and signer-independent partitioning.
- **`sanket.vision`**: Temporal subsampling, letterboxing, MediaPipe landmark extraction, pose anchoring, and coordinate normalization.
- **`sanket.features`**: Kinematic extraction producing 215-dim frame vectors and 645-dim pooled representations.
- **`sanket.models`**: Standardized classifier interfaces and baseline model implementations.
- **`sanket.evaluation`**: Unified benchmark evaluation harness, metric logging, and manifest export.
