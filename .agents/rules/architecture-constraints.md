# Architecture Constraints & Boundaries

## Module Dependency Hierarchy
Dependencies must strictly flow in a single direction down the architecture stack:

```text
sanket.core
    └── sanket.data
            └── sanket.vision
                    └── sanket.features
                            └── sanket.models
                                    └── sanket.evaluation
                                            └── sanket.realtime
                                                    └── sanket.ui
```

- **Zero Circular Dependencies**: Lower-level modules (`core`, `vision`, `features`) must never import from higher-level modules (`models`, `evaluation`, `ui`).
- **Feature Vector Shape Contract**: The feature extractor must produce exactly a 645-dimensional numerical vector:
  - 126 raw coordinates
  - 126 normalized coordinates
  - 60 joint angles
  - 210 pairwise Euclidean distances
  - 123 temporal statistical moments
- **Model Invariance**: Models consume feature vectors only; classifiers must never take raw video frames or un-normalized landmarks directly.
- **Edge Deployment Compatibility**: No cloud network calls or GPU requirements permitted in core inference pipelines.

