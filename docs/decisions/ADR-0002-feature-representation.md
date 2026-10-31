# ADR-0002: Feature Vector Representation for Isolated Sign Recognition

- **Status**: Accepted
- **Date**: 2026-10-31
- **Deciders**: Sanket Core AI & Vision Team
- **Related Issue**: SANKET-027 (Issue #29)

---

## Decision
Adopt a hybrid anatomical and kinematic feature representation combining:
1. **Normalized 3D skeletal coordinates** ($126$ dimensions: $21 \times 3 \times 2$ hands), wrist-centered and palm-scaled.
2. **Joint flexion and abduction angles** ($38$ dimensions: $19 \times 2$ hands) computed in radians $[0, \pi]$ via vector dot products.
3. **Anatomical pairwise Euclidean distances** ($48$ dimensions: $24 \times 2$ hands) capturing finger curling, span, and tip-to-palm proximity.
4. **Inter-hand interaction distances** ($3$ dimensions: inter-wrist, inter-index-tip, inter-thumb-tip) for two-handed signs.
5. **Temporal moment pooling** across sequence frames aggregating `mean`, `standard_deviation`, and `net_displacement` into a fixed $645$-dimensional tabular feature vector for baseline classifiers (Random Forest, SVM, MLP).

---

## Context & Problem Statement
Machine learning classifiers (Random Forest, Support Vector Machines, Multi-Layer Perceptrons) require fixed-length, dense tabular input vectors. Raw video sequences vary in duration (20–90 frames), camera framing, and signer hand morphology. 
Simply flattening unnormalized landmarks or taking raw pixel grids fails to generalize across signers (RQ1, RQ2).
Furthermore, while raw Cartesian coordinates provide spatial positioning, tree-based models and shallow linear SVMs struggle to discover non-linear trigonometric relationships (joint angles and finger spans) without explicit geometric feature engineering.

---

## Options Considered

1. **Option 1: Raw Landmark Coordinate Flattening ($126$ dims)**
   - *Pros*: Minimal computational overhead; simple implementation.
   - *Cons*: Highly sensitive to orientation and finger articulation non-linearities; ignores temporal velocity and movement dynamics.

2. **Option 2: Pure Angular / Kinematic Representation (Angles Only, $38$ dims)**
   - *Pros*: Invariant to uniform hand scaling; compact dimensionality.
   - *Cons*: Discards spatial separation between hands and relative hand position; cannot distinguish two-handed gestures from one-handed gestures in identical handshapes.

3. **Option 3: Hybrid Anatomical & Temporal Moment Representation ($645$ dims)** (Chosen)
   - *Pros*: Combines global spatial layout (normalized coordinates), local articulation (joint flexion & abduction angles), shape deformation (pairwise distances), inter-hand interaction, and temporal dynamics (mean, std, delta velocity).
   - *Cons*: Moderate vector size ($645$ floats); requires ~1.5 ms CPU compute per sequence.

---

## Chosen Approach
We chose **Option 3: Hybrid Anatomical & Temporal Moment Representation** implemented via `src/sanket/features/`.
- Per-frame feature dimension: $D_{\text{frame}} = 126 + 38 + 48 + 3 = 215$.
- Temporal aggregation: $3 \times D_{\text{frame}} = 645$ features (`mean`, `std`, `delta`).
- Frame-level temporal matrix representation $(T \times 215)$ is preserved for Phase 3 recurrent architectures (LSTM / GRU).

---

## Reasons & Justification
1. **Phonological Discriminability**: Captures the four primary phonological parameters of sign language: Handshape (joint angles and curl distances), Orientation (rotation alignment), Movement (temporal std and delta), and Location/Interaction (inter-hand distances).
2. **Tabular Classifier Compatibility**: Dense $645$-dimensional vector is ideal for Random Forest, SVM (RBF kernel), and MLP training without gradient vanishing or curse of dimensionality.
3. **Execution Efficiency**: Extraction runs at $< 2.0$ ms per 30-frame sequence on standard multi-core CPUs, satisfying the real-time latency target (RQ5).

---

## Consequences
- **Positive Consequences**:
  - Eliminates the need for models to learn non-linear trigonometric functions from coordinates.
  - Generates deterministic, fully reproducible tabular datasets partitioned by signer-independent splits.
  - Fulfills the Phase 1 milestone gate requirement for feature representation ADR.
- **Negative Consequences / Trade-offs**:
  - $645$ features per sample require slight memory caching during batch training.
- **Risks & Mitigation**:
  - *Risk*: Empty/undetected hands introduce zeros that could skew standard deviation.
  - *Mitigation*: Implemented boolean detection masks in `compute_temporal_statistics` filtering out unobserved frames prior to moment calculation.
