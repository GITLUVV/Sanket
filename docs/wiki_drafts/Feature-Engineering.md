# Feature Engineering & Kinematic Modeling

Formally documented in **ADR-0002**, Sanket transforms skeletal landmarks into rich, discriminative tabular feature vectors.

## Feature Components (215 Dimensions per Frame)
1. **Normalized Coordinates**: 42 hand landmarks $	imes 3 = 126$ dimensions.
2. **Joint Flexion & Abduction Angles**: 15 flexion + 4 abduction angles per hand $	imes 2 = 38$ dimensions in $[0, \pi]$ radians.
3. **Pairwise Euclidean Distances**: 24 anatomical spans per hand $	imes 2 = 48$ dimensions with proven translation invariance.
4. **Inter-Hand Interaction**: 3 distances (inter-wrist, inter-index-tip, inter-thumb-tip).

## Temporal Dynamic Moments (645 Dimensions)
For isolated sign classification, frame vectors are aggregated across time using statistical moments:
- **Mean Vector**: $\mu \in \mathbb{R}^{215}$
- **Standard Deviation Vector**: $\sigma \in \mathbb{R}^{215}$
- **Trajectory Delta**: $\Delta = v_{	ext{end}} - v_{	ext{start}} \in \mathbb{R}^{215}$
- **Total Pooled Dimensionality**: $3 	imes 215 = 645$ dimensions.
