# Research Methodology

## Signer-Independent Protocol
In sign language recognition, models often suffer from **signer leakage**—memorizing personal clothing, body proportions, skin tone, or background clutter rather than true sign kinematics.

To prevent over-optimistic evaluation:
- All datasets are partitioned strictly by **signer ID**.
- Signers present in the training set **never appear in the validation or test sets**.
- Random sample shuffling across splits is strictly prohibited.

## Core Research Questions (RQs)
- **RQ1 (Representation)**: Do normalized skeletal landmarks provide sufficient discriminability compared to raw pixel CNNs?
- **RQ2 (Anatomical vs Metric Features)**: Do joint flexion angles and pairwise distances outperform raw normalized coordinates?
- **RQ3 (Model Complexity)**: Can lightweight classical models (Random Forest, SVM) match deep neural networks for isolated ISL signs?
- **RQ4 (Pose Anchors)**: Does incorporating upper-body pose reference anchors improve signs articulated in spatial proximity to the torso and face?
- **RQ5 (Inference Latency)**: Can end-to-end inference execute in $\le 50$ ms on consumer CPU hardware?
