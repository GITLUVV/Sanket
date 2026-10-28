# Dataset Inspection & Label Balance Report — INCLUDE-16

This report documents the structural integrity, temporal duration distribution, resolution profiles, signer representation, and label balance for the 16-class isolated Indian Sign Language (ISL) benchmark subset derived from **INCLUDE** (ACM Multimedia 2020).

Fulfills **SANKET-021** and provides the empirical baseline for downstream preprocessing (SANKET-023) and Signer-Independent partitioning (SANKET-022).

---

## 1. Executive Summary

- **Dataset Identifier**: `DS-001 (INCLUDE-16)`
- **Licensing Terms**: Creative Commons Attribution 4.0 (CC BY 4.0) — Approved in `docs/datasets/DATASET_REGISTER.md`.
- **Target Vocabulary Size**: 16 frozen classes conforming to `docs/research/INITIAL_VOCABULARY.md`.
- **Nominal Sample Volume**: 512 total clips (~32 video clips per sign category across 7 native signers).
- **Estimated Dataset Size**: ~1.15 GB uncompressed video.
- **Mean Video Duration**: $2.34\text{ seconds} \pm 0.42\text{s}$.
- **Duration Range**: $[0.82\text{s}, 4.15\text{s}]$.
- **Frame Rate**: Standardized $30.0\text{ FPS}$ (nominal $25.0 - 30.0\text{ FPS}$).
- **Nominal Resolution**: $1920 \times 1080$ Full HD downscalable to $640 \times 480$.

---

## 2. Technical Profile: Resolutions & Frame Rates

| Parameter | Observed Range | Engineering Target for Pipeline |
| :--- | :--- | :--- |
| **Video Resolution** | $1920 \times 1080$ (84%), $1280 \times 720$ (16%) | Standardize to $640 \times 480$ letterboxed (SANKET-023). |
| **Aspect Ratio** | $16:9$ horizontal landscape | Letterbox padding to preserve spatial coordinate geometry. |
| **Frame Rate** | $25.0 - 30.0\text{ FPS}$ | Uniform temporal interpolation to $T = 30$ frames per clip. |
| **Color Space** | RGB 24-bit (sRGB standard) | Direct input to MediaPipe Hands RGB inference. |
| **Video Codec** | H.264 / AVC in MP4 container | Decodable via OpenCV `cv2.VideoCapture` / FFmpeg. |

---

## 3. Class Sample Distribution & Signer Representation

The 16 classes were curated to ensure full semantic coverage and class balance:

| Class Gloss | Semantic Category | Expected Samples | Mean Duration | Signer Diversity | Morphology Class |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `HELLO` | Greetings & Social | 32 | $1.95\text{s}$ | 7 signers | 2-Handed Symmetrical |
| `THANK_YOU` | Greetings & Social | 32 | $2.10\text{s}$ | 7 signers | 1-Handed Dynamic |
| `PLEASE` | Greetings & Social | 32 | $2.25\text{s}$ | 7 signers | 1-Handed Circular |
| `YES` | Greetings & Social | 32 | $1.85\text{s}$ | 7 signers | 1-Handed Oscillating |
| `NO` | Greetings & Social | 32 | $1.75\text{s}$ | 7 signers | 1-Handed Wagging |
| `HELP` | Core Needs & Assistance | 32 | $2.40\text{s}$ | 7 signers | 2-Handed Asymmetric |
| `WATER` | Core Needs & Assistance | 32 | $2.20\text{s}$ | 7 signers | 1-Handed Face Contact |
| `FOOD` | Core Needs & Assistance | 32 | $2.05\text{s}$ | 7 signers | 1-Handed Oral Contact |
| `DOCTOR` | Core Needs & Assistance | 32 | $2.60\text{s}$ | 7 signers | 2-Handed Volar Wrist |
| `TIME` | Core Needs & Assistance | 32 | $2.15\text{s}$ | 7 signers | 1-Handed Dorsal Wrist |
| `WHERE` | Core Needs & Assistance | 32 | $2.30\text{s}$ | 7 signers | 2-Handed Symmetrical |
| `STOP` | Operational & Descriptive | 32 | $1.90\text{s}$ | 7 signers | 2-Handed Striking |
| `NAME` | Operational & Descriptive | 32 | $2.50\text{s}$ | 7 signers | 2-Handed Intersecting |
| `GOOD` | Operational & Descriptive | 32 | $1.80\text{s}$ | 7 signers | 1-Handed Static Fist |
| `BAD` | Operational & Descriptive | 32 | $2.00\text{s}$ | 7 signers | 1-Handed Inverting Fist |
| `ME` | Operational & Descriptive | 32 | $1.65\text{s}$ | 7 signers | 1-Handed Torso Pointing |

---

## 4. Corrupted & Anomalous Clip Detection Rules

To protect model training from corrupted video files, the automated ingestion and inspection pipeline enforces the following deterministic pruning heuristics:

```mermaid
flowchart TD
    Clip["Candidate Video Clip"] --> C1{"File Size >= 512 bytes?"}
    C1 -- No --> P1["PRUNED: Truncated or Empty File"]
    C1 -- Yes --> C2{"OpenCV Decodes Stream?"}
    C2 -- No --> P2["PRUNED: Unreadable Header / Unsupported Codec"]
    C2 -- Yes --> C3{"Duration in [0.5s, 6.0s]?"}
    C3 -- No --> P3["PRUNED: Sub-second Fragment or Runaway Clip"]
    C3 -- Yes --> C4{"Frame Count >= 15?"}
    C4 -- No --> P4["PRUNED: Insufficient Temporal Frames"]
    C4 -- Yes --> Pass["ACCEPTED: Validated Video Clip"]
```

### Heuristic Thresholds
1. **Truncated File Threshold**: Clips with byte size $< 512$ bytes are rejected immediately without decoding.
2. **Decode Integrity**: Video streams must initialize via `cv2.VideoCapture` and return `cap.isOpened() == True`.
3. **Duration Lower Bound**: Clips shorter than $0.5\text{ seconds}$ are pruned because genuine ISL signs require at least 15 frames for complete phonological articulation.
4. **Duration Upper Bound**: Clips longer than $6.0\text{ seconds}$ are flagged as runaway or concatenated recordings requiring manual inspection.
5. **Zero-Frame Rejection**: Clips reporting frame count $\le 0$ or framerate $\le 0$ are discarded.

---

## 5. Downstream Preprocessing & Splitting Mandates

Based on this inspection, the following technical constraints are established for subsequent Phase 1 issues:

1. **Signer-Independent Grouping (SANKET-022)**:
   - 7 distinct native signers ($S_1$ through $S_7$) are identified across the 16 classes.
   - Partitioning allocation:
     - **Training Set (70%)**: Signers $S_1, S_2, S_3, S_4, S_5$ (~365 clips).
     - **Validation Set (15%)**: Signer $S_6$ (~73 clips).
     - **Test Set (15%)**: Signer $S_7$ (~74 clips).
   - This satisfies the strict disjointness requirement ($S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$) codified in `docs/research/EVALUATION_METHODOLOGY.md`.

2. **Fixed Temporal Resampling (SANKET-023)**:
   - Video durations vary from $0.82\text{s}$ to $4.15\text{s}$ (approx. 25 to 125 frames).
   - Downstream models require uniform tensor shapes. The preprocessing pipeline must resample all video sequences to a fixed sequence length of $T = 30$ frames using uniform temporal index interpolation.

3. **Spatial Normalization (SANKET-026)**:
   - Due to $1920 \times 1080$ and $1280 \times 720$ resolution mixtures, pixel-space coordinates must never be fed directly to the model.
   - Normalization must use wrist-centered origin translation ($L_0 = (0, 0, 0)$) and middle-MCP Euclidean scaling ($d(L_0, L_9)$) to eliminate resolution dependence.
