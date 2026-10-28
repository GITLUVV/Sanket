# GitHub Wiki Architecture & Knowledge Base Structure

## Purpose
The **GitHub Wiki** serves as the human-readable, educational, and conceptual knowledge base for **Sanket**. 

While the repository's `docs/` directory stores version-controlled, auditable engineering and research artifacts (such as ADRs, dataset registers, and experiment logs), the Wiki explains concepts, operational workflows, and domain background for students, researchers, and community contributors.


---

## Wiki Page Taxonomy & Sitemap

```text
Wiki Root
├── Home (Landing page, project vision, quick navigation)
├── Project Overview (Context of Indian Sign Language accessibility, goals, ethics)
├── Architecture (End-to-end pipeline overview: Camera → Vision → ML → NLP → Speech)
├── Development Workflow (Git branch strategy, issue/PR lifecycle, coding standards)
├── Research Methodology (Signer independence, scientific hypotheses, empirical rigor)
├── Datasets (Overview of ISL datasets, licensing caveats, collection protocols)
├── ISL Vocabulary (Linguistic introduction to ISL phonology, hands, and vocab tiers)
├── Model Development (Feature extraction with MediaPipe, classical vs deep models)
├── Experiments (Guide to interpreting experiment logs and confusion metrics)
├── Evaluation (Benchmarking guidelines, FPS, latency, and robustness criteria)
├── Deployment (Packaging, on-device execution, and real-time inference)
├── Troubleshooting (Common camera issues, landmark drops, lighting fixes)
└── Project History (Chronological milestones, lessons learned, and acknowledgments)
```

---

## Detailed Page Outlines & Drafting Notes

### 1. `Home.md`
- **Focus**: Welcome message, project mission statement, active phase status indicator (Phase 0), and links to primary Wiki sections and GitHub repository.
- **Audience**: General visitors, new contributors, academic evaluators.

### 2. `Project-Overview.md`
- **Focus**: Demographic context of the Deaf and Hard-of-Hearing community in India; the linguistic richness and distinctiveness of Indian Sign Language (ISL) from ASL or BSL; Sanket's objective as an assistive accessibility interface.
- **Disclaimer**: Clear note clarifying that the initial project scope targets constrained vocabulary recognition rather than unrestricted continuous discourse.

### 3. `Architecture.md`
- **Focus**: High-level block diagram illustrating data flow:
  1. Video Stream Ingestion (OpenCV)
  2. Spatial Landmark Tracking (MediaPipe Hands & Pose)
  3. Geometric Landmark Normalization (Rotation/Scale invariant)
  4. Temporal Windowing (Sliding buffer)
  5. Classifier & Sequence Model (RF/SVM baseline → LSTM/GRU)
  6. Natural Language & Sentence Structuring (Grammar layer)
  7. Accessible User Interface & TTS Synthesis

### 4. `Development-Workflow.md`
- **Focus**: Step-by-step guide for developers on cloning the repository, installing `requirements-dev.txt`, creating `feature/SANKET-XXX` branches, opening PRs, running linting (`ruff`) and test suites (`pytest`), and understanding human merge control.

### 5. `Research-Methodology.md`
- **Focus**: Research standards: why signer-independent splits are mandatory, how to avoid camera background overfitting, how to define null hypotheses, and why negative experimental results must be faithfully recorded in `docs/experiments/`.

### 6. `Datasets.md`
- **Focus**: Guide to Indian Sign Language datasets (e.g. INCLUDE, ISL-CSLTR), explaining the differences between isolated sign clips and continuous sentences, as well as the imperative of auditing third-party licenses in `docs/datasets/DATASET_REGISTER.md`.


### 7. `ISL-Vocabulary.md`
- **Focus**: Linguistic parameters of Indian Sign Language (handshape, location, movement, palm orientation, non-manual markers). Details the initial 10–25 sign constrained vocabulary and roadmap for expansion.

### 8. `Model-Development.md`
- **Focus**: Landmark-based classification vs raw CNN pixel processing. Explains the computational advantages of extracting 21 skeletal hand coordinates for real-time inference on low-power devices.

### 9. `Experiments.md`
- **Focus**: How to read and contribute to experiment reports; explanation of metrics (Top-1 accuracy, Top-5 accuracy, Macro F1, confusion matrices, latency).

### 10. `Evaluation.md`
- **Focus**: Rigorous testing across unconstrained environmental conditions (lighting shifts, signer distances, cluttered backgrounds, hand occlusions) and real-time FPS benchmarking.

### 11. `Deployment.md`
- **Focus**: Running the pipeline locally; configuring video device IDs; packaging models using ONNX runtime; privacy considerations regarding on-device video memory buffers.

### 12. `Troubleshooting.md`
- **Focus**: Resolving webcam permission errors, MediaPipe landmark flickering under bad lighting, virtual environment package conflicts, and Windows/Linux path issues.

### 13. `Project-History.md`
- **Focus**: Archival notes on project origins, major refactorings, completed milestones, and team contributors.

---

## Initializing the GitHub Wiki
Because GitHub automatically creates the wiki Git repository upon the creation of the first page via the web UI:
1. Navigate to `https://github.com/anveshna25/Sanket/wiki`.
2. Click **Create the first page** to create `Home.md`.
3. Once initialized, the wiki repository `https://github.com/anveshna25/Sanket.wiki.git` becomes active and can be synced with these drafts.
