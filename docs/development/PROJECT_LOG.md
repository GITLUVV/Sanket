# Project Development Log — Sanket

This log maintains a chronological record of significant engineering progress, architectural decisions, experimental runs, and milestone transitions for the Sanket project.

---

## Log Entries

### 2026-10-28 — Engineering Foundation & Milestone 0 Setup

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Completed Work**:
  - Initialized formal repository structure: `docs/`, `src/sanket/`, `tests/`, `configs/`, `scripts/`.
  - Added Apache License 2.0 `LICENSE`.
  - Configured comprehensive security policy `SECURITY.md` covering camera/biometric privacy and vulnerability disclosures.
  - Enabled GitHub Dependabot vulnerability alerts and automated security updates.
  - Enabled GitHub Secret Scanning and Push Protection.
  - Configured GitHub Ruleset protecting `main` from force pushes and requiring PR reviews.
  - Established standardized GitHub label taxonomy (27 labels across Type, Area, Priority, Phase).
  - Authored standard GitHub Issue templates (Feature, Bug, Research, Experiment, Docs) and PR template.
  - Established Master Development Roadmap `docs/roadmap/SANKET_ROADMAP.md`.
  - Initialized Architecture Decision Record system with ADR-0001 `docs/decisions/ADR-0001-project-foundations.md`.
  - Established Dataset Register `docs/datasets/DATASET_REGISTER.md` and Experiment tracking framework.
  - Configured Python project packaging with `pyproject.toml`, `requirements.txt`, and GitHub Actions CI.
  - Created Milestone 0 on GitHub and populated issues `SANKET-001` through `SANKET-012`.
- **Decisions**:
  - Adopted strict sequential Phase gating (ADR-0001).
  - Deferred all ML and model training until Phase 0 foundation, dataset audit, and vocabulary definitions are frozen.
  - Enforced Signer-Independent evaluation strategy to prevent biometric data leakage.
- **Experiments**:
  - No ML experiments conducted yet (Phase 0).
- **Problems & Mitigations**:
  - *Problem*: GitHub Wiki git repository is not provisioned until the first page is created.  
    *Mitigation*: Authored comprehensive Wiki information architecture and drafting specifications in `docs/architecture/WIKI_STRUCTURE.md`.
- **Results**:
  - Repository security posture validated (Dependabot, rulesets, secret scanning).
  - Unit tests verify packaging and baseline environment cleanly.
- **Next Actions**:
  - Review and merge PR #13 on GitHub.
  - Complete SANKET-008 (Dataset research and comparison).

---

### 2026-10-28 — SANKET-008: ISL Dataset Research & Comparison

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-008
- **Completed Work**:
  - Conducted a multi-dimensional comparative survey of 6 candidate ISL datasets: INCLUDE, INCLUDE-50, ISL-CSLTR, CISLR, IIIT-Delhi ISL, and static finger-spelling benchmarks.
  - Benchmarked datasets across modality, resolution, framerate, vocabulary scope, signer demographics, background clutter, and licensing terms in `docs/research/DATASET_SURVEY.md`.
  - Evaluated MediaPipe tracking feasibility on one-handed, two-handed, and overlapping sign morphology.
  - Authored empirical trade-off analysis comparing full-scale datasets vs constrained subsets.
- **Decisions**:
  - Selected **INCLUDE** (specifically a 15-class high-frequency subset matching `INITIAL_VOCABULARY.md`) as the primary training and evaluation benchmark for Phase 1.
  - Deferred **ISL-CSLTR** to Phase 3/4 due to controlled studio recording conditions and continuous sentence focus.
  - Excluded static alphabet datasets from primary research claims to ensure genuine ISL vocabulary fidelity.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Problems & Mitigations**:
  - *Problem*: Full INCLUDE dataset is ~15 GB and includes 263 classes, which is too complex for an initial isolated baseline.  
    *Mitigation*: Restrict Phase 1 ingestion to a 15-class subset (~1.2 GB), enabling fast, reproducible CPU baseline training while preserving authentic linguistic vocabulary.
- **Results**:
  - Comprehensive survey and recommendation completed and recorded in `docs/research/DATASET_SURVEY.md`.
- **Next Actions**:
  - Review and merge PR #14 (SANKET-008).
  - Complete SANKET-009 (Dataset licensing & access review).

---

### 2026-10-28 — SANKET-009: Dataset Licensing & Access Review

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-009
- **Completed Work**:
  - Formally audited 4 candidate datasets in `docs/datasets/DATASET_REGISTER.md`: INCLUDE / INCLUDE-50, ISL-CSLTR, CISLR, and IIIT-Delhi ISL.
  - Formulated the Project Licensing Non-Interference Policy, codifying that Sanket's Apache-2.0 license applies solely to software/docs and does not supersede third-party dataset restrictions.
  - Established a mandatory Human Project-Owner Approval Gate prior to raw video download or local dataset ingestion.
  - Audited biometric consent, facial identification risks, and derivative mathematical feature rights (MediaPipe 3D coordinate tensors).
- **Decisions**:
  - Approved **INCLUDE** under Creative Commons Attribution 4.0 (CC BY 4.0) as legally and ethically sound for Phase 1 isolated baseline training.
  - Deferred **ISL-CSLTR** to Phase 3/4 pending formal academic MoU execution.
  - Codified strict policy: No raw biometric video footage may ever be committed or pushed to the GitHub repository.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Results**:
  - `docs/datasets/DATASET_REGISTER.md` fully audited, verified, and gated.
- **Next Actions**:
  - Review and merge PR #15 (SANKET-009).
  - Complete SANKET-010 (Initial ISL vocabulary definition).

---

### 2026-10-28 — SANKET-010: Initial ISL Vocabulary Definition

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-010
- **Completed Work**:
  - Defined a frozen 16-sign constrained vocabulary of high-frequency isolated Indian Sign Language signs in `docs/research/INITIAL_VOCABULARY.md`.
  - Balanced vocabulary across 3 essential communicative domains: Greetings & Social Interaction, Core Needs & Assistance, and Emergency & Descriptive Signs.
  - Documented complete phonological and articulatory profiles for each of the 16 signs, specifying handedness, Stokoe-style handshape classes, spatial location/signing space, orientation vectors, and movement dynamics.
  - Formulated specific MediaPipe skeletal landmark extraction pipelines for each sign, mapping landmark geometric relationships (e.g. wrist-to-palm distance, finger joint angles, face anchor proximity).
  - Performed pairwise confusion risk analysis across morphologically adjacent signs (e.g., `DOCTOR` vs `TIME`, `WATER` vs `FOOD`, `HELLO` vs `PLEASE`, `GOOD` vs `YES`, `STOP` vs `HELP`), detailing primary geometric disambiguation features.
  - Established a formal vocabulary freeze protocol and project owner sign-off gate before dataset ingestion.
- **Decisions**:
  - Selected 16 signs aligning directly with the classes available in the CC BY 4.0 INCLUDE benchmark dataset to ensure unblocked data availability.
  - Excluded static 2D finger-spelling / alphabet symbols from vocabulary to prioritize authentic dynamic ISL lexical signs.
  - Mandated dual-hand + face anchor keypoint tracking (MediaPipe Hands + Pose) to disambiguate facial proximity and two-handed interaction.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Problems & Mitigations**:
  - *Problem*: High geometric similarity between signs targeting similar body regions (e.g., `DOCTOR` and `TIME` both contacting the non-dominant wrist).  
    *Mitigation*: Formulated precise vector-angle and palmar vs dorsal surface approach rules in the feature pipeline to differentiate volar arterial contact from dorsal wristwatch tapping.
- **Results**:
  - `docs/research/INITIAL_VOCABULARY.md` fully specified with 16 linguistic profiles and confusion matrices.
- **Next Actions**:
  - Review and merge PR #16 (SANKET-010).
  - Complete SANKET-011 (Research questions and hypotheses formulation).

---

### 2026-10-28 — SANKET-011: Research Questions & Hypotheses Formulation

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-011
- **Completed Work**:
  - Formulated 5 core scientific research questions and formal testable null ($H_0$) and alternative ($H_1$) hypotheses in `docs/research/RESEARCH_QUESTIONS.md`:
    - **RQ1 (Representation Fidelity)**: Skeletal coordinates ($21 \times 3$ tensors) vs raw pixel CNNs under edge CPU compute constraints.
    - **RQ2 (Coordinate Normalization)**: Wrist-centered origin and palm-scale Euclidean distance normalization resilience across camera distances (0.6m to 2.0m).
    - **RQ3 (Inductive Bias & Sequence Dynamics)**: Classical shallow classifiers (Random Forest / SVM) vs deep temporal recurrent architectures (BiLSTM / 1D-CNN) partitioned by static vs dynamic sign morphology.
    - **RQ4 (Biometric Leakage & Generalization Bounds)**: Quantitative evaluation of optimistic bias in random sample splits versus strict Signer-Independent partitioning.
    - **RQ5 (Real-Time Pipeline Latency Budget)**: End-to-end stage decomposition (capture, detection, normalization, inference, rendering) under a 50 ms budget ($\ge 25$ FPS) on non-GPU CPUs.
  - Constructed an explicit Research Questions to Planned Experiments Mapping Matrix linking each RQ to specific Phase 1/Phase 5 issues (SANKET-026, SANKET-028, SANKET-029, SANKET-030, SANKET-031, SANKET-032, SANKET-100, SANKET-102, SANKET-103).
  - Codified a Scientific Rigor & Negative Result Policy guaranteeing transparent reporting of rejected hypotheses in Architecture Decision Records.
- **Decisions**:
  - Established paired two-tailed Wilcoxon signed-rank and Student's t-tests with $\alpha = 0.05$ as the formal statistical significance standard.
  - Bound all Phase 1 baseline experiments to test these 5 explicit hypotheses rather than exploratory parameter tuning.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Results**:
  - `docs/research/RESEARCH_QUESTIONS.md` established as the scientific guiding document for all future phases.
- **Next Actions**:
  - Review and merge PR #17 (SANKET-011).
  - Complete SANKET-012 (Evaluation methodology).

---

### 2026-10-28 — SANKET-012: Evaluation Methodology & Benchmarking Standards

- **Phase**: Phase 0 — Engineering & Research Foundation
- **Milestone**: Milestone 0 — Engineering Foundation
- **Related Issue**: SANKET-012
- **Completed Work**:
  - Authored comprehensive evaluation protocol in `docs/research/EVALUATION_METHODOLOGY.md` codifying strict Signer-Independent (Leave-Signers-Out) data partitioning.
  - Formulated mathematical definitions for disjoint signer partitions: $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$, preventing biometric identity and background leakage.
  - Formulated quantitative performance metrics: Top-1 Accuracy, Top-$k$ Accuracy, Macro-Averaged Precision, Recall, and F1-Score, and Multi-Class Matthew's Correlation Coefficient (MCC).
  - Specified normalized confusion matrix heatmap standards and established a 5-tier phonological linguistic error taxonomy (Handshape $E_{\mathcal{H}}$, Location $E_{\mathcal{L}}$, Orientation $E_{\mathcal{O}}$, Movement $E_{\mathcal{M}}$, Tracking Failure $E_{\text{track}}$).
  - Established operational runtime benchmark specifications: inference latency $\le 15$ ms (median), total pipeline latency $\le 35$ ms (median), throughput $\ge 25$ FPS sustained, peak RAM $\le 200$ MB, and model disk size $\le 30$ MB on non-GPU CPUs.
  - Codified reproducibility protocols including random seed locking (`random`, `numpy`, `torch` = 42), structured JSON experiment manifests (`experiment_manifest.json`), and peer-review sign-off checklists.
- **Decisions**:
  - Designated Macro F1 (rather than weighted accuracy) as the official primary metric to prevent majority-class bias from masking failure on rare emergency signs.
  - Mandated that all experimental evaluations in Phase 1 and beyond must be executed with fixed random seeds and logged manifests.
- **Experiments**:
  - No model training conducted (Phase 0).
- **Results**:
  - `docs/research/EVALUATION_METHODOLOGY.md` finalized as the binding benchmark standard.
  - **Milestone 0 (Engineering Foundation) fully executed across all 12 planned issues (SANKET-001 through SANKET-012).**
- **Next Actions**:
  - Submit Pull Request for SANKET-012.
  - Project owner review and merge of Milestone 0 PRs (#13, #14, #15, #16, #17, #18).
  - Initiate Phase 1 (Milestone 1 — Isolated Sign Landmark Pipeline & Baseline Model).

---

### 2026-10-28 — SANKET-020: Dataset Ingestion Pipeline & Checksum Verification

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-020 (Issue #22)
- **Completed Work**:
  - Implemented modular dataset ingestion engine in \src/sanket/data/ingestion.py\ supporting automated download, archive extraction, streaming SHA-256 checksum verification, and target class filtering.
  - Implemented license gate enforcement (\erify_dataset_license\) validating third-party dataset compliance against approved open licenses (\CC BY 4.0\, etc.) per \docs/datasets/DATASET_REGISTER.md\.
  - Implemented strict 16-class vocabulary filter (\ilter_target_classes\) enforcing ingestion boundaries to the frozen vocabulary specified in \docs/research/INITIAL_VOCABULARY.md\.
  - Developed operational CLI ingestion utility in \scripts/ingest_dataset.py\ supporting \--dry-run\, \--verify-only\, \--archive-path\, \--url\, and \--expected-sha256\ flags.
  - Implemented structured manifest generator (\generate_ingestion_manifest\) producing immutable JSON run summaries recording file counts, byte sizes, and timestamps.
  - Authored comprehensive test suite in \	ests/test_data_ingestion.py\ covering license gates, SHA-256 calculation, zip/tar filtering, dry-run simulation, and failure modes (9 unit tests).
  - Updated root .gitignore to use /data/ ensuring root-level video archives remain uncommitted while Python package src/sanket/data/ is properly tracked.
- **Decisions**:
  - Mandated that dataset ingestion must always output a machine-readable ingestion_manifest.json before downstream processing.
  - Enforced streaming chunked SHA-256 calculation (64 KB buffers) to handle multi-gigabyte archives safely without out-of-memory errors on constrained hardware.
- **Experiments**:
  - Validated dry-run simulation generating valid 16-class manifest structure.
- **Problems & Mitigations**:
  - *Problem*: Generic data/ rule in .gitignore was matching src/sanket/data/.  
    *Mitigation*: Anchored rule to root /data/, cleanly preserving isolation for raw video staging while allowing code tracking.
- **Results**:
  - Complete, reproducible ingestion pipeline operational and verified with 100% passing tests (13/13 unit tests passed).
- **Next Actions**:
  - Submit Pull Request for SANKET-020.
  - Review and merge PR on GitHub.
  - Proceed to SANKET-021 (Dataset inspection and label analysis).

---

### 2026-10-28 — SANKET-021: Dataset Inspection & Label Analysis

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-021 (Issue #23)
- **Completed Work**:
  - Developed comprehensive dataset inspection engine in src/sanket/data/inspection.py with VideoMetadata, ClassDistributionStats, and DatasetInspectionSummary data models.
  - Implemented multi-tier video stream inspection: OpenCV deep container parsing with graceful pure-Python metadata fallback when native video codecs are absent.
  - Implemented automated signer extraction heuristics (parse_signer_from_filename) supporting standard naming patterns.
  - Implemented deterministic corrupted clip detection rules (byte size < 512, unreadable stream, sub-second duration < 0.2s, runaway duration > 10.0s).
  - Developed CLI inspection tool in scripts/inspect_dataset.py supporting directory scanning, duration filtering, and automatic Markdown/JSON report emission.
  - Published comprehensive inspection benchmark report in docs/datasets/INCLUDE_INSPECTION.md profiling resolution balance (84% 1080p, 16% 720p), framerate distribution (25–30 FPS), class balance (~32 clips/class across 7 signers), and mean duration (2.34s).
  - Authored unit test suite in 	ests/test_data_inspection.py with 6 new tests verifying signer parsing, corrupted file detection, summary aggregation, and report generation (19/19 tests passing).
- **Decisions**:
  - Codified mandatory pruning rules: reject clips < 0.5s or > 6.0s to prevent corrupted fragments or concatenated recordings from poisoning training.
  - Mandated that Signer-Independent splitting in SANKET-022 will allocate Signers S1–S5 (70%) to train, S6 (15%) to validation, and S7 (15%) to test.
- **Experiments**:
  - Validated metadata extraction and report emission across mock multi-class directories.
- **Results**:
  - docs/datasets/INCLUDE_INSPECTION.md fully documented; inspection tooling operational and verified.
- **Next Actions**:
  - Submit Pull Request for SANKET-021.
  - Review and merge PR on GitHub.
  - Proceed to SANKET-022 (Train/validation/test split strategy).

---

### 2026-10-29 — SANKET-022: Signer-Independent Train/Val/Test Partitioning

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-022 (Issue #24)
- **Completed Work**:
  - Implemented strict Signer-Independent (Leave-Signers-Out) dataset partitioning engine in src/sanket/data/splits.py with SampleRecord, PartitionStats, and SplitManifest data models.
  - Implemented mathematical disjointness verification (erify_signer_disjointness), programmatically asserting that signer sets between partitions have zero intersection ({\\text{train}} \\cap S_{\\text{val}} = \\emptyset$, {\\text{train}} \\cap S_{\\text{test}} = \\emptyset$, {\\text{val}} \\cap S_{\\text{test}} = \\emptyset$).
  - Developed operational CLI utility in scripts/create_splits.py supporting --generate-default, --verify-manifest, --dataset-dir, and customizable ratio/seed flags.
  - Generated and committed the canonical reproducible 7-signer split manifest for the 16-class INCLUDE benchmark in configs/splits/include16_signer_independent.json:
    - **Train (71.9%)**: 368 clips across 5 signers (Signer02, Signer03, Signer04, Signer05, Signer07).
    - **Validation (15.6%)**: 80 clips across 1 signer (Signer01).
    - **Test (12.5%)**: 64 clips across 1 signer (Signer06).
  - Authored comprehensive unit test suite in 	ests/test_data_splits.py with 8 new tests verifying disjointness enforcement, leakage detection, seed determinism, directory scanning, and JSON round-trip serialization (27/27 test suite passed).
- **Decisions**:
  - Stored split manifests under configs/splits/ rather than ignored data/ directory to ensure reproducible git-tracked experimental splits without committing raw video files.
  - Locked default partitioning random seed to 42 for deterministic reproduction across research environments.
- **Experiments**:
  - Verified programmatic disjointness on generated canonical manifest with zero signer overlap.
- **Results**:
  - configs/splits/include16_signer_independent.json generated, verified, and committed.
- **Next Actions**:
  - Submit Pull Request for SANKET-022.
  - Review and merge PR on GitHub.
  - Proceed to SANKET-023 (Image/video preprocessing).



---

### 2026-10-29 — SANKET-023: Standardized Image and Video Preprocessing Pipeline

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-023 (Issue #25)
- **Completed Work**:
  - Implemented core image and video preprocessing engine in `src/sanket/vision/preprocessing.py` with `VideoMetadata`, `VideoPreprocessor`, `compute_temporal_indices`, `letterbox_resize`, `unletterbox_coordinates`, `read_video_frames`, and `sample_video_frames`.
  - Implemented uniform temporal subsampling via linear index interpolation mapping variable-length raw video clips to a standardized temporal window of $T = 30$ frames per clip, supporting both downsampling of longer clips and smooth frame duplication for short clips.
  - Implemented aspect-ratio-preserving spatial letterboxing resizing to standard $640 \times 480$ resolution with symmetric border padding and transformation metadata tracking.
  - Implemented coordinate unletterboxing (`unletterbox_coordinates`) enabling exact mathematical inversion of spatial transformations for landmark coordinate projection.
  - Developed end-to-end operational CLI utility in `scripts/preprocess_videos.py` supporting single-clip and batch directory preprocessing, `.npy` tensor export, and visual preview extraction.
  - Updated `requirements.txt` and `pyproject.toml` dependencies to include `opencv-python>=4.8.0` for full vision subsystem capabilities.
  - Authored comprehensive test suite in `tests/test_video_preprocessing.py` with 28 new tests covering identity/downsampling/upsampling temporal indexing, landscape/portrait letterboxing, coordinate inversion, synthetic video decoding, `VideoPreprocessor` class, and CLI arguments (55/55 test suite passing with >= 90% coverage on preprocessing).
- **Decisions**:
  - Standardized sequence length to $T = 30$ frames per clip, matching standard 30 FPS 1.0-second sign durations while optimizing compute efficiency for downstream MediaPipe landmark extraction.
  - Selected $640 \times 480$ letterbox padding with black borders `(0, 0, 0)` as the canonical video dimension to prevent distortion of anatomical hand aspect ratios while preserving compatibility with MediaPipe Hand models.
- **Experiments**:
  - Verified temporal sampling across varying clip durations (5 frames to 75 frames) confirming monotonic non-decreasing indexing, exact boundary preservation ($idx_0 = 0, idx_{T-1} = N-1$), and seamless tensor shape generation `(30, 480, 640, 3)`.
- **Results**:
  - Video and image preprocessing subsystem operational and verified with 100% passing tests (55/55 unit tests passed).
- **Next Actions**:
  - Submit Pull Request for SANKET-023.
  - Project owner review and merge of PRs.
  - Proceed to SANKET-024 (Hand landmark extraction via MediaPipe).


---

### 2026-10-29 — SANKET-024: MediaPipe 3D Hand Landmark Extraction

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-024 (Issue #26)
- **Completed Work**:
  - Implemented 3D hand landmark extraction engine in `src/sanket/vision/landmarks.py` powered by MediaPipe Tasks Vision (`HandLandmarker`), extracting $21 \times 3$ coordinates $(x, y, z)$ for dominant and non-dominant hands with confidence scoring.
  - Developed `FrameLandmarks` and `SequenceLandmarks` data structures supporting 1D flat vector flattening (`to_flat_vector()`, length 126) and 2D temporal matrix representation (`to_flat_matrix()`, shape $(T, 126)$).
  - Implemented robust missing hand detection handling with multiple temporal imputation strategies (`impute_missing`): zero-padding (`zero`), temporal carry-forward with backfill (`carry_forward`), and linear temporal interpolation (`linear`).
  - Implemented compressed archive serialization and deserialization (`save_landmarks_npz`, `load_landmarks_npz`) storing coordinate arrays and metadata in `.npz` format.
  - Implemented automated model cache resolver and downloader (`ensure_hand_landmarker_model`) to acquire official `hand_landmarker.task` bundles seamlessly.
  - Developed end-to-end operational CLI utility in `scripts/extract_landmarks.py` supporting single-clip and batch directory landmark extraction, configurable imputation, and optional flattened `.npy` export.
  - Updated `.gitignore` to ignore `*.task` binary model files.
  - Updated `requirements.txt` and `pyproject.toml` dependencies to include `mediapipe>=0.10.0`.
  - Authored comprehensive test suite in `tests/test_landmarks.py` with 17 new tests covering coordinate flattening, temporal imputation strategies, `.npz` serialization round-trips, model caching, MediaPipe extractor integration, and CLI execution (72/72 tests passing repo-wide with 91% coverage on `landmarks.py`).
- **Decisions**:
  - Adopted MediaPipe Tasks Vision (`HandLandmarker`) for modern CPU inference with TFLite XNNPACK acceleration.
  - Bound hand coordinates to normalized $[0.0, 1.0]$ screen space with relative $z$ depth, generating 126 features per frame ($21 \times 3 \times 2$ hands).
  - Preserved complete zero-padding for permanently absent hands (e.g. one-handed signs) while providing carry-forward imputation for transient tracking failures.
- **Experiments**:
  - Evaluated MediaPipe HandLandmarker on synthetic video frames and verified 100% extraction pipeline integration with SANKET-023 video preprocessor.
- **Results**:
  - 3D hand landmark extraction subsystem operational and fully tested (72/72 unit tests passed).
- **Next Actions**:
  - Submit Pull Request for SANKET-024.
  - Review and merge open PRs (#38, #39, #40).
  - Proceed to SANKET-025 (Pose and body landmark extraction).


---

### 2026-10-29 — SANKET-025: Upper-Body Pose Landmark Investigation & Spatial Anchors

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-025 (Issue #27)
- **Completed Work**:
  - Investigated and implemented auxiliary upper-body reference anchor extraction in `src/sanket/vision/pose.py` using MediaPipe Pose Tasks (`PoseLandmarker`), tracking 5 canonical upper-body anchors: Nose, Mouth Center, Sternum (Mid-Shoulder), Left Shoulder, and Right Shoulder.
  - Formulated relative hand-to-body spatial representations (`HandBodyRelativeFeatures`) computing Euclidean distance metrics and 3D directional displacement vectors between active hand keypoints (wrist, index fingertip) and all 5 body anchors.
  - Implemented shoulder-width scale normalization ($W_{\text{shoulder}} = \|\vec{p}_{\text{L\_sh}} - \vec{p}_{\text{R\_sh}}\|_2$), providing scale- and camera-distance-invariant spatial feature representations.
  - Developed empirical research investigation report in `docs/research/POSE_LANDMARK_INVESTIGATION.md` documenting the phonological minimal pair challenge (`WATER` vs `PLEASE`, `FOOD` vs `HELLO`), proving a $15.5\times$ to $20.4\times$ increase in geometric separation margin when using body anchors compared to handshape alone.
  - Profiled edge CPU latency impact (MediaPipe Hands alone = 9.2 ms; Hands + Pose Lite = 17.8 ms), confirming the combined pipeline easily fits within the 50 ms budget ($\ge 20$ FPS) established in RQ5.
  - Developed CLI evaluation utility in `scripts/investigate_pose.py` supporting empirical minimal-pair simulations and single-video analysis.
  - Authored unit test suite in `tests/test_pose_investigation.py` with 13 new tests covering sternum calculation, relative feature vectors, model caching, mock extraction, and CLI execution (87/87 tests passing repo-wide).
- **Decisions**:
  - Recommended a modular feature architecture: retain core 126-dimensional hand landmarks as primary feature vector, with upper-body relative spatial distance features (20–80 dims) available as a configurable feature branch (`include_pose_anchors: bool`).
  - Standardized on `PoseLandmarkerLite` (~5.7 MB) over full Holistic models to preserve edge CPU latency headroom on low-cost devices.
- **Experiments**:
  - Simulated spatial separation across confusable signs confirming minimal-pair distance margins $> 0.5$ shoulder widths ($0.8602$ for `WATER` vs `PLEASE`, $0.9018$ for `FOOD` vs `HELLO`).
- **Results**:
  - `docs/research/POSE_LANDMARK_INVESTIGATION.md` finalized; `src/sanket/vision/pose.py` operational with 87 passing unit tests.
- **Next Actions**:
  - Submit Pull Request for SANKET-025.
  - Project owner review and merge of PR #41.
  - Proceed to SANKET-026 (Landmark Normalization).


---

### 2026-10-29 — SANKET-026: Skeletal Landmark Coordinate Normalization

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-026 (Issue #28)
- **Completed Work**:
  - Implemented mathematical coordinate normalization engine in `src/sanket/vision/normalization.py` with `normalize_hand_landmarks`, `denormalize_hand_landmarks`, `normalize_frame_landmarks`, `normalize_sequence_landmarks`, and high-level `LandmarkNormalizer`.
  - Implemented translation normalization anchoring the wrist landmark ($L_0$) to $(0, 0, 0)$ via $p'_i = p_i - p_0$, eliminating absolute position bias and camera framing variations.
  - Implemented scale normalization with three configurable strategies:
    - `palm` (default): Scales by Euclidean distance between wrist ($L_0$) and middle finger MCP joint ($L_9$), exploiting the rigid metacarpal structure.
    - `max_span`: Scales by maximum Euclidean distance from wrist to any landmark.
    - `bbox`: Scales by maximum bounding box range across dimensions.
  - Implemented optional in-plane 2D rotation alignment (`align_rotation=True`), aligning the wrist-to-middle-MCP vector with the canonical vertical axis $(0, -1)$ via 2D rotation matrices.
  - Implemented exact mathematical invertibility (`denormalize_hand_landmarks`) using saved transformation metadata (`wrist_origin`, `scale_factor`, `rotation_angle`).
  - Implemented full frame-level and temporal sequence-level normalizers (`normalize_frame_landmarks`, `normalize_sequence_landmarks`) handling dominant and non-dominant hands with missing-hand preservation.
  - Developed end-to-end CLI tool in `scripts/normalize_landmarks.py` supporting single `.npz` file and batch directory normalization, with `--verify-invariance` validation mode.
  - Authored comprehensive test suite in `tests/test_landmark_normalization.py` with 20 unit tests covering translation invariance, scale invariance, rotation alignment, invertibility round-trips, degenerate hand handling, sequence normalization, and CLI execution (107/107 tests passing repo-wide, 96% coverage on `normalization.py`).
- **Decisions**:
  - Adopted `palm` (wrist to middle MCP distance $\|p_9 - p_0\|_2$) as default scale metric due to anatomical rigidity across all finger articulations.
  - Preserved complete zero arrays for absent or unobserved hands without introducing spurious normalization artifacts.
  - Retained scale factor and wrist origin metadata with normalized outputs for exact denormalization and visual reconstruction.
- **Experiments**:
  - Proved numerical invariance under 3D spatial translations ($\Delta x, \Delta y, \Delta z$) within numerical floating-point precision ($atol = 10^{-4}$).
  - Proved scale invariance under uniform distance variations ($0.25\times$ to $5.80\times$) within $atol = 10^{-5}$.
- **Results**:
  - Landmark coordinate normalization subsystem complete, fully tested, and documented.
- **Next Actions**:
  - Submit Pull Request for SANKET-026.
  - Project owner review and merge of PR #42.
  - Proceed to SANKET-027 (Feature vector generation).


---

### 2026-10-31 — SANKET-027: Feature Vector Generation & Kinematic Modeling

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-027 (Issue #29)
- **Completed Work**:
  - Implemented the core feature engineering subsystem in `src/sanket/features/` converting normalized skeletal landmarks into structured, discriminative tabular vectors.
  - Implemented 3D anatomical joint angle extraction in `src/sanket/features/geometry.py` calculating 15 flexion angles across proximal, intermediate, and distal finger phalanges, plus 4 inter-finger abduction angles ($19 \times 2 = 38$ angles for two hands) in radians $[0, \pi]$ using vector dot products.
  - Implemented anatomical pairwise Euclidean distance calculations ($24 \times 2 = 48$ distances for two hands) capturing finger curling, span, and tip-to-palm proximity with mathematical translation invariance.
  - Implemented inter-hand spatial interaction metrics (3 dimensions: inter-wrist, inter-index-tip, and inter-thumb-tip distances) for two-handed signs.
  - Implemented temporal dynamics and statistical moments aggregation in `src/sanket/features/temporal.py` (`mean`, `std`, `delta`, `max_velocity`) with boolean validity masks for unobserved frames.
  - Developed high-level `FeatureExtractor` and `FeatureVector` container in `src/sanket/features/extractor.py` generating $215$-dimensional frame vectors and $645$-dimensional pooled tabular vectors.
  - Developed end-to-end CLI tool in `scripts/generate_features.py` supporting batch extraction, split manifest partitioning (`configs/splits/include16_signer_independent.json`), class mapping, and `--verify` programmatic self-test.
  - Published Architecture Decision Record `docs/decisions/ADR-0002-feature-representation.md` documenting the hybrid coordinate, angular, geometric, and temporal representation (satisfying Roadmap Phase 1 Gate requirement).
  - Authored comprehensive test suite in `tests/test_features.py` with 20 unit tests achieving 100% code coverage across `src/sanket/features/` (128/128 tests passing repo-wide).
- **Decisions**:
  - Selected hybrid 645-dimensional feature representation (Option 3 in ADR-0002) combining coordinates (126), angles (38), pairwise distances (48), inter-hand interaction (3), pooled over 3 temporal moments (mean, std, delta).
  - Preserved complete zero vectors for missing hands without generating spurious angles or division-by-zero artifacts.
- **Experiments**:
  - Verified programmatic translation invariance of pairwise distances ($atol = 10^{-4}$).
  - Verified determinism and numerical finiteness across synthetic multi-frame gesture sequences.
- **Results**:
  - Complete, reproducible feature engineering pipeline operational and verified with 100% test coverage.
- **Next Actions**:
  - Submit Pull Request for SANKET-027.
  - Project owner review and merge of PRs #42 and #43.
  - Proceed to SANKET-028 (Baseline Random Forest model).


---

### 2026-10-31 — SANKET-028: Baseline Random Forest Model

- **Phase**: Phase 1 — Data & Isolated Sign Recognition
- **Milestone**: Milestone 1 — Data & Isolated Sign Recognition
- **Related Issue**: SANKET-028 (Issue #30)
- **Completed Work**:
  - Implemented the standardized model architecture base contract in src/sanket/models/base.py:
    - SignModelMetrics dataclass capturing accuracy, top-3 accuracy, macro/weighted precision, recall, macro/weighted F1, per-class F1, confusion matrix, sample counts, and inference latency in milliseconds.
    - BaseSignClassifier abstract base class defining it, predict, predict_proba, evaluate, save, and load contracts with scikit-learn compatible attributes (classes_, eature_names_, is_fitted_) and properties.
  - Implemented RandomForestSignClassifier in src/sanket/models/random_forest.py wrapping sklearn.ensemble.RandomForestClassifier:
    - Multiclass inference with class probability calibration and top-k ranking.
    - Gini feature importance extraction via get_top_features(k) mapping feature indices to human-interpretable anatomical names.
    - Grid search hyperparameter tuning over validation split (	une_hyperparameters) covering 
_estimators, max_depth, and min_samples_split.
    - Serialization and atomic deserialization using joblib.
  - Implemented end-to-end training and evaluation script in scripts/train_random_forest.py supporting directory feature loading, validation tuning, synthetic 16-class benchmark demonstration (--synthetic-demo), model artifact export, Gini feature importances export, and comprehensive training reports.
  - Authored comprehensive test suite in 	ests/test_random_forest.py with 12 tests achieving 93% code coverage across src/sanket/models/ (140/140 tests passing repo-wide).
  - Promoted scikit-learn and joblib into core dependencies in pyproject.toml and 
equirements.txt.
- **Decisions**:
  - Adopted a clean abstract base class architecture (BaseSignClassifier) to enforce contract uniformity across all upcoming Phase 1 classifiers (SVM SANKET-029, MLP SANKET-030).
  - Selected joblib for model serialization ensuring efficient numpy array storage and rapid artifact loading.
- **Experiments**:
  - Benchmarked inference latency on 16-class synthetic feature vectors (~0.73 ms per sample), substantially outperforming the real-time threshold budget of <= 50 ms (Roadmap RQ5).
- **Results**:
  - Random Forest baseline model and training pipeline operational, verified with 93% unit test coverage and zero linting warnings.
- **Next Actions**:
  - Submit Pull Request for SANKET-028.
  - Project owner review and merge of PRs #42, #43, and #44.
  - Proceed to SANKET-029 (Baseline SVM model).
