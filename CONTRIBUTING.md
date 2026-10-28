# Contributing to Sanket

Thank you for your interest in contributing to **Sanket — AI-Assisted Indian Sign Language Accessibility System**.

Sanket is an academic and research-oriented AI engineering project. To preserve scientific rigor, reproducible experimentation, and architectural modularity, all contributors and automated assistants must adhere to the engineering guidelines below.

---

## 1. Project Principles

1. **Keep the project modular**: Separate vision ingestion, feature extraction, model inference, and language rendering into clean interfaces.
2. **Keep research decisions documented**: Every significant algorithmic or architectural choice must be recorded in an Architecture Decision Record (`docs/decisions/`).
3. **Do not silently change technical decisions**: Major shifts in models, dataset choices, or dependencies require discussion and documentation.
4. **Record experiments and results**: Every ML experiment must be logged in `docs/experiments/` with reproducible hyperparameters, metrics, and error analyses.
5. **Keep GitHub Issues focused**: Issues must correspond to concrete, actionable research or engineering tasks.
6. **Use Pull Requests for meaningful changes**: Never push directly to `main`.
7. **Small, reviewable PRs**: Prefer focused PRs addressing a single issue over massive monolithic submissions.
8. **Do not overclaim**: Never claim "full continuous sign language translation" when the system currently implements constrained isolated sign recognition.
9. **Strict dataset licensing**: Audit dataset licenses and redistribution rights in `docs/datasets/DATASET_REGISTER.md` prior to ingestion.
10. **Privacy first**: Never commit camera footage, raw biometric recordings, credentials, or private datasets.


---

## 2. Branching & Merge Workflow

The `main` branch is protected and represents validated, passing milestones. Direct commits to `main` are strictly forbidden.

### Branch Lifecycle

```text
main
  ↓
feature / research / fix / docs / refactor branch
  ↓
Pull Request
  ↓
Automated CI checks + Project Owner Review
  ↓
Merge into main
```

### Branch Naming Conventions

Use standardized branch names matching assigned issue IDs:

- `feature/SANKET-XXX-description` — New features or pipeline additions
- `fix/SANKET-XXX-description` — Bug fixes or defect resolutions
- `research/SANKET-XXX-description` — Exploration, dataset surveys, feasibility investigations
- `docs/SANKET-XXX-description` — Documentation, ADRs, logs, or Wiki updates
- `refactor/SANKET-XXX-description` — Code improvements with identical external behavior
- `chore/SANKET-XXX-description` — Tooling, dependency, or configuration maintenance

---

## 3. Human Merge Control Rules

- **All Pull Requests must be reviewed and merged by the project owner.**
- Automated AI assistants and contributors must create the branch, implement changes, and submit a detailed Pull Request.
- **Never independently merge Pull Requests.** The project owner controls progression.
- Work should proceed **one issue at a time**:
  ```text
  Issue → Implementation → PR → Human Review & Merge → Next Issue
  ```

---

## 4. Label Taxonomy

Apply labels consistently across four standardized dimensions:

### Type
- `type:feature` — New feature or pipeline capability
- `type:bug` — Defect, regression, or calculation error
- `type:research` — Investigation, literature review, dataset exploration
- `type:experiment` — Machine learning experiment or baseline comparison
- `type:documentation` — Documentation or record updates
- `type:refactor` — Code refactoring without behavior change
- `type:chore` — Repository maintenance, CI, dependencies, or configuration

### Area
- `area:dataset` — Datasets, ingestion, annotations, and registers
- `area:vision` — Computer vision, landmarks, frame extraction, MediaPipe
- `area:ml` — Machine learning models, features, training, inference
- `area:temporal` — Temporal modeling, sequential processing, LSTM/GRU
- `area:nlp` — Natural language processing, sentence formation, grammar
- `area:ui` — User interface, feedback display, visualization
- `area:api` — API endpoints, service interfaces
- `area:deployment` — Packaging, containerization, edge inference
- `area:security` — Security, privacy, credentials, compliance
- `area:testing` — Unit tests, integration tests, benchmarks, evaluation

### Priority
- `priority:P0` — Critical blocker; address immediately
- `priority:P1` — High priority; necessary for current milestone
- `priority:P2` — Medium priority; scheduled standard sequence
- `priority:P3` — Low priority; nice to have or deferred

### Phase
- `phase:0` through `phase:6` matching the roadmap phases

---

## 5. Commit Guidelines

Follow the Conventional Commits specification:

```text
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `research`.  
Example: `feat(vision): add landmark normalization utility (Closes #1)`

---

## 6. Pre-PR Checklist

Before opening a Pull Request:
1. Ensure all unit tests pass: `pytest`
2. Ensure code style is formatted: `ruff check .`
3. Verify no secrets, camera videos, or raw datasets are staged.
4. Update corresponding documentation (ADRs, roadmap, project log).
