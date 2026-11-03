# SANKET Documentation Formatting & Style Guide

> **Authoritative Standards for Repository Documentation, Issues, PRs, and Architecture Records**  
> *Enforced across all Markdown documents, technical specifications, and research records.*

---

## 1. File Naming & Organization

- **Root & Meta Documentation**: Use uppercase snake/hyphen naming for repository-level files:
  - `README.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, `ARCHITECTURE.md`, `TECH_STACK.md`, `TESTING.md`, `AGENTS.md`
- **Subsystem & Technical Documentation**: Use uppercase naming inside topic directories:
  - `docs/technical/API_DOCUMENTATION.md`
  - `docs/technical/DESIGN_SYSTEM.md`
  - `docs/technical/BUSINESS_LOGIC.md`
  - `docs/operational/DEPLOYMENT.md`
  - `docs/operational/TROUBLESHOOTING.md`
- **Architecture Decision Records (ADRs)**: Stored in `docs/decisions/` with zero-padded identifiers:
  - `ADR-0001-project-foundations.md`
  - `ADR-0002-feature-representation.md`
  - `ADR-0003-baseline-classifier-selection.md`
- **Experiment Records**: Stored in `docs/experiments/`:
  - `EXP-0001-random-forest-baseline.md`
  - `EXP-P1-001-baseline-comparison.md`

---

## 2. Typography, Punctuation & Spacing

### 2.1 Spacing Inside Delimiters
Never insert loose spaces inside parentheses, backticks, or brackets, and never place spaces before punctuation marks:

```markdown
<!-- CORRECT -->
Establish a modular repository structure (`src/sanket/`, `docs/`, `tests/`, `configs/`, `scripts/`).

<!-- INCORRECT: Notice loose spaces before commas and inside parentheses -->
Establish a modular repository structure ( src/sanket/ , docs/ , tests/ , configs/ , scripts/ ) .
```

### 2.2 Character Encoding & Special Symbols
- **UTF-8 Encoding**: Every file must be encoded in UTF-8 without BOM.
- **Dashes**: Use standard em dashes (`—`) or standard double hyphens (`--`). Never use corrupted byte sequences (`?\"` or ``).
- **Arrows**: Use standard arrows in prose (`->` or `$\rightarrow$`). In Mermaid diagrams, use standard `-->`.
- **Mathematical Symbols**: Use KaTeX / LaTeX math blocks for formulas: `$s = \|\mathbf{p}_9 - \mathbf{p}_0\|_2$`.

---

## 3. Standard Document Headers

Every major technical specification or ADR must begin with a standardized header block:

```markdown
# Title of Document

> **Brief 1-2 sentence high-level summary or subtitle**  
> *Authoritative scope and audience description.*

---

## 1. Section Name
...
```

For Architecture Decision Records (ADRs):

```markdown
# ADR-XXXX: Title of Decision

- **Status**: Proposed | Accepted | Deprecated | Superseded
- **Date**: YYYY-MM-DD
- **Deciders**: [Project Maintainer / Sanket Team]
- **Related Issue**: SANKET-XXX

---

## Decision
...
```

---

## 4. Tone, Authorship & Attribution Rules

1. **Grounded Technical Tone**: Write in clear, professional, concise engineering prose. Avoid bloated corporate jargon, self-aggrandizing role titles (e.g., "Principal Software Architect", "Lead Governance Engineer"), and hype.
2. **No AI Self-Attribution**: AI assistants and IDEs (e.g., Antigravity, Copilot, ChatGPT) must **never** be cited as deciders, contributors, co-authors, or team members in ADRs, commit messages, or documentation. Authorship belongs to the human project owner (`Anveshna`) and the repository team (`Sanket Team`).
3. **Verified Facts vs Assumptions**: Clearly distinguish empirical benchmarks (with recorded commit hashes and parameters) from planned future explorations.

---

## 5. Privacy & Path Cleanliness

> [!CAUTION]
> **Strict Non-Negotiable Rule**:
> - Never include absolute local machine paths (e.g., `C:\Users\...` or `/home/username/...`) in any documentation, commit, PR description, or issue.
> - Always use repository-relative paths: `src/sanket/vision/capture.py`, `docs/experiments/README.md`.
> - Never expose API keys, credentials, or private configuration.

---

## 6. Code Blocks & Tables

- **Syntax Highlighting**: Always declare the language in fenced code blocks:
  ````markdown
  ```python
  def normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
      ...
  ```
  ````
- **Tables**: Use formatted Markdown tables with aligned header delimiters:
  ```markdown
  | Metric | Random Forest | Support Vector Machine | Target |
  | :--- | :---: | :---: | :---: |
  | Top-1 Accuracy | 93.3% | 80.0% | >= 90.0% |
  ```
