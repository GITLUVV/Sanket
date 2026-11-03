# Git Governance & Pull Request Rules

## Branch Naming Policy
Always create task branches from `main` using standardized prefixes matching assigned issue IDs:
- `feat/SANKET-XXX-short-description` (Features & pipeline additions)
- `fix/SANKET-XXX-short-description` (Bug fixes)
- `research/SANKET-XXX-short-description` (Investigations, diagnostics, comparisons)
- `docs/SANKET-XXX-short-description` (Documentation, ADRs, logs)
- `test/SANKET-XXX-short-description` (Test harnesses & benchmarks)
- `security/SANKET-XXX-short-description` (Vulnerability fixes & hardening)

## Pull Request Lifecycle
- Work on **one issue at a time**: `Issue -> Branch -> Implementation -> PR -> Human Review & Merge -> Next Issue`.
- Never push directly to `main`.
- Never independently merge Pull Requests; all PRs require human project owner review and merge.

## PR Metadata Enforcement (`pr-metadata.yml`)
Every PR must include:
1. **Title**: Conventional Commits format, e.g. `feat(vision): add webcam capture pipeline (SANKET-040)`.
2. **Milestone**: Assigned to active milestone (e.g., `Milestone 2 — Real-Time Sanket MVP`).
3. **Labels**: Four distinct categories (`type:*`, `area:*`, `priority:*`, `phase:*`).
4. **Closing Reference**: Explicit closing keyword in body: `Closes #51`.
5. **Testing Proof**: Verifiable local test output and acceptance criteria checkboxes.

