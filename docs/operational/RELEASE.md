# SANKET (संकेत) — Release Engineering & Versioning Protocol

> **Semantic Versioning, Milestone Gating, Model Checkpointing, and Tagging Workflows**  
> *Authoritative specifications for creating, tagging, and distributing releases.*

---

## 1. Versioning Strategy

SANKET follows **Semantic Versioning 2.0.0 (`vMAJOR.MINOR.PATCH`)**:

- **MAJOR (`X.0.0`)**: Backwards-incompatible changes, architectural redesigns, or foundational vocabulary restructuring.
- **MINOR (`0.X.0`)**: Gated phase completions (e.g., Phase 1 certified baseline $\rightarrow$ `v0.1.0`; Phase 2 Real-Time MVP $\rightarrow$ `v0.2.0`; Phase 3 Dynamic Recognition $\rightarrow$ `v0.3.0`).
- **PATCH (`0.0.X`)**: Bug fixes, performance optimizations, documentation updates, or non-breaking feature enhancements.

---

## 2. Milestone Release Checklist & Quality Gates

Before any milestone release is tagged, the following gates must be satisfied:

```markdown
### Milestone Release Gate Checklist

- [ ] All issues assigned to the active milestone are resolved, reviewed, and closed.
- [ ] Pytest test suite passes with 100% success rate across Python 3.10 and 3.11 (`pytest -q`).
- [ ] Code style and linting pass with zero warnings (`ruff check .`).
- [ ] Setup verification passes on clean virtual environment (`python scripts/verify_setup.py`).
- [ ] Formal benchmark manifest generated and archived in `docs/experiments/manifests/`.
- [ ] Master roadmap updated in `docs/roadmap/SANKET_ROADMAP.md`.
- [ ] Project log updated in `docs/development/PROJECT_LOG.md`.
- [ ] No personal paths or sensitive data committed (`git diff origin/main`).
```

---

## 3. Release & Tagging Procedure

1. **Verify Clean Main Branch**:
   ```bash
   git checkout main
   git pull origin main
   pytest -q
   ruff check .
   ```

2. **Update Version in Packaging**:
   Update `version = "X.Y.Z"` in `pyproject.toml` and record changes in `docs/operational/CHANGELOG.md`.

3. **Create Annotated Git Tag**:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0: Phase 1 Data & Isolated Sign Recognition Certification"
   git push origin v0.1.0
   ```

4. **Create GitHub Release**:
   ```bash
   gh release create v0.1.0 \
       --title "SANKET v0.1.0 — Phase 1 Certification & Baseline Models" \
       --notes-file docs/experiments/EXP-P1-001-baseline-comparison.md
   ```

