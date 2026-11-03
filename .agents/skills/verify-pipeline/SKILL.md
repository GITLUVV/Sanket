---
name: verify-pipeline
description: Verifies local environment setup, Ruff code formatting and linting, and runs the complete Pytest test suite.
---

# Verify Pipeline Skill

Use this procedure before opening any Pull Request or after making code changes to ensure repository integrity.

## Execution Steps

1. **Verify Environment Setup**:
   ```bash
   python scripts/verify_setup.py
   ```
   Confirm all 4 checks pass: Python version, dependencies, package imports, and configuration YAML.

2. **Verify Code Style & Formatting**:
   ```bash
   ruff check .
   ```
   If any style issues appear, fix them or run `ruff check --fix .`.

3. **Verify Unit & Integration Tests**:
   ```bash
   pytest -q
   ```
   Confirm all 206+ tests pass without errors.

4. **Verify Privacy & Path Hygiene**:
   Ensure no local personal Windows paths or temporary files are staged in Git:
   ```bash
   git status
   ```

