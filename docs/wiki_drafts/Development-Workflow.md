# Development Workflow

Sanket enforces rigorous engineering hygiene and human merge control.

## Principles
1. **Never push directly to `main`**: All changes must proceed through an issue-linked branch and a reviewed Pull Request.
2. **Issue Traceability**: Every branch must correspond to an assigned GitHub issue.
3. **Branch Naming**:
   - `feat/SANKET-XXX-description`
   - `fix/SANKET-XXX-description`
   - `research/SANKET-XXX-description`
   - `docs/SANKET-XXX-description`
   - `test/SANKET-XXX-description`
   - `security/SANKET-XXX-description`
4. **Pre-PR Verification**:
   - `pytest` (all tests must pass)
   - `ruff check .` (zero linting errors)
   - `python scripts/verify_setup.py` (foundation checks must pass)
5. **Human Merge Control**: Pull Requests are merged exclusively by the project owner.
