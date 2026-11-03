# Security & Privacy Guidelines

## Biometric Privacy Mandate
- **Strict Prohibition**: Never write raw video recordings, unanonymized human facial crops, or identifiable body imagery to disk storage.
- Video frames captured from live cameras must reside strictly in volatile memory (RAM) within bounded circular queues and be released immediately after landmark extraction.
- Only anonymized, non-reversible skeletal keypoint coordinates (`x, y, z`) may be persisted for dataset curation.

## Repository Privacy & Path Cleanliness
- **Zero Local Environment Paths**: Never include absolute Windows directory paths (e.g., `C:\Users\...`) or machine usernames in code, commits, PR bodies, issue comments, or documentation.
- Use repository-relative paths exclusively (e.g., `src/sanket/vision/capture.py`).
- Add machine-specific configuration, virtual environments (`venv/`), caches (`.pytest_cache`, `.ruff_cache`), and model checkpoints (`*.joblib`) to `.gitignore`.

## Secrets & Dataset Governance
- Never commit API keys, cloud credentials, or secrets.
- Never download, ingest, or process external datasets without verified entries in `docs/datasets/DATASET_REGISTER.md` and documented owner authorization.
- Pin all third-party GitHub Actions in `.github/workflows/` to immutable full commit SHAs.

