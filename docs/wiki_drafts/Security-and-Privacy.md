# Security & Privacy Safeguards

## Core Principles
1. **Camera Privacy**: Video feeds are processed in volatile memory and never transmitted over external networks without explicit user authorization.
2. **Zero Biometric Media Tracking**: Raw camera footage, participant likenesses, or private datasets are never committed to Git.
3. **No Unsafe Deserialization**: Model loading is restricted to trusted local model paths.
4. **Least-Privilege GitHub Actions**: All workflows declare explicit least-privilege permissions (`contents: read`).
5. **Dependency Supply Chain**: Dependencies are pinned and monitored via Dependabot alerts.

## Reporting Vulnerabilities
Do NOT open public GitHub issues for security vulnerabilities. Send confidential reports to `anveshna2025@gmail.com` as outlined in `SECURITY.md`.
