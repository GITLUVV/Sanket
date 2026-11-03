# SANKET (संकेत) — Security & Privacy Incident Response Plan

> **Procedures for Handling Privacy Breaches, Credential Leaks, Licensing Disputes, and CI Failures**  
> *Authoritative response playbooks for maintainers and contributors.*

---

## 1. Incident Classification & Severity Levels

| Severity Level | Definition | Examples | Response Window |
| :--- | :--- | :--- | :---: |
| **SEV-1 (Critical)** | Direct exposure of secrets or biometric data. | Personal API keys pushed to public repo; raw face video frames committed. | **< 1 hour** |
| **SEV-2 (High)** | Data governance or licensing violation. | Ingesting third-party dataset without documented owner approval; compromised CI action. | **< 6 hours** |
| **SEV-3 (Medium)** | Privacy policy violation (non-biometric). | Local absolute Windows path (`C:\Users\...`) committed in documentation or issue. | **< 24 hours** |
| **SEV-4 (Low)** | Non-blocking CI failure or broken documentation link. | Linting failure in PR; broken anchor link in Wiki draft. | **Next PR / Sprint** |

---

## 2. Incident Response Playbooks

### Playbook A: Accidental Secret or Credential Leak (SEV-1)
1. **Immediate Revocation**: Invalidate the leaked token or API key in the provider console immediately.
2. **History Rewriting**:
   - Use `git filter-repo` or BFG Repo-Cleaner to purge the sensitive commit from the Git tree:
     ```bash
     git filter-repo --invert-paths --path sensitive_file.txt
     git push origin main --force
     ```
3. **Audit**: Inspect Git log and GitHub audit log for unauthorized access.

### Playbook B: Accidental Commit of Biometric Media (SEV-1)
1. **Purge Commit**: Remove raw images or video files from Git history using `git filter-repo`.
2. **Hard Push**: Force-push cleaned history to GitHub (`origin/main`).
3. **Notify Maintainer**: Alert repository lead to ensure local developer working copies are re-cloned.

### Playbook C: Accidental Path Privacy Exposure (SEV-3)
1. **Edit Content**: Replace absolute paths (`C:\Users\...`) with repository-relative paths (`src/sanket/...`).
2. **Amend / Commit**:
   - For an unmerged branch: `git commit --amend` and force-push to the feature branch.
   - For merged code: Open a corrective documentation pull request: `docs(privacy): sanitize local file paths`.

### Playbook D: Third-Party Dataset Licensing Dispute (SEV-2)
1. **Halt Processing**: Immediately stop all automated pipelines referencing the disputed dataset.
2. **Quarantine Data**: Remove local cached archives from `data/`.
3. **Registry Audit**: Review and update `docs/datasets/DATASET_REGISTER.md`. If licensing terms cannot be reconciled, remove dataset references and revert models to verified synthetic or permissive open-access data.

