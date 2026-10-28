# Dataset Register — Sanket

> [!IMPORTANT]
> **Dataset Licensing Notice**: Never assume the Sanket project license (Apache-2.0) applies to third-party datasets. Every dataset incorporated or used for benchmarking must be formally audited and registered in this document prior to download or ingestion.

---

## Dataset Audit Table

| ID | Dataset Name | License | Modality | Samples / Classes | Status | Date Checked |
| :- | :----------- | :------ | :------- | :---------------- | :----- | :----------- |
| DS-001 | INCLUDE | CC BY 4.0 / Academic | Video (RGB) | 4,287 clips / 263 classes | Candidate (Survey) | 2026-10-28 |
| DS-002 | ISL-CSLTR | Academic Research Use | Video (RGB) | 5,000+ clips / Isolated+Continuous | Candidate (Survey) | 2026-10-28 |
| DS-003 | CISLR | Custom Academic | Video (RGB) | Isolated signs / ~100 classes | Candidate (Survey) | 2026-10-28 |

---

## Detailed Records

### DS-001: INCLUDE Dataset
- **Dataset Name**: INCLUDE (Indian Sign Language Dataset)
- **Source**: Artificial Intelligence Institute of South Carolina / IIT Madras (ACM Multimedia 2020)
- **Purpose**: High-quality benchmark for isolated Indian Sign Language recognition.
- **Vocabulary**: 263 word signs across 15 semantic domains (e.g. Greetings, Actions, Numbers).
- **Number / Type of Samples**: 4,287 video clips recorded by multiple experienced signers in varied unconstrained environments.
- **License**: CC BY 4.0 (attribution required) / Academic Use.
- **Access Requirements**: Publicly downloadable repository / institutional request.
- **Known Limitations**: Variable lighting and resolutions; some dynamic signs require upper-body context beyond hands.
- **Intended Sanket Use**: Candidate source for initial isolated sign subset extraction (Phase 1).
- **Redistribution Restrictions**: Derived feature vectors may be distributed; raw video re-hosting must retain author attribution and original licensing terms.
- **Date Checked**: 2026-10-28

---

### DS-002: ISL-CSLTR Dataset
- **Dataset Name**: ISL-CSLTR (Continuous & Isolated Sign Language Translation/Recognition)
- **Source**: CDAC / IIIT Hyderabad
- **Purpose**: Academic research on continuous and isolated Indian Sign Language sentences.
- **Vocabulary**: 100+ isolated root signs and 500+ continuous grammatical sentences.
- **Number / Type of Samples**: 5,000+ video sequences from 7 native signers in controlled studio conditions.
- **License**: Non-commercial academic research agreement upon request.
- **Access Requirements**: Formal academic data request and signed license agreement.
- **Known Limitations**: Studio lighting is uniform and less representative of real-world noisy camera environments.
- **Intended Sanket Use**: Benchmarking dynamic signs (Phase 3) and sentence formation (Phase 4).
- **Redistribution Restrictions**: Strict prohibition against raw video redistribution.
- **Date Checked**: 2026-10-28

---

## Dataset Registration Template

When registering a new dataset, copy this section:

```markdown
### DS-XXX: [Dataset Name]
- **Dataset Name**:
- **Source**: [Author, Institution, URL, Paper Citation]
- **Purpose**:
- **Vocabulary**:
- **Number / Type of Samples**:
- **License**: [e.g. CC BY-NC 4.0, Custom Academic]
- **Access Requirements**: [Public URL, Form / NDA required]
- **Known Limitations**:
- **Intended Sanket Use**: [e.g. Phase 1 isolated baseline, Phase 3 sequence training]
- **Redistribution Restrictions**: [Strict no redistribution, features-only, etc.]
- **Date Checked**: YYYY-MM-DD
```
