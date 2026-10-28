# Security & Privacy Policy — Sanket

Sanket is an academic and research-oriented AI project focused on Indian Sign Language (ISL) recognition. Because the project interfaces with computer vision pipelines and will eventually process camera/video streams, **security, privacy, and responsible data governance** are core priorities from day one.

---

## 1. Supported Versions

Security and vulnerability updates are applied to active development releases:

| Version | Supported          | Status             |
| :------ | :----------------- | :----------------- |
| `0.1.x` | :white_check_mark: | Active Development |
| `< 0.1` | :x:                | Unsupported        |

---

## 2. Reporting Security & Privacy Vulnerabilities

**Do NOT report security vulnerabilities, sensitive data leaks, or privacy issues through public GitHub Issues or Pull Requests.**

If you discover a security vulnerability or privacy concern in Sanket, please follow this protocol:

1. **Email the Maintainers**: Send a confidential report to `anveshna2025@gmail.com` with the subject line `[SECURITY] Vulnerability Report - Sanket`.
2. **Include Key Details**:
   - Description of the vulnerability or privacy defect.
   - Clear steps to reproduce the issue (proof-of-concept script, configuration, or environment details).
   - Potential impact on users, video capture feeds, or host systems.
   - Any suggested mitigations or patches.
3. **Encrypted / Secure Channel**: If you require encrypted communication, mention this in your initial message to coordinate PGP keys.

---

## 3. What Information Must NOT Be Publicly Disclosed

To protect researchers, contributors, and sign-language community participants, never publicly post or commit:

- **Camera recordings, participant video footage, or raw sign language recordings** containing identifiable biometric or facial likenesses without verified consent and formal release documentation.
- **Private datasets or copyrighted annotations** with restricted redistribution clauses.
- **API keys, access tokens, cloud credentials, or private configuration files** (`.env`, service account JSONs).
- **Vulnerability exploit details or zero-day demonstration payloads** prior to maintainer review and patching.

---

## 4. Expected Handling & Response Timeline

When a security report is submitted:

1. **Acknowledgment**: The maintainers will acknowledge receipt of the report within **48 hours**.
2. **Assessment & Verification**: The vulnerability will be evaluated, reproduced, and prioritized within **5 business days**.
3. **Remediation & Patching**: A patch or workaround will be developed on a private security branch and validated.
4. **Coordinated Disclosure**: Once a resolution is merged, a security advisory will be published, and appropriate credit will be attributed to the reporter (unless anonymity is requested).

---

## 5. Camera & Biometric Privacy Safeguards

- **No Remote Streaming Without Consent**: Sanket is architected to perform client-side or on-device inference; raw video frames are not transmitted over external networks without explicit user authorization.
- **No Unencrypted Storage of Video Feeds**: Raw camera frames used for landmark extraction should be processed in memory and discarded unless the user has opted into an explicit dataset collection session with documented consent.
- **Synthetic & Normalized Features**: Where possible, downstream recognition models should consume abstract landmark coordinates (normalized geometry) rather than raw facial or bodily imagery to minimize biometric exposure.
