# Frequently Asked Questions (FAQ)

### Q: Why does Sanket use skeletal landmarks instead of raw video pixels?
**A**: MediaPipe landmark extraction isolates the signer's hands and body geometry while abstracting away background clutter, lighting variations, clothing, and private facial pixel data. Landmarks significantly reduce computational complexity, allowing real-time inference on low-power consumer CPUs without requiring expensive GPUs.

### Q: Is Sanket limited to 16 signs forever?
**A**: No. The initial 16-sign vocabulary is a controlled research baseline designed to prove the viability of the normalization and kinematic feature pipeline. The vocabulary expands in subsequent project phases.

### Q: Can Sanket translate full continuous sentences right now?
**A**: No. Phase 1 focuses strictly on isolated signs. Continuous discourse modeling (Phase 3) and grammatical sentence translation (Phase 4) follow sequentially upon empirical validation of the isolated sign baselines.

### Q: How do I initialize the live GitHub Wiki?
**A**: GitHub only activates the Wiki Git repository after the first page is created through the GitHub web interface:
1. Navigate to `https://github.com/GITLUVV/Sanket/wiki`.
2. Click **Create the first page** and save `Home.md`.
3. Once initialized, the Wiki git repository becomes active and can be updated from these drafts.
