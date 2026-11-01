# Deployment & Runtime Execution

## Local Execution
Sanket supports local CLI execution across macOS, Linux, and Windows:
```bash
# Verify environment and foundation checks
python scripts/verify_setup.py

# Run unit test suite
pytest

# Extract features from normalized landmark directory
python scripts/generate_features.py --help

# Train baseline classifier
python scripts/train_random_forest.py --synthetic-demo
```

## System Requirements
- Python 3.10, 3.11, or 3.12
- OpenCV and MediaPipe system dependencies (Linux: `libegl1`, `libgl1`, `libgles2`)
- 4GB+ RAM, multi-core CPU (no dedicated GPU required for Phase 1 inference)
