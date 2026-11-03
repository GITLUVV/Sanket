# SANKET (संकेत) — Environment & Runtime Specifications

> **System Prerequisites, Hardware Baselines, OS Matrix, and Environment Variables**  
> *Authoritative specifications for host configurations and operational environments.*

---

## 1. Operating System Compatibility Matrix

| Operating System | Architecture | Supported | Validation Status |
| :--- | :--- | :---: | :--- |
| **Windows 11 / 10** | x86_64 | **Yes** | Fully verified on Python 3.12.10 local workstation. |
| **Ubuntu 22.04 LTS** | x86_64 | **Yes** | Fully verified on GitHub Actions CI runner (`ubuntu-latest`). |
| **Ubuntu 20.04 LTS** | x86_64 | **Yes** | Verified with standard GL/GLIB libraries installed. |
| **macOS (Ventura / Sonoma)** | ARM64 / x86_64 | **Yes** | Verified with Apple Silicon native Python 3.11+. |
| **Debian 11 / 12 (Raspberry Pi OS)** | ARM64 | **Yes** | Supported on Raspberry Pi 4/5 (4 GB+ RAM). |

---

## 2. Hardware Resource Requirements

### Minimum Requirements (Evaluation & CLI Mode)
- **CPU**: Dual-core x86_64 or ARM64 @ 1.8 GHz.
- **RAM**: 2 GB available system memory.
- **Disk Storage**: 500 MB free space (codebase, models, and dependencies).
- **Video Input**: Synthetic mock video or video files on disk.

### Recommended Requirements (Live Camera MVP Mode)
- **CPU**: Quad-core x86_64 or ARM64 @ 2.4 GHz+ (Intel Core i5/i7, AMD Ryzen 5/7, Apple M1/M2/M3).
- **RAM**: 4 GB+ system memory.
- **Disk Storage**: 1 GB free space.
- **Camera**: Standard UVC USB or integrated RGB webcam supporting $640 \times 480$ resolution at 30 FPS.

---

## 3. Environment Variables Configuration

SANKET supports optional configuration overrides via environment variables:

```bash
# SANKET Environment Configuration (.env.example)

# Path to custom YAML configuration file (defaults to configs/default.yaml)
SANKET_CONFIG_PATH="configs/default.yaml"

# Default video capture camera device ID (0, 1, 2)
SANKET_CAMERA_ID=0

# Log level for application logging (DEBUG, INFO, WARNING, ERROR)
SANKET_LOG_LEVEL="INFO"

# Path to serialized model weights artifact
SANKET_MODEL_WEIGHTS="models/baseline_random_forest.joblib"

# Headless CI testing mode toggle (true/false)
SANKET_HEADLESS="false"
```

To use custom environment variables, copy `.env.example` to `.env` (which is excluded from Git via `.gitignore`).

