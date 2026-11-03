# SANKET (संकेत) — Troubleshooting & Diagnostic Guide

> **Diagnostic Procedures, Known Issues, and Actionable Technical Remedies**  
> *Quick reference guide for developers and system operators encountering runtime issues.*

---

## 1. Vision & Hardware Acquisition Issues

### 1.1 Camera Device Not Found (`CameraCaptureError: Device 0 not available`)
- **Root Cause**: The default camera index `0` is inaccessible, occupied by another application (e.g., Zoom/Teams), or camera permissions are disabled.
- **Remedies**:
  1. Test alternate camera device indices:
     ```bash
     python scripts/run_realtime_mvp.py --camera-id 1
     ```
  2. On Windows: Open *Settings $\rightarrow$ Privacy & Security $\rightarrow$ Camera* and verify "Let desktop apps access your camera" is enabled.
  3. On Linux: Verify permissions for `/dev/video0`: `sudo usermod -aG video $USER`.
  4. In Headless / CI environments: Run with synthetic mock video:
     ```bash
     python scripts/run_realtime_mvp.py --mock-video --headless
     ```

### 1.2 OpenCV GUI / Headless Display Crash (`cv2.error: The function is not implemented`)
- **Root Cause**: Attempting to execute `cv2.imshow()` in a headless Linux server or Docker container without an X11 display server.
- **Remedies**:
  - Pass the `--headless` flag to disable visual GUI rendering:
    ```bash
    python scripts/run_realtime_mvp.py --headless
    ```
  - For Docker / CI environments, install `opencv-python-headless` or run in headless mode.

### 1.3 MediaPipe C++ DLL / Library Import Failures
- **Root Cause**: Missing runtime C++ dependencies on Windows or missing Mesa/GLIB libraries on Linux.
- **Remedies**:
  - On Windows: Install the latest *Microsoft Visual C++ Redistributable (x64)*.
  - On Linux / Docker: Install graphics libraries:
    ```bash
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    ```

---

## 2. Model Training & Optimization Warnings

### 2.1 MLP Convergence Warning (`ConvergenceWarning: Maximum iterations reached`)
- **Root Cause**: The stochastic gradient descent optimizer reached `max_iter` before parameter convergence on small synthetic datasets.
- **Remedies**:
  - Increase iterations: `python scripts/train_mlp.py --max-iter 300`.
  - Alternatively, adjust learning rate: `python scripts/train_mlp.py --learning-rate 0.002`.

### 2.2 Incompatible Feature Dimension Error (`IncompatibleFeatureShapeError`)
- **Root Cause**: Model was trained on a feature space different from the current 645-dimensional pipeline.
- **Remedies**:
  - Re-extract features using the latest `FeatureExtractor`:
    ```bash
    python scripts/generate_features.py --landmarks-dir data/landmarks --output-features data/features/features.npz
    ```

---

## 3. Code Quality & Testing Failures

### 3.1 Ruff Linter Failures (`ruff check .`)
- **Remedies**:
  - Automatically fix linting and import ordering:
    ```bash
    ruff check --fix .
    ```
  - Format source code:
    ```bash
    black src tests scripts
    ```

### 3.2 Verification Script Failure (`python scripts/verify_setup.py`)
- **Remedies**:
  - If package import fails, ensure the package is installed in editable mode:
    ```bash
    pip install -e ".[dev]"
    ```

