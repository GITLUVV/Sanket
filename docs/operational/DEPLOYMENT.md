# SANKET (संकेत) — Deployment & Packaging Guide

> **Packaging, Local Workstation Installation, Containerization, and Edge Distribution**  
> *Authoritative guide for packaging and executing SANKET across diverse runtime environments.*

---

## 1. Local Workstation Deployment (Recommended)

SANKET is distributed as a standard Python package following PEP 517 / PEP 621 specifications (`pyproject.toml`).

### 1.1 Direct Package Installation
```bash
# Clone the repository
git clone https://github.com/GITLUVV/Sanket.git
cd Sanket

# Create clean virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install in standard production mode
pip install --upgrade pip
pip install .
```

### 1.2 Running the Real-Time Application
```bash
# Launch real-time camera recognition (default camera 0, Random Forest baseline)
python scripts/run_realtime_mvp.py --camera-id 0 --mirror

# Launch in headless test mode with mock video input
python scripts/run_realtime_mvp.py --mock-video --headless
```

---

## 2. Docker Container Deployment (Headless / Edge)

For headless benchmarking, continuous integration, or edge container environments:

### 2.1 Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY models/ ./models/

RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/verify_setup.py"]
```

### 2.2 Building and Running the Container
```bash
docker build -t sanket:latest .
docker run --rm sanket:latest python scripts/verify_setup.py
```

---

## 3. Edge Workstation Specifications

- **Target Architecture**: x86_64 or ARM64 (Raspberry Pi 5, Apple Silicon M-series, Intel NUC).
- **GPU Acceleration**: None required. All inference is optimized for CPU multi-threading.
- **Portability**: Self-contained with no network calls required during live execution.

