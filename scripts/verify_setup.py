#!/usr/bin/env python3
"""Sanket environment and repository foundation verification script."""

import sys
from pathlib import Path


def main() -> int:
    print("=" * 60)
    print("Sanket Engineering Foundation Verification")
    print("=" * 60)

    # 1. Check Python version
    major, minor = sys.version_info[:2]
    print(f"[1/4] Python Version: {sys.version.split()[0]}", end=" ")
    if (major, minor) < (3, 10):
        print("-> FAILED (Requires >= 3.10)")
        return 1
    print("-> PASSED")

    # 2. Check Directory Structure
    repo_root = Path(__file__).resolve().parent.parent
    required_dirs = [
        "docs",
        "docs/roadmap",
        "docs/architecture",
        "docs/research",
        "docs/datasets",
        "docs/experiments",
        "docs/decisions",
        "docs/development",
        "src/sanket",
        "tests",
        "configs",
        "scripts",
    ]
    print("[2/4] Verifying Directory Layout:")
    all_dirs_ok = True
    for d in required_dirs:
        target = repo_root / d
        exists = target.is_dir()
        status = "OK" if exists else "MISSING"
        if not exists:
            all_dirs_ok = False
        print(f"      - {d:25} [{status}]")

    if not all_dirs_ok:
        print("-> FAILED: One or more required directories missing.")
        return 1
    print("-> PASSED")

    # 3. Check Configuration Loading
    print("[3/4] Testing Configuration Subsystem:", end=" ")
    try:
        sys.path.insert(0, str(repo_root / "src"))
        from sanket.core.config import load_config
        config_file = repo_root / "configs" / "default.yaml"
        cfg = load_config(config_file)
        assert cfg.camera.fps == 30
        assert cfg.landmarks.min_detection_confidence == 0.7
        print(f"-> PASSED (Model: {cfg.model.architecture})")
    except Exception as e:
        print(f"-> FAILED ({e})")
        return 1

    # 4. Check Key Foundation Files
    required_files = [
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        ".gitignore",
        "pyproject.toml",
        "docs/roadmap/SANKET_ROADMAP.md",
        "docs/development/PROJECT_LOG.md",
        "docs/datasets/DATASET_REGISTER.md",
        "docs/decisions/ADR-0001-project-foundations.md",
    ]
    print("[4/4] Verifying Key Governance & Documentation Files:")
    all_files_ok = True
    for f in required_files:
        target = repo_root / f
        exists = target.is_file()
        status = "OK" if exists else "MISSING"
        if not exists:
            all_files_ok = False
        print(f"      - {f:35} [{status}]")

    if not all_files_ok:
        print("-> FAILED: One or more key foundation files are missing.")
        return 1
    print("-> PASSED")

    print("=" * 60)
    print("ALL FOUNDATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
