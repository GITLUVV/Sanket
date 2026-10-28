"""Dataset ingestion pipeline and integrity verification for Sanket.

Fulfills SANKET-020 by providing automated archive verification, SHA-256 integrity
checks, license gate validation, target class filtering, and manifest generation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tarfile
from typing import Any
import urllib.request
import zipfile

from pydantic import BaseModel, Field

logger = logging.getLogger("sanket.data.ingestion")

APPROVED_VOCABULARY_V1: list[str] = [
    "HELLO",
    "THANK_YOU",
    "PLEASE",
    "YES",
    "NO",
    "HELP",
    "WATER",
    "FOOD",
    "DOCTOR",
    "TIME",
    "WHERE",
    "STOP",
    "NAME",
    "GOOD",
    "BAD",
    "ME",
]

APPROVED_LICENSES: list[str] = [
    "CC BY 4.0",
    "CC-BY-4.0",
    "Creative Commons Attribution 4.0",
    "Creative Commons Attribution 4.0 International",
    "Apache-2.0",
    "MIT",
]


class IngestionConfig(BaseModel):
    """Configuration schema for dataset ingestion."""

    dataset_id: str = Field(default="DS-001", description="Identifier matching DATASET_REGISTER.md")
    name: str = Field(default="INCLUDE", description="Dataset name")
    license: str = Field(default="CC BY 4.0", description="Dataset licensing terms")
    archive_url: str | None = Field(default=None, description="Download URL if available")
    expected_sha256: str | None = Field(default=None, description="Expected SHA-256 checksum")
    output_dir: Path = Field(
        default_factory=lambda: Path("data/raw/include"),
        description="Local staging destination",
    )
    target_classes: list[str] = Field(
        default_factory=lambda: list(APPROVED_VOCABULARY_V1),
        description="List of class glosses to ingest",
    )
    dry_run: bool = Field(default=False, description="Simulate ingestion without downloading files")


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hexadecimal digest of a file.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte chunk size for streaming computation.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    """Verify whether a file's SHA-256 matches an expected digest.

    Args:
        file_path: Path to the file.
        expected_hash: Expected hex string.

    Returns:
        True if checksums match, False otherwise.
    """
    if not expected_hash:
        return True
    actual_hash = compute_file_sha256(file_path)
    return actual_hash.lower() == expected_hash.strip().lower()


def verify_dataset_license(license_name: str) -> bool:
    """Validate whether the dataset license is among the approved open licenses.

    Args:
        license_name: License identifier to test.

    Returns:
        True if license is approved for ingestion, False otherwise.
    """
    normalized = license_name.strip().lower()
    return any(normalized == approved.lower() for approved in APPROVED_LICENSES)


def filter_target_classes(
    candidate_classes: list[str],
    approved_vocabulary: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Filter class list against approved vocabulary.

    Args:
        candidate_classes: Candidate class glosses from dataset.
        approved_vocabulary: List of approved classes (defaults to APPROVED_VOCABULARY_V1).

    Returns:
        Tuple of (accepted_classes, rejected_classes).
    """
    approved = set(approved_vocabulary or APPROVED_VOCABULARY_V1)
    accepted: list[str] = []
    rejected: list[str] = []

    for cls in candidate_classes:
        normalized = cls.strip().upper()
        if normalized in approved:
            accepted.append(normalized)
        else:
            rejected.append(cls)

    return accepted, rejected


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    target_classes: list[str] | None = None,
) -> list[Path]:
    """Extract an archive file to output_dir, filtering for target classes.

    Args:
        archive_path: Path to zip or tar archive.
        output_dir: Directory where files should be extracted.
        target_classes: Optional list of class names to filter.

    Returns:
        List of paths to extracted files.
    """
    extracted_paths: list[Path] = []
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_set = {c.upper() for c in target_classes} if target_classes else None

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Determine class name from path structure
                parts = Path(member.filename).parts
                class_match = any(p.upper() in target_set for p in parts) if target_set else True
                if class_match:
                    zf.extract(member, output_dir)
                    extracted_paths.append(output_dir / member.filename)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                parts = Path(member.name).parts
                class_match = any(p.upper() in target_set for p in parts) if target_set else True
                if class_match:
                    tf.extract(member, output_dir)
                    extracted_paths.append(output_dir / member.name)
    else:
        msg = f"Unsupported archive format: {archive_path.name}"
        raise ValueError(msg)

    return extracted_paths


def download_file(url: str, destination: Path) -> Path:
    """Download a remote file using streaming urllib.

    Args:
        url: Remote URL.
        destination: Local path to save file.

    Returns:
        Path to downloaded file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return destination


def generate_ingestion_manifest(
    config: IngestionConfig,
    extracted_files: list[Path],
    archive_hash: str | None = None,
) -> dict[str, Any]:
    """Generate structured manifest of the ingestion run.

    Args:
        config: IngestionConfig used for the run.
        extracted_files: List of extracted or staged file paths.
        archive_hash: Optional SHA-256 hash of the archive.

    Returns:
        Dictionary representation of the manifest.
    """
    classes_found: set[str] = set()
    total_bytes = 0
    file_records: list[dict[str, Any]] = []

    for f in extracted_files:
        p = Path(f)
        if p.exists() and p.is_file():
            size = p.stat().st_size
            total_bytes += size
            # Infer class label from parent folder
            class_name = p.parent.name.upper()
            classes_found.add(class_name)
            file_records.append({
                "filename": p.name,
                "class": class_name,
                "bytes": size,
            })

    return {
        "dataset_id": config.dataset_id,
        "dataset_name": config.name,
        "license": config.license,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dry_run": config.dry_run,
        "archive_sha256": archive_hash,
        "total_files": len(extracted_files),
        "total_bytes": total_bytes,
        "unique_classes_count": len(classes_found),
        "classes": sorted(classes_found),
        "sample_files": file_records[:20],
    }


def ingest_dataset(
    config: IngestionConfig,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end dataset ingestion workflow.

    Args:
        config: Ingestion configuration parameters.
        archive_path: Optional path to a local archive (if not downloading).

    Returns:
        Structured ingestion manifest.
    """
    # 1. License Gate Validation
    if not verify_dataset_license(config.license):
        msg = f"License '{config.license}' is not in approved open licenses list."
        raise PermissionError(msg)

    # 2. Dry Run Mode
    if config.dry_run:
        logger.info("Executing dry-run ingestion for %s", config.name)
        simulated_files = [
            config.output_dir / cls / f"{cls.lower()}_sample_01.mp4" for cls in config.target_classes
        ]
        manifest = generate_ingestion_manifest(config, simulated_files, config.expected_sha256)
        manifest_path = config.output_dir / "ingestion_manifest.json"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    # 3. Obtain Archive
    archive_file: Path
    if archive_path and Path(archive_path).exists():
        archive_file = Path(archive_path)
    elif config.archive_url:
        archive_file = config.output_dir / "raw_archive.zip"
        logger.info("Downloading dataset archive from %s...", config.archive_url)
        download_file(config.archive_url, archive_file)
    else:
        msg = "No local archive provided and no archive_url configured."
        raise FileNotFoundError(msg)

    # 4. Checksum Verification
    computed_hash = compute_file_sha256(archive_file)
    if config.expected_sha256 and not verify_checksum(archive_file, config.expected_sha256):
        msg = (
            f"Checksum verification failed for {archive_file.name}. "
            f"Expected: {config.expected_sha256}, Got: {computed_hash}"
        )
        raise ValueError(msg)

    # 5. Extract and Filter for Target Classes
    extracted = extract_archive(
        archive_path=archive_file,
        output_dir=config.output_dir,
        target_classes=config.target_classes,
    )

    # 6. Generate and Save Manifest
    manifest = generate_ingestion_manifest(config, extracted, computed_hash)
    manifest_path = config.output_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Ingestion complete. Manifest saved to %s", manifest_path)

    return manifest
