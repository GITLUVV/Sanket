"""Unit tests for dataset ingestion, checksum verification, and class filtering."""

import zipfile
from pathlib import Path

import pytest

from sanket.data.ingestion import (
    APPROVED_VOCABULARY_V1,
    IngestionConfig,
    compute_file_sha256,
    extract_archive,
    filter_target_classes,
    ingest_dataset,
    verify_checksum,
    verify_dataset_license,
)


def test_approved_vocabulary_contains_16_classes() -> None:
    """Ensure the frozen V1 vocabulary has exactly 16 target classes."""
    assert len(APPROVED_VOCABULARY_V1) == 16
    assert "HELLO" in APPROVED_VOCABULARY_V1
    assert "HELP" in APPROVED_VOCABULARY_V1
    assert "DOCTOR" in APPROVED_VOCABULARY_V1


def test_verify_dataset_license_approved() -> None:
    """Verify standard open licenses pass verification."""
    assert verify_dataset_license("CC BY 4.0") is True
    assert verify_dataset_license("cc-by-4.0") is True
    assert verify_dataset_license("Creative Commons Attribution 4.0") is True
    assert verify_dataset_license("Apache-2.0") is True


def test_verify_dataset_license_unapproved() -> None:
    """Verify unapproved or restrictive licenses fail verification."""
    assert verify_dataset_license("Proprietary") is False
    assert verify_dataset_license("Unknown") is False
    assert verify_dataset_license("") is False


def test_filter_target_classes() -> None:
    """Ensure candidate class list is partitioned into approved and rejected sets."""
    candidates = ["HELLO", "WATER", "RANDOM_WORD", "FOOD", "ALPHABET_A"]
    accepted, rejected = filter_target_classes(candidates)
    assert "HELLO" in accepted
    assert "WATER" in accepted
    assert "FOOD" in accepted
    assert "RANDOM_WORD" in rejected
    assert "ALPHABET_A" in rejected


def test_compute_file_sha256_and_verify(tmp_path: Path) -> None:
    """Test SHA-256 calculation and checksum matching."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Sanket ISL Dataset Ingestion", encoding="utf-8")

    digest = compute_file_sha256(sample_file)
    assert len(digest) == 64
    assert verify_checksum(sample_file, digest) is True
    assert verify_checksum(sample_file, "wrong_hash_value") is False


def test_extract_archive_with_class_filtering(tmp_path: Path) -> None:
    """Test extracting archive and filtering files strictly for approved classes."""
    zip_path = tmp_path / "test_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("HELLO/clip_01.mp4", b"dummy_video_content")
        zf.writestr("WATER/clip_02.mp4", b"dummy_video_content")
        zf.writestr("UNAPPROVED/clip_03.mp4", b"dummy_video_content")

    out_dir = tmp_path / "extracted"
    extracted = extract_archive(
        archive_path=zip_path,
        output_dir=out_dir,
        target_classes=["HELLO", "WATER"],
    )

    extracted_filenames = [p.name for p in extracted]
    assert "clip_01.mp4" in extracted_filenames
    assert "clip_02.mp4" in extracted_filenames
    assert "clip_03.mp4" not in extracted_filenames


def test_ingest_dataset_dry_run(tmp_path: Path) -> None:
    """Test that dry-run ingestion generates a valid manifest without downloading."""
    out_dir = tmp_path / "raw_include"
    config = IngestionConfig(
        dataset_id="DS-001",
        name="INCLUDE",
        license="CC BY 4.0",
        output_dir=out_dir,
        dry_run=True,
    )

    manifest = ingest_dataset(config)
    assert manifest["dataset_id"] == "DS-001"
    assert manifest["dry_run"] is True
    assert manifest["total_files"] == 16
    assert (out_dir / "ingestion_manifest.json").exists()


def test_ingest_dataset_checksum_mismatch(tmp_path: Path) -> None:
    """Ensure ValueError is raised when archive checksum does not match expected."""
    archive_file = tmp_path / "corrupted_archive.zip"
    with zipfile.ZipFile(archive_file, "w") as zf:
        zf.writestr("HELLO/clip_01.mp4", b"video_data")

    config = IngestionConfig(
        output_dir=tmp_path / "out",
        expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        dry_run=False,
    )

    with pytest.raises(ValueError, match="Checksum verification failed"):
        ingest_dataset(config, archive_path=archive_file)


def test_ingest_dataset_unapproved_license_fails(tmp_path: Path) -> None:
    """Ensure PermissionError is raised when dataset license is unapproved."""
    config = IngestionConfig(license="Commercial-Proprietary", output_dir=tmp_path / "out")
    with pytest.raises(PermissionError, match="not in approved open licenses list"):
        ingest_dataset(config)
