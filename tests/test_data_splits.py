"""Unit tests for Signer-Independent dataset splitting and manifest generation."""

from pathlib import Path

import pytest

from sanket.data.splits import (
    PartitionStats,
    SampleRecord,
    SplitManifest,
    collect_samples_from_directory,
    create_signer_independent_split,
    generate_default_include16_split,
    load_split_manifest,
    save_split_manifest,
    verify_signer_disjointness,
)


def test_verify_signer_disjointness_valid() -> None:
    """Ensure disjoint signer sets validate successfully."""
    manifest = SplitManifest(
        dataset_name="TEST",
        partitions={
            "train": PartitionStats(
                partition_name="train",
                total_samples=10,
                signers=["Signer01", "Signer02"],
            ),
            "val": PartitionStats(
                partition_name="val",
                total_samples=5,
                signers=["Signer03"],
            ),
            "test": PartitionStats(
                partition_name="test",
                total_samples=5,
                signers=["Signer04"],
            ),
        },
    )
    assert verify_signer_disjointness(manifest) is True


def test_verify_signer_disjointness_leakage_detected() -> None:
    """Ensure overlapping signers fail verification."""
    manifest = SplitManifest(
        dataset_name="TEST",
        partitions={
            "train": PartitionStats(
                partition_name="train",
                total_samples=10,
                signers=["Signer01", "Signer02"],
            ),
            "test": PartitionStats(
                partition_name="test",
                total_samples=5,
                signers=["Signer02", "Signer03"],  # Signer02 leaked into test!
            ),
        },
    )
    assert verify_signer_disjointness(manifest) is False


def test_create_signer_independent_split_insufficient_signers() -> None:
    """Ensure ValueError is raised when fewer than 3 signers are provided."""
    samples = [
        SampleRecord(
            sample_id="1", class_name="HELLO", signer_id="S1", relative_path="clip1.mp4"
        ),
        SampleRecord(
            sample_id="2", class_name="HELLO", signer_id="S2", relative_path="clip2.mp4"
        ),
    ]
    with pytest.raises(ValueError, match="requires >= 3 signers"):
        create_signer_independent_split(samples)


def test_create_signer_independent_split_disjointness() -> None:
    """Ensure create_signer_independent_split guarantees zero identity overlap."""
    samples = []
    signers = [f"Signer0{i}" for i in range(1, 9)]  # 8 signers
    classes = ["HELLO", "WATER", "FOOD", "HELP"]

    for signer in signers:
        for cls in classes:
            samples.append(
                SampleRecord(
                    sample_id=f"{cls}_{signer}",
                    class_name=cls,
                    signer_id=signer,
                    relative_path=f"{cls}/{signer}_{cls}.mp4",
                )
            )

    manifest = create_signer_independent_split(
        samples=samples,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    assert manifest.is_signer_disjoint is True
    assert verify_signer_disjointness(manifest) is True

    train_s = set(manifest.partitions["train"].signers)
    val_s = set(manifest.partitions["val"].signers)
    test_s = set(manifest.partitions["test"].signers)

    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)
    assert len(train_s) + len(val_s) + len(test_s) == 8


def test_split_determinism_with_seed() -> None:
    """Ensure identical random seeds yield identical signer allocations."""
    samples = [
        SampleRecord(
            sample_id=f"sample_{i}",
            class_name="HELLO",
            signer_id=f"Signer_{i % 5}",
            relative_path=f"clip_{i}.mp4",
        )
        for i in range(20)
    ]

    manifest1 = create_signer_independent_split(samples, random_seed=123)
    manifest2 = create_signer_independent_split(samples, random_seed=123)
    manifest3 = create_signer_independent_split(samples, random_seed=999)

    assert manifest1.signer_allocation == manifest2.signer_allocation
    # Different seed should yield different allocation
    assert manifest1.signer_allocation != manifest3.signer_allocation


def test_collect_samples_from_directory(tmp_path: Path) -> None:
    """Test discovering and parsing samples from a directory of class folders."""
    hello_dir = tmp_path / "HELLO"
    hello_dir.mkdir(parents=True)
    (hello_dir / "Signer01_HELLO_01.mp4").write_bytes(b"dummy")
    (hello_dir / "Signer02_HELLO_02.mp4").write_bytes(b"dummy")

    water_dir = tmp_path / "WATER"
    water_dir.mkdir(parents=True)
    (water_dir / "Signer01_WATER_01.mp4").write_bytes(b"dummy")

    samples = collect_samples_from_directory(
        dataset_dir=tmp_path, target_classes=["HELLO", "WATER"]
    )
    assert len(samples) == 3
    sample_ids = [s.sample_id for s in samples]
    assert "HELLO_Signer01_HELLO_01" in sample_ids
    assert "WATER_Signer01_WATER_01" in sample_ids


def test_save_and_load_split_manifest(tmp_path: Path) -> None:
    """Test JSON manifest round-trip persistence."""
    manifest = generate_default_include16_split(random_seed=42)
    target_path = tmp_path / "manifest.json"

    saved_path = save_split_manifest(manifest, target_path)
    assert saved_path.exists()

    loaded = load_split_manifest(saved_path)
    assert loaded.dataset_name == manifest.dataset_name
    assert loaded.random_seed == manifest.random_seed
    assert loaded.signer_allocation == manifest.signer_allocation
    assert verify_signer_disjointness(loaded) is True


def test_canonical_default_include16_split() -> None:
    """Test generating canonical 7-signer baseline split for INCLUDE-16."""
    manifest = generate_default_include16_split(random_seed=42)
    assert manifest.dataset_name == "INCLUDE-16"
    assert len(manifest.target_classes) == 16
    assert len(manifest.samples) == 512  # 16 classes * 32 clips
    assert verify_signer_disjointness(manifest) is True

    # Check that all 16 classes are represented
    all_classes_in_samples = {s.class_name for s in manifest.samples}
    assert len(all_classes_in_samples) == 16
