"""Signer-Independent dataset partitioning and split manifest generation.

Fulfills SANKET-022 by providing strict Leave-Signers-Out (LSO) train/val/test
splitting, ensuring zero biometric data leakage across evaluation partitions.
"""

from __future__ import annotations

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sanket.data.ingestion import APPROVED_VOCABULARY_V1
from sanket.data.inspection import parse_signer_from_filename

logger = logging.getLogger("sanket.data.splits")


class SampleRecord(BaseModel):
    """Metadata record for a single sign video sample."""

    sample_id: str
    class_name: str
    signer_id: str
    relative_path: str


class PartitionStats(BaseModel):
    """Aggregate statistics for a single split partition."""

    partition_name: str
    total_samples: int
    signers: list[str] = Field(default_factory=list)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    sample_ids: list[str] = Field(default_factory=list)


class SplitManifest(BaseModel):
    """Deterministic, immutable manifest documenting dataset split partitions."""

    dataset_name: str = "INCLUDE-16"
    created_at_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    random_seed: int = 42
    target_classes: list[str] = Field(default_factory=lambda: list(APPROVED_VOCABULARY_V1))
    signer_allocation: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from signer_id to partition name (train, val, test)",
    )
    partitions: dict[str, PartitionStats] = Field(default_factory=dict)
    is_signer_disjoint: bool = True
    samples: list[SampleRecord] = Field(default_factory=list)


def verify_signer_disjointness(manifest: SplitManifest) -> bool:
    """Verify mathematically that no signer appears in multiple partitions.

    Args:
        manifest: SplitManifest instance.

    Returns:
        True if all signer sets are pairwise disjoint, False otherwise.
    """
    partition_signers: dict[str, set[str]] = {}
    for part_name, part_stats in manifest.partitions.items():
        partition_signers[part_name] = set(part_stats.signers)

    partition_names = list(partition_signers.keys())
    for i in range(len(partition_names)):
        for j in range(i + 1, len(partition_names)):
            name_a, name_b = partition_names[i], partition_names[j]
            overlap = partition_signers[name_a].intersection(partition_signers[name_b])
            if overlap:
                logger.error(
                    "Signer leakage detected between '%s' and '%s': %s",
                    name_a,
                    name_b,
                    overlap,
                )
                return False

    return True


def create_signer_independent_split(
    samples: list[SampleRecord],
    dataset_name: str = "INCLUDE-16",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    target_classes: list[str] | None = None,
) -> SplitManifest:
    """Partition samples into train/val/test using strict Leave-Signers-Out grouping.

    Args:
        samples: List of all dataset SampleRecord items.
        dataset_name: Identifier for dataset.
        train_ratio: Target proportion of signers for training (~0.70).
        val_ratio: Target proportion of signers for validation (~0.15).
        test_ratio: Target proportion of signers for testing (~0.15).
        random_seed: Deterministic random seed for shuffling.
        target_classes: Active class glosses.

    Returns:
        Validated SplitManifest.
    """
    classes = target_classes or APPROVED_VOCABULARY_V1
    unique_signers = sorted({s.signer_id for s in samples if s.signer_id})

    if len(unique_signers) < 3:
        msg = f"Signer-Independent split requires >= 3 signers, found {len(unique_signers)}"
        raise ValueError(msg)

    # Deterministic signer shuffling
    rng = random.Random(random_seed)
    shuffled_signers = list(unique_signers)
    rng.shuffle(shuffled_signers)

    n_signers = len(shuffled_signers)
    # Guarantee at least 1 signer in val and test
    n_test = max(1, round(n_signers * test_ratio))
    n_val = max(1, round(n_signers * val_ratio))
    n_train = n_signers - (n_test + n_val)

    if n_train <= 0:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train_signers = sorted(shuffled_signers[:n_train])
    val_signers = sorted(shuffled_signers[n_train: n_train + n_val])
    test_signers = sorted(shuffled_signers[n_train + n_val:])

    signer_allocation: dict[str, str] = {}
    for s in train_signers:
        signer_allocation[s] = "train"
    for s in val_signers:
        signer_allocation[s] = "val"
    for s in test_signers:
        signer_allocation[s] = "test"

    partition_samples: dict[str, list[SampleRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for sample in samples:
        part = signer_allocation.get(sample.signer_id)
        if part in partition_samples:
            partition_samples[part].append(sample)

    partitions: dict[str, PartitionStats] = {}
    for part_name, part_list in partition_samples.items():
        class_dist: dict[str, int] = {}
        for s in part_list:
            class_dist[s.class_name] = class_dist.get(s.class_name, 0) + 1

        signers_in_part = sorted({s.signer_id for s in part_list})
        partitions[part_name] = PartitionStats(
            partition_name=part_name,
            total_samples=len(part_list),
            signers=signers_in_part,
            class_distribution=class_dist,
            sample_ids=[s.sample_id for s in part_list],
        )

    manifest = SplitManifest(
        dataset_name=dataset_name,
        random_seed=random_seed,
        target_classes=classes,
        signer_allocation=signer_allocation,
        partitions=partitions,
        is_signer_disjoint=True,
        samples=samples,
    )

    if not verify_signer_disjointness(manifest):
        msg = "Fatal error: Signer disjointness assertion failed during split creation."
        raise AssertionError(msg)

    return manifest


def collect_samples_from_directory(
    dataset_dir: Path,
    target_classes: list[str] | None = None,
) -> list[SampleRecord]:
    """Scan dataset directory and parse samples with class and signer identities.

    Args:
        dataset_dir: Path to directory containing class subfolders.
        target_classes: Optional list of approved classes.

    Returns:
        List of SampleRecord instances.
    """
    dataset_dir = Path(dataset_dir)
    approved = set(target_classes or APPROVED_VOCABULARY_V1)
    samples: list[SampleRecord] = []

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for class_folder in sorted(dataset_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name.upper()
        if class_name not in approved:
            continue

        for video_file in sorted(class_folder.iterdir()):
            if video_file.suffix.lower() not in video_exts:
                continue

            signer_id = parse_signer_from_filename(video_file.name) or "Signer01"
            sample_id = f"{class_name}_{video_file.stem}"
            rel_path = str(video_file.relative_to(dataset_dir)).replace("\\", "/")

            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    class_name=class_name,
                    signer_id=signer_id,
                    relative_path=rel_path,
                )
            )

    return samples


def generate_default_include16_split(random_seed: int = 42) -> SplitManifest:
    """Generate the canonical baseline 7-signer split manifest for INCLUDE-16.

    Simulates the standardized 7 native signers (S1–S7) with 32 clips per class,
    allocating S1–S5 (71%) to train, S6 (14%) to val, and S7 (14%) to test.

    Args:
        random_seed: Seed for deterministic manifest metadata.

    Returns:
        Canonical SplitManifest instance.
    """
    samples: list[SampleRecord] = []
    signers = [f"Signer0{i}" for i in range(1, 8)]

    for cls in APPROVED_VOCABULARY_V1:
        for idx in range(1, 33):
            # Distribute 32 clips evenly across 7 signers
            signer = signers[(idx - 1) % len(signers)]
            sample_id = f"{cls}_{signer}_{idx:02d}"
            rel_path = f"{cls}/{signer}_{cls}_{idx:02d}.mp4"
            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    class_name=cls,
                    signer_id=signer,
                    relative_path=rel_path,
                )
            )

    return create_signer_independent_split(
        samples=samples,
        dataset_name="INCLUDE-16",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
    )


def save_split_manifest(manifest: SplitManifest, output_path: Path) -> Path:
    """Save split manifest to JSON file.

    Args:
        manifest: SplitManifest instance.
        output_path: Path to target JSON file.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Split manifest saved to %s", output_path)
    return output_path


def load_split_manifest(input_path: Path) -> SplitManifest:
    """Load split manifest from JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        Validated SplitManifest instance.
    """
    content = Path(input_path).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(content)
    return SplitManifest(**data)
