"""Data ingestion, processing, inspection, and splitting subsystem for Sanket."""

from sanket.data.ingestion import (
    APPROVED_VOCABULARY_V1,
    IngestionConfig,
    compute_file_sha256,
    ingest_dataset,
    verify_checksum,
    verify_dataset_license,
)
from sanket.data.inspection import (
    ClassDistributionStats,
    DatasetInspectionSummary,
    VideoMetadata,
    compute_inspection_summary,
    generate_markdown_report,
    inspect_dataset_directory,
    inspect_video_file,
    parse_signer_from_filename,
    save_inspection_report,
)
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

__all__ = [
    "APPROVED_VOCABULARY_V1",
    "ClassDistributionStats",
    "DatasetInspectionSummary",
    "IngestionConfig",
    "PartitionStats",
    "SampleRecord",
    "SplitManifest",
    "VideoMetadata",
    "collect_samples_from_directory",
    "compute_file_sha256",
    "compute_inspection_summary",
    "create_signer_independent_split",
    "generate_default_include16_split",
    "generate_markdown_report",
    "ingest_dataset",
    "inspect_dataset_directory",
    "inspect_video_file",
    "load_split_manifest",
    "parse_signer_from_filename",
    "save_inspection_report",
    "save_split_manifest",
    "verify_checksum",
    "verify_dataset_license",
    "verify_signer_disjointness",
]
