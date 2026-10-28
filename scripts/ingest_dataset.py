#!/usr/bin/env python3
"""CLI utility for dataset ingestion and integrity verification in Sanket.

Usage examples:
    python scripts/ingest_dataset.py --dry-run
    python scripts/ingest_dataset.py --archive-path data/archives/include.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from sanket.data.ingestion import (  # noqa: E402
    IngestionConfig,
    compute_file_sha256,
    ingest_dataset,
    verify_checksum,
    verify_dataset_license,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ingest and verify Indian Sign Language datasets for Sanket."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="DS-001",
        help="Dataset identifier in docs/datasets/DATASET_REGISTER.md",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="INCLUDE",
        help="Dataset name",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="CC BY 4.0",
        help="Dataset license identifier",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Path to local raw archive (.zip or .tar.gz)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Download URL for dataset archive",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 checksum",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/include"),
        help="Destination directory for extracted videos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate ingestion without downloading or extracting raw videos",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify license and archive checksum without extracting",
    )
    return parser


def main() -> int:
    """Execute CLI dataset ingestion workflow."""
    parser = build_parser()
    args = parser.parse_args()

    # 1. License Check
    if not verify_dataset_license(args.license):
        logger.error("License '%s' is not approved. Ingestion aborted.", args.license)
        return 1

    # 2. Checksum Verification Only
    if args.verify_only:
        if not args.archive_path or not args.archive_path.exists():
            logger.error("Archive path required for --verify-only.")
            return 1
        digest = compute_file_sha256(args.archive_path)
        logger.info("File: %s", args.archive_path)
        logger.info("SHA-256: %s", digest)
        if args.expected_sha256:
            matches = verify_checksum(args.archive_path, args.expected_sha256)
            if matches:
                logger.info("Checksum MATCHES expected hash.")
                return 0
            logger.error("Checksum MISMATCH! Expected %s", args.expected_sha256)
            return 2
        return 0

    # 3. Full Ingestion Workflow
    config = IngestionConfig(
        dataset_id=args.dataset_id,
        name=args.name,
        license=args.license,
        archive_url=args.url,
        expected_sha256=args.expected_sha256,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    try:
        manifest = ingest_dataset(config=config, archive_path=args.archive_path)
        logger.info("Ingestion completed successfully.")
        print(json.dumps(manifest, indent=2))
        return 0
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
