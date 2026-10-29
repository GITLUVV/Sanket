#!/usr/bin/env python3
"""CLI utility for generating and verifying Signer-Independent dataset splits in Sanket.

Usage examples:
    python scripts/create_splits.py --generate-default
    python scripts/create_splits.py --dataset-dir data/raw/include
    python scripts/create_splits.py --verify-manifest configs/splits/manifest.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from sanket.data.splits import (  # noqa: E402
    collect_samples_from_directory,
    create_signer_independent_split,
    generate_default_include16_split,
    load_split_manifest,
    save_split_manifest,
    verify_signer_disjointness,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("split_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Create and verify Signer-Independent dataset partitions for Sanket."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Root directory of raw video files containing class folders",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("configs/splits/include16_signer_independent.json"),
        help="Path where the output split manifest JSON will be written",
    )
    parser.add_argument(
        "--generate-default",
        action="store_true",
        help="Generate the canonical 7-signer baseline split for INCLUDE-16",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="Path to an existing split manifest to verify for signer disjointness",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for signer shuffling",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Target proportion of signers allocated to training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target proportion of signers allocated to validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Target proportion of signers allocated to testing",
    )
    return parser


def main() -> int:
    """Execute CLI split generation or verification workflow."""
    parser = build_parser()
    args = parser.parse_args()

    # 1. Verification-only mode
    if args.verify_manifest:
        if not args.verify_manifest.exists():
            logger.error("Manifest '%s' does not exist.", args.verify_manifest)
            return 1
        manifest = load_split_manifest(args.verify_manifest)
        is_disjoint = verify_signer_disjointness(manifest)
        if is_disjoint:
            logger.info("Manifest verified: Strict signer disjointness confirmed.")
            for name, part in manifest.partitions.items():
                logger.info(
                    "  Partition '%s': %d samples, %d signers %s",
                    name,
                    part.total_samples,
                    len(part.signers),
                    part.signers,
                )
            return 0
        logger.error("Manifest FAILED verification: Signer overlap detected!")
        return 1

    # 2. Canonical default split mode
    if args.generate_default or not args.dataset_dir:
        logger.info("Generating canonical baseline split for INCLUDE-16...")
        manifest = generate_default_include16_split(random_seed=args.seed)
        save_split_manifest(manifest, args.output_manifest)
        logger.info("Saved default split manifest to %s", args.output_manifest)
        return 0

    # 3. Directory scanning mode
    if not args.dataset_dir.exists():
        logger.error("Dataset directory '%s' does not exist.", args.dataset_dir)
        return 1

    samples = collect_samples_from_directory(args.dataset_dir)
    if not samples:
        logger.error("No valid video samples found in '%s'.", args.dataset_dir)
        return 1

    logger.info("Collected %d samples across directory.", len(samples))
    manifest = create_signer_independent_split(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    save_split_manifest(manifest, args.output_manifest)
    logger.info("Saved split manifest to %s", args.output_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
