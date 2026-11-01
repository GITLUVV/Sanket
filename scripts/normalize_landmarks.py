#!/usr/bin/env python3
"""CLI utility for skeletal hand landmark coordinate normalization (SANKET-026).

Applies wrist-centering, palm-scale Euclidean normalization, and optional in-plane
rotational alignment to extracted landmark archives (.npz).

Usage examples:
    python scripts/normalize_landmarks.py --input sample_landmarks.npz --output-dir data/normalized
    python scripts/normalize_landmarks.py --verify-invariance
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

import numpy as np  # noqa: E402

from sanket.vision.landmarks import (  # noqa: E402
    load_landmarks_npz,
    save_landmarks_npz,
)
from sanket.vision.normalization import (  # noqa: E402
    LandmarkNormalizer,
    normalize_hand_landmarks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("normalize_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Normalize skeletal hand landmarks for translation and scale invariance."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=Path,
        help="Path to single .npz landmark archive to normalize",
    )
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing .npz landmark archives to process in batch",
    )
    group.add_argument(
        "--verify-invariance",
        action="store_true",
        help="Execute synthetic translation and scaling perturbation invariance verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory to save normalized .npz archives",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="palm",
        choices=["palm", "max_span", "bbox"],
        help="Scale normalization method (default: palm)",
    )
    parser.add_argument(
        "--align-rotation",
        action="store_true",
        help="Apply 2D in-plane rotational alignment",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    return parser


def run_invariance_verification() -> bool:
    """Demonstrate numerical invariance across synthetic spatial perturbations."""
    logger.info("Executing synthetic landmark perturbation invariance verification...")

    # Generate synthetic 21-point hand coordinates
    np.random.seed(42)
    base_hand = np.random.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)

    # Base normalized landmarks
    base_norm, _ = normalize_hand_landmarks(base_hand, method="palm", align_rotation=False)

    # 1. Test translation perturbation: shift by (12.5, -8.3, 4.1)
    shift = np.array([12.5, -8.3, 4.1], dtype=np.float32)
    translated_hand = base_hand + shift
    trans_norm, _ = normalize_hand_landmarks(translated_hand, method="palm", align_rotation=False)
    trans_err = float(np.max(np.abs(trans_norm - base_norm)))

    # 2. Test scale perturbation: scale by 4.25x
    scaled_hand = base_hand * 4.25
    scale_norm, _ = normalize_hand_landmarks(scaled_hand, method="palm", align_rotation=False)
    scale_err = float(np.max(np.abs(scale_norm - base_norm)))

    # 3. Test combined translation + scale perturbation
    combined_hand = (base_hand * 0.35) + shift
    comb_norm, _ = normalize_hand_landmarks(combined_hand, method="palm", align_rotation=False)
    comb_err = float(np.max(np.abs(comb_norm - base_norm)))

    logger.info("--- Invariance Perturbation Results ---")
    logger.info("  * Translation Invariance Max Error: %.2e", trans_err)
    logger.info("  * Scaling Invariance Max Error:     %.2e", scale_err)
    logger.info("  * Combined Invariance Max Error:    %.2e", comb_err)

    tolerance = 1e-5
    passed = trans_err < tolerance and scale_err < tolerance and comb_err < tolerance
    if passed:
        logger.info("SUCCESS: All coordinate invariance tests PASSED within 1e-5 tolerance.")
    else:
        logger.error("FAILURE: Invariance error exceeded tolerance.")
    return passed


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    if args.verify_invariance:
        ok = run_invariance_verification()
        return 0 if ok else 1

    normalizer = LandmarkNormalizer(
        method=args.method,
        align_rotation=args.align_rotation,
    )

    if args.input:
        if not args.input.is_file():
            logger.error("Input file not found: %s", args.input)
            return 1
        if not args.output_dir:
            logger.error("--output-dir is required when normalizing files")
            return 1

        seq, meta = load_landmarks_npz(args.input)
        norm_seq = normalizer.normalize_sequence(seq)
        meta["normalization_method"] = args.method
        meta["align_rotation"] = args.align_rotation

        args.output_dir.mkdir(parents=True, exist_ok=True)
        dest = args.output_dir / f"{args.input.stem}_norm.npz"
        save_landmarks_npz(norm_seq, dest, metadata=meta)
        logger.info("Saved normalized archive: %s", dest)
        return 0

    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error("Input directory not found: %s", args.input_dir)
            return 1
        if not args.output_dir:
            logger.error("--output-dir is required when processing directories")
            return 1

        npz_files = list(args.input_dir.rglob("*.npz"))
        if not npz_files:
            logger.warning("No .npz files found in %s", args.input_dir)
            return 0

        logger.info("Found %d archives to normalize", len(npz_files))
        for p in npz_files:
            seq, meta = load_landmarks_npz(p)
            norm_seq = normalizer.normalize_sequence(seq)
            meta["normalization_method"] = args.method
            meta["align_rotation"] = args.align_rotation

            rel = p.relative_to(args.input_dir).parent
            dest_dir = args.output_dir / rel
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{p.stem}_norm.npz"
            save_landmarks_npz(norm_seq, dest, metadata=meta)

        logger.info("Batch normalization completed for %d archives", len(npz_files))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
