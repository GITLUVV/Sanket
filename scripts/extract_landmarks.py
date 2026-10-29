#!/usr/bin/env python3
"""CLI utility for extracting 3D skeletal hand landmarks from video clips (SANKET-024).

Usage examples:
    python scripts/extract_landmarks.py --video sample.mp4 --output-dir data/landmarks
    python scripts/extract_landmarks.py --input-dir data/raw/include --output-dir data/landmarks
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
    HandLandmarkExtractor,
    save_landmarks_npz,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("landmarks_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract 3D skeletal hand landmarks from video clips using MediaPipe."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=Path,
        help="Path to a single video clip",
    )
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing video clips to process in batch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save extracted .npz landmark archives",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=30,
        help="Standardized sequence length T (default: 30 frames)",
    )
    parser.add_argument(
        "--impute-strategy",
        type=str,
        default="carry_forward",
        choices=["carry_forward", "zero", "linear", "none"],
        help="Strategy for handling missing hand detections across time (default: carry_forward)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to local hand_landmarker.task bundle",
    )
    parser.add_argument(
        "--export-flat-npy",
        action="store_true",
        help="Also export flattened (T, 126) coordinate matrix as .npy",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    return parser


def process_video_file(
    extractor: HandLandmarkExtractor,
    video_path: Path,
    output_dir: Path,
    target_frames: int,
    impute_strategy: str,
    export_flat_npy: bool,
) -> bool:
    """Process a single video file and save extracted landmarks."""
    try:
        strategy = None if impute_strategy == "none" else impute_strategy
        seq = extractor.extract_video(
            video_path=video_path,
            target_frames=target_frames,
            impute_strategy=strategy,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / f"{video_path.stem}_landmarks.npz"

        metadata = {
            "source_video": video_path.name,
            "target_frames": target_frames,
            "impute_strategy": impute_strategy,
            "left_detected_count": int(np.sum(seq.left_detected)),
            "right_detected_count": int(np.sum(seq.right_detected)),
        }
        save_landmarks_npz(seq, archive_path, metadata=metadata)

        if export_flat_npy:
            matrix = seq.to_flat_matrix()
            npy_path = output_dir / f"{video_path.stem}_landmarks_flat.npy"
            np.save(str(npy_path), matrix)

        logger.info(
            "Extracted: %s (L detected: %d/%d, R detected: %d/%d)",
            video_path.name,
            metadata["left_detected_count"],
            target_frames,
            metadata["right_detected_count"],
            target_frames,
        )
        return True
    except Exception as exc:
        logger.error("Failed processing %s: %s", video_path, exc)
        return False


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    with HandLandmarkExtractor(
        model_path=args.model_path,
        min_detection_confidence=args.confidence,
    ) as extractor:
        if args.video:
            if not args.video.is_file():
                logger.error("Video file not found: %s", args.video)
                return 1
            success = process_video_file(
                extractor=extractor,
                video_path=args.video,
                output_dir=args.output_dir,
                target_frames=args.target_frames,
                impute_strategy=args.impute_strategy,
                export_flat_npy=args.export_flat_npy,
            )
            return 0 if success else 1

        if args.input_dir:
            if not args.input_dir.is_dir():
                logger.error("Input directory not found: %s", args.input_dir)
                return 1

            extensions = {".mp4", ".avi", ".mov", ".mkv"}
            videos = [p for p in args.input_dir.rglob("*") if p.suffix.lower() in extensions]
            if not videos:
                logger.warning("No video files found in %s", args.input_dir)
                return 0

            logger.info("Found %d video clips for landmark extraction", len(videos))
            succeeded = 0
            failed = 0

            for v in sorted(videos):
                rel_dir = v.relative_to(args.input_dir).parent
                target_sub = args.output_dir / rel_dir
                ok = process_video_file(
                    extractor=extractor,
                    video_path=v,
                    output_dir=target_sub,
                    target_frames=args.target_frames,
                    impute_strategy=args.impute_strategy,
                    export_flat_npy=args.export_flat_npy,
                )
                if ok:
                    succeeded += 1
                else:
                    failed += 1

            logger.info("Extraction finished: %d succeeded, %d failed", succeeded, failed)
            return 0 if failed == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
