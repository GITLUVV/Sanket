#!/usr/bin/env python3
"""CLI utility for video preprocessing, uniform temporal sampling, and tensor export.

Usage examples:
    python scripts/preprocess_videos.py --video data/raw/sample.mp4 --output-dir data/processed
    python scripts/preprocess_videos.py --input-dir data/raw/include --output-dir data/processed
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sanket.vision.preprocessing import (  # noqa: E402
    VideoPreprocessor,
    get_video_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("preprocess_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Preprocess video files with uniform temporal sampling and letterboxing."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=Path,
        help="Path to a single video clip to preprocess",
    )
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing video clips to process in batch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory to save preprocessed .npy tensors and previews",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=30,
        help="Standardized sequence length T (default: 30 frames)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Standardized frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Standardized frame height (default: 480)",
    )
    parser.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Disable aspect ratio letterboxing (use direct linear resize)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize pixel intensities to float32 [0.0, 1.0]",
    )
    parser.add_argument(
        "--export-preview",
        action="store_true",
        help="Save preview PNG frames (start, middle, end) to output directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    return parser


def export_preview_frames(
    tensor: np.ndarray,
    stem: str,
    output_dir: Path,
) -> None:
    """Save preview frames (first, middle, last) as PNG images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t = tensor.shape[0]
    preview_indices = [0, t // 2, t - 1]

    for idx in preview_indices:
        frame = tensor[idx]
        if tensor.dtype == np.float32:
            frame_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_u8 = frame.copy()

        # Convert RGB back to BGR for OpenCV imwrite
        bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
        out_path = output_dir / f"{stem}_frame{idx:02d}.png"
        cv2.imwrite(str(out_path), bgr)
        logger.info("Saved preview image: %s", out_path)


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    preprocessor = VideoPreprocessor(
        target_frames=args.target_frames,
        target_size=(args.width, args.height),
        letterbox=not args.no_letterbox,
        to_rgb=True,
        normalize_pixels=args.normalize,
    )

    if args.video:
        video_path = args.video
        if not video_path.is_file():
            logger.error("Video file not found: %s", video_path)
            return 1

        try:
            meta = get_video_metadata(video_path)
            logger.info(
                "Source: %s [%dx%d @ %.1ffps, %d frames, %.2fs]",
                meta.path,
                meta.width,
                meta.height,
                meta.fps,
                meta.frame_count,
                meta.duration_seconds,
            )

            tensor = preprocessor.preprocess(video_path)
            logger.info("Preprocessed tensor shape: %s, dtype: %s", tensor.shape, tensor.dtype)

            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                npy_path = args.output_dir / f"{video_path.stem}_preprocessed.npy"
                np.save(str(npy_path), tensor)
                logger.info("Saved serialized tensor: %s", npy_path)

                if args.export_preview:
                    export_preview_frames(tensor, video_path.stem, args.output_dir)

            return 0
        except Exception as exc:
            logger.error("Failed to preprocess %s: %s", video_path, exc)
            return 1

    if args.input_dir:
        input_dir = args.input_dir
        if not input_dir.is_dir():
            logger.error("Input directory not found: %s", input_dir)
            return 1

        extensions = {".mp4", ".avi", ".mov", ".mkv"}
        video_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]
        if not video_files:
            logger.warning("No video files found in %s", input_dir)
            return 0

        logger.info("Found %d video clips to preprocess", len(video_files))
        succeeded = 0
        failed = 0

        for vid in sorted(video_files):
            try:
                tensor = preprocessor.preprocess(vid)
                if args.output_dir:
                    rel_dir = vid.relative_to(input_dir).parent
                    dest_dir = args.output_dir / rel_dir
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    npy_path = dest_dir / f"{vid.stem}_preprocessed.npy"
                    np.save(str(npy_path), tensor)

                    if args.export_preview and succeeded == 0:
                        export_preview_frames(tensor, vid.stem, dest_dir)
                succeeded += 1
            except Exception as exc:
                logger.warning("Failed processing %s: %s", vid, exc)
                failed += 1

        logger.info("Batch preprocessing finished: %d succeeded, %d failed", succeeded, failed)
        return 0 if failed == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
