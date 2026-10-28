#!/usr/bin/env python3
"""CLI utility for video dataset inspection, validation, and label analysis.

Usage examples:
    python scripts/inspect_dataset.py --input-dir data/raw/include
    python scripts/inspect_dataset.py --report-markdown docs/datasets/INCLUDE_INSPECTION.md
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

from sanket.data.inspection import (  # noqa: E402
    inspect_dataset_directory,
    save_inspection_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("inspect_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Inspect and profile Indian Sign Language video datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/include"),
        help="Root directory of raw video files to inspect",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="INCLUDE-16",
        help="Dataset name identifier",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.2,
        help="Minimum acceptable video duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum acceptable video duration in seconds",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("docs/datasets/INCLUDE_INSPECTION.md"),
        help="Output path for markdown inspection report",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for JSON structured summary output",
    )
    return parser


def main() -> int:
    """Execute CLI dataset inspection workflow."""
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        logger.warning(
            "Input directory '%s' does not exist yet. Creating placeholder summary.",
            input_dir,
        )

    summary = inspect_dataset_directory(
        dataset_dir=input_dir,
        dataset_name=args.dataset_name,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    save_inspection_report(
        summary=summary,
        output_markdown_path=args.report_markdown,
        output_json_path=args.report_json,
    )

    logger.info(
        "Total clips: %d (Valid: %d, Pruned: %d)",
        summary.total_videos,
        summary.valid_videos,
        summary.corrupted_videos,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
