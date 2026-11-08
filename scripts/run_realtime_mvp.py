#!/usr/bin/env python3
"""Run SANKET real-time Indian Sign Language recognition camera MVP (SANKET-048).

Usage:
    python scripts/run_realtime_mvp.py [OPTIONS]

Examples:
    # Run with default web camera:
    python scripts/run_realtime_mvp.py

    # Run in headless mode with synthetic mock video (CI / benchmarking):
    python scripts/run_realtime_mvp.py --mock --headless --max-frames 100

    # Run with custom camera index and strict thresholding:
    python scripts/run_realtime_mvp.py --camera-id 1 --sensitivity strict
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sanket.realtime.pipeline import PipelineConfig, RealtimePipeline
from sanket.realtime.thresholding import SensitivityProfile


def setup_logging(verbose: bool = False) -> None:
    """Configure console log formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SANKET Real-Time Indian Sign Language Camera MVP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Camera video device index.")
    parser.add_argument("--width", type=int, default=640, help="Target resolution width.")
    parser.add_argument("--height", type=int, default=480, help="Target resolution height.")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate.")
    parser.add_argument(
        "--no-mirror", action="store_true", help="Disable selfie horizontal mirroring."
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to model artifact (.joblib)."
    )
    parser.add_argument("--label-map", type=str, default=None, help="Path to JSON label map.")
    parser.add_argument(
        "--confidence", type=float, default=0.65, help="Minimum decision confidence threshold."
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default="balanced",
        choices=["permissive", "balanced", "strict"],
        help="Gating sensitivity preset profile.",
    )
    parser.add_argument("--no-hud", action="store_true", help="Disable visual HUD rendering.")
    parser.add_argument(
        "--mock", action="store_true", help="Use synthetic mock video instead of webcam."
    )
    parser.add_argument(
        "--mock-pattern", type=str, default="moving_box", help="Pattern for mock video."
    )
    parser.add_argument("--video-file", type=str, default=None, help="Path to video file.")
    parser.add_argument(
        "--headless", action="store_true", help="Run without creating an OpenCV GUI window."
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process before exiting."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    return parser.parse_args()


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("sanket.realtime")

    # Build Pipeline Configuration
    config = PipelineConfig(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
        use_mock=args.mock,
        mock_pattern=args.mock_pattern,
        mock_file=args.video_file,
        model_path=args.model_path,
        label_map_path=args.label_map,
        min_confidence=args.confidence,
        sensitivity_profile=SensitivityProfile(args.sensitivity),
        show_hud=not args.no_hud,
        headless=args.headless,
    )

    logger.info(
        "Initializing SANKET Real-Time MVP (Camera: %s, Headless: %s)",
        args.camera_id,
        args.headless,
    )

    # Initialize Pipeline
    pipeline = RealtimePipeline(config=config)

    # Signal Handling for Graceful Shutdown
    def handle_signal(sig: int, frame: Any) -> None:
        logger.info("Termination signal received. Shutting down gracefully...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Subscriber to log new subtitle emissions to stdout
    def log_sign_emission(entry: Any) -> None:
        tag = f"{entry.english_text} ({entry.hindi_text})"
        print(f"\n[SIGN EMITTED] {tag} - Conf: {entry.confidence:.1%}")

    pipeline.text_display.subscribe(log_sign_emission)

    try:
        pipeline.run_loop(max_frames=args.max_frames)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        pipeline.stop()
        full_transcript = pipeline.text_display.get_full_transcript()
        if full_transcript:
            print("\n" + "=" * 50)
            print("SESSION TRANSCRIPT:")
            print(full_transcript)
            print("=" * 50)

    logger.info(
        "Pipeline exited cleanly. Total frames processed: %d",
        pipeline.telemetry.processed_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
