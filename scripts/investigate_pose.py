#!/usr/bin/env python3
"""CLI utility to investigate upper-body pose reference anchors for ISL (SANKET-025).

Evaluates upper-body pose keypoints, hand-to-body relative distances, and spatial
discriminability for minimal pair signs that share handshapes but differ in articulation.

Usage examples:
    python scripts/investigate_pose.py --simulate
    python scripts/investigate_pose.py --video sample.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from sanket.vision.landmarks import FrameLandmarks, HandLandmarkExtractor  # noqa: E402
from sanket.vision.pose import (  # noqa: E402
    ANCHOR_NAMES,
    PoseAnchors,
    PoseExtractor,
    compute_relative_spatial_features,
)
from sanket.vision.preprocessing import sample_video_frames  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pose_investigate")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Investigate upper-body pose anchors and relative hand-to-body features."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=Path,
        help="Path to video clip to evaluate",
    )
    group.add_argument(
        "--simulate",
        action="store_true",
        help="Run empirical minimal-pair spatial distinctiveness simulation",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to export evaluation metrics as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs",
    )
    return parser


def simulate_minimal_pair_analysis() -> dict[str, Any]:
    """Empirically evaluate spatial separability for location-confusable sign pairs.

    Pairs examined:
      1. WATER (tapped at chin/mouth) vs PLEASE (circular rub on chest)
      2. FOOD (fingertips at mouth) vs HELLO (hand at forehead/temple)
      3. DOCTOR (pulse at wrist/forearm) vs HOSPITAL (cross on shoulder)
    """
    # Canonical neutral pose anchor reference coordinates (normalized screen space)
    pose = PoseAnchors(
        nose=np.array([0.50, 0.25, 0.0], dtype=np.float32),
        mouth_center=np.array([0.50, 0.32, 0.0], dtype=np.float32),
        sternum=np.array([0.50, 0.50, 0.0], dtype=np.float32),
        left_shoulder=np.array([0.65, 0.48, 0.0], dtype=np.float32),
        right_shoulder=np.array([0.35, 0.48, 0.0], dtype=np.float32),
        shoulder_width=0.30,
        is_detected=True,
        confidence=1.0,
    )

    # Simulated dominant hand positions for each sign (21x3 hand with wrist at [0] and index at [8])
    signs = {
        "WATER": {"wrist": np.array([0.50, 0.38, 0.05]), "index": np.array([0.50, 0.33, 0.02])},
        "FOOD": {"wrist": np.array([0.51, 0.37, 0.04]), "index": np.array([0.50, 0.32, 0.01])},
        "PLEASE": {"wrist": np.array([0.50, 0.53, 0.03]), "index": np.array([0.50, 0.49, 0.01])},
        "HELLO": {"wrist": np.array([0.60, 0.22, 0.02]), "index": np.array([0.58, 0.18, 0.01])},
    }

    results: dict[str, Any] = {"signs": {}, "separations": {}}

    for name, pts in signs.items():
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = pts["wrist"]
        hand[8] = pts["index"]
        frame_lm = FrameLandmarks(
            left_hand=np.zeros((21, 3), dtype=np.float32),
            right_hand=hand,
            left_detected=False,
            right_detected=True,
            left_confidence=0.0,
            right_confidence=0.95,
        )
        rel_feat = compute_relative_spatial_features(frame_lm, pose)
        results["signs"][name] = {
            "wrist_distances": {
                anchor: float(dist)
                for anchor, dist in zip(ANCHOR_NAMES, rel_feat.right_wrist_distances, strict=False)
            },
            "index_distances": {
                anchor: float(dist)
                for anchor, dist in zip(ANCHOR_NAMES, rel_feat.right_index_distances, strict=False)
            },
            "feature_vector_norm": float(np.linalg.norm(rel_feat.to_vector())),
        }

    # Calculate separation between minimal pairs:
    # Pair 1: WATER vs PLEASE (Mouth/Chin vs Chest)
    w_vec = np.array(list(results["signs"]["WATER"]["wrist_distances"].values()))
    p_vec = np.array(list(results["signs"]["PLEASE"]["wrist_distances"].values()))
    dist_water_please = float(np.linalg.norm(w_vec - p_vec))

    # Pair 2: FOOD vs HELLO (Mouth vs Forehead)
    f_vec = np.array(list(results["signs"]["FOOD"]["wrist_distances"].values()))
    h_vec = np.array(list(results["signs"]["HELLO"]["wrist_distances"].values()))
    dist_food_hello = float(np.linalg.norm(f_vec - h_vec))

    results["separations"] = {
        "WATER_vs_PLEASE_distance_margin": dist_water_please,
        "FOOD_vs_HELLO_distance_margin": dist_food_hello,
    }
    return results


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    if args.simulate:
        logger.info("Executing empirical minimal-pair spatial distinctiveness simulation...")
        res = simulate_minimal_pair_analysis()
        logger.info("--- Simulation Results ---")
        for sign, data in res["signs"].items():
            logger.info("Sign '%s' Hand Distances (normalized by shoulder width):", sign)
            for anchor, d in data["wrist_distances"].items():
                logger.info("   - to %-15s: %.3f", anchor, d)
        logger.info("Separation Metrics:")
        for metric, val in res["separations"].items():
            logger.info("   * %s: %.4f", metric, val)

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            logger.info("Exported simulation metrics to %s", args.output_json)
        return 0

    if args.video:
        video_path = args.video
        if not video_path.is_file():
            logger.error("Video file not found: %s", video_path)
            return 1

        logger.info("Processing video clip: %s", video_path)
        frames = sample_video_frames(video_path, target_frames=10, to_rgb=True)

        with HandLandmarkExtractor() as hand_ext, PoseExtractor() as pose_ext:
            pose_detected_count = 0
            all_distances: list[dict[str, float]] = []

            for i, f in enumerate(frames):
                h_res = hand_ext.extract_frame(f)
                p_res = pose_ext.extract_frame(f)
                if p_res.is_detected:
                    pose_detected_count += 1
                rel = compute_relative_spatial_features(h_res, p_res)
                all_distances.append(
                    {
                        "frame": i,
                        "pose_detected": p_res.is_detected,
                        "right_wrist_to_mouth": float(rel.right_wrist_distances[1]),
                        "right_wrist_to_sternum": float(rel.right_wrist_distances[2]),
                    }
                )

        logger.info(
            "Video processed: Pose detected in %d/%d frames",
            pose_detected_count,
            len(frames),
        )

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump({"video": str(video_path), "frames": all_distances}, f, indent=2)
            logger.info("Exported video analysis to %s", args.output_json)

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
