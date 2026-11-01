"""CLI tool for generating structured tabular feature matrices from landmark archives.

Converts raw or normalized SequenceLandmarks (.npz archives) into tabular ML feature
arrays (X, y) partitioned by train, val, and test splits according to
configs/splits/include16_signer_independent.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from sanket.features import FeatureExtractor, FeatureVector
from sanket.vision.landmarks import (
    COORDINATE_DIM,
    NUM_HAND_LANDMARKS,
    SequenceLandmarks,
    load_landmarks_npz,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sanket.generate_features")


def extract_features_from_archive(
    archive_path: Path,
    extractor: FeatureExtractor,
) -> FeatureVector:
    """Load a single SequenceLandmarks .npz archive and extract features."""
    seq, _ = load_landmarks_npz(archive_path)
    return extractor.extract_sequence_features(seq)


def run_feature_generation(
    input_dir: Path,
    output_dir: Path,
    splits_file: Path | None = None,
    pooling: str = "mean_std_delta",
    normalize: bool = True,
) -> dict[str, Any]:
    """Batch generate feature vectors from landmark archives.

    Args:
        input_dir: Directory containing .npz landmark archives.
        output_dir: Destination directory for feature datasets.
        splits_file: Optional JSON split manifest.
        pooling: Temporal pooling strategy.
        normalize: Whether to run landmark normalization.

    Returns:
        Summary dictionary with counts, shapes, and partition metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor(
        normalize_input=normalize,
        pooling_strategy=pooling,
    )

    npz_files = sorted(input_dir.rglob("*.npz"))
    if not npz_files:
        logger.warning("No .npz landmark files found in %s", input_dir)
        return {"total_samples": 0, "status": "no_files_found"}

    logger.info("Found %d landmark archives in %s", len(npz_files), input_dir)

    feature_names = extractor.get_feature_names()
    dim = extractor.output_dimension

    # Save feature names mapping
    names_path = output_dir / "feature_names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Process all archives
    records: list[tuple[str, np.ndarray, str, str]] = []
    # (file_stem, feature_vector, class_name, signer_id)

    for npz_path in npz_files:
        fvec = extract_features_from_archive(npz_path, extractor)

        # Infer class name and signer from path / filename
        class_name = npz_path.parent.name
        stem = npz_path.stem
        # Heuristic signer extraction: 'SignerXX' in stem or parent
        signer = "unknown"
        for part in stem.split("_") + [npz_path.parent.name]:
            if part.lower().startswith("signer"):
                signer = part
                break

        records.append((stem, fvec.features, class_name, signer))

    # Compile global dataset
    x_all = np.array([r[1] for r in records], dtype=np.float32)
    y_classes = [r[2] for r in records]

    # Map class labels to integers
    unique_classes = sorted(list(set(y_classes)))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_all = np.array([class_to_idx[c] for c in y_classes], dtype=np.int64)

    # Save master class index mapping
    class_map_path = output_dir / "class_to_index.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # If split manifest is provided, partition into train/val/test
    partitions: dict[str, dict[str, int]] = {}
    if splits_file is not None and splits_file.is_file():
        logger.info("Partitioning according to split manifest: %s", splits_file)
        with open(splits_file, encoding="utf-8") as f:
            manifest_data = json.load(f)

        split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        split_map: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            sample_ids = manifest_data.get("partitions", {}).get(split_name, {}).get("samples", [])
            for s in sample_ids:
                split_map[s] = split_name

        for idx, (stem, _, _, _) in enumerate(records):
            assigned_split = split_map.get(stem, "train")
            if assigned_split in split_indices:
                split_indices[assigned_split].append(idx)

        for s_name, indices in split_indices.items():
            if indices:
                sub_x = x_all[indices]
                sub_y = y_all[indices]
                np.save(output_dir / f"X_{s_name}.npy", sub_x)
                np.save(output_dir / f"y_{s_name}.npy", sub_y)
                partitions[s_name] = {"count": len(indices), "shape": list(sub_x.shape)}
                logger.info("Saved %s partition: X shape %s", s_name, sub_x.shape)
    else:
        # Save monolithic dataset
        np.save(output_dir / "X.npy", x_all)
        np.save(output_dir / "y.npy", y_all)
        partitions["all"] = {"count": len(x_all), "shape": list(x_all.shape)}

    summary = {
        "total_samples": len(records),
        "feature_dimension": dim,
        "num_classes": len(unique_classes),
        "classes": unique_classes,
        "partitions": partitions,
        "pooling": pooling,
    }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Feature generation complete. Output saved to %s", output_dir)
    return summary


def verify_feature_generation() -> bool:
    """Run programmatic self-verification on synthetic landmarks."""
    logger.info("Running synthetic feature generation verification...")
    t = 15
    np.random.seed(42)
    left = np.random.rand(t, NUM_HAND_LANDMARKS, COORDINATE_DIM).astype(np.float32)
    right = np.random.rand(t, NUM_HAND_LANDMARKS, COORDINATE_DIM).astype(np.float32)

    seq = SequenceLandmarks(
        left_hand=left,
        right_hand=right,
        left_detected=np.ones(t, dtype=bool),
        right_detected=np.ones(t, dtype=bool),
        left_confidence=np.full(t, 0.95, dtype=np.float32),
        right_confidence=np.full(t, 0.95, dtype=np.float32),
    )

    extractor = FeatureExtractor(
        normalize_input=True,
        pooling_strategy="mean_std_delta",
    )
    fvec = extractor.extract_sequence_features(seq)

    # Frame dimension: 126 coords + 38 angles + 48 distances + 3 inter-hand = 215
    expected_frame_dim = 215
    expected_output_dim = expected_frame_dim * 3  # mean, std, delta = 645

    assert extractor.frame_dimension == expected_frame_dim, (
        f"Frame dim mismatch: {extractor.frame_dimension} != {expected_frame_dim}"
    )
    assert fvec.dimension == expected_output_dim, (
        f"Output dim mismatch: {fvec.dimension} != {expected_output_dim}"
    )
    assert len(fvec.feature_names) == expected_output_dim, "Feature names count mismatch"
    assert np.all(np.isfinite(fvec.features)), "Non-finite values detected in feature vector"

    # Determinism verification
    fvec2 = extractor.extract_sequence_features(seq)
    assert np.allclose(fvec.features, fvec2.features), "Feature extraction is not deterministic!"

    logger.info(
        "Verification successful: Frame dim=%d, Aggregated dim=%d, "
        "Finite=True, Deterministic=True.",
        expected_frame_dim,
        expected_output_dim,
    )
    return True


def main() -> int:
    """CLI entrypoint for feature vector generation."""
    parser = argparse.ArgumentParser(
        description="Extract fixed-dimension tabular features from landmark sequences.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/landmarks"),
        help="Directory containing .npz landmark files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Destination directory for feature datasets.",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=Path("configs/splits/include16_signer_independent.json"),
        help="Path to signer-independent JSON split manifest.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_std_delta",
        choices=["mean_std_delta", "mean_std", "mean", "none"],
        help="Temporal aggregation strategy.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip coordinate normalization (if input is already normalized).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run self-test verification on synthetic landmark sequence.",
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_feature_generation()
        return 0 if success else 1

    summary = run_feature_generation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits_file=args.splits_file if args.splits_file.is_file() else None,
        pooling=args.pooling,
        normalize=not args.no_normalize,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
