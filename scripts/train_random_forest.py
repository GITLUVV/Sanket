"""CLI script to train, tune, and evaluate the Random Forest baseline model (SANKET-028).

Trains on the signer-independent training set, evaluates on validation and test sets,
logs classification metrics, extracts feature importances, and serializes the model artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from sanket.models import RandomForestSignClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sanket.train_random_forest")


def generate_synthetic_benchmark_dataset(
    num_classes: int = 16,
    num_features: int = 645,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate reproducible synthetic tabular dataset mimicking the 16-class INCLUDE benchmark."""
    rng = np.random.RandomState(random_seed)

    # Class centers in feature space to give separable distributions
    centers = rng.randn(num_classes, num_features).astype(np.float32)

    def sample_split(n_samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        x_list: list[np.ndarray] = []
        y_list: list[int] = []
        for c in range(num_classes):
            noise = rng.randn(n_samples_per_class, num_features).astype(np.float32) * 0.5
            x_list.append(centers[c] + noise)
            y_list.extend([c] * n_samples_per_class)
        return np.vstack(x_list).astype(np.float32), np.array(y_list, dtype=np.int64)

    # Mimic INCLUDE 16-class counts: ~23 train/class, ~5 val/class, ~4 test/class
    x_train, y_train = sample_split(15)
    x_val, y_val = sample_split(4)
    x_test, y_test = sample_split(4)

    classes = [
        "HELLO", "THANK_YOU", "YES", "NO", "PLEASE",
        "WATER", "FOOD", "HELP", "GOOD", "BAD",
        "FAMILY", "FRIEND", "WORK", "NAME", "WHERE", "HOW",
    ][:num_classes]

    return x_train, y_train, x_val, y_val, x_test, y_test, classes


def train_and_evaluate_rf(
    features_dir: Path | None,
    output_dir: Path,
    tune: bool = False,
    n_estimators: int = 150,
    max_depth: int | None = 20,
    synthetic_demo: bool = False,
) -> dict[str, Any]:
    """Train Random Forest model and emit metrics and artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if synthetic_demo or features_dir is None or not (features_dir / "X_train.npy").is_file():
        logger.info("Using synthetic 16-class benchmark dataset for demonstration/testing.")
        x_train, y_train, x_val, y_val, x_test, y_test, classes = (
            generate_synthetic_benchmark_dataset()
        )
        feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]
    else:
        logger.info("Loading feature arrays from %s...", features_dir)
        x_train = np.load(features_dir / "X_train.npy")
        y_train = np.load(features_dir / "y_train.npy")
        x_val = np.load(features_dir / "X_val.npy")
        y_val = np.load(features_dir / "y_val.npy")
        x_test = np.load(features_dir / "X_test.npy")
        y_test = np.load(features_dir / "y_test.npy")

        # Load class map and feature names
        class_map_file = features_dir / "class_to_index.json"
        if class_map_file.is_file():
            with open(class_map_file, encoding="utf-8") as f:
                c_map = json.load(f)
            classes = sorted(c_map, key=c_map.get)
        else:
            classes = [str(c) for c in np.unique(y_train)]

        feat_names_file = features_dir / "feature_names.json"
        if feat_names_file.is_file():
            with open(feat_names_file, encoding="utf-8") as f:
                feature_names = json.load(f)
        else:
            feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]

    logger.info(
        "Dataset shapes: Train %s, Val %s, Test %s (Classes: %d)",
        x_train.shape,
        x_val.shape,
        x_test.shape,
        len(classes),
    )

    clf = RandomForestSignClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    tuning_results: dict[str, Any] | None = None
    if tune:
        logger.info("Performing validation hyperparameter search...")
        tuning_results = clf.tune_hyperparameters(
            x_train,
            y_train,
            x_val,
            y_val,
            classes=classes,
            feature_names=feature_names,
        )
    else:
        clf.fit(x_train, y_train, classes=classes, feature_names=feature_names)

    # Evaluate on all splits
    train_metrics = clf.evaluate(x_train, y_train, classes=classes)
    val_metrics = clf.evaluate(x_val, y_val, classes=classes)
    test_metrics = clf.evaluate(x_test, y_test, classes=classes)

    logger.info("=== Evaluation Results ===")
    logger.info(
        "Train Accuracy: %.4f | Macro F1: %.4f",
        train_metrics.accuracy,
        train_metrics.macro_f1,
    )
    logger.info(
        "Val   Accuracy: %.4f | Macro F1: %.4f",
        val_metrics.accuracy,
        val_metrics.macro_f1,
    )
    logger.info(
        "Test  Accuracy: %.4f | Macro F1: %.4f",
        test_metrics.accuracy,
        test_metrics.macro_f1,
    )
    logger.info(
        "Test  Top-3 Acc: %.4f | Latency: %.2f ms/sample",
        test_metrics.top_3_accuracy,
        test_metrics.inference_latency_ms,
    )

    # Save trained model
    model_path = output_dir / "random_forest_baseline.joblib"
    clf.save(model_path)

    # Save feature importances
    top_features = clf.get_top_features(k=25)
    top_feat_path = output_dir / "rf_feature_importances.json"
    with open(top_feat_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"feature": name, "importance": score} for name, score in top_features],
            f,
            indent=2,
        )

    # Save summary report
    report = {
        "model": "RandomForestSignClassifier",
        "timestamp": time.time(),
        "hyperparameters": clf.get_params(),
        "tuning": tuning_results,
        "train_metrics": train_metrics.to_dict(),
        "val_metrics": val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "top_features": top_features[:10],
    }

    report_path = output_dir / "rf_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved model artifact and metrics report to %s", output_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Random Forest training."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest baseline sign recognition classifier.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing X_train.npy, y_train.npy, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/baselines"),
        help="Directory to save model artifacts and metrics reports.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning over validation set before final fit.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Number of decision trees (default: 150).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum tree depth (default: 20).",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Train on synthetic 16-class benchmark dataset.",
    )

    args = parser.parse_args(argv)

    report = train_and_evaluate_rf(
        features_dir=args.features_dir if args.features_dir.is_dir() else None,
        output_dir=args.output_dir,
        tune=args.tune,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        synthetic_demo=args.synthetic_demo,
    )

    print(json.dumps({
        "train_accuracy": report["train_metrics"]["accuracy"],
        "val_accuracy": report["val_metrics"]["accuracy"],
        "test_accuracy": report["test_metrics"]["accuracy"],
        "test_macro_f1": report["test_metrics"]["macro_f1"],
        "test_top3_accuracy": report["test_metrics"]["top_3_accuracy"],
        "inference_latency_ms": report["test_metrics"]["inference_latency_ms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
