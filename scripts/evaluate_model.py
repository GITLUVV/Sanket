"""CLI tool to execute standardized model evaluation and manifest generation.

Evaluates trained sign recognition models against test datasets, profiles operational
runtime latency, checks target benchmark thresholds, and outputs structured JSON manifests
compliant with docs/research/EVALUATION_METHODOLOGY.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from sanket.evaluation import SignModelEvaluator, lock_seed
from sanket.models import RandomForestSignClassifier, SVMSignClassifier
from sanket.models.base import BaseSignClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sanket.evaluate_model")

INITIAL_16_VOCABULARY = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "PLEASE",
    "WATER",
    "FOOD",
    "HELP",
    "GOOD",
    "BAD",
    "FAMILY",
    "FRIEND",
    "WORK",
    "NAME",
    "WHERE",
    "HOW",
]


def generate_synthetic_test_data(
    num_classes: int = 16,
    num_features: int = 645,
    samples_per_class: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate reproducible synthetic test dataset for isolated benchmarking."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_classes, num_features).astype(np.float32)

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for c in range(num_classes):
        noise = rng.randn(samples_per_class, num_features).astype(np.float32) * 0.4
        x_list.append(centers[c] + noise)
        y_list.extend([c] * samples_per_class)

    x_arr = np.vstack(x_list).astype(np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    classes = INITIAL_16_VOCABULARY[:num_classes]
    return x_arr, y_arr, classes


def load_model_from_file(model_path: Path) -> BaseSignClassifier:
    """Load a trained classifier from disk."""
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    # Check model type by trying specialized load methods
    for loader in (RandomForestSignClassifier.load, SVMSignClassifier.load):
        try:
            return loader(model_path)
        except Exception:
            pass

    import joblib

    loaded = joblib.load(model_path)
    if isinstance(loaded, BaseSignClassifier):
        return loaded

    msg = f"Loaded object {type(loaded)} does not implement BaseSignClassifier."
    raise TypeError(msg)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate sign language models and generate structured benchmark manifests.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to serialized trained model artifact (.joblib).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Path to .npz file containing test features ('x' or 'x_test') and "
            "labels ('y' or 'y_test')."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Unique identifier for the experiment manifest (e.g. EXP-P1-001-RF).",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("docs/experiments/manifests"),
        help="Directory to save structured JSON manifests (default: docs/experiments/manifests).",
    )
    parser.add_argument(
        "--latency-trials",
        type=int,
        default=100,
        help="Number of latency benchmarking trials (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation determinism (default: 42).",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run self-contained demonstration using synthetic 16-class benchmark.",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parsed = parse_args(args)
    lock_seed(parsed.seed)

    if parsed.synthetic_demo or parsed.model_path is None:
        logger.info("Executing evaluation in synthetic benchmark demo mode.")
        x_test, y_test, classes = generate_synthetic_test_data(
            num_classes=16,
            num_features=645,
            samples_per_class=5,
            seed=parsed.seed,
        )

        if parsed.model_path is not None and parsed.model_path.exists():
            logger.info("Loading model from %s", parsed.model_path)
            model = load_model_from_file(parsed.model_path)
        else:
            logger.info(
                "Fitting demonstration RandomForestSignClassifier on synthetic training split."
            )
            x_train, y_train, _ = generate_synthetic_test_data(
                num_classes=16,
                num_features=645,
                samples_per_class=15,
                seed=parsed.seed + 1,
            )
            model = RandomForestSignClassifier(n_estimators=50, random_state=parsed.seed)
            model.fit(x_train, y_train, classes=classes)

        exp_id = parsed.experiment_id or "EXP-DEMO-SYNTHETIC-RF"
    else:
        if parsed.data_path is None or not parsed.data_path.exists():
            logger.error("Data path must be provided and exist when not in --synthetic-demo mode.")
            return 1

        logger.info("Loading evaluation data from %s", parsed.data_path)
        npz_data = np.load(parsed.data_path, allow_pickle=True)
        x_key = "x_test" if "x_test" in npz_data else ("x" if "x" in npz_data else "features")
        y_key = "y_test" if "y_test" in npz_data else ("y" if "y" in npz_data else "labels")

        x_test = npz_data[x_key]
        y_test = npz_data[y_key]
        classes = (
            list(npz_data["classes"])
            if "classes" in npz_data
            else (INITIAL_16_VOCABULARY if len(np.unique(y_test)) == 16 else None)
        )

        logger.info("Loading model from %s", parsed.model_path)
        model = load_model_from_file(parsed.model_path)
        exp_id = parsed.experiment_id or f"EXP-{model.__class__.__name__.upper()}-EVAL"

    # Execute evaluation harness
    evaluator = SignModelEvaluator(
        seed=parsed.seed,
        latency_trials=parsed.latency_trials,
        warm_up_trials=10,
    )

    report = evaluator.evaluate(
        model=model,
        x_test=x_test,
        y_test=y_test,
        classes=classes,
        experiment_id=exp_id,
        save_manifest_dir=parsed.manifest_dir,
    )

    # Print summary table
    print("\n" + report.summary() + "\n")
    logger.info("Manifest saved to %s/%s.json", parsed.manifest_dir, exp_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
