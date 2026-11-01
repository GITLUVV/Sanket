"""CLI tool to train, tune, and evaluate the SVM baseline model (SANKET-029).

Trains an RBF Support Vector Machine on isolated sign features with feature standardization,
Platt probability calibration, optional validation grid search, and unified manifest output.
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
from sanket.models import SVMSignClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sanket.train_svm")

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


def generate_synthetic_benchmark_dataset(
    num_classes: int = 16,
    num_features: int = 645,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate reproducible synthetic dataset mimicking the 16-class isolated sign benchmark."""
    rng = np.random.RandomState(random_seed)
    centers = rng.randn(num_classes, num_features).astype(np.float32)

    def sample_split(n_samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        x_list: list[np.ndarray] = []
        y_list: list[int] = []
        for c in range(num_classes):
            noise = rng.randn(n_samples_per_class, num_features).astype(np.float32) * 0.4
            x_list.append(centers[c] + noise)
            y_list.extend([c] * n_samples_per_class)
        return np.vstack(x_list).astype(np.float32), np.array(y_list, dtype=np.int64)

    x_train, y_train = sample_split(15)
    x_val, y_val = sample_split(4)
    x_test, y_test = sample_split(4)
    classes = INITIAL_16_VOCABULARY[:num_classes]

    return x_train, y_train, x_val, y_val, x_test, y_test, classes


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train, tune, and evaluate the SVM baseline classifier for ISL recognition.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to .npz file containing train/val/test feature arrays.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/baseline_svm.joblib"),
        help=(
            "Destination path for serialized model artifact "
            "(default: models/baseline_svm.joblib)."
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("docs/experiments/manifests/EXP-P1-002-SVM-BASELINE.json"),
        help="Destination path for evaluation manifest JSON.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=10.0,
        help="Regularization parameter C (default: 10.0).",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient gamma: 'scale', 'auto', or float (default: 'scale').",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="Kernel type: 'rbf', 'linear', 'poly' (default: 'rbf').",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler normalization.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform validation grid search over C and gamma before testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42).",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Execute pipeline on synthetic 16-class benchmark.",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parsed = parse_args(args)
    lock_seed(parsed.seed)

    if parsed.synthetic_demo or parsed.data_path is None:
        logger.info("Running SVM baseline training in synthetic benchmark demonstration mode.")
        x_tr, y_tr, x_v, y_v, x_te, y_te, classes = generate_synthetic_benchmark_dataset(
            num_classes=16,
            num_features=645,
            random_seed=parsed.seed,
        )
    else:
        if not parsed.data_path.exists():
            logger.error("Dataset file not found: %s", parsed.data_path)
            return 1
        logger.info("Loading feature dataset from %s", parsed.data_path)
        npz = np.load(parsed.data_path, allow_pickle=True)
        x_tr = npz["x_train"]
        y_tr = npz["y_train"]
        x_v = npz.get("x_val", x_tr)
        y_v = npz.get("y_val", y_tr)
        x_te = npz["x_test"]
        y_te = npz["y_test"]
        classes = (
            list(npz["classes"])
            if "classes" in npz
            else INITIAL_16_VOCABULARY[: len(np.unique(y_tr))]
        )

    gamma_val: str | float
    try:
        gamma_val = float(parsed.gamma)
    except ValueError:
        gamma_val = parsed.gamma

    svm = SVMSignClassifier(
        c=parsed.c,
        kernel=parsed.kernel,
        gamma=gamma_val,
        probability=True,
        class_weight="balanced",
        random_state=parsed.seed,
        scale_features=not parsed.no_scale,
    )

    if parsed.tune:
        logger.info("Tuning SVM hyperparameters on validation split...")
        tune_res = svm.tune_hyperparameters(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_v,
            y_val=y_v,
            classes=classes,
        )
        logger.info("Tuning complete: best parameters = %s", tune_res["best_params"])
    else:
        logger.info("Fitting SVMSignClassifier (C=%.2f, kernel=%s)...", svm.c, svm.kernel)
        svm.fit(x_tr, y_tr, classes=classes)

    # Save model artifact
    saved_model = svm.save(parsed.output_model)
    logger.info("Model artifact serialized to %s", saved_model)

    # Standardized evaluation harness
    evaluator = SignModelEvaluator(seed=parsed.seed, latency_trials=50, warm_up_trials=10)
    report = evaluator.evaluate(
        model=svm,
        x_test=x_te,
        y_test=y_te,
        classes=classes,
        experiment_id="EXP-P1-002-SVM-BASELINE",
    )
    if report.manifest is not None:
        report.manifest.save(parsed.output_manifest)

    print("\n" + report.summary() + "\n")
    logger.info("Evaluation manifest saved to %s", parsed.output_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
