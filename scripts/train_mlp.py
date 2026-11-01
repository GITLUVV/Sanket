"""CLI tool to train, tune, and evaluate the Multi-Layer Perceptron (MLP) baseline (SANKET-030).

Trains a 3-layer neural network on isolated sign features with feature standardization,
Adam optimizer, adaptive learning rate, early stopping, and standardized manifest output.
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
from sanket.models import MLPSignClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sanket.train_mlp")

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
        description="Train, tune, and evaluate the MLP baseline classifier for ISL recognition.",
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
        default=Path("models/baseline_mlp.joblib"),
        help=(
            "Destination path for serialized model artifact "
            "(default: models/baseline_mlp.joblib)."
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("docs/experiments/manifests/EXP-P1-003-MLP-BASELINE.json"),
        help="Destination path for evaluation manifest JSON.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="256,128,64",
        help="Comma-separated neuron counts for hidden layers (default: 256,128,64).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="L2 regularization penalty alpha (default: 0.0001).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum training iterations/epochs (default: 200).",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler normalization.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform validation grid search over architecture and regularization.",
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
        logger.info("Running MLP baseline training in synthetic benchmark demonstration mode.")
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

    hidden_sizes = tuple(int(dim.strip()) for dim in parsed.hidden_layers.split(",") if dim.strip())

    mlp = MLPSignClassifier(
        hidden_layer_sizes=hidden_sizes,
        alpha=parsed.alpha,
        learning_rate_init=parsed.lr,
        max_iter=parsed.max_iter,
        early_stopping=True,
        random_state=parsed.seed,
        scale_features=not parsed.no_scale,
    )

    if parsed.tune:
        logger.info("Tuning MLP hyperparameters on validation split...")
        tune_res = mlp.tune_hyperparameters(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_v,
            y_val=y_v,
            classes=classes,
        )
        logger.info("Tuning complete: best parameters = %s", tune_res["best_params"])
    else:
        logger.info("Fitting MLPSignClassifier (layers=%s)...", mlp.hidden_layer_sizes)
        mlp.fit(x_tr, y_tr, classes=classes)

    # Save model artifact
    saved_model = mlp.save(parsed.output_model)
    logger.info("Model artifact serialized to %s", saved_model)

    # Standardized evaluation harness
    evaluator = SignModelEvaluator(seed=parsed.seed, latency_trials=50, warm_up_trials=10)
    report = evaluator.evaluate(
        model=mlp,
        x_test=x_te,
        y_test=y_te,
        classes=classes,
        experiment_id="EXP-P1-003-MLP-BASELINE",
    )
    if report.manifest is not None:
        report.manifest.save(parsed.output_manifest)

    print("\n" + report.summary() + "\n")
    logger.info("Evaluation manifest saved to %s", parsed.output_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
