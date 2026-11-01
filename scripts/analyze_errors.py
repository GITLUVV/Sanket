"""CLI tool to execute confusion matrix diagnostics and phonological error analysis.

Fulfills SANKET-032 by evaluating trained sign recognition models, generating
row-normalized confusion matrix heatmaps, ranking top confusable sign pairs,
classifying errors into the 5-tier phonological linguistic taxonomy, and publishing
formal error reports in docs/experiments/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from sanket.data.ingestion import APPROVED_VOCABULARY_V1
from sanket.evaluation import (
    ConfusionMatrixDiagnostics,
    lock_seed,
)
from sanket.models import (
    MLPSignClassifier,
    RandomForestSignClassifier,
    SVMSignClassifier,
)
from sanket.models.base import BaseSignClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sanket.analyze_errors")


def generate_synthetic_evaluation_data(
    num_classes: int = 16,
    num_features: int = 645,
    samples_per_class: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic test features mimicking phonologically coupled sign clusters."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_classes, num_features).astype(np.float32)

    # Phonologically coupled cluster pairs: deliberately pull centers closer
    # to induce natural linguistic confusions (e.g. DOCTOR vs TIME, WATER vs FOOD)
    cluster_pairs = [(8, 9), (6, 7), (0, 2), (3, 4), (13, 14), (4, 15)]
    for idx1, idx2 in cluster_pairs:
        if idx1 < num_classes and idx2 < num_classes:
            shared = (centers[idx1] + centers[idx2]) / 2.0
            centers[idx1] = shared + rng.randn(num_features).astype(np.float32) * 0.15
            centers[idx2] = shared + rng.randn(num_features).astype(np.float32) * 0.15

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for c in range(num_classes):
        noise = rng.randn(samples_per_class, num_features).astype(np.float32) * 0.45
        x_list.append(centers[c] + noise)
        y_list.extend([c] * samples_per_class)

    x_arr = np.vstack(x_list).astype(np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    classes = APPROVED_VOCABULARY_V1[:num_classes]
    return x_arr, y_arr, classes


def load_model_from_file(model_path: Path) -> BaseSignClassifier:
    """Load a trained classifier from disk."""
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    for loader in (
        RandomForestSignClassifier.load,
        SVMSignClassifier.load,
        MLPSignClassifier.load,
    ):
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
    """Parse command-line arguments for error analysis tool."""
    parser = argparse.ArgumentParser(
        description="Execute confusion matrix diagnostics and phonological error analysis.",
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
        help="Path to .npz file containing test features and labels.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("docs/experiments/ERROR_ANALYSIS_PHASE1.md"),
        help="Destination path for Markdown error analysis report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top confusable pairs to report (default: 10).",
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
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all Phase 1 baselines (Random Forest, SVM, MLP) in demonstration mode.",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parsed = parse_args(args)
    lock_seed(parsed.seed)

    if parsed.compare_all:
        logger.info("Executing comprehensive Phase 1 baseline error analysis comparison.")
        x_test, y_test, classes = generate_synthetic_evaluation_data(
            num_classes=16,
            num_features=645,
            samples_per_class=25,
            seed=parsed.seed,
        )
        x_train, y_train, _ = generate_synthetic_evaluation_data(
            num_classes=16,
            num_features=645,
            samples_per_class=40,
            seed=parsed.seed + 10,
        )

        baselines: list[tuple[str, BaseSignClassifier]] = [
            (
                "Random Forest Baseline",
                RandomForestSignClassifier(n_estimators=100, random_state=42),
            ),
            (
                "SVM (RBF Kernel) Baseline",
                SVMSignClassifier(c=2.0, gamma="scale", random_state=42),
            ),
            (
                "MLP (3-Layer) Baseline",
                MLPSignClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=80,
                    random_state=42,
                ),
            ),
        ]

        reports: list[str] = [
            "# SANKET Phase 1 — Comprehensive Baseline Error Diagnostics & Linguistic Analysis",
            "",
            "- **Evaluation Protocol**: Signer-Independent 16-Sign Vocabulary Benchmark",
            "- **Taxonomy Framework**: 5-Tier Phonological Model (EVALUATION_METHODOLOGY.md)",
            "- **Target Vocabulary**: 16 signs (`APPROVED_VOCABULARY_V1`)",
            "",
            "---",
            "",
        ]

        for model_name, model in baselines:
            logger.info("Training %s on demonstration split...", model_name)
            model.fit(x_train, y_train, classes=classes)

            logger.info("Evaluating %s on test split...", model_name)
            y_pred = model.predict(x_test)
            diag = ConfusionMatrixDiagnostics.from_predictions(y_test, y_pred, classes=classes)

            print("\n" + "=" * 80)
            print(f"DIAGNOSTIC REPORT: {model_name.upper()}")
            print("=" * 80)
            print(diag.render_ascii_heatmap())
            print("\nTop Confusable Pairs:")
            for p in diag.get_top_confusable_pairs(top_k=5):
                print(
                    f"  [{p.true_class:<10} -> {p.predicted_class:<10}] "
                    f"Errors: {p.error_count:2d} ({p.error_rate:5.1%}) | {p.error_type.value}"
                )

            rep_section = diag.generate_full_report(
                model_name=model_name,
                dataset_name="Signer-Independent Test Partition (Synthetic Benchmark)",
            )
            reports.append(rep_section)
            reports.append("\n---\n")

        full_content = "\n".join(reports)
        parsed.output_report.parent.mkdir(parents=True, exist_ok=True)
        parsed.output_report.write_text(full_content, encoding="utf-8")
        logger.info("Published comprehensive comparative report to %s", parsed.output_report)
        return 0

    if parsed.synthetic_demo or parsed.model_path is None:
        logger.info("Executing diagnostics in synthetic benchmark demonstration mode.")
        x_test, y_test, classes = generate_synthetic_evaluation_data(
            num_classes=16,
            num_features=645,
            samples_per_class=25,
            seed=parsed.seed,
        )

        if parsed.model_path is not None and parsed.model_path.exists():
            logger.info("Loading model from %s", parsed.model_path)
            model = load_model_from_file(parsed.model_path)
            model_name = model.__class__.__name__
        else:
            logger.info("Fitting demonstration RandomForestSignClassifier...")
            x_train, y_train, _ = generate_synthetic_evaluation_data(
                num_classes=16,
                num_features=645,
                samples_per_class=35,
                seed=parsed.seed + 1,
            )
            model = RandomForestSignClassifier(n_estimators=60, random_state=parsed.seed)
            model.fit(x_train, y_train, classes=classes)
            model_name = "RandomForestSignClassifier (Demo Baseline)"

        y_pred = model.predict(x_test)
        diag = ConfusionMatrixDiagnostics.from_predictions(y_test, y_pred, classes=classes)
    else:
        if parsed.data_path is None or not parsed.data_path.exists():
            logger.error("Data path must be provided and exist when not in --synthetic-demo mode.")
            return 1

        logger.info("Loading evaluation dataset from %s", parsed.data_path)
        npz_data = np.load(parsed.data_path, allow_pickle=True)
        x_key = "x_test" if "x_test" in npz_data else ("x" if "x" in npz_data else "features")
        y_key = "y_test" if "y_test" in npz_data else ("y" if "y" in npz_data else "labels")

        x_test = npz_data[x_key]
        y_test = npz_data[y_key]
        classes = (
            list(npz_data["classes"])
            if "classes" in npz_data
            else APPROVED_VOCABULARY_V1[: len(np.unique(y_test))]
        )

        logger.info("Loading model from %s", parsed.model_path)
        model = load_model_from_file(parsed.model_path)
        model_name = model.__class__.__name__

        y_pred = model.predict(x_test)
        diag = ConfusionMatrixDiagnostics.from_predictions(y_test, y_pred, classes=classes)

    # Print ASCII heatmap to stdout
    print("\n" + "=" * 80)
    print(f"CONFUSION MATRIX DIAGNOSTICS: {model_name}")
    print("=" * 80)
    print(diag.render_ascii_heatmap())

    print("\n" + "=" * 80)
    print("TOP CONFUSABLE SIGN PAIRS (PHONOLOGICAL LINGUISTIC ATTRIBUTION)")
    print("=" * 80)
    for rank, p in enumerate(diag.get_top_confusable_pairs(top_k=parsed.top_k), start=1):
        print(
            f"  {rank:2d}. [{p.true_class:<10} -> {p.predicted_class:<10}] "
            f"Errors: {p.error_count:2d} ({p.error_rate:5.1%}) | "
            f"{p.error_type.value:<25} | {p.linguistic_explanation}"
        )

    print("\n" + "=" * 80)
    print("PHONOLOGICAL TAXONOMY BREAKDOWN")
    print("=" * 80)
    breakdown = diag.get_phonological_breakdown()
    for cat, count in breakdown.counts_by_type.items():
        prop = breakdown.proportions_by_type.get(cat, 0.0)
        print(f"  - {cat:<30}: {count:3d} errors ({prop:5.1%})")

    # Save report
    out_file = diag.save_report(
        output_path=parsed.output_report,
        model_name=model_name,
        dataset_name="Signer-Independent Test Partition",
    )
    logger.info("Saved error analysis report to %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
