# Model Development

Sanket evaluates multiple baseline model paradigms on identical feature representations and signer-independent splits:

## 1. Random Forest Classifier (SANKET-028)
- Decorrelated ensemble of decision trees.
- Fast training, built-in Gini feature importance ranking, and ultra-low latency (~0.73 ms/sample).

## 2. Support Vector Machine (SANKET-029)
- Maximum-margin separation with Radial Basis Function (RBF) kernel.
- Feature standardization via `StandardScaler` fitted strictly on training data.
- Platt scaling for calibrated multiclass probability estimation.

## 3. Multi-Layer Perceptron (SANKET-030)
- Feed-forward neural network baseline.
- Multi-layer hidden representations with adaptive learning rates and early stopping on validation Macro F1.

All models adhere to the unified `BaseSignClassifier` interface in `src/sanket/models/base.py`.
